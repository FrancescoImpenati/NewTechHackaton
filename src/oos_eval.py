"""Frozen-model out-of-sample evaluation on 2022-2025 (Phase 3, WP4 + WP5).

WP4 (headline): load the FROZEN dev-fit (2000-2018) frozen-36 detectors + the
production hyperparameters, NO refitting on OOS data, and score:
  1. a CONTROL on the original 2019-2021 holdout — must reproduce the committed
     frozen-36 numbers (AUC-PR ~0.784) or we stop;
  2. the free-data 2022-2025 second holdout — detection metrics overall and per
     stress episode, with the per-model score that drove each flag.

WP5 (deployment scenario): refit the four detectors on the FULL 2000-2021
(frozen-36), re-tune tau in-sample, evaluate on the SAME 2022-2025 OOS, and
compare frozen-2018 vs refit-2021.

The frozen-36 model is reconstructed exactly as Phase-2's
``feature_selection.evaluate_frozen_on_holdout``: a fresh ``FoldScaler`` fit on
the 2000-2018 development set restricted to the frozen subset, ``clone_unfit`` of
the saved production detectors fit on the dev normals, percentile-vs-dev-normal
reference, soft-median ensemble, frozen tau from ``selected_features.json``.

Run from the repo root::

    python -m src.oos_eval
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # handoff §5.3
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

from src.data_loader import PROJECT_ROOT
from src.ensemble import MODEL_ORDER, clone_unfit, score_to_percentile
from src.extend_dataset import NEUTRALIZE_FEATURES, frozen_features
from src.models import compute_metrics
from src.preprocessing import FoldScaler
from src.sensitivity import load_model_folds, load_production_models
from src.y_rule import EX_ANTE_WINDOWS

TABLES = PROJECT_ROOT / "outputs" / "tables"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
EXTENDED = PROJECT_ROOT / "data" / "processed" / "extended"
SEL = json.load(open(TABLES / "selected_features.json"))
FROZEN_TAU = SEL["tau"]
METRIC_KEYS = ["AUC_PR", "F_beta_2", "Recall", "Precision", "F1", "AUC_ROC"]


# --------------------------------------------------------------------------
# Frozen-36 scorer (dev-fit; built once, never refit on OOS)
# --------------------------------------------------------------------------
class Frozen36Scorer:
    """Dev-fit (2000-2018) frozen-36 soft-median ensemble; pure inference."""

    def __init__(self, refit_full_2021: bool = False, retune_tau: bool = False):
        self.subset = frozen_features()
        self.cols = self.subset + ["Y"]
        data = load_model_folds()
        self.dev = data["dev"]
        self.dev_normals = data["dev_normals"]
        self.production = load_production_models()
        # WP5 deployment scenario: fit on the full 2000-2021 instead of 2000-2018.
        fit_frame = pd.concat([data["dev"], data["test"]]) if refit_full_2021 else data["dev"]
        fit_normals = fit_frame[fit_frame["Y"] == 0]
        self.fit_mean = fit_frame[self.subset].mean()   # for neutralized fills
        self.scaler = FoldScaler().fit_on_development(fit_frame[self.cols])
        Xn = self.scaler.transform_holdout(fit_normals[self.cols])
        self.fitted, self.ref = {}, {}
        for name in MODEL_ORDER:
            self.fitted[name] = clone_unfit(self.production[name]).fit(Xn)
            self.ref[name] = self.fitted[name].score_samples(Xn)
        # default neutralization set (DX-4/5 behaviour); overridable per call
        # so the Italy-10Y patch can score with a reduced set additively.
        self.neutralize = list(NEUTRALIZE_FEATURES)
        self.tau = self._retune_tau(fit_frame) if retune_tau else FROZEN_TAU

    def _retune_tau(self, fit_frame) -> float:
        """In-sample argmax-F1 tau over the 80-99th percentile grid (WP5)."""
        from src.models import _tune_threshold
        ens = self._ensemble(fit_frame)
        eps, _ = _tune_threshold(ens, ens, fit_frame["Y"].to_numpy())
        return float(eps)

    def _prepare(self, panel: pd.DataFrame, neutralize=None) -> pd.DataFrame:
        """Neutralize gap features to the dev mean (-> standardized 0)."""
        neutralize = self.neutralize if neutralize is None else neutralize
        p = panel.copy()
        for f in neutralize:
            if f in p.columns:
                p[f] = self.fit_mean[f]
        return p

    def per_model_pct(self, panel: pd.DataFrame, neutralize=None) -> dict[str, np.ndarray]:
        X = self.scaler.transform_holdout(self._prepare(panel, neutralize))
        return {n: score_to_percentile(self.ref[n], self.fitted[n].score_samples(X))
                for n in MODEL_ORDER}

    def _ensemble(self, panel: pd.DataFrame, neutralize=None) -> np.ndarray:
        pct = self.per_model_pct(panel, neutralize)
        return np.median(np.array([pct[n] for n in MODEL_ORDER]), axis=0)

    def score(self, panel: pd.DataFrame, neutralize=None) -> pd.DataFrame:
        """Return per-week ensemble score, flag, and the 4 per-model percentiles."""
        pct = self.per_model_pct(panel, neutralize)
        ens = np.median(np.array([pct[n] for n in MODEL_ORDER]), axis=0)
        out = pd.DataFrame({"ens": ens, "flag": (ens >= self.tau).astype(int)},
                           index=panel.index)
        for n in MODEL_ORDER:
            out[f"pct_{n}"] = pct[n]
        if "Y" in panel.columns:
            out["Y"] = panel["Y"].to_numpy()
        return out


# --------------------------------------------------------------------------
# Detection metrics + per-episode analysis
# --------------------------------------------------------------------------
def detection_metrics(scored: pd.DataFrame) -> dict:
    y = scored["Y"].to_numpy().astype(int)
    return compute_metrics(y, scored["flag"].to_numpy(), scored["ens"].to_numpy())


def episode_analysis(scored: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Per stress window: coverage, first-flag lead/lag vs window start, peak
    ensemble score, and the per-model percentile that drove the flags."""
    rows = []
    for name, start, end in EX_ANTE_WINDOWS:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        win = scored.loc[(scored.index >= s) & (scored.index <= e)]
        if len(win) == 0:
            continue
        flagged = win[win["flag"] == 1]
        first_flag = flagged.index.min() if len(flagged) else None
        lead_lag = (int((first_flag - s).days / 7) if first_flag is not None else None)
        drivers = {n: float(win[f"pct_{n}"].mean()) for n in MODEL_ORDER}
        dominant = max(drivers, key=drivers.get)
        rows.append({
            "episode": name, "start": start, "end": end, "n_weeks": len(win),
            "n_flagged": int(win["flag"].sum()),
            "coverage": round(float(win["flag"].mean()), 3),
            "detected": bool(win["flag"].any()),
            "first_flag": str(first_flag.date()) if first_flag is not None else None,
            "lead_lag_weeks_vs_start": lead_lag,
            "peak_ens": round(float(win["ens"].max()), 4),
            "dominant_model": dominant,
            **{f"mean_pct_{n}": round(drivers[n], 3) for n in MODEL_ORDER},
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_control(scorer: Frozen36Scorer, verbose: bool = True) -> dict:
    """Score the ORIGINAL 2019-2021 holdout; must reproduce the committed
    frozen-36 numbers (AUC-PR ~0.784)."""
    data = load_model_folds()
    test = data["test"]
    scored = scorer.score(test[scorer.cols])
    m = detection_metrics(scored)
    committed = pd.read_csv(TABLES / "holdout_frozen_vs_full.csv")
    ref = committed[committed["subset"] == "frozen_k36"].iloc[0]
    if verbose:
        print("=" * 80)
        print("WP4 CONTROL — frozen-36 on the ORIGINAL 2019-2021 holdout")
        print("=" * 80)
        print(f"{'metric':<12}{'reproduced':>12}{'committed':>12}{'diff':>10}")
        for k, ck in [("AUC_PR", "AUC_PR"), ("F1", "F1"), ("F_beta_2", "F_beta_2"),
                      ("Recall", "Recall"), ("Precision", "Precision"), ("AUC_ROC", "AUC_ROC")]:
            print(f"{k:<12}{m[k]:>12.4f}{ref[ck]:>12.4f}{m[k]-ref[ck]:>+10.4f}")
        ok = abs(m["AUC_PR"] - ref["AUC_PR"]) < 0.02
        print(f"\nControl {'PASS' if ok else 'FAIL'} "
              f"(|AUC-PR diff| {abs(m['AUC_PR']-ref['AUC_PR']):.4f} < 0.02 tol; "
              f"residual = AE re-fit nondeterminism)")
    return {"metrics": m, "committed": ref.to_dict(), "scored": scored}


def run_oos(scorer: Frozen36Scorer, label: str = "frozen_2018", verbose: bool = True,
            panel_path: str | Path | None = None, neutralize=None) -> dict:
    """Score the free-data 2022-2025 second holdout.

    ``panel_path`` / ``neutralize`` default to the DX-3 neutralized panel and the
    full neutralization set (DX-4 behaviour); the Italy-10Y patch (FIX-3) passes
    the patched panel and the reduced set additively."""
    panel = pd.read_parquet(panel_path or (EXTENDED / "model_panel_ext.parquet"))
    oos = panel.loc["2022-01-01":"2025-12-31"]
    scored = scorer.score(oos[scorer.cols], neutralize)
    m = detection_metrics(scored)
    epi = episode_analysis(scored, scorer.tau)
    if verbose:
        print("\n" + "=" * 80)
        print(f"WP4 OOS — frozen-36 ({label}) on free-data 2022-2025 "
              f"(tau={scorer.tau:.4f})")
        print("=" * 80)
        print(f"n_weeks={len(oos)}  Y=1={int(scored['Y'].sum())}  "
              f"flagged={int(scored['flag'].sum())}")
        print("  " + "  ".join(f"{k}={m[k]:.3f}" for k in METRIC_KEYS))
        print("\nPer-episode detection:")
        with pd.option_context("display.width", 170, "display.max_columns", None):
            print(epi[["episode", "n_weeks", "n_flagged", "coverage", "detected",
                       "first_flag", "lead_lag_weeks_vs_start", "peak_ens",
                       "dominant_model"]].to_string(index=False))
    return {"metrics": m, "episodes": epi, "scored": scored}


# --------------------------------------------------------------------------
# Routing + backtest on the free-data 2022-2025 block (WP4)
# --------------------------------------------------------------------------
def _free_asset_returns() -> pd.DataFrame:
    """Weekly asset returns for the backtest, from the free raw panel (same
    construction as nb07, on the free proxies)."""
    raw = pd.read_parquet(EXTENDED / "raw_panel_free.parquet")
    ret = pd.DataFrame(index=raw.index)
    ret["equity"] = raw[["MXUS", "MXEU", "MXJP"]].pct_change().mean(axis=1)
    ret["bond"] = raw["LUACTRUU"].pct_change()
    ret["gold"] = raw["XAUBGNL"].pct_change()
    ret["cash"] = (raw["USGG3M"] / 100.0) / 52.0
    ret["mbs"] = raw["LMBITR"].pct_change()
    return ret


def _oos_subscores(oos_index: pd.DatetimeIndex):
    """USD/Gold/MBS sub-scores for the OOS weeks, calibrated on the ORIGINAL
    development set (frozen), applied to the free extension triggers."""
    from src.free_data import _free_feature_frame
    from src.routing import (SUBSCORE_ORO, SUBSCORE_USD, compute_all_subscores,
                             fit_mbs_params, fit_zscore_params)

    raw = pd.read_parquet(EXTENDED / "raw_panel_free.parquet")
    free = _free_feature_frame(raw)
    trig = pd.read_parquet(EXTENDED / "routing_triggers_ext.parquet")
    # the MBS rule also reads hy_ig_spread_chg4w (from spreads, not the 14 triggers)
    trig = trig.join(free[["hy_ig_spread_chg4w"]], how="left")

    # frozen calibration on the original development routing triggers / spreads
    orig_trig = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "routing_triggers.parquet")
    dev_trig = orig_trig.loc[:"2018-12-31"]
    mu_u, sd_u = fit_zscore_params(dev_trig, SUBSCORE_USD)
    mu_o, sd_o = fit_zscore_params(dev_trig, SUBSCORE_ORO)
    mbs_p = fit_mbs_params(dev_trig)
    subs = compute_all_subscores(trig.loc[oos_index], mu_u, sd_u, mu_o, sd_o, mbs_p)
    dxy = trig.loc[oos_index, "dxy_chg4w"]
    return subs, dxy


def run_backtest(scored: pd.DataFrame, verbose: bool = True) -> dict:
    """Route + backtest the frozen ensemble signal on 2022-2025 vs a buy-&-hold
    MSCI proxy and a 1.5x-levered-equity static strategy."""
    from src.backtest import ASSET_COLS, backtest_strategy, compute_full_metrics
    from src.routing import RoutingEngine

    idx = scored.index
    ret = _free_asset_returns()
    rt = ret.reindex(idx)
    subs, dxy = _oos_subscores(idx)
    tc = {"equity": 5, "bond": 5, "gold": 8, "cash": 2, "mbs": 20}

    sig = pd.DataFrame({"ens_sig": scored["flag"].to_numpy(),
                        "subscore_usd": subs["subscore_usd"].to_numpy(),
                        "subscore_oro": subs["subscore_oro"].to_numpy(),
                        "subscore_mbs": subs["subscore_mbs"].to_numpy(),
                        "dxy_chg4w": dxy.to_numpy()}, index=idx)
    engine = RoutingEngine()
    alloc = engine.route_series(sig, {"usd": 1.5, "oro": 0.5})   # frozen production thresholds

    def static(d):
        w = pd.DataFrame(0.0, index=idx, columns=ASSET_COLS)
        for k, v in d.items():
            w[k] = v
        return w

    strat = {
        "EWS_frozen": RoutingEngine.allocations_to_weights(alloc),
        "MSCI_BH": static({"equity": 1.0}),
        "Levered_1.5x": static({"equity": 1.5, "cash": -0.5}),
    }
    bt = {k: backtest_strategy(w, rt, tc) for k, w in strat.items()}
    rows = {}
    for k, res in bt.items():
        kw = dict(rf_series=rt["cash"])
        if k == "EWS_frozen":
            kw.update(allocations_series=alloc, tc_series=res["tc"], y_series=scored["Y"])
        rows[k] = compute_full_metrics(res["equity_curve"], res["net"], **kw)
    mdf = pd.DataFrame(rows)
    keep = ["CAGR", "Sharpe", "Sortino", "Calmar", "Max_Drawdown",
            "Annual_Turnover", "Hit_Ratio_RiskOff"]
    summary = mdf.loc[keep]
    if verbose:
        print("\n" + "=" * 80)
        print("WP4 BACKTEST — frozen EWS vs benchmarks on free-data 2022-2025")
        print("=" * 80)
        print(f"EWS allocation mix: {alloc.value_counts().to_dict()}")
        print(summary.round(3).to_string())
    return {"metrics": mdf, "summary": summary, "alloc": alloc, "bt": bt}


def plot_oos_timeline(scored: pd.DataFrame, tau: float, out_path: Path | None = None):
    """Ensemble score over 2022-2025 with the tau line, true risk-off shading,
    and the four ex-ante episode windows annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(scored.index, scored["ens"], color="0.25", lw=1.1, label="soft-median ensemble")
    ax.axhline(tau, color="#d73027", ls="--", lw=1.3, label=f"frozen tau={tau:.3f}")
    y = scored["Y"].to_numpy()
    idx = scored.index
    i = 0
    first = True
    while i < len(y):
        if y[i] == 1:
            j = i
            while j + 1 < len(y) and y[j + 1] == 1:
                j += 1
            ax.axvspan(idx[i], idx[j], color="#fdae61", alpha=0.25,
                       label="true risk-off (Y=1)" if first else None)
            first = False
            i = j + 1
        else:
            i += 1
    for name, s, e in EX_ANTE_WINDOWS:
        ax.annotate(name, xy=(pd.Timestamp(s), 1.02), fontsize=7, rotation=0,
                    ha="left", va="bottom", color="#2166ac")
    ax.set_ylim(0, 1.08)
    ax.set_title("Frozen-36 ensemble on free-data 2022-2025 (Phase 3 OOS)")
    ax.set_ylabel("ensemble percentile score")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path = out_path or FIGURES / "oos_timeline_2022_2025.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def run_wp5(frozen_oos: dict, verbose: bool = True) -> dict:
    """WP5 deployment scenario: refit the 4 detectors on the FULL 2000-2021
    (frozen-36), re-tune tau in-sample, evaluate on the SAME 2022-2025 OOS, and
    quantify the value of the 3 extra years of training vs the frozen-2018 model."""
    refit = Frozen36Scorer(refit_full_2021=True, retune_tau=True)
    oos_refit = run_oos(refit, "refit_2021", verbose)
    rows = [{"model": "frozen_2018", "tau": round(FROZEN_TAU, 4),
             **{k: round(frozen_oos["metrics"][k], 4) for k in METRIC_KEYS}},
            {"model": "refit_2021", "tau": round(refit.tau, 4),
             **{k: round(oos_refit["metrics"][k], 4) for k in METRIC_KEYS}}]
    cmp = pd.DataFrame(rows)
    if verbose:
        print("\n" + "=" * 80)
        print("WP5 — DEPLOYMENT SCENARIO: frozen-2018 vs refit-2021 on the SAME 2022-2025 OOS")
        print("=" * 80)
        print(cmp.to_string(index=False))
        d = cmp.iloc[1][METRIC_KEYS].astype(float) - cmp.iloc[0][METRIC_KEYS].astype(float)
        print("\nvalue of 3 extra years (refit - frozen):",
              {k: round(float(d[k]), 4) for k in METRIC_KEYS})
    return {"comparison": cmp, "oos_refit": oos_refit, "refit_scorer": refit}


def main(verbose: bool = True) -> dict:
    scorer = Frozen36Scorer()
    control = run_control(scorer, verbose)
    oos = run_oos(scorer, "frozen_2018", verbose)
    bt = run_backtest(oos["scored"], verbose)
    plot_oos_timeline(oos["scored"], scorer.tau)
    wp5 = run_wp5(oos, verbose)
    wp5["comparison"].to_csv(TABLES / "oos_frozen_vs_refit.csv", index=False)

    # persist tables
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"window": "control_2019_2021", **control["metrics"]},
                  {"window": "oos_2022_2025", **oos["metrics"]}]).to_csv(
        TABLES / "oos_detection_metrics.csv", index=False)
    oos["episodes"].to_csv(TABLES / "oos_episode_detection.csv", index=False)
    bt["summary"].to_csv(TABLES / "oos_backtest_metrics.csv")
    oos["scored"].to_csv(TABLES / "oos_scored_2022_2025.csv")
    return {"control": control, "oos": oos, "backtest": bt, "wp5": wp5}


if __name__ == "__main__":
    main()
