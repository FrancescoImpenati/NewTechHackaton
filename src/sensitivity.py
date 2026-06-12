"""Sensitivity analysis: persistent fold-score cache + tau curves.

The single biggest cost in any threshold / hyperparameter sweep is the ~80 s
leakage-free refit of the 4 detectors across the 5 walk-forward folds (handoff
§4). This module persists those scores once, so every subsequent tau sweep is a
pure numpy re-aggregation (seconds, not minutes), and the AE nondeterminism is
frozen into the cache instead of drifting run-to-run.

Cache layout (``outputs/cache/``)::

    fold_scores/fold_{fid}.npz   per-fold raw scores: ref_{name} (train-normal
                                 scores of the fold-refit model), val_{name}
                                 (fold-val scores), y_val (fold-val labels)
    fold_scores/thresholds.json  per-model production thresholds (the tuned
                                 ``threshold_`` of the saved final detectors)
    final_path.npz               dev-fit reference scores ref_{name} (the saved
                                 final models scored on the development normals)
                                 + holdout_{name} (raw scores on the sealed test
                                 holdout, stored for the single end-of-WP3 shot;
                                 NO holdout labels are cached)
    meta.json                    model order, fold row counts, creation note

``load_fold_scores()`` reproduces exactly the in-memory dict consumed by
``EnsembleDetector.fit_threshold_walkforward(fold_scores=...)``:
``{fid: {'ref': {name: arr}, 'val': {name: arr}, 'thr': {name: float},
'y_val': arr}}`` (the extra ``y_val`` key is ignored by the ensemble and used
by the tau sweeps).

Run from the repo root: ``python -m src.sensitivity`` (builds the cache if
missing, then writes tau_curve.csv / tau_stability.csv + the tau figure).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Handoff §5.3: reduce AE drift. Must be set before tensorflow is imported
# (src.models also sets them, but a direct import of this module must be safe).
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

from src.ensemble import MODEL_ORDER, EnsembleDetector, clone_unfit, score_to_percentile
from src.models import (
    AutoencoderDetector,
    IsolationForestDetector,
    MVGAnomalyDetector,
    OneClassSVMDetector,
    weighted_median,
    weighted_mean,
)
from src.preprocessing import FoldScaler
from src.splits import walkforward_split

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "outputs" / "models"
CACHE_DIR = ROOT / "outputs" / "cache"
FOLD_CACHE_DIR = CACHE_DIR / "fold_scores"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
TRIGGERS_PATH = ROOT / "data" / "processed" / "routing_triggers.parquet"

PRODUCTION_TAU = 0.9769  # deployed soft_median tau (ensemble_final.pkl meta)

# tau sweep grid: [0.80, 1.00] in steps of 0.0025 (81 points).
TAU_GRID = np.round(np.linspace(0.80, 1.00, 81), 4)


# --------------------------------------------------------------------------
# Data + production-artifact assembly
# --------------------------------------------------------------------------
def load_model_folds() -> dict:
    """Build the enriched 56-feature model panel exactly as in handoff §2.

    Returns ``{'folds': [...], 'dev': df, 'dev_normals': df, 'test': df}``
    where folds is the list consumed by the walk-forward tuning methods.
    """
    wf = walkforward_split(save=False, verbose=False)
    ebc = pd.read_parquet(TRIGGERS_PATH)["equity_bond_corr_13w"]

    def enrich(df):
        return df.join(ebc, how="left").dropna(subset=["equity_bond_corr_13w"])

    folds = [{"fold_id": f["fold_id"], "train": enrich(f["train"]),
              "val": enrich(f["val"]), "crisis_captured": f["crisis_captured"]}
             for f in wf["models"]["cv_folds"]]
    dev = enrich(wf["models"]["development"])
    test = enrich(wf["models"]["test_holdout"])
    return {"folds": folds, "dev": dev, "dev_normals": dev[dev["Y"] == 0], "test": test}


def load_production_models() -> dict:
    """The 4 saved final (development-fit) detectors, keyed by MODEL_ORDER."""
    return {
        "mvg": MVGAnomalyDetector.load(MODELS_DIR / "mvg_final.pkl"),
        "svm": OneClassSVMDetector.load(MODELS_DIR / "svm_final.pkl"),
        "ae": AutoencoderDetector.load(MODELS_DIR / "autoencoder_final.keras"),
        "if": IsolationForestDetector.load(MODELS_DIR / "if_final.pkl"),
    }


# --------------------------------------------------------------------------
# WP1 — persistent fold-score cache
# --------------------------------------------------------------------------
def cache_fold_scores(rebuild: bool = False, verbose: bool = True) -> dict:
    """Build (or load) the persistent fold-score cache under outputs/cache/.

    Per fold the 4 detectors are re-fit leakage-free (``clone_unfit`` of the
    production models + a fresh ``FoldScaler`` fit on fold-train only), exactly
    as ``EnsembleDetector.fit_threshold_walkforward`` does internally. The
    final inference path (dev-fit reference scores + sealed-holdout raw scores)
    is computed from the SAVED final models + the saved production FoldScaler,
    so it reproduces the handoff §2 inference path bit-for-bit.

    Returns the in-memory fold_scores dict (see module docstring for layout).
    """
    if not rebuild and (FOLD_CACHE_DIR / "fold_1.npz").exists():
        return load_fold_scores()

    t0 = time.time()
    data = load_model_folds()
    folds = data["folds"]
    models = load_production_models()

    scaler = FoldScaler()
    scaler.fit_per_fold(folds)
    scaler.fit_on_development(data["dev"])

    FOLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fold_scores: dict[int, dict] = {}
    thr = {name: float(models[name].threshold_) for name in MODEL_ORDER}

    for fold in folds:
        fid, train, val = fold["fold_id"], fold["train"], fold["val"]
        tn = train[train["Y"] == 0]
        Xtn, Xval = scaler.transform(tn, fid), scaler.transform(val, fid)
        ref, valsc = {}, {}
        for name in MODEL_ORDER:
            fm = clone_unfit(models[name]).fit(Xtn)
            ref[name] = fm.score_samples(Xtn)
            valsc[name] = fm.score_samples(Xval)
        y_val = val["Y"].to_numpy()
        fold_scores[fid] = {"ref": ref, "val": valsc, "thr": dict(thr), "y_val": y_val}
        np.savez(FOLD_CACHE_DIR / f"fold_{fid}.npz", y_val=y_val,
                 **{f"ref_{n}": ref[n] for n in MODEL_ORDER},
                 **{f"val_{n}": valsc[n] for n in MODEL_ORDER})
        if verbose:
            print(f"fold {fid}: cached ({len(val)} val rows, {int(y_val.sum())} pos, "
                  f"{time.time() - t0:.0f}s elapsed)")

    with open(FOLD_CACHE_DIR / "thresholds.json", "w") as f:
        json.dump(thr, f, indent=2)

    # Final path: SAVED dev-fit models + saved production scaler (handoff §2).
    prod_scaler = FoldScaler.load(MODELS_DIR / "scaler_final.pkl")
    X_dev_normals = prod_scaler.transform_holdout(data["dev_normals"])
    X_test = prod_scaler.transform_holdout(data["test"])
    final = {}
    for name in MODEL_ORDER:
        final[f"ref_{name}"] = models[name].score_samples(X_dev_normals)
        final[f"holdout_{name}"] = models[name].score_samples(X_test)
    np.savez(CACHE_DIR / "final_path.npz", **final)

    meta_ens = EnsembleDetector.load_meta(MODELS_DIR / "ensemble_final.pkl")
    drift = {n: float(np.max(np.abs(final[f"ref_{n}"]
                                    - np.asarray(meta_ens["reference_scores_dict"][n]))))
             for n in MODEL_ORDER}
    with open(CACHE_DIR / "meta.json", "w") as f:
        json.dump({"model_order": MODEL_ORDER,
                   "n_val_rows": {f["fold_id"]: len(f["val"]) for f in folds},
                   "build_seconds": round(time.time() - t0, 1),
                   "max_abs_diff_vs_ensemble_meta_refs": drift,
                   "note": "fold refits via clone_unfit + fresh FoldScaler; "
                           "final path via saved models + scaler_final.pkl"}, f, indent=2)
    if verbose:
        print(f"cache built in {time.time() - t0:.0f}s -> {CACHE_DIR}")
        print("dev-ref drift vs ensemble_final.pkl meta (max abs):",
              {k: f"{v:.2e}" for k, v in drift.items()})
    return fold_scores


def load_fold_scores() -> dict:
    """Load the cache; reproduces exactly the dict consumed by
    ``EnsembleDetector.fit_threshold_walkforward(fold_scores=...)``."""
    with open(FOLD_CACHE_DIR / "thresholds.json") as f:
        thr = {k: float(v) for k, v in json.load(f).items()}
    out = {}
    for fid in (1, 2, 3, 4, 5):
        z = np.load(FOLD_CACHE_DIR / f"fold_{fid}.npz")
        out[fid] = {"ref": {n: z[f"ref_{n}"] for n in MODEL_ORDER},
                    "val": {n: z[f"val_{n}"] for n in MODEL_ORDER},
                    "thr": dict(thr), "y_val": z["y_val"]}
    return out


def load_final_path() -> dict:
    """Dev-fit reference scores + sealed-holdout raw scores (handoff §2 path)."""
    z = np.load(CACHE_DIR / "final_path.npz")
    return {"ref": {n: z[f"ref_{n}"] for n in MODEL_ORDER},
            "holdout": {n: z[f"holdout_{n}"] for n in MODEL_ORDER}}


# --------------------------------------------------------------------------
# WP1 — tau sensitivity
# --------------------------------------------------------------------------
def _ensemble_val_percentiles(fold_scores: dict, mode: str) -> dict[int, np.ndarray]:
    """Percentile-aggregated ensemble score of each fold val (vs fold ref)."""
    agg = np.mean if mode == "soft_mean" else np.median
    out = {}
    for fid, fs in fold_scores.items():
        pct = np.array([score_to_percentile(fs["ref"][n], fs["val"][n])
                        for n in MODEL_ORDER])
        out[fid] = agg(pct, axis=0)
    return out


def tau_curve(fold_scores: dict | None = None,
              modes: tuple[str, ...] = ("soft_median", "soft_mean"),
              taus: np.ndarray = TAU_GRID, save: bool = True) -> pd.DataFrame:
    """Sweep tau over the percentile-aggregated ensemble scores.

    Per fold and per tau: F1, F2, precision, recall and coverage (= fraction of
    val weeks flagged risk-off), plus the n_pos-weighted aggregate row
    (fold='weighted', coverage weighted the same way for consistency).
    """
    fold_scores = fold_scores or load_fold_scores()
    rows = []
    for mode in modes:
        scores = _ensemble_val_percentiles(fold_scores, mode)
        per_fold = {}
        for fid, s in scores.items():
            y = fold_scores[fid]["y_val"]
            n_pos = int(y.sum())
            for tau in taus:
                pred = (s >= tau).astype(int)
                rows.append({"mode": mode, "fold": str(fid), "tau": float(tau),
                             "n_pos": n_pos,
                             "F1": f1_score(y, pred, zero_division=0),
                             "F2": fbeta_score(y, pred, beta=2, zero_division=0),
                             "precision": precision_score(y, pred, zero_division=0),
                             "recall": recall_score(y, pred, zero_division=0),
                             "coverage": float(pred.mean())})
            per_fold[fid] = n_pos
        # n_pos-weighted aggregate curve (same convention as _weighted_summary).
        w = np.array([per_fold[fid] for fid in sorted(per_fold)], dtype=float)
        sub = pd.DataFrame([r for r in rows if r["mode"] == mode and r["fold"] != "weighted"])
        for tau in taus:
            at = sub[sub["tau"] == float(tau)].set_index("fold").loc[
                [str(f) for f in sorted(per_fold)]]
            rows.append({"mode": mode, "fold": "weighted", "tau": float(tau),
                         "n_pos": int(w.sum()),
                         **{m: weighted_mean(at[m].to_numpy(), w)
                            for m in ("F1", "F2", "precision", "recall", "coverage")}})
    df = pd.DataFrame(rows)
    if save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TABLES_DIR / "tau_curve.csv", index=False)
    return df


def tau_stability(curve: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    """Per-fold argmax-F1 / argmax-F2 tau* (ties -> higher tau, as in
    ``_tune_threshold``); min/max/std across folds + n_pos-weighted median."""
    curve = curve if curve is not None else tau_curve(save=False)
    rows = []
    for mode in curve["mode"].unique():
        sub = curve[(curve["mode"] == mode) & (curve["fold"] != "weighted")]
        stars = {}
        for metric in ("F1", "F2"):
            tau_star, n_pos = [], []
            for fid, g in sub.groupby("fold"):
                g = g.sort_values("tau")
                best = g[g[metric] == g[metric].max()]["tau"].max()  # higher tau on ties
                tau_star.append(float(best))
                n_pos.append(int(g["n_pos"].iloc[0]))
                rows.append({"mode": mode, "metric": metric, "fold": fid,
                             "tau_star": float(best), "n_pos": int(g["n_pos"].iloc[0]),
                             "best_value": float(g[metric].max())})
            ts, w = np.array(tau_star), np.array(n_pos, dtype=float)
            stars[metric] = {"min": ts.min(), "max": ts.max(), "std": ts.std(),
                             "wmedian": weighted_median(ts, w)}
            rows.append({"mode": mode, "metric": metric, "fold": "summary",
                         "tau_star": stars[metric]["wmedian"], "n_pos": int(w.sum()),
                         "best_value": np.nan,
                         "tau_min": stars[metric]["min"], "tau_max": stars[metric]["max"],
                         "tau_std": stars[metric]["std"]})
    df = pd.DataFrame(rows)
    if save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TABLES_DIR / "tau_stability.csv", index=False)
    return df


def plot_tau_curve(curve: pd.DataFrame, metric: str = "F1",
                   out_path: Path | None = None):
    """One figure: a thin line per fold + the thick weighted line, per mode;
    vertical marker at the production tau."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(curve["mode"].unique())
    fig, axes = plt.subplots(1, len(modes), figsize=(6.5 * len(modes), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, mode in zip(axes, modes):
        sub = curve[curve["mode"] == mode]
        for fid in ("1", "2", "3", "4", "5"):
            g = sub[sub["fold"] == fid].sort_values("tau")
            ax.plot(g["tau"], g[metric], lw=1.0, alpha=0.65,
                    label=f"fold {fid} (n_pos={int(g['n_pos'].iloc[0])})")
        g = sub[sub["fold"] == "weighted"].sort_values("tau")
        ax.plot(g["tau"], g[metric], lw=2.6, color="black", label="n_pos-weighted")
        ax.axvline(PRODUCTION_TAU, color="#d73027", ls="--", lw=1.4,
                   label=f"production tau={PRODUCTION_TAU:.3f}")
        ax.set_title(f"{mode}: {metric} vs tau")
        ax.set_xlabel("tau (ensemble percentile threshold)")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_path or FIGURES_DIR / "tau_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# WP4 — hyperparameter x tau surfaces (full 56 set; AE excluded)
# --------------------------------------------------------------------------
SVM_NU_GRID = [0.05, 0.10, 0.15, 0.22]
IF_CONTAMINATION_GRID = [0.05, 0.10, 0.15, 0.22]
PRODUCTION_SVM = {"nu": 0.10, "gamma": 0.001}
PRODUCTION_IF_CONTAMINATION = 0.10


def hyperparam_tau_surface(model: str, grid: list | None = None,
                           taus: np.ndarray = TAU_GRID,
                           fold_scores: dict | None = None,
                           save: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Single-model hyperparameter x tau surface (n_pos-weighted F2 + AUC-PR).

    ``model`` is "svm" (nu grid, gamma fixed at the production 0.001) or "if"
    (contamination grid). Per hyperparameter value and fold, raw val scores are
    mapped to percentiles vs the fold train-normal reference and thresholded at
    each tau; F2 is aggregated n_pos-weighted across folds. AUC-PR (threshold-
    free) is computed on the RAW val scores — the percentile map collapses all
    val scores above the reference max into ties at 1.0, which distorts AP;
    raw-score AP matches the models_comparison.csv convention. The AE is
    excluded from surfaces (cost + nondeterminism).

    Cache reuse: the production combo (svm nu=0.10/gamma=0.001) reuses the WP1
    cached scores. IF ``score_samples`` is contamination-invariant with fixed
    random_state (handoff §1.9.4) — contamination only moves the operating
    point, never the scores — so ALL contamination rows reuse the cached
    scores; the invariance is asserted by refitting fold 1 at the first
    non-production value.
    """
    from sklearn.metrics import average_precision_score

    assert model in ("svm", "if")
    fold_scores = fold_scores or load_fold_scores()
    grid = grid if grid is not None else (SVM_NU_GRID if model == "svm"
                                          else IF_CONTAMINATION_GRID)
    fids = sorted(fold_scores)
    y = {fid: fold_scores[fid]["y_val"] for fid in fids}
    w = np.array([y[fid].sum() for fid in fids], dtype=float)

    # Per-hyperparameter raw + percentile val scores ({h: {fid: array}}).
    raw_scores: dict[float, dict[int, np.ndarray]] = {}
    pct_scores: dict[float, dict[int, np.ndarray]] = {}
    needs_refit = (model == "svm")
    if needs_refit:
        data = load_model_folds()
        scaler = FoldScaler()
        scaler.fit_per_fold(data["folds"])
    for h in grid:
        if model == "if" or (model == "svm" and h == PRODUCTION_SVM["nu"]):
            raw_scores[h] = {fid: fold_scores[fid]["val"][model] for fid in fids}
            pct_scores[h] = {fid: score_to_percentile(fold_scores[fid]["ref"][model],
                                                      fold_scores[fid]["val"][model])
                             for fid in fids}
        else:
            raw_scores[h], pct_scores[h] = {}, {}
            for fold in data["folds"]:
                fid = fold["fold_id"]
                tn = fold["train"][fold["train"]["Y"] == 0]
                det = OneClassSVMDetector(nu=h, gamma=PRODUCTION_SVM["gamma"]).fit(
                    scaler.transform(tn, fid))
                ref = det.score_samples(scaler.transform(tn, fid))
                val = det.score_samples(scaler.transform(fold["val"], fid))
                raw_scores[h][fid] = val
                pct_scores[h][fid] = score_to_percentile(ref, val)
            if verbose:
                print(f"svm nu={h}: refit 5 folds")

    if model == "if":
        # Assert score invariance once (fold 1, first non-production value).
        data = load_model_folds()
        scaler = FoldScaler()
        scaler.fit_per_fold(data["folds"])
        fold1 = data["folds"][0]
        tn = fold1["train"][fold1["train"]["Y"] == 0]
        c_alt = next(c for c in grid if c != PRODUCTION_IF_CONTAMINATION)
        det = IsolationForestDetector(contamination=c_alt).fit(scaler.transform(tn, 1))
        val_alt = det.score_samples(scaler.transform(fold1["val"], 1))
        assert np.allclose(val_alt, fold_scores[1]["val"]["if"]), \
            "IF score_samples is NOT contamination-invariant — surface must refit"
        if verbose:
            print(f"IF invariance asserted: contamination={c_alt} scores == cached "
                  f"(max abs diff {np.max(np.abs(val_alt - fold_scores[1]['val']['if'])):.2e})")

    from sklearn.metrics import fbeta_score
    rows = []
    for h in grid:
        aucpr = weighted_mean(np.array(
            [average_precision_score(y[fid], raw_scores[h][fid]) for fid in fids]), w)
        for tau in taus:
            f2 = weighted_mean(np.array(
                [fbeta_score(y[fid], (pct_scores[h][fid] >= tau).astype(int),
                             beta=2, zero_division=0) for fid in fids]), w)
            rows.append({"model": model, "hyperparam": h, "tau": float(tau),
                         "F2_w": f2, "AUC_PR_w": aucpr})
    df = pd.DataFrame(rows)
    if save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        name = "svm_nu_tau_surface" if model == "svm" else "if_contamination_tau_surface"
        df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    return df


def plot_hyperparam_tau_surface(surface: pd.DataFrame, model: str,
                                out_path: Path | None = None):
    """Heatmap hyperparam x tau of weighted F2 + side column of weighted AUC-PR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hname = "nu" if model == "svm" else "contamination"
    prod = PRODUCTION_SVM["nu"] if model == "svm" else PRODUCTION_IF_CONTAMINATION
    piv = surface.pivot(index="hyperparam", columns="tau", values="F2_w")
    aucpr = surface.groupby("hyperparam")["AUC_PR_w"].first()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 3.6), width_ratios=[12, 1.2])
    im = ax.imshow(piv.to_numpy(), cmap="viridis", aspect="auto",
                   extent=[piv.columns.min(), piv.columns.max(), len(piv) - 0.5, -0.5])
    ax.set_yticks(range(len(piv)), [str(i) for i in piv.index])
    ax.axvline(PRODUCTION_TAU, color="#d73027", ls="--", lw=1.3)
    ax.plot([], [], color="#d73027", ls="--", label=f"production tau={PRODUCTION_TAU:.3f}")
    prod_row = list(piv.index).index(prod)
    ax.plot(piv.columns.min(), prod_row, marker=">", color="#d73027", clip_on=False)
    ax.set_xlabel("tau")
    ax.set_ylabel(hname)
    ax.set_title(f"{model.upper()}: n_pos-weighted F2 ({hname} x tau); "
                 f"arrow = production {hname}={prod}")
    ax.legend(loc="lower left", fontsize=8)
    fig.colorbar(im, ax=ax, label="F2_w")

    im2 = ax2.imshow(aucpr.to_numpy().reshape(-1, 1), cmap="magma", aspect="auto")
    ax2.set_yticks(range(len(aucpr)), [str(i) for i in aucpr.index])
    ax2.set_xticks([])
    ax2.set_title("AUC-PR_w", fontsize=9)
    for i, v in enumerate(aucpr.to_numpy()):
        ax2.text(0, i, f"{v:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.tight_layout()
    name = "svm_nu_tau_surface" if model == "svm" else "if_contamination_tau_surface"
    out_path = out_path or FIGURES_DIR / f"{name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# WP4 — inputs for the routing-threshold grid (reuses the WP1 cache)
# --------------------------------------------------------------------------
def routing_inputs(fold_scores: dict | None = None, tau: float = PRODUCTION_TAU) -> dict:
    """Assemble the inputs of ``optimize_routing_thresholds`` from committed
    artifacts + the WP1 cache (no model refits).

    Ensemble flags per fold = soft-median percentile of the cached fold scores
    thresholded at the production tau; sub-scores and routing fold-vals are the
    committed parquets; prices/TC replicate the notebook-07 setup.
    """
    from src.data_loader import load_dataset

    fold_scores = fold_scores or load_fold_scores()
    data = load_model_folds()

    raw = load_dataset(verbose=False)
    prices = pd.DataFrame(index=raw.index)
    prices["equity"] = raw[["MXUS", "MXEU", "MXJP"]].pct_change().mean(axis=1)
    prices["bond"] = raw["LUACTRUU"].pct_change()
    prices["gold"] = raw["XAUBGNL"].pct_change()
    prices["cash"] = (raw["USGG3M"] / 100.0) / 52.0
    prices["mbs"] = raw["LMBITR"].pct_change()
    tc = {"equity": 5, "bond": 5, "gold": 8, "cash": 2, "mbs": 20}

    ens_pct = _ensemble_val_percentiles(fold_scores, "soft_median")
    sub_dir = ROOT / "data" / "processed" / "subscores" / "folds"
    wf_dir = ROOT / "data" / "processed" / "walkforward"
    ensemble_per_fold, subscores_per_fold, folds_routing, y_per_fold = {}, {}, {}, {}
    for fold in data["folds"]:
        fid, val = fold["fold_id"], fold["val"]
        ensemble_per_fold[fid] = pd.Series((ens_pct[fid] >= tau).astype(int),
                                           index=val.index)
        y_per_fold[fid] = val["Y"]
        subscores_per_fold[fid] = pd.read_parquet(
            sub_dir / f"fold_{fid}_val.parquet").set_index("date_index")
        folds_routing[fid] = pd.read_parquet(wf_dir / f"routing_fold_{fid}_val.parquet")
    return {"ensemble_per_fold": ensemble_per_fold,
            "subscores_per_fold": subscores_per_fold,
            "folds_routing": folds_routing, "y_per_fold": y_per_fold,
            "prices": prices, "tc": tc}


if __name__ == "__main__":
    fs = cache_fold_scores()
    t0 = time.time()
    curve = tau_curve(fs)
    stab = tau_stability(curve)
    print(f"tau sweep + stability in {time.time() - t0:.1f}s "
          f"(cache hit -> no refits)")
    fig = plot_tau_curve(curve)
    print(f"wrote {TABLES_DIR / 'tau_curve.csv'}, {TABLES_DIR / 'tau_stability.csv'}, {fig}")
    summ = stab[stab["fold"] == "summary"]
    print(summ.to_string(index=False))
