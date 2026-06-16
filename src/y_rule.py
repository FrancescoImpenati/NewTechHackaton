"""Reverse-engineer the Y labelling rule (Phase 3, WP1).

The raw dataset ships the binary ``Y`` (1 = risk-off week) but NOT the rule that
generated it. To label the 2022-2025 extension we must first recover that rule
from the 2000-2021 panel — or prove no simple parametric rule reproduces it, in
which case we fall back to ex-ante public stress windows (decided by the
operator, never fabricated here).

We fit candidate parametric rule families on a GLOBAL equity basket and pick the
one whose flags match the shipped ``Y`` on the largest fraction of weeks:

    (i)   rolling max drawdown over a w-week window <= -d
    (ii)  k-week realized vol >= the p-th percentile of its own history
    (iii) trailing/forward k-week return <= r
    (iv)  AND / OR combinations of the above (incl. a VIX-level condition)

Each rule is also tested at temporal alignments lag in {-2,-1,0,+1,+2} (does the
rule lead/lag Y?). The basket is an equal-weight developed-market composite
(MXUS, MXEU, MXJP); MXUS-only is tried as a contrast.

DECISION GATE (printed and acted on):
    match >= 99%      -> adopt the rule;
    97% <= match < 99% -> adopt, list residual mismatch weeks;
    match < 97%       -> STOP, report the top-3 rules, DO NOT fabricate labels;
                         the fallback is operator-confirmed ex-ante windows.

Run from the repo root::

    python -m src.y_rule
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import PROJECT_ROOT, TARGET_COL, load_dataset

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
Y_RULE_PATH = TABLES_DIR / "y_rule.json"

DM_EQUITY = ["MXUS", "MXEU", "MXJP"]      # developed-market basket
WEEKS_PER_YEAR = 52

# Grids (kept explicit so the search is fully auditable / reproducible).
DD_WINDOWS = [8, 13, 26, 52]
DD_DEPTHS = np.round(np.arange(0.04, 0.40, 0.01), 2)
VOL_WINDOWS = [4, 8, 13, 26]
VOL_PCTILES = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
RET_WINDOWS = [2, 4, 8, 13, 26]
RET_THRESHOLDS = np.round(np.arange(-0.25, -0.01, 0.005), 3)
VIX_THRESHOLDS = np.round(np.arange(15, 45, 1), 0)
LAGS = [-2, -1, 0, 1, 2]


# --------------------------------------------------------------------------
# Basket + signal construction
# --------------------------------------------------------------------------
def equity_composite(df: pd.DataFrame, cols=DM_EQUITY) -> pd.Series:
    """Equal-weight price index of ``cols`` (each normalized to 1 at the start)."""
    norm = df[cols] / df[cols].iloc[0]
    return norm.mean(axis=1)


def _drawdown(price: pd.Series, w: int) -> pd.Series:
    return price / price.rolling(w, min_periods=1).max() - 1.0


def _realized_vol(price: pd.Series, k: int) -> pd.Series:
    return np.log(price).diff().rolling(k).std() * np.sqrt(WEEKS_PER_YEAR)


def _trailing_return(price: pd.Series, k: int) -> pd.Series:
    return price / price.shift(k) - 1.0


def _forward_return(price: pd.Series, k: int) -> pd.Series:
    return price.shift(-k) / price - 1.0


def _match(pred: pd.Series, y: pd.Series, lag: int) -> float:
    """Fraction of identical weeks at temporal alignment ``lag`` (rule shifted
    by ``lag`` weeks; NaNs from shifting/rolling count as 0 = risk-on)."""
    p = pred.shift(lag).reindex(y.index).fillna(0).astype(int)
    return float((p == y).mean())


# --------------------------------------------------------------------------
# Family search
# --------------------------------------------------------------------------
def _search_single(signals: dict[str, pd.Series], y: pd.Series) -> list[dict]:
    """Scan every single-condition rule across all baskets, params and lags."""
    rows = []
    for basket, sig in signals.items():
        px = sig["price"]
        # (i) drawdown
        for w in DD_WINDOWS:
            dd = _drawdown(px, w)
            for d in DD_DEPTHS:
                pred = (dd <= -d).astype(int)
                for lag in LAGS:
                    rows.append({"family": "drawdown", "basket": basket,
                                 "params": {"w": int(w), "d": float(d)},
                                 "lag": lag, "match": _match(pred, y, lag)})
        # (ii) realized vol >= percentile
        for k in VOL_WINDOWS:
            v = _realized_vol(px, k)
            for p in VOL_PCTILES:
                thr = v.quantile(p)
                pred = (v >= thr).astype(int)
                for lag in LAGS:
                    rows.append({"family": "realized_vol", "basket": basket,
                                 "params": {"k": int(k), "pctile": float(p),
                                            "thr": float(thr)},
                                 "lag": lag, "match": _match(pred, y, lag)})
        # (iii) trailing / forward return <= r
        for k in RET_WINDOWS:
            for kind, fn in (("trailing", _trailing_return), ("forward", _forward_return)):
                ret = fn(px, k)
                for r in RET_THRESHOLDS:
                    pred = (ret <= r).astype(int)
                    for lag in LAGS:
                        rows.append({"family": f"{kind}_return", "basket": basket,
                                     "params": {"k": int(k), "r": float(r)},
                                     "lag": lag, "match": _match(pred, y, lag)})
    # (iv-a) VIX level (basket-independent)
    vix = signals[next(iter(signals))]["vix"]
    for thr in VIX_THRESHOLDS:
        pred = (vix >= thr).astype(int)
        for lag in LAGS:
            rows.append({"family": "vix_level", "basket": "VIX",
                         "params": {"thr": float(thr)}, "lag": lag,
                         "match": _match(pred, y, lag)})
    return rows


def _search_combos(signals: dict[str, pd.Series], y: pd.Series) -> list[dict]:
    """Scan AND / OR two-condition combinations on the MXUS basket + VIX
    (the two strongest single signals), over a focused 2-D grid at lag 0."""
    px = signals["MXUS"]["price"]
    vix = signals["MXUS"]["vix"]
    dd13 = _drawdown(px, 13)
    vol8 = _realized_vol(px, 8)
    rows = []
    for d in np.round(np.arange(0.05, 0.25, 0.01), 2):
        for thr in VIX_THRESHOLDS:
            cond_dd = dd13 <= -d
            cond_vix = vix >= thr
            for op, pred in (("or", (cond_dd | cond_vix)), ("and", (cond_dd & cond_vix))):
                pred = pred.astype(int)
                rows.append({"family": f"drawdown_{op}_vix", "basket": "MXUS",
                             "params": {"w": 13, "d": float(d), "vix_thr": float(thr)},
                             "lag": 0, "match": _match(pred, y, 0)})
    for p in VOL_PCTILES:
        vthr = vol8.quantile(p)
        for thr in VIX_THRESHOLDS:
            cond_v = vol8 >= vthr
            cond_vix = vix >= thr
            for op, pred in (("or", (cond_v | cond_vix)), ("and", (cond_v & cond_vix))):
                pred = pred.astype(int)
                rows.append({"family": f"realized_vol_{op}_vix", "basket": "MXUS",
                             "params": {"k": 8, "pctile": float(p), "vthr": float(vthr),
                                        "vix_thr": float(thr)},
                             "lag": 0, "match": _match(pred, y, 0)})
    return rows


def search_rules(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run the full family + combo search; return a ranked match table."""
    df = load_dataset(verbose=False) if df is None else df
    y = df[TARGET_COL].astype(int)
    signals = {
        "DM": {"price": equity_composite(df), "vix": df["VIX"]},
        "MXUS": {"price": df["MXUS"], "vix": df["VIX"]},
    }
    rows = _search_single(signals, y) + _search_combos(signals, y)
    out = pd.DataFrame(rows).sort_values("match", ascending=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Reconstruct a rule's flags (for confusion / by-year diagnostics)
# --------------------------------------------------------------------------
def reconstruct(rule: dict, df: pd.DataFrame) -> pd.Series:
    """Rebuild the 0/1 flag series for a ranked-table row at its alignment."""
    fam, basket, p, lag = rule["family"], rule["basket"], rule["params"], rule["lag"]
    px = equity_composite(df) if basket == "DM" else df.get("MXUS")
    vix = df["VIX"]
    if fam == "drawdown":
        pred = (_drawdown(px, p["w"]) <= -p["d"])
    elif fam == "realized_vol":
        pred = (_realized_vol(px, p["k"]) >= p["thr"])
    elif fam == "trailing_return":
        pred = (_trailing_return(px, p["k"]) <= p["r"])
    elif fam == "forward_return":
        pred = (_forward_return(px, p["k"]) <= p["r"])
    elif fam == "vix_level":
        pred = (vix >= p["thr"])
    elif fam.startswith("drawdown_"):
        cond = _drawdown(px, p["w"]) <= -p["d"]
        pred = (cond | (vix >= p["vix_thr"])) if "_or_" in fam else (cond & (vix >= p["vix_thr"]))
    elif fam.startswith("realized_vol_"):
        cond = _realized_vol(px, p["k"]) >= p["vthr"]
        pred = (cond | (vix >= p["vix_thr"])) if "_or_" in fam else (cond & (vix >= p["vix_thr"]))
    else:
        raise ValueError(f"unknown family {fam}")
    return pred.astype(int).shift(lag).reindex(df.index).fillna(0).astype(int)


def diagnostics(rule: dict, df: pd.DataFrame) -> dict:
    """Confusion matrix + by-year match table for a single rule vs Y."""
    y = df[TARGET_COL].astype(int)
    pred = reconstruct(rule, df)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    by_year = (pd.DataFrame({"y": y, "pred": pred, "match": (y == pred).astype(int)})
               .groupby(df.index.year)
               .agg(n=("y", "size"), y_pos=("y", "sum"),
                    pred_pos=("pred", "sum"), match=("match", "mean")))
    by_year["match"] = (by_year["match"] * 100).round(1)
    return {"confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "by_year": by_year}


# --------------------------------------------------------------------------
# Decision gate + persistence
# --------------------------------------------------------------------------
def resolve(df: pd.DataFrame | None = None, save: bool = True, verbose: bool = True) -> dict:
    """Search, apply the decision gate, persist y_rule.json."""
    df = load_dataset(verbose=False) if df is None else df
    ranked = search_rules(df)
    top3 = ranked.head(3).to_dict("records")
    best = top3[0]
    match = best["match"]
    gate = ("adopt" if match >= 0.99
            else "adopt_with_residuals" if match >= 0.97
            else "stop")
    diag = diagnostics(best, df)

    resolved = {
        "status": gate,
        "best_match_pct": round(match * 100, 2),
        "best_rule": best,
        "top3_rules": top3,
        "confusion": diag["confusion"],
        "by_year_match": {int(k): float(v) for k, v in diag["by_year"]["match"].items()},
        "n_obs": int(len(df)),
        "y_prevalence": round(float(df[TARGET_COL].mean()), 4),
        "note": ("match below 97%: NO 2022-2025 labels fabricated; awaiting "
                 "operator-confirmed ex-ante stress windows as the fallback."
                 if gate == "stop" else
                 "rule adopted to label the 2022-2025 extension."),
    }
    if save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        with open(Y_RULE_PATH, "w") as f:
            json.dump(resolved, f, indent=2)

    if verbose:
        print("=" * 78)
        print("WP1 — REVERSE-ENGINEERING THE Y LABELLING RULE (2000-2021)")
        print("=" * 78)
        print(f"Y: {int(df[TARGET_COL].sum())}/{len(df)} risk-off weeks "
              f"({df[TARGET_COL].mean():.1%} prevalence)\n")
        print("TOP-3 RULES (by % of identical weeks vs shipped Y):")
        for i, r in enumerate(top3, 1):
            print(f"  {i}. {r['match']*100:5.2f}%  {r['family']:<22} basket={r['basket']:<5} "
                  f"lag={r['lag']:+d}  {r['params']}")
        c = diag["confusion"]
        print(f"\nBest-rule confusion vs Y: TP={c['tp']} FP={c['fp']} "
              f"FN={c['fn']} TN={c['tn']}")
        print("\nBy-year match % (best rule):")
        print(diag["by_year"].to_string())
        print("\n" + "=" * 78)
        print(f"DECISION GATE: best match = {match*100:.2f}%  ->  {gate.upper()}")
        if gate == "stop":
            print("  match < 97%: STOP. No 2022-2025 labels fabricated.")
            print("  Fallback = operator-confirmed ex-ante public stress windows.")
        elif gate == "adopt_with_residuals":
            y = df[TARGET_COL].astype(int)
            pred = reconstruct(best, df)
            mis = df.index[(pred != y)]
            print(f"  97-99%: adopt; {len(mis)} residual mismatch weeks listed in y_rule.json.")
            resolved["residual_mismatch_weeks"] = [str(d.date()) for d in mis]
            if save:
                with open(Y_RULE_PATH, "w") as f:
                    json.dump(resolved, f, indent=2)
        else:
            print("  >= 99%: adopt the rule.")
        print("=" * 78)
        if save:
            print(f"\nSaved -> {Y_RULE_PATH}")
    return resolved


if __name__ == "__main__":
    resolve()
