"""Build + label the 2022-2025 out-of-sample extension (Phase 3, WP3).

Pulls the free raw panel (``src.free_data``), runs the UNMODIFIED ``src.features``
transforms, slices the frozen-36 model features (``selected_features.json``) and
the 14 routing triggers, and labels the post-2021 weeks with the operator-confirmed
ex-ante rule (``src.y_rule.label_ex_ante``: stress windows OR weekly VIX >= 30).

Coverage handling (documented, never silently substituted):
  * ``NEUTRALIZE_FEATURES`` have NO usable free instrument (BDIY, ECSURPUS,
    Italy-10Y -> GTITL10YR/it_de_10y/it_de_10y_chg4w) or a definitional break
    (US0001M: SOFR is secured, 1M LIBOR unsecured). They are left NaN here and
    filled with the development mean at inference time (-> standardized 0 = neutral),
    so the frozen model can score without inventing a signal. Every other feature
    uses its real (if noisier) free series — see proxy_validation.csv for fidelity.
  * Level shifts between Bloomberg and the free sources are absorbed by the
    return/diff transforms; raw levels are never re-based by hand.

Run from the repo root (uses the committed free-data cache; no network needed)::

    python -m src.extend_dataset
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import PROJECT_ROOT, TARGET_COL, load_dataset
from src.free_data import EXTENDED_DIR, ROUTING_TRIGGERS_14, _free_feature_frame
from src.y_rule import label_ex_ante

SEL_PATH = PROJECT_ROOT / "outputs" / "tables" / "selected_features.json"
DEV_END = "2018-12-31"          # frozen models are fit on 2000-2018
ORIGINAL_END = "2021-04-20"     # last week of the shipped panel
OOS_START = "2022-01-01"        # the labelled second holdout
OOS_END = "2025-12-31"

# Frozen-36 features with no usable free instrument / a definitional break.
# Filled with the development mean at inference (neutral standardized 0).
NEUTRALIZE_FEATURES = [
    "BDIY", "ECSURPUS", "GTITL10YR", "it_de_10y", "it_de_10y_chg4w", "US0001M",
]


def frozen_features() -> list[str]:
    sel = json.load(open(SEL_PATH))
    return sel["data_driven"] + sel["hypothesis"]


def build_extended(verbose: bool = True) -> dict:
    """Build the extended model panel (36 features + Y) and routing triggers,
    labelled with the ex-ante rule for the post-2021 weeks. Returns a dict and
    writes parquet + a coverage report under data/processed/extended/."""
    raw = pd.read_parquet(EXTENDED_DIR / "raw_panel_free.parquet")
    free = _free_feature_frame(raw)
    frozen = frozen_features()

    model = free[frozen].copy()
    triggers = free[ROUTING_TRIGGERS_14].copy()

    # Labels: original Y up to the shipped end, ex-ante rule afterwards.
    orig = load_dataset(verbose=False)
    y = pd.Series(index=model.index, dtype="float64")
    overlap = model.index <= pd.Timestamp(ORIGINAL_END)
    y[overlap] = orig[TARGET_COL].reindex(model.index[overlap]).to_numpy()
    post = model.index > pd.Timestamp(ORIGINAL_END)
    y[post] = label_ex_ante(model.index[post], raw["VIX"]).to_numpy()
    model[TARGET_COL] = y

    EXTENDED_DIR.mkdir(parents=True, exist_ok=True)
    model.to_parquet(EXTENDED_DIR / "model_panel_ext.parquet")
    triggers.to_parquet(EXTENDED_DIR / "routing_triggers_ext.parquet")

    oos = model.loc[OOS_START:OOS_END]
    coverage = {
        "span": [str(model.index.min().date()), str(model.index.max().date())],
        "n_rows_total": int(len(model)),
        "oos_window": [OOS_START, OOS_END],
        "n_rows_oos": int(len(oos)),
        "n_features": len(frozen),
        "neutralized_features": NEUTRALIZE_FEATURES,
        "neutralized_reason": "no usable free instrument or SOFR/LIBOR break; "
                              "filled with development mean at inference",
        "oos_label_rule": "ex_ante windows OR weekly VIX>=30 (y_rule.json)",
        "oos_y_prevalence": round(float(oos[TARGET_COL].mean()), 4),
        "oos_n_pos": int(oos[TARGET_COL].sum()),
        "features_with_gaps_in_oos": {c: int(oos[c].isna().sum())
                                      for c in frozen if oos[c].isna().any()},
    }
    with open(EXTENDED_DIR / "coverage_report.json", "w") as f:
        json.dump(coverage, f, indent=2)

    if verbose:
        print("=" * 80)
        print("WP3 — EXTENDED DATASET (free data -> src.features transforms -> label)")
        print("=" * 80)
        print(f"model panel : {model.shape} {coverage['span'][0]} -> {coverage['span'][1]}")
        print(f"OOS 2022-25 : {len(oos)} weeks | Y=1 prevalence {coverage['oos_y_prevalence']:.1%} "
              f"({coverage['oos_n_pos']} weeks)")
        print(f"neutralized (no free instrument / break): {NEUTRALIZE_FEATURES}")
        print(f"OOS Y by year:")
        print(oos[TARGET_COL].groupby(oos.index.year).agg(['sum', 'size']).to_string())
        print(f"\nSaved -> {EXTENDED_DIR}/model_panel_ext.parquet, "
              f"routing_triggers_ext.parquet, coverage_report.json")
    return {"model": model, "triggers": triggers, "coverage": coverage}


def sanity_overlap(verbose: bool = True) -> pd.DataFrame:
    """Reconstruct the 2015-2021 overlap from free data and report, per
    high-fidelity feature, the correlation vs the original transformed panel
    (the panel-level analogue of the WP2 harness)."""
    from src.free_data import _original_feature_frame
    raw = pd.read_parquet(EXTENDED_DIR / "raw_panel_free.parquet")
    free = _free_feature_frame(raw)
    orig = _original_feature_frame()
    frozen = frozen_features()
    grid = pd.date_range("2015-01-06", ORIGINAL_END, freq="W-TUE")
    rows = []
    for f in frozen:
        if f not in free.columns or f not in orig.columns:
            continue
        a = free[f].reindex(grid)
        b = orig[f].reindex(grid)
        m = a.notna() & b.notna()
        if m.sum() >= 30:
            rows.append({"feature": f, "n": int(m.sum()),
                         "corr": round(float(np.corrcoef(a[m], b[m])[0, 1]), 4)})
    df = pd.DataFrame(rows).sort_values("corr", ascending=False)
    if verbose:
        hi = (df["corr"] > 0.95).sum()
        print(f"\nOverlap sanity (2015-2021): {hi}/{len(df)} frozen features reconstruct "
              f"at corr>0.95; median corr {df['corr'].median():.3f}")
    return df


if __name__ == "__main__":
    build_extended()
    sanity_overlap()
