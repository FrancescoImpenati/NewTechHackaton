"""Temporal train / CV / test split that respects time ordering.

Two aligned feature sets are produced from the cleaned Prompt-3.1 outputs:

* **models**  — ``features_stationary_clean.parquet`` (40 cols, incl. ``Y``)
  concatenated column-wise with ``spreads_clean.parquet`` (16 cols) = 56 cols.
* **routing** — ``routing_triggers.parquet`` (14 trigger cols, no ``Y``).

Both are sliced on the *same* date boundaries (the row counts differ only
because the routing triggers start later, after a 52-week rolling warm-up):

* train: ... <= 2015-12-31  (GFC 2008, euro crisis 2011, taper tantrum 2013)
* cv   : 2016-01-01 .. 2018-12-31  (Q4-2018 selloff)
* test : 2019-01-01 .. end  (COVID-19 2020)

The split is a pure chronological cut — **never** ``train_test_split`` /
shuffling — so there is no look-ahead leakage across the boundaries.

    python -m src.splits
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import PROJECT_ROOT, TARGET_COL

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"

FEATURES_CLEAN_PATH = PROCESSED_DIR / "features_stationary_clean.parquet"
SPREADS_CLEAN_PATH = PROCESSED_DIR / "spreads_clean.parquet"
ROUTING_PATH = PROCESSED_DIR / "routing_triggers.parquet"

# Known stress episodes (inclusive windows) used for the per-split crisis report
# and the sanity checks.
KNOWN_CRISES: dict[str, tuple[str, str]] = {
    "GFC 2008": ("2007-07-01", "2009-06-30"),
    "Euro debt crisis 2011": ("2011-07-01", "2012-07-31"),
    "Taper tantrum 2013": ("2013-05-01", "2013-09-30"),
    "Q4-2018 selloff": ("2018-10-01", "2018-12-31"),
    "COVID-19 crash": ("2020-02-15", "2020-04-30"),
}


def _load_model_features() -> pd.DataFrame:
    """Concatenate the cleaned stationary features and spreads (inner-aligned)."""
    feats = pd.read_parquet(FEATURES_CLEAN_PATH)
    spreads = pd.read_parquet(SPREADS_CLEAN_PATH)
    combined = pd.concat([feats, spreads], axis=1, join="inner")
    return combined.sort_index()


def _load_routing_features() -> pd.DataFrame:
    return pd.read_parquet(ROUTING_PATH).sort_index()


def _slice(df: pd.DataFrame, train_end: pd.Timestamp, cv_end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Chronological cut into train/cv/test by date masks (no shuffling)."""
    idx = df.index
    return {
        "train": df[idx <= train_end],
        "cv": df[(idx > train_end) & (idx <= cv_end)],
        "test": df[idx > cv_end],
    }


def _crises_in(index: pd.DatetimeIndex) -> list[str]:
    captured = []
    for name, (start, end) in KNOWN_CRISES.items():
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        if mask.any():
            captured.append(name)
    return captured


def _assert_no_overlap(splits: dict[str, pd.DataFrame], label: str) -> None:
    tr, cv, te = splits["train"].index, splits["cv"].index, splits["test"].index
    assert tr.max() < cv.min(), f"{label}: train/cv overlap"
    assert cv.max() < te.min(), f"{label}: cv/test overlap"


def _print_report(splits: dict[str, pd.DataFrame], overall_prev: float) -> None:
    print("\nPer-split report (model feature set):")
    print(f"{'split':<6} {'rows':>5} {'period':<25} {'Y=1 prev':>9}  crises captured")
    for name in ("train", "cv", "test"):
        s = splits[name]
        prev = s[TARGET_COL].mean()
        period = f"{s.index.min().date()} -> {s.index.max().date()}"
        crises = ", ".join(_crises_in(s.index)) or "-"
        print(f"{name:<6} {len(s):>5} {period:<25} {prev:>8.1%}  {crises}")
    print(f"\nOverall Y=1 prevalence: {overall_prev:.1%}")


def _print_sanity(splits: dict[str, pd.DataFrame], overall_prev: float) -> None:
    print("\nSanity checks:")

    def in_window(index, key):
        s, e = KNOWN_CRISES[key]
        return ((index >= pd.Timestamp(s)) & (index <= pd.Timestamp(e))).any()

    checks = [
        ("COVID-19 (Feb-Apr 2020) in TEST", in_window(splits["test"].index, "COVID-19 crash")),
        ("GFC 2008 in TRAIN", in_window(splits["train"].index, "GFC 2008")),
        ("Q4-2018 selloff in CV", in_window(splits["cv"].index, "Q4-2018 selloff")),
    ]
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
        assert ok, f"Sanity check failed: {desc}"

    # Prevalence: reasonably close to overall, but allowed to vary by regime.
    test_prev = splits["test"][TARGET_COL].mean()
    close = abs(test_prev - overall_prev) <= 0.10
    flag = "PASS" if close else "NOTE"
    print(f"  [{flag}] TEST Y=1 prevalence {test_prev:.1%} vs overall {overall_prev:.1%} "
          f"(regime-dependent; COVID lifts it)")


def time_split(
    train_end: str = "2015-12-31",
    cv_end: str = "2018-12-31",
    save: bool = True,
    verbose: bool = True,
) -> dict[str, object]:
    """Build aligned temporal train/CV/test splits for models and routing.

    Returns a dict with:
      * ``models``  -> {'train','cv','test'} (56 cols incl. ``Y``)
      * ``routing`` -> {'train','cv','test'} (14 trigger cols)
      * ``train_normals_only`` -> model train rows with ``Y == 0`` only
        (for unsupervised novelty detection: MVG, AE, OneClassSVM, IF)
    """
    train_end_ts = pd.Timestamp(train_end)
    cv_end_ts = pd.Timestamp(cv_end)

    model_df = _load_model_features()
    routing_df = _load_routing_features()

    model_splits = _slice(model_df, train_end_ts, cv_end_ts)
    routing_splits = _slice(routing_df, train_end_ts, cv_end_ts)

    _assert_no_overlap(model_splits, "models")
    _assert_no_overlap(routing_splits, "routing")

    train_normals_only = model_splits["train"][model_splits["train"][TARGET_COL] == 0]
    overall_prev = model_df[TARGET_COL].mean()

    if verbose:
        print("=" * 78)
        print("PROMPT 4 — TEMPORAL SPLIT (chronological, no shuffle)")
        print("=" * 78)
        print(f"Model feature set : {model_df.shape[1]} cols (incl. '{TARGET_COL}'), {len(model_df)} rows")
        print(f"Routing feature set: {routing_df.shape[1]} cols, {len(routing_df)} rows")
        print(f"Boundaries        : train <= {train_end} | cv <= {cv_end} | test > {cv_end}")
        _print_report(model_splits, overall_prev)
        print("\nRouting split rows: "
              + ", ".join(f"{k}={len(v)}" for k, v in routing_splits.items()))
        print(f"train_normals_only: {len(train_normals_only)} rows (Y==0 from model train)")
        _print_sanity(model_splits, overall_prev)

    if save:
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in model_splits.items():
            df.to_parquet(SPLITS_DIR / f"{name}.parquet")
        for name, df in routing_splits.items():
            df.to_parquet(SPLITS_DIR / f"routing_{name}.parquet")
        train_normals_only.to_parquet(SPLITS_DIR / "train_normals.parquet")
        if verbose:
            print(f"\nSaved splits -> {SPLITS_DIR}")

    return {
        "models": model_splits,
        "routing": routing_splits,
        "train_normals_only": train_normals_only,
    }


if __name__ == "__main__":
    time_split()
