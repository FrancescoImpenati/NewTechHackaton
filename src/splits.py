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
WALKFORWARD_DIR = PROCESSED_DIR / "walkforward"

FEATURES_CLEAN_PATH = PROCESSED_DIR / "features_stationary_clean.parquet"
SPREADS_CLEAN_PATH = PROCESSED_DIR / "spreads_clean.parquet"
ROUTING_PATH = PROCESSED_DIR / "routing_triggers.parquet"

# Boundary between the development set and the sealed out-of-sample test set.
DEV_END = "2018-12-31"

# Walk-forward fold definitions: (fold_id, train_end, val_end, crisis label).
# Train always starts at the dataset start (expanding window). The validation
# window opens ``embargo_weeks`` observations after ``train_end`` (the embargo
# gap), which lands it around February of the following year.
WALKFORWARD_FOLDS: list[tuple[int, str, str, str]] = [
    (1, "2006-12-31", "2009-12-31", "GFC 2008"),
    (2, "2009-12-31", "2012-12-31", "Euro 2011"),
    (3, "2012-12-31", "2014-12-31", "Taper 2013"),
    (4, "2014-12-31", "2016-12-31", "China-Oil 2015-16"),
    (5, "2016-12-31", "2018-12-31", "Q4 2018 selloff"),
]

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

    DEPRECATED: use walkforward_split() instead. Kept for backwards
    compatibility; later prompts rely only on walkforward_split().

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


def _build_walkforward(
    df: pd.DataFrame, embargo_weeks: int, has_target: bool
) -> dict[str, object]:
    """Build development/test split + expanding walk-forward folds for one set."""
    dev_end_ts = pd.Timestamp(DEV_END)
    dev = df[df.index <= dev_end_ts]
    test_holdout = df[df.index > dev_end_ts]

    cv_folds: list[dict] = []
    train_normals_per_fold: list[dict] = []
    for fid, train_end, val_end, crisis in WALKFORWARD_FOLDS:
        train_end_ts = pd.Timestamp(train_end)
        val_end_ts = pd.Timestamp(val_end)

        train = dev[dev.index <= train_end_ts]
        after = dev[dev.index > train_end_ts]
        # Embargo: exclude the first ``embargo_weeks`` observations after the
        # train end from BOTH train and val to kill 4w-lookback leakage.
        val_start_ts = after.index[embargo_weeks]
        val = dev[(dev.index >= val_start_ts) & (dev.index <= val_end_ts)]

        # Embargo / no-overlap verification.
        gap = dev[(dev.index > train.index.max()) & (dev.index < val.index.min())]
        assert train.index.max() < val.index.min(), f"fold {fid}: train/val overlap"
        assert len(gap) == embargo_weeks, (
            f"fold {fid}: embargo gap {len(gap)} != {embargo_weeks}"
        )

        cv_folds.append(
            {"fold_id": fid, "train": train, "val": val, "crisis_captured": crisis}
        )
        if has_target:
            train_normals_per_fold.append(
                {"fold_id": fid, "train_normals": train[train[TARGET_COL] == 0]}
            )

    out: dict[str, object] = {
        "development": dev,
        "test_holdout": test_holdout,
        "cv_folds": cv_folds,
    }
    if has_target:
        out["train_normals_per_fold"] = train_normals_per_fold
    return out


def _print_walkforward_report(models_wf: dict, embargo_weeks: int) -> None:
    print("\nWalk-forward folds (model feature set):")
    header = (
        f"{'fold':<5}{'train period':<27}{'val period':<27}"
        f"{'n_tr':>6}{'n_val':>6}{'Y=1%':>7}{'#Y=1':>6}  crisis"
    )
    print(header)
    for fold in models_wf["cv_folds"]:
        tr, val = fold["train"], fold["val"]
        prev = val[TARGET_COL].mean()
        n_pos = int(val[TARGET_COL].sum())
        tr_period = f"{tr.index.min().date()}->{tr.index.max().date()}"
        val_period = f"{val.index.min().date()}->{val.index.max().date()}"
        print(
            f"{fold['fold_id']:<5}{tr_period:<27}{val_period:<27}"
            f"{len(tr):>6}{len(val):>6}{prev:>6.1%}{n_pos:>6}  {fold['crisis_captured']}"
        )


def _print_walkforward_sanity(models_wf: dict) -> None:
    print("\nSanity checks:")

    # Test holdout must cover the COVID crash.
    test_idx = models_wf["test_holdout"].index
    s, e = KNOWN_CRISES["COVID-19 crash"]
    covid_ok = ((test_idx >= pd.Timestamp(s)) & (test_idx <= pd.Timestamp(e))).any()
    print(f"  [{'PASS' if covid_ok else 'FAIL'}] COVID-19 (Feb-Apr 2020) in test_holdout")
    assert covid_ok, "COVID not in test_holdout"

    # Per-fold prevalence and positive-count checks (non-blocking warnings).
    for fold in models_wf["cv_folds"]:
        val = fold["val"]
        fid = fold["fold_id"]
        prev = val[TARGET_COL].mean()
        n_pos = int(val[TARGET_COL].sum())

        if prev < 0.10:
            tag = "WARN"
        elif prev < 0.12:
            tag = "NOTE"  # below 12% but acceptable (esp. fold 5)
        else:
            tag = "PASS"
        print(f"  [{tag}] fold {fid} val Y=1 prevalence {prev:.1%}"
              + (" (< 10% — low-volatility regime)" if tag == "WARN" else ""))

        count_tag = "PASS" if n_pos > 20 else "WARN"
        print(f"  [{count_tag}] fold {fid} val absolute Y=1 count = {n_pos} (>20 expected)")


def _save_walkforward(result: dict) -> None:
    WALKFORWARD_DIR.mkdir(parents=True, exist_ok=True)
    for kind, prefix in (("models", ""), ("routing", "routing_")):
        wf = result[kind]
        wf["development"].to_parquet(WALKFORWARD_DIR / f"{prefix}development.parquet")
        wf["test_holdout"].to_parquet(WALKFORWARD_DIR / f"{prefix}test_holdout.parquet")
        for fold in wf["cv_folds"]:
            i = fold["fold_id"]
            fold["train"].to_parquet(WALKFORWARD_DIR / f"{prefix}fold_{i}_train.parquet")
            fold["val"].to_parquet(WALKFORWARD_DIR / f"{prefix}fold_{i}_val.parquet")
        if "train_normals_per_fold" in wf:
            for item in wf["train_normals_per_fold"]:
                i = item["fold_id"]
                item["train_normals"].to_parquet(
                    WALKFORWARD_DIR / f"{prefix}fold_{i}_train_normals.parquet"
                )


def walkforward_split(
    embargo_weeks: int = 4,
    save: bool = True,
    verbose: bool = True,
) -> dict[str, object]:
    """Purged expanding walk-forward CV with embargo + sealed test holdout.

    Returns ``{'models': wf_models, 'routing': wf_routing}`` where each ``wf``
    dict has the structure::

        {
            'development': df_dev,            # all weeks <= 2018-12-31
            'test_holdout': df_test,          # 2019 -> end (contains COVID)
            'cv_folds': [                     # 5 expanding folds
                {'fold_id', 'train', 'val', 'crisis_captured'}, ...
            ],
            'train_normals_per_fold': [       # models only (needs Y)
                {'fold_id', 'train_normals'}, ...
            ],
        }

    ``embargo_weeks`` observations immediately after each fold's train end are
    excluded from both train and validation, preventing leakage from the 4-week
    lookback features. The same boundaries are applied to the 56-column model
    feature set and the 14-column routing feature set.
    """
    model_df = _load_model_features()
    routing_df = _load_routing_features()

    result = {
        "models": _build_walkforward(model_df, embargo_weeks, has_target=True),
        "routing": _build_walkforward(routing_df, embargo_weeks, has_target=False),
    }

    if verbose:
        print("=" * 78)
        print("PROMPT 4.1 — PURGED WALK-FORWARD CV (5 expanding folds + embargo)")
        print("=" * 78)
        dev, test = result["models"]["development"], result["models"]["test_holdout"]
        print(f"Embargo            : {embargo_weeks} weeks")
        print(f"Development (models): {len(dev)} weeks "
              f"({dev.index.min().date()} -> {dev.index.max().date()})")
        print(f"Test holdout (models): {len(test)} weeks "
              f"({test.index.min().date()} -> {test.index.max().date()})")
        print(f"Routing dev/test    : {len(result['routing']['development'])} / "
              f"{len(result['routing']['test_holdout'])} weeks")
        _print_walkforward_report(result["models"], embargo_weeks)
        _print_walkforward_sanity(result["models"])

    if save:
        _save_walkforward(result)
        if verbose:
            print(f"\nSaved walk-forward splits -> {WALKFORWARD_DIR}")

    return result


if __name__ == "__main__":
    walkforward_split()
