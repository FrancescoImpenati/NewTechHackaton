"""Load and clean the Bloomberg weekly risk-on/risk-off dataset.

The raw file is an Excel export (sheet ``Markets``) of weekly Bloomberg
observations. Column ``Y`` is the response variable: 1 = risk-off week,
0 = risk-on week. A companion ``Metadata`` sheet maps tickers to
human-readable descriptions.

Run as a script to print a coverage report:

    python -m src.data_loader
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_FILE = RAW_DIR / "04_May_Zenti_exercises.xlsx"

# Column holding the weekly date in the raw export (Italian "Data").
RAW_DATE_COL = "Data"
DATE_COL = "Date"
TARGET_COL = "Y"


def _resolve_path(path: str | Path | None) -> Path:
    if path is None:
        return DEFAULT_FILE
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_metadata(path: str | Path | None = None) -> pd.DataFrame:
    """Return the ticker -> description/type mapping from the Metadata sheet."""
    file_path = _resolve_path(path)
    meta = pd.read_excel(file_path, sheet_name="Metadata")
    meta.columns = ["variable", "description", "type"]
    meta["description"] = meta["description"].str.strip()
    return meta


def load_raw(path: str | Path | None = None, sheet: str = "Markets") -> pd.DataFrame:
    """Load the raw Markets sheet with a parsed weekly DatetimeIndex.

    No rows are dropped here; this is the unfiltered series so that the
    coverage report can describe the data before cleaning.
    """
    file_path = _resolve_path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            f"Place the Bloomberg Excel export in {RAW_DIR}."
        )

    df = pd.read_excel(file_path, sheet_name=sheet)
    if RAW_DATE_COL not in df.columns:
        raise KeyError(
            f"Expected date column '{RAW_DATE_COL}' not found. "
            f"Columns: {list(df.columns)}"
        )

    df = df.rename(columns={RAW_DATE_COL: DATE_COL})
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).set_index(DATE_COL)
    df.index.name = DATE_COL
    return df


def trim_to_common_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the warm-up period so every feature is fully populated.

    Features can begin and end on different dates. We restrict the panel to
    the window [max(first valid date), min(last valid date)] across all
    columns, then drop any straggler rows that still contain NaNs. This
    removes the leading rows with incomplete coverage without forward-filling
    values that were never observed.
    """
    if df.empty:
        return df

    first_valid = df.apply(lambda s: s.first_valid_index()).max()
    last_valid = df.apply(lambda s: s.last_valid_index()).min()
    trimmed = df.loc[first_valid:last_valid]

    before = len(trimmed)
    trimmed = trimmed.dropna(axis=0, how="any")
    dropped = before - len(trimmed)
    if dropped:
        print(f"[data_loader] Dropped {dropped} interior rows with NaNs after trimming.")
    return trimmed


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature coverage: first/last valid date, missing count, % missing, length."""
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        rows.append(
            {
                "feature": col,
                "first_valid": s.first_valid_index(),
                "last_valid": s.last_valid_index(),
                "n_obs": int(s.notna().sum()),
                "n_missing": n_missing,
                "pct_missing": round(100 * n_missing / n, 2) if n else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def print_coverage_report(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    """Print a human-readable coverage report for raw and cleaned series."""
    print("=" * 78)
    print("BLOOMBERG DATASET — COVERAGE REPORT")
    print("=" * 78)
    print(f"Raw series   : {len(raw):>5} rows | {raw.index.min().date()} -> {raw.index.max().date()}")
    print(f"Clean series : {len(clean):>5} rows | {clean.index.min().date()} -> {clean.index.max().date()}")
    print(f"Features      : {clean.shape[1]} columns (incl. target '{TARGET_COL}')")

    freq = raw.index.to_series().diff().dt.days.mode()
    if not freq.empty:
        print(f"Sampling      : ~{int(freq.iloc[0])} days between observations (weekly)")

    print("\nPer-feature coverage (raw series):")
    report = coverage_report(raw).copy()
    report["first_valid"] = pd.to_datetime(report["first_valid"]).dt.date
    report["last_valid"] = pd.to_datetime(report["last_valid"]).dt.date
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(report.to_string())

    total_missing = int(report["n_missing"].sum())
    print(f"\nTotal missing cells (raw): {total_missing}")
    if TARGET_COL in clean.columns:
        counts = clean[TARGET_COL].value_counts().sort_index()
        prevalence = clean[TARGET_COL].mean()
        print(
            f"Target balance (clean): risk-on(0)={counts.get(0, 0)}, "
            f"risk-off(1)={counts.get(1, 0)} | risk-off prevalence={prevalence:.1%}"
        )
    print("=" * 78)


def load_dataset(
    path: str | Path | None = None,
    sheet: str = "Markets",
    verbose: bool = True,
) -> pd.DataFrame:
    """Load, clean, and (optionally) report on the Bloomberg dataset.

    Returns the cleaned DataFrame indexed by weekly date, with full coverage
    across all features.
    """
    raw = load_raw(path, sheet=sheet)
    clean = trim_to_common_coverage(raw)
    if verbose:
        print_coverage_report(raw, clean)
    return clean


if __name__ == "__main__":
    load_dataset()
