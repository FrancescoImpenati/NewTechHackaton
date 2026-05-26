"""Stationarity transforms for the Bloomberg weekly dataset.

Each raw series is mapped to one transformation *family* based on its economic
nature (see ``TRANSFORM_MAP``):

* ``log_return`` — weekly log-returns ``ln(x_t) - ln(x_{t-1})`` for **price-like
  series**: equity indices, FX spot rates, gold, oil, commodity indices and
  bond *total-return* indices. Prices are non-stationary in level (random-walk
  with drift) but their log-returns are approximately stationary.
* ``diff`` — first differences ``x_t - x_{t-1}`` for **yields, rates and
  spreads**. These are already expressed in percentage points; differencing
  removes the stochastic trend while keeping an interpretable "change in bps".
* ``level`` — no transformation for series that are **already stationary**:
  the VIX (mean-reverting volatility), the US Economic Surprise Index, and the
  binary target ``Y``.

Run as a script to transform the data, ADF-test every feature, print the
results table and write ``data/processed/features_stationary.parquet``.

    python -m src.features
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.data_loader import PROJECT_ROOT, TARGET_COL, load_dataset

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "features_stationary.parquet"

LOG_RETURN = "log_return"
DIFF = "diff"
LEVEL = "level"

# Per-column transformation choice, grouped by family for documentation.
TRANSFORM_MAP: dict[str, str] = {
    # --- Prices -> weekly log-returns -------------------------------------
    # Equity indices (MSCI)
    "MXUS": LOG_RETURN,
    "MXEU": LOG_RETURN,
    "MXJP": LOG_RETURN,
    "MXBR": LOG_RETURN,
    "MXRU": LOG_RETURN,
    "MXIN": LOG_RETURN,
    "MXCN": LOG_RETURN,
    # FX spot rates
    "DXY": LOG_RETURN,
    "GBP": LOG_RETURN,
    "JPY": LOG_RETURN,
    # Commodities / gold / oil
    "XAUBGNL": LOG_RETURN,   # Gold spot
    "Cl1": LOG_RETURN,       # 1st CL (crude oil) future
    "CRY": LOG_RETURN,       # CRB commodity index
    "BDIY": LOG_RETURN,      # Baltic Dry Index (trending freight-rate index)
    # Bond total-return indices (price-like levels)
    "EMUSTRUU": LOG_RETURN,
    "LF94TRUU": LOG_RETURN,
    "LF98TRUU": LOG_RETURN,
    "LG30TRUU": LOG_RETURN,
    "LMBITR": LOG_RETURN,
    "LP01TREU": LOG_RETURN,
    "LUACTRUU": LOG_RETURN,
    "LUMSTRUU": LOG_RETURN,
    # --- Yields / rates -> first differences ------------------------------
    "GT10": DIFF,        # US 10Y
    "USGG2YR": DIFF,
    "USGG30YR": DIFF,
    "USGG3M": DIFF,
    "US0001M": DIFF,     # USD 1M LIBOR
    "EONIA": DIFF,       # EUR overnight rate
    "GTDEM2Y": DIFF,
    "GTDEM10Y": DIFF,
    "GTDEM30Y": DIFF,
    "GTGBP2Y": DIFF,
    "GTGBP20Y": DIFF,
    "GTGBP30Y": DIFF,
    "GTITL2YR": DIFF,
    "GTITL10YR": DIFF,
    "GTITL30YR": DIFF,
    "GTJPY2YR": DIFF,
    "GTJPY10YR": DIFF,
    "GTJPY30YR": DIFF,
    # --- Already stationary -> keep level ---------------------------------
    "VIX": LEVEL,        # mean-reverting volatility index
    "ECSURPUS": LEVEL,   # economic surprise index (oscillates around 0)
    TARGET_COL: LEVEL,   # binary response variable (passthrough)
}


def _apply(series: pd.Series, transform: str) -> pd.Series:
    if transform == LOG_RETURN:
        if (series <= 0).any():
            raise ValueError(
                f"Cannot take log-returns of '{series.name}': non-positive values present."
            )
        return np.log(series).diff()
    if transform == DIFF:
        return series.diff()
    if transform == LEVEL:
        return series
    raise ValueError(f"Unknown transform '{transform}' for column '{series.name}'.")


def make_stationary(
    df: pd.DataFrame,
    transform_map: dict[str, str] = TRANSFORM_MAP,
    dropna: bool = True,
) -> pd.DataFrame:
    """Apply the per-family transformation in ``transform_map`` to each column.

    Columns missing from the map default to ``level`` (with a warning). The
    leading row produced by differencing/log-returns is dropped when
    ``dropna`` is True so the result is fully populated.
    """
    out = {}
    for col in df.columns:
        transform = transform_map.get(col)
        if transform is None:
            print(f"[features] '{col}' not in TRANSFORM_MAP -> defaulting to '{LEVEL}'.")
            transform = LEVEL
        out[col] = _apply(df[col], transform)

    result = pd.DataFrame(out, index=df.index)
    if dropna:
        result = result.dropna(axis=0, how="any")
    return result


def adf_table(df: pd.DataFrame, signif: float = 0.05) -> pd.DataFrame:
    """Run the Augmented Dickey-Fuller test on every column.

    Returns one row per feature with the ADF statistic, p-value, lags used,
    number of observations, the 1%/5%/10% critical values, the applied
    transform, and a boolean ``stationary`` flag (p-value < ``signif``).
    """
    rows = []
    for col in df.columns:
        series = df[col].dropna()
        stat, pvalue, usedlag, nobs, crit, _ = adfuller(series, autolag="AIC")
        rows.append(
            {
                "feature": col,
                "transform": TRANSFORM_MAP.get(col, LEVEL),
                "adf_stat": round(stat, 4),
                "p_value": round(pvalue, 5),
                "used_lag": usedlag,
                "n_obs": nobs,
                "crit_1%": round(crit["1%"], 3),
                "crit_5%": round(crit["5%"], 3),
                "crit_10%": round(crit["10%"], 3),
                "stationary": bool(pvalue < signif),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def build(verbose: bool = True) -> pd.DataFrame:
    """Load, transform to stationarity, ADF-test, save parquet, return the df."""
    df = load_dataset(verbose=verbose)
    stationary = make_stationary(df)

    table = adf_table(stationary)
    if verbose:
        print("\n" + "=" * 78)
        print("AUGMENTED DICKEY-FULLER TEST — transformed features")
        print("=" * 78)
        with pd.option_context("display.max_rows", None, "display.width", 140):
            print(table.to_string())
        n_stat = int(table["stationary"].sum())
        print(f"\nStationary at 5%: {n_stat}/{len(table)} features")
        non_stat = table.index[~table["stationary"]].tolist()
        if non_stat:
            print(f"NOT stationary at 5%: {non_stat}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stationary.to_parquet(OUTPUT_PATH)
    if verbose:
        print(f"\nSaved {stationary.shape[0]} x {stationary.shape[1]} -> {OUTPUT_PATH}")
    return stationary


if __name__ == "__main__":
    build()
