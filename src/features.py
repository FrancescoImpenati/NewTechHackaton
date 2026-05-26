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
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PATH = PROCESSED_DIR / "features_stationary.parquet"
SPREADS_PATH = PROCESSED_DIR / "spreads.parquet"

# Number of weekly observations used as the "4-week" horizon and the
# realized-vol window (~20 trading days = 4 weeks).
HORIZON_4W = 4
REALIZED_VOL_WEEKS = 4
WEEKS_PER_YEAR = 52

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


def adf_table(
    df: pd.DataFrame,
    signif: float = 0.05,
    transform_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Run the Augmented Dickey-Fuller test on every column.

    Returns one row per feature with the ADF statistic, p-value, lags used,
    number of observations, the 1%/5%/10% critical values, the applied
    transform label, and a boolean ``stationary`` flag (p-value < ``signif``).
    ``transform_map`` labels the transform shown per column (defaults to
    ``TRANSFORM_MAP``); unknown columns are labelled ``level``.
    """
    labels = TRANSFORM_MAP if transform_map is None else transform_map
    rows = []
    for col in df.columns:
        series = df[col].dropna()
        stat, pvalue, usedlag, nobs, crit, _ = adfuller(series, autolag="AIC")
        rows.append(
            {
                "feature": col,
                "transform": labels.get(col, LEVEL),
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


# ---------------------------------------------------------------------------
# Engineered spreads / relative-value features
# ---------------------------------------------------------------------------
# Documentation of every constructed feature and the kind of series it is.
# Yield/sovereign spreads are true differences in percentage points. Credit/EM
# "spreads" are RETURN-BASED PROXIES: the dataset ships credit *total-return
# indices* (not yields) and has no Treasury or Global-Aggregate total-return
# index, so we proxy them with log price-index ratios (safe leg minus risky
# leg) — a rising value means the risky leg is underperforming, i.e. spreads
# widening / risk-off. ``MSCI World`` is proxied by ``MXUS`` and ``Global
# Aggregate`` by ``LUACTRUU`` (US IG Corporate), neither being in the dataset.
SPREAD_KINDS: dict[str, str] = {
    "us_term_10y_3m": "yield_spread",
    "us_term_10y_2y": "yield_spread",
    "de_term_10y_2y": "yield_spread",
    "it_de_10y": "sovereign_spread",
    "us_de_10y": "sovereign_spread",
    "hy_spread": "credit_proxy(logratio)",
    "hy_ig_spread": "credit_proxy(logratio)",
    "em_spread": "credit_proxy(logratio)",
}
# Standalone derived features (single series each, not level+change spreads).
STANDALONE_KINDS: dict[str, str] = {
    "equity_bond_rot": "ret4w_diff",
    "gold_oil_ratio": "ratio_level",
    "vrp": "level",
    "jpy_strength": "neg_ret4w",
}


def _logret_4w(series: pd.Series) -> pd.Series:
    return np.log(series).diff(HORIZON_4W)


def build_spreads(df: pd.DataFrame, horizon: int = HORIZON_4W) -> pd.DataFrame:
    """Construct cross-asset spreads and relative-value features.

    Expects the cleaned raw-level dataset (e.g. from ``load_dataset``). For
    every spread in ``SPREAD_KINDS`` both the **level** and its **4-week
    change** (``_chg4w``) are returned. Standalone features in
    ``STANDALONE_KINDS`` are returned as a single series each. See the module
    note above for the proxy choices on missing series.
    """
    out: dict[str, pd.Series] = {}

    # --- Term-structure spreads (yield points) ----------------------------
    out["us_term_10y_3m"] = df["GT10"] - df["USGG3M"]
    out["us_term_10y_2y"] = df["GT10"] - df["USGG2YR"]
    out["de_term_10y_2y"] = df["GTDEM10Y"] - df["GTDEM2Y"]

    # --- Sovereign spreads (yield points) ---------------------------------
    out["it_de_10y"] = df["GTITL10YR"] - df["GTDEM10Y"]   # BTP - Bund
    out["us_de_10y"] = df["GT10"] - df["GTDEM10Y"]

    # --- Credit / EM proxies (log price-index ratios, safe - risky) -------
    log_hy = np.log(df["LF98TRUU"])     # US High Yield
    log_ig = np.log(df["LUACTRUU"])     # US IG Corporate
    log_mbs = np.log(df["LUMSTRUU"])    # US MBS (high grade / quasi-govt)
    log_em = np.log(df["EMUSTRUU"])     # EM USD Aggregate
    out["hy_spread"] = log_mbs - log_hy     # HY credit-risk premium proxy
    out["hy_ig_spread"] = log_ig - log_hy   # quality spread (IG vs HY)
    out["em_spread"] = log_ig - log_em      # EM vs IG

    # Add the 4-week change for every spread above.
    for name in list(out):
        out[f"{name}_chg4w"] = out[name].diff(horizon)

    # --- Standalone relative-value features -------------------------------
    # Equity-bond rotation: 4w return of MSCI World proxy minus Global Agg proxy
    out["equity_bond_rot"] = _logret_4w(df["MXUS"]) - _logret_4w(df["LUACTRUU"])
    # Gold/oil ratio (rises when oil collapses -> classic risk-off)
    out["gold_oil_ratio"] = df["XAUBGNL"] / df["Cl1"]
    # Variance risk premium: implied (VIX) minus annualized realized vol of MXUS
    realized_vol = (
        np.log(df["MXUS"]).diff().rolling(REALIZED_VOL_WEEKS).std()
        * np.sqrt(WEEKS_PER_YEAR) * 100
    )
    out["vrp"] = df["VIX"] - realized_vol
    # JPY strength: negative 4-week return of USDJPY (yen up = risk-off haven)
    out["jpy_strength"] = -_logret_4w(df["JPY"])

    return pd.DataFrame(out, index=df.index)


def correlation_heatmap(
    originals: pd.DataFrame,
    spreads: pd.DataFrame,
    path: Path = FIG_DIR / "feature_spread_correlation.png",
    redundancy_threshold: float = 0.9,
) -> pd.DataFrame:
    """Save a correlation heatmap of originals + spreads; return redundant pairs.

    Aligns both frames on the common index, drops the target, and computes the
    Pearson correlation. Returns the pairs with ``|corr| >=
    redundancy_threshold``. Note: spread *levels* (yield/sovereign spreads,
    gold/oil ratio, credit log-ratios) are persistent, so their correlations
    with stationary returns should be read qualitatively.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    orig = originals.drop(columns=[TARGET_COL], errors="ignore")
    combined = orig.join(spreads, how="inner").dropna(axis=0, how="any")
    corr = combined.corr()

    n = corr.shape[1]
    fig, ax = plt.subplots(figsize=(0.32 * n + 4, 0.32 * n + 3))
    sns.heatmap(
        corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True,
        linewidths=0.3, cbar_kws={"shrink": 0.6}, ax=ax,
    )
    ax.set_title("Correlation: original features + engineered spreads")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)

    # Extract highly correlated (redundant) pairs from the upper triangle.
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = (
        upper.stack()
        .rename("corr")
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
    )
    redundant = pairs[pairs["corr"].abs() >= redundancy_threshold]
    return redundant.reindex(
        redundant["corr"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


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

    # --- Engineered spreads ------------------------------------------------
    spreads = build_spreads(df).dropna(axis=0, how="any")
    spread_labels = {**SPREAD_KINDS, **STANDALONE_KINDS}
    spread_labels.update({f"{k}_chg4w": "spread_chg4w" for k in SPREAD_KINDS})
    spread_adf = adf_table(spreads, transform_map=spread_labels)
    if verbose:
        print("\n" + "=" * 78)
        print("AUGMENTED DICKEY-FULLER TEST — engineered spreads")
        print("=" * 78)
        with pd.option_context("display.max_rows", None, "display.width", 140):
            print(spread_adf.to_string())
        n_stat = int(spread_adf["stationary"].sum())
        print(f"\nStationary at 5%: {n_stat}/{len(spread_adf)} spreads")
        non_stat = spread_adf.index[~spread_adf["stationary"]].tolist()
        if non_stat:
            print(f"NOT stationary at 5% (persistent levels, expected): {non_stat}")

    spreads.to_parquet(SPREADS_PATH)
    if verbose:
        print(f"Saved {spreads.shape[0]} x {spreads.shape[1]} -> {SPREADS_PATH}")

    # --- Redundancy: correlation heatmap (originals + spreads) ------------
    redundant = correlation_heatmap(stationary, spreads)
    if verbose:
        print("\n" + "=" * 78)
        print("REDUNDANCY — pairs with |corr| >= 0.90 (originals + spreads)")
        print("=" * 78)
        if redundant.empty:
            print("None.")
        else:
            with pd.option_context("display.max_rows", None, "display.width", 120):
                print(redundant.round(3).to_string(index=False))
        print(f"\nSaved heatmap -> {FIG_DIR / 'feature_spread_correlation.png'}")
    return stationary


if __name__ == "__main__":
    build()
