"""Domain sub-scores (USD, Gold, MBS) built from the 14 routing triggers.

USD and Gold sub-scores are the mean of *signed z-scores* of their triggers,
calibrated (mu, sigma) on the development set. The MBS sub-score is a binary
rule-based filter (active in moderate stress, blocked in acute crises). These
feed the Prompt-9 routing engine to pick which safe haven to rotate into.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Signed z-score weights per domain (sign = direction that means "risk-off for
# this domain"). Comments give the economic reading.
SUBSCORE_USD = {
    "libor_3m_spread_chg4w": +1,   # USD funding stress -> USD strong
    "dxy_chg4w": +1,               # USD appreciates -> USD trigger
    "vrp": +1,                     # forward-looking US equity stress
    "us_10y_diff_chg4w": -1,       # yields down = flight-to-quality -> USD up
    "usa_world_relative": +1,      # US outperforms = non-US stress -> USD haven
}
SUBSCORE_ORO = {
    "real_yield_proxy_chg4w": -1,  # real rates down -> gold up (no carry cost)
    "dxy_chg4w": -1,               # weak USD -> strong gold
    "jpy_strength": +1,            # strong yen confirms flight from USD fiat
    "equity_bond_corr_13w": +1,    # positive corr = diversification fails -> gold
    "gold_oil_ratio_chg4w": +1,    # gold/oil up = real macro stress
}

# Columns the MBS rule reads (hy_ig_spread_chg4w must be joined from spreads).
MBS_REQUIRED = [
    "vix_level", "us_term_10y_2y_level", "mxus_drawdown_52w",
    "libor_3m_spread_level", "hy_ig_spread_chg4w",
]


def fit_zscore_params(df_dev: pd.DataFrame, sign_dict: dict) -> tuple[dict, dict]:
    """Estimate mu and sigma of each trigger in ``sign_dict`` on the development set."""
    mu = {c: float(df_dev[c].mean()) for c in sign_dict}
    sigma = {c: float(df_dev[c].std()) for c in sign_dict}
    return mu, sigma


def compute_subscore_zscore(triggers_target, mu_dict, sigma_dict, sign_dict) -> np.ndarray:
    """Mean of signed z-scores across the domain's triggers, per row."""
    cols = list(sign_dict)
    z = np.empty((len(triggers_target), len(cols)))
    for j, c in enumerate(cols):
        sigma = sigma_dict[c] if sigma_dict[c] not in (0, None) else 1.0
        z[:, j] = sign_dict[c] * (triggers_target[c].to_numpy() - mu_dict[c]) / sigma
    return z.mean(axis=1)


def fit_mbs_params(df_dev: pd.DataFrame) -> dict:
    """90th-percentile of libor_3m_spread_level on the development set."""
    return {"p90_dev": float(df_dev["libor_3m_spread_level"].quantile(0.90))}


def compute_subscore_mbs(triggers_target, params_dict) -> np.ndarray:
    """Binary MBS sub-score: 1 when moderate stress (active) and not in an acute
    crisis (blocked), else 0. Requires the columns in ``MBS_REQUIRED``."""
    p90 = params_dict["p90_dev"]
    vix = triggers_target["vix_level"].to_numpy()
    term = triggers_target["us_term_10y_2y_level"].to_numpy()
    dd = triggers_target["mxus_drawdown_52w"].to_numpy()
    libor = triggers_target["libor_3m_spread_level"].to_numpy()
    hyig = triggers_target["hy_ig_spread_chg4w"].to_numpy()

    active = (vix >= 20) & (vix <= 28) & (term > 0) & (dd >= -0.12) & (dd <= -0.05)
    blocked = (vix > 30) | (libor > p90) | (hyig > 0.005)
    return (active & ~blocked).astype(int)


def compute_all_subscores(triggers_target, mu_usd, sigma_usd, mu_oro, sigma_oro,
                          mbs_params) -> pd.DataFrame:
    """Convenience: USD/Gold z-score sub-scores + binary MBS, indexed like input."""
    return pd.DataFrame(
        {
            "subscore_usd": compute_subscore_zscore(triggers_target, mu_usd, sigma_usd, SUBSCORE_USD),
            "subscore_oro": compute_subscore_zscore(triggers_target, mu_oro, sigma_oro, SUBSCORE_ORO),
            "subscore_mbs": compute_subscore_mbs(triggers_target, mbs_params),
        },
        index=triggers_target.index,
    )
