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


# ---------------------------------------------------------------------------
# Routing engine (Prompt 9) + threshold optimization (Prompt 10)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {"usd": 1.0, "oro": 1.0}

# Allocation label -> asset weights (over equity/bond/gold/cash/mbs).
ALLOCATION_WEIGHTS = {
    "LEVERED_EQUITY": {"equity": 1.5, "cash": -0.5},
    "CASH_USD": {"cash": 1.0},
    "GOLD": {"gold": 1.0},
    "MBS": {"mbs": 1.0},
}


def route_allocation(ens_sig, sub_usd, sub_oro, sub_mbs, dxy_chg4w, th) -> str:
    """Map the ensemble risk signal + domain sub-scores to an allocation regime."""
    if ens_sig == 0:
        return "LEVERED_EQUITY"
    if sub_usd > th["usd"] and dxy_chg4w > 0:
        return "CASH_USD"
    if sub_oro > th["oro"]:
        return "GOLD"
    if sub_mbs == 1:
        return "MBS"
    return "CASH_USD"


class RoutingEngine:
    """Turns per-week signals into allocation labels and portfolio weights."""

    SIGNAL_COLS = ["ens_sig", "subscore_usd", "subscore_oro", "subscore_mbs", "dxy_chg4w"]

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)

    def route_series(self, df: pd.DataFrame, thresholds: dict | None = None) -> pd.Series:
        th = thresholds or self.thresholds
        ens = df["ens_sig"].to_numpy()
        u = df["subscore_usd"].to_numpy()
        o = df["subscore_oro"].to_numpy()
        mb = df["subscore_mbs"].to_numpy()
        dxy = df["dxy_chg4w"].to_numpy()
        out = [route_allocation(ens[i], u[i], o[i], mb[i], dxy[i], th) for i in range(len(df))]
        return pd.Series(out, index=df.index, name="allocation")

    @staticmethod
    def allocations_to_weights(alloc_series: pd.Series) -> pd.DataFrame:
        cols = ["equity", "bond", "gold", "cash", "mbs"]
        w = pd.DataFrame(0.0, index=alloc_series.index, columns=cols)
        for label, wd in ALLOCATION_WEIGHTS.items():
            mask = (alloc_series == label).to_numpy()
            for asset, val in wd.items():
                w.loc[mask, asset] = val
        return w


def optimize_routing_thresholds(folds_data, ensemble_per_fold, subscores_per_fold,
                                prices, tc_dict,
                                usd_grid=(0.5, 0.75, 1.0, 1.25, 1.5),
                                oro_grid=(0.5, 0.75, 1.0, 1.25, 1.5)):
    """Grid-search (usd, oro) thresholds; score each combo by the **duration-
    weighted median** Calmar across the fold validation windows.

    ``folds_data`` maps fold_id -> routing fold-val DataFrame (needs dxy_chg4w);
    ``ensemble_per_fold`` maps fold_id -> binary ensemble signal series;
    ``subscores_per_fold`` maps fold_id -> DataFrame(subscore_usd/oro/mbs);
    ``prices`` is the weekly per-asset returns DataFrame. Returns
    (best_thresholds, grid_results_df)."""
    from src.backtest import backtest_strategy
    from src.models import weighted_median

    engine = RoutingEngine()
    rows = []
    for u in usd_grid:
        for o in oro_grid:
            th = {"usd": u, "oro": o}
            calmars, weeks = [], []
            for fid, fold_df in folds_data.items():
                idx = fold_df.index
                sig = pd.DataFrame({
                    "ens_sig": pd.Series(ensemble_per_fold[fid], index=idx),
                    "subscore_usd": subscores_per_fold[fid]["subscore_usd"].reindex(idx),
                    "subscore_oro": subscores_per_fold[fid]["subscore_oro"].reindex(idx),
                    "subscore_mbs": subscores_per_fold[fid]["subscore_mbs"].reindex(idx),
                    "dxy_chg4w": fold_df["dxy_chg4w"].reindex(idx),
                }, index=idx)
                alloc = engine.route_series(sig, th)
                w = RoutingEngine.allocations_to_weights(alloc)
                res = backtest_strategy(w, prices.reindex(idx), tc_dict)
                ec, net = res["equity_curve"], res["net"]
                m = len(net)
                cagr = ec.iloc[-1] ** (52 / m) - 1.0
                maxdd = (ec / ec.cummax() - 1.0).min()
                calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
                calmars.append(calmar)
                weeks.append(m)
            cal = np.array(calmars, dtype=float)
            wk = np.array(weeks, dtype=float)
            ok = ~np.isnan(cal)
            med = weighted_median(cal[ok], wk[ok]) if ok.any() else np.nan
            rows.append({"usd": u, "oro": o, "calmar_wmedian": med})

    grid = pd.DataFrame(rows)
    best = grid.loc[grid["calmar_wmedian"].idxmax()]
    best_thresholds = {"usd": float(best["usd"]), "oro": float(best["oro"])}
    return best_thresholds, grid
