"""Weekly portfolio backtester + full performance-metric suite.

Weights are decided at week t (from info available at t) and applied to week
t+1 returns (one-week lag, no look-ahead). Transaction costs are charged on the
change in effective weights (one-way bps per asset touched).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ASSET_COLS = ["equity", "bond", "gold", "cash", "mbs"]
WEEKS_PER_YEAR = 52
COVID_CRASH = ("2020-02-15", "2020-04-15")


def backtest_strategy(weights_ts: pd.DataFrame, returns_ts: pd.DataFrame, tc_dict: dict) -> dict:
    """Backtest a weekly weight schedule.

    ``weights_ts`` and ``returns_ts`` are weekly DataFrames over ASSET_COLS.
    Returns dict with equity_curve, net/gross weekly returns and the tc series.
    """
    w = weights_ts.reindex(columns=ASSET_COLS).fillna(0.0)
    r = returns_ts.reindex(columns=ASSET_COLS).reindex(w.index).fillna(0.0)

    eff_w = w.shift(1)            # decided at t-1, applied to week-t return (no look-ahead)
    eff_w.iloc[0] = 0.0
    gross = (eff_w * r).sum(axis=1)

    tc_frac = pd.Series({k: tc_dict[k] / 10000.0 for k in ASSET_COLS})
    dw = eff_w.diff().abs()
    dw.iloc[0] = eff_w.iloc[0].abs()
    tc = (dw * tc_frac).sum(axis=1)

    net = gross - tc
    equity_curve = (1.0 + net).cumprod()
    return {"equity_curve": equity_curve, "net": net, "gross": gross, "tc": tc}


def risk_parity_weights(returns_df: pd.DataFrame, window: int = 52) -> pd.DataFrame:
    """Inverse-rolling-vol weights (normalized to 1) over the given columns."""
    vols = returns_df.rolling(window).std()
    inv = 1.0 / vols
    return inv.div(inv.sum(axis=1), axis=0)


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def _max_dd_duration(equity: pd.Series) -> int:
    underwater = (equity < equity.cummax()).to_numpy()
    longest = run = 0
    for u in underwater:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return int(longest)


def _covid_recovery_weeks(equity: pd.Series) -> float:
    cs, _ = COVID_CRASH
    pre = equity.loc[:cs]
    post = equity.loc[cs:]
    if pre.empty or post.empty:
        return float("nan")
    pre_peak = pre.max()
    trough_pos = int(np.argmin(post.to_numpy()))
    after = post.iloc[trough_pos:]
    recovered = np.where(after.to_numpy() >= pre_peak)[0]
    return float(recovered[0]) if len(recovered) else float("nan")


def compute_full_metrics(equity_curve, weekly_returns, allocations_series=None,
                         rf_series=None, benchmark_returns=None,
                         tc_series=None, y_series=None) -> dict:
    """Full metric suite (return / risk / risk-adjusted / crisis / operational)."""
    r = weekly_returns.dropna()
    ec = equity_curve.reindex(r.index)
    n = len(r)
    ann = np.sqrt(WEEKS_PER_YEAR)
    rf = (rf_series.reindex(r.index).fillna(0.0) if rf_series is not None
          else pd.Series(0.0, index=r.index))
    excess = r - rf
    downside = np.sqrt(np.mean(np.clip(r.to_numpy(), None, 0.0) ** 2))
    dd = ec / ec.cummax() - 1.0
    cagr = float(ec.iloc[-1] ** (WEEKS_PER_YEAR / n) - 1.0)
    maxdd = _max_drawdown(ec)
    comp4 = (1.0 + r).rolling(4).apply(np.prod, raw=True) - 1.0
    q95, q05 = r.quantile(0.95), r.quantile(0.05)
    top, bot = r[r >= q95].mean(), r[r <= q05].mean()

    cs, ce = COVID_CRASH
    cov = ec.loc[cs:ce]
    covid_dd = float((cov / cov.iloc[0]).min() - 1.0) if len(cov) else float("nan")

    m = {
        "CAGR": cagr,
        "Total_Return": float(ec.iloc[-1] - 1.0),
        "Best_Week": float(r.max()),
        "Worst_Week": float(r.min()),
        "Vol_Annualized": float(r.std() * ann),
        "Downside_Deviation": float(downside * ann),
        "Max_Drawdown": maxdd,
        "Max_DD_Duration_Weeks": _max_dd_duration(ec),
        "Time_Underwater_Pct": float((dd < 0).mean() * 100),
        "Skewness": float(stats.skew(r)),
        "Kurtosis": float(stats.kurtosis(r)),
        "Sharpe": float(excess.mean() * ann / r.std()) if r.std() > 0 else float("nan"),
        "Sortino": float(excess.mean() * WEEKS_PER_YEAR / (downside * ann)) if downside > 0 else float("nan"),
        "Calmar": float(cagr / abs(maxdd)) if maxdd < 0 else float("nan"),
        "Information_Ratio_vs_6040": float("nan"),
        "COVID_Crash_DD": covid_dd,
        "COVID_Recovery_Weeks": _covid_recovery_weeks(ec),
        "Worst_4w_Period": float(comp4.min()),
        "Tail_Ratio": float(abs(top) / abs(bot)) if bot != 0 else float("nan"),
        "Annual_Turnover": None,
        "Cumulative_TC": None,
        "Hit_Ratio_RiskOff": None,
    }

    if benchmark_returns is not None:
        active = (r - benchmark_returns.reindex(r.index)).dropna()
        if active.std() > 0:
            m["Information_Ratio_vs_6040"] = float(active.mean() * ann / active.std())

    if allocations_series is not None:
        a = allocations_series.reindex(r.index)
        switches = int((a != a.shift()).sum() - 1)
        m["Annual_Turnover"] = float(switches / (n / WEEKS_PER_YEAR))
        if tc_series is not None:
            m["Cumulative_TC"] = float(tc_series.reindex(r.index).sum())
        if y_series is not None:
            y = y_series.reindex(r.index)
            riskoff = y == 1
            if riskoff.sum() > 0:
                in_haven = a[riskoff] != "LEVERED_EQUITY"
                m["Hit_Ratio_RiskOff"] = float(in_haven.mean())
    return m
