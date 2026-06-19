"""Italy-10Y patch for the Phase-3 free-data OOS panel (FIX-1 / FIX-2).

In Phase 3 Italy 10Y had no reachable free source, so ``GTITL10YR`` and the two
BTP-Bund-spread features (``it_de_10y``, ``it_de_10y_chg4w``) were neutralized to
the development mean in the 2022-2025 OOS panel. But ``it_de_10y`` is the #1
importance cluster from Phase-2 and the BTP-Bund spread widened materially in the
2022 inflation bear — exactly the episode the headline is about — so neutralizing
it specifically handicaps that episode.

This module recovers Italy 10Y weekly (free) and rebuilds the three Italy-
dependent features through the UNMODIFIED ``src.features`` pipeline, saving a
PATCHED panel ALONGSIDE (never overwriting) the neutralized DX-3 panel. Nothing
else changes: every other raw series and feature is taken verbatim from the
committed free raw panel.

Source resolution order (FIX-1): stooq, investpy, investing.com, worldgovernment-
bonds — kept the first that yields a clean weekly series. In this environment
stooq returns a JavaScript anti-bot wall and investpy/Yahoo expose no Italy-10Y
symbol; the investing.com historical API (pair 23738, "Italy 10-Year") responds
cleanly and is used. The raw daily pull is cached to ``data/raw/free/``.

Run from the repo root::

    python -m src.it10y_patch          # source + validate + build patched panel
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import PROJECT_ROOT
from src.free_data import EXTENDED_DIR, _free_feature_frame, to_weekly, tuesday_grid

RAW_FREE_DIR = PROJECT_ROOT / "data" / "raw" / "free"
IT10Y_CACHE = RAW_FREE_DIR / "it10y_daily.csv"
MANUAL_CSV = RAW_FREE_DIR / "it10y_weekly.csv"   # operator fallback drop point
INVESTING_PID = 23738                            # "Italy 10-Year" bond yield

PATCHED_RAW = EXTENDED_DIR / "raw_panel_free_it10y.parquet"
PATCHED_MODEL = EXTENDED_DIR / "model_panel_ext_it10y.parquet"
ITALY_FEATURES = ["GTITL10YR", "it_de_10y", "it_de_10y_chg4w"]

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


# --------------------------------------------------------------------------
# FIX-1 — source Italy 10Y (free), in the spec's order
# --------------------------------------------------------------------------
def _from_stooq(start: str, end: str) -> pd.Series:
    import requests
    s = pd.Timestamp(start).strftime("%Y%m%d")
    e = pd.Timestamp(end).strftime("%Y%m%d")
    url = f"https://stooq.com/q/d/l/?s=10ity.b&d1={s}&d2={e}&i=d"
    txt = requests.get(url, headers={"User-Agent": _UA}, timeout=25).text
    if "<html" in txt[:200].lower() or "Date" not in txt[:20]:
        raise RuntimeError("stooq: anti-bot / no CSV")
    d = pd.read_csv(io.StringIO(txt))
    return pd.Series(d["Close"].to_numpy(), index=pd.to_datetime(d["Date"])).sort_index()


def _from_investpy(start: str, end: str) -> pd.Series:
    import investpy  # noqa: F401  (defunct on modern investing.com; tried per spec)
    df = investpy.bonds.get_bond_historical_data(
        bond="Italy 10Y",
        from_date=pd.Timestamp(start).strftime("%d/%m/%Y"),
        to_date=pd.Timestamp(end).strftime("%d/%m/%Y"))
    return df["Close"].sort_index()


def _from_investing(start: str, end: str) -> pd.Series:
    """investing.com historical API (pair 23738 = Italy 10-Year)."""
    import requests
    headers = {"User-Agent": _UA, "Accept": "application/json",
               "Origin": "https://www.investing.com",
               "Referer": "https://www.investing.com/", "domain-id": "www"}
    url = (f"https://api.investing.com/api/financialdata/historical/{INVESTING_PID}"
           f"?start-date={start}&end-date={end}&time-frame=Daily&add-missing-rows=false")
    data = json.loads(requests.get(url, headers=headers, timeout=40).text)["data"]
    df = pd.DataFrame(data)
    idx = pd.to_datetime(df["rowDateTimestamp"])
    return pd.Series(pd.to_numeric(df["last_closeRaw"], errors="coerce").to_numpy(),
                     index=idx).sort_index().dropna()


def _from_worldgovbonds(start: str, end: str) -> pd.Series:
    """worldgovernmentbonds embeds the history as a JS array of [ms, yield]."""
    import re
    import requests
    url = "https://www.worldgovernmentbonds.com/bond-historical-data/italy/10-years/"
    html = requests.get(url, headers={"User-Agent": _UA}, timeout=25).text
    m = re.search(r"data:\s*(\[\[.*?\]\])", html, re.DOTALL)
    if not m:
        raise RuntimeError("wgb: no embedded data array")
    arr = json.loads(m.group(1))
    s = pd.Series({pd.Timestamp(p[0], unit="ms").normalize(): float(p[1]) for p in arr})
    return s.sort_index()


def _norm_index(s: pd.Series) -> pd.Series:
    """Force a tz-naive, midnight-normalized DatetimeIndex (the panel grid is
    tz-naive; mixing tz-aware breaks the weekly union/ffill alignment)."""
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = s.copy()
    s.index = idx.normalize()
    return s.sort_index()


def source_italy10y(start: str = "2014-01-01", end: str = "2025-12-31",
                    use_cache: bool = True, verbose: bool = True) -> tuple[pd.Series, str]:
    """Return (daily Italy-10Y series, source-name). Tries the free providers in
    the spec's order, caches the winning raw pull to data/raw/free/."""
    RAW_FREE_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and IT10Y_CACHE.exists():
        d = pd.read_csv(IT10Y_CACHE, parse_dates=["date"]).set_index("date")["it10y"]
        return _norm_index(d), "cache"

    attempts = [("stooq", _from_stooq), ("investpy", _from_investpy),
                ("investing.com", _from_investing), ("worldgovernmentbonds", _from_worldgovbonds)]
    for name, fn in attempts:
        try:
            s = _norm_index(fn(start, end))
            if len(s) > 200 and s.notna().sum() > 200:
                s.rename("it10y").to_frame().rename_axis("date").to_csv(IT10Y_CACHE)
                if verbose:
                    print(f"  Italy 10Y sourced from {name}: {len(s)} daily obs "
                          f"{s.index.min().date()}..{s.index.max().date()}")
                return s.sort_index(), name
            if verbose:
                print(f"  {name}: too few rows ({len(s)})")
        except Exception as e:
            if verbose:
                print(f"  {name}: {type(e).__name__} {str(e)[:60]}")

    if MANUAL_CSV.exists():   # operator fallback (FIX-1)
        d = pd.read_csv(MANUAL_CSV)
        d.columns = [c.lower() for c in d.columns]
        dcol = [c for c in d.columns if "date" in c][0]
        vcol = [c for c in d.columns if c not in (dcol,)][0]
        s = pd.Series(pd.to_numeric(d[vcol], errors="coerce").to_numpy(),
                      index=pd.to_datetime(d[dcol])).sort_index().dropna()
        return s, "manual_csv"
    raise RuntimeError(
        "All free Italy-10Y sources failed (anti-bot / unavailable). "
        f"Drop a weekly CSV at {MANUAL_CSV} (columns: date,yield) and re-run.")


# --------------------------------------------------------------------------
# Validation (same harness convention: transformed-series Pearson on overlap)
# --------------------------------------------------------------------------
def validate_italy10y(it10y_weekly: pd.Series, overlap=("2015-01-06", "2021-04-20"),
                      verbose: bool = True) -> dict:
    """Push the new IT10Y through the SAME src.features transform as GTITL10YR
    (first difference) and compare to the original on the overlap; also check the
    it_de_10y (BTP-Bund) spread level. Reports corr + fidelity tier."""
    from src.data_loader import load_dataset
    orig = load_dataset(verbose=False)
    s, e = pd.Timestamp(overlap[0]), pd.Timestamp(overlap[1])
    grid = pd.date_range(s, e, freq="W-TUE")

    free_lvl = it10y_weekly.reindex(grid)
    orig_lvl = orig["GTITL10YR"].reindex(grid)
    # GTITL10YR transform = first difference (DIFF family)
    a, b = free_lvl.diff(), orig_lvl.diff()
    m = a.notna() & b.notna()
    corr_diff = float(np.corrcoef(a[m], b[m])[0, 1])
    corr_lvl = float(np.corrcoef(free_lvl[m], orig_lvl[m])[0, 1])
    # it_de_10y spread (BTP-Bund) = IT10Y - DE10Y; free uses the ECB German 10Y.
    free_de = pd.read_parquet(EXTENDED_DIR / "raw_panel_free.parquet")["GTDEM10Y"].reindex(grid)
    spread_free = free_lvl - free_de
    spread_orig = (orig["GTITL10YR"] - orig["GTDEM10Y"]).reindex(grid)
    ms = spread_free.notna() & spread_orig.notna()
    corr_spread = float(np.corrcoef(spread_free[ms], spread_orig[ms])[0, 1])

    tier = ("PASS" if corr_diff > 0.99 else "STRONG" if corr_diff > 0.95
            else "MODERATE" if corr_diff > 0.85 else "WEAK")
    out = {"n_overlap": int(m.sum()), "corr_level": round(corr_lvl, 4),
           "corr_diff_transformed": round(corr_diff, 4),
           "corr_it_de_10y_spread": round(corr_spread, 4), "tier": tier,
           "flag": "OK" if corr_diff >= 0.95 else "MODERATE_FIDELITY"}
    if verbose:
        print("\nFIX-1 fidelity (Italy 10Y, transformed-series corr on 2015-2021):")
        print(f"  level corr            : {corr_lvl:.4f}")
        print(f"  GTITL10YR diff corr   : {corr_diff:.4f}  -> tier {tier}")
        print(f"  it_de_10y spread corr : {corr_spread:.4f}")
        if corr_diff < 0.95:
            print("  [FLAG] transformed corr < 0.95 -> moderate fidelity, not silent")
    return out


# --------------------------------------------------------------------------
# FIX-2 — patched raw + model panels (standard pipeline, alongside originals)
# --------------------------------------------------------------------------
def build_patched_panels(verbose: bool = True) -> dict:
    """Inject Italy 10Y into the committed free raw panel, re-run the unmodified
    src.features pipeline, and persist patched raw + model panels alongside the
    neutralized DX-3 outputs. Returns a summary dict."""
    from src.extend_dataset import (ORIGINAL_END, OOS_END, OOS_START, frozen_features)
    from src.data_loader import load_dataset
    from src.y_rule import label_ex_ante

    it_daily, source = source_italy10y(verbose=verbose)
    grid = tuesday_grid("2014-01-01", "2025-12-31")
    it_weekly = to_weekly(it_daily, grid)
    val = validate_italy10y(it_weekly, verbose=verbose)

    raw = pd.read_parquet(EXTENDED_DIR / "raw_panel_free.parquet").copy()
    raw["GTITL10YR"] = it_weekly.reindex(raw.index)   # replace the all-NaN gap column
    raw.to_parquet(PATCHED_RAW)

    free = _free_feature_frame(raw)
    frozen = frozen_features()
    model = free[frozen].copy()

    orig = load_dataset(verbose=False)
    y = pd.Series(index=model.index, dtype="float64")
    overlap = model.index <= pd.Timestamp(ORIGINAL_END)
    y[overlap] = orig["Y"].reindex(model.index[overlap]).to_numpy()
    post = model.index > pd.Timestamp(ORIGINAL_END)
    y[post] = label_ex_ante(model.index[post], raw["VIX"]).to_numpy()
    model["Y"] = y
    model.to_parquet(PATCHED_MODEL)

    oos = model.loc[OOS_START:OOS_END]
    italy_na = {c: int(oos[c].isna().sum()) for c in ITALY_FEATURES}
    summary = {"source": source, "fidelity": val,
               "italy_features_na_in_oos": italy_na,
               "patched_raw": str(PATCHED_RAW.relative_to(PROJECT_ROOT)),
               "patched_model": str(PATCHED_MODEL.relative_to(PROJECT_ROOT)),
               "neutralized_remaining": ["BDIY", "ECSURPUS", "US0001M"]}
    with open(EXTENDED_DIR / "it10y_patch_report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if verbose:
        print("\nFIX-2 patched panels:")
        print(f"  Italy features NaN in 2022-2025 OOS: {italy_na} (expect all 0)")
        print(f"  it_de_10y OOS 2022 range: "
              f"{oos.loc['2022','it_de_10y'].min():.3f}..{oos.loc['2022','it_de_10y'].max():.3f}")
        print(f"  saved -> {PATCHED_RAW.name}, {PATCHED_MODEL.name}, it10y_patch_report.json")
    return summary


# --------------------------------------------------------------------------
# FIX-3 — re-run WP4 (frozen) + WP5 (refit) on the patched panel; before/after
# --------------------------------------------------------------------------
REDUCED_NEUTRALIZE = ["BDIY", "ECSURPUS", "US0001M"]   # Italy features now real
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def run_fix3(verbose: bool = True) -> dict:
    """Re-score frozen-2018 and refit-2021 on the neutralized (DX-3) AND the
    Italy-10Y-patched 2022-2025 panels, in one process (same fitted models, so
    the before/after delta isolates the feature gap, not AE drift)."""
    from src.oos_eval import (METRIC_KEYS, Frozen36Scorer, run_control, run_oos)

    frozen = Frozen36Scorer()
    control = run_control(frozen, verbose)
    ref_aucpr = control["committed"]["AUC_PR"]
    if abs(control["metrics"]["AUC_PR"] - ref_aucpr) >= 0.02:
        raise RuntimeError(
            f"CONTROL MOVED: AUC-PR {control['metrics']['AUC_PR']:.4f} vs "
            f"{ref_aucpr:.4f} — something leaked into the 2019-2021 control; stop.")

    fb = run_oos(frozen, "frozen_neutralized", verbose=False)
    fa = run_oos(frozen, "frozen_it10y", verbose=False,
                 panel_path=PATCHED_MODEL, neutralize=REDUCED_NEUTRALIZE)
    refit = Frozen36Scorer(refit_full_2021=True, retune_tau=True)
    rb = run_oos(refit, "refit_neutralized", verbose=False)
    ra = run_oos(refit, "refit_it10y", verbose=False,
                 panel_path=PATCHED_MODEL, neutralize=REDUCED_NEUTRALIZE)

    runs = [("frozen_2018", "neutralized", fb), ("frozen_2018", "it10y_patched", fa),
            ("refit_2021", "neutralized", rb), ("refit_2021", "it10y_patched", ra)]
    table = pd.DataFrame([{"model": m, "panel": p, "tau": round(
        (frozen.tau if m == "frozen_2018" else refit.tau), 4),
        **{k: round(r["metrics"][k], 4) for k in METRIC_KEYS}} for m, p, r in runs])

    epi_rows = []
    for m, p, r in runs:
        for _, e in r["episodes"].iterrows():
            epi_rows.append({"model": m, "panel": p, "episode": e["episode"],
                             "coverage": e["coverage"], "n_flagged": int(e["n_flagged"]),
                             "lead_lag_weeks": e["lead_lag_weeks_vs_start"],
                             "dominant_model": e["dominant_model"]})
    epi = pd.DataFrame(epi_rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "oos_it10y_before_after.csv", index=False)
    epi.to_csv(TABLES_DIR / "oos_it10y_episode_before_after.csv", index=False)
    _plot_before_after(table)

    if verbose:
        print("\n" + "=" * 86)
        print("FIX-3 — BEFORE (neutralized) vs AFTER (Italy-10Y patched), same fitted models")
        print("=" * 86)
        print(table.to_string(index=False))
        print("\n2022 inflation-bear episode (coverage / flags / lag):")
        bear = epi[epi.episode == "2022 inflation bear"]
        print(bear.to_string(index=False))
        fd = table[table.model == "frozen_2018"].set_index("panel")
        rd = table[table.model == "refit_2021"].set_index("panel")
        print("\nINTERPRETATION (<=8 lines):")
        print(f"- Frozen AUC-PR {fd.loc['neutralized','AUC_PR']:.3f} -> "
              f"{fd.loc['it10y_patched','AUC_PR']:.3f} "
              f"({fd.loc['it10y_patched','AUC_PR']-fd.loc['neutralized','AUC_PR']:+.3f}) "
              f"once the BTP-Bund spread is restored.")
        print(f"- Refit  AUC-PR {rd.loc['neutralized','AUC_PR']:.3f} -> "
              f"{rd.loc['it10y_patched','AUC_PR']:.3f} "
              f"({rd.loc['it10y_patched','AUC_PR']-rd.loc['neutralized','AUC_PR']:+.3f}).")
        gap_share = fd.loc['it10y_patched','AUC_PR'] - fd.loc['neutralized','AUC_PR']
        total_gap = 0.784 - fd.loc['neutralized','AUC_PR']
        print(f"- Of the 0.78->{fd.loc['neutralized','AUC_PR']:.2f} frozen OOS drop, the Italy gap "
              f"explains ~{gap_share:+.3f} AUC-PR; the remainder is genuine regime-transfer failure.")
        print(f"- frozen->refit AUC-PR delta: neutralized "
              f"{rd.loc['neutralized','AUC_PR']-fd.loc['neutralized','AUC_PR']:+.3f} vs patched "
              f"{rd.loc['it10y_patched','AUC_PR']-fd.loc['it10y_patched','AUC_PR']:+.3f} "
              f"(does closing the gap change the refit verdict?).")
    return {"table": table, "episodes": epi, "control": control}


def _plot_before_after(table: pd.DataFrame, out_path: Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = ["AUC_PR", "F_beta_2", "Recall", "Precision"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, model in zip(axes, ["frozen_2018", "refit_2021"]):
        sub = table[table.model == model].set_index("panel")
        x = np.arange(len(metrics)); w = 0.38
        ax.bar(x - w / 2, [sub.loc["neutralized", k] for k in metrics], w,
               label="neutralized (before)", color="#bdbdbd")
        ax.bar(x + w / 2, [sub.loc["it10y_patched", k] for k in metrics], w,
               label="Italy-10Y patched (after)", color="#3182bd")
        ax.set_xticks(x); ax.set_xticklabels(["AUC-PR", "F2", "Recall", "Prec"])
        ax.set_title(model); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.suptitle("Frozen-36 OOS 2022-2025: before vs after Italy-10Y patch")
    fig.tight_layout()
    out_path = out_path or FIGURES_DIR / "oos_it10y_before_after.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    build_patched_panels()
    run_fix3()
