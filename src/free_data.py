"""Free-data sourcing for the 2022-2025 out-of-sample extension (Phase 3, WP2).

The original panel is a proprietary Bloomberg weekly export (Tuesdays). To extend
the EWS out-of-sample on FREE data we re-source every RAW series that feeds the
frozen-36 model features (``selected_features.json``) and the 14 routing triggers,
map each Bloomberg ticker to a free provider, then push the result through the
unchanged ``src.features`` transforms.

Providers actually reachable from this environment (probed 2026-06-16):
  * Yahoo Finance (yfinance)        — equities, FX, commodities, VIX, DXY, US-yield
                                      indices, bond total-return ETFs
  * US Treasury par-yield XML       — US 2Y / 3M / 10Y / 30Y (key-free)
  * NY Fed markets API              — SOFR (USD funding, for the LIBOR splice)
  * ECB Data Portal (data-api)      — euro-area AAA yield curve (German proxy), EONIA/€STR
  * Bank of England IADB            — UK gilt yields
  * MoF Japan JGB CSV               — JGB 2Y / 30Y
The key-free FRED ``fredgraph.csv`` host is BLOCKED here; the keyed ``api.stlouisfed``
host is reachable but a key was deliberately avoided (operator chose key-free first,
and Treasury+NY Fed+ECB cover the US/EU rates FRED would have provided).

Coverage gaps that NO probed free source covers (documented, never silently
substituted with an unrelated live series — see ``GAP_TICKERS``):
  * BDIY    — Baltic Dry Index (no free historical API)
  * GTITL10YR — Italy 10Y (ECB publishes only AAA + GDP-weighted-average curves)
  * ECSURPUS — US economic surprise index (proprietary; dropped per the WP2 brief)

Splices (documented): US0001M (1M USD LIBOR, discontinued) -> SOFR; EONIA -> €STR
from 2022-01 (EONIA ceased 2022-01-03).

Run from the repo root (network required; uses outputs/cache for pulls)::

    python -m src.free_data            # build + cache the extended raw panel
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "extended" / "raw_cache"
EXTENDED_DIR = PROJECT_ROOT / "data" / "processed" / "extended"
PANEL_WEEKDAY = "Tue"   # the original panel is sampled on Tuesdays

# Raw Bloomberg tickers we cannot source from any probed free provider.
GAP_TICKERS = {
    "BDIY": "Baltic Dry Index — no free historical API",
    "GTITL10YR": "Italy 10Y — ECB exposes only AAA / GDP-weighted curves, not IT",
    "ECSURPUS": "US economic surprise index — proprietary (dropped per WP2 brief)",
}

# Documented splice points (date from which the second leg replaces the first).
SPLICES = {
    "US0001M": "1M USD LIBOR proxied by SOFR (LIBOR ceased 2023-06; SOFR used throughout)",
    "EONIA": "EONIA -> €STR from 2022-01-03 (EONIA discontinued)",
}

_HEADERS = {"User-Agent": "Mozilla/5.0 (research; free-data sourcing)"}


# --------------------------------------------------------------------------
# Weekly grid
# --------------------------------------------------------------------------
def tuesday_grid(start: str, end: str) -> pd.DatetimeIndex:
    """Weekly Tuesday index spanning [start, end] (matches the original panel)."""
    return pd.date_range(start=start, end=end, freq="W-TUE")


def to_weekly(daily: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """As-of map a daily series onto the Tuesday grid (last value on/before each
    Tuesday; ffill across holidays). Returns a series indexed by ``grid``."""
    s = daily.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(grid)).ffill().reindex(grid)


# --------------------------------------------------------------------------
# Source fetchers (each returns a daily pd.Series indexed by date)
# --------------------------------------------------------------------------
def _req(url: str, timeout: int = 30):
    import requests
    return requests.get(url, headers=_HEADERS, timeout=timeout)


def fetch_yahoo(symbol: str, start: str, end: str) -> pd.Series:
    import yfinance as yf
    d = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if d is None or len(d) == 0:
        raise RuntimeError(f"yahoo: no data for {symbol}")
    close = d["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = close.dropna()
    s.index = pd.to_datetime(s.index)
    return s


def fetch_treasury(maturity: str, start: str, end: str) -> pd.Series:
    """US par yield (maturity in {'1 Mo','3 Mo','2 Yr','10 Yr','30 Yr'}) from the
    Treasury daily XML feed (one calendar year per request)."""
    import xml.etree.ElementTree as ET
    ns = {"a": "http://www.w3.org/2005/Atom",
          "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
          "d": "http://schemas.microsoft.com/ado/2007/08/dataservices"}
    field = {"1 Mo": "BC_1MONTH", "3 Mo": "BC_3MONTH", "2 Yr": "BC_2YEAR",
             "10 Yr": "BC_10YEAR", "30 Yr": "BC_30YEAR"}[maturity]
    out = {}
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        url = ("https://home.treasury.gov/resource-center/data-chart-center/"
               "interest-rates/pages/xml?data=daily_treasury_yield_curve"
               f"&field_tdr_date_value={year}")
        r = _req(url, timeout=40)
        root = ET.fromstring(r.content)
        for entry in root.findall("a:entry", ns):
            props = entry.find("a:content/m:properties", ns)
            if props is None:
                continue
            date = props.find("d:NEW_DATE", ns)
            val = props.find(f"d:{field}", ns)
            if date is not None and val is not None and val.text:
                out[pd.Timestamp(date.text[:10])] = float(val.text)
    return pd.Series(out).sort_index()


def fetch_sofr(start: str, end: str) -> pd.Series:
    """SOFR daily from the NY Fed markets API (USD funding rate)."""
    url = (f"https://markets.newyorkfed.org/api/rates/secured/sofr/search.json"
           f"?startDate={start}&endDate={end}&type=rate")
    r = _req(url, timeout=40)
    data = r.json().get("refRates", [])
    s = pd.Series({pd.Timestamp(x["effectiveDate"]): float(x["percentRate"]) for x in data})
    return s.sort_index()


def fetch_ecb(flow: str, key: str, start: str, end: str) -> pd.Series:
    url = (f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}"
           f"?startPeriod={start}&endPeriod={end}&format=csvdata")
    r = _req(url, timeout=40)
    d = pd.read_csv(io.StringIO(r.text))
    s = pd.Series(d["OBS_VALUE"].to_numpy(),
                  index=pd.to_datetime(d["TIME_PERIOD"])).sort_index()
    return s.dropna()


def fetch_boe(series_code: str, start: str, end: str) -> pd.Series:
    """BoE IADB daily series (e.g. UK gilt yields)."""
    d1 = pd.Timestamp(start).strftime("%d/%b/%Y")
    d2 = pd.Timestamp(end).strftime("%d/%b/%Y")
    url = ("https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?"
           f"csv.x=yes&Datefrom={d1}&Dateto={d2}&SeriesCodes={series_code}"
           "&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
    r = _req(url, timeout=40)
    d = pd.read_csv(io.StringIO(r.text))
    d.columns = ["date", "value"]
    s = pd.Series(d["value"].to_numpy(), index=pd.to_datetime(d["date"], format="%d %b %Y"))
    return s.sort_index().dropna()


def fetch_mof_japan(maturity: str, start: str, end: str) -> pd.Series:
    """JGB par yields from MoF Japan. ``maturity`` in {'2Y','30Y',...}. The
    'jgbcme_all.csv' file holds the full history (fiscal-year rows)."""
    for url in ("https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/"
                "historical/jgbcme_all.csv",
                "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv"):
        try:
            r = _req(url, timeout=40)
            if r.status_code != 200 or "<html" in r.text[:200].lower():
                continue
            d = pd.read_csv(io.StringIO(r.text), skiprows=1)
            d = d.rename(columns={d.columns[0]: "date"})
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d = d.dropna(subset=["date"])
            if maturity not in d.columns:
                continue
            s = pd.to_numeric(d.set_index("date")[maturity], errors="coerce").dropna()
            s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
            if len(s):
                return s.sort_index()
        except Exception:
            continue
    raise RuntimeError(f"mof_japan: could not fetch {maturity}")


# --------------------------------------------------------------------------
# Source map: original Bloomberg ticker -> (provider, args, model transform note)
# --------------------------------------------------------------------------
# provider in {yahoo, treasury, sofr, ecb, boe, mof, splice}; the harness checks
# the TRANSFORMED series (log-return / diff / level) against the original.
SOURCE_MAP: dict[str, dict] = {
    # equities (MSCI USD price indices -> closest USD index / ETF; returns basis)
    "MXUS": {"provider": "yahoo", "symbol": "^GSPC"},
    "MXEU": {"provider": "yahoo", "symbol": "VGK"},
    "MXJP": {"provider": "yahoo", "symbol": "EWJ"},
    "MXBR": {"provider": "yahoo", "symbol": "EWZ"},
    "MXCN": {"provider": "yahoo", "symbol": "MCHI"},
    # FX (panel JPY = USDJPY level, GBP = GBPUSD level)
    "JPY": {"provider": "yahoo", "symbol": "JPY=X"},
    "GBP": {"provider": "yahoo", "symbol": "GBPUSD=X"},
    "DXY": {"provider": "yahoo", "symbol": "DX-Y.NYB"},
    # commodities / vol
    "XAUBGNL": {"provider": "yahoo", "symbol": "GC=F"},
    "Cl1": {"provider": "yahoo", "symbol": "CL=F"},
    "VIX": {"provider": "yahoo", "symbol": "^VIX"},
    # US yields (Treasury XML)
    "GT10": {"provider": "treasury", "maturity": "10 Yr"},
    "USGG2YR": {"provider": "treasury", "maturity": "2 Yr"},
    "USGG3M": {"provider": "treasury", "maturity": "3 Mo"},
    # USD 1M funding: LIBOR discontinued -> SOFR (documented splice)
    "US0001M": {"provider": "sofr"},
    # euro overnight: EONIA -> €STR (documented splice)
    "EONIA": {"provider": "splice_eonia"},
    # euro govt yields (ECB AAA curve = German-equivalent)
    "GTDEM10Y": {"provider": "ecb", "flow": "YC", "key": "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y"},
    "GTDEM2Y": {"provider": "ecb", "flow": "YC", "key": "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y"},
    # UK 2Y gilt (BoE IADB; IUDSNPY = British Govt Securities nominal par yield)
    "GTGBP2Y": {"provider": "boe", "series": "IUDSNPY"},
    # Japan JGB (MoF)
    "GTJPY2YR": {"provider": "mof", "maturity": "2Y"},
    "GTJPY30YR": {"provider": "mof", "maturity": "30Y"},
    # bond total-return indices -> total-return ETF proxies
    "LUACTRUU": {"provider": "yahoo", "symbol": "LQD"},   # US IG corp
    "LF98TRUU": {"provider": "yahoo", "symbol": "HYG"},   # US high yield
    "EMUSTRUU": {"provider": "yahoo", "symbol": "EMB"},   # EM USD aggregate
    "LMBITR": {"provider": "yahoo", "symbol": "MBB"},     # US MBS
    "LF94TRUU": {"provider": "yahoo", "symbol": "TIP"},   # inflation-linked (US TIPS proxy)
    "LP01TREU": {"provider": "yahoo", "symbol": "IEAG.L"},  # Euro aggregate bond
    # LUMSTRUU feeds ONLY the collinear-dropped hy_spread (never a frozen-36
    # feature); sourced so the unmodified src.features pipeline runs end-to-end.
    "LUMSTRUU": {"provider": "yahoo", "symbol": "VMBS"},
}


def fetch_ticker(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch one original ticker from its mapped free source (daily series)."""
    spec = SOURCE_MAP[ticker]
    p = spec["provider"]
    if p == "yahoo":
        return fetch_yahoo(spec["symbol"], start, end)
    if p == "treasury":
        return fetch_treasury(spec["maturity"], start, end)
    if p == "sofr":
        return fetch_sofr(start, end)
    if p == "ecb":
        return fetch_ecb(spec["flow"], spec["key"], start, end)
    if p == "boe":
        return fetch_boe(spec["series"], start, end)
    if p == "mof":
        return fetch_mof_japan(spec["maturity"], start, end)
    if p == "splice_eonia":
        # EONIA (ECB EON dataflow) until 2022-01-03, then €STR (ECB EST).
        eonia = fetch_ecb("EON", "D.EONIA_TO.RATE", start, "2022-01-03")
        try:
            estr = fetch_ecb("EST", "B.EU000A2X2A25.WT", "2021-10-01", end)
        except Exception:
            estr = pd.Series(dtype=float)
        return pd.concat([eonia[eonia.index < "2022-01-03"],
                          estr[estr.index >= "2022-01-03"]]).sort_index()
    raise ValueError(f"unknown provider {p}")


# --------------------------------------------------------------------------
# Panel assembly (with on-disk cache)
# --------------------------------------------------------------------------
def build_raw_panel(start: str, end: str, use_cache: bool = True,
                    verbose: bool = True) -> pd.DataFrame:
    """Assemble a raw-level weekly (Tuesday) panel with ORIGINAL ticker column
    names, sourced entirely from free providers. Cached per ticker under
    ``CACHE_DIR``. Gap tickers (``GAP_TICKERS``) are returned as all-NaN columns
    so the column set stays aligned; they are handled downstream as documented
    coverage gaps, never silently substituted."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    grid = tuesday_grid(start, end)
    cols = {}
    for ticker in list(SOURCE_MAP) + list(GAP_TICKERS):
        if ticker in GAP_TICKERS:
            cols[ticker] = pd.Series(np.nan, index=grid)
            continue
        cache = CACHE_DIR / f"{ticker}.parquet"
        if use_cache and cache.exists():
            daily = pd.read_parquet(cache)["value"]
            daily.index = pd.to_datetime(daily.index)
        else:
            for attempt in range(3):
                try:
                    daily = fetch_ticker(ticker, start, end)
                    break
                except Exception as e:
                    if attempt == 2:
                        if verbose:
                            print(f"  [WARN] {ticker} fetch failed ({e}); -> gap column")
                        daily = pd.Series(dtype=float)
                    else:
                        time.sleep(1.5)
            if len(daily):
                daily.rename("value").to_frame().to_parquet(cache)
        cols[ticker] = to_weekly(daily, grid) if len(daily) else pd.Series(np.nan, index=grid)
        if verbose:
            n = int(cols[ticker].notna().sum())
            print(f"  {ticker:10} {SOURCE_MAP.get(ticker, {}).get('provider', 'GAP'):12} "
                  f"{n:4d}/{len(grid)} weekly obs")
    panel = pd.DataFrame(cols, index=grid)
    panel.index.name = "Date"
    return panel


# --------------------------------------------------------------------------
# Feature reconstruction (unchanged src.features pipeline) + validation harness
# --------------------------------------------------------------------------
ROUTING_TRIGGERS_14 = [
    "libor_3m_spread_chg4w", "dxy_chg4w", "vrp", "us_10y_diff_chg4w",
    "usa_world_relative", "real_yield_proxy_chg4w", "jpy_strength",
    "equity_bond_corr_13w", "gold_oil_ratio_chg4w", "vix_level",
    "us_10y_vol_4w", "us_term_10y_2y_level", "libor_3m_spread_level",
    "mxus_drawdown_52w",
]


def build_free_features(raw_panel: pd.DataFrame):
    """Run the UNMODIFIED src.features transforms on a free raw panel.

    Returns (stationary_clean, spreads_clean, routing_triggers). ``dropna`` is
    disabled on the stationarity step so the all-NaN gap columns (BDIY, Italy
    10Y, ECSURPUS) don't wipe every row; gap-dependent features simply stay NaN.
    """
    from src.features import (COLLINEAR_DROP, build_routing_triggers, build_spreads,
                              make_stationary, remove_collinear_features)
    stat = make_stationary(raw_panel, dropna=False)
    spreads = build_spreads(raw_panel)
    triggers = build_routing_triggers(raw_panel, stat, spreads)
    stat_clean = remove_collinear_features(stat, COLLINEAR_DROP)
    spreads_clean = remove_collinear_features(spreads, COLLINEAR_DROP)
    return stat_clean, spreads_clean, triggers


def _free_feature_frame(raw_panel: pd.DataFrame) -> pd.DataFrame:
    """Single frame with every model feature + routing trigger (free data)."""
    stat, spreads, trig = build_free_features(raw_panel)
    return pd.concat([stat, spreads, trig.add_prefix("")], axis=1).loc[
        :, lambda d: ~d.columns.duplicated()]


def _original_feature_frame() -> pd.DataFrame:
    """The committed original transformed features (2000-2021) for comparison."""
    proc = PROJECT_ROOT / "data" / "processed"
    stat = pd.read_parquet(proc / "features_stationary_clean.parquet")
    spreads = pd.read_parquet(proc / "spreads_clean.parquet")
    trig = pd.read_parquet(proc / "routing_triggers.parquet")
    out = pd.concat([stat, spreads, trig], axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def validate_proxies(raw_panel: pd.DataFrame | None = None,
                     overlap=("2015-01-01", "2021-04-20"),
                     corr_threshold: float = 0.99, save: bool = True,
                     verbose: bool = True) -> pd.DataFrame:
    """Compare every frozen-36 feature + routing trigger reconstructed from free
    data against the original, on the TRANSFORMED series over the overlap window.
    PASS iff Pearson corr > ``corr_threshold``. FAIL / GAP features are reported,
    never silently substituted. Writes outputs/tables/proxy_validation.csv."""
    import json

    if raw_panel is None:
        raw_panel = pd.read_parquet(EXTENDED_DIR / "raw_panel_free.parquet")
    free = _free_feature_frame(raw_panel)
    orig = _original_feature_frame()

    sel = json.load(open(PROJECT_ROOT / "outputs" / "tables" / "selected_features.json"))
    frozen = sel["data_driven"] + sel["hypothesis"]

    # raw-source attribution per feature (for the report)
    dep = {f: [] for f in frozen + ROUTING_TRIGGERS_14}
    s, e = pd.Timestamp(overlap[0]), pd.Timestamp(overlap[1])
    rows = []
    targets = [("frozen36", f) for f in frozen] + [("routing", t) for t in ROUTING_TRIGGERS_14]
    seen = set()
    for role, feat in targets:
        if feat in seen:
            continue
        seen.add(feat)
        in_free = feat in free.columns
        in_orig = feat in orig.columns
        if not in_free or not in_orig:
            rows.append({"feature": feat, "role": role, "n_overlap": 0,
                         "corr": np.nan, "status": "GAP",
                         "note": "feature not reconstructable from free sources"})
            continue
        a = free[feat].reindex(pd.date_range(s, e, freq="W-TUE"))
        b = orig[feat].reindex(a.index)
        common = a.notna() & b.notna()
        n = int(common.sum())
        if n < 30:
            corr = np.nan
            status = "GAP"
            tier = "GAP"
            note = "no free source / insufficient overlap"
        else:
            corr = float(np.corrcoef(a[common], b[common])[0, 1])
            status = "PASS" if corr > corr_threshold else "FAIL"  # strict spec bar
            tier = ("PASS" if corr > 0.99 else "STRONG" if corr > 0.95
                    else "MODERATE" if corr > 0.85 else "WEAK" if corr > 0.50
                    else "GAP")
            note = _fail_note(feat, corr)
        rows.append({"feature": feat, "role": role, "n_overlap": n,
                     "corr": round(corr, 4) if np.isfinite(corr) else np.nan,
                     "status": status, "tier": tier,
                     "usable": tier not in ("GAP",), "note": note})

    df = pd.DataFrame(rows)
    # attach the free provider/proxy used (best-effort attribution)
    prov = {t: f"{v['provider']}:{v.get('symbol', v.get('maturity', v.get('key', v.get('series', ''))))}"
            for t, v in SOURCE_MAP.items()}
    df["proxies"] = df["feature"].map(lambda f: _attribute_sources(f, prov))
    df = df[["feature", "role", "n_overlap", "corr", "status", "tier", "usable",
             "proxies", "note"]]
    if save:
        (PROJECT_ROOT / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
        df.to_csv(PROJECT_ROOT / "outputs" / "tables" / "proxy_validation.csv", index=False)

    if verbose:
        npass = int((df["status"] == "PASS").sum())
        nfail = int((df["status"] == "FAIL").sum())
        ngap = int((df["status"] == "GAP").sum())
        tier_counts = df["tier"].value_counts().to_dict()
        print("=" * 95)
        print("WP2 — PROXY VALIDATION HARNESS (transformed series, overlap "
              f"{overlap[0]}..{overlap[1]}, PASS iff corr > {corr_threshold})")
        print("=" * 95)
        with pd.option_context("display.max_rows", None, "display.width", 175):
            print(df[["feature", "role", "n_overlap", "corr", "status", "tier", "note"]].to_string(index=False))
        print(f"\nStrict spec bar (>{corr_threshold}): PASS={npass}  FAIL={nfail}  GAP={ngap}  (of {len(df)})")
        print(f"Fidelity tiers: {tier_counts}")
        print("True GAP (no usable free instrument):",
              df[df.tier == "GAP"]["feature"].tolist())
        print("=> usable proxies (tier != GAP):", int(df["usable"].sum()), "of", len(df))
    return df


def _fail_note(feature: str, corr: float) -> str:
    """Diagnose why a feature falls below the strict 0.99 transformed-corr bar."""
    if corr > 0.99:
        return ""
    libor = {"US0001M", "libor_3m_spread_chg4w", "libor_3m_spread_level"}
    etf_tr = {"LMBITR", "LF94TRUU", "LP01TREU", "hy_ig_spread", "em_spread_chg4w"}
    if feature in libor:
        return "SOFR (secured) vs 1M LIBOR (unsecured): economic break in the spread"
    if feature in etf_tr:
        return "ETF total-return proxy vs Bloomberg index TR (different basket/duration)"
    if corr > 0.85:
        return "correct instrument; cross-vendor weekly-snapshot return noise"
    return "weak proxy: correct asset class, materially different series"


def _attribute_sources(feature: str, prov: dict) -> str:
    """Best-effort listing of which free proxies feed a feature (for the report)."""
    feat_dep = {
        "de_term_10y_2y": ["GTDEM10Y", "GTDEM2Y"], "de_term_10y_2y_chg4w": ["GTDEM10Y", "GTDEM2Y"],
        "em_spread_chg4w": ["LUACTRUU", "EMUSTRUU"], "equity_bond_rot": ["MXUS", "LUACTRUU"],
        "gold_oil_ratio": ["XAUBGNL", "Cl1"], "hy_ig_spread": ["LUACTRUU", "LF98TRUU"],
        "it_de_10y": ["GTITL10YR", "GTDEM10Y"], "it_de_10y_chg4w": ["GTITL10YR", "GTDEM10Y"],
        "jpy_strength": ["JPY"], "us_de_10y_chg4w": ["GT10", "GTDEM10Y"],
        "us_term_10y_2y_chg4w": ["GT10", "USGG2YR"], "vrp": ["VIX", "MXUS"],
        "equity_bond_corr_13w": ["MXUS", "LUACTRUU"], "dxy_chg4w": ["DXY"],
        "libor_3m_spread_chg4w": ["US0001M", "USGG3M"], "libor_3m_spread_level": ["US0001M", "USGG3M"],
        "us_10y_diff_chg4w": ["GT10"], "usa_world_relative": ["MXUS", "MXEU", "MXJP"],
        "real_yield_proxy_chg4w": ["GT10", "LF94TRUU"], "gold_oil_ratio_chg4w": ["XAUBGNL", "Cl1"],
        "vix_level": ["VIX"], "us_10y_vol_4w": ["GT10"], "us_term_10y_2y_level": ["GT10", "USGG2YR"],
        "mxus_drawdown_52w": ["MXUS"],
    }
    tickers = feat_dep.get(feature, [feature])
    return ", ".join(f"{t}->{prov.get(t, 'GAP')}" for t in tickers)


if __name__ == "__main__":
    EXTENDED_DIR.mkdir(parents=True, exist_ok=True)
    # Pull warm-up (2014) + the validation overlap + the extension in one span.
    # The 2014 lead gives clean rolling-52w / chg4w / corr-13w from 2015 on.
    panel = build_raw_panel("2014-01-01", "2025-12-31")
    panel.to_parquet(EXTENDED_DIR / "raw_panel_free.parquet")
    print(f"\nSaved free raw panel {panel.shape} -> "
          f"{EXTENDED_DIR / 'raw_panel_free.parquet'}")
    miss = {c: int(panel[c].isna().sum()) for c in panel.columns if panel[c].isna().any()}
    print("columns with missing weeks:", miss)
