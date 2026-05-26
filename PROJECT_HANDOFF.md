# PROJECT HANDOFF — Early-Warning System for Risk-Off Regimes + Allocation Routing

> **Purpose of this document.** Complete, self-contained state of the project as of commit
> `156d0dd` (Prompt 4.1). It is written so that a **second Claude Code instance can work in
> parallel** without re-reading the whole conversation. It documents: the problem, the dataset,
> everything already built (Prompts 1 → 4.1) with exact paths / signatures / schemas / findings,
> the **shared methodology for Prompts 5–12**, and the **detailed spec of each upcoming Prompt
> 5–12**. Read this top-to-bottom before writing any code.

---

## 0. TL;DR / Orientation

- **Goal.** Build an Early-Warning System (EWS) that flags **risk-off** weeks (binary target `Y`,
  1 = risk-off) on weekly Bloomberg data, then a **routing engine** that maps the risk signal +
  domain sub-scores into an allocation regime (LEVERED_EQUITY / CASH_USD / GOLD / MBS), and a
  **backtest** proving it beats 60/40 and buy-and-hold, with a **COVID stress test**.
- **Modeling style.** Unsupervised **novelty/anomaly detection** trained on *normal* (Y=0) weeks
  (MVG, One-Class SVM, Autoencoder, Isolation Forest), combined in an **ensemble**, validated with
  **purged expanding walk-forward CV + embargo**, evaluated on a **sealed 2019→2021 test holdout**
  that contains COVID.
- **Status.** Data pipeline, feature engineering, stationarity, collinearity cleanup, routing
  triggers, and the temporal / walk-forward splits are **done and committed**. Prompts 5–12
  (models, ensemble, routing, backtest, stress, deliverables) are **not started**.
- **Branch.** All work is on `claude/intelligent-einstein-DIHCx`. Commit there. Never push to `main`.

---

## 1. Environment & conventions

- **Working dir:** repo root (`NewTechHackaton`). Python 3.11. Linux.
- **Run modules as packages from repo root:** `python -m src.features`, `python -m src.splits`,
  etc. (the `src/` modules do `from src.data_loader import ...`, so the CWD must be the repo root).
- **Dependencies:** see `requirements.txt` — pandas, numpy, openpyxl, pyarrow, scikit-learn,
  tensorflow (bundles Keras), scipy, statsmodels, matplotlib, seaborn, jupyter. Installed ad-hoc in
  this environment via `pip install ...` (no venv). `tensorflow` is **not yet installed** here — it
  will be needed for the Autoencoder (Prompt 6).
- **Parquet engine:** pyarrow. All processed artifacts are `.parquet` with a weekly
  `DatetimeIndex` named `Date`.
- **Figures:** saved to `reports/figures/` so far (dpi≈120). **Prompts 5–12 should save figures to
  `outputs/figures/` at dpi=200** per the new spec (see §7). Models → `outputs/models/`, tables →
  `outputs/tables/`. These `outputs/` dirs **do not exist yet** — create them.
- **Notebooks:** built programmatically with `nbformat`, executed headless with
  `jupyter nbconvert --to notebook --execute --inplace <nb> --ExecutePreprocessor.timeout=...`.
  The one existing builder script (`build_notebook.py`) was a one-off and was deleted after use;
  follow the same pattern (write a temp builder, execute, delete) or author the `.ipynb` directly.
- **Commit style:** one commit per prompt, message prefix `Prompt N: ...`. Footer line
  `https://claude.ai/code/session_...` is auto-added by the commit helper. **Do not** put the model
  identifier in commits/code.
- **Git push:** `git push -u origin claude/intelligent-einstein-DIHCx` (retry w/ backoff on network
  errors). Do **not** open a PR unless explicitly asked.

### Repo tree (current)

```
src/
  data_loader.py        # load + clean raw Bloomberg xlsx, coverage report
  features.py           # stationarity transforms, spreads, routing triggers, collinearity cleanup
  splits.py             # time_split() [DEPRECATED] + walkforward_split()
data/
  raw/04_May_Zenti_exercises.xlsx
  processed/
    features_stationary.parquet          (1110 x 43)  incl. Y
    features_stationary_clean.parquet    (1110 x 40)  incl. Y  <-- collinearity-cleaned
    spreads.parquet                      (1107 x 20)
    spreads_clean.parquet                (1107 x 16)           <-- collinearity-cleaned
    routing_triggers.parquet             (1060 x 14)
    splits/        (Prompt 4 — linear split, DEPRECATED but kept)
      train/cv/test.parquet, routing_{train,cv,test}.parquet, train_normals.parquet
    walkforward/   (Prompt 4.1 — USE THIS)
      development.parquet, test_holdout.parquet
      fold_{1..5}_train.parquet, fold_{1..5}_val.parquet, fold_{1..5}_train_normals.parquet
      routing_development.parquet, routing_test_holdout.parquet
      routing_fold_{1..5}_train.parquet, routing_fold_{1..5}_val.parquet
notebooks/
  01_eda.ipynb          (executed)
reports/figures/        (EDA + correlation heatmaps)
requirements.txt
.gitignore              (ignores __pycache__, .ipynb_checkpoints, venv, .DS_Store)
```

---

## 2. The dataset

- **Source file:** `data/raw/04_May_Zenti_exercises.xlsx`. Two sheets:
  - `Markets` — 1111 weekly rows, 44 columns. Date column is **`Data`** (Italian), renamed to
    `Date` on load. Weekly frequency, exact 7-day spacing. Range **2000-01-11 → 2021-04-20**.
  - `Metadata` — ticker → description/type mapping (3 cols).
- **Target `Y`** = "Is it a risk-off period?" (1 = risk-off, 0 = risk-on). Binary.
  **Class balance:** 874 risk-on / 237 risk-off → **~21.3% risk-off prevalence overall.**
- **No missing values.** Every column is fully populated across all 1111 rows (the cleaning logic
  in `data_loader` is general-purpose but a no-op on this file).

### Y=1 (risk-off) distribution by year — **critical for CV design**

Risk-off clusters are concentrated in **2000–2002, 2008–2009, 2010–2012, and 2020**. Several
mid-decade years are calm:

| year | #Y=1 | year | #Y=1 | year | #Y=1 |
|---|---|---|---|---|---|
| 2000 | 19 | 2007 | 6  | 2014 | 2  |
| 2001 | 24 | 2008 | 27 | 2015 | 6  |
| 2002 | 27 | 2009 | 20 | 2016 | 4  |
| 2003 | 6  | 2010 | 14 | 2017 | **0** |
| 2004 | 0  | 2011 | 24 | 2018 | 6  |
| 2005 | 0  | 2012 | 27 | 2019 | **0** |
| 2006 | 2  | 2013 | **0** | 2020 | 23 |
|      |    |      |    | 2021 | 0 (16 wks) |

**Consequence:** walk-forward folds whose validation windows fall in 2013–2018 have **very few
positives** (see §6). Weighting fold metrics by `n_pos` (the shared methodology) is essential.

### Columns present in `Markets` (43 features + `Y`)

Equity (MSCI price indices): `MXUS MXEU MXJP MXBR MXRU MXIN MXCN`
FX spot: `DXY GBP JPY`  ·  Commodities: `XAUBGNL`(gold) `Cl1`(WTI 1st future) `CRY`(CRB) `BDIY`(Baltic Dry)
Bond total-return indices: `EMUSTRUU LF94TRUU LF98TRUU LG30TRUU LMBITR LP01TREU LUACTRUU LUMSTRUU`
US yields/rates: `GT10 USGG2YR USGG30YR USGG3M US0001M`  ·  EUR rate: `EONIA`
Sovereign yields: `GTDEM2Y GTDEM10Y GTDEM30Y`, `GTGBP2Y GTGBP20Y GTGBP30Y`,
`GTITL2YR GTITL10YR GTITL30YR`, `GTJPY2YR GTJPY10YR GTJPY30YR`
Stationary-ish: `VIX ECSURPUS`  ·  Target: `Y`

### ⚠️ Dataset gaps & proxy decisions (IMPORTANT — keep consistent across all prompts)

The Metadata sheet lists tickers that are **NOT in the Markets data**. We adopted these proxies
(documented in code) — **the parallel agent must use the same proxies**:

- **`MXWO` (MSCI World) is absent.** `msci_world_proxy` =
  - in EDA / `equity_bond_rot` spread: single name **`MXUS`**;
  - in `usa_world_relative` routing trigger: an **equal-weight DM composite of `MXUS, MXEU, MXJP`**
    (using `MXUS` alone there would make the feature identically zero).
  - For the **backtest benchmarks (Prompt 10)** the spec says "msci_world_proxy" — use **`MXUS`**
    (the established single-name proxy) unless you deliberately decide otherwise and document it.
- **`LEGATRUU` (Global Aggregate) is absent.** `global_agg_proxy` = **`LUACTRUU`** (US IG Corporate
  total-return). Used in `equity_bond_rot` and is the 60/40 bond leg in the backtest.
- **No credit *yields*, only total-return indices.** Credit/EM "spreads" are **log price-index
  ratios** (safe leg − risky leg), i.e. relative-performance proxies, NOT true OAS yield spreads.
  A rising value = risky leg underperforming = spreads widening = risk-off.

---

## 3. `src/data_loader.py` (Prompt 1)

Loads & cleans the raw Bloomberg export and prints a coverage report.

Key API (all importable):
- `PROJECT_ROOT`, `RAW_DIR`, `DEFAULT_FILE` — paths. `TARGET_COL = "Y"`, `DATE_COL = "Date"`.
- `load_metadata(path=None) -> DataFrame` — columns `[variable, description, type]`.
- `load_raw(path=None, sheet="Markets") -> DataFrame` — renames `Data`→`Date`, parses datetime,
  sorts, sets `Date` index. No row dropping.
- `trim_to_common_coverage(df)` — restricts to `[max(first_valid), min(last_valid)]` across columns
  then drops residual NaN rows (no-op on this dataset).
- `coverage_report(df) -> DataFrame` — per-feature first/last valid date, n_obs, n_missing, %missing.
- `load_dataset(path=None, sheet="Markets", verbose=True) -> DataFrame` — **the main entry point.**
  Returns the cleaned panel indexed by weekly `Date` (1111 rows, 43 feats + `Y`). Prints the
  coverage report when `verbose`.

Run: `python -m src.data_loader`.

---

## 4. `src/features.py` (Prompts 2, 3, 3.1)

### 4a. Stationarity transforms (Prompt 2)

`TRANSFORM_MAP: dict[col -> {'log_return','diff','level'}]` documents the per-column family:
- **`log_return`** = `np.log(x).diff()` — all price-like series: 7 MSCI equities, FX (`DXY GBP JPY`),
  `XAUBGNL Cl1 CRY BDIY`, and the 8 bond total-return indices.
- **`diff`** = `x.diff()` — all yields/rates: `GT10 USGG2YR USGG30YR USGG3M US0001M EONIA` and all
  GTDEM/GTGBP/GTITL/GTJPY tenors.
- **`level`** = passthrough — `VIX`, `ECSURPUS`, and `Y`.

Functions:
- `make_stationary(df, transform_map=TRANSFORM_MAP, dropna=True) -> DataFrame` — applies per-column
  transform, drops the single leading NaN row (1111→1110 rows).
- `adf_table(df, signif=0.05, transform_map=None) -> DataFrame` — Augmented Dickey-Fuller per column:
  `adf_stat, p_value, used_lag, n_obs, crit_1/5/10%, transform, stationary(bool)`.
- `build(verbose=True)` — load → make_stationary → ADF → save → also runs spreads + heatmap (below).
- **Output:** `data/processed/features_stationary.parquet` (1110 × 43, incl. `Y`).
- **Result:** all 43 features stationary at 5% (weakest: VIX p=0.0015, USGG3M p≈4e-5).

### 4b. Cross-asset spreads (Prompt 3): `build_spreads(df, horizon=4)`

Operates on the **cleaned raw-level** df (`load_dataset()`), returns spread features. For each
spread → **both level and `_chg4w`** (4-week change `= level.diff(4)`). Standalone derived features
get a single series.

- Term/sovereign (true yield-point spreads): `us_term_10y_3m` (GT10−USGG3M), `us_term_10y_2y`
  (GT10−USGG2YR), `de_term_10y_2y` (GTDEM10Y−GTDEM2Y), `it_de_10y` (BTP−Bund = GTITL10YR−GTDEM10Y),
  `us_de_10y` (GT10−GTDEM10Y).
- Credit/EM (log price-index ratios, safe−risky): `hy_spread` = `log(LUMSTRUU)−log(LF98TRUU)`,
  `hy_ig_spread` = `log(LUACTRUU)−log(LF98TRUU)`, `em_spread` = `log(LUACTRUU)−log(EMUSTRUU)`.
- Standalone (single series, NOT level+chg): `equity_bond_rot` = R4w(MXUS)−R4w(LUACTRUU);
  `gold_oil_ratio` = XAUBGNL/Cl1; `vrp` = VIX − annualized 4-week realized vol of MXUS
  (`log(MXUS).diff().rolling(4).std()*sqrt(52)*100`); `jpy_strength` = −R4w(JPY) (yen up = risk-off).
- `correlation_heatmap(originals, spreads, ...)` → saves `reports/figures/feature_spread_correlation.png`,
  returns redundant pairs |corr|≥0.9.
- **Output:** `data/processed/spreads.parquet` (1107 × 20).
- **ADF result:** the 8 spread **levels are persistent (non-stationary)** — expected; all `_chg4w`
  and the 4 standalone features are stationary at 5%.

### 4c. Collinearity cleanup + routing triggers (Prompt 3.1)

`remove_collinear_features(df, drop_list) -> df` drops the present columns in `drop_list`.
`COLLINEAR_DROP = [hy_spread, hy_spread_chg4w, us_term_10y_3m, us_term_10y_3m_chg4w, GTGBP20Y,
GTDEM30Y, USGG30YR]` (each dropped twin had |corr|>0.9 with a kept survivor: keep `hy_ig_spread`,
`us_term_10y_2y`, `GTGBP30Y`, `GTDEM10Y`, `GT10`).
- **Outputs:** `features_stationary_clean.parquet` (43→**40**), `spreads_clean.parquet` (20→**16**).

`build_routing_triggers(df_raw, df_stationary, df_spreads, horizon=4) -> DataFrame` — **14 unique
triggers** across 3 domains (`dxy_chg4w` is shared by USD & Oro with opposite economic reading):

```
USD: libor_3m_spread_chg4w, dxy_chg4w, vrp, us_10y_diff_chg4w, usa_world_relative
Oro: real_yield_proxy_chg4w, dxy_chg4w, jpy_strength, equity_bond_corr_13w, gold_oil_ratio_chg4w
MBS: vix_level, us_10y_vol_4w, us_term_10y_2y_level, libor_3m_spread_level, mxus_drawdown_52w
```
Construction details (exact):
- `libor_3m_spread_chg4w` = `(US0001M − USGG3M).diff(4)`
- `dxy_chg4w` = `log(DXY).diff(4)`
- `vrp` = reused from spreads
- `us_10y_diff_chg4w` = `GT10.diff().diff(4)` (4w change of the 1st difference)
- `usa_world_relative` = `log(MXUS).diff(4) − mean(log[MXUS,MXEU,MXJP].diff(4))`
- `real_yield_proxy_chg4w` = `(GT10 − log(LF94TRUU).diff(4)).diff(4)`
- `jpy_strength` = reused from spreads
- `equity_bond_corr_13w` = `df_stationary['MXUS'].rolling(13).corr(df_stationary['LUACTRUU'])`
- `gold_oil_ratio_chg4w` = `df_spreads['gold_oil_ratio'].diff(4)`
- `vix_level` = `VIX`
- `us_10y_vol_4w` = `GT10.diff().rolling(4).std()`
- `us_term_10y_2y_level` = `GT10 − USGG2YR`
- `libor_3m_spread_level` = `US0001M − USGG3M`
- `mxus_drawdown_52w` = `MXUS / MXUS.rolling(52).max() − 1`
- **Output:** `data/processed/routing_triggers.parquet` (**1060 × 14**, starts 2001-01-02 because of
  the 52-week rolling warm-up).

`routing_correlation(triggers, ...)` → 15-slot domain-ordered Pearson matrix (dxy duplicated),
heatmap `reports/figures/routing_triggers_correlation.png` with block separators, prints |corr|>0.7
pairs annotated intra/cross-domain. **Findings:** `USD:dxy_chg4w ~ Oro:dxy_chg4w` = 1.00
(deliberate, NOT removed), `vix_level ~ mxus_drawdown_52w` = −0.83 (intra-MBS, expected).

`routing_trigger_summary(triggers)` — ADF sanity excluding the rule-based level filters
`RULE_BASED_LEVELS = {vix_level, us_term_10y_2y_level, libor_3m_spread_level, mxus_drawdown_52w}`.
**Result: 11/11 tested triggers stationary at 5%.**

`build_routing(verbose=True)` orchestrates Prompt 3.1 (cleanup → triggers → corr → ADF). `__main__`
runs `build(); build_routing()`.

Constants the parallel agent will reuse: `HORIZON_4W=4`, `REALIZED_VOL_WEEKS=4`, `WEEKS_PER_YEAR=52`,
`ROUTING_DOMAINS`, `TRIGGER_TYPE`, `RULE_BASED_LEVELS`, the path constants
`PROCESSED_DIR, FIG_DIR, FEATURES_CLEAN_PATH, SPREADS_CLEAN_PATH, ROUTING_PATH`.

---

## 5. `src/splits.py` — Prompt 4 (DEPRECATED) + Prompt 4.1 (USE THIS)

### 5a. `time_split(train_end='2015-12-31', cv_end='2018-12-31', save=True, verbose=True)` — **DEPRECATED**
Linear train/cv/test. Kept for back-compat only. Marked `DEPRECATED: use walkforward_split()`.
Wrote `data/processed/splits/`. **Do not use in Prompts 5–12.**

### 5b. `walkforward_split(embargo_weeks=4, save=True, verbose=True)` — **canonical splitter**

Returns `{'models': wf_models, 'routing': wf_routing}`. Each `wf` dict:
```python
{
  'development':   df_dev,        # all weeks <= 2018-12-31  (models: 986 rows; routing: 939)
  'test_holdout':  df_test,       # 2019-01-01 -> 2021-04-20 (121 rows, contains COVID)
  'cv_folds': [ {'fold_id':i, 'train':df, 'val':df, 'crisis_captured':str}, ... ],  # 5 folds
  'train_normals_per_fold': [ {'fold_id':i, 'train_normals':df_Y0}, ... ],  # MODELS ONLY (needs Y)
}
```
- **Model feature set = `features_stationary_clean` (40) ⨝ `spreads_clean` (16) inner-join = 56 cols
  incl. `Y`.** (`_load_model_features()`.) Routing feature set = `routing_triggers` (14).
- **Expanding** train (always starts at dataset start). **Embargo:** exactly `embargo_weeks`
  observations immediately after each train end are excluded from BOTH train and val (kills 4w
  lookback leakage). Asserted: `train.max() < val.min()` and gap == `embargo_weeks`.
- `DEV_END='2018-12-31'`. `WALKFORWARD_FOLDS = [(id, train_end, val_end, crisis), ...]`:
  `(1,'2006-12-31','2009-12-31','GFC 2008')`, `(2,'2009-12-31','2012-12-31','Euro 2011')`,
  `(3,'2012-12-31','2014-12-31','Taper 2013')`, `(4,'2014-12-31','2016-12-31','China-Oil 2015-16')`,
  `(5,'2016-12-31','2018-12-31','Q4 2018 selloff')`. (val_end of fold i == train_end of fold i+1.)
- **Outputs:** `data/processed/walkforward/` — `development.parquet`, `test_holdout.parquet`,
  `fold_{i}_train/val.parquet`, `fold_{i}_train_normals.parquet`, and `routing_*` analogues.
- Run: `python -m src.splits`.

### 5c. Walk-forward fold reality (MEASURED) — read before tuning

| fold | train period | val period | n_tr | n_val | val Y=1% | val #Y=1 | crisis |
|---|---|---|---|---|---|---|---|
| 1 | 2000-02 → 2006-12 | 2007-01 → 2009-12 | 360 | 153 | **34.6%** | **53** | GFC 2008 |
| 2 | 2000-02 → 2009-12 | 2010-02 → 2012-12 | 517 | 152 | **42.1%** | **64** | Euro 2011 |
| 3 | 2000-02 → 2012-12 | 2013-01 → 2014-12 | 673 | 101 | 2.0% | **2** | Taper 2013 |
| 4 | 2000-02 → 2014-12 | 2015-02 → 2016-12 | 778 | 100 | 10.0% | 10 | China-Oil 2015-16 |
| 5 | 2000-02 → 2016-12 | 2017-01 → 2018-12 | 882 | 100 | 6.0% | 6 | Q4 2018 selloff |

- Sanity checks implemented: COVID in `test_holdout` (PASS, hard assert); per-fold non-blocking
  WARN if prevalence <10% / NOTE if <12%; WARN if #pos ≤ 20.
- **Folds 1–2 are the statistically reliable folds (53 & 64 positives). Folds 3–5 are thin**
  (2, 10, 6 positives) because 2013–2018 is a low-volatility regime in this `Y` labeling. This is a
  **data property, not a bug.** ⇒ The weighted-by-`n_pos` aggregation in the shared methodology is
  what makes the CV meaningful: **fold 1–2 dominate, fold 3–5 contribute at the margin.**

---

## 6. SHARED METHODOLOGY for Prompts 5–12 (apply everywhere)

1. **Weighted walk-forward tuning.** For every hyperparameter / threshold, compute the metric
   (default **F1**) on each fold's `val`, then aggregate across folds with a **mean/median weighted
   by the fold's number of positives `n_pos`**. Folds 1–2 dominate; folds 3–5 contribute at the
   margin (they are statistically noisy). Pick the config maximizing the weighted/median metric.
2. **Scaler per fold.** `StandardScaler` fit **only on each fold's train** (then applied to that
   fold's val). For the **test holdout**, fit the scaler on the **entire development set
   (2000–2018)**. NEVER fit a scaler on the full dataset or on any val/test data. This is the
   `FoldScaler` class to be built in Prompt 5 (`src/preprocessing.py`).
3. **Extended model feature set = 57 columns.** Add **`equity_bond_corr_13w`** (currently only in
   `routing_triggers.parquet`) to the 56-col model feature set → **57 cols (of which 1 is `Y`)**.
   Rationale: it discriminates liquidity-driven from inflation-driven crises. ⇒ When loading model
   data for Prompts 5–12, join `equity_bond_corr_13w` from `routing_triggers.parquet` onto the
   model walk-forward frames (align on `Date`; mind that routing starts 2001 so the earliest
   model-train weeks may get NaN for this one column — handle by dropping those rows or
   forward-considering; **document whatever you choose**). So **X has 56 numeric features + `Y`.**
4. **Metrics to report EVERYWHERE:** **F1, AUC-PR (average precision), AUC-ROC, Precision, Recall,
   and F-beta with beta=2** (recall-weighted, because missing a crisis is costlier than a false
   alarm). Report per-fold AND on the test holdout.

> Anomaly-detection framing: models are **fit on normal weeks only (`Y==0`)** using the
> `train_normals` / `train_normals_per_fold` artifacts. They output an **anomaly score**; a
> **threshold** converts score→binary risk-off flag; the threshold is tuned by weighted
> walk-forward. Higher score = more anomalous = more risk-off.

---

## 7. UPCOMING PROMPTS 5–12 (full spec to implement)

> Conventions for all: save **models → `outputs/models/`**, **tables → `outputs/tables/`**,
> **figures → `outputs/figures/` at dpi=200**, **notebooks → `notebooks/`**. Create these dirs.
> Use `walkforward_split()` for all CV. Use the 57-col extended feature set. Report the 6 metrics.

### Prompt 5 — MVG baseline
- **`src/preprocessing.py`** with **`FoldScaler`**: per-fold `StandardScaler` (fit on fold train) +
  a "final" scaler (fit on full development set) for the test holdout. (See methodology #2.)
- **`src/models.py`** with **`MVGAnomalyDetector`**:
  - **Ledoit-Wolf shrinkage mandatory** (`sklearn.covariance.LedoitWolf`) — Σ is unstable with 57
    features and ~360 normals in Fold 1.
  - Fit on **normals (Y=0)** of the train. Score = **squared Mahalanobis distance**.
  - Tune threshold **ε** via walk-forward **weighted by `n_pos`**.
- **Outputs:** notebook `notebooks/03_models_mvg.ipynb`; final model `outputs/models/mvg_final.pkl`;
  per-fold + test-holdout table `outputs/tables/mvg_walkforward.csv`.

### Prompt 6 — SVM, AE, IF (same interface as MVG)
Extend `src/models.py` with three classes, **identical interface** to `MVGAnomalyDetector`
(`.fit(X_normals)`, `.score(X)`, threshold tuning, `.predict(X)`):
- **`OneClassSVMDetector`**: RBF kernel; grid `nu ∈ {0.05,0.10,0.15,0.22}` × `gamma ∈
  {scale,auto,0.01,0.001}`; select by **weighted F1**.
- **`AutoencoderDetector`**: architecture **57→24→12→6→12→24→57**, dropout **0.15**, MSE loss, Adam
  **lr=1e-3**, **early stopping on a temporal sub-split of the train** (NEVER on the val fold — that
  would leak), **TF seed=42**. Score = reconstruction MSE.
- **`IsolationForestDetector`**: `n_estimators=200`, `random_state=42`; grid
  `contamination ∈ {0.05,0.10,0.15,0.22}`.
- **Outputs:** notebook `notebooks/04_models_all.ipynb`; serialized models in `outputs/models/`;
  comparative table of all 4 models.
- **Note:** needs `tensorflow` installed (not yet present in this env).

### Prompt 7 — Ensemble (soft + hard voting)
**`src/ensemble.py`**:
- `score_to_percentile(s_train_normals, s_new)`: empirical percentile of new scores vs the
  train-normals score distribution (per model, to make scores comparable).
- **`EnsembleDetector`** with 3 modes: **hard voting** (majority 3/4), **soft voting mean**, **soft
  voting median** (on percentile-mapped scores).
- Tune threshold **τ** of the soft versions via walk-forward weighted.
- **Error-correlation analysis 4×4:** if mean pairwise correlation of the binary errors > **0.85**,
  print a warning (ensemble adds little).
- **Outputs:** notebook `notebooks/05_ensemble.ipynb`; table comparing 4 singles + 3 ensembles;
  percentile scatter between models; error-correlation heatmap.

### Prompt 8 — Domain sub-scores (USD, Oro, MBS)
**`src/routing.py`** with **`compute_subscore_zscore()`**. Signed z-score weights:
```python
SUBSCORE_USD = {'libor_3m_spread_chg4w': +1, 'dxy_chg4w': +1, 'vrp': +1,
                'us_10y_diff_chg4w': -1, 'usa_world_relative': +1}
SUBSCORE_ORO = {'real_yield_proxy_chg4w': -1, 'dxy_chg4w': -1, 'jpy_strength': +1,
                'equity_bond_corr_13w': +1, 'gold_oil_ratio_chg4w': +1}
```
- Sub-score = **mean of signed z-scores**, with **μ, σ estimated on the development set only**
  (never on test). (z = (x−μ_dev)/σ_dev, times the sign, averaged over the 5 features.)
- **MBS sub-score is binary rule-based:**
  - **active (=1)** if `VIX ∈ [20,28]` AND `us_term_10y_2y_level > 0` AND
    `mxus_drawdown_52w ∈ [−12%, −5%]`.
  - **blocked (=0)** if `VIX > 30` OR `libor_3m_spread_level > p90_dev` OR
    `hy_ig_spread_chg4w > +50bps`. (`hy_ig_spread_chg4w` lives in `spreads_clean.parquet`;
    `p90_dev` = 90th pct of `libor_3m_spread_level` on development set.)
- **Outputs:** notebook `notebooks/06_subscores.ipynb`; parquet with sub-scores for the test holdout
  and for each fold val.

### Prompt 9 — Routing engine + decision matrix
Extend `src/routing.py` with **`RoutingEngine`** / `route_allocation(...)`:
```python
def route_allocation(ensemble_signal, sub_usd, sub_oro, sub_mbs, dxy_chg4w, thresholds):
    if ensemble_signal == 0:
        return 'LEVERED_EQUITY'          # 1.5x equity, -0.5x cash
    if sub_usd > thresholds['usd'] and dxy_chg4w > 0:
        return 'CASH_USD'
    elif sub_oro > thresholds['oro']:
        return 'GOLD'
    elif sub_mbs == 1:
        return 'MBS'
    else:
        return 'CASH_USD'                # default
```
Default thresholds `{'usd': 1.0, 'oro': 1.0}` (optimization deferred to Prompt 10).
- **Outputs:** notebook `notebooks/07_routing.ipynb`; test-holdout timeline colored by allocation
  regime; regime-count table.

### Prompt 10 — Backtest + threshold optimization
**`src/backtest.py`** with `backtest_strategy(allocations, prices, tc_dict)`:
- Allocation → weights: **LEVERED_EQUITY = +1.5x equity / −0.5x cash; CASH_USD = +1.0x cash;
  GOLD = +1.0x gold; MBS = +1.0x MBS.**
- Transaction costs (bps): **LEVERED=5, CASH=2, GOLD=8, MBS=20.**
- **Benchmark 1:** static 60/40 = 60% `msci_world_proxy` (=MXUS) + 40% `LUACTRUU`.
- **Benchmark 2:** `msci_world_proxy` (MXUS) buy-and-hold.
- Metrics: **CAGR, annual vol, Sharpe, Sortino, max DD, Calmar, turnover, risk-off hit ratio,
  per-regime performance.**
- **`optimize_routing_thresholds()`:** grid `usd ∈ {0.5,0.75,1.0,1.25,1.5,2.0}` ×
  `oro ∈ {0.5,0.75,1.0,1.25,1.5}`. For each combo, backtest on each fold val, **Calmar weighted by
  duration**; pick thresholds with **max median Calmar**.
- Complementary metric: **cost-weighted score `C = 0.10·n_FN + 0.005·n_FP`** (α/β = 20 → asymmetric:
  a missed crisis costs 20× a false alarm).
- **Cash/equity/gold/MBS price series** for the backtest come from the **raw levels** in
  `load_dataset()`: equity = `MXUS`, gold = `XAUBGNL`, MBS = `LUMSTRUU` total-return, cash = a
  short-rate accrual (e.g. from `USGG3M`/`US0001M`; **decide & document** — likely accrue
  `USGG3M/52` weekly). 60/40 bond leg = `LUACTRUU`.
- **Outputs:** notebook `notebooks/08_backtest.ipynb`; equity curve vs benchmarks; drawdown chart;
  Calmar grid-search heatmap; metrics table.

### Prompt 11 — COVID stress test (test-holdout deep dive)
Notebook `notebooks/09_stress.ipynb`. Windows:
1. **COVID crash** 2020-02-15 → 2020-04-15
2. **COVID recovery** 2020-04-15 → 2020-12-31
3. **Reflation / early 2021** 2021-01-01 → end of test
Per window: cumulative return strategy vs benchmark, max DD, dominant safe-haven asset, #switches,
**lead time** (weeks the first risk-off flag precedes the real drawdown).
- **Outputs:** crisis-zoom timeline with regime shading; multi-window summary table; narrative md.

### Prompt 12 — Final deliverables
1. **End-to-end notebook `notebooks/EWS_GSoM_PoliMI.ipynb`**: orchestrates all `src/` modules in
   sequence, **14 sections** with narrative markdown, **runtime < 15 min**.
2. **Extended abstract `outputs/abstract.md`** (1–2 pages): problem, methodology (walk-forward CV +
   ensemble + routing), key results, business implications, limitations.
3. **Pitch deck outline `outputs/deck_outline.md`** (15 slides): cover, problem, dataset, pipeline,
   stationarity, walk-forward, 4 models, ensemble, routing engine, decision matrix, threshold
   tuning, backtest, COVID stress test, limitations, Q&A backup.
- High-res figures (**dpi=200**) in `outputs/figures/`.

---

## 8. Risks / gotchas for the parallel agent

- **Thin folds 3–5.** Do not trust unweighted CV means; always weight by `n_pos`. Fold 3 has only 2
  positives — its F1 is near-degenerate. Consider reporting both weighted-mean and median.
- **57th feature alignment.** `equity_bond_corr_13w` starts later (needs 13w + the routing 52w
  warm-up). Joining it onto early model-train weeks introduces NaNs — decide drop-vs-keep and keep
  it consistent across folds AND the test-holdout final scaler. Document it.
- **Scaler leakage.** The #1 way to silently break this project. Per-fold scaler on fold train only;
  test-holdout scaler on development only. The Autoencoder early-stopping split must come **from
  within the train**, never the val fold.
- **Proxy consistency.** MSCI World→MXUS (single name) for backtest/EDA; DM composite only inside
  `usa_world_relative`. Global Agg→LUACTRUU. Credit spreads are log-ratios, not yields.
- **Sign conventions.** Higher anomaly score = risk-off. `jpy_strength` positive = yen strengthening
  = risk-off. `dxy_chg4w` shared USD/Oro with **opposite** sign in the sub-score weights.
- **`Y` is in the model frames.** Remember to drop `Y` from `X` before scoring/fitting; fit anomaly
  models on `X[Y==0]` rows only.
- **Determinism.** Seeds: TF=42, IsolationForest random_state=42. Set numpy seed where relevant.
- **Runtime budget.** Prompt 12 end-to-end < 15 min ⇒ keep AE epochs modest with early stopping;
  cache intermediate parquets.

---

## 9. Quick-start for the parallel session

```bash
# from repo root, on branch claude/intelligent-einstein-DIHCx
pip install -r requirements.txt          # tensorflow needed for Prompt 6
python -m src.data_loader                # sanity: coverage report
python -m src.features                   # rebuilds all processed parquets + heatmaps
python -m src.splits                     # rebuilds data/processed/walkforward/*
# then start Prompt 5: create src/preprocessing.py (FoldScaler) + src/models.py (MVG)
```

Load walk-forward splits in code:
```python
from src.splits import walkforward_split
wf = walkforward_split(embargo_weeks=4, save=False, verbose=False)
folds = wf['models']['cv_folds']         # list of {fold_id, train, val, crisis_captured}
normals = wf['models']['train_normals_per_fold']   # fit anomaly models on these (Y==0)
dev, test = wf['models']['development'], wf['models']['test_holdout']
# remember: add equity_bond_corr_13w (57-col set), drop Y from X, scale per methodology
```

---
*End of handoff. Generated at Prompt 4.1 (commit `156d0dd`). No existing code/data/notebooks were
modified to produce this document.*
