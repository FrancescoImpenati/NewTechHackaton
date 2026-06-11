# PROJECT HANDOFF V2 — Final State + Next-Phase Context

> **Supersedes `PROJECT_HANDOFF.md`** (written at Prompt 4.1, now stale). This documents the
> **final** state at commit `505b866` on branch `claude/intelligent-einstein-DIHCx`.
> Project: an Early-Warning System (EWS) for risk-off detection + a routing engine that allocates
> 1.5× levered equity in calm weeks and rotates into USD-cash / Gold / MBS havens in risk-off weeks,
> with a walk-forward backtest vs 5 benchmarks. All numbers below are read from the committed
> notebook outputs / `outputs/tables/*.csv` and were re-verified live while writing this file.

Quick facts: Python 3.11 · pandas/numpy/scikit-learn/statsmodels/tensorflow-cpu 2.21/pyarrow/
matplotlib/seaborn/networkx/python-pptx. `src/__init__.py` exists (regular package). Run modules
as `python -m src.X` **from repo root**. TF/AE is the only nondeterministic component.

---

## 1. FINAL STATE OF `src/`

Pipeline order: `data_loader → features → splits → preprocessing → models → ensemble → routing →
backtest`. Cross-module imports use `from src.X import Y` (so cwd must be repo root or `src/` on path).

### 1.1 `data_loader.py`
| Symbol | Signature → returns |
|---|---|
| consts | `PROJECT_ROOT`, `RAW_DIR`, `DEFAULT_FILE=data/raw/04_May_Zenti_exercises.xlsx`, `RAW_DATE_COL="Data"`, `DATE_COL="Date"`, `TARGET_COL="Y"` |
| `load_metadata(path=None)` | → DataFrame `[variable, description, type]` (Metadata sheet) |
| `load_raw(path=None, sheet="Markets")` | → DataFrame, `Data`→`Date` index, sorted, no rows dropped |
| `trim_to_common_coverage(df)` | → DataFrame trimmed to `[max first-valid, min last-valid]`, drops residual NaN rows |
| `coverage_report(df)` | → per-feature first/last valid, n_obs, n_missing, pct_missing |
| `load_dataset(path=None, sheet="Markets", verbose=True)` | **main entry**: → cleaned weekly DataFrame **(1111×43, incl. Y)** |

Raw data has **no missing values**; `trim_to_common_coverage` is effectively a no-op on this file.

### 1.2 `features.py`
| Symbol | Signature → returns / notes |
|---|---|
| consts | `HORIZON_4W=4`, `REALIZED_VOL_WEEKS=4`, `WEEKS_PER_YEAR=52`; paths `OUTPUT_PATH`, `SPREADS_PATH`, `FEATURES_CLEAN_PATH`, `SPREADS_CLEAN_PATH`, `ROUTING_PATH` |
| `TRANSFORM_MAP` | dict col→{`log_return`,`diff`,`level`} (per-family stationarity) |
| `make_stationary(df, transform_map=TRANSFORM_MAP, dropna=True)` | → stationary DataFrame (drops 1 leading row) |
| `adf_table(df, signif=0.05, transform_map=None)` | → ADF stat/p/lags/crit/stationary per column |
| `build_spreads(df, horizon=4)` | → 20 spread cols (8 spreads × {level,`_chg4w`} + 4 standalone). Needs raw-level df |
| `correlation_heatmap(originals, spreads, path=…, redundancy_threshold=0.9)` | saves PNG, → redundant pairs |
| `COLLINEAR_DROP` | `[hy_spread, hy_spread_chg4w, us_term_10y_3m, us_term_10y_3m_chg4w, GTGBP20Y, GTDEM30Y, USGG30YR]` |
| `remove_collinear_features(df, drop_list)` | → df without present drop cols |
| `ROUTING_DOMAINS` | `{USD:[5], Oro:[5], MBS:[5]}` — 15 slots, `dxy_chg4w` shared USD+Oro (13 unique cols, but `vrp` also appears → see below) |
| `TRIGGER_TYPE`, `RULE_BASED_LEVELS` | level/variation labels; 4 MBS levels excluded from ADF |
| `build_routing_triggers(df_raw, df_stationary, df_spreads, horizon=4)` | → **14 unique trigger cols** |
| `routing_correlation(triggers, …)` | saves 15-slot heatmap, → \|corr\|>0.7 pairs |
| `routing_trigger_summary(triggers, signif=0.05)` | → ADF summary (excludes rule-based levels) |
| `build(verbose=True)` | writes `features_stationary.parquet`, `spreads.parquet` + heatmap; → stationary df |
| `build_routing(verbose=True)` | writes `*_clean.parquet` + `routing_triggers.parquet`; → triggers |
| `__main__` | runs `build()` then `build_routing()` (full regen from raw) |

Spread families: `us_term_10y_3m`, `us_term_10y_2y`, `de_term_10y_2y` (yield spreads); `it_de_10y`,
`us_de_10y` (sovereign); `hy_spread`, `hy_ig_spread`, `em_spread` (credit **log-ratio proxies**, NOT
OAS bps); standalone `equity_bond_rot`, `gold_oil_ratio`, `vrp`, `jpy_strength`.

### 1.3 `splits.py`
| Symbol | Signature → returns |
|---|---|
| consts | `DEV_END="2018-12-31"`, `WALKFORWARD_FOLDS` (5×(id,train_end,val_end,crisis)), `KNOWN_CRISES` |
| `time_split(train_end="2015-12-31", cv_end="2018-12-31", save, verbose)` | **DEPRECATED** (still works, writes `data/processed/splits/`) |
| `walkforward_split(embargo_weeks=4, save=True, verbose=True)` | **canonical**: → `{'models':wf, 'routing':wf}` |

`wf` dict = `{'development', 'test_holdout', 'cv_folds':[{fold_id,train,val,crisis_captured}],
'train_normals_per_fold':[{fold_id,train_normals}]}` (last key models-only). Model feature set =
`features_stationary_clean` (40 incl Y) ⨝ `spreads_clean` (16) = **56 columns (55 features + Y)**.
Embargo = exactly `embargo_weeks` obs excluded between each train_end and val_start (asserted).

### 1.4 `preprocessing.py` — `FoldScaler(target_col="Y")`
`fit_per_fold(folds_data)` (one `StandardScaler` per fold-train, all rows) · `transform(X, fold_id)` ·
`fit_on_development(df_dev)` · `transform_holdout(X)` (final scaler) · `save(path)` / `load(path)`.
Drops `Y`, locks `feature_cols_` order on first call. **`feature_cols_` = 56** (the 57-col set minus Y).

### 1.5 `models.py` — 4 detectors + shared helpers
Shared: `compute_metrics(y_true,y_pred,scores)`→{F1,Precision,Recall,AUC_ROC,AUC_PR,F_beta_2};
`weighted_median(values,weights)`; `weighted_mean(values,weights)`; `_tune_threshold(scores_train_normals,
scores_val,y_val,percentiles=range(80,100))`→`(eps,metrics)`; `_weighted_summary(results,threshold)`.

Common interface (all 4): `fit(X_normals)` · `score_samples(X)` (higher=more anomalous) ·
`predict(X)` · `weighted_summary()` · `save(path)` / `load(path)` · attrs `threshold_`,
`walkforward_results_`, `best_params_`.

| Model | tuning method | grids / arch |
|---|---|---|
| `MVGAnomalyDetector(shrinkage="ledoit-wolf")` | `fit_threshold_walkforward(folds,scaler,percentiles=range(80,100))` | score=squared Mahalanobis via LedoitWolf precision; ε = n_pos-weighted **median** of per-fold F1-optimal thresholds |
| `OneClassSVMDetector(nu=.1,gamma="scale")` | `fit_hyperparams_walkforward(folds,scaler,…)` | `NU_GRID=[.05,.10,.15,.22]` × `GAMMA_GRID=["scale","auto",.01,.001]` (16); pick by weighted-median F1 |
| `IsolationForestDetector(contamination=.1,n_estimators=200,random_state=42)` | `fit_hyperparams_walkforward(folds,scaler)` | `CONTAMINATION_GRID=[.05,.10,.15,.22]`; score=`-score_samples` |
| `AutoencoderDetector(dropout=.15,lr=1e-3,max_epochs=200,batch_size=32,patience=15,seed=42)` | `fit_threshold_walkforward(folds,scaler,…)` | arch **56-24-12-6-12-24-56**, ReLU + linear out, dropout on 3 encoder layers; score=row MSE |

### 1.6 `ensemble.py`
`MODEL_ORDER=["mvg","svm","ae","if"]`. `score_to_percentile(s_reference,s_new)`→[0,1] empirical CDF.
`clone_unfit(model)`→fresh detector same hyperparams.
`EnsembleDetector(mode, models_dict, scalers_dict)`, `mode∈{hard,soft_mean,soft_median}`:
`decision_scores(X,reference_scores_dict,fold_id=None)` · `score_samples(...)` · `predict(...)` ·
`fit_threshold_walkforward(folds_data, fold_scores=None, target_col="Y", percentiles=range(80,100))`
→ per-fold df, sets `threshold_`(τ) for soft modes · `weighted_summary()` ·
`save(path, reference_scores_dict)` (META only) · `load_meta(path)`→dict.

### 1.7 `routing.py`
`SUBSCORE_USD`, `SUBSCORE_ORO` (signed-weight dicts), `MBS_REQUIRED` cols, `DEFAULT_THRESHOLDS={usd:1,oro:1}`,
`ALLOCATION_WEIGHTS` (LEVERED_EQUITY=equity1.5/cash-0.5; CASH_USD=cash1; GOLD=gold1; MBS=mbs1).
`fit_zscore_params(df_dev,sign_dict)`→(mu,sigma) · `compute_subscore_zscore(triggers,mu,sigma,sign)`→array ·
`fit_mbs_params(df_dev)`→{p90_dev} · `compute_subscore_mbs(triggers,params)`→binary array ·
`compute_all_subscores(...)`→DataFrame[subscore_usd,subscore_oro,subscore_mbs] ·
`route_allocation(ens_sig,sub_usd,sub_oro,sub_mbs,dxy_chg4w,th)`→label ·
`RoutingEngine(thresholds=None).route_series(df,thresholds=None)`→Series (needs cols
`ens_sig,subscore_usd,subscore_oro,subscore_mbs,dxy_chg4w`) · `RoutingEngine.allocations_to_weights(alloc)`→weights ·
`optimize_routing_thresholds(folds_data, ensemble_per_fold, subscores_per_fold, prices, tc_dict,
usd_grid=(.5,.75,1,1.25,1.5), oro_grid=(.5,.75,1,1.25,1.5))`→`(best_thresholds, grid_df)` (duration-weighted median Calmar).
MBS active: `20≤vix≤28 & term_10y_2y>0 & -.12≤dd≤-.05`; blocked: `vix>30 | libor>p90_dev | hy_ig_spread_chg4w>0.005`.

### 1.8 `backtest.py`
`ASSET_COLS=["equity","bond","gold","cash","mbs"]`, `WEEKS_PER_YEAR=52`, `COVID_CRASH=("2020-02-15","2020-04-15")`.
`backtest_strategy(weights_ts, returns_ts, tc_dict)`→{equity_curve,net,gross,tc} — **weights lagged 1 week**
(`w.shift(1)`), TC on |Δweight|. `risk_parity_weights(returns_df, window=52)`→inv-vol weights.
`compute_full_metrics(equity_curve, weekly_returns, allocations_series=None, rf_series=None,
benchmark_returns=None, tc_series=None, y_series=None)`→22-key dict (CAGR…Hit_Ratio_RiskOff).

### 1.9 DIVERGENCES from the v1 prompt spec (read carefully)
1. **AE width 56, not 57.** Spec said `57→…→57`. The "57" counted `Y`; `Y` is dropped from X → input is
   **56 features**. `AutoencoderDetector` builds from `input_dim_=X.shape[1]=56`. (`56 = 55 from the split set + equity_bond_corr_13w`.)
2. **"56 features + Y = 57 columns"** is the real model matrix. Module docstrings/labels saying "57-col"
   mean *columns incl Y*; the scaler/model see **56**. Don't confuse the two.
3. **AE persistence**: saved as `*.keras` **plus** a sidecar `*.meta.json` (threshold_, input_dim_,
   hyperparams). `AutoencoderDetector.load()` needs **both** files.
4. **IsolationForest tuning**: `score_samples` is contamination-invariant, so a per-contamination F1
   threshold would be degenerate. Contamination is used as the **operating point** (boundary `=-offset_`);
   grid picked by n_pos-weighted median val-F1. **Optimal = 0.10** (not ≥0.15 as the spec hoped).
5. **MVG ε transfer**: ε tuned on per-fold (per-fold-scaler) models is applied to the development-fit
   final model. Valid because standardized squared-Mahalanobis ≈ χ²(56), but it is an approximation.
6. **`EnsembleDetector.save()` stores META only** (mode, τ, model_order, reference_scores_dict,
   walkforward_results) — the Keras AE isn't picklable. There is **no** `EnsembleDetector.load()`;
   reconstruct inference manually (see §2 snippet) or rebuild `EnsembleDetector(mode, models, scalers)` and
   set `.threshold_=tau`.
7. **`vrp`, `jpy_strength`, `gold_oil_ratio` are in BOTH the 56 model features (via `spreads_clean`) AND
   the routing triggers** (`build_routing_triggers` reuses `df_spreads["vrp"]` etc.). For feature
   selection: `vrp` and `jpy_strength` are literally duplicated across the model set and the routing set;
   `gold_oil_ratio` (level) vs `gold_oil_ratio_chg4w` (routing) differ by a diff.
8. **`equity_bond_corr_13w` is only in `routing_triggers.parquet`.** The model panel needs it joined on
   (`join(...).dropna(subset=["equity_bond_corr_13w"])`), which drops early rows: model panel 1107→**1060**
   rows once enriched (dev 939 + test 121). Fold-1 train 360→**313** (254 normals + 59 pos).
9. **MVG docstring "~250-360 normals" is stale**: actual fold-1 normals = **254** (post-enrichment).
10. **Internal naming**: "Oro" (=gold) in `SUBSCORE_ORO`/`ROUTING_DOMAINS["Oro"]`/`subscore_oro`; the
    allocation label is `GOLD`. Routing-correlation strings use Italian "intra-dominio"/"cross-dominio".
11. **`time_split()` deprecated** but present; it still writes `data/processed/splits/`. Use `walkforward_split`.
12. **Routing grid Calmar values are all negative** (see §5).

---

## 2. ARTIFACTS MAP + tested reload/inference

### `outputs/models/`
| File | What | Produced by |
|---|---|---|
| `scaler_final.pkl` | `FoldScaler` (5 per-fold scalers + final dev scaler), `feature_cols_`=56 | nb 03 |
| `mvg_final.pkl` | `MVGAnomalyDetector`, `threshold_`=81.7413 | nb 03 |
| `svm_final.pkl` | `OneClassSVMDetector` nu=0.10 γ=0.001, `threshold_`=0.5376 | nb 04 |
| `autoencoder_final.keras` (+`.meta.json`) | `AutoencoderDetector` 56-24-12-6-…, `threshold_`=1.44642 | nb 04 |
| `if_final.pkl` | `IsolationForestDetector` contamination=0.10, n_est=200, `threshold_`=0.4708 | nb 04 |
| `ensemble_final.pkl` | META dict: mode=`soft_median`, tau=**0.9769**, model_order, `reference_scores_dict` (4×744 dev-normal scores), walkforward_results | nb 05 |

### `outputs/tables/`
| File | What | Produced by |
|---|---|---|
| `mvg_walkforward.csv` | MVG per-fold + weighted_mean + test rows | nb 03 |
| `models_comparison.csv` | 4 models × (wCV + test) metrics + hyperparams | nb 04 |
| `ensemble_comparison.csv` | 4 singles + 3 ensembles × (wCV + test) | nb 05 |
| `backtest_metrics_all.csv` | 22 metrics × 6 strategies | nb 07 |
| `equity_curves.csv` | 121-week growth-of-$1, 6 strategies | nb 07 |
| `threshold_grid.csv` | 25-row (usd×oro) duration-weighted median Calmar | nb 07 |

Other processed parquets: `data/processed/{features_stationary,features_stationary_clean,spreads,
spreads_clean,routing_triggers}.parquet`; `walkforward/` (dev/test/fold_i_{train,val,train_normals} ×
models+routing); `subscores/{test_holdout,folds/fold_i_val}.parquet`; `allocations/test_holdout.parquet`;
`splits/` (deprecated).

### Tested reload + end-to-end inference (verified — flags 11/121 on the test holdout)
```python
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import numpy as np, pandas as pd
from src.splits import walkforward_split
from src.preprocessing import FoldScaler
from src.models import (MVGAnomalyDetector, OneClassSVMDetector,
                        AutoencoderDetector, IsolationForestDetector)
from src.ensemble import EnsembleDetector, score_to_percentile

# 1. Build the 57-col model matrix (56 features + Y) for any new period.
#    Pipeline must already have produced data/processed/*; otherwise run
#    `python -m src.features` (regenerates clean parquets + routing_triggers).
wf  = walkforward_split(save=False, verbose=False)
ebc = pd.read_parquet("data/processed/routing_triggers.parquet")["equity_bond_corr_13w"]
enrich = lambda d: d.join(ebc, how="left").dropna(subset=["equity_bond_corr_13w"])
new_df = enrich(wf["models"]["test_holdout"])          # <-- any DataFrame w/ the 56 feature cols (+Y)

# 2. Reload artifacts.
scaler = FoldScaler.load("outputs/models/scaler_final.pkl")
models = {"mvg": MVGAnomalyDetector.load("outputs/models/mvg_final.pkl"),
          "svm": OneClassSVMDetector.load("outputs/models/svm_final.pkl"),
          "ae":  AutoencoderDetector.load("outputs/models/autoencoder_final.keras"),
          "if":  IsolationForestDetector.load("outputs/models/if_final.pkl")}
meta = EnsembleDetector.load_meta("outputs/models/ensemble_final.pkl")
ORD, tau, ref = meta["model_order"], meta["tau"], meta["reference_scores_dict"]

# 3. Inference: dev-fit scaler -> per-model score -> percentile vs dev normals -> soft median -> tau.
X   = scaler.transform_holdout(new_df)                 # drops Y, 56 cols
pct = np.array([score_to_percentile(ref[n], models[n].score_samples(X)) for n in ORD])
ens_score = np.median(pct, axis=0)
ens_sig   = (ens_score >= tau).astype(int)             # 1 = risk-off
```
For per-model binary flags instead of the ensemble: `models[n].predict(X)` (uses each `threshold_`).

---

## 3. KEY RESULTS

### Per-model & ensemble (weighted-CV = n_pos-weighted across 5 folds; test = sealed 2019-2021)
| Model | F1 wCV | AUC-PR wCV | AUC-ROC wCV | F1 test | AUC-PR test | AUC-ROC test | Fβ2 test | hyperparams |
|---|---|---|---|---|---|---|---|---|
| MVG | 0.572 | 0.734 | 0.808 | 0.635 | 0.764 | **0.883** | 0.758 | ledoit-wolf |
| SVM | **0.649** | 0.729 | 0.814 | 0.605 | 0.738 | 0.853 | 0.580 | nu=0.10, γ=0.001 |
| AE | 0.617 | 0.718 | 0.800 | 0.588 | **0.778** | 0.876 | 0.485 | 56-24-12-6-12-24-56, dropout 0.15 |
| IF | 0.576 | 0.652 | 0.752 | 0.571 | 0.671 | 0.806 | 0.541 | contamination=0.10, n_est=200 |
| ENS_hard | 0.593 | 0.531 | 0.740 | 0.615 | 0.663 | 0.886 | 0.556 | ≥3/4 vote |
| ENS_soft_mean | 0.655 | 0.670 | 0.784 | 0.634 | 0.742 | 0.858 | 0.591 | τ tuned |
| **ENS_soft_median** | 0.650 | 0.606 | 0.776 | **0.647** | 0.753 | 0.860 | 0.534 | **τ=0.977 (DEPLOYED)** |

Mean pairwise binary-error correlation (folds 1+2) = **0.507** (notebook value; AE refit drifts 0.47-0.51).

### MVG per fold (illustrates fold imbalance — all models share these n_pos)
| fold | val window | n_train | n_val | n_pos | crisis | F1 |
|---|---|---|---|---|---|---|
| 1 | 2007-01→2009-12 | 313 | 153 | **53** | GFC 2008 | 0.596 |
| 2 | 2010-02→2012-12 | 470 | 152 | **64** | Euro 2011 | 0.618 |
| 3 | 2013-01→2014-12 | 626 | 101 | 2 | Taper 2013 | 0.077 |
| 4 | 2015-02→2016-12 | 731 | 100 | 10 | China-Oil | 0.294 |
| 5 | 2017-01→2018-12 | 835 | 100 | 6 | Q4-2018 | 0.500 |

### Routing thresholds & backtest (test holdout, EWS = our strategy)
Chosen routing thresholds: **τ_usd = 1.5, τ_oro = 0.5** (duration-weighted median Calmar; see §5 caveat).
TC bps one-way: equity 5 / bond 5 / gold 8 / cash 2 / mbs 20.

| Strategy | CAGR | Sharpe | Sortino | Calmar | MaxDD | COVID DD | Recovery wk | final ×$1 |
|---|---|---|---|---|---|---|---|---|
| **EWS** | **0.310** | **1.45** | **2.22** | **1.39** | −0.222 | −0.214 | **22** | **1.874** |
| MSCI BH | 0.168 | 0.98 | 1.31 | 0.61 | −0.277 | −0.272 | 35 | 1.435 |
| 60/40 | 0.137 | 1.10 | 1.47 | 0.68 | −0.203 | −0.201 | 33 | 1.349 |
| Butterfly | 0.127 | 1.32 | 1.80 | 1.06 | −0.120 | −0.120 | 10 | 1.321 |
| Permanent | 0.107 | 1.34 | 1.81 | 1.07 | −0.100 | −0.100 | 11 | 1.268 |
| Risk Parity | 0.124 | 1.31 | 1.80 | 1.11 | −0.111 | −0.111 | 10 | 1.313 |

EWS operational: Annual_Turnover=1.29 switches/yr, Cumulative_TC=0.46%, Hit_Ratio_RiskOff=0.478.
Honest nuance: always-defensive blends have smaller **absolute** DD (never fully invested); EWS wins all
**risk-adjusted** metrics + CAGR. EWS flags **11/121** test weeks risk-off (high-precision / low-recall τ).

---

## 4. RUNTIME NOTES (measured, single CPU container)

| Step | Wall-clock | Notes |
|---|---|---|
| `load_dataset()` | <1 s | Excel read |
| `python -m src.features` (full regen) | ~20-40 s | ADF tests + heatmaps dominate |
| `python -m src.splits` | ~2-5 s | parquet slicing |
| **AE single fit** (fold 1, 254 normals) | **~14 s** | grows with fold size (fold 5 ≈ 646 normals) |
| **Full leakage-free refit of all 4 models × 5 folds** | **~80 s** | dominated by 5 AE fits; MVG/SVM/IF single-fits are ~0.1-1 s each |
| SVM grid (nb 04): 16 combos × 5 folds | ~1-2 min | 80 cheap SVM fits |
| AE walk-forward tuning (nb 04/05): 5 AE fits | ~1-2 min | AE is the cost driver everywhere |
| nb 04 / nb 05 end-to-end | ~3-6 min each | AE + ensemble per-fold refit |
| nb 07 (routing+backtest) | ~2-4 min | re-derives per-fold ensemble signal (refits 4×5) |
| **`notebooks/EWS_GSoM_PoliMI.ipynb` end-to-end** | **~30 s** | loads saved artifacts + displays figures; **no retraining** |

**Caching that exists:** none persisted. The only in-memory cache is the `fold_scores` dict accepted by
`EnsembleDetector.fit_threshold_walkforward(fold_scores=…)` — built once per notebook run and reused across
the 3 ensemble modes. **The single biggest win for the next phase is to persist this fold-score cache**
(see §7) so τ/threshold sweeps don't pay the ~80 s refit each time. AE nondeterminism means a persisted
cache also stabilizes the error-correlation / ensemble numbers across runs.

---

## 5. KNOWN ISSUES / TECH DEBT

1. **Routing threshold grid is all-negative & weakly identified.** Every `calmar_wmedian` in
   `threshold_grid.csv` is negative (range −0.242 … −0.073) because per-fold-val backtests over thin crisis
   windows lose on a duration-weighted-median Calmar basis. The "best" (1.5, 0.5) is merely least-negative;
   at `oro=0.5` several `usd` values tie at −0.0933 (only `usd=1.5` reaches −0.0728), so `oro` drives the
   choice and `usd` is barely identified. Next phase should try alternative objectives (Sortino, the
   cost-weighted `C=0.10·n_FN+0.005·n_FP`, or Calmar on the concatenated fold path).
2. **Sparse labels post-2013** → folds 3-5 have 2/10/6 positives. CV is essentially carried by folds 1-2
   (117/135 = 87% of positives). `weighted_median`/`weighted_mean` by `n_pos` mitigate but cannot remove the
   noise; fold-3 F1 is near-degenerate (0.077 on n=2).
3. **AE nondeterminism**: even with `seed=42`, oneDNN op ordering shifts AE scores run-to-run; error
   correlation lands 0.47-0.51 (committed = 0.507). Any recomputed AE-dependent figure will differ slightly
   from the committed artifacts. Set `TF_ENABLE_ONEDNN_OPTS=0` + `TF_DETERMINISTIC_OPS=1` to reduce drift.
4. **Feature leakage of meaning, not data**: `vrp` & `jpy_strength` appear in **both** the 56 model
   features and the routing triggers; clustered feature-selection must treat them as one concept.
5. **Proxies baked in** (must stay consistent if data extended): MSCI World→`MXUS` (and an equal-weight
   `MXUS,MXEU,MXJP` composite inside `usa_world_relative`); Global Agg→`LUACTRUU`; credit spreads are
   log-ratios of total-return indices, **not** OAS in bps; risk-free/cash = `USGG3M/52`.
6. **MVG ε cross-fit transfer** (per-fold ε applied to dev-fit model) is a χ²-approximation, not exact.
7. **`EnsembleDetector` has no full (de)serialization** — only META; you must reload the 4 base models.
8. **LibreOffice cannot render in this sandbox** ("source file could not be loaded" on any file) → the
   deck QA used `render_preview.py` (rasterizes pptx geometry via matplotlib), not `soffice`.
9. **`COVID_Crash_DD` window is fixed** (`2020-02-15…04-15`) inside `backtest.py`; recovery-week logic
   assumes the COVID structure — both are test-holdout-specific and would need generalizing for new data.
10. `git` working tree currently has untracked helper scripts at repo root (`gen_assets.py`,
    `build_deck_v2.py`, `render_preview.py`) — these were committed in `505b866`; nothing else is pending.

---

## 6. CONVENTIONS

- **Branch**: develop & push only to `claude/intelligent-einstein-DIHCx`. **Never push to `main`.**
  `git push -u origin claude/intelligent-einstein-DIHCx` (retry w/ backoff on network errors).
- **Commits**: one per deliverable, message prefixed (`Prompt N: …` historically). A
  `https://claude.ai/code/session_…` footer is auto-appended. Do **not** put the model id in commits.
  Do **not** open PRs unless asked.
- **Notebooks are built programmatically**: a one-off `build_nb_*.py` writes the `.ipynb` via `nbformat`,
  then `jupyter nbconvert --to notebook --execute --inplace <nb> --ExecutePreprocessor.timeout=…`; the
  builder is deleted after a clean run. Stage notebooks live in `notebooks/0X_*.ipynb`.
- **Output dirs**: processed data → `data/processed/{,splits,walkforward,subscores,allocations}`;
  models → `outputs/models/`; tables (CSV) → `outputs/tables/`; figures → `reports/figures/` (stage
  notebooks, dpi≈120-130) and `outputs/deck_assets/` (deck, dpi=200); deliverables → `outputs/`.
- **Seeds**: TF `tf.random.set_seed(42)` + `np.random.seed(42)` inside AE; IsolationForest
  `random_state=42`. MVG/SVM deterministic. No global seed is set elsewhere.
- **`src/__init__.py` exists** so `from src.X import Y` resolves under a notebook kernel.

---

## 7. NEXT-PHASE CONTEXT (documentation only — do NOT design here)

Three planned tasks. Everything they need from the current code:

### (a) 3-stage feature-selection pipeline
- Operates on the **56 model features** (55 from `features_stationary_clean ⨝ spreads_clean`, + joined
  `equity_bond_corr_13w`). Build the matrix exactly as in §2 (`enrich`), then drop `Y`.
- A **univariate AP-ranking prototype** will be provided separately (stage 1). Stage 2 = clustered
  permutation importance; stage 3 = nested subset curve with **per-subset τ re-tuning**.
- Per-subset τ re-tuning hooks: for each candidate feature subset, refit the 4 detectors per fold (use
  `clone_unfit` + `FoldScaler` restricted to the subset — note `FoldScaler` locks `feature_cols_` on first
  call, so **construct a fresh `FoldScaler` per subset** or pass a pre-sliced df), recompute fold scores,
  then call `EnsembleDetector(mode,…).fit_threshold_walkforward(folds, fold_scores=cache)` which returns
  the per-fold df and sets `threshold_` (= n_pos-weighted median of per-fold `_tune_threshold` ε).
- Watch the **meaning-duplication** (§5.4) and the **fold-3 degeneracy** (n=2) when ranking.

### (b) Sensitivity module
- **Persistent fold-score cache** — the key enabler. Natural location: `data/processed/fold_scores/`
  (one file per `(model, fold_id[, hyperparam])`, e.g. `npz`/parquet of `ref` and `val` raw scores), or
  `outputs/cache/`. Structure mirrors the in-memory dict consumed today:
  `fold_scores[fid] = {'ref':{name:array}, 'val':{name:array}, 'thr':{name:float}}`.
  With this cache, τ-curves and threshold grids cost ~0 refit instead of ~80 s.
- **τ curves per fold**: reuse cached `ref`/`val` scores → `score_to_percentile` → aggregate (mean/median)
  → sweep the threshold and call `compute_metrics` at each; `_tune_threshold` already does the
  argmax-F1 version (returns one ε + metrics) and shows the exact percentile-grid convention (80-99).
- **Hyperparameter × τ surfaces**: SVM `NU_GRID×GAMMA_GRID`, IF `CONTAMINATION_GRID`, AE
  (dropout/lr/arch) — each combo needs its own cached fold scores (AE = the cost; cache it).
- **Routing-threshold grid with alternative metrics**: `optimize_routing_thresholds` currently hard-codes
  duration-weighted median Calmar; generalize the inner scalarization to Sortino / cost-weighted
  `C=0.10·n_FN+0.005·n_FP` / path-concatenated Calmar (see §5.1).

### (c) Extend dataset to 2022-2025 (free sources) + reverse-engineer the `Y` labelling rule
- **How new raw data enters**: `data_loader.load_raw` reads `data/raw/04_May_Zenti_exercises.xlsx`,
  sheet `Markets`, with date col **`Data`** and the **43 columns incl `Y`** (42 tickers + Y). New rows must
  match the same ticker columns. Then the whole pipeline is deterministic from raw:
  `python -m src.features` (regenerates `features_stationary*`, `spreads*`, `routing_triggers`) →
  `python -m src.splits` (regenerates `walkforward/`). Models then re-fit via the stage notebooks.
- **The `Y` labelling rule is unknown** (course-provided) and must be reverse-engineered before 2022-2025
  rows can be labelled. `Y` is ~21.3% prevalent, clustered in 2000-02 / 2008-09 / 2010-12 / 2020, with
  **zero** positives in 2013/2017/2019 — suggesting a volatility/drawdown-threshold rule on equities (likely
  VIX- and/or MXUS-drawdown-based). Validate any reconstructed rule against the existing `Y` before
  applying to new data.
- Proxy/ticker availability from free sources will differ (no Bloomberg credit TR indices, no `MXWO`); keep
  the §5.5 proxy substitutions or document new ones. New raw tickers may break `TRANSFORM_MAP` /
  `build_spreads` / `build_routing_triggers` column lookups — those reference exact ticker names.

---

## 8. Deck / presentation / non-code (compressed)
`outputs/EWS_pitch_deck_v2.pptx` (15 slides, dark "trading-terminal" design, hero = slide-9 trigger-map
network) + `outputs/presentation_speech.md` (~1,450 words) + `notebooks/EWS_GSoM_PoliMI.ipynb` (14
sections, ~30 s). Built by `gen_assets.py` (dpi=200 matplotlib visuals) + `build_deck_v2.py`; QA via
`render_preview.py` (LibreOffice can't render here). Affiliation **PoliMI GSoM only** (no Bocconi), year
**2026**, team Ercelli · Galli · Impenati · Mazzini. README has a Colab badge.
