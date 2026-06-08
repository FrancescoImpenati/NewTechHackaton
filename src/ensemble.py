"""Ensemble of the four novelty detectors (MVG, SVM, AE, IF).

Raw anomaly scores live on incomparable scales, so they are mapped to empirical
percentiles against a reference distribution (the train normals of a fold, or
the development normals for the final test). Three aggregation modes:

* ``hard``        — majority vote (>=3 of 4) of each model's tuned binary flag.
* ``soft_mean``   — mean of the 4 percentiles; threshold tau tuned walk-forward.
* ``soft_median`` — median of the 4 percentiles (robust to one model blowing up).

Walk-forward tuning re-fits the per-fold models leakage-free (via ``clone_unfit``
or a precomputed ``fold_scores`` cache); the final test uses the development-fit
models passed in ``models_dict``.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import (
    AutoencoderDetector,
    IsolationForestDetector,
    MVGAnomalyDetector,
    OneClassSVMDetector,
    _tune_threshold,
    _weighted_summary,
    compute_metrics,
    weighted_median,
)

MODEL_ORDER = ["mvg", "svm", "ae", "if"]


def score_to_percentile(s_reference, s_new) -> np.ndarray:
    """Empirical percentile of ``s_new`` against ``s_reference`` in [0, 1].

    ``searchsorted`` on the sorted reference / len(reference): values above the
    reference max -> 1.0, at/below the min -> 0.0."""
    ref_sorted = np.sort(np.asarray(s_reference, dtype=float))
    return np.searchsorted(ref_sorted, np.asarray(s_new, dtype=float)) / len(ref_sorted)


def clone_unfit(model):
    """Fresh, unfitted detector carrying the same hyperparameters."""
    if isinstance(model, MVGAnomalyDetector):
        return MVGAnomalyDetector(shrinkage=model.shrinkage)
    if isinstance(model, OneClassSVMDetector):
        return OneClassSVMDetector(nu=model.nu, gamma=model.gamma)
    if isinstance(model, IsolationForestDetector):
        return IsolationForestDetector(contamination=model.contamination,
                                       n_estimators=model.n_estimators,
                                       random_state=model.random_state)
    if isinstance(model, AutoencoderDetector):
        return AutoencoderDetector(dropout=model.dropout, lr=model.lr,
                                   max_epochs=model.max_epochs, batch_size=model.batch_size,
                                   patience=model.patience, seed=model.seed)
    raise TypeError(f"clone_unfit: unknown model type {type(model)}")


class EnsembleDetector:
    def __init__(self, mode: str, models_dict: dict, scalers_dict: dict):
        assert mode in ("hard", "soft_mean", "soft_median")
        self.mode = mode
        self.models = models_dict
        self.scalers = scalers_dict
        self.threshold_ = None
        self.walkforward_results_ = None
        self.model_order = [m for m in MODEL_ORDER if m in models_dict]

    # --- scoring helpers ---------------------------------------------------
    def _raw_scores(self, X, fold_id=None) -> dict[str, np.ndarray]:
        out = {}
        for name in self.model_order:
            scaler = self.scalers[name]
            Xs = scaler.transform(X, fold_id) if fold_id is not None else scaler.transform_holdout(X)
            out[name] = self.models[name].score_samples(Xs)
        return out

    def _aggregate(self, pct_dict: dict[str, np.ndarray]) -> np.ndarray:
        arr = np.array([pct_dict[name] for name in self.model_order])
        return arr.mean(axis=0) if self.mode == "soft_mean" else np.median(arr, axis=0)

    def _votes(self, raw: dict[str, np.ndarray]) -> np.ndarray:
        return np.sum([(raw[name] >= self.models[name].threshold_).astype(int)
                       for name in self.model_order], axis=0)

    def decision_scores(self, X, reference_scores_dict, fold_id=None) -> np.ndarray:
        """Continuous decision score for ANY mode (hard -> vote fraction)."""
        raw = self._raw_scores(X, fold_id)
        if self.mode == "hard":
            return self._votes(raw) / len(self.model_order)
        pct = {n: score_to_percentile(reference_scores_dict[n], raw[n]) for n in self.model_order}
        return self._aggregate(pct)

    def score_samples(self, X, reference_scores_dict, fold_id=None) -> np.ndarray:
        if self.mode == "hard":
            return (self._votes(self._raw_scores(X, fold_id)) >= 3).astype(int)
        return self.decision_scores(X, reference_scores_dict, fold_id)

    def predict(self, X, reference_scores_dict, fold_id=None) -> np.ndarray:
        if self.mode == "hard":
            return self.score_samples(X, reference_scores_dict, fold_id)
        return (self.score_samples(X, reference_scores_dict, fold_id) >= self.threshold_).astype(int)

    # --- walk-forward tuning ----------------------------------------------
    def fit_threshold_walkforward(self, folds_data, fold_scores=None,
                                  target_col="Y", percentiles=range(80, 100)):
        """Tune tau (soft modes) or just report CV metrics (hard).

        ``fold_scores`` (optional cache) maps fold_id ->
        {'ref': {name: train-normal raw scores}, 'val': {name: val raw scores},
         'thr': {name: per-model threshold}}; if absent the per-fold models are
        re-fit leakage-free via clone_unfit."""
        rows = []
        for fold in folds_data:
            fid, train, val = fold["fold_id"], fold["train"], fold["val"]
            y_val = val[target_col].to_numpy()

            if fold_scores is not None and fid in fold_scores:
                ref, valsc = fold_scores[fid]["ref"], fold_scores[fid]["val"]
                thr = fold_scores[fid]["thr"]
            else:
                tn = train[train[target_col] == 0]
                ref, valsc, thr = {}, {}, {}
                for name in self.model_order:
                    sc = self.scalers[name]
                    fm = clone_unfit(self.models[name]).fit(sc.transform(tn, fid))
                    ref[name] = fm.score_samples(sc.transform(tn, fid))
                    valsc[name] = fm.score_samples(sc.transform(val, fid))
                    thr[name] = self.models[name].threshold_

            base = {"fold_id": fid, "n_train": int(len(train)), "n_val": int(len(val)),
                    "n_pos_val": int(y_val.sum())}
            if self.mode == "hard":
                votes = np.sum([(valsc[n] >= thr[n]).astype(int) for n in self.model_order], axis=0)
                pred = (votes >= 3).astype(int)
                metrics = compute_metrics(y_val, pred, votes / len(self.model_order))
                rows.append({**base, "epsilon_optimal": np.nan, **metrics})
            else:
                pct_ref = {n: score_to_percentile(ref[n], ref[n]) for n in self.model_order}
                pct_val = {n: score_to_percentile(ref[n], valsc[n]) for n in self.model_order}
                eps, metrics = _tune_threshold(self._aggregate(pct_ref),
                                               self._aggregate(pct_val), y_val, percentiles)
                rows.append({**base, "epsilon_optimal": eps, **metrics})

        df = pd.DataFrame(rows)
        if self.mode != "hard":
            self.threshold_ = weighted_median(df["epsilon_optimal"].to_numpy(),
                                              df["n_pos_val"].to_numpy())
        self.walkforward_results_ = df
        return df

    def weighted_summary(self) -> dict[str, float]:
        return _weighted_summary(self.walkforward_results_, self.threshold_)

    def save(self, path, reference_scores_dict) -> None:
        """Persist lightweight metadata (NOT the models, which are saved
        separately and include a non-picklable Keras AE)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "mode": self.mode,
            "tau": self.threshold_,
            "model_order": self.model_order,
            "reference_scores_dict": {k: np.asarray(v) for k, v in reference_scores_dict.items()},
            "walkforward_results": self.walkforward_results_,
        }
        with open(path, "wb") as f:
            pickle.dump(meta, f)

    @staticmethod
    def load_meta(path) -> dict:
        with open(path, "rb") as f:
            return pickle.load(f)
