"""Anomaly-detection models for risk-off detection.

Models are fit on *normal* (Y=0) weeks only and emit an anomaly score; a tuned
threshold turns the score into a 0/1 risk-off flag (higher score = more
anomalous = risk-off). ``MVGAnomalyDetector`` is the multivariate-Gaussian
baseline with mandatory Ledoit-Wolf shrinkage of the covariance.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# --- shared helpers (reused by later models / ensemble) --------------------
def compute_metrics(y_true, y_pred, scores) -> dict[str, float]:
    """The six metrics reported everywhere: F1, Precision, Recall, AUC-ROC,
    AUC-PR, F-beta(beta=2)."""
    y_true = np.asarray(y_true)
    both_classes = len(np.unique(y_true)) > 1
    return {
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "AUC_ROC": roc_auc_score(y_true, scores) if both_classes else float("nan"),
        "AUC_PR": average_precision_score(y_true, scores) if both_classes else float("nan"),
        "F_beta_2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
    }


def weighted_median(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0:
        return float(np.median(values))
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, weights.sum() / 2.0))
    return float(v[min(idx, len(v) - 1)])


def weighted_mean(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values)
    if not mask.any() or weights[mask].sum() == 0:
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


class MVGAnomalyDetector:
    """Multivariate-Gaussian novelty detector with Ledoit-Wolf shrinkage.

    Score = squared Mahalanobis distance to the mean of the normal class. With
    56 features and only ~250-360 normals in the early folds the empirical
    covariance is unstable/near-singular, so Ledoit-Wolf shrinkage is used by
    default to obtain a well-conditioned, invertible Sigma.
    """

    def __init__(self, shrinkage: str = "ledoit-wolf"):
        self.shrinkage = shrinkage
        self.mu_ = None
        self.covariance_ = None
        self.precision_ = None  # Sigma^{-1}
        self.threshold_ = None
        self.walkforward_results_ = None
        self.fold_details_ = None

    def fit(self, X_train_normals) -> "MVGAnomalyDetector":
        """Estimate mu (sample mean) and Sigma (Ledoit-Wolf) on normals only."""
        X = np.asarray(X_train_normals, dtype=float)
        self.mu_ = X.mean(axis=0)
        if self.shrinkage == "ledoit-wolf":
            lw = LedoitWolf().fit(X)
            self.covariance_ = lw.covariance_
            self.precision_ = lw.precision_
        else:
            self.covariance_ = np.cov(X, rowvar=False)
            self.precision_ = np.linalg.pinv(self.covariance_)
        return self

    def score_samples(self, X) -> np.ndarray:
        """Squared Mahalanobis distance per row."""
        diff = np.asarray(X, dtype=float) - self.mu_
        return np.einsum("ni,ij,nj->n", diff, self.precision_, diff)

    def predict(self, X) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("threshold_ not set; call fit_threshold_walkforward first")
        return (self.score_samples(X) >= self.threshold_).astype(int)

    def fit_threshold_walkforward(
        self, folds_data: list[dict], scaler, percentiles=range(80, 100)
    ) -> pd.DataFrame:
        """Tune the threshold epsilon across walk-forward folds.

        For each fold: fit an MVG on the (scaled) train normals, build a
        threshold grid from the 80th-99th percentiles of the train-normals
        Mahalanobis distribution, pick the threshold maximizing val F1, and
        record F1/Precision/Recall/AUC-ROC/AUC-PR/F-beta2 at that threshold.
        The final epsilon is the n_pos-weighted **median** of the per-fold
        optima (folds with few positives weigh less). Returns the per-fold
        results table and stores it on the instance.
        """
        target = scaler.target_col
        grid_pcts = list(percentiles)
        rows, details = [], []
        for fold in folds_data:
            fid = fold["fold_id"]
            train, val = fold["train"], fold["val"]
            train_normals = train[train[target] == 0]

            Xtr = scaler.transform(train_normals, fid)
            det = type(self)(shrinkage=self.shrinkage).fit(Xtr)
            s_train = det.score_samples(Xtr)

            Xval = scaler.transform(val, fid)
            s_val = det.score_samples(Xval)
            y_val = val[target].to_numpy()

            grid = np.percentile(s_train, grid_pcts)
            f1s = np.array(
                [f1_score(y_val, (s_val >= t).astype(int), zero_division=0) for t in grid]
            )
            best = int(np.where(f1s == f1s.max())[0][-1])  # highest thr among ties
            eps = float(grid[best])
            metrics = compute_metrics(y_val, (s_val >= eps).astype(int), s_val)

            rows.append(
                {
                    "fold_id": fid,
                    "n_train": int(len(train)),
                    "n_val": int(len(val)),
                    "n_pos_val": int(y_val.sum()),
                    "epsilon_optimal": eps,
                    **metrics,
                }
            )
            details.append(
                {
                    "fold_id": fid,
                    "scores_train_normals": s_train,
                    "scores_val": s_val,
                    "y_val": y_val,
                    "epsilon": eps,
                }
            )

        results = pd.DataFrame(rows)
        weights = results["n_pos_val"].to_numpy()
        self.threshold_ = weighted_median(results["epsilon_optimal"].to_numpy(), weights)
        self.walkforward_results_ = results
        self.fold_details_ = details
        return results

    def weighted_summary(self) -> dict[str, float]:
        """n_pos-weighted mean of each metric across folds (the 'weighted_mean'
        row of the report). Epsilon is the weighted median (= self.threshold_)."""
        r = self.walkforward_results_
        w = r["n_pos_val"].to_numpy()
        out = {m: weighted_mean(r[m].to_numpy(), w) for m in
               ["F1", "Precision", "Recall", "AUC_ROC", "AUC_PR", "F_beta_2"]}
        out["epsilon_optimal"] = self.threshold_
        return out

    def save(self, path: str | Path, drop_details: bool = True) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        details = self.fold_details_
        if drop_details:
            self.fold_details_ = None  # keep the pickle small
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self.fold_details_ = details

    @staticmethod
    def load(path: str | Path) -> "MVGAnomalyDetector":
        with open(path, "rb") as f:
            return pickle.load(f)
