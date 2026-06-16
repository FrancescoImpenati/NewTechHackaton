"""Anomaly-detection models for risk-off detection.

Models are fit on *normal* (Y=0) weeks only and emit an anomaly score; a tuned
threshold turns the score into a 0/1 risk-off flag (higher score = more
anomalous = risk-off). ``MVGAnomalyDetector`` is the multivariate-Gaussian
baseline with mandatory Ledoit-Wolf shrinkage of the covariance.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

# Reduce AE run-to-run drift (handoff §5.3). Must be set before tensorflow is
# imported anywhere; the AE imports TF lazily, so module-import time is early
# enough. setdefault keeps any explicit user override.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
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


# --- shared walk-forward helpers (SVM / AE / IF) ---------------------------
def _tune_threshold(scores_train_normals, scores_val, y_val, percentiles=range(80, 100)):
    """Pick the threshold (from the 80-99th pct of train-normals scores) that
    maximizes val F1; ties resolved toward the higher threshold. Returns
    (epsilon, metrics dict)."""
    grid = np.percentile(scores_train_normals, list(percentiles))
    f1s = np.array([f1_score(y_val, (scores_val >= t).astype(int), zero_division=0) for t in grid])
    best = int(np.where(f1s == f1s.max())[0][-1])
    eps = float(grid[best])
    return eps, compute_metrics(y_val, (scores_val >= eps).astype(int), scores_val)


def _weighted_summary(results: pd.DataFrame, threshold: float) -> dict[str, float]:
    """n_pos-weighted mean of each metric across folds (+ epsilon = threshold)."""
    w = results["n_pos_val"].to_numpy()
    out = {m: weighted_mean(results[m].to_numpy(), w) for m in
           ["F1", "Precision", "Recall", "AUC_ROC", "AUC_PR", "F_beta_2"]}
    out["epsilon_optimal"] = threshold
    return out


class OneClassSVMDetector:
    """One-Class SVM (RBF) novelty detector. score = -decision_function."""

    NU_GRID = [0.05, 0.10, 0.15, 0.22]
    GAMMA_GRID = ["scale", "auto", 0.01, 0.001]

    def __init__(self, nu: float = 0.1, gamma="scale"):
        self.nu = nu
        self.gamma = gamma
        self.model_ = None
        self.threshold_ = None
        self.best_params_ = None
        self.walkforward_results_ = None
        self.fold_details_ = None

    def fit(self, X_train_normals) -> "OneClassSVMDetector":
        self.model_ = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma).fit(
            np.asarray(X_train_normals, dtype=float))
        return self

    def score_samples(self, X) -> np.ndarray:
        return -self.model_.decision_function(np.asarray(X, dtype=float))

    def predict(self, X) -> np.ndarray:
        return (self.score_samples(X) >= self.threshold_).astype(int)

    def fit_hyperparams_walkforward(self, folds_data, scaler, percentiles=range(80, 100)):
        """Grid-search nu x gamma (4x4). For each combo, tune a per-fold F1-optimal
        threshold; select the combo with the highest n_pos-weighted **median** F1.
        Final threshold = n_pos-weighted median of the winning combo's per-fold
        thresholds."""
        target = scaler.target_col
        best = None
        for nu in self.NU_GRID:
            for gamma in self.GAMMA_GRID:
                rows, details = [], []
                for fold in folds_data:
                    fid, train, val = fold["fold_id"], fold["train"], fold["val"]
                    tn = train[train[target] == 0]
                    Xtr = scaler.transform(tn, fid)
                    m = OneClassSVMDetector(nu=nu, gamma=gamma).fit(Xtr)
                    s_tr = m.score_samples(Xtr)
                    s_val = m.score_samples(scaler.transform(val, fid))
                    y_val = val[target].to_numpy()
                    eps, metrics = _tune_threshold(s_tr, s_val, y_val, percentiles)
                    rows.append({"fold_id": fid, "n_train": int(len(train)),
                                 "n_val": int(len(val)), "n_pos_val": int(y_val.sum()),
                                 "epsilon_optimal": eps, **metrics})
                    details.append({"fold_id": fid, "scores_val": s_val, "y_val": y_val})
                df = pd.DataFrame(rows)
                wf1 = weighted_median(df["F1"].to_numpy(), df["n_pos_val"].to_numpy())
                if best is None or wf1 > best["wf1"]:
                    best = {"wf1": wf1, "nu": nu, "gamma": gamma, "results": df, "details": details}
        self.nu, self.gamma = best["nu"], best["gamma"]
        self.best_params_ = {"nu": best["nu"], "gamma": best["gamma"]}
        self.walkforward_results_ = best["results"]
        self.fold_details_ = best["details"]
        self.threshold_ = weighted_median(
            best["results"]["epsilon_optimal"].to_numpy(),
            best["results"]["n_pos_val"].to_numpy())
        return best["results"]

    def weighted_summary(self) -> dict[str, float]:
        return _weighted_summary(self.walkforward_results_, self.threshold_)

    def save(self, path, drop_details: bool = True) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = self.fold_details_
        if drop_details:
            self.fold_details_ = None
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self.fold_details_ = d

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


class IsolationForestDetector:
    """Isolation Forest novelty detector. score = -IsolationForest.score_samples.

    score_samples is contamination-invariant, so contamination is used as the
    decision-boundary (operating point): per fold we evaluate F1 at the
    contamination-induced threshold (-offset_) and pick the contamination with
    the highest n_pos-weighted median val F1. Final threshold = weighted median
    of the per-fold boundaries."""

    CONTAMINATION_GRID = [0.05, 0.10, 0.15, 0.22]

    def __init__(self, contamination: float = 0.1, n_estimators: int = 200,
                 random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_ = None
        self.threshold_ = None
        self.best_params_ = None
        self.walkforward_results_ = None
        self.fold_details_ = None

    def fit(self, X_train_normals) -> "IsolationForestDetector":
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators, contamination=self.contamination,
            max_samples="auto", random_state=self.random_state
        ).fit(np.asarray(X_train_normals, dtype=float))
        return self

    def score_samples(self, X) -> np.ndarray:
        return -self.model_.score_samples(np.asarray(X, dtype=float))

    def predict(self, X) -> np.ndarray:
        return (self.score_samples(X) >= self.threshold_).astype(int)

    def fit_hyperparams_walkforward(self, folds_data, scaler):
        target = scaler.target_col
        best = None
        for c in self.CONTAMINATION_GRID:
            rows, details = [], []
            for fold in folds_data:
                fid, train, val = fold["fold_id"], fold["train"], fold["val"]
                tn = train[train[target] == 0]
                m = IsolationForestDetector(contamination=c,
                                            n_estimators=self.n_estimators,
                                            random_state=self.random_state).fit(scaler.transform(tn, fid))
                s_val = m.score_samples(scaler.transform(val, fid))
                eps = float(-m.model_.offset_)  # contamination-induced boundary
                y_val = val[target].to_numpy()
                metrics = compute_metrics(y_val, (s_val >= eps).astype(int), s_val)
                rows.append({"fold_id": fid, "n_train": int(len(train)),
                             "n_val": int(len(val)), "n_pos_val": int(y_val.sum()),
                             "epsilon_optimal": eps, **metrics})
                details.append({"fold_id": fid, "scores_val": s_val, "y_val": y_val})
            df = pd.DataFrame(rows)
            wf1 = weighted_median(df["F1"].to_numpy(), df["n_pos_val"].to_numpy())
            if best is None or wf1 > best["wf1"]:
                best = {"wf1": wf1, "contamination": c, "results": df, "details": details}
        self.contamination = best["contamination"]
        self.best_params_ = {"contamination": best["contamination"]}
        self.walkforward_results_ = best["results"]
        self.fold_details_ = best["details"]
        self.threshold_ = weighted_median(
            best["results"]["epsilon_optimal"].to_numpy(),
            best["results"]["n_pos_val"].to_numpy())
        return best["results"]

    def weighted_summary(self) -> dict[str, float]:
        return _weighted_summary(self.walkforward_results_, self.threshold_)

    def save(self, path, drop_details: bool = True) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = self.fold_details_
        if drop_details:
            self.fold_details_ = None
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self.fold_details_ = d

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


class AutoencoderDetector:
    """Dense autoencoder novelty detector (56-24-12-6-12-24-56).

    score = per-row reconstruction MSE. Encoder layers use dropout; early
    stopping monitors an internal validation split = the last 15% (temporal) of
    the train normals (never the val fold). Final model saved via Keras
    ``model.save(*.keras)``."""

    def __init__(self, dropout: float = 0.15, lr: float = 1e-3, max_epochs: int = 200,
                 batch_size: int = 32, patience: int = 15, seed: int = 42):
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.model_ = None
        self.threshold_ = None
        self.input_dim_ = None
        self.walkforward_results_ = None
        self.fold_details_ = None

    def _build(self, input_dim: int):
        import tensorflow as tf
        from tensorflow.keras import layers, models, optimizers
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        inp = layers.Input(shape=(input_dim,))
        x = layers.Dense(24, activation="relu")(inp)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(12, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(6, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(12, activation="relu")(x)
        x = layers.Dense(24, activation="relu")(x)
        out = layers.Dense(input_dim, activation="linear")(x)
        model = models.Model(inp, out)
        model.compile(optimizer=optimizers.Adam(self.lr), loss="mse")
        return model

    def fit(self, X_train_normals) -> "AutoencoderDetector":
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X_train_normals, dtype=float)
        self.input_dim_ = X.shape[1]
        cut = int(np.floor(len(X) * 0.85))  # last 15% temporal = internal val
        X_tr, X_iv = X[:cut], X[cut:]
        self.model_ = self._build(self.input_dim_)
        es = EarlyStopping(monitor="val_loss", patience=self.patience,
                           restore_best_weights=True)
        self.model_.fit(X_tr, X_tr, validation_data=(X_iv, X_iv),
                        epochs=self.max_epochs, batch_size=self.batch_size,
                        callbacks=[es], verbose=0)
        return self

    def score_samples(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        recon = self.model_.predict(X, verbose=0)
        return np.mean((X - recon) ** 2, axis=1)

    def predict(self, X) -> np.ndarray:
        return (self.score_samples(X) >= self.threshold_).astype(int)

    def fit_threshold_walkforward(self, folds_data, scaler, percentiles=range(80, 100)):
        target = scaler.target_col
        rows, details = [], []
        for fold in folds_data:
            fid, train, val = fold["fold_id"], fold["train"], fold["val"]
            tn = train[train[target] == 0]
            Xtr = scaler.transform(tn, fid)
            det = AutoencoderDetector(dropout=self.dropout, lr=self.lr,
                                      max_epochs=self.max_epochs, batch_size=self.batch_size,
                                      patience=self.patience, seed=self.seed).fit(Xtr)
            s_tr = det.score_samples(Xtr)
            s_val = det.score_samples(scaler.transform(val, fid))
            y_val = val[target].to_numpy()
            eps, metrics = _tune_threshold(s_tr, s_val, y_val, percentiles)
            rows.append({"fold_id": fid, "n_train": int(len(train)),
                         "n_val": int(len(val)), "n_pos_val": int(y_val.sum()),
                         "epsilon_optimal": eps, **metrics})
            details.append({"fold_id": fid, "scores_val": s_val, "y_val": y_val})
        results = pd.DataFrame(rows)
        self.walkforward_results_ = results
        self.fold_details_ = details
        self.threshold_ = weighted_median(results["epsilon_optimal"].to_numpy(),
                                          results["n_pos_val"].to_numpy())
        return results

    def weighted_summary(self) -> dict[str, float]:
        return _weighted_summary(self.walkforward_results_, self.threshold_)

    def save(self, path) -> None:
        """Save the Keras model to ``path`` (*.keras) + a sidecar meta json with
        the threshold (so the detector is fully reloadable)."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save(path)
        meta = {"threshold_": self.threshold_, "input_dim_": self.input_dim_,
                "dropout": self.dropout, "lr": self.lr, "max_epochs": self.max_epochs,
                "batch_size": self.batch_size, "patience": self.patience, "seed": self.seed}
        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(path) -> "AutoencoderDetector":
        import json
        import tensorflow as tf
        path = Path(path)
        with open(path.with_suffix(".meta.json")) as f:
            meta = json.load(f)
        det = AutoencoderDetector(dropout=meta["dropout"], lr=meta["lr"],
                                  max_epochs=meta["max_epochs"], batch_size=meta["batch_size"],
                                  patience=meta["patience"], seed=meta["seed"])
        det.model_ = tf.keras.models.load_model(path)
        det.threshold_ = meta["threshold_"]
        det.input_dim_ = meta["input_dim_"]
        return det
