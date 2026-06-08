"""Per-fold feature scaling for leakage-free walk-forward CV.

A single ``StandardScaler`` fit on the whole dataset would leak information
across folds (validation statistics bleed into training). ``FoldScaler`` keeps
one scaler per fold (fit on that fold's *train* only) plus a "final" scaler fit
on the entire development set, used to transform the sealed test holdout.

The target column ``Y`` is never scaled; it is dropped from the feature matrix.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


class FoldScaler:
    def __init__(self, target_col: str = "Y"):
        self.target_col = target_col
        self.fold_scalers: dict[int, StandardScaler] = {}
        self.final_scaler: StandardScaler | None = None
        self.feature_cols_: list[str] | None = None

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select the feature columns (drop the target), locking column order."""
        if self.feature_cols_ is None:
            self.feature_cols_ = [c for c in df.columns if c != self.target_col]
        missing = [c for c in self.feature_cols_ if c not in df.columns]
        if missing:
            raise KeyError(f"FoldScaler: missing feature columns {missing}")
        return df[self.feature_cols_]

    def fit_per_fold(self, folds_data: list[dict]) -> "FoldScaler":
        """Fit one scaler per fold on the fold's full train (all rows, not only
        normals — the scaler captures feature geometry, not the anomaly model)."""
        for fold in folds_data:
            X = self._features(fold["train"])
            self.fold_scalers[fold["fold_id"]] = StandardScaler().fit(X)
        return self

    def transform(self, X: pd.DataFrame, fold_id: int):
        if fold_id not in self.fold_scalers:
            raise KeyError(f"FoldScaler: no scaler fit for fold {fold_id}")
        return self.fold_scalers[fold_id].transform(self._features(X))

    def fit_on_development(self, df_dev: pd.DataFrame) -> "FoldScaler":
        """Fit the final scaler on the entire development set (2000-2018)."""
        self.final_scaler = StandardScaler().fit(self._features(df_dev))
        return self

    def transform_holdout(self, X: pd.DataFrame):
        if self.final_scaler is None:
            raise RuntimeError("FoldScaler: call fit_on_development() first")
        return self.final_scaler.transform(self._features(X))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "FoldScaler":
        with open(path, "rb") as f:
            return pickle.load(f)
