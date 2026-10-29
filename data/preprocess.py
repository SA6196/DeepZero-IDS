"""Tabular data preprocessing pipeline fit strictly on TRAIN data.

Source of Truth: SPEC.md & User Request PART B.
Global Rule 1: Fit ALL scalers, encoders, and transforms on TRAIN only.
Never use TEST statistics anywhere.

Pipeline steps:
1. Drop constant columns (determined on TRAIN only).
2. For non-negative, heavy-tailed features (skewness on TRAIN > skew_threshold),
   apply log1p. Sentinel columns (Init_Win_bytes_forward / backward = -1) are excluded.
3. StandardScaler fit on TRAIN only, handling zero-variance safely.
4. Serializable and reloadable per evaluation split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from data.cicids2017 import SENTINEL_COLUMNS


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessing pipeline enforcing strict TRAIN-only parameter estimation."""

    def __init__(
        self,
        skew_threshold: float = 3.0,
        sentinel_cols: Optional[Sequence[str]] = None,
        clip_value: Optional[float] = 10.0,
    ):
        self.skew_threshold = float(skew_threshold)
        self.sentinel_cols = [c.lower() for c in (sentinel_cols or SENTINEL_COLUMNS)]
        self.clip_value = float(clip_value) if clip_value is not None else None
        
        # Fitted attributes
        self.is_fitted: bool = False
        self.feature_names_in_: List[str] = []
        self.constant_cols_: List[str] = []
        self.retained_cols_: List[str] = []
        self.log1p_cols_: List[str] = []
        self.scaler: Optional[StandardScaler] = None

    def fit(self, X_train: pd.DataFrame, y: Any = None) -> TabularPreprocessor:
        """Fit preprocessing parameters strictly on TRAIN data."""
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame with feature column names.")

        self.feature_names_in_ = list(X_train.columns)

        # 1. Identify constant columns on TRAIN only
        constant_cols = []
        for col in self.feature_names_in_:
            # Check unique values (ignoring or including NaNs, but data is already clean)
            if X_train[col].nunique(dropna=False) <= 1:
                constant_cols.append(col)

        self.constant_cols_ = sorted(constant_cols)
        self.retained_cols_ = [c for c in self.feature_names_in_ if c not in self.constant_cols_]

        X_active = X_train[self.retained_cols_].copy()

        # 2. Identify heavy-tailed non-negative features on TRAIN
        log1p_candidates = []
        for col in self.retained_cols_:
            if col.lower() in self.sentinel_cols:
                # Sentinel columns like Init_Win_bytes can have -1 and must NOT be log1p-transformed
                continue

            vals = X_active[col].to_numpy(dtype=np.float64)
            # Only apply log1p to non-negative features
            if (vals >= 0.0).all():
                skewness = float(X_active[col].skew())
                if skewness > self.skew_threshold:
                    log1p_candidates.append(col)

        self.log1p_cols_ = sorted(log1p_candidates)

        # Apply log1p on TRAIN for the identified columns
        for col in self.log1p_cols_:
            X_active[col] = np.log1p(X_active[col].clip(lower=0.0))

        # 3. Fit StandardScaler on TRAIN only with safe zero-variance handling
        self.scaler = StandardScaler(copy=True)
        self.scaler.fit(X_active.to_numpy(dtype=np.float64))

        # Handle zero variance safely: if any scale is 0.0, replace with 1.0 to avoid NaN
        if hasattr(self.scaler, "scale_") and self.scaler.scale_ is not None:
            self.scaler.scale_[self.scaler.scale_ == 0.0] = 1.0

        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform data using fitted TRAIN parameters."""
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError("Preprocessor must be fitted on TRAIN data before transform.")

        if isinstance(X, pd.DataFrame):
            # Check if all retained columns are present
            missing = [c for c in self.retained_cols_ if c not in X.columns]
            if missing:
                raise ValueError(f"Input DataFrame is missing required features: {missing}")
            X_active = X[self.retained_cols_].copy()
        else:
            # Assume numpy array matches feature_names_in_ order
            if X.shape[1] == len(self.feature_names_in_):
                df_temp = pd.DataFrame(X, columns=self.feature_names_in_)
                X_active = df_temp[self.retained_cols_].copy()
            elif X.shape[1] == len(self.retained_cols_):
                X_active = pd.DataFrame(X, columns=self.retained_cols_)
            else:
                raise ValueError(
                    f"Expected input with {len(self.feature_names_in_)} or {len(self.retained_cols_)} columns, "
                    f"got {X.shape[1]}"
                )

        # Apply log1p using the fitted TRAIN column list
        for col in self.log1p_cols_:
            X_active[col] = np.log1p(X_active[col].clip(lower=0.0))

        # Scale using fitted parameters
        X_scaled = self.scaler.transform(X_active.to_numpy(dtype=np.float64))

        # Fixed, data-independent post-standardization clipping
        if self.clip_value is not None:
            X_scaled = np.clip(X_scaled, -self.clip_value, self.clip_value)

        return X_scaled

    def fit_transform(self, X_train: pd.DataFrame, y: Any = None) -> np.ndarray:
        return self.fit(X_train, y).transform(X_train)

    @property
    def mean_(self) -> np.ndarray:
        if self.scaler is not None and hasattr(self.scaler, "mean_"):
            return self.scaler.mean_
        raise AttributeError("Preprocessor is not fitted or has no mean_")

    @property
    def scale_(self) -> np.ndarray:
        if self.scaler is not None and hasattr(self.scaler, "scale_"):
            return self.scaler.scale_
        raise AttributeError("Preprocessor is not fitted or has no scale_")

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize fitted pipeline to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> TabularPreprocessor:
        """Deserialize fitted pipeline from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Fitted preprocessor not found at {filepath}")
        loaded = joblib.load(filepath)
        if not isinstance(loaded, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return loaded
