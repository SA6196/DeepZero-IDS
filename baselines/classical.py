"""Classical anomaly detection baselines fit strictly on benign TRAIN data.

Source of Truth: SPEC.md Section 5 & User Request PART D.
Baselines implemented:
1. Isolation Forest (Liu et al. ICDM 2008)
2. One-Class SVM (Schölkopf et al. 2001)
3. Local Outlier Factor with novelty=True (Breunig et al. SIGMOD 2000)
4. PCA Reconstruction Error

Convention:
- All models fit on benign TRAIN only.
- Anomaly score outputs are scaled such that higher values indicate higher abnormality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detection models."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> BaseAnomalyDetector:
        """Fit detector on benign training samples."""
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute continuous anomaly scores where higher = more anomalous."""
        pass


class IsolationForestBaseline(BaseAnomalyDetector):
    """Isolation Forest anomaly detector."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[str, int, float] = "auto",
        contamination: str = "auto",
        random_state: int = 0,
        n_jobs: int = 4,
    ):
        if n_jobs == -1 or n_jobs is None or n_jobs <= 0:
            n_jobs = 4
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray) -> IsolationForestBaseline:
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # sklearn score_samples is the opposite of anomaly score (lower is more anomalous)
        # We negate it so higher score = more anomalous
        return -self.model.score_samples(X)


class OneClassSVMBaseline(BaseAnomalyDetector):
    """One-Class Support Vector Machine."""

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        nu: float = 0.05,
    ):
        self.model = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=nu,
        )

    def fit(self, X: np.ndarray) -> OneClassSVMBaseline:
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # decision_function is positive for inliers, negative for outliers
        # Negate so higher = more anomalous
        return -self.model.decision_function(X)


class LocalOutlierFactorBaseline(BaseAnomalyDetector):
    """Local Outlier Factor in novelty detection mode."""

    def __init__(
        self,
        n_neighbors: int = 20,
        novelty: bool = True,
        n_jobs: int = 4,
    ):
        if not novelty:
            raise ValueError("novelty must be True to score unseen test samples.")
        if n_jobs == -1 or n_jobs is None or n_jobs <= 0:
            n_jobs = 4
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=novelty,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray) -> LocalOutlierFactorBaseline:
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # decision_function is positive for inliers, negative for outliers
        # Negate so higher = more anomalous
        return -self.model.decision_function(X)


class PCAReconstructionBaseline(BaseAnomalyDetector):
    """PCA reconstruction error anomaly detector."""

    def __init__(
        self,
        n_components: int = 16,
        random_state: int = 0,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.model: Optional[PCA] = None

    def fit(self, X: np.ndarray) -> PCAReconstructionBaseline:
        # Handle case where features < requested components
        k = min(self.n_components, X.shape[1])
        self.model = PCA(n_components=k, random_state=self.random_state)
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before scoring.")
        # Project and reconstruct
        X_latent = self.model.transform(X)
        X_rec = self.model.inverse_transform(X_latent)
        # Mean squared reconstruction error per sample
        mse = np.mean((X - X_rec) ** 2, axis=1)
        return mse
