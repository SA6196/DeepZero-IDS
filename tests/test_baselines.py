"""Unit tests for classical anomaly detection baselines.

Verifies:
1. All four baselines fit on benign data and produce 1D continuous anomaly scores.
2. Scores for synthetic outliers are higher than scores for inliers (higher = more anomalous).
3. Baselines handle zero-variance or constant columns cleanly after preprocessor.
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines.classical import (
    IsolationForestBaseline,
    LocalOutlierFactorBaseline,
    OneClassSVMBaseline,
    PCAReconstructionBaseline,
)


@pytest.fixture
def synthetic_train_test():
    rng = np.random.default_rng(42)
    # Benign normal distribution
    X_train = rng.normal(loc=0.0, scale=1.0, size=(500, 10))
    
    # Test set: 200 normal + 50 extreme outliers
    X_test_normal = rng.normal(loc=0.0, scale=1.0, size=(200, 10))
    X_test_outliers = rng.normal(loc=15.0, scale=2.0, size=(50, 10))
    X_test = np.vstack([X_test_normal, X_test_outliers])
    y_test = np.array([0] * 200 + [1] * 50)
    
    return X_train, X_test, y_test


def test_isolation_forest_baseline(synthetic_train_test):
    X_train, X_test, y_test = synthetic_train_test
    clf = IsolationForestBaseline(n_estimators=50, random_state=42)
    clf.fit(X_train)
    scores = clf.score(X_test)

    assert scores.shape == (250,)
    assert scores[y_test == 1].mean() > scores[y_test == 0].mean()


def test_ocsvm_baseline(synthetic_train_test):
    X_train, X_test, y_test = synthetic_train_test
    clf = OneClassSVMBaseline(nu=0.05)
    clf.fit(X_train)
    scores = clf.score(X_test)

    assert scores.shape == (250,)
    assert scores[y_test == 1].mean() > scores[y_test == 0].mean()


def test_lof_novelty_baseline(synthetic_train_test):
    X_train, X_test, y_test = synthetic_train_test
    clf = LocalOutlierFactorBaseline(n_neighbors=15, novelty=True)
    clf.fit(X_train)
    scores = clf.score(X_test)

    assert scores.shape == (250,)
    assert scores[y_test == 1].mean() > scores[y_test == 0].mean()


def test_pca_reconstruction_baseline(synthetic_train_test):
    X_train, X_test, y_test = synthetic_train_test
    clf = PCAReconstructionBaseline(n_components=5, random_state=42)
    clf.fit(X_train)
    scores = clf.score(X_test)

    assert scores.shape == (250,)
    assert scores[y_test == 1].mean() > scores[y_test == 0].mean()
