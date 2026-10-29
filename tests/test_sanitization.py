"""Unit tests for seed-guided sanitization.

DEV DATA, NOT FOR REPORTED RESULTS
Requirements:
(a) S is disjoint from the pool, VAL, and TEST.
(b) The sanitization function receives only unlabeled features and S, no labels.
(c) Kept rows plus dropped rows equals the pool size.
"""

from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
import pytest

from robustness.sanitization import draw_trusted_seed_set, filter_pool_with_seed
from splits.protocols import generate_p4_setting_b_splits


def test_seed_set_disjointness(synthetic_ids_dataframe: pd.DataFrame):
    """(a) Verify that seed set S is strictly disjoint from contaminated pool, VAL-BENIGN, and TEST."""
    df = synthetic_ids_dataframe
    seed = 42
    n_train_max = 500
    n_seed = 100

    S_indices, S_fit, S_cal = draw_trusted_seed_set(
        df,
        seed=seed,
        n_seed=n_seed,
        n_train_max=n_train_max,
        val_benign_ratio=0.2,
        test_benign_ratio=0.2,
    )

    assert len(S_indices) == n_seed
    assert len(S_fit) + len(S_cal) == n_seed
    assert len(set(S_fit).intersection(set(S_cal))) == 0

    # Test disjointness against Setting B split across multiple contamination levels
    for c in [0.0, 0.02, 0.05, 0.10]:
        split_b = generate_p4_setting_b_splits(
            df,
            seed=seed,
            family="DoS/DDoS",
            contamination_rate=c,
            n_train_total=n_train_max,
        )

        S_set = set(S_indices)
        pool_set = set(split_b.train_indices)
        val_set = set(split_b.val_indices)
        test_set = set(split_b.test_indices)

        # 1. Disjoint from contaminated pool
        overlap_pool = S_set.intersection(pool_set)
        assert len(overlap_pool) == 0, f"S overlaps with contaminated pool by {len(overlap_pool)} rows at c={c}!"

        # 2. Disjoint from VAL-BENIGN
        overlap_val = S_set.intersection(val_set)
        assert len(overlap_val) == 0, f"S overlaps with VAL-BENIGN by {len(overlap_val)} rows at c={c}!"

        # 3. Disjoint from TEST
        overlap_test = S_set.intersection(test_set)
        assert len(overlap_test) == 0, f"S overlaps with TEST by {len(overlap_test)} rows at c={c}!"


def test_filter_pool_signature_no_labels():
    """(b) Verify sanitization function accepts only unlabeled features and S, no labels."""
    sig = inspect.signature(filter_pool_with_seed)
    params = list(sig.parameters.keys())

    # Check required inputs
    assert "X_pool_raw" in params
    assert "X_seed_fit_raw" in params
    assert "X_seed_cal_raw" in params

    # Strictly check that no label parameter exists
    forbidden = ["label", "target", "ground_truth", "gt"]
    for p in params:
        p_lower = p.lower()
        for f in forbidden:
            assert f not in p_lower, f"Forbidden label parameter '{p}' found in filter_pool_with_seed!"
        assert p_lower != "y", f"Forbidden parameter 'y' found in filter_pool_with_seed!"
        assert not p_lower.startswith("y_"), f"Forbidden parameter '{p}' found in filter_pool_with_seed!"
        assert not p_lower.endswith("_y"), f"Forbidden parameter '{p}' found in filter_pool_with_seed!"


def test_kept_plus_dropped_equals_pool_size():
    """(c) Verify kept rows plus dropped rows equals pool size on both AE and PCA."""
    rng = np.random.default_rng(999)
    n_pool = 300
    n_fit = 80
    n_cal = 20
    d = 8

    # Synthetic unlabeled features
    X_pool_raw = pd.DataFrame(rng.normal(0, 1, size=(n_pool, d)), columns=[f"f_{i}" for i in range(d)])
    X_seed_fit_raw = pd.DataFrame(rng.normal(0, 1, size=(n_fit, d)), columns=[f"f_{i}" for i in range(d)])
    X_seed_cal_raw = pd.DataFrame(rng.normal(0, 1, size=(n_cal, d)), columns=[f"f_{i}" for i in range(d)])

    # Inject a few outliers into pool
    X_pool_raw.iloc[:10] += 15.0

    # Test with model_type="pca"
    keep_pca, drop_pca, thresh_pca, _, _ = filter_pool_with_seed(
        X_pool_raw,
        X_seed_fit_raw,
        X_seed_cal_raw,
        model_type="pca",
        q=0.05,
        seed=42,
    )
    assert len(keep_pca) == n_pool
    assert len(drop_pca) == n_pool
    assert np.sum(keep_pca) + np.sum(drop_pca) == n_pool
    # Verify extreme outliers were dropped
    assert np.sum(drop_pca[:10]) > 0

    # Test with model_type="ae"
    keep_ae, drop_ae, thresh_ae, _, _ = filter_pool_with_seed(
        X_pool_raw,
        X_seed_fit_raw,
        X_seed_cal_raw,
        model_type="ae",
        q=0.05,
        seed=42,
        max_epochs=3,
        batch_size=32,
    )
    assert len(keep_ae) == n_pool
    assert len(drop_ae) == n_pool
    assert np.sum(keep_ae) + np.sum(drop_ae) == n_pool
