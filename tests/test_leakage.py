"""Unit tests for leakage verification per Global Rule 4 and research invariants.

Source of Truth: SPEC.md Section 8 (Rule 4) & User Specifications:
(a) The scaler's mean equals the mean of TRAIN only (never contaminated with VAL or TEST).
(b) No TEST row index appears in TRAIN or VAL.
(c) Injected contamination equals exactly c% of TRAIN for c in {0, 1, 2, 5, 10}%.
(d) In leave-one-family-out, the held-out family never appears in TRAIN unless
    the contamination variant is on.
(e) Protocol P4 Confound Elimination:
    - len(TRAIN) is IDENTICAL for all c in {0, 1, 2, 5, 10}%.
    - Benign rows for larger c are a strict nested subset of benign rows for smaller c.
(f) Order of Operations:
    - Duplicates removed BEFORE splitting, so no identical flow exists across TRAIN and TEST.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.cicids2017 import CICIDS2017Cleaner, TrainOnlyScaler
from data.preprocess import TabularPreprocessor
from splits.protocols import (
    generate_p1_lofo_splits,
    generate_p2_temporal_splits,
    generate_p4_contamination_splits,
    generate_p4_setting_b_splits,
)


@pytest.fixture
def synthetic_ids_dataframe() -> pd.DataFrame:
    """Generate reproducible synthetic flow dataframe mimicking CIC-IDS2017."""
    rng = np.random.default_rng(42)
    n_rows = 5000

    days = ["Monday"] * 2000 + ["Tuesday"] * 750 + ["Wednesday"] * 750 + ["Thursday"] * 750 + ["Friday"] * 750

    labels = []
    families = []
    for day in days:
        if day == "Monday":
            labels.append("BENIGN")
            families.append("Benign")
        elif day == "Tuesday":
            if rng.random() < 0.6:
                labels.append("BENIGN")
                families.append("Benign")
            else:
                lbl = rng.choice(["FTP-Patator", "SSH-Patator"])
                labels.append(lbl)
                families.append("BruteForce")
        elif day == "Wednesday":
            if rng.random() < 0.6:
                labels.append("BENIGN")
                families.append("Benign")
            else:
                lbl = rng.choice(["DoS slowloris", "DoS Slowhttptest", "DoS Hulk", "DoS GoldenEye"])
                labels.append(lbl)
                families.append("DoS/DDoS")
        elif day == "Thursday":
            if rng.random() < 0.6:
                labels.append("BENIGN")
                families.append("Benign")
            else:
                lbl = rng.choice(["Web Attack – Brute Force", "Web Attack – XSS"])
                labels.append(lbl)
                families.append("WebAttack")
        elif day == "Friday":
            if rng.random() < 0.6:
                labels.append("BENIGN")
                families.append("Benign")
            else:
                lbl = rng.choice(["PortScan", "Bot"])
                family = "PortScan" if lbl == "PortScan" else "Botnet"
                labels.append(lbl)
                families.append(family)

    features = {
        "Flow Duration": rng.uniform(10.0, 100000.0, size=n_rows),
        "Total Fwd Packets": rng.integers(1, 100, size=n_rows).astype(float),
        "Total Backward Packets": rng.integers(0, 100, size=n_rows).astype(float),
        "Flow Bytes/s": rng.uniform(100.0, 500000.0, size=n_rows),
        "Flow Packets/s": rng.uniform(1.0, 5000.0, size=n_rows),
        "Flow IAT Mean": rng.uniform(1.0, 20000.0, size=n_rows),
        "Init_Win_bytes_forward": rng.integers(0, 65535, size=n_rows).astype(float),
    }

    df = pd.DataFrame(features)
    df["Day"] = days
    df["Label"] = labels
    df["Attack_Family"] = families

    attack_mask = df["Attack_Family"] != "Benign"
    df.loc[attack_mask, "Flow Duration"] += 50000.0
    df.loc[attack_mask, "Flow Bytes/s"] += 200000.0

    return df


def test_rule_4a_scaler_mean_equals_train_only(synthetic_ids_dataframe: pd.DataFrame):
    """Leakage Test 4(a):

    The scaler's mean must equal the mean of TRAIN only.
    It must NOT match the mean of VAL, TEST, or ALL data.
    """
    df = synthetic_ids_dataframe
    feature_cols = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Init_Win_bytes_forward",
    ]

    p1_splits = generate_p1_lofo_splits(df, seed=42, val_benign_ratio=0.2, test_benign_ratio=0.2)
    split = p1_splits["DoS/DDoS"]

    train_df = df.loc[split.train_indices, feature_cols]
    val_df = df.loc[split.val_indices, feature_cols]
    test_df = df.loc[split.test_indices, feature_cols]

    scaler = TrainOnlyScaler(scaler_type="standard")
    scaler.fit(train_df)

    expected_train_mean = train_df.mean(axis=0).to_numpy()
    actual_scaler_mean = scaler.mean_

    np.testing.assert_allclose(
        actual_scaler_mean,
        expected_train_mean,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Scaler mean does not match TRAIN mean!",
    )
    val_mean = val_df.mean(axis=0).to_numpy()
    assert not np.allclose(actual_scaler_mean, val_mean, atol=1e-2), (
        "Scaler mean unexpectedly identical to VAL mean!"
    )
    test_mean = test_df.mean(axis=0).to_numpy()
    assert not np.allclose(actual_scaler_mean, test_mean, atol=1e-2), (
        "Scaler mean leaked TEST statistics!"
    )
    all_mean = df[feature_cols].mean(axis=0).to_numpy()
    assert not np.allclose(actual_scaler_mean, all_mean, atol=1e-2), (
        "Scaler mean leaked combined dataset statistics!"
    )


def test_rule_4b_no_test_index_in_train_or_val(synthetic_ids_dataframe: pd.DataFrame):
    """Leakage Test 4(b):

    No TEST row index appears in TRAIN or VAL across P1, P2, and P4 protocols.
    """
    df = synthetic_ids_dataframe

    # Protocol P1
    p1_splits = generate_p1_lofo_splits(df, seed=42)
    for family, split in p1_splits.items():
        train_set = set(split.train_indices)
        val_set = set(split.val_indices)
        test_set = set(split.test_indices)

        assert len(test_set.intersection(train_set)) == 0, f"P1 [{family}]: TEST and TRAIN overlap!"
        assert len(test_set.intersection(val_set)) == 0, f"P1 [{family}]: TEST and VAL overlap!"
        assert len(train_set.intersection(val_set)) == 0, f"P1 [{family}]: TRAIN and VAL overlap!"

    # Protocol P2
    p2_split = generate_p2_temporal_splits(df, seed=42)
    assert len(set(p2_split.test_indices).intersection(set(p2_split.train_indices))) == 0
    assert len(set(p2_split.test_indices).intersection(set(p2_split.val_indices))) == 0
    assert len(set(p2_split.train_indices).intersection(set(p2_split.val_indices))) == 0

    # Protocol P4
    for c in [0.0, 0.01, 0.02, 0.05, 0.10]:
        p4_split = generate_p4_contamination_splits(
            df, seed=42, lofo_family="DoS/DDoS", contamination_rate=c
        )
        assert len(set(p4_split.test_indices).intersection(set(p4_split.train_indices))) == 0
        assert len(set(p4_split.test_indices).intersection(set(p4_split.val_indices))) == 0
        assert len(set(p4_split.train_indices).intersection(set(p4_split.val_indices))) == 0


def test_rule_4c_injected_contamination_equals_exact_percentage(synthetic_ids_dataframe: pd.DataFrame):
    """Leakage Test 4(c):

    Injected contamination equals exactly c% of TRAIN for c in {0, 1, 2, 5, 10}%.
    """
    df = synthetic_ids_dataframe
    contamination_levels = [0.0, 0.01, 0.02, 0.05, 0.10]

    for c in contamination_levels:
        split = generate_p4_contamination_splits(
            df,
            seed=42,
            lofo_family="DoS/DDoS",
            contamination_rate=c,
            include_test_family=False,
        )

        train_indices = split.train_indices
        train_labels = df.loc[train_indices, "Attack_Family"].to_numpy()

        n_total_train = len(train_indices)
        n_attack_in_train = int(np.sum(train_labels != "Benign"))

        expected_attack_count = int(round(c * n_total_train))
        assert n_attack_in_train == expected_attack_count, (
            f"Contamination count mismatch at c={c}: "
            f"expected {expected_attack_count}, got {n_attack_in_train}"
        )
        actual_rate = n_attack_in_train / n_total_train
        np.testing.assert_allclose(actual_rate, c, atol=1e-5)


def test_rule_4d_lofo_held_out_family_isolation(synthetic_ids_dataframe: pd.DataFrame):
    """Leakage Test 4(d):

    In leave-one-family-out (P1 and P4), the held-out family never appears in TRAIN
    unless the contamination variant is explicitly enabled.
    """
    df = synthetic_ids_dataframe
    all_attack_families = ["DoS/DDoS", "BruteForce", "WebAttack", "PortScan", "Botnet"]

    # 1. P1
    p1_splits = generate_p1_lofo_splits(df, seed=42, target_families=all_attack_families)
    for family in all_attack_families:
        split = p1_splits[family]
        train_families = set(df.loc[split.train_indices, "Attack_Family"].unique())
        val_families = set(df.loc[split.val_indices, "Attack_Family"].unique())
        assert family not in train_families
        assert family not in val_families

    # 2. P4 Headline variant
    for family in all_attack_families:
        for c in [0.01, 0.05, 0.10]:
            split = generate_p4_contamination_splits(
                df,
                seed=42,
                lofo_family=family,
                contamination_rate=c,
                include_test_family=False,
            )
            train_families = set(df.loc[split.train_indices, "Attack_Family"].unique())
            assert family not in train_families, (
                f"P4 Headline Variant: '{family}' found in TRAIN at c={c}!"
            )


def test_p4_confound_len_train_identical_and_benign_nested(synthetic_ids_dataframe: pd.DataFrame):
    """Protocol P4 Confound Elimination Test:

    1. Total TRAIN size N is IDENTICAL for every c in {0, 1, 2, 5, 10}%.
    2. N is a fixed multiple of 100.
    3. Benign rows for larger c are a strict nested subset of benign rows for smaller c.
    """
    df = synthetic_ids_dataframe
    contamination_levels = [0.0, 0.01, 0.02, 0.05, 0.10]
    train_sizes = []
    benign_subsets = []

    for c in contamination_levels:
        split = generate_p4_contamination_splits(
            df,
            seed=42,
            lofo_family="DoS/DDoS",
            contamination_rate=c,
            include_test_family=False,
        )
        train_sizes.append(len(split.train_indices))

        train_df = df.loc[split.train_indices]
        benign_rows = set(train_df.index[train_df["Attack_Family"] == "Benign"])
        benign_subsets.append((c, benign_rows))

    # 1. Assert len(TRAIN) is IDENTICAL for all c
    assert len(set(train_sizes)) == 1, (
        f"Confound detected! TRAIN sizes vary across contamination levels: {dict(zip(contamination_levels, train_sizes))}"
    )
    N = train_sizes[0]

    # 2. Assert N is a fixed multiple of 100
    assert N % 100 == 0, f"TRAIN size N={N} is not a multiple of 100!"

    # 3. Assert nesting: benign rows for larger c must be a subset of benign rows for smaller c
    for i in range(len(benign_subsets) - 1):
        c_small, benign_small = benign_subsets[i]
        c_large, benign_large = benign_subsets[i + 1]
        assert c_large > c_small
        assert benign_large.issubset(benign_small), (
            f"Nesting violation! Benign rows at c={c_large} are not a subset of benign rows at c={c_small}"
        )
        # Expected exact benign count: N * (1 - c/100)
        expected_benign_count = int(round(N * (1.0 - c_large)))
        assert len(benign_large) == expected_benign_count, (
            f"Expected {expected_benign_count} benign rows at c={c_large}, got {len(benign_large)}"
        )


def test_order_of_operations_duplicates_removed_before_split():
    """Verify that duplicates are removed BEFORE splitting.

    Identical flow vectors must NOT land in both TRAIN and TEST.
    """
    rng = np.random.default_rng(123)
    feature_cols = ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s"]

    # Base dataset
    n_base = 600
    base_data = {
        "Flow Duration": rng.uniform(10.0, 1000.0, size=n_base),
        "Total Fwd Packets": rng.integers(1, 20, size=n_base).astype(float),
        "Flow Bytes/s": rng.uniform(100.0, 10000.0, size=n_base),
        "Label": ["BENIGN"] * 500 + ["DoS Hulk"] * 100,
        "Day": ["Monday"] * 500 + ["Wednesday"] * 100,
    }
    raw_df = pd.DataFrame(base_data)

    # Intentionally inject duplicates of flow 0 into multiple rows
    duplicate_rows = raw_df.iloc[[0] * 50].copy()
    raw_df_with_dups = pd.concat([raw_df, duplicate_rows], ignore_index=True)

    # Process through cleaner
    cleaner = CICIDS2017Cleaner()
    cleaned_df, report = cleaner.clean(raw_df_with_dups, print_negative_summary=False)

    assert report.duplicate_rows_dropped >= 50
    # Assert no duplicate rows remain in cleaned_df across features
    assert not cleaned_df[feature_cols].duplicated().any(), "Cleaned dataset contains duplicates!"

    # Generate P1 splits
    p1_splits = generate_p1_lofo_splits(cleaned_df, seed=42)
    split = p1_splits["DoS/DDoS"]

    train_features = cleaned_df.loc[split.train_indices, feature_cols].values
    test_features = cleaned_df.loc[split.test_indices, feature_cols].values

    # Check for identical flows between TRAIN and TEST
    # Convert rows to tuples for exact set membership
    train_tuples = {tuple(row) for row in train_features}
    for test_row in test_features:
        assert tuple(test_row) not in train_tuples, (
            "Identical flow vector found in both TRAIN and TEST! Leakage through duplicate flows!"
        )


def test_tabular_preprocessor_pipeline_train_only_statistics(tmp_path):
    """Leakage Test for Preprocessor Pipeline:

    1. Constant columns determined on TRAIN only.
    2. Log1p candidate features determined by TRAIN skewness only; sentinels excluded.
    3. Scaler mean and variance equal TRAIN-only statistics and strictly differ from VAL/TEST.
    4. Serialized pipeline reloads with identical fitted parameters.
    """
    from data.preprocess import TabularPreprocessor

    rng = np.random.default_rng(999)
    n_train = 500
    n_val = 100
    n_test = 200

    # Create synthetic feature sets with distinct distributions
    # Col 1: normal feature
    train_c1 = rng.normal(10.0, 2.0, size=n_train)
    val_c1 = rng.normal(50.0, 5.0, size=n_val)
    test_c1 = rng.normal(100.0, 10.0, size=n_test)

    # Col 2: constant in TRAIN only (should be dropped)
    train_c2 = np.zeros(n_train)
    val_c2 = rng.uniform(0.0, 1.0, size=n_val)
    test_c2 = rng.uniform(0.0, 1.0, size=n_test)

    # Col 3: heavy-tailed non-negative (high positive skew in TRAIN, should get log1p)
    train_c3 = rng.lognormal(mean=0.0, sigma=1.5, size=n_train)
    val_c3 = rng.normal(0.0, 1.0, size=n_val)
    test_c3 = rng.normal(0.0, 1.0, size=n_test)

    # Col 4: sentinel column with -1 (must NOT get log1p despite skew)
    train_c4 = rng.choice([-1.0, 1024.0, 2048.0], size=n_train)
    val_c4 = rng.choice([-1.0, 1024.0], size=n_val)
    test_c4 = rng.choice([-1.0, 1024.0], size=n_test)

    # Col 5: constant in TEST only (must NOT be dropped because it is NOT constant in TRAIN)
    train_c5 = rng.uniform(1.0, 5.0, size=n_train)
    val_c5 = np.ones(n_val)
    test_c5 = np.ones(n_test)

    train_df = pd.DataFrame({
        "Feat_Normal": train_c1,
        "Feat_TrainConstant": train_c2,
        "Feat_HeavyTail": train_c3,
        "Init_Win_bytes_forward": train_c4,
        "Feat_TestConstant": train_c5,
    })
    val_df = pd.DataFrame({
        "Feat_Normal": val_c1,
        "Feat_TrainConstant": val_c2,
        "Feat_HeavyTail": val_c3,
        "Init_Win_bytes_forward": val_c4,
        "Feat_TestConstant": val_c5,
    })
    test_df = pd.DataFrame({
        "Feat_Normal": test_c1,
        "Feat_TrainConstant": test_c2,
        "Feat_HeavyTail": test_c3,
        "Init_Win_bytes_forward": test_c4,
        "Feat_TestConstant": test_c5,
    })

    prep = TabularPreprocessor(skew_threshold=2.0)
    X_train_trans = prep.fit_transform(train_df)

    # 1. Constant columns determined from TRAIN only:
    # Feat_TrainConstant dropped; Feat_TestConstant NOT dropped
    assert "Feat_TrainConstant" in prep.constant_cols_
    assert "Feat_TestConstant" not in prep.constant_cols_

    # 2. Heavy-tailed non-negative feature identified on TRAIN; sentinel column excluded
    assert "Feat_HeavyTail" in prep.log1p_cols_
    assert "Init_Win_bytes_forward" not in prep.log1p_cols_

    # 3. Scaler mean equals TRAIN transformed mean exactly
    expected_train_c1_mean = train_df["Feat_Normal"].mean()
    expected_train_c3_mean = np.log1p(train_df["Feat_HeavyTail"]).mean()

    feat_idx_c1 = prep.retained_cols_.index("Feat_Normal")
    feat_idx_c3 = prep.retained_cols_.index("Feat_HeavyTail")

    np.testing.assert_allclose(prep.mean_[feat_idx_c1], expected_train_c1_mean, rtol=1e-5)
    np.testing.assert_allclose(prep.mean_[feat_idx_c3], expected_train_c3_mean, rtol=1e-5)

    # Assert scaler mean does NOT match VAL or TEST statistics
    assert not np.isclose(prep.mean_[feat_idx_c1], val_df["Feat_Normal"].mean(), atol=1.0)
    assert not np.isclose(prep.mean_[feat_idx_c1], test_df["Feat_Normal"].mean(), atol=1.0)

    # 4. Persistence: save and reload
    save_path = tmp_path / "pipeline.joblib"
    prep.save(save_path)
    loaded_prep = TabularPreprocessor.load(save_path)

    assert loaded_prep.constant_cols_ == prep.constant_cols_
    assert loaded_prep.log1p_cols_ == prep.log1p_cols_
    assert loaded_prep.retained_cols_ == prep.retained_cols_
    np.testing.assert_array_equal(loaded_prep.mean_, prep.mean_)
    np.testing.assert_array_equal(loaded_prep.scale_, prep.scale_)

    # Transform on test with loaded pipeline
    X_test_trans = loaded_prep.transform(test_df)
    assert X_test_trans.shape == (n_test, len(prep.retained_cols_))


def test_tabular_preprocessor_post_standardization_clipping(tmp_path: Path):
    """Verify that TabularPreprocessor strictly bounds output features to [-clip_value, clip_value]."""
    rng = np.random.default_rng(1234)
    train_df = pd.DataFrame({
        "feat_a": rng.normal(loc=0.0, scale=1.0, size=200),
        "feat_b": rng.uniform(0.0, 100.0, size=200),
    })

    # Extreme test samples far beyond 10 standard deviations
    extreme_test_df = pd.DataFrame({
        "feat_a": [1e9, -1e9, 0.0, 5.0],
        "feat_b": [-1e9, 1e9, 50.0, 20.0],
    })

    prep = TabularPreprocessor(clip_value=10.0)
    prep.fit(train_df)

    X_train_trans = prep.transform(train_df)
    assert np.max(np.abs(X_train_trans)) <= 10.0

    X_extreme_trans = prep.transform(extreme_test_df)
    # (a) Output max absolute value must be <= 10.0
    assert np.max(np.abs(X_extreme_trans)) <= 10.0
    assert np.isclose(X_extreme_trans[0, 0], 10.0)
    assert np.isclose(X_extreme_trans[1, 0], -10.0)

    # Verify persistence of clip_value
    save_path = tmp_path / "clipping_pipeline.joblib"
    prep.save(save_path)
    loaded = TabularPreprocessor.load(save_path)
    assert loaded.clip_value == 10.0
    X_loaded_trans = loaded.transform(extreme_test_df)
    assert np.max(np.abs(X_loaded_trans)) <= 10.0


def test_p4_c0_train_indices_identical_across_families(synthetic_ids_dataframe: pd.DataFrame):
    """Verify that at c=0, P4 train indices are 100% identical across all held-out families."""
    df = synthetic_ids_dataframe
    families = ["BruteForce", "PortScan", "WebAttack", "DoS/DDoS"]

    splits = {
        fam: generate_p4_contamination_splits(
            df,
            seed=0,
            lofo_family=fam,
            contamination_rate=0.0,
            n_train_total=1000,
        )
        for fam in families
    }

    base_train = splits["BruteForce"].train_indices
    for fam in families[1:]:
        np.testing.assert_array_equal(
            splits[fam].train_indices,
            base_train,
            err_msg=f"c=0 train indices for {fam} do not match BruteForce!",
        )


def test_p4_setting_b_no_overlap_between_train_and_test(synthetic_ids_dataframe: pd.DataFrame):
    """Verify Setting B leakage control: zero overlap between TRAIN and TEST for attack family F."""
    df = synthetic_ids_dataframe
    # Ensure synthetic dataset has enough attack rows for DoS/DDoS
    for c in [0.0, 0.02, 0.05, 0.10]:
        split_b = generate_p4_setting_b_splits(
            df,
            seed=42,
            family="DoS/DDoS",
            contamination_rate=c,
            n_train_total=1000,
        )
        train_set = set(split_b.train_indices)
        test_set = set(split_b.test_indices)
        val_set = set(split_b.val_indices)

        # Invariant 1: zero overlap between train and test
        assert len(train_set.intersection(test_set)) == 0, f"Train and test overlap at c={c}!"
        # Invariant 2: zero overlap between train and val
        assert len(train_set.intersection(val_set)) == 0, f"Train and val overlap at c={c}!"
        # Invariant 3: zero overlap between val and test
        assert len(val_set.intersection(test_set)) == 0, f"Val and test overlap at c={c}!"



