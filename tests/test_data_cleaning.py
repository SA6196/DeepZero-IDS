"""Unit tests for CIC-IDS2017 data cleaner, negative inspection, schema validation, and persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from data.cicids2017 import (
    CICIDS2017Cleaner,
    CICIDS2017Dataset,
    SchemaValidationError,
    inspect_negative_values,
)
from splits.protocols import (
    SplitIndices,
    generate_p1_lofo_splits,
    save_splits,
    load_splits,
)


def test_cleaner_inspects_negatives_and_preserves_sentinels():
    """Verify negative value inspection, sentinel column preservation, and selective filtering."""
    data = {
        "Flow ID": ["id1", "id2", "id3", "id4", "id5", "id6", "id1"],
        "Source IP": ["192.168.1.1"] * 7,
        "Destination Port": [80, 443, 22, 8080, 80, 443, 80],
        "Timestamp": ["2017-07-03 09:00:00"] * 7,
        "Flow Duration": [100.0, 200.0, -5.0, 400.0, 500.0, 600.0, 100.0],  # row 2 has negative duration
        "Flow Bytes/s": [10.0, np.inf, 30.0, np.nan, 50.0, 60.0, 10.0],     # row 1 inf, row 3 nan
        "Total Fwd Packets": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.0],
        "Init_Win_bytes_forward": [-1.0, -1.0, -1.0, 1024.0, -1.0, 2048.0, -1.0],  # Legitimate sentinels (-1)
        "Label": [
            "BENIGN",
            "BENIGN",
            "DoS Hulk",
            "PortScan",
            "FTP-Patator",
            "BENIGN",
            "BENIGN",
        ],
    }
    raw_df = pd.DataFrame(data)

    # 1. Negative inspection function test
    neg_report = inspect_negative_values(raw_df, label_col="Label", print_summary=False)
    assert not neg_report.empty
    # Check that Init_Win_bytes_forward is recognized as sentinel
    sentinel_entry = neg_report[neg_report["column"] == "Init_Win_bytes_forward"]
    assert not sentinel_entry.empty
    assert bool(sentinel_entry.iloc[0]["is_sentinel"]) is True

    # 2. Default cleaner: drop_negative_artifacts=False
    # Rows with negative values are NOT dropped yet
    cleaner_default = CICIDS2017Cleaner(drop_negative_artifacts=False)
    clean_df, report = cleaner_default.clean(raw_df, print_negative_summary=False)

    assert report.negative_value_rows_dropped == 0
    # Sentinel column is preserved in feature matrix
    assert "Init_Win_bytes_forward" in clean_df.columns
    assert (clean_df["Init_Win_bytes_forward"] == -1.0).any()

    # 3. Cleaner with drop_negative_artifacts=True
    # Rows with negative values in physically impossible columns (Flow Duration < 0) are dropped,
    # but sentinels (Init_Win_bytes_forward = -1) are KEPT!
    cleaner_drop = CICIDS2017Cleaner(drop_negative_artifacts=True)
    clean_df_drop, report_drop = cleaner_drop.clean(raw_df, print_negative_summary=False)

    assert report_drop.negative_value_rows_dropped == 1
    # Check that rows with Init_Win_bytes_forward = -1 are still retained!
    assert (clean_df_drop["Init_Win_bytes_forward"] == -1.0).sum() >= 1
    # Verify report records rows dropped per label
    assert "DoS Hulk" in report_drop.rows_dropped_per_label


def test_schema_validation_missing_and_unexpected_columns(tmp_path: Path):
    """Verify loader raises clear SchemaValidationError for missing and unexpected columns."""
    sample_data = {
        "Flow Duration": [100.0, 200.0],
        "Total Fwd Packets": [2.0, 3.0],
        "Label": ["BENIGN", "BENIGN"],
        "Unexpected_Column_XYZ": [1, 2],  # Not in schema
    }
    df = pd.DataFrame(sample_data)
    csv_path = tmp_path / "test_missing_cols.csv"
    df.to_csv(csv_path, index=False)

    loader = CICIDS2017Dataset(
        data_path=csv_path,
        schema_path="configs/schema_cicids2017.yaml",
        strict_schema=True,
    )

    with pytest.raises(SchemaValidationError) as exc_info:
        loader.load_raw()

    err = str(exc_info.value)
    assert "Missing expected columns" in err
    assert "Unexpected extra columns" in err
    assert "Unexpected_Column_XYZ" in err


def test_unmapped_label_detection():
    """Verify cleaner detects unmapped labels against YAML mapping instead of guessing."""
    data = {
        "Flow Duration": [100.0, 200.0],
        "Label": ["BENIGN", "Unseen_New_Attack_Variant_2026"],
    }
    df = pd.DataFrame(data)

    cleaner = CICIDS2017Cleaner(family_mapping_path="configs/attack_families_cicids2017.yaml")
    clean_df, report = cleaner.clean(df, print_negative_summary=False)

    assert "Unseen_New_Attack_Variant_2026" in report.unmapped_labels_found
    # Unmapped label mapped to Unknown_Attack
    mapped = clean_df.loc[clean_df["Label"] == "Unseen_New_Attack_Variant_2026", "Attack_Family"].iloc[0]
    assert mapped == "Unknown_Attack"


def test_split_indices_save_and_load(tmp_path: Path):
    """Verify split persistence to JSON and exact reconstruction."""
    train = np.array([0, 1, 2, 5, 8], dtype=np.int64)
    val = np.array([3, 4], dtype=np.int64)
    test = np.array([6, 7, 9, 10], dtype=np.int64)

    original = SplitIndices(
        protocol="P1",
        seed=123,
        family="PortScan",
        train_indices=train,
        val_indices=val,
        test_indices=test,
        contamination_rate=0.0,
        metadata={"note": "test_save"},
    )

    json_path = tmp_path / "test_split.json"
    original.save(json_path)

    loaded = load_splits(json_path)

    assert loaded.protocol == original.protocol
    assert loaded.seed == original.seed
    assert loaded.family == original.family
    assert loaded.contamination_rate == original.contamination_rate
    np.testing.assert_array_equal(loaded.train_indices, original.train_indices)
    np.testing.assert_array_equal(loaded.val_indices, original.val_indices)
    np.testing.assert_array_equal(loaded.test_indices, original.test_indices)
    assert loaded.metadata["note"] == "test_save"
