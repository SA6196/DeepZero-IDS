"""Validation script executing Steps 1 to 5 on real raw CSV data.

DEV DATA, NOT FOR REPORTED RESULTS
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles to prevent charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from data.cicids2017 import (
    CICIDS2017Cleaner,
    CICIDS2017Dataset,
    inspect_negative_values,
)


def run_validation():
    print("=" * 80)
    print("DEV DATA, NOT FOR REPORTED RESULTS: CIC-IDS2017 Pipeline Validation")
    print("=" * 80)

    raw_dir = REPO_ROOT / "data" / "raw"
    loader = CICIDS2017Dataset(
        data_path=raw_dir,
        schema_path=REPO_ROOT / "configs" / "schema_cicids2017.yaml",
        family_mapping_path=REPO_ROOT / "configs" / "attack_families_cicids2017.yaml",
        strict_schema=True,
    )

    files = loader.discover_files()
    print(f"\n[Step 1: Schema Validation]")
    print(f"DEV DATA, NOT FOR REPORTED RESULTS: Found {len(files)} CSV files in {raw_dir}")
    for f in files:
        print(f"  - {f.name}")

    # Step 1: Validate Schema on every file
    all_valid = True
    for f in files:
        df_head = pd.read_csv(f, nrows=5, encoding="cp1252")
        df_head.columns = [str(c).strip() for c in df_head.columns]
        if "Fwd Header Length.1" in df_head.columns:
            df_head = df_head.drop(columns=["Fwd Header Length.1"])
        missing, unexpected = loader.validate_schema(df_head.columns)
        if missing or unexpected:
            print(f"  FAILED on {f.name}: missing={missing}, unexpected={unexpected}")
            all_valid = False
        else:
            print(f"  PASSED schema validation: {f.name}")

    print(f"Overall Schema Validation Result: {'PASSED' if all_valid else 'FAILED'}")

    # Load all files
    print("\n[Loading All 8 CSV Files...]")
    raw_df = loader.load_raw()
    print(f"DEV DATA, NOT FOR REPORTED RESULTS: Total Loaded Rows: {len(raw_df):,}, Columns: {len(raw_df.columns)}")

    # Step 2: Distinct Labels and Unmapped Labels
    print("\n[Step 2: Distinct Labels & Attack Family Mapping]")
    label_col = "Label"
    raw_labels = raw_df[label_col].astype(str).str.strip()
    counts = raw_labels.value_counts()
    print("DEV DATA, NOT FOR REPORTED RESULTS: Distinct Raw Labels Found:")
    for lbl, cnt in counts.items():
        print(f"  - {repr(lbl)}: {cnt:,} rows")

    cleaner = loader.cleaner
    unmapped = []
    for lbl in counts.index:
        if (
            lbl not in cleaner.family_mapping
            and lbl.lower() not in cleaner.family_mapping
        ):
            # check normalize
            norm = cleaner._normalize_columns(pd.DataFrame({"x": [lbl]}))
            norm_str = str(lbl).replace("\ufffd", "-").replace("–", "-").replace("—", "-").strip()
            if norm_str not in cleaner.family_mapping and norm_str.lower() not in cleaner.family_mapping:
                unmapped.append(lbl)

    print("\nDEV DATA, NOT FOR REPORTED RESULTS: Unmapped Labels:")
    if unmapped:
        print(f"  Unmapped labels detected: {unmapped}")
    else:
        print("  None. All distinct labels are cleanly mapped in configs/attack_families_cicids2017.yaml.")

    # Step 3: Run inspect_negative_values()
    print("\n[Step 3: Negative Value Inspection (Full Per-Column, Per-Label Table)]")
    print("DEV DATA, NOT FOR REPORTED RESULTS: Inspecting negative values (NO rows dropped yet)...")
    neg_table = inspect_negative_values(raw_df, label_col="Label", print_summary=True)

    # Step 4: Run Cleaning with drop_negative_artifacts=True
    print("\n[Step 4: Dataset Cleaning with drop_negative_artifacts=True]")
    cleaner.drop_negative_artifacts = True
    cleaned_df, report = cleaner.clean(raw_df, dataset_name="CIC-IDS2017 (corrected)", print_negative_summary=False)

    print("\nDEV DATA, NOT FOR REPORTED RESULTS: Cleaning Report:")
    print(f"  - Raw rows total: {report.raw_rows_total:,}")
    print(f"  - Duplicate rows dropped: {report.duplicate_rows_dropped:,}")
    print(f"  - Inf / NaN rows dropped: {report.inf_nan_rows_dropped:,}")
    print(f"  - Negative value rows dropped: {report.negative_value_rows_dropped:,} (Flow Duration < 0 corrupt flows)")
    print(f"  - Total rows dropped: {report.total_rows_dropped:,}")
    print(f"  - Clean rows total: {report.clean_rows_total:,}")
    print(f"  - Identifier columns removed: {report.identifier_columns_removed}")
    print(f"  - Features retained count: {report.features_retained_count}")
    print(f"\nDEV DATA, NOT FOR REPORTED RESULTS: Clipped Physical Metric Columns ({len(report.clipped_columns)}):")
    for col, count in sorted(report.clipped_columns.items()):
        print(f"  - {col}: {count:,} values clipped to 0")

    print("\nDEV DATA, NOT FOR REPORTED RESULTS: Rows Dropped Per Label Class:")
    for lbl, dropped_cnt in sorted(report.rows_dropped_per_label.items(), key=lambda x: -x[1]):
        before = report.label_distribution_before.get(lbl, 0)
        after = report.label_distribution_after.get(lbl, 0)
        pct = (dropped_cnt / before * 100.0) if before > 0 else 0.0
        print(f"  - {lbl}: before={before:,}, after={after:,}, dropped={dropped_cnt:,} ({pct:.2f}%)")

    # Save cleaned data to disk so experiments can quickly load it
    cache_path = REPO_ROOT / "data" / "processed" / "cicids2017_cleaned_dev.parquet"
    report_path = REPO_ROOT / "data" / "processed" / "cleaning_report_partA.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[Saving cleaned dataframe to {cache_path}...]")
    cleaned_df.to_parquet(cache_path, index=False)
    print(f"[Saving cleaning report to {report_path}...]")
    report.save_json(report_path)
    print("DEV DATA, NOT FOR REPORTED RESULTS: Cleaned dataset and report successfully saved.")


if __name__ == "__main__":
    run_validation()
