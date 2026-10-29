"""Summarize splits for protocols P1, P2, and P4.

Source of Truth: SPEC.md Section 3 & User Request Task 6.
Prints tabular summaries of row counts per role (TRAIN / VAL-BENIGN / TEST)
per attack family, and per day for Protocol P2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from data.cicids2017 import CICIDS2017Cleaner, CICIDS2017Dataset
from splits.protocols import (
    generate_p1_lofo_splits,
    generate_p2_temporal_splits,
    generate_p4_contamination_splits,
)


def create_synthetic_demo_data() -> pd.DataFrame:
    """Fallback synthetic dataset matching CIC-IDS2017 structure."""
    rng = np.random.default_rng(42)
    n_rows = 6000
    days = (
        ["Monday"] * 2400
        + ["Tuesday"] * 900
        + ["Wednesday"] * 900
        + ["Thursday"] * 900
        + ["Friday"] * 900
    )
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
                labels.append("FTP-Patator")
                families.append("BruteForce")
        elif day == "Wednesday":
            if rng.random() < 0.6:
                labels.append("BENIGN")
                families.append("Benign")
            else:
                labels.append("DoS Hulk")
                families.append("DoS/DDoS")
        elif day == "Thursday":
            if rng.random() < 0.6:
                labels.append("BENIGN")
                families.append("Benign")
            else:
                labels.append("Web Attack – Brute Force")
                families.append("WebAttack")
        elif day == "Friday":
            if rng.random() < 0.6:
                labels.append("BENIGN")
                families.append("Benign")
            else:
                labels.append("PortScan")
                families.append("PortScan")

    df = pd.DataFrame({
        "Flow Duration": rng.uniform(10.0, 50000.0, size=n_rows),
        "Total Fwd Packets": rng.integers(1, 50, size=n_rows).astype(float),
        "Flow Bytes/s": rng.uniform(10.0, 50000.0, size=n_rows),
        "Day": days,
        "Label": labels,
        "Attack_Family": families,
    })
    return df


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Render a formatted ASCII table."""
    print(f"\n{title}")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)))


def summarize(data_path: Optional[str] = None, seed: int = 42) -> None:
    if data_path and Path(data_path).exists():
        p = Path(data_path)
        if p.suffix.lower() == ".parquet":
            print(f"\nDEV DATA, NOT FOR REPORTED RESULTS: Loading pre-cleaned data from {data_path}...")
            df = pd.read_parquet(p)
        else:
            print(f"\nDEV DATA, NOT FOR REPORTED RESULTS: Loading and cleaning data from {data_path}...")
            loader = CICIDS2017Dataset(data_path)
            df, report = loader.load_and_clean()
    else:
        print("[Summarize] Real data path not provided or not found; generating synthetic sample dataset.")
        df = create_synthetic_demo_data()

    print(f"DEV DATA, NOT FOR REPORTED RESULTS: Total dataset rows: {len(df):,}")
    print(f"DEV DATA, NOT FOR REPORTED RESULTS: Attack Family breakdown:")
    for fam, cnt in df["Attack_Family"].value_counts().items():
        print(f"  - {fam}: {cnt:,} rows")

    # 1. PROTOCOL P1 SUMMARY
    p1_splits = generate_p1_lofo_splits(df, seed=seed)
    p1_rows = []
    for family, split in sorted(p1_splits.items()):
        test_df = df.loc[split.test_indices]
        test_benign_cnt = int((test_df["Attack_Family"] == "Benign").sum())
        test_attack_cnt = int((test_df["Attack_Family"] == family).sum())
        p1_rows.append([
            family,
            str(len(split.train_indices)),
            str(len(split.val_indices)),
            str(test_benign_cnt),
            str(test_attack_cnt),
            str(len(split.test_indices)),
        ])

    print_table(
        title="=== [DEV DATA, NOT FOR REPORTED RESULTS] Protocol P1: Leave-One-Family-Out (LOFO) Split Summary ===",
        headers=["Target Family", "TRAIN (Benign)", "VAL-BENIGN", "TEST (Benign)", "TEST (Family Attack)", "Total TEST"],
        rows=p1_rows,
    )

    # 2. PROTOCOL P2 SUMMARY (Temporal by Day)
    p2_split = generate_p2_temporal_splits(df, seed=seed)
    train_df = df.loc[p2_split.train_indices]
    val_df = df.loc[p2_split.val_indices]
    test_df = df.loc[p2_split.test_indices]

    p2_rows = [
        ["TRAIN", "Monday", str(len(train_df)), "Benign only (baseline)"],
        ["VAL-BENIGN", "Monday", str(len(val_df)), "Benign only (held-out)"],
    ]
    for day in ["Tuesday", "Wednesday", "Thursday", "Friday"]:
        day_sub = test_df[test_df["Day"] == day]
        if len(day_sub) > 0:
            attack_types = day_sub.loc[day_sub["Attack_Family"] != "Benign", "Attack_Family"].unique().tolist()
            att_str = ", ".join(attack_types) if attack_types else "None (Benign only)"
            p2_rows.append(["TEST", day, str(len(day_sub)), f"Benign + Attacks: [{att_str}]"])

    p2_rows.append(["TEST TOTAL", "Tue-Fri", str(len(test_df)), "All traffic on later days (temporal drift)"])

    print_table(
        title="=== [DEV DATA, NOT FOR REPORTED RESULTS] Protocol P2: Temporal Split (by Day) Summary ===",
        headers=["Role", "Day", "Flow Count", "Attack Families / Characteristics"],
        rows=p2_rows,
    )

    # 3. PROTOCOL P4 SUMMARY (Contamination Sweep)
    sample_family = list(p1_splits.keys())[0] if p1_splits else "DoS/DDoS"
    p4_rows = []
    levels = [0.0, 0.01, 0.02, 0.05, 0.10]
    previous_benign_indices = None
    for c in levels:
        split = generate_p4_contamination_splits(
            df,
            seed=seed,
            lofo_family=sample_family,
            contamination_rate=c,
            include_test_family=False,
        )
        meta = split.metadata
        benign_in_train = meta["n_benign_train"]
        attack_in_train = meta["n_injected_contam"]
        total_train = meta["n_train_total"]

        # Check nested subset property
        current_train_df = df.loc[split.train_indices]
        current_benign = set(current_train_df.index[current_train_df["Attack_Family"] == "Benign"])
        nested_check = "N/A (base c=0)"
        if previous_benign_indices is not None:
            is_subset = current_benign.issubset(previous_benign_indices)
            nested_check = "Strict Subset (Pass)" if is_subset else "FAIL (Not a subset)"
        previous_benign_indices = current_benign

        p4_rows.append([
            f"{int(c * 100)}%",
            str(total_train),
            str(benign_in_train),
            str(attack_in_train),
            f"{meta['actual_contam_rate']*100:.2f}%",
            str(len(split.val_indices)),
            str(len(split.test_indices)),
            nested_check,
        ])

    print_table(
        title=f"=== [DEV DATA, NOT FOR REPORTED RESULTS] Protocol P4: Contamination Sweep Summary (Held-out Family: {sample_family}) ===",
        headers=["c (%)", "Fixed Total TRAIN (N)", "TRAIN Benign", "TRAIN Injected", "Actual Contam %", "VAL-BENIGN", "Total TEST", "Benign Nesting Check"],
        rows=p4_rows,
    )
    print("\n[Summarize] All split summaries printed successfully.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize evaluation splits for P1, P2, and P4.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to cleaned dataset file/directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits.")
    args = parser.parse_args()
    summarize(args.data_path, args.seed)
