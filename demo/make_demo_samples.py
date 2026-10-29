"""Build sample flow CSVs for the live demo from held-out rows.

DEV DATA, NOT FOR REPORTED RESULTS

This script constructs:
1. `demo/demo_samples.csv`: 700 raw flow feature rows (no labels).
   Composed of:
   - 300 held-out Benign flows (strictly NOT in TRAIN or VAL).
   - 100 DoS/DDoS attack flows.
   - 100 PortScan attack flows.
   - 100 BruteForce attack flows.
   - 100 WebAttack attack flows.
2. `demo/demo_samples_labels.csv`: Ground-truth labels and attack families
   saved separately for optional evaluation display.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from splits.protocols import generate_p1_lofo_splits

STATUS_TAG = "DEV DATA, NOT FOR REPORTED RESULTS"


def make_demo_samples(
    data_path: Path = REPO_ROOT / "data" / "processed" / "cicids2017_cleaned_dev.parquet",
    output_dir: Path = REPO_ROOT / "demo",
    seed: int = 0,
) -> None:
    print("=" * 80)
    print(f"[{STATUS_TAG}]")
    print("BUILDING DEMO FLOW SAMPLES FROM HELD-OUT ROWS")
    print("=" * 80)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {data_path}. Please verify data directory."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {data_path} ...")
    t0 = time.time()
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} flows in {time.time() - t0:.2f}s")

    feature_cols = [
        c for c in df.columns if c not in ["Label", "Attack_Family", "Day", "Destination Port"]
    ]
    print(f"Using {len(feature_cols)} raw flow feature columns.")

    # Generate P1 splits to isolate held-out benign rows
    print("Partitioning benign flows via P1 protocol (seed=0) ...")
    splits = generate_p1_lofo_splits(
        df,
        seed=seed,
        val_benign_ratio=0.2,
        test_benign_ratio=0.2,
        target_families=["DoS/DDoS"],
    )
    ref_split = splits["DoS/DDoS"]
    train_indices = set(ref_split.train_indices)
    val_indices = set(ref_split.val_indices)

    # Benign rows strictly excluded from both TRAIN and VAL
    benign_all = df.index[df["Attack_Family"] == "Benign"].to_numpy()
    held_out_benign = np.array(
        [idx for idx in benign_all if idx not in train_indices and idx not in val_indices],
        dtype=np.int64,
    )
    print(f"Total held-out Benign flows available: {len(held_out_benign):,}")

    rng = np.random.default_rng(seed)

    # 1. Select 300 benign flows
    n_benign = min(300, len(held_out_benign))
    sampled_benign = rng.choice(held_out_benign, size=n_benign, replace=False)

    # 2. Select 100 flows each from 4 attack families
    attack_families = ["DoS/DDoS", "PortScan", "BruteForce", "WebAttack"]
    sampled_attacks = {}
    for fam in attack_families:
        fam_idx = df.index[df["Attack_Family"] == fam].to_numpy()
        if len(fam_idx) == 0:
            raise ValueError(f"No rows found for attack family: {fam}")
        n_fam = min(100, len(fam_idx))
        chosen = rng.choice(fam_idx, size=n_fam, replace=False)
        sampled_attacks[fam] = chosen
        print(f"Sampled {len(chosen)} flows for {fam} (available: {len(fam_idx):,})")

    # Combine all sample indices: 300 Benign + 400 Attacks = 700 total flows
    all_sampled_indices = np.concatenate(
        [sampled_benign] + [sampled_attacks[fam] for fam in attack_families]
    )
    print(f"\nTotal demo sample flows selected: {len(all_sampled_indices):,}")

    # Extract raw feature columns only (no labels, no attack family)
    demo_samples_df = df.loc[all_sampled_indices, feature_cols].copy().reset_index(drop=True)
    samples_out_path = output_dir / "demo_samples.csv"
    demo_samples_df.to_csv(samples_out_path, index=False)
    print(f"Saved raw feature samples to: {samples_out_path}")
    print(f"   Shape: {demo_samples_df.shape} | Labels present: {'Label' in demo_samples_df.columns}")

    # Extract corresponding ground-truth labels into separate file
    labels_df = pd.DataFrame(
        {
            "flow_index": range(len(all_sampled_indices)),
            "Attack_Family": df.loc[all_sampled_indices, "Attack_Family"].values,
            "Label": df.loc[all_sampled_indices, "Label"].values,
            "is_attack": (df.loc[all_sampled_indices, "Attack_Family"].values != "Benign").astype(int),
        }
    )
    labels_out_path = output_dir / "demo_samples_labels.csv"
    labels_df.to_csv(labels_out_path, index=False)
    print(f"Saved ground-truth labels separately to: {labels_out_path}")
    print(f"   Label distribution:\n{labels_df['Attack_Family'].value_counts().to_string()}")

    print("\n" + "=" * 80)
    print(f"[{STATUS_TAG}] Demo samples created successfully.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"[{STATUS_TAG}] Build demo samples from held-out data."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(REPO_ROOT / "data" / "processed" / "cicids2017_cleaned_dev.parquet"),
        help="Path to processed parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "demo"),
        help="Directory to output demo CSVs.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random sampling seed.")
    args = parser.parse_args()

    make_demo_samples(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )
