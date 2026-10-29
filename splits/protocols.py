"""Evaluation protocol split generators for P1, P2, and P4.

Source of Truth: SPEC.md (Section 3, Section 8 Module 2)

Global Rules applied:
1. Fit ALL scalers/encoders/PCA on TRAIN data only. Never touch TEST statistics.
2. Thresholds and hyperparameters use VAL-BENIGN only.
3. Explicit seed in every split generator for deterministic reproducibility.
4. Leakage-free protocols:
   - TRAIN: benign (+ injected contamination for P4).
   - VAL-BENIGN: held-out benign only.
   - TEST: benign + attacks.
   - P1 (LOFO): test family F never appears in TRAIN unless contamination variant is explicitly enabled.
   - P2 (Temporal): train/val on Monday benign, test on later days with no day-shuffling.
   - P4 (Contamination):
     * Total TRAIN size N is IDENTICAL for every contamination level c in {0, 1, 2, 5, 10}.
     * N is a fixed multiple of 100.
     * TRAIN = (N * (1 - c/100)) benign + (N * c/100) attack rows.
     * Benign rows for larger c are a strict nested subset of benign rows for smaller c.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class SplitIndices:
    """Container for split indices and metadata."""
    protocol: str
    seed: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    family: Optional[str] = None
    contamination_rate: float = 0.0
    include_test_family_in_contamination: bool = False
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "seed": self.seed,
            "family": self.family,
            "contamination_rate": float(self.contamination_rate),
            "include_test_family_in_contamination": self.include_test_family_in_contamination,
            "train_indices": [int(x) for x in self.train_indices],
            "val_indices": [int(x) for x in self.val_indices],
            "test_indices": [int(x) for x in self.test_indices],
            "counts": {
                "train": len(self.train_indices),
                "val": len(self.val_indices),
                "test": len(self.test_indices),
            },
            "metadata": self.metadata or {},
        }

    def save(self, filepath: Union[str, Path]) -> Path:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix.lower() == ".json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif filepath.suffix.lower() in [".npz"]:
            np.savez_compressed(
                filepath,
                train_indices=self.train_indices,
                val_indices=self.val_indices,
                test_indices=self.test_indices,
                metadata=json.dumps(
                    {
                        "protocol": self.protocol,
                        "seed": self.seed,
                        "family": self.family,
                        "contamination_rate": self.contamination_rate,
                        "include_test_family_in_contamination": self.include_test_family_in_contamination,
                        "metadata": self.metadata or {},
                    }
                ),
            )
        elif filepath.suffix.lower() == ".parquet":
            df_parts = pd.DataFrame(
                {
                    "row_index": np.concatenate(
                        [self.train_indices, self.val_indices, self.test_indices]
                    ),
                    "split_role": (
                        ["TRAIN"] * len(self.train_indices)
                        + ["VAL_BENIGN"] * len(self.val_indices)
                        + ["TEST"] * len(self.test_indices)
                    ),
                }
            )
            df_parts.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        return filepath


def load_splits(filepath: Union[str, Path]) -> SplitIndices:
    """Load split indices from disk."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Split file not found: {filepath}")

    if filepath.suffix.lower() == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SplitIndices(
            protocol=data["protocol"],
            seed=data["seed"],
            family=data.get("family"),
            contamination_rate=data.get("contamination_rate", 0.0),
            include_test_family_in_contamination=data.get(
                "include_test_family_in_contamination", False
            ),
            train_indices=np.array(data["train_indices"], dtype=np.int64),
            val_indices=np.array(data["val_indices"], dtype=np.int64),
            test_indices=np.array(data["test_indices"], dtype=np.int64),
            metadata=data.get("metadata", {}),
        )
    elif filepath.suffix.lower() == ".npz":
        data = np.load(filepath, allow_pickle=True)
        meta = json.loads(str(data["metadata"]))
        return SplitIndices(
            protocol=meta["protocol"],
            seed=meta["seed"],
            family=meta.get("family"),
            contamination_rate=meta.get("contamination_rate", 0.0),
            include_test_family_in_contamination=meta.get(
                "include_test_family_in_contamination", False
            ),
            train_indices=data["train_indices"],
            val_indices=data["val_indices"],
            test_indices=data["test_indices"],
            metadata=meta.get("metadata", {}),
        )
    elif filepath.suffix.lower() == ".parquet":
        df = pd.read_parquet(filepath)
        train_idx = df.loc[df["split_role"] == "TRAIN", "row_index"].to_numpy(dtype=np.int64)
        val_idx = df.loc[df["split_role"] == "VAL_BENIGN", "row_index"].to_numpy(dtype=np.int64)
        test_idx = df.loc[df["split_role"] == "TEST", "row_index"].to_numpy(dtype=np.int64)
        return SplitIndices(
            protocol="Unknown",
            seed=0,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
        )
    else:
        raise ValueError(f"Unsupported split format: {filepath.suffix}")


def save_splits(
    splits: Union[SplitIndices, Dict[str, SplitIndices]],
    output_dir: Union[str, Path],
    file_format: str = "json",
) -> List[Path]:
    """Save split indices to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    if isinstance(splits, SplitIndices):
        splits_dict = {f"{splits.protocol}_{splits.family or 'default'}": splits}
    else:
        splits_dict = splits

    for name, split in splits_dict.items():
        fname = f"{name}_seed{split.seed}.{file_format}"
        p = split.save(output_dir / fname)
        saved_paths.append(p)

    return saved_paths


def generate_p1_lofo_splits(
    df: pd.DataFrame,
    seed: int,
    val_benign_ratio: float = 0.2,
    test_benign_ratio: float = 0.2,
    target_families: Optional[List[str]] = None,
) -> Dict[str, SplitIndices]:
    """Generate Protocol P1 (Leave-One-Family-Out) splits.
    
    Per SPEC.md Section 3:
    "For each attack family F: train on benign only;
     test on benign test split + family F only.
     Report per-family metrics, then macro-average."
    """
    if "Attack_Family" not in df.columns:
        raise ValueError("DataFrame must contain 'Attack_Family' column.")

    rng = np.random.default_rng(seed)

    benign_indices = df.index[df["Attack_Family"] == "Benign"].to_numpy()
    if len(benign_indices) == 0:
        raise ValueError("No benign samples found in dataset.")

    shuffled_benign = rng.permutation(benign_indices)
    n_benign = len(shuffled_benign)

    n_val = int(round(n_benign * val_benign_ratio))
    n_test = int(round(n_benign * test_benign_ratio))
    n_train = n_benign - n_val - n_test

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test} "
            f"for total benign={n_benign}"
        )

    benign_train_idx = shuffled_benign[:n_train]
    benign_val_idx = shuffled_benign[n_train : n_train + n_val]
    benign_test_idx = shuffled_benign[n_train + n_val :]

    all_families = [
        f for f in df["Attack_Family"].unique() if f != "Benign" and f != "Unknown_Attack"
    ]
    if target_families is not None:
        all_families = [f for f in all_families if f in target_families]

    lofo_splits: Dict[str, SplitIndices] = {}
    for family in sorted(all_families):
        family_attack_idx = df.index[df["Attack_Family"] == family].to_numpy()
        test_combined = np.concatenate([benign_test_idx, family_attack_idx])

        split = SplitIndices(
            protocol="P1",
            seed=seed,
            family=family,
            train_indices=np.sort(benign_train_idx),
            val_indices=np.sort(benign_val_idx),
            test_indices=np.sort(test_combined),
            contamination_rate=0.0,
            include_test_family_in_contamination=False,
            metadata={
                "n_benign_train": len(benign_train_idx),
                "n_benign_val": len(benign_val_idx),
                "n_benign_test": len(benign_test_idx),
                "n_family_attack_test": len(family_attack_idx),
            },
        )
        lofo_splits[family] = split

    return lofo_splits


def generate_p2_temporal_splits(
    df: pd.DataFrame,
    seed: int,
    val_benign_ratio: float = 0.2,
    train_days: Optional[List[str]] = None,
    test_days: Optional[List[str]] = None,
) -> SplitIndices:
    """Generate Protocol P2 (Temporal Split) without cross-day shuffling.
    
    Per SPEC.md Section 3:
    "CIC-IDS2017 is organized by day. Train on Monday benign (+ optionally early-week benign),
     validate on held-out benign, test on later days with attacks. No shuffling across days."
    """
    if "Day" not in df.columns:
        raise ValueError("DataFrame must contain 'Day' column for temporal protocol P2.")
    if "Attack_Family" not in df.columns:
        raise ValueError("DataFrame must contain 'Attack_Family' column.")

    train_days = train_days or ["Monday"]
    test_days = test_days or ["Tuesday", "Wednesday", "Thursday", "Friday"]

    rng = np.random.default_rng(seed)

    train_day_mask = df["Day"].isin(train_days) & (df["Attack_Family"] == "Benign")
    train_day_benign_indices = df.index[train_day_mask].to_numpy()

    if len(train_day_benign_indices) == 0:
        raise ValueError(f"No benign traffic found for training days: {train_days}")

    shuffled_train_benign = rng.permutation(train_day_benign_indices)
    n_total_train_day = len(shuffled_train_benign)
    n_val = int(round(n_total_train_day * val_benign_ratio))
    n_train = n_total_train_day - n_val

    train_idx = shuffled_train_benign[:n_train]
    val_idx = shuffled_train_benign[n_train:]

    test_day_mask = df["Day"].isin(test_days)
    test_idx = df.index[test_day_mask].to_numpy()

    if len(test_idx) == 0:
        raise ValueError(f"No traffic found for test days: {test_days}")

    return SplitIndices(
        protocol="P2",
        seed=seed,
        family="Temporal_All",
        train_indices=np.sort(train_idx),
        val_indices=np.sort(val_idx),
        test_indices=np.sort(test_idx),
        contamination_rate=0.0,
        include_test_family_in_contamination=False,
        metadata={
            "train_days": train_days,
            "test_days": test_days,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
        },
    )


def generate_p4_contamination_splits(
    df: pd.DataFrame,
    seed: int,
    lofo_family: str,
    contamination_rate: float,
    n_train_total: Optional[int] = None,
    include_test_family: bool = False,
    val_benign_ratio: float = 0.2,
    test_benign_ratio: float = 0.2,
) -> SplitIndices:
    """Generate Protocol P4 (Contamination Injection) under P1 LOFO.
    
    Rigorous Leakage & Confound Controls:
    1. Total TRAIN size N is IDENTICAL for every contamination level c in {0, 1, 2, 5, 10}.
    2. N is a fixed multiple of 100.
    3. TRAIN = (N * (1 - c/100)) benign + (N * c/100) attack rows.
    4. Benign rows for larger c are a strict nested subset of benign rows for smaller c
       (nested from the same seed-ordered benign permutation).
    5. Held-out family F never appears in TRAIN unless include_test_family is True.
    """
    if "Attack_Family" not in df.columns:
        raise ValueError("DataFrame must contain 'Attack_Family' column.")

    if not (0.0 <= contamination_rate < 1.0):
        raise ValueError(f"contamination_rate must be in [0.0, 1.0), got {contamination_rate}")

    rng = np.random.default_rng(seed)

    # Partition benign flows into train pool, val-benign, test-benign
    benign_indices = df.index[df["Attack_Family"] == "Benign"].to_numpy()
    shuffled_benign = rng.permutation(benign_indices)
    n_benign = len(shuffled_benign)

    n_val = int(round(n_benign * val_benign_ratio))
    n_test = int(round(n_benign * test_benign_ratio))
    benign_train_pool = shuffled_benign[: n_benign - n_val - n_test]
    benign_val_idx = shuffled_benign[n_benign - n_val - n_test : n_benign - n_test]
    benign_test_idx = shuffled_benign[n_benign - n_test :]

    pool_len = len(benign_train_pool)

    # Determine fixed N as a multiple of 100
    if n_train_total is not None:
        N = n_train_total
        if N % 100 != 0:
            raise ValueError(f"n_train_total must be a fixed multiple of 100, got {N}")
        if N > pool_len:
            raise ValueError(
                f"Requested n_train_total={N} exceeds available benign training pool size {pool_len}"
            )
    else:
        N = (pool_len // 100) * 100
        if N == 0:
            raise ValueError(
                f"Benign training pool ({pool_len}) too small to form a fixed multiple of 100"
            )

    # Compute exact row counts: TRAIN = (N * (1 - c/100)) benign + (N * c/100) attack
    pct = int(round(contamination_rate * 100))
    n_attack = (N * pct) // 100
    n_benign_train = N - n_attack

    # Nested benign selection: always taking the first n_benign_train from benign_train_pool
    # Ensures larger c uses a strict subset of benign rows used by smaller c
    selected_benign = benign_train_pool[:n_benign_train]

    # Select attack contamination rows
    if include_test_family:
        contam_candidates = df.index[
            (df["Attack_Family"] != "Benign") & (df["Attack_Family"] != "Unknown_Attack")
        ].to_numpy()
    else:
        # Headline variant: strictly exclude test family from candidate pool
        contam_candidates = df.index[
            (df["Attack_Family"] != "Benign")
            & (df["Attack_Family"] != "Unknown_Attack")
            & (df["Attack_Family"] != lofo_family)
        ].to_numpy()

    if n_attack > 0:
        if len(contam_candidates) < n_attack:
            raise ValueError(
                f"Requested {n_attack} contamination flows, but only {len(contam_candidates)} "
                f"available in candidate pool (excluding {lofo_family if not include_test_family else 'none'})."
            )
        # Use separate deterministic seed branch for attack permutation
        rng_attack = np.random.default_rng(seed + 9999)
        shuffled_attacks = rng_attack.permutation(contam_candidates)
        selected_attack = shuffled_attacks[:n_attack]
        train_idx = np.concatenate([selected_benign, selected_attack])
    else:
        selected_attack = np.array([], dtype=np.int64)
        train_idx = selected_benign

    # Invariant: len(train_idx) must equal N exactly
    assert len(train_idx) == N, f"Internal error: len(train_idx)={len(train_idx)} != N={N}"

    # Test set: Benign test split + test family F only (P1 rule)
    family_attack_idx = df.index[df["Attack_Family"] == lofo_family].to_numpy()
    test_idx = np.concatenate([benign_test_idx, family_attack_idx])

    return SplitIndices(
        protocol="P4",
        seed=seed,
        family=lofo_family,
        train_indices=np.sort(train_idx),
        val_indices=np.sort(benign_val_idx),
        test_indices=np.sort(test_idx),
        contamination_rate=contamination_rate,
        include_test_family_in_contamination=include_test_family,
        metadata={
            "fixed_N": N,
            "n_train_total": len(train_idx),
            "n_benign_train": len(selected_benign),
            "n_injected_contam": len(selected_attack),
            "actual_contam_rate": len(selected_attack) / len(train_idx),
            "n_benign_val": len(benign_val_idx),
            "n_test_total": len(test_idx),
            "n_family_attack_test": len(family_attack_idx),
        },
    )


def generate_p4_setting_b_splits(
    df: pd.DataFrame,
    seed: int,
    family: str,
    contamination_rate: float,
    n_train_total: int = 200000,
    val_benign_ratio: float = 0.2,
    test_benign_ratio: float = 0.2,
) -> SplitIndices:
    """Generate Protocol P4 Setting B splits (contamination drawn ONLY from F itself).
    
    Rigorous Leakage Controls:
    1. F's attack rows are partitioned 50/50 using a seeded permutation into:
       - f_contam_pool (50%)
       - f_test_pool (50%)
    2. Contamination rows are drawn ONLY from f_contam_pool.
    3. TEST evaluates on all test benign rows + f_test_pool.
    4. Guarantees zero overlap between TRAIN and TEST: no F flow is ever in both.
    5. Benign train rows are nested across contamination levels c identical to Setting A.
    """
    if "Attack_Family" not in df.columns:
        raise ValueError("DataFrame must contain 'Attack_Family' column.")

    if not (0.0 <= contamination_rate < 1.0):
        raise ValueError(f"contamination_rate must be in [0.0, 1.0), got {contamination_rate}")

    rng = np.random.default_rng(seed)

    # 1. Benign partition identical to Setting A
    benign_indices = df.index[df["Attack_Family"] == "Benign"].to_numpy()
    shuffled_benign = rng.permutation(benign_indices)
    n_benign = len(shuffled_benign)

    n_val = int(round(n_benign * val_benign_ratio))
    n_test = int(round(n_benign * test_benign_ratio))
    benign_train_pool = shuffled_benign[: n_benign - n_val - n_test]
    benign_val_idx = shuffled_benign[n_benign - n_val - n_test : n_benign - n_test]
    benign_test_idx = shuffled_benign[n_benign - n_test :]

    N = n_train_total
    pct = int(round(contamination_rate * 100))
    n_attack = (N * pct) // 100
    n_benign_train = N - n_attack

    selected_benign = benign_train_pool[:n_benign_train]

    # 2. Split F's rows 50/50 into disjoint contamination and test pools
    f_indices = df.index[df["Attack_Family"] == family].to_numpy()
    if len(f_indices) == 0:
        raise ValueError(f"No samples found for family {family}")

    rng_f = np.random.default_rng(seed + 7777)
    shuffled_f = rng_f.permutation(f_indices)
    split_pt = len(shuffled_f) // 2
    f_contam_pool = shuffled_f[:split_pt]
    f_test_pool = shuffled_f[split_pt:]

    if n_attack > 0:
        if len(f_contam_pool) < n_attack:
            raise ValueError(
                f"Requested {n_attack} contamination flows from {family}, but contamination pool only has {len(f_contam_pool)}"
            )
        # Nested selection from f_contam_pool
        selected_attack = f_contam_pool[:n_attack]
        train_idx = np.concatenate([selected_benign, selected_attack])
    else:
        selected_attack = np.array([], dtype=np.int64)
        train_idx = selected_benign

    assert len(train_idx) == N
    test_idx = np.concatenate([benign_test_idx, f_test_pool])

    # Zero leakage check
    overlap = set(train_idx).intersection(set(test_idx))
    assert len(overlap) == 0, f"Leakage detected between TRAIN and TEST in Setting B: {len(overlap)} shared indices!"

    return SplitIndices(
        protocol="P4_Setting_B",
        seed=seed,
        family=family,
        train_indices=np.sort(train_idx),
        val_indices=np.sort(benign_val_idx),
        test_indices=np.sort(test_idx),
        contamination_rate=contamination_rate,
        include_test_family_in_contamination=True,
        metadata={
            "setting": "B",
            "fixed_N": N,
            "n_train_total": len(train_idx),
            "n_benign_train": len(selected_benign),
            "n_injected_contam": len(selected_attack),
            "actual_contam_rate": len(selected_attack) / len(train_idx),
            "n_benign_val": len(benign_val_idx),
            "n_test_total": len(test_idx),
            "n_family_attack_test": len(f_test_pool),
            "n_family_attack_contam_pool": len(f_contam_pool),
        },
    )

