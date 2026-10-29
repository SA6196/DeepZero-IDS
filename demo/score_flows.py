"""Score network flows and flag suspicious anomalies.

DEV DATA, NOT FOR REPORTED RESULTS

This module provides:
1. `score_flows(df)`: Computes reconstruction anomaly scores and threshold flags
   using the trained demo MLP Autoencoder. Labels are strictly forbidden and never accepted.
2. CLI interface: `python demo/score_flows.py flows.csv`
   Prints a scoring summary and writes `flows_scored.csv`.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocess import TabularPreprocessor
from models.autoencoders import MLPAutoencoder

STATUS_TAG = "DEV DATA, NOT FOR REPORTED RESULTS"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "demo" / "artifacts"

# Explicit label rejection: labels must NEVER be accepted or used
FORBIDDEN_LABEL_KEYWORDS = {
    "label",
    "attack",
    "attack_family",
    "target",
    "class",
    "ground_truth",
    "is_attack",
    "category",
}


class ScoredFlows(NamedTuple):
    """Container holding anomaly scores and binary suspicion flags."""
    scores: np.ndarray
    flags: np.ndarray

    def __iter__(self):
        yield self.scores
        yield self.flags


_CACHED_ARTIFACTS: dict = {}


def check_no_labels(df: pd.DataFrame) -> None:
    """Ensure no ground-truth label columns are passed into score_flows."""
    lowered_cols = {str(c).strip().lower() for c in df.columns}
    forbidden_present = lowered_cols.intersection(FORBIDDEN_LABEL_KEYWORDS)
    if forbidden_present:
        raise ValueError(
            f"[{STATUS_TAG}] Labels are never accepted or used in score_flows. "
            f"Detected forbidden label column(s): {sorted(forbidden_present)}. "
            f"Please supply raw unlabelled network flow features only."
        )


def load_demo_artifacts(
    artifacts_dir: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
) -> Tuple[MLPAutoencoder, TabularPreprocessor, float, dict]:
    """Load and cache the trained demo model, preprocessor, and POT threshold."""
    art_dir = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
    cache_key = str(art_dir.resolve())

    if cache_key in _CACHED_ARTIFACTS and not force_reload:
        return _CACHED_ARTIFACTS[cache_key]

    weights_path = art_dir / "model_weights.pt"
    preprocessor_path = art_dir / "preprocessor.joblib"
    threshold_path = art_dir / "threshold.json"
    config_path = art_dir / "model_config.json"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            f"Run 'python demo/train_demo_model.py' to produce artifacts."
        )
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Fitted preprocessor not found at {preprocessor_path}. "
            f"Run 'python demo/train_demo_model.py' to produce artifacts."
        )
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Threshold metadata not found at {threshold_path}. "
            f"Run 'python demo/train_demo_model.py' to produce artifacts."
        )

    # 1. Load fitted preprocessor
    preprocessor = TabularPreprocessor.load(preprocessor_path)

    # 2. Load threshold metadata
    with open(threshold_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    threshold = float(meta["threshold"])

    # 3. Model dimensions (load config if present, otherwise default to accepted MLP AE)
    hidden_dims = (64, 32)
    bottleneck_dim = 16
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            hidden_dims = tuple(cfg.get("hidden_dims", [64, 32]))
            bottleneck_dim = int(cfg.get("bottleneck_dim", 16))
        except Exception:
            pass

    input_dim = len(preprocessor.retained_cols_)
    model = MLPAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        bottleneck_dim=bottleneck_dim,
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    _CACHED_ARTIFACTS[cache_key] = (model, preprocessor, threshold, meta)
    return _CACHED_ARTIFACTS[cache_key]


def score_flows(
    df: pd.DataFrame,
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> ScoredFlows:
    """Score network flows and return anomaly scores and alarm flags.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing network flow features. Must not contain labels.
    artifacts_dir : Path or str, optional
        Custom directory containing model_weights.pt, preprocessor.joblib, and threshold.json.

    Returns
    -------
    ScoredFlows
        NamedTuple of (scores: np.ndarray, flags: np.ndarray).
        - scores: Mean squared reconstruction error per flow (higher = more anomalous).
        - flags: Boolean array (True = suspicious anomaly exceeding POT threshold).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df)}")

    # Strict invariant: labels are never accepted or used
    check_no_labels(df)

    if len(df) == 0:
        return ScoredFlows(scores=np.empty(0, dtype=np.float32), flags=np.empty(0, dtype=bool))

    model, preprocessor, threshold, _ = load_demo_artifacts(artifacts_dir=artifacts_dir)

    # Transform flows through the fitted pipeline
    X_scaled = preprocessor.transform(df)

    # Compute reconstruction anomaly scores
    scores = model.compute_reconstruction_error(X_scaled)
    scores = np.asarray(scores, dtype=np.float32)

    # Flag flows exceeding POT threshold
    flags = scores > threshold

    return ScoredFlows(scores=scores, flags=flags)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"[{STATUS_TAG}] Score network flows and flag suspicious anomalies."
    )
    parser.add_argument("csv_path", type=str, help="Path to input flows CSV file.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="flows_scored.csv",
        help="Path to output scored CSV file (default: flows_scored.csv).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Path to artifacts directory containing trained model and threshold.",
    )
    args = parser.parse_args()

    input_path = Path(args.csv_path)
    if not input_path.exists():
        print(f"Error: input file {input_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    print("=" * 75)
    print(f"[{STATUS_TAG}]")
    print(f"Scoring network flows from: {input_path}")
    print("=" * 75)

    df_raw = pd.read_csv(input_path)
    total_flows = len(df_raw)
    print(f"Loaded {total_flows:,} flows.")

    try:
        res = score_flows(df_raw, artifacts_dir=args.artifacts_dir)
    except ValueError as e:
        print(f"\nValidation Error: {e}", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as e:
        print(f"\nMissing Artifacts Error: {e}", file=sys.stderr)
        sys.exit(3)

    scores, flags = res.scores, res.flags
    flagged_count = int(np.sum(flags))
    pct_flagged = (flagged_count / max(1, total_flows)) * 100.0

    _, _, threshold, _ = load_demo_artifacts(artifacts_dir=args.artifacts_dir)

    print("\n" + "-" * 40 + " SCORING SUMMARY " + "-" * 40)
    print(f"Total flows scored: {total_flows:,}")
    print(f"Alarm threshold (POT q=0.01): {threshold:.6f}")
    print(f"Suspicious flows flagged: {flagged_count:,} ({pct_flagged:.2f}%)")
    print(f"Benign-rated flows: {total_flows - flagged_count:,} ({100.0 - pct_flagged:.2f}%)")
    if total_flows > 0:
        print(f"Anomaly Score Stats:")
        print(f"   Min:    {scores.min():.6f}")
        print(f"   25%:    {np.percentile(scores, 25):.6f}")
        print(f"   Median: {np.median(scores):.6f}")
        print(f"   75%:    {np.percentile(scores, 75):.6f}")
        print(f"   95%:    {np.percentile(scores, 95):.6f}")
        print(f"   Max:    {scores.max():.6f}")
    print("-" * 97)

    # Attach scores and flags to output dataframe
    df_out = df_raw.copy()
    df_out.insert(0, "flagged_anomaly", flags)
    df_out.insert(1, "anomaly_score", scores)

    out_path = Path(args.output)
    df_out.to_csv(out_path, index=False)
    print(f"Scored flows written to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
