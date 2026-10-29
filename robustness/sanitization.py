"""Seed-Guided Sanitization Defense for Contaminated Anomaly Detection.

DEV DATA, NOT FOR REPORTED RESULTS
Source of Truth: User Request PART C.

Assumption:
A small TRUSTED clean benign "seed" set S exists (e.g., a verified baseline capture).

Pipeline:
1. Draw seed set S of n_seed clean benign flows (seeded), strictly disjoint from
   the contaminated pool, VAL-BENIGN, and TEST.
   Split S into S_fit (80%) and S_cal (20%).
2. Stage 1: Fit a preprocessor on S_fit ONLY. Train a stage-1 detector (AE or PCA)
   on S_fit with early stopping on S_cal reconstruction loss.
3. Score every row of the unlabeled contaminated pool with the stage-1 detector.
   Threshold = POT(S_cal scores, q = 0.01).
   Drop every pool row whose score is above the threshold.
4. Stage 2:
   - For AE: Fit a new preprocessor on (kept pool rows + S) and train the final AE on them,
     early stopping on VAL-BENIGN as before.
   - For PCA: Fit stage-2 preprocessor and train final PCA on kept pool rows.
5. Invariant: Labels are NEVER passed to or used within the sanitization method.
   Labels are only used afterwards to report what was removed.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from baselines.classical import PCAReconstructionBaseline
from data.preprocess import TabularPreprocessor
from models.autoencoders import MLPAutoencoder, train_autoencoder
from thresholds.pot import fit_pot_threshold


def draw_trusted_seed_set(
    df: pd.DataFrame,
    seed: int,
    n_seed: int = 10000,
    n_train_max: int = 200000,
    val_benign_ratio: float = 0.2,
    test_benign_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a trusted clean benign seed set S strictly disjoint from TRAIN pool, VAL, and TEST.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing 'Attack_Family' column.
    seed : int
        RNG seed matching the evaluation seed.
    n_seed : int, default 10000
        Number of clean benign flows in trusted seed set S.
    n_train_max : int, default 200000
        Maximum benign rows reserved for training pool across all c.
    val_benign_ratio : float, default 0.2
        Fraction of benign flows allocated to VAL-BENIGN.
    test_benign_ratio : float, default 0.2
        Fraction of benign flows allocated to TEST-BENIGN.
        
    Returns
    -------
    S_indices : np.ndarray
        Array of all n_seed indices in S.
    S_fit : np.ndarray
        80% subset of S for stage-1 fitting.
    S_cal : np.ndarray
        20% subset of S for stage-1 calibration / POT thresholding.
    """
    if "Attack_Family" not in df.columns:
        raise ValueError("DataFrame must contain 'Attack_Family' column.")

    rng = np.random.default_rng(seed)

    benign_indices = df.index[df["Attack_Family"] == "Benign"].to_numpy()
    shuffled_benign = rng.permutation(benign_indices)
    n_benign = len(shuffled_benign)

    n_val = int(round(n_benign * val_benign_ratio))
    n_test = int(round(n_benign * test_benign_ratio))
    benign_train_pool_len = n_benign - n_val - n_test

    if n_train_max + n_seed > benign_train_pool_len:
        raise ValueError(
            f"Requested n_train_max={n_train_max} + n_seed={n_seed} ({n_train_max + n_seed}) "
            f"exceeds available benign train pool ({benign_train_pool_len})."
        )

    # Draw S immediately after the first n_train_max benign pool rows
    S_indices = shuffled_benign[n_train_max : n_train_max + n_seed]

    # Split S into S_fit (80%) and S_cal (20%)
    n_fit = int(round(0.80 * n_seed))
    S_fit = S_indices[:n_fit]
    S_cal = S_indices[n_fit:]

    return S_indices, S_fit, S_cal


def filter_pool_with_seed(
    X_pool_raw: pd.DataFrame,
    X_seed_fit_raw: pd.DataFrame,
    X_seed_cal_raw: pd.DataFrame,
    model_type: str = "ae",
    q: float = 0.01,
    seed: int = 0,
    skew_threshold: float = 3.0,
    clip_value: float = 10.0,
    max_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 0.001,
    patience: int = 3,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, Any, TabularPreprocessor]:
    """Filter an unlabeled contaminated pool using a trusted clean seed set S.
    
    RULE: Receives ONLY feature DataFrames. No label or target information is accepted.
    
    Parameters
    ----------
    X_pool_raw : pd.DataFrame
        Unlabeled features of the contaminated pool (shape N x D).
    X_seed_fit_raw : pd.DataFrame
        Features of the trusted clean S_fit set (shape N_fit x D).
    X_seed_cal_raw : pd.DataFrame
        Features of the trusted clean S_cal set (shape N_cal x D).
    model_type : str, default "ae"
        Detector type: "ae" (MLP Autoencoder) or "pca" (PCA Reconstruction).
    q : float, default 0.01
        Target POT exceedance probability on calibration scores.
    seed : int, default 0
        RNG seed for model initialization.
        
    Returns
    -------
    keep_mask : np.ndarray of bool
        Boolean mask of shape (N,) indicating kept pool rows (score <= threshold).
    drop_mask : np.ndarray of bool
        Boolean mask of shape (N,) indicating dropped pool rows (score > threshold).
    threshold : float
        The POT estimated anomaly threshold z_q.
    stage1_model : Any
        Trained stage-1 detector.
    prep_stage1 : TabularPreprocessor
        Fitted stage-1 preprocessor (fit strictly on S_fit only).
    """
    if not isinstance(X_pool_raw, pd.DataFrame):
        raise TypeError("X_pool_raw must be a pandas DataFrame of features.")
    if not isinstance(X_seed_fit_raw, pd.DataFrame):
        raise TypeError("X_seed_fit_raw must be a pandas DataFrame of features.")
    if not isinstance(X_seed_cal_raw, pd.DataFrame):
        raise TypeError("X_seed_cal_raw must be a pandas DataFrame of features.")

    # Stage 1: Fit preprocessor on S_fit ONLY
    prep_stage1 = TabularPreprocessor(skew_threshold=skew_threshold, clip_value=clip_value)
    prep_stage1.fit(X_seed_fit_raw)

    X_s_fit = prep_stage1.transform(X_seed_fit_raw)
    X_s_cal = prep_stage1.transform(X_seed_cal_raw)
    X_pool = prep_stage1.transform(X_pool_raw)

    if model_type == "ae":
        stage1_ae = MLPAutoencoder(
            input_dim=X_s_fit.shape[1],
            hidden_dims=(64, 32, 16),
            bottleneck_dim=16,
            seed=seed,
        )
        trained_model, _ = train_autoencoder(
            stage1_ae,
            X_train=X_s_fit,
            X_val_benign=X_s_cal,
            seed=seed,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            trim_frac=0.0,
            warmup_epochs=0,
            verbose=verbose,
        )
        cal_scores = trained_model.compute_reconstruction_error(X_s_cal)
        pool_scores = trained_model.compute_reconstruction_error(X_pool)

    elif model_type == "pca":
        stage1_pca = PCAReconstructionBaseline(
            n_components=10,
            random_state=seed,
        )
        stage1_pca.fit(X_s_fit)
        cal_scores = stage1_pca.score(X_s_cal)
        pool_scores = stage1_pca.score(X_pool)
        trained_model = stage1_pca

    else:
        raise ValueError(f"Unknown model_type: '{model_type}', expected 'ae' or 'pca'.")

    # Estimate POT threshold on clean calibration scores
    threshold = fit_pot_threshold(cal_scores, q=q, percentile_u=90.0)

    # Filter contaminated pool
    keep_mask = pool_scores <= threshold
    drop_mask = pool_scores > threshold

    # Invariant: kept + dropped == pool size
    assert len(keep_mask) == len(X_pool_raw), "Mask length mismatch with pool size."
    assert np.sum(keep_mask) + np.sum(drop_mask) == len(X_pool_raw), "Sum of kept and dropped rows must equal pool size."

    return keep_mask, drop_mask, threshold, trained_model, prep_stage1
