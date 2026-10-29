"""Peaks-Over-Threshold (POT) Extreme Value Theory Thresholding.

DEV DATA, NOT FOR REPORTED RESULTS
Source of Truth: User Request PART B & Extreme Value Theory.

Formula:
Given calibration anomaly scores S:
1. Initial threshold u = percentile_u (default 90th percentile) of S.
2. Excesses Y = S[S > u] - u.
3. Fit Generalized Pareto Distribution (GPD) with location fixed at 0:
   Y ~ GPD(xi, sigma) via scipy.stats.genpareto.fit(Y, floc=0)
4. For target risk q (exceedance probability q, default 0.01):
   Let N = len(S), N_u = len(Y). Empirical exceedance rate p_u = N_u / N.
   If |xi| <= 1e-6 (exponential tail limit):
       z_q = u + sigma * ln(N_u / (q * N))
   Else:
       z_q = u + (sigma / xi) * [ ((q * N) / N_u)^(-xi) - 1 ]
"""

from __future__ import annotations

import warnings
from typing import Sequence, Union

import numpy as np
from scipy.stats import genpareto


def fit_pot_threshold(
    scores: Union[np.ndarray, Sequence[float]],
    q: float = 0.01,
    percentile_u: float = 90.0,
) -> float:
    """Compute label-free anomaly threshold via Peaks-Over-Threshold (POT).
    
    Parameters
    ----------
    scores : array-like of shape (N,)
        Calibration anomaly scores (higher = more anomalous). Must not contain labels or test data.
    q : float, default 0.01
        Target risk / false alarm exceedance probability (e.g. 0.01 for 1% tail).
    percentile_u : float, default 90.0
        Percentile used to set the initial extreme-value threshold u.
        
    Returns
    -------
    threshold : float
        The POT estimated anomaly threshold z_q.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    if scores_arr.ndim != 1:
        scores_arr = scores_arr.ravel()

    n = len(scores_arr)
    if n == 0:
        raise ValueError("scores array cannot be empty.")

    if not (0.0 < q < 1.0):
        raise ValueError(f"Target risk q must be in (0, 1), got {q}")

    if not (0.0 < percentile_u < 100.0):
        raise ValueError(f"percentile_u must be in (0, 100), got {percentile_u}")

    u = float(np.percentile(scores_arr, percentile_u))
    excesses = scores_arr[scores_arr > u] - u
    n_u = len(excesses)

    if n_u < 5:
        warnings.warn(
            f"Too few excesses ({n_u}) above initial threshold u={u:.4f} (N={n}). "
            f"Falling back to empirical percentile {100.0 * (1.0 - q):.1f}%."
        )
        return float(np.percentile(scores_arr, 100.0 * (1.0 - q)))

    # Fit GPD with location fixed at 0
    try:
        # scipy genpareto parameterization: c is shape xi, scale is sigma
        c_shape, _, scale = genpareto.fit(excesses, floc=0)
    except Exception as e:
        warnings.warn(f"GPD fit failed: {e}. Falling back to exponential tail fit.")
        scale = float(np.mean(excesses))
        c_shape = 0.0

    # Warning for extreme shape parameter
    if abs(c_shape) > 1.0:
        warnings.warn(
            f"GPD shape parameter xi={c_shape:.4f} is extreme (|xi| > 1.0). "
            f"Estimated threshold may have high variance."
        )

    # Standard POT quantile formula
    # P(X > z_q) = (N_u / N) * [1 + xi * (z_q - u) / sigma]^(-1/xi) = q
    ratio = (q * n) / n_u  # q / (N_u / N)

    if abs(c_shape) <= 1e-6:
        # Exponential tail limit as xi -> 0
        threshold = u + scale * np.log(n_u / (q * n))
    else:
        threshold = u + (scale / c_shape) * (ratio ** (-c_shape) - 1.0)

    return float(threshold)
