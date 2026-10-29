"""Unit tests for Peaks-Over-Threshold (POT) extreme value theory thresholding.

DEV DATA, NOT FOR REPORTED RESULTS
Requirements:
(1) On a large synthetic heavy-tailed sample, realized exceedance rate on a fresh sample is within 30% relative of q = 0.01.
(2) The xi near 0 branch (exponential distribution).
(3) The function signature accepts only scores (no labels, no test data).
"""

from __future__ import annotations

import inspect
import numpy as np
import pytest
from scipy.stats import expon, genpareto

from thresholds.pot import fit_pot_threshold


def test_pot_heavy_tailed_exceedance_rate():
    """(1) On large synthetic heavy-tailed sample, realized exceedance rate is within ~30% of q=0.01."""
    rng = np.random.default_rng(2026)
    q_target = 0.01
    
    # Heavy-tailed sample from Generalized Pareto with shape xi=0.25, scale=2.0
    cal_scores = genpareto.rvs(c=0.25, scale=2.0, size=50000, random_state=rng)
    thresh = fit_pot_threshold(cal_scores, q=q_target, percentile_u=90.0)

    # Fresh independent test sample
    test_scores = genpareto.rvs(c=0.25, scale=2.0, size=100000, random_state=rng)
    realized_exceedance = float(np.mean(test_scores > thresh))

    rel_error = abs(realized_exceedance - q_target) / q_target
    print(f"Heavy-tail POT: Target={q_target}, Realized={realized_exceedance:.4f}, RelError={rel_error:.2%}")
    assert rel_error <= 0.30, f"Relative error {rel_error:.2%} exceeds 30% tolerance!"


def test_pot_xi_near_zero_branch():
    """(2) Test the xi near 0 branch (exponential distribution limit)."""
    rng = np.random.default_rng(42)
    q_target = 0.01

    # Exponential distribution has xi = 0 exactly in GPD
    cal_scores = expon.rvs(scale=3.0, size=50000, random_state=rng)
    thresh = fit_pot_threshold(cal_scores, q=q_target, percentile_u=90.0)

    # Theoretical exact quantile for exponential with scale 3.0:
    # P(X > z) = exp(-z / 3.0) = 0.01 => z = -3.0 * ln(0.01) = 3.0 * ln(100) = 13.8155
    expected_exact = float(-3.0 * np.log(q_target))

    test_scores = expon.rvs(scale=3.0, size=100000, random_state=rng)
    realized_exceedance = float(np.mean(test_scores > thresh))
    rel_error = abs(realized_exceedance - q_target) / q_target

    print(f"Exponential POT: Estimated Thresh={thresh:.4f}, Theoretical={expected_exact:.4f}, Realized Exceedance={realized_exceedance:.4f}")
    assert np.isclose(thresh, expected_exact, rtol=0.05), "Threshold diverges from theoretical exponential quantile!"
    assert rel_error <= 0.30, f"Relative error {rel_error:.2%} exceeds 30% tolerance!"


def test_pot_function_signature():
    """(3) The function signature accepts only scores (and config floats), no labels, no test data."""
    sig = inspect.signature(fit_pot_threshold)
    param_names = list(sig.parameters.keys())

    # Verify first parameter is 'scores'
    assert param_names[0] == "scores"

    # Strictly verify that no label, ground truth, or test data parameters exist
    forbidden_terms = ["label", "target", "test", "y", "ground_truth", "gt"]
    for p in param_names:
        for term in forbidden_terms:
            assert term not in p.lower(), f"Forbidden parameter '{p}' found in fit_pot_threshold signature!"
