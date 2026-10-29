"""Unit tests for PyTorch Autoencoder and VAE anomaly detection models.

Requirements per User Request STEP 2:
1. AE reconstruction error is higher for synthetic outliers than inliers after short training.
2. Same seed gives the same result (strict determinism).
3. The training function only receives TRAIN and VAL-BENIGN arrays, so the test arrays cannot be reached.
"""

from __future__ import annotations

import inspect
import numpy as np
import pytest
import torch

from models.autoencoders import (
    MLPAutoencoder,
    TrimmedAutoencoder,
    VariationalAutoencoder,
    compute_trimmed_mse_loss,
    train_autoencoder,
    train_trimmed_autoencoder,
)


@pytest.fixture
def synthetic_train_val_test():
    rng = np.random.default_rng(42)
    # 1000 benign train samples, 200 benign val samples
    X_train = rng.normal(loc=0.0, scale=1.0, size=(1000, 10))
    X_val = rng.normal(loc=0.0, scale=1.0, size=(200, 10))

    # Test set: 200 inliers + 50 distinct outliers
    X_test_inliers = rng.normal(loc=0.0, scale=1.0, size=(200, 10))
    X_test_outliers = rng.normal(loc=12.0, scale=2.0, size=(50, 10))
    X_test = np.vstack([X_test_inliers, X_test_outliers])
    y_test = np.array([0] * 200 + [1] * 50)

    return X_train, X_val, X_test, y_test


def test_ae_synthetic_outlier_detection(synthetic_train_val_test):
    """Test (1): AE reconstruction error is higher for synthetic outliers than inliers."""
    X_train, X_val, X_test, y_test = synthetic_train_val_test

    ae = MLPAutoencoder(input_dim=10, hidden_dims=(32, 16), bottleneck_dim=8)
    trained_ae, _ = train_autoencoder(
        ae,
        X_train=X_train,
        X_val_benign=X_val,
        seed=42,
        max_epochs=10,
        batch_size=128,
        patience=3,
    )

    errors = trained_ae.compute_reconstruction_error(X_test)
    assert errors.shape == (250,)
    assert errors[y_test == 1].mean() > errors[y_test == 0].mean()

    # Repeat for VAE
    vae = VariationalAutoencoder(input_dim=10, hidden_dims=(32, 16), latent_dim=8)
    trained_vae, _ = train_autoencoder(
        vae,
        X_train=X_train,
        X_val_benign=X_val,
        seed=42,
        max_epochs=10,
        batch_size=128,
        patience=3,
    )

    vae_errors = trained_vae.compute_reconstruction_error(X_test)
    assert vae_errors.shape == (250,)
    assert vae_errors[y_test == 1].mean() > vae_errors[y_test == 0].mean()


def test_ae_seed_reproducibility(synthetic_train_val_test):
    """Test (2): Same seed gives the same result."""
    X_train, X_val, X_test, _ = synthetic_train_val_test

    # Run 1
    ae1 = MLPAutoencoder(input_dim=10, hidden_dims=(32, 16), bottleneck_dim=8, seed=123)
    trained_ae1, hist1 = train_autoencoder(
        ae1, X_train=X_train, X_val_benign=X_val, seed=123, max_epochs=5, batch_size=128
    )
    err1 = trained_ae1.compute_reconstruction_error(X_test)

    # Run 2
    ae2 = MLPAutoencoder(input_dim=10, hidden_dims=(32, 16), bottleneck_dim=8, seed=123)
    trained_ae2, hist2 = train_autoencoder(
        ae2, X_train=X_train, X_val_benign=X_val, seed=123, max_epochs=5, batch_size=128
    )
    err2 = trained_ae2.compute_reconstruction_error(X_test)

    # Verify identical loss progression and reconstruction errors
    assert [h["val_recon_loss"] for h in hist1] == [h["val_recon_loss"] for h in hist2]
    np.testing.assert_allclose(err1, err2, atol=1e-6)


def test_training_function_leakage_safety():
    """Test (3): The training function only receives TRAIN and VAL-BENIGN arrays, so test arrays cannot be reached."""
    sig = inspect.signature(train_autoencoder)
    param_names = list(sig.parameters.keys())

    # Assert exactly that X_train and X_val_benign are accepted, and NO test parameter exists
    assert "X_train" in param_names
    assert "X_val_benign" in param_names
    assert not any("test" in p.lower() for p in param_names), (
        f"train_autoencoder parameter signature must not accept TEST data! Found: {param_names}"
    )


def test_trimmed_ae_trim_zero_identical_to_plain_ae(synthetic_train_val_test):
    """Test Part B (1): with trim_frac = 0 the result is identical to plain AE with same seed."""
    X_train, X_val, X_test, _ = synthetic_train_val_test

    ae_plain = MLPAutoencoder(input_dim=10, hidden_dims=(32, 16), bottleneck_dim=8, seed=42)
    trained_plain, hist_plain = train_autoencoder(
        ae_plain,
        X_train=X_train,
        X_val_benign=X_val,
        seed=42,
        max_epochs=5,
        batch_size=128,
        trim_frac=0.0,
        verbose=False,
    )
    err_plain = trained_plain.compute_reconstruction_error(X_test)

    ae_trimmed = TrimmedAutoencoder(input_dim=10, hidden_dims=(32, 16), bottleneck_dim=8, seed=42)
    trained_trimmed, hist_trimmed = train_trimmed_autoencoder(
        ae_trimmed,
        X_train=X_train,
        X_val_benign=X_val,
        seed=42,
        max_epochs=5,
        batch_size=128,
        trim_frac=0.0,
        warmup_epochs=0,
        verbose=False,
    )
    err_trimmed = trained_trimmed.compute_reconstruction_error(X_test)

    assert [h["val_recon_loss"] for h in hist_plain] == [h["val_recon_loss"] for h in hist_trimmed]
    np.testing.assert_array_equal(err_plain, err_trimmed)


def test_trimmed_ae_sample_exclusion_and_ranking():
    """Test Part B (2): on a small synthetic batch, exactly ceil(trim_frac * batch_size)
    samples are excluded and they are the highest-error ones."""
    import math

    batch_size = 20
    dim = 8
    trim_frac = 0.15  # ceil(0.15 * 20) = 3 samples dropped

    # Create dummy target and reconstruction with controlled errors
    target = torch.zeros((batch_size, dim), dtype=torch.float32)
    recon = torch.zeros((batch_size, dim), dtype=torch.float32)

    # Set distinct error magnitudes per sample: sample i has squared error (i + 1)^2
    for i in range(batch_size):
        recon[i, 0] = float(i + 1)

    expected_errors = [(i + 1) ** 2 / dim for i in range(batch_size)]
    expected_n_trim = int(math.ceil(trim_frac * batch_size))
    assert expected_n_trim == 3

    # The 3 highest error samples must be indices 17, 18, 19
    expected_dropped_indices = {17, 18, 19}

    loss, excluded_indices = compute_trimmed_mse_loss(recon, target, trim_frac=trim_frac)

    assert len(excluded_indices) == expected_n_trim
    assert set(excluded_indices.tolist()) == expected_dropped_indices

    # Verify retained loss equals mean of lowest 17 errors
    expected_retained_loss = np.mean([expected_errors[i] for i in range(17)])
    np.testing.assert_allclose(loss.item(), expected_retained_loss, rtol=1e-5)


def test_trimmed_ae_excluded_samples_receive_zero_gradient():
    """Test Part B (3): the excluded samples receive zero gradient."""
    batch_size = 10
    dim = 5
    trim_frac = 0.2  # drop top 2

    # Synthetic batch where each sample requires grad
    x = torch.randn(batch_size, dim, requires_grad=True)
    model = MLPAutoencoder(input_dim=dim, hidden_dims=(16, 8), bottleneck_dim=4, seed=99)
    recon = model(x)

    loss, excluded_indices = compute_trimmed_mse_loss(recon, x, trim_frac=trim_frac)
    loss.backward()

    excluded_set = set(excluded_indices.tolist())
    assert len(excluded_set) == 2

    # Verify that gradients for excluded samples are exactly zero
    for idx in range(batch_size):
        sample_grad_norm = x.grad[idx].abs().sum().item()
        if idx in excluded_set:
            assert sample_grad_norm == 0.0, f"Sample {idx} was excluded but received non-zero grad {sample_grad_norm}"
        else:
            assert sample_grad_norm > 0.0, f"Sample {idx} was kept but received zero grad"
