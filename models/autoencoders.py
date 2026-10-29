"""PyTorch Autoencoder and Variational Autoencoder architectures for anomaly detection.

Source of Truth: SPEC.md Section 4 & User Request STEP 2.

Architectures:
1. Plain MLP Autoencoder:
   - input -> 64 -> 32 -> 16 (bottleneck) -> 32 -> 64 -> input
   - ReLU activations, linear output
   - Trained on benign TRAIN only with MSE loss and Adam optimizer (lr=1e-3, batch=1024, max 30 epochs)
   - Early stopping (patience 3) on VAL-BENIGN reconstruction loss ONLY
   - Anomaly score = per-sample mean squared reconstruction error

2. Variational Autoencoder (VAE):
   - Same layer dimensions, latent dim 16
   - Standard KL divergence term
   - Anomaly score = deterministic reconstruction error using latent mean (no sampling)
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLPAutoencoder(nn.Module):
    """Plain MLP Autoencoder for tabular flow reconstruction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, int] = (64, 32),
        bottleneck_dim: int = 16,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.bottleneck_dim = bottleneck_dim

        # Encoder: input -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], bottleneck_dim),
        )

        # Decoder: 16 -> 32 -> 64 -> input (linear output)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    @torch.no_grad()
    def compute_reconstruction_error(
        self,
        X: Union[np.ndarray, torch.Tensor],
        batch_size: int = 4096,
        device: str = "cpu",
    ) -> np.ndarray:
        """Compute per-sample mean squared reconstruction error."""
        self.eval()
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X.astype(np.float32))
        else:
            X_tensor = X.float()

        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        errors: List[np.ndarray] = []
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            recon = self.forward(batch_x)
            mse_per_sample = torch.mean((batch_x - recon) ** 2, dim=1).cpu().numpy()
            errors.append(mse_per_sample)

        return np.concatenate(errors, axis=0)


def compute_trimmed_mse_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    trim_frac: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute trimmed MSE loss dropping the top trim_frac fraction of highest-error samples.

    Per User Request Part B:
    - For every batch compute per-sample squared error.
    - DROP the top trim_frac highest-error samples from the loss (their loss contributes zero gradient).
    - Average over the rest.

    Returns:
        loss: Scalar mean loss over retained samples (gradient flows through retained samples only).
        excluded_indices: Tensor of batch indices that were dropped.
    """
    sample_errors = torch.mean((recon - target) ** 2, dim=1)
    B = recon.shape[0]
    n_trim = int(math.ceil(trim_frac * B))
    if n_trim <= 0:
        return torch.mean(sample_errors), torch.empty(0, dtype=torch.long, device=recon.device)
    if n_trim >= B:
        n_trim = B - 1

    k = B - n_trim
    topk_vals, _ = torch.topk(sample_errors, k=k, largest=False)
    loss = torch.mean(topk_vals)
    _, excluded_indices = torch.topk(sample_errors, k=n_trim, largest=True)
    return loss, excluded_indices


class TrimmedAutoencoder(MLPAutoencoder):
    """Trimmed-loss MLP Autoencoder for contamination-robust representation learning.

    Shares identical architecture with MLPAutoencoder:
    input -> 64 -> 32 -> 16 (bottleneck) -> 32 -> 64 -> input with ReLU activations and linear output.
    Trained with batch-wise trimmed MSE loss dropping top trim_frac highest-error samples after warmup.
    """
    pass


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder (VAE) for tabular flow reconstruction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, int] = (64, 32),
        latent_dim: int = 16,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Shared encoder: input -> 64 -> 32
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        # Latent heads: mean and log-variance
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)

        # Decoder: 16 -> 32 -> 64 -> input (linear output)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @torch.no_grad()
    def compute_reconstruction_error(
        self,
        X: Union[np.ndarray, torch.Tensor],
        batch_size: int = 4096,
        device: str = "cpu",
    ) -> np.ndarray:
        """Deterministic reconstruction error using the latent mean (no sampling)."""
        self.eval()
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X.astype(np.float32))
        else:
            X_tensor = X.float()

        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        errors: List[np.ndarray] = []
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            mu, _ = self.encode(batch_x)
            recon = self.decode(mu)
            mse_per_sample = torch.mean((batch_x - recon) ** 2, dim=1).cpu().numpy()
            errors.append(mse_per_sample)

        return np.concatenate(errors, axis=0)


def train_autoencoder(
    model: Union[MLPAutoencoder, VariationalAutoencoder, TrimmedAutoencoder],
    X_train: np.ndarray,
    X_val_benign: np.ndarray,
    seed: int = 0,
    max_epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    patience: int = 3,
    device: str = "cpu",
    trim_frac: float = 0.0,
    warmup_epochs: int = 2,
    log_csv_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Tuple[Union[MLPAutoencoder, VariationalAutoencoder, TrimmedAutoencoder], List[Dict[str, Any]]]:
    """Train Autoencoder strictly on benign TRAIN with early stopping on VAL-BENIGN.

    Leakage invariant:
    - This function accepts ONLY X_train and X_val_benign.
    - Test arrays cannot be reached.
    """
    # Seed PyTorch for strict reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_tensor = torch.from_numpy(X_train.astype(np.float32))
    val_tensor = torch.from_numpy(X_val_benign.astype(np.float32))

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=batch_size * 2,
        shuffle=False,
    )

    best_val_loss = float("inf")
    best_epoch = 1
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    stopped_epoch = max_epochs

    history: List[Dict[str, Any]] = []

    is_vae = isinstance(model, VariationalAutoencoder)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_total = 0.0
        n_train_samples = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            if is_vae:
                recon, mu, logvar = model(batch_x)
                mse_loss = F.mse_loss(recon, batch_x, reduction="mean")
                # Normalized KL loss per feature
                kl_loss = -0.5 * torch.mean(
                    torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                ) / model.input_dim
                loss = mse_loss + kl_loss
            else:
                recon = model(batch_x)
                if trim_frac > 0.0 and epoch > warmup_epochs:
                    loss, _ = compute_trimmed_mse_loss(recon, batch_x, trim_frac=trim_frac)
                else:
                    loss = F.mse_loss(recon, batch_x, reduction="mean")

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * len(batch_x)
            n_train_samples += len(batch_x)

        train_loss = train_loss_total / max(1, n_train_samples)

        # Validation on VAL-BENIGN reconstruction loss ONLY (untrimmed MSE per SPEC)
        model.eval()
        val_recon_loss_total = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                if is_vae:
                    mu, _ = model.encode(batch_x)
                    recon = model.decode(mu)
                else:
                    recon = model(batch_x)

                mse_val = F.mse_loss(recon, batch_x, reduction="mean")
                val_recon_loss_total += mse_val.item() * len(batch_x)
                n_val_samples += len(batch_x)

        val_recon_loss = val_recon_loss_total / max(1, n_val_samples)
        epoch_time = time.time() - t0

        if verbose:
            print(
                f"      Epoch {epoch:02d}/{max_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Recon Loss: {val_recon_loss:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_recon_loss": round(val_recon_loss, 6),
            "epoch_time_seconds": round(epoch_time, 2),
        }
        history.append(epoch_record)

        # Early stopping check on VAL-BENIGN reconstruction loss ONLY
        if val_recon_loss < best_val_loss - 1e-5:
            best_val_loss = val_recon_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        stopped_epoch = epoch
        if epochs_no_improve >= patience:
            break

    # Restore best weights
    model.load_state_dict(best_weights)

    early_stop_str = (
        f"Early stopping triggered at Epoch {stopped_epoch}/{max_epochs} | Best validation epoch: {best_epoch} "
        f"(val loss: {best_val_loss:.6f})"
    )
    print(f"      [AE Early Stopping] {early_stop_str}")

    if log_csv_path is not None:
        log_csv_path = Path(log_csv_path)
        log_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(log_csv_path, index=False)

    return model, history


def train_trimmed_autoencoder(
    model: Union[MLPAutoencoder, TrimmedAutoencoder],
    X_train: np.ndarray,
    X_val_benign: np.ndarray,
    trim_frac: float = 0.05,
    warmup_epochs: int = 2,
    seed: int = 0,
    max_epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    patience: int = 3,
    device: str = "cpu",
    log_csv_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Tuple[Union[MLPAutoencoder, TrimmedAutoencoder], List[Dict[str, Any]]]:
    """Train Trimmed-Loss Autoencoder with warmup and top-k sample trimming per batch.

    Per User Request Part B:
    - Same architecture as plain AE.
    - During training, for every batch compute per-sample squared error, DROP the top trim_frac = 5%
      highest-error samples from the loss (their loss contributes zero gradient), and average over the rest.
    - Warm-up: no trimming for the first 2 epochs, then trim.
    - Early stopping on VAL-BENIGN untrimmed MSE, patience 3, same as plain AE.
    """
    return train_autoencoder(
        model=model,
        X_train=X_train,
        X_val_benign=X_val_benign,
        seed=seed,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        device=device,
        trim_frac=trim_frac,
        warmup_epochs=warmup_epochs,
        log_csv_path=log_csv_path,
        verbose=verbose,
    )
