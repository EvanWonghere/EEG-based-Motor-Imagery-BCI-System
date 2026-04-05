"""EEGNet: compact CNN for EEG-based BCI (Lawhern et al., 2018).

Architecture:
  1. Temporal convolution  — learns frequency filters (like bandpass)
  2. Depthwise convolution — learns spatial filters (like CSP)
  3. Separable convolution — learns temporal patterns from the filtered signal
  4. Classification head   — dense layer → softmax

This implementation uses PyTorch and is designed for small-sample MI data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models.base import BaseModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for EEGNet. Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class _EEGNetModule(nn.Module):
        """Raw EEGNet architecture."""

        def __init__(
            self,
            n_channels: int = 22,
            n_times: int = 500,
            n_classes: int = 2,
            dropout: float = 0.5,
            F1: int = 8,
            D: int = 2,
            F2: int = 16,
            kernel_length: int = 64,
        ):
            super().__init__()

            # Block 1: Temporal + Depthwise spatial convolution
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
                nn.BatchNorm2d(F1),
                # Depthwise: each F1 filter gets D spatial filters
                nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(dropout),
            )

            # Block 2: Separable convolution
            self.block2 = nn.Sequential(
                # Depthwise temporal
                nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                # Pointwise
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(dropout),
            )

            # Compute flattened size
            self._flat_size = self._get_flat_size(n_channels, n_times)

            self.classifier = nn.Linear(self._flat_size, n_classes)

        def _get_flat_size(self, n_channels: int, n_times: int) -> int:
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self.block1(x)
            x = self.block2(x)
            return x.numel()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, 1, n_channels, n_times)
            x = self.block1(x)
            x = self.block2(x)
            x = x.flatten(1)
            return self.classifier(x)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class EEGNetModel(BaseModel):
    """EEGNet classifier conforming to the ``BaseModel`` interface.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_times : int
        Number of time samples per epoch.
    n_classes : int
        Number of output classes.
    dropout : float
    learning_rate : float
    weight_decay : float
    batch_size : int
    n_epochs : int
        Maximum training epochs.
    patience : int
        Early-stopping patience (epochs without validation improvement).
    early_stopping : bool
        Whether to stop training early when validation loss stalls.
    device : str
        ``"cuda"`` / ``"mps"`` / ``"cpu"`` / ``"auto"``.
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_times: int = 500,
        n_classes: int = 2,
        dropout: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        n_epochs: int = 200,
        patience: int = 30,
        early_stopping: bool = True,
        val_split: float = 0.1,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 10,
        min_delta: float = 1e-4,
        device: str = "auto",
        **kwargs: Any,
    ):
        _check_torch()
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.dropout = dropout
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.early_stopping = early_stopping
        self.val_split = val_split
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.min_delta = min_delta
        self.device = self._resolve_device(device)

        self._net: _EEGNetModule | None = None
        self._label_map: dict[int, int] | None = None
        self._label_inv: dict[int, int] | None = None
        self._norm_mean: np.ndarray | None = None
        self._norm_std: np.ndarray | None = None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_norm_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-channel normalization stats from training data."""
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return mean, std

    @staticmethod
    def _apply_normalization(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (X - mean) / std

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train EEGNet with optional early stopping.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)
        y : np.ndarray, shape (n_trials,)
        """
        # Map labels to contiguous 0..n_classes-1
        classes = sorted(set(y.tolist()))
        self._label_map = {c: i for i, c in enumerate(classes)}
        self._label_inv = {i: c for c, i in self._label_map.items()}
        y_mapped = np.array([self._label_map[c] for c in y])

        # Auto-detect dimensions from data
        self.n_channels = X.shape[1]
        self.n_times = X.shape[2]
        self.n_classes = len(classes)

        self._net = _EEGNetModule(
            n_channels=self.n_channels,
            n_times=self.n_times,
            n_classes=self.n_classes,
            dropout=self.dropout,
            F1=self.F1,
            D=self.D,
            F2=self.F2,
            kernel_length=self.kernel_length,
        ).to(self.device)
        net = self._net

        # Train/val split for optional early stopping and LR scheduling.
        from sklearn.model_selection import train_test_split
        use_val = self.val_split > 0.0
        if use_val:
            min_class_count = np.min(np.bincount(y_mapped))
            # Need at least 2 samples per class to stratify a split.
            use_val = min_class_count >= 2

        if use_val:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X,
                y_mapped,
                test_size=self.val_split,
                stratify=y_mapped,
                random_state=42,
            )
        else:
            X_tr, y_tr = X, y_mapped
            X_val = np.empty((0, X.shape[1], X.shape[2]), dtype=X.dtype)
            y_val = np.empty((0,), dtype=y_mapped.dtype)

        # Channel-wise normalization using training split statistics only.
        self._norm_mean, self._norm_std = self._compute_norm_stats(X_tr)
        X_tr = self._apply_normalization(X_tr, self._norm_mean, self._norm_std)
        if use_val:
            X_val = self._apply_normalization(X_val, self._norm_mean, self._norm_std)

        train_loader = self._make_loader(X_tr, y_tr, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False) if use_val else None

        if self.early_stopping:
            logger.info(
                "Early stopping enabled (patience=%d, min_delta=%.6f, val_split=%.2f)",
                self.patience,
                self.min_delta,
                self.val_split,
            )
        else:
            logger.info("Early stopping disabled; training runs full %d epochs", self.n_epochs)

        optimiser = torch.optim.AdamW(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode="min",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(1, self.n_epochs + 1):
            # --- train ---
            net.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimiser.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(X_tr)

            # --- validate ---
            if use_val and val_loader is not None:
                net.eval()
                val_loss = 0.0
                correct = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        logits = net(xb)
                        val_loss += criterion(logits, yb).item() * xb.size(0)
                        correct += (logits.argmax(1) == yb).sum().item()
                val_loss /= len(X_val)
                val_acc = correct / len(X_val)
            else:
                val_loss = train_loss
                val_acc = float("nan")

            if epoch % 20 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%s  lr=%.6f",
                    epoch,
                    self.n_epochs,
                    train_loss,
                    val_loss,
                    f"{val_acc * 100:.2f}%" if np.isfinite(val_acc) else "n/a",
                    optimiser.param_groups[0]["lr"],
                )

            # Early stopping
            scheduler.step(val_loss)
            if val_loss < (best_val_loss - self.min_delta):
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if self.early_stopping and wait >= self.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.patience)
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._label_inv is None:
            raise RuntimeError("Model is not fitted yet")
        proba = self.predict_proba(X)
        indices = proba.argmax(axis=1)
        return np.array([self._label_inv[i] for i in indices])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model is not fitted yet")
        if self._norm_mean is None or self._norm_std is None:
            raise RuntimeError("Model is not fitted yet")
        X = self._apply_normalization(X, self._norm_mean, self._norm_std)
        self._net.eval()
        loader = self._make_loader(X, labels=None, shuffle=False)
        all_proba = []
        with torch.no_grad():
            for (xb,) in loader:
                logits = self._net(xb.to(self.device))
                all_proba.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(all_proba)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        X: np.ndarray,
        labels: np.ndarray | None = None,
        shuffle: bool = False,
    ) -> DataLoader:
        # (n, ch, t) → (n, 1, ch, t) for Conv2d
        X_t = torch.from_numpy(X).float().unsqueeze(1)
        if labels is not None:
            y_t = torch.from_numpy(labels).long()
            ds = TensorDataset(X_t, y_t)
        else:
            ds = TensorDataset(X_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)
