"""Artifact detection and removal: ICA and epoch rejection."""

from __future__ import annotations

import mne
import numpy as np
from mne.io import BaseRaw
from mne.preprocessing import ICA

from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_ica_artifact_removal(
    raw: BaseRaw,
    n_components: int = 15,
    method: str = "fastica",
    eog_channels: list[str] | None = None,
    random_state: int = 42,
) -> tuple[BaseRaw, ICA]:
    """Run ICA and automatically remove EOG-correlated components.

    Parameters
    ----------
    raw : BaseRaw
        Raw data (should already be filtered ≥1 Hz for ICA stability).
    n_components : int
        Number of ICA components to fit.
    method : str
        ICA algorithm (``"fastica"`` or ``"infomax"``).
    eog_channels : list[str] | None
        Explicit EOG channel names. If ``None``, MNE auto-detects.
    random_state : int
        For reproducibility.

    Returns
    -------
    raw : BaseRaw
        Data with artifact components removed (in-place).
    ica : ICA
        Fitted ICA object (useful for visualization).
    """
    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw, verbose=False)

    # Auto-detect EOG components
    eog_indices = []
    if eog_channels:
        for ch in eog_channels:
            indices, _ = ica.find_bads_eog(raw, ch_name=ch, verbose=False)
            eog_indices.extend(indices)
    else:
        # Try default EOG detection (works if EOG channels exist in data)
        try:
            indices, _ = ica.find_bads_eog(raw, verbose=False)
            eog_indices.extend(indices)
        except RuntimeError:
            # No EOG channels found — use correlation-based heuristic
            logger.info("No EOG channels found; skipping automatic EOG component detection")

    if eog_indices:
        ica.exclude = list(set(eog_indices))
        logger.info("ICA: excluding %d EOG component(s): %s", len(ica.exclude), ica.exclude)
        ica.apply(raw, verbose=False)
    else:
        logger.info("ICA: no EOG components detected, no components removed")

    return raw, ica


def reject_bad_epochs(
    epochs: mne.Epochs,
    reject_threshold: float = 100e-6,
) -> mne.Epochs:
    """Drop epochs exceeding peak-to-peak amplitude threshold.

    Parameters
    ----------
    epochs : mne.Epochs
        Must be preloaded.
    reject_threshold : float
        Threshold in Volts (default 100 µV).

    Returns
    -------
    epochs : mne.Epochs
        Epochs with bad trials dropped.
    """
    n_before = len(epochs)
    epochs.drop_bad(reject=dict(eeg=reject_threshold), verbose=False)
    n_dropped = n_before - len(epochs)
    if n_dropped > 0:
        logger.info(
            "Epoch rejection: dropped %d / %d (%.1f%%)",
            n_dropped, n_before, 100.0 * n_dropped / n_before,
        )
    return epochs
