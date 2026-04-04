"""Event detection, epoching, and array extraction."""

from __future__ import annotations

from typing import Any

import mne
import numpy as np
from mne.io import BaseRaw


def create_epochs(
    raw: BaseRaw,
    events: np.ndarray,
    event_id: dict[str, int],
    tmin: float = -0.5,
    tmax: float = 3.0,
    baseline: tuple[float, float] | None = None,
    picks: Any | None = None,
) -> mne.Epochs:
    """Create Epochs from raw data and events.

    Parameters
    ----------
    raw : BaseRaw
    events : np.ndarray, shape (n_events, 3)
    event_id : dict[str, int]
    tmin, tmax : float
        Epoch window relative to event onset (seconds).
    baseline : tuple | None
        Baseline correction interval. ``None`` = no baseline correction.
    picks : str | list | None
        Channel selection. Defaults to EEG channels only.
    """
    if picks is None:
        picks = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
    epochs = mne.Epochs(
        raw, events, event_id,
        tmin=tmin,
        tmax=tmax,
        proj=True,
        picks=picks,
        baseline=baseline,
        preload=True,
        verbose=False,
    )
    return epochs


def epochs_to_arrays(
    epochs: mne.Epochs,
    tmin: float | None = None,
    tmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(X, y)`` arrays from Epochs.

    Parameters
    ----------
    epochs : mne.Epochs
    tmin, tmax : float | None
        If given, crop the epochs to this window before extracting.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_channels, n_times)
    y : np.ndarray, shape (n_epochs,)
    """
    if tmin is not None or tmax is not None:
        epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)
    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1]
    return X, y
