"""
Filtering, artifact handling, and epoching for MI EEG data.
Uses MNE; supports BCI Competition IV 2a/2b (.mat or .gdf) or PhysioNet EEGBCI (.edf).
"""
import mne
from mne.io import BaseRaw
from mne.epochs import Epochs
import numpy as np
from typing import Tuple, Optional, Dict, Any

from .utils import BAND_LOW_HZ, BAND_HIGH_HZ, EPOCH_TMIN, EPOCH_TMAX


def bandpass_filter(
    raw: BaseRaw,
    l_freq: float = BAND_LOW_HZ,
    h_freq: float = BAND_HIGH_HZ,
    skip_by_annotation: str = "edge",
) -> BaseRaw:
    """Apply bandpass filter (Mu + Beta band). Modifies raw in place."""
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design="firwin",
        skip_by_annotation=skip_by_annotation,
    )
    return raw


def get_epochs(
    raw: BaseRaw,
    events: np.ndarray,
    event_id: Dict[str, int],
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
    picks: Optional[Any] = None,
    baseline: Optional[Tuple[float, float]] = None,
) -> Epochs:
    """Build Epochs from raw and events."""
    if picks is None:
        picks = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=tmin,
        tmax=tmax,
        proj=True,
        picks=picks,
        baseline=baseline,
        preload=True,
    )
    return epochs


def epochs_to_arrays(
    epochs: Epochs,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (X, y) for sklearn: X (n_epochs, n_channels, n_times), y (n_epochs,).
    Optionally crop epochs to [tmin, tmax] before extracting.
    """
    if tmin is not None or tmax is not None:
        epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)
    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1]
    return X, y
