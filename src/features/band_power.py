"""Band-power feature extraction via Welch PSD."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


# Standard EEG frequency bands
DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "theta": (4, 8),
    "mu": (8, 13),
    "beta": (13, 30),
    "low_gamma": (30, 45),
}


class BandPowerExtractor:
    """Extract average log-band-power features per channel.

    For each channel and each frequency band, estimate PSD with Welch's method
    and average within the band.

    Parameters
    ----------
    bands : dict[str, tuple[float, float]] | None
        Mapping of band name → (low_hz, high_hz).  Defaults to
        theta / mu / beta / low_gamma.
    sfreq : float
        Sampling rate (Hz) of the input epochs.
    """

    def __init__(
        self,
        bands: dict[str, tuple[float, float]] | None = None,
        sfreq: float = 250.0,
    ):
        self.bands = bands or DEFAULT_BANDS
        self.sfreq = sfreq

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BandPowerExtractor:
        """No-op (stateless extractor). Kept for pipeline compatibility."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract band-power features.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        features : np.ndarray, shape (n_trials, n_channels * n_bands)
        """
        n_trials, n_channels, _ = X.shape
        n_bands = len(self.bands)
        features = np.zeros((n_trials, n_channels * n_bands))

        for i in range(n_trials):
            for ch in range(n_channels):
                freqs, psd = welch(X[i, ch], fs=self.sfreq, nperseg=min(256, X.shape[2]))
                for bi, (_, (lo, hi)) in enumerate(self.bands.items()):
                    mask = (freqs >= lo) & (freqs < hi)
                    avg = np.mean(psd[mask]) if mask.any() else 1e-30
                    features[i, ch * n_bands + bi] = np.log(avg + 1e-30)

        return features

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
