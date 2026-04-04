"""Bandpass and notch filters."""

from __future__ import annotations

import mne
from mne.io import BaseRaw


def bandpass_filter(
    raw: BaseRaw,
    l_freq: float,
    h_freq: float,
    method: str = "fir",
    fir_design: str = "firwin",
) -> BaseRaw:
    """Apply FIR bandpass filter in-place.

    Parameters
    ----------
    raw : BaseRaw
    l_freq, h_freq : float
        Low and high cutoff in Hz.
    """
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method=method,
        fir_design=fir_design,
        verbose=False,
    )
    return raw


def notch_filter(
    raw: BaseRaw,
    freqs: list[float] | float = 50.0,
) -> BaseRaw:
    """Apply notch filter to remove power-line noise in-place.

    Parameters
    ----------
    freqs : list[float] | float
        Frequencies to notch out (e.g. 50.0 for EU, 60.0 for US).
    """
    if isinstance(freqs, (int, float)):
        freqs = [float(freqs)]
    raw.notch_filter(freqs=freqs, verbose=False)
    return raw
