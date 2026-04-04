"""EEG re-referencing methods."""

from __future__ import annotations

from mne.io import BaseRaw


def set_common_average_reference(raw: BaseRaw) -> BaseRaw:
    """Apply Common Average Reference (CAR) in-place.

    Subtracts the mean of all EEG channels at each time point,
    reducing common-mode noise and emphasizing local spatial differences.
    """
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw
