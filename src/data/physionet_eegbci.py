"""PhysioNet EEG Motor Movement/Imagery Dataset loader.

109 subjects, 64 EEG channels, 160 Hz.
Uses runs 4/8/12 (left fist vs right fist motor imagery).

Reference: Schalk et al., 2004. BCI2000.
Data source: https://physionet.org/content/eegmmidb/1.0.0/
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from src.data.base import EEGDataset
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Runs for left fist vs right fist motor imagery
_DEFAULT_RUNS = [4, 8, 12]

# Map annotation labels to canonical event codes
_EVENT_ID = {"Left Hand": 769, "Right Hand": 770}


class PhysioNetEEGBCIDataset(EEGDataset):
    """PhysioNet EEG Motor Movement/Imagery Dataset."""

    n_subjects: int = 109
    sfreq: float = 160.0
    ch_names: list[str] = []  # populated dynamically (64 channels)

    def __init__(
        self,
        data_dir: str | Path,
        subjects: list[int] | None = None,
        runs: list[int] | None = None,
    ):
        self.runs = runs or _DEFAULT_RUNS
        super().__init__(data_dir, subjects)

    def get_event_id(self) -> dict[str, int]:
        return dict(_EVENT_ID)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_raw(self, subject_id: int, session: str | None = None) -> mne.io.Raw:
        raw, _, _ = self.load_data(subject_id, session)
        return raw

    def load_data(
        self, subject_id: int, session: str | None = None
    ) -> tuple[mne.io.Raw, np.ndarray, dict[str, int]]:
        """Load and concatenate runs for one subject.

        Parameters
        ----------
        subject_id : int
            Subject number (1-109).
        session : str or None
            Ignored (PhysioNet has no session concept).
        """
        raw_fnames = eegbci.load_data(
            subject_id, self.runs, path=str(self.data_dir),
        )
        raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
        raw = concatenate_raws(raws)

        # Standardize channel names and set montage
        eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="warn")

        # Update ch_names from actual data
        if not self.ch_names:
            self.__class__.ch_names = raw.ch_names.copy()

        # Extract events from annotations
        # T0 = rest, T1 = left fist (runs 4/8/12), T2 = right fist (runs 4/8/12)
        ann_mapping = {"T1": 769, "T2": 770}
        events, _ = mne.events_from_annotations(
            raw, event_id=ann_mapping, verbose=False,
        )

        event_id = {"Left Hand": 769, "Right Hand": 770}

        logger.info(
            "PhysioNet subject %d: %d events (%s)",
            subject_id, len(events),
            {k: int(np.sum(events[:, 2] == v)) for k, v in event_id.items()},
        )

        return raw, events, event_id
