"""Abstract base class for all EEG dataset loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import mne
import numpy as np


class EEGDataset(ABC):
    """Unified interface for EEG dataset loading.

    Each concrete subclass handles one dataset family (BCI IV 2a, 2b, etc.)
    and knows how to find files, parse events, and return MNE objects.
    """

    def __init__(self, data_dir: str | Path, subjects: list[int] | None = None):
        self.data_dir = Path(data_dir)
        self.subjects = subjects or list(range(1, self.n_subjects + 1))

    @abstractmethod
    def load_raw(self, subject_id: int, session: str | None = None) -> mne.io.Raw:
        """Load raw EEG data for one subject/session."""
        ...

    @abstractmethod
    def load_data(
        self, subject_id: int, session: str | None = None
    ) -> tuple[mne.io.Raw, np.ndarray, dict[str, int]]:
        """Load raw data with events already extracted.

        Returns
        -------
        raw : mne.io.Raw
        events : np.ndarray, shape (n_events, 3)
        event_id : dict[str, int]
            Mapping of class name to event code.
        """
        ...

    @abstractmethod
    def get_event_id(self) -> dict[str, int]:
        """Return the canonical event-ID mapping for this dataset."""
        ...

    @property
    @abstractmethod
    def n_subjects(self) -> int:
        ...

    @property
    @abstractmethod
    def sfreq(self) -> float:
        ...

    @property
    @abstractmethod
    def ch_names(self) -> list[str]:
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(data_dir={self.data_dir!r}, "
            f"subjects={self.subjects})"
        )
