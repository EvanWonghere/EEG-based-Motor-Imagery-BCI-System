"""BCI Competition IV Dataset 2a loader.

9 subjects, 22 EEG channels, 250 Hz.
4-class MI: Left Hand (769), Right Hand (770), Feet (771), Tongue (772).
This project uses the Left/Right 2-class subset.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
from mne.io import read_raw_gdf

from src.data._compat import patch_numpy_fromstring
from src.data._mat_loader import mat_to_raw_events
from src.data.base import EEGDataset
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Canonical event IDs for left/right hand
_EVENT_ID = {"Left Hand": 769, "Right Hand": 770}

# GDF marker codes that are *not* MI class labels
_SKIP_CODES = {276, 277, 1023, 1072, 32766}

_CH_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]


class BCIIV2aDataset(EEGDataset):
    """BCI Competition IV Dataset 2a."""

    n_subjects: int = 9
    sfreq: float = 250.0
    ch_names: list[str] = _CH_NAMES

    def __init__(
        self,
        data_dir: str | Path,
        subjects: list[int] | None = None,
        data_subdir: str = "MNE-bnci-data/database/data-sets/001-2014",
    ):
        super().__init__(data_dir, subjects)
        self._dataset_dir = self.data_dir / data_subdir

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
        session = session or "T"
        session_letter = session[-1].upper()  # "session_T" → "T", "T" → "T"

        # Try GDF first
        raw_events = self._try_gdf(subject_id, session_letter)
        if raw_events is not None:
            return raw_events

        # Try the other session as fallback
        alt = "E" if session_letter == "T" else "T"
        raw_events = self._try_gdf(subject_id, alt)
        if raw_events is not None:
            logger.info(
                "Subject %d: using session %s (requested %s not found)",
                subject_id, alt, session_letter,
            )
            return raw_events

        # Fallback to .mat
        raw_events = self._try_mat(subject_id)
        if raw_events is not None:
            return raw_events

        raise FileNotFoundError(
            f"BCI IV 2a subject {subject_id}: no .gdf or .mat found under "
            f"{self._dataset_dir}"
        )

    # ------------------------------------------------------------------
    # GDF loading
    # ------------------------------------------------------------------

    def _try_gdf(
        self, subject_id: int, session_letter: str
    ) -> tuple[mne.io.Raw, np.ndarray, dict[str, int]] | None:
        fname = f"A{subject_id:02d}{session_letter}.gdf"
        path = self._find_file(fname)
        if path is None:
            # Also try single-digit naming (A01T vs A1T)
            fname_alt = f"A{subject_id}{session_letter}.gdf"
            path = self._find_file(fname_alt)
        if path is None:
            return None

        patch_numpy_fromstring()
        raw = read_raw_gdf(str(path), preload=True, verbose=False)
        raw = self._normalize_channel_metadata(raw)

        # GDF annotations use string codes ("769", "770", ...).
        # Map the MI class annotations directly to our event_id.
        event_id_from_ann = {"769": "Left Hand", "770": "Right Hand"}
        ann_descriptions = set(raw.annotations.description)

        if "769" in ann_descriptions and "770" in ann_descriptions:
            # Standard path: annotation text matches event codes
            mapping = {desc: _EVENT_ID[name]
                       for desc, name in event_id_from_ann.items()
                       if desc in ann_descriptions}
            events, _ = mne.events_from_annotations(raw, event_id=mapping, verbose=False)
            event_id = dict(_EVENT_ID)
        else:
            # Fallback: try find_events (older GDF format)
            try:
                events = mne.find_events(raw, shortest_event=1, verbose=False)
                codes = np.unique(events[:, 2])
                class_codes = [c for c in codes if c not in _SKIP_CODES][:2]
                if len(class_codes) >= 2:
                    event_id = {"Left Hand": int(class_codes[0]), "Right Hand": int(class_codes[1])}
                else:
                    event_id = dict(_EVENT_ID)
            except (ValueError, RuntimeError):
                events, _ = mne.events_from_annotations(raw, verbose=False)
                event_id = dict(_EVENT_ID)

        # Keep only left/right events
        events = events[np.isin(events[:, 2], list(event_id.values()))]
        return raw, events, event_id

    # ------------------------------------------------------------------
    # .mat fallback
    # ------------------------------------------------------------------

    def _try_mat(
        self, subject_id: int
    ) -> tuple[mne.io.Raw, np.ndarray, dict[str, int]] | None:
        for pattern in (
            f"A{subject_id:02d}E.mat", f"A{subject_id:02d}T.mat",
            f"A{subject_id}E.mat", f"A{subject_id}T.mat",
        ):
            path = self._find_file(pattern)
            if path is not None:
                raw, events, event_id = mat_to_raw_events(
                    path, self.sfreq,
                    keep_classes=[1, 2],
                    event_id_map={1: 769, 2: 770},
                )
                raw = self._normalize_channel_metadata(raw)
                return raw, events, event_id
        return None

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _find_file(self, filename: str) -> Path | None:
        """Search for *filename* under the dataset directory."""
        direct = self._dataset_dir / filename
        if direct.exists():
            return direct
        # Recursive search
        for p in self._dataset_dir.rglob(filename):
            return p
        # Broader search under data_dir
        for p in self.data_dir.rglob(filename):
            return p
        return None

    def _normalize_channel_metadata(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Fix channel types/names so preprocessing can reliably pick EEG channels."""
        # In GDF files, EOG channels may be typed as EEG; correct them explicitly.
        eog_like = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
        if eog_like:
            raw.set_channel_types({ch: "eog" for ch in eog_like}, verbose=False)

        # Standardize first 22 non-EOG channels to canonical BCI IV 2a names.
        eeg_like = [ch for ch in raw.ch_names if ch not in eog_like]
        if len(eeg_like) >= len(_CH_NAMES):
            rename_map = {
                old: new
                for old, new in zip(eeg_like[:len(_CH_NAMES)], _CH_NAMES)
                if old != new
            }
            if rename_map:
                raw.rename_channels(rename_map)

        return raw
