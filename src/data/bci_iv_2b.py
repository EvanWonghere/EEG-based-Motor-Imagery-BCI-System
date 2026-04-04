"""BCI Competition IV Dataset 2b loader.

9 subjects, 3 bipolar EEG channels (C3, Cz, C4), 250 Hz.
2-class MI: Left Hand (769), Right Hand (770).
5 sessions per subject: 3 training (no feedback) + 2 evaluation (with feedback).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import mne
import numpy as np
from mne.io import read_raw_gdf

from src.data._compat import patch_numpy_fromstring
from src.data._mat_loader import mat_to_raw_events
from src.data.base import EEGDataset
from src.utils.logging import get_logger

logger = get_logger(__name__)

_EVENT_ID = {"Left Hand": 769, "Right Hand": 770}
_CH_NAMES = ["C3", "Cz", "C4"]

# GDF file naming for 2b: B{subject:02d}{run:02d}{session}.gdf
# Training runs: 01-03 (T), Evaluation runs: 04-05 (E)
_GDF_PATTERNS_EVAL = [
    "B{sub:02d}04E.gdf",
    "B{sub:02d}05E.gdf",
]
_GDF_PATTERNS_TRAIN = [
    "B{sub:02d}01T.gdf",
    "B{sub:02d}02T.gdf",
    "B{sub:02d}03T.gdf",
]


class BCIIV2bDataset(EEGDataset):
    """BCI Competition IV Dataset 2b."""

    n_subjects: int = 9
    sfreq: float = 250.0
    ch_names: list[str] = _CH_NAMES

    def __init__(
        self,
        data_dir: str | Path,
        subjects: list[int] | None = None,
        data_subdir: str = "MNE-bnci-data/database/data-sets/004-2014",
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
        # Try GDF files (eval sessions first — they have feedback)
        patterns = _GDF_PATTERNS_EVAL + _GDF_PATTERNS_TRAIN
        for pat in patterns:
            fname = pat.format(sub=subject_id)
            path = self._find_file(fname)
            if path is not None:
                return self._load_gdf(path)

        # Fallback to .mat
        raw_events = self._try_mat(subject_id)
        if raw_events is not None:
            return raw_events

        raise FileNotFoundError(
            f"BCI IV 2b subject {subject_id}: no .gdf or .mat found under "
            f"{self._dataset_dir}"
        )

    # ------------------------------------------------------------------
    # GDF loading
    # ------------------------------------------------------------------

    def _load_gdf(
        self, path: Path
    ) -> tuple[mne.io.Raw, np.ndarray, dict[str, int]]:
        patch_numpy_fromstring()
        raw = read_raw_gdf(str(path), preload=True, verbose=False)

        # Prefer annotation-based extraction (GDF annotations are strings)
        ann_descriptions = set(raw.annotations.description)
        if "769" in ann_descriptions and "770" in ann_descriptions:
            mapping = {"769": 769, "770": 770}
            events, _ = mne.events_from_annotations(raw, event_id=mapping, verbose=False)
            event_id = {"Left Hand": 769, "Right Hand": 770}
        else:
            # Fallback for older GDF or non-standard annotations
            try:
                events = mne.find_events(raw, shortest_event=1, verbose=False)
            except (ValueError, RuntimeError):
                events, _ = mne.events_from_annotations(raw, verbose=False)

            uniq = set(np.unique(events[:, 2]).tolist())
            if 769 in uniq and 770 in uniq:
                event_id = {"Left Hand": 769, "Right Hand": 770}
            elif 1 in uniq and 2 in uniq:
                event_id = {"Left Hand": 1, "Right Hand": 2}
            else:
                counts = Counter(events[:, 2])
                top2 = [code for code, _ in counts.most_common(2)]
                event_id = {"Left Hand": int(top2[0]), "Right Hand": int(top2[1])}

        events = events[np.isin(events[:, 2], list(event_id.values()))]
        return raw, events, event_id

    # ------------------------------------------------------------------
    # .mat fallback
    # ------------------------------------------------------------------

    def _try_mat(
        self, subject_id: int
    ) -> tuple[mne.io.Raw, np.ndarray, dict[str, int]] | None:
        for pattern in (
            f"B{subject_id:02d}E.mat", f"B{subject_id:02d}T.mat",
            f"B{subject_id}E.mat", f"B{subject_id}T.mat",
        ):
            path = self._find_file(pattern)
            if path is not None:
                return mat_to_raw_events(
                    path, self.sfreq,
                    keep_classes=[1, 2],
                    event_id_map={1: 769, 2: 770},
                )
        return None

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _find_file(self, filename: str) -> Path | None:
        direct = self._dataset_dir / filename
        if direct.exists():
            return direct
        for p in self._dataset_dir.rglob(filename):
            return p
        for p in self.data_dir.rglob(filename):
            return p
        return None
