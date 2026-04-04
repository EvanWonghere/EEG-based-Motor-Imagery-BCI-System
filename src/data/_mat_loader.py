"""Shared helper for loading BCI Competition IV ``.mat`` files.

The .mat files from BCI IV have inconsistent internal structures depending on
the exporter.  This module consolidates the resilient parsing logic that was
previously inlined in ``train_model.py``.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
from mne.io import RawArray
from scipy.io import loadmat


def mat_to_raw_events(
    mat_path: Path,
    sfreq: float,
    keep_classes: list[int],
    event_id_map: dict[int, int],
) -> tuple[mne.io.Raw, np.ndarray, dict[str, int]]:
    """Load a ``.mat`` file and construct MNE Raw + events.

    Parameters
    ----------
    mat_path : Path
        Path to the ``.mat`` file.
    sfreq : float
        Sampling rate in Hz.
    keep_classes : list[int]
        Class labels to keep (e.g. ``[1, 2]`` for left/right hand).
    event_id_map : dict[int, int]
        Mapping from class label in the file to canonical event code
        (e.g. ``{1: 769, 2: 770}``).

    Returns
    -------
    raw : mne.io.Raw
    events : np.ndarray, shape (n_events, 3)
    event_id : dict[str, int]
    """
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    data = _extract_data(mat)
    labels = _extract_labels(mat, data)

    # Reshape to (n_trials, n_channels, n_times)
    X = _to_trials_array(data, labels)

    # Keep only the requested classes
    keep = np.isin(labels, keep_classes)
    X = X[keep]
    labels = labels[keep]
    if len(labels) == 0:
        raise ValueError(
            f"No trials left after keeping classes {keep_classes}"
        )

    # Build continuous Raw + events
    n_trials, n_ch, n_times = X.shape
    data_concat = X.reshape(n_ch, -1)  # (n_ch, total_samples)

    events_list = []
    for i, lab in enumerate(labels):
        code = event_id_map.get(int(lab), int(lab))
        events_list.append([i * n_times, 0, code])
    events = np.array(events_list, dtype=int)

    info = mne.create_info(
        ch_names=[f"EEG{i + 1}" for i in range(n_ch)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    raw = RawArray(data_concat, info, verbose=False)

    codes_in_use = sorted(np.unique(events[:, 2]).tolist())
    if len(codes_in_use) == 2:
        names = ["Left Hand", "Right Hand"]
    else:
        names = [f"class_{c}" for c in codes_in_use]
    event_id = dict(zip(names, codes_in_use))

    return raw, events, event_id


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_data(mat: dict) -> np.ndarray:
    for key in ("data", "Data", "X", "x"):
        if key in mat and isinstance(mat[key], np.ndarray):
            return np.asarray(mat[key])
    raise ValueError(
        f"Cannot find data array in .mat; "
        f"keys: {[k for k in mat if not k.startswith('_')]}"
    )


def _extract_labels(mat: dict, data: np.ndarray) -> np.ndarray:
    # Top-level label keys
    for key in ("label", "labels", "y", "class", "Y"):
        if key in mat and isinstance(mat[key], np.ndarray):
            return np.asarray(mat[key]).ravel()

    # Try extracting from struct fields inside data
    flat = data.ravel()
    if len(flat) == 0:
        raise ValueError("Empty data array, cannot extract labels")

    first = flat[0]

    # Single struct element with a label attribute
    for attr in ("y", "label", "labels", "class", "t", "trial_type", "type"):
        if hasattr(first, attr):
            arr = getattr(first, attr)
            if isinstance(arr, np.ndarray):
                return np.asarray(arr).ravel()

    # _fieldnames fallback
    if hasattr(first, "_fieldnames"):
        for fn in first._fieldnames:
            if fn.lower() in ("y", "label", "class", "t"):
                arr = getattr(first, fn, None)
                if isinstance(arr, np.ndarray):
                    return np.asarray(arr).ravel()

    # Struct-per-trial: each element has .y
    if data.size > 1 and hasattr(flat[0], "y"):
        labels = np.array([getattr(t, "y", None) for t in flat])
        if labels.dtype == object:
            labels = np.array([int(l) if l is not None else -1 for l in labels])
        return labels

    # (1,1) struct
    if data.size == 1 and hasattr(first, "y"):
        return np.asarray(getattr(first, "y")).ravel()

    raise ValueError(
        f"Cannot find labels in .mat; "
        f"keys: {[k for k in mat if not k.startswith('_')]}"
    )


def _to_trials_array(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return data as (n_trials, n_channels, n_times)."""
    flat = data.ravel()
    first = flat[0] if len(flat) > 0 else None

    # Struct array — each row is a trial with .x / .X
    if first is not None and (data.ndim == 1 or (data.ndim == 2 and data.size > 1)):
        attr = "x" if hasattr(first, "x") else "X" if hasattr(first, "X") else None
        if attr is not None:
            parts = [np.asarray(getattr(t, attr), dtype=float) for t in flat]
            if all(p.ndim == 3 for p in parts):
                data = np.concatenate(parts, axis=0)
            elif len(parts) == 1:
                data = parts[0]

    # (1,1) struct wrapping the whole dataset
    if first is not None and data.ndim != 3 and data.size == 1:
        for attr in ("x", "X"):
            if hasattr(first, attr):
                data = np.asarray(getattr(first, attr), dtype=float)
                break

    if data.ndim != 3:
        raise ValueError(f"Expected 3-D data array, got ndim={data.ndim}")

    a, b, c = data.shape
    n = labels.shape[0]
    if a == n:
        return np.asarray(data, dtype=float)
    if c == n:
        return np.transpose(data, (2, 0, 1)).astype(float)
    if b == n:
        return np.transpose(data, (1, 0, 2)).astype(float)

    raise ValueError(
        f"data shape {data.shape} does not match labels length {n}"
    )
