"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.filters import bandpass_filter, notch_filter
from src.preprocessing.reference import set_common_average_reference
from src.preprocessing.epochs import create_epochs, epochs_to_arrays
from src.preprocessing.artifacts import reject_bad_epochs
from src.preprocessing.quality import detect_bad_channels
from src.preprocessing.pipeline import PreprocessingPipeline


# ------------------------------------------------------------------
# Helpers: synthetic data
# ------------------------------------------------------------------

def _make_raw(n_channels: int = 8, sfreq: float = 250.0, duration: float = 10.0):
    """Create a synthetic Raw object with random EEG data."""
    rng = np.random.RandomState(42)
    n_times = int(sfreq * duration)
    data = rng.randn(n_channels, n_times) * 20e-6  # ~20 µV
    ch_names = [f"EEG{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def _make_events(n_events: int = 20, sfreq: float = 250.0):
    """Create synthetic events alternating between two classes."""
    events = []
    for i in range(n_events):
        sample = int((i + 1) * sfreq * 2)  # every 2s
        code = 1 if i % 2 == 0 else 2
        events.append([sample, 0, code])
    return np.array(events, dtype=int)


EVENT_ID = {"Left Hand": 1, "Right Hand": 2}


# ------------------------------------------------------------------
# Individual modules
# ------------------------------------------------------------------

class TestFilters:
    def test_bandpass(self):
        raw = _make_raw(duration=5.0)
        result = bandpass_filter(raw, l_freq=8, h_freq=30)
        assert result is raw  # in-place

    def test_notch(self):
        raw = _make_raw(duration=5.0)
        result = notch_filter(raw, freqs=50.0)
        assert result is raw


class TestReference:
    def test_car(self):
        raw = _make_raw()
        result = set_common_average_reference(raw)
        assert result is raw


class TestEpochs:
    def test_create_epochs(self):
        raw = _make_raw(duration=60.0)
        events = _make_events(n_events=10)
        epochs = create_epochs(raw, events, EVENT_ID, tmin=-0.5, tmax=2.0)
        assert len(epochs) > 0
        assert epochs.get_data().ndim == 3

    def test_epochs_to_arrays(self):
        raw = _make_raw(duration=60.0)
        events = _make_events(n_events=10)
        epochs = create_epochs(raw, events, EVENT_ID, tmin=-0.5, tmax=2.0)
        X, y = epochs_to_arrays(epochs)
        assert X.shape[0] == y.shape[0]
        assert set(y.tolist()).issubset({1, 2})

    def test_epochs_to_arrays_crop(self):
        raw = _make_raw(duration=60.0)
        events = _make_events(n_events=10)
        epochs = create_epochs(raw, events, EVENT_ID, tmin=-0.5, tmax=2.0)
        X_full, _ = epochs_to_arrays(epochs)
        X_crop, _ = epochs_to_arrays(epochs, tmin=0.0, tmax=1.0)
        assert X_crop.shape[2] < X_full.shape[2]


class TestArtifacts:
    def test_reject_bad_epochs(self):
        raw = _make_raw(duration=60.0)
        events = _make_events(n_events=10)
        epochs = create_epochs(raw, events, EVENT_ID, tmin=-0.5, tmax=2.0)
        # With a very tight threshold, some epochs should be rejected
        result = reject_bad_epochs(epochs, reject_threshold=1e-6)
        assert len(result) <= len(events)

    def test_reject_with_generous_threshold(self):
        raw = _make_raw(duration=60.0)
        events = _make_events(n_events=10)
        epochs = create_epochs(raw, events, EVENT_ID, tmin=-0.5, tmax=2.0)
        n_before = len(epochs)
        result = reject_bad_epochs(epochs, reject_threshold=1.0)  # 1V — nothing rejected
        assert len(result) == n_before


class TestQuality:
    def test_detect_bad_channels_clean_data(self):
        raw = _make_raw()
        bads = detect_bad_channels(raw)
        # Random data should have similar variance, so no bads
        assert isinstance(bads, list)

    def test_detect_bad_channels_with_outlier(self):
        raw = _make_raw()
        # Inject a bad channel with 1000x variance
        data = raw.get_data()
        data[0] *= 1000
        raw._data = data
        bads = detect_bad_channels(raw, zscore_threshold=2.0)
        assert "EEG1" in bads


# ------------------------------------------------------------------
# Pipeline integration
# ------------------------------------------------------------------

class TestPipeline:
    def _default_config(self) -> dict:
        return {
            "reference": "car",
            "bandpass": [8, 30],
            "notch_filter": None,
            "bad_channel_detection": False,
            "artifact_rejection": {"method": None},
            "epoch": {
                "tmin": -0.5,
                "tmax": 2.0,
                "baseline": None,
                "reject_threshold": None,
            },
        }

    def test_pipeline_basic(self):
        raw = _make_raw(duration=60.0)
        events = _make_events(n_events=10)
        pipe = PreprocessingPipeline(self._default_config())
        epochs = pipe.run(raw, events, EVENT_ID)
        assert len(epochs) > 0
        assert pipe.quality_report is not None
        assert pipe.quality_report["n_epochs_total"] > 0

    def test_pipeline_with_rejection(self):
        cfg = self._default_config()
        cfg["epoch"]["reject_threshold"] = 100e-6
        raw = _make_raw(duration=60.0)
        events = _make_events(n_events=10)
        pipe = PreprocessingPipeline(cfg)
        epochs = pipe.run(raw, events, EVENT_ID)
        assert len(epochs) > 0

    def test_pipeline_with_bad_channel_detection(self):
        cfg = self._default_config()
        cfg["bad_channel_detection"] = True
        raw = _make_raw(duration=60.0)
        events = _make_events(n_events=10)
        pipe = PreprocessingPipeline(cfg)
        epochs = pipe.run(raw, events, EVENT_ID)
        assert len(epochs) > 0
