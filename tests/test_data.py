"""Tests for the data management layer.

These tests verify that:
1. The dataset registry and factory work
2. Dataset classes instantiate correctly
3. Data loading works when files are present (skipped if data is missing)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import create_dataset, DATASET_REGISTRY, BCIIV2aDataset, BCIIV2bDataset
from src.data.base import EEGDataset
from src.utils.paths import get_data_dir


# ------------------------------------------------------------------
# Registry & factory
# ------------------------------------------------------------------

def test_registry_has_datasets():
    assert "bci_iv_2a" in DATASET_REGISTRY
    assert "bci_iv_2b" in DATASET_REGISTRY


def test_factory_creates_dataset():
    ds = create_dataset("bci_iv_2a", data_dir=get_data_dir())
    assert isinstance(ds, EEGDataset)
    assert isinstance(ds, BCIIV2aDataset)


def test_factory_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        create_dataset("nonexistent", data_dir=".")


# ------------------------------------------------------------------
# Dataset properties
# ------------------------------------------------------------------

def test_2a_properties():
    ds = BCIIV2aDataset(data_dir=get_data_dir())
    assert ds.n_subjects == 9
    assert ds.sfreq == 250.0
    assert len(ds.ch_names) == 22
    assert ds.subjects == list(range(1, 10))
    event_id = ds.get_event_id()
    assert event_id["Left Hand"] == 769
    assert event_id["Right Hand"] == 770


def test_2b_properties():
    ds = BCIIV2bDataset(data_dir=get_data_dir())
    assert ds.n_subjects == 9
    assert ds.sfreq == 250.0
    assert len(ds.ch_names) == 3
    event_id = ds.get_event_id()
    assert event_id["Left Hand"] == 769


# ------------------------------------------------------------------
# Data loading (requires downloaded data — skip if missing)
# ------------------------------------------------------------------

def _data_available(dataset_cls: type[EEGDataset]) -> bool:
    ds = dataset_cls(data_dir=get_data_dir())
    try:
        ds.load_data(subject_id=1)
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(
    not _data_available(BCIIV2aDataset),
    reason="BCI IV 2a data not downloaded",
)
class TestBCIIV2aLoading:
    def test_load_data_returns_triple(self):
        ds = BCIIV2aDataset(data_dir=get_data_dir())
        raw, events, event_id = ds.load_data(subject_id=1)
        assert hasattr(raw, "info")
        assert events.ndim == 2 and events.shape[1] == 3
        assert "Left Hand" in event_id
        assert "Right Hand" in event_id

    def test_load_raw(self):
        ds = BCIIV2aDataset(data_dir=get_data_dir())
        raw = ds.load_raw(subject_id=1)
        assert raw.info["sfreq"] > 0


@pytest.mark.skipif(
    not _data_available(BCIIV2bDataset),
    reason="BCI IV 2b data not downloaded",
)
class TestBCIIV2bLoading:
    def test_load_data_returns_triple(self):
        ds = BCIIV2bDataset(data_dir=get_data_dir())
        raw, events, event_id = ds.load_data(subject_id=1)
        assert events.shape[1] == 3
        assert len(event_id) >= 2
