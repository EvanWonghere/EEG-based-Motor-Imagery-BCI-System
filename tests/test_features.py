"""Tests for feature extraction modules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.csp import CSPExtractor
from src.features.fbcsp import FBCSPExtractor
from src.features.band_power import BandPowerExtractor
from src.features.selector import mutual_information_selection
from src.features import create_extractor


def _make_data(n_trials=40, n_channels=8, n_times=500, seed=42):
    """Synthetic 2-class EEG data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_trials, n_channels, n_times) * 20e-6
    y = np.array([0, 1] * (n_trials // 2))
    # Inject slight class difference on channel 0
    X[y == 0, 0, :] += 5e-6
    X[y == 1, 0, :] -= 5e-6
    return X, y


class TestCSP:
    def test_fit_transform(self):
        X, y = _make_data()
        csp = CSPExtractor(n_components=4)
        feats = csp.fit_transform(X, y)
        assert feats.shape == (40, 4)

    def test_patterns(self):
        X, y = _make_data()
        csp = CSPExtractor(n_components=4)
        csp.fit(X, y)
        patterns = csp.get_patterns()
        # patterns_ shape is (n_channels, n_channels) from MNE CSP
        assert patterns.ndim == 2
        assert patterns.shape[1] == 8


class TestFBCSP:
    def test_fit_transform(self):
        X, y = _make_data()
        fbcsp = FBCSPExtractor(
            filter_bands=[(4, 8), (8, 12), (12, 16)],
            n_components=2,
            n_select=4,
            sfreq=250.0,
        )
        feats = fbcsp.fit_transform(X, y)
        assert feats.shape == (40, 4)

    def test_band_info(self):
        X, y = _make_data()
        fbcsp = FBCSPExtractor(
            filter_bands=[(4, 8), (8, 12), (12, 16)],
            n_components=2,
            n_select=4,
            sfreq=250.0,
        )
        fbcsp.fit(X, y)
        info = fbcsp.get_selected_band_info()
        assert len(info) == 4
        assert all("band_range" in item for item in info)


class TestBandPower:
    def test_fit_transform(self):
        X, y = _make_data()
        bp = BandPowerExtractor(sfreq=250.0)
        feats = bp.fit_transform(X, y)
        # 8 channels × 4 bands = 32 features
        assert feats.shape == (40, 32)


class TestSelector:
    def test_mutual_info(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 20)
        y = (X[:, 0] > 0).astype(int)  # feature 0 is informative
        X_sel, indices = mutual_information_selection(X, y, n_select=5)
        assert X_sel.shape == (100, 5)
        assert 0 in indices


class TestFactory:
    def test_create_csp(self):
        ext = create_extractor({"method": "csp", "params": {"n_components": 4}})
        assert isinstance(ext, CSPExtractor)

    def test_create_fbcsp(self):
        ext = create_extractor({
            "method": "fbcsp",
            "params": {
                "filter_bands": [[4, 8], [8, 12]],
                "n_components": 2,
                "selection": {"method": "mutual_info", "n_select": 3},
            },
        }, sfreq=250.0)
        assert isinstance(ext, FBCSPExtractor)

    def test_create_null(self):
        ext = create_extractor({"method": None})
        assert ext is None
