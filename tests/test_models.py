"""Tests for classification models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import create_model
from src.models.traditional import TraditionalModel


def _make_features(n=60, d=4, seed=42):
    """Synthetic 2D feature data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = np.array([0, 1] * (n // 2))
    X[y == 0] += 0.5
    X[y == 1] -= 0.5
    return X, y


class TestTraditional:
    @pytest.mark.parametrize("model_type", ["lda", "svm", "rf"])
    def test_fit_predict(self, model_type):
        X, y = _make_features()
        m = TraditionalModel(model_type=model_type)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (60,)

    @pytest.mark.parametrize("model_type", ["lda", "svm", "rf"])
    def test_predict_proba(self, model_type):
        X, y = _make_features()
        m = TraditionalModel(model_type=model_type)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (60, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_save_load(self, tmp_path):
        X, y = _make_features()
        m = TraditionalModel(model_type="lda")
        m.fit(X, y)
        path = m.save(tmp_path / "test_model.pkl")
        loaded = TraditionalModel.load(path)
        np.testing.assert_array_equal(m.predict(X), loaded.predict(X))

    def test_unknown_type(self):
        with pytest.raises(ValueError):
            TraditionalModel(model_type="unknown")


class TestFactory:
    def test_create_lda(self):
        m = create_model({"type": "lda", "params": {}})
        assert isinstance(m, TraditionalModel)
        assert m.model_type == "lda"

    def test_create_svm_with_params(self):
        m = create_model({"type": "svm", "params": {"C": 10.0, "kernel": "rbf"}})
        assert isinstance(m, TraditionalModel)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_model({"type": "nonexistent"})


# EEGNet tests (skipped if no PyTorch)
HAS_TORCH = False
try:
    import torch
    # PyTorch Conv2d segfaults on Python 3.14 — skip until upstream fix
    if sys.version_info >= (3, 14):
        HAS_TORCH = False
    else:
        HAS_TORCH = True
except ImportError:
    pass


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available or unstable on this Python version")
class TestEEGNet:
    def _make_epochs(self, n=40, ch=8, t=250):
        rng = np.random.RandomState(42)
        X = rng.randn(n, ch, t).astype(np.float32) * 20e-6
        y = np.array([0, 1] * (n // 2))
        return X, y

    def test_fit_predict(self):
        from src.models.eegnet import EEGNetModel
        X, y = self._make_epochs()
        m = EEGNetModel(n_epochs=5, patience=3, batch_size=16)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (40,)

    def test_predict_proba(self):
        from src.models.eegnet import EEGNetModel
        X, y = self._make_epochs()
        m = EEGNetModel(n_epochs=5, patience=3, batch_size=16)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (40, 2)
