"""Model base class, registry, and serialisation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.utils.logging import get_logger
from src.utils.paths import ensure_dir, get_models_dir

logger = get_logger(__name__)


class BaseModel(ABC):
    """Uniform interface for all classifiers (traditional and deep)."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (n_samples, n_classes)."""
        ...

    def save(self, path: str | Path) -> Path:
        """Serialise the model with joblib."""
        path = Path(path)
        ensure_dir(path.parent)
        joblib.dump(self, path)
        logger.info("Model saved: %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> BaseModel:
        """Load a previously saved model."""
        return joblib.load(path)
