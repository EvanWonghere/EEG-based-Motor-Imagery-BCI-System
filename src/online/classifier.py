"""Online classifier: load trained model + extractor for real-time inference."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

from src.models.base import BaseModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Event code → human-readable label
LABEL_MAP = {769: "left_hand", 770: "right_hand"}


class OnlineClassifier:
    """Load a trained model (+ optional feature extractor) for online prediction.

    Parameters
    ----------
    model_path : str or Path
        Path to a saved ``.pkl`` model file.
    extractor_path : str or Path or None
        Path to a saved fitted extractor (CSP / FBCSP).
        If ``None``, the model receives raw epoch data directly (e.g. EEGNet).
    """

    def __init__(
        self,
        model_path: str | Path,
        extractor_path: str | Path | None = None,
    ):
        self.model: BaseModel = BaseModel.load(model_path)
        logger.info("Model loaded: %s", model_path)

        self.extractor = None
        if extractor_path is not None:
            ext_path = Path(extractor_path)
            if ext_path.exists():
                self.extractor = joblib.load(ext_path)
                logger.info("Extractor loaded: %s", ext_path)

    def predict_trial(self, epoch: np.ndarray) -> dict:
        """Classify a single trial.

        Parameters
        ----------
        epoch : (1, n_channels, n_times) or (n_channels, n_times)
            A single preprocessed EEG epoch.

        Returns
        -------
        dict
            ``{label: int, label_name: str, confidence: float, latency_ms: float}``
        """
        if epoch.ndim == 2:
            epoch = epoch[np.newaxis, ...]  # (1, ch, t)

        t0 = time.perf_counter()

        if self.extractor is not None:
            features = self.extractor.transform(epoch)
        else:
            features = epoch

        pred = self.model.predict(features)
        label = int(pred[0])

        confidence = 0.5
        try:
            proba = self.model.predict_proba(features)
            confidence = float(np.max(proba[0]))
        except (AttributeError, NotImplementedError, IndexError):
            pass

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "label": label,
            "label_name": LABEL_MAP.get(label, str(label)),
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2),
        }

    @staticmethod
    def infer_extractor_path(model_path: str | Path) -> Path | None:
        """Guess extractor path from model path convention.

        ``{name}_sub{id}.pkl`` → ``{name}_sub{id}_extractor.pkl``
        """
        model_path = Path(model_path)
        ext_path = model_path.with_name(
            model_path.stem + "_extractor" + model_path.suffix
        )
        return ext_path if ext_path.exists() else None
