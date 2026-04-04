"""Classification models: traditional (LDA/SVM/RF) and deep learning (EEGNet)."""

from __future__ import annotations

from typing import Any

from src.models.base import BaseModel
from src.models.traditional import TraditionalModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "lda": TraditionalModel,
    "svm": TraditionalModel,
    "rf": TraditionalModel,
}

# EEGNet registered only if PyTorch is available
try:
    from src.models.eegnet import EEGNetModel
    MODEL_REGISTRY["eegnet"] = EEGNetModel
except ImportError:
    EEGNetModel = None  # type: ignore[assignment,misc]


def create_model(config: dict[str, Any], **kwargs: Any) -> BaseModel:
    """Instantiate a model from a ``model`` config block.

    Parameters
    ----------
    config : dict
        The ``model`` section of an experiment config::

            type: "svm"
            params:
              kernel: "rbf"
              C: 1.0
    **kwargs
        Extra keyword arguments forwarded to the constructor (e.g.
        ``n_channels``, ``n_times`` for EEGNet).
    """
    model_type = config["type"].lower()
    params = dict(config.get("params") or {})

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type {model_type!r}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[model_type]

    if cls is TraditionalModel:
        return TraditionalModel(model_type=model_type, params=params)

    # Deep learning models (EEGNet etc.) — merge params + kwargs
    params.update(kwargs)
    return cls(**params)


__all__ = [
    "BaseModel",
    "TraditionalModel",
    "MODEL_REGISTRY",
    "create_model",
]
if EEGNetModel is not None:
    __all__.append("EEGNetModel")
