"""Evaluation metrics for MI-BCI classification.

Computes accuracy, Cohen's kappa, F1, precision, recall, ROC-AUC,
and confusion matrix using scikit-learn.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Metrics that require only y_true and y_pred
_PRED_METRICS = {
    "accuracy": accuracy_score,
    "kappa": cohen_kappa_score,
    "f1_weighted": lambda y, p: f1_score(y, p, average="weighted"),
    "precision": lambda y, p: precision_score(y, p, average="weighted", zero_division=0),
    "recall": lambda y, p: recall_score(y, p, average="weighted", zero_division=0),
}

DEFAULT_METRICS = ["accuracy", "kappa", "f1_weighted"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute classification metrics.

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        Ground-truth labels.
    y_pred : array of shape (n_samples,)
        Predicted labels.
    y_proba : array of shape (n_samples, n_classes) or None
        Predicted class probabilities.  Required for ``roc_auc``.
    metrics : list of str or None
        Which metrics to compute.  ``None`` → :data:`DEFAULT_METRICS`.
        Supported: accuracy, kappa, f1_weighted, precision, recall, roc_auc.

    Returns
    -------
    dict
        ``{metric_name: value, ..., "confusion_matrix": [[...], ...]}``.
        Confusion matrix is always included.
    """
    if metrics is None:
        metrics = list(DEFAULT_METRICS)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result: dict[str, Any] = {}

    for name in metrics:
        if name in _PRED_METRICS:
            result[name] = float(_PRED_METRICS[name](y_true, y_pred))
        elif name == "roc_auc":
            if y_proba is not None:
                try:
                    n_classes = y_proba.shape[1] if y_proba.ndim == 2 else 2
                    if n_classes == 2:
                        result["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                    else:
                        result["roc_auc"] = float(
                            roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                        )
                except ValueError:
                    result["roc_auc"] = float("nan")
            else:
                result["roc_auc"] = float("nan")

    # Always include confusion matrix
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    result["labels"] = [int(lb) for lb in labels]

    return result
