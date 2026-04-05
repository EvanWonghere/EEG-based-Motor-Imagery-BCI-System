"""Cross-validation with full per-fold metric collection.

Replaces the simple ``cross_val_score`` in train.py with a richer
evaluation that captures predictions, probabilities, and multiple metrics
per fold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import compute_metrics, DEFAULT_METRICS
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FoldResult:
    """Results from a single CV fold."""

    fold: int
    y_true: list[int]
    y_pred: list[int]
    y_proba: list[list[float]] | None
    metrics: dict[str, Any]


@dataclass
class CVResult:
    """Aggregated cross-validation results."""

    n_splits: int
    metric_names: list[str]
    folds: list[FoldResult] = field(default_factory=list)

    # Aggregated (computed after all folds)
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)
    # Full concatenated predictions for overall confusion matrix
    all_y_true: list[int] = field(default_factory=list)
    all_y_pred: list[int] = field(default_factory=list)
    overall_metrics: dict[str, Any] = field(default_factory=dict)

    def aggregate(self) -> None:
        """Compute mean/std from per-fold metrics and overall metrics."""
        if not self.folds:
            return

        # Per-metric mean and std across folds
        for name in self.metric_names:
            vals = [f.metrics[name] for f in self.folds if name in f.metrics]
            if vals:
                self.mean_metrics[name] = float(np.mean(vals))
                self.std_metrics[name] = float(np.std(vals))

        # Concatenate all predictions for overall confusion matrix
        self.all_y_true = []
        self.all_y_pred = []
        all_proba = []
        for f in self.folds:
            self.all_y_true.extend(f.y_true)
            self.all_y_pred.extend(f.y_pred)
            if f.y_proba is not None:
                all_proba.extend(f.y_proba)

        proba_arr = np.array(all_proba) if all_proba else None
        self.overall_metrics = compute_metrics(
            np.array(self.all_y_true),
            np.array(self.all_y_pred),
            y_proba=proba_arr,
            metrics=self.metric_names,
        )

    def to_dict(self, include_predictions: bool = True) -> dict[str, Any]:
        """Serialisable dict (for JSON).

        Parameters
        ----------
        include_predictions : bool
            If True, include per-fold y_true, y_pred, y_proba
            (needed for ROC curves and detailed analysis).
        """
        fold_dicts = []
        for f in self.folds:
            fd: dict[str, Any] = {
                "fold": f.fold,
                "metrics": f.metrics,
                "n_samples": len(f.y_true),
            }
            if include_predictions:
                fd["y_true"] = f.y_true
                fd["y_pred"] = f.y_pred
                fd["y_proba"] = f.y_proba
            fold_dicts.append(fd)

        return {
            "n_splits": self.n_splits,
            "metric_names": self.metric_names,
            "mean_metrics": self.mean_metrics,
            "std_metrics": self.std_metrics,
            "overall_metrics": self.overall_metrics,
            "folds": fold_dicts,
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return path


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    extractor: Any | None,
    model_factory: Callable[[], Any],
    n_splits: int = 10,
    seed: int = 42,
    metrics: list[str] | None = None,
    is_deep: bool = False,
) -> CVResult:
    """Run stratified K-fold CV with full metric collection.

    Parameters
    ----------
    X : (n_trials, n_channels, n_times) or (n_trials, n_features)
    y : (n_trials,)
    extractor : feature extractor with fit_transform/transform, or None
    model_factory : callable returning a fresh model instance per fold
    n_splits : number of folds
    seed : random seed
    metrics : list of metric names (default: DEFAULT_METRICS + roc_auc)
    is_deep : if True, skip extractor (deep models take raw epochs)
    """
    if metrics is None:
        metrics = list(DEFAULT_METRICS) + ["roc_auc"]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = CVResult(n_splits=n_splits, metric_names=metrics)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_factory()

        if extractor is not None and not is_deep:
            # Clone extractor for each fold to avoid data leakage
            import copy
            ext = copy.deepcopy(extractor)
            X_train_feat = ext.fit_transform(X_train, y_train)
            X_test_feat = ext.transform(X_test)
        else:
            X_train_feat = X_train
            X_test_feat = X_test

        model.fit(X_train_feat, y_train)
        y_pred = model.predict(X_test_feat)

        # Probabilities
        y_proba = None
        try:
            y_proba = model.predict_proba(X_test_feat)
        except (AttributeError, NotImplementedError):
            pass

        fold_metrics = compute_metrics(y_test, y_pred, y_proba=y_proba, metrics=metrics)

        fold_result = FoldResult(
            fold=fold_idx,
            y_true=y_test.tolist(),
            y_pred=y_pred.tolist(),
            y_proba=y_proba.tolist() if y_proba is not None else None,
            metrics=fold_metrics,
        )
        result.folds.append(fold_result)

        acc = fold_metrics.get("accuracy", 0)
        kappa = fold_metrics.get("kappa", 0)
        logger.info(
            "  Fold %d/%d: acc=%.2f%% kappa=%.3f",
            fold_idx + 1, n_splits, acc * 100, kappa,
        )

    result.aggregate()
    return result
