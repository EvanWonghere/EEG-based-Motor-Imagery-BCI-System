"""Tests for the evaluation framework."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics
from src.evaluation.cross_validation import run_cv, CVResult
from src.evaluation.statistics import (
    confidence_interval,
    wilcoxon_test,
    paired_permutation_test,
    friedman_nemenyi,
)
from src.evaluation.comparison import ComparisonTable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predictions(n=100, seed=42):
    rng = np.random.default_rng(seed)
    y_true = rng.choice([0, 1], size=n)
    # ~80% accuracy
    y_pred = y_true.copy()
    flip_idx = rng.choice(n, size=n // 5, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    y_proba = np.zeros((n, 2))
    for i in range(n):
        if y_pred[i] == 1:
            y_proba[i, 1] = rng.uniform(0.6, 1.0)
            y_proba[i, 0] = 1 - y_proba[i, 1]
        else:
            y_proba[i, 0] = rng.uniform(0.6, 1.0)
            y_proba[i, 1] = 1 - y_proba[i, 0]
    return y_true, y_pred, y_proba


# ---------------------------------------------------------------------------
# Test metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_basic_metrics(self):
        y_true, y_pred, y_proba = _make_predictions()
        result = compute_metrics(y_true, y_pred, y_proba)
        assert "accuracy" in result
        assert "kappa" in result
        assert "confusion_matrix" in result
        assert 0 <= result["accuracy"] <= 1

    def test_perfect_predictions(self):
        y = np.array([0, 1, 0, 1, 0])
        result = compute_metrics(y, y)
        assert result["accuracy"] == 1.0
        assert result["kappa"] == 1.0

    def test_roc_auc(self):
        y_true, y_pred, y_proba = _make_predictions()
        result = compute_metrics(y_true, y_pred, y_proba, metrics=["roc_auc"])
        assert "roc_auc" in result
        assert 0.5 <= result["roc_auc"] <= 1.0

    def test_roc_auc_no_proba(self):
        y_true, y_pred, _ = _make_predictions()
        result = compute_metrics(y_true, y_pred, metrics=["roc_auc"])
        assert np.isnan(result["roc_auc"])

    def test_confusion_matrix_shape(self):
        y_true, y_pred, _ = _make_predictions()
        result = compute_metrics(y_true, y_pred)
        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2


# ---------------------------------------------------------------------------
# Test cross-validation
# ---------------------------------------------------------------------------

class TestCrossValidation:
    def test_run_cv_basic(self):
        """CV with a simple sklearn-compatible model."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 4))
        y = np.concatenate([np.zeros(40), np.ones(40)]).astype(int)

        def model_factory():
            class _Wrapper:
                def __init__(self):
                    self._clf = LinearDiscriminantAnalysis()
                def fit(self, X, y):
                    self._clf.fit(X, y)
                def predict(self, X):
                    return self._clf.predict(X)
                def predict_proba(self, X):
                    return self._clf.predict_proba(X)
            return _Wrapper()

        result = run_cv(X, y, extractor=None, model_factory=model_factory, n_splits=5)
        assert result.n_splits == 5
        assert len(result.folds) == 5
        assert "accuracy" in result.mean_metrics
        assert "kappa" in result.mean_metrics

    def test_cv_result_json(self, tmp_path):
        """CVResult can be serialised to JSON."""
        from src.evaluation.cross_validation import FoldResult

        fold = FoldResult(
            fold=0,
            y_true=[0, 1, 0],
            y_pred=[0, 1, 1],
            y_proba=None,
            metrics={"accuracy": 0.667, "kappa": 0.333},
        )
        cv = CVResult(
            n_splits=1,
            metric_names=["accuracy", "kappa"],
            folds=[fold],
        )
        cv.aggregate()
        path = cv.save_json(tmp_path / "test_cv.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["n_splits"] == 1


# ---------------------------------------------------------------------------
# Test statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_confidence_interval(self):
        scores = [0.8, 0.85, 0.9, 0.75, 0.88]
        low, high = confidence_interval(scores)
        mean = np.mean(scores)
        assert low < mean < high
        assert low > 0
        assert high < 1

    def test_wilcoxon_identical(self):
        a = [0.8, 0.85, 0.9]
        stat, p = wilcoxon_test(a, a)
        assert p == 1.0

    def test_wilcoxon_different(self):
        a = [0.9, 0.92, 0.95, 0.88, 0.91, 0.93, 0.90, 0.94]
        b = [0.6, 0.62, 0.65, 0.58, 0.61, 0.63, 0.60, 0.64]
        stat, p = wilcoxon_test(a, b)
        assert p < 0.05

    def test_paired_permutation(self):
        a = [0.95, 0.93, 0.92, 0.94, 0.91, 0.96, 0.93, 0.92]
        b = [0.60, 0.62, 0.58, 0.61, 0.59, 0.63, 0.60, 0.61]
        p, diff = paired_permutation_test(a, b)
        assert diff > 0
        assert p < 0.05

    def test_friedman_nemenyi(self):
        # 9 subjects, 3 methods
        rng = np.random.default_rng(42)
        scores = np.column_stack([
            rng.uniform(0.8, 1.0, 9),   # method A (best)
            rng.uniform(0.6, 0.8, 9),   # method B
            rng.uniform(0.4, 0.6, 9),   # method C (worst)
        ])
        result = friedman_nemenyi(scores, ["A", "B", "C"])
        assert result.p_value < 0.05
        assert result.mean_ranks["A"] < result.mean_ranks["C"]
        assert result.cd > 0


# ---------------------------------------------------------------------------
# Test comparison table
# ---------------------------------------------------------------------------

class TestComparisonTable:
    def test_latex_output(self):
        table = ComparisonTable(
            method_names=["CSP+LDA", "FBCSP+SVM"],
            subject_ids=[1, 2, 3],
            accuracy_matrix=np.array([
                [0.85, 0.90],
                [0.80, 0.88],
                [0.75, 0.82],
            ]),
        )
        table.analyze()
        latex = table.to_latex()
        assert r"\begin{tabular}" in latex
        assert "CSP+LDA" in latex
        assert r"\textbf{" in latex

    def test_to_dict(self):
        table = ComparisonTable(
            method_names=["A", "B"],
            subject_ids=[1, 2],
            accuracy_matrix=np.array([[0.8, 0.9], [0.7, 0.85]]),
        )
        table.analyze()
        d = table.to_dict()
        assert "per_method" in d
        assert "A" in d["per_method"]
