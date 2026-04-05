"""Evaluation framework: metrics, cross-validation, statistics, comparison."""

from src.evaluation.metrics import compute_metrics
from src.evaluation.cross_validation import run_cv, CVResult
from src.evaluation.statistics import (
    confidence_interval,
    wilcoxon_test,
    paired_permutation_test,
    friedman_nemenyi,
)
from src.evaluation.comparison import compare_methods, ComparisonTable

__all__ = [
    "compute_metrics",
    "run_cv",
    "CVResult",
    "confidence_interval",
    "wilcoxon_test",
    "paired_permutation_test",
    "friedman_nemenyi",
    "compare_methods",
    "ComparisonTable",
]
