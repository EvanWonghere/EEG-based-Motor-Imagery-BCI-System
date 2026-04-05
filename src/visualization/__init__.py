"""Visualization: topomaps, ERD/ERS, result plots, auto-report."""

from src.visualization.topo import plot_csp_patterns
from src.visualization.erds import plot_erds_maps
from src.visualization.results import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_subject_boxplot,
)
from src.visualization.report import generate_report

__all__ = [
    "plot_csp_patterns",
    "plot_erds_maps",
    "plot_accuracy_comparison",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_subject_boxplot",
    "generate_report",
]
