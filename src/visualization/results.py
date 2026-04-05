"""Result visualization: accuracy bars, confusion matrices, ROC curves, boxplots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_accuracy_comparison(
    results: dict[str, dict[str, float]],
    title: str = "Method Comparison",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing mean accuracy ± std across methods.

    Parameters
    ----------
    results : {method_name: {"mean": float, "std": float}}
    """
    methods = list(results.keys())
    means = [results[m]["mean"] for m in methods]
    stds = [results[m]["std"] for m in methods]

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.5), 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#4C72B0", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)

    # Value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
            f"{m:.1%}", ha="center", va="bottom", fontsize=10,
        )

    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray | list[list[int]],
    labels: list[str | int],
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot confusion matrix as a heatmap.

    Parameters
    ----------
    cm : (n_classes, n_classes) array or nested list
    labels : class labels for axis ticks
    """
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_roc_curves(
    roc_data: dict[str, dict[str, Any]],
    title: str = "ROC Curves",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot ROC curves for multiple methods.

    Parameters
    ----------
    roc_data : {method_name: {"y_true": array, "y_proba": array}}
        Each entry has ground truth and predicted probabilities for the
        positive class.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))

    for (name, data), color in zip(roc_data.items(), colors):
        y_true = np.asarray(data["y_true"])
        y_proba = np.asarray(data["y_proba"])

        # For binary: use the probability of the positive class
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]

        # Binarize labels: map to 0/1
        classes = sorted(np.unique(y_true))
        if len(classes) == 2:
            y_binary = (y_true == classes[1]).astype(int)
        else:
            y_binary = y_true

        fpr, tpr, _ = roc_curve(y_binary, y_proba)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_subject_boxplot(
    accuracy_dict: dict[str, list[float]],
    title: str = "Accuracy Distribution by Method",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Boxplot of per-subject accuracies for each method.

    Parameters
    ----------
    accuracy_dict : {method_name: [acc_sub1, acc_sub2, ...]}
    """
    methods = list(accuracy_dict.keys())
    data = [accuracy_dict[m] for m in methods]

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.5), 5))
    bp = ax.boxplot(data, labels=methods, patch_artist=True, showmeans=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticklabels(methods, rotation=30, ha="right")

    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
