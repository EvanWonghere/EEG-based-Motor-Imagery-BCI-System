"""Auto-generate all thesis figures from experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.comparison import load_experiment_results, compare_methods
from src.visualization.results import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_subject_boxplot,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_report(
    results_dir: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate all figures from experiment results.

    Returns list of generated file paths.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    experiments = load_experiment_results(results_dir)
    if not experiments:
        logger.warning("No experiments found in %s", results_dir)
        return generated

    # --- 1. Accuracy comparison bar chart ---
    acc_data = {}
    for name, data in experiments.items():
        valid = [s for s in data.get("subjects", []) if not s.get("skipped")]
        if valid:
            accs = [s["mean_accuracy"] for s in valid]
            acc_data[name] = {"mean": np.mean(accs), "std": np.std(accs)}

    if acc_data:
        path = output_dir / "accuracy_comparison.png"
        plot_accuracy_comparison(acc_data, save_path=path)
        plt.close()
        generated.append(path)
        logger.info("Generated %s", path)

    # --- 2. Per-experiment confusion matrices ---
    for exp_name, data in experiments.items():
        # Load per-subject CV results for overall confusion matrix
        exp_dir = results_dir / exp_name
        cm_all = None
        labels = None
        for cv_json in sorted(exp_dir.glob("cv_sub*.json")):
            with open(cv_json, encoding="utf-8") as fh:
                cv_data = json.load(fh)
            om = cv_data.get("overall_metrics", {})
            cm = om.get("confusion_matrix")
            if cm is not None:
                cm_arr = np.array(cm)
                if cm_all is None:
                    cm_all = cm_arr
                    labels = om.get("labels", list(range(cm_arr.shape[0])))
                else:
                    cm_all = cm_all + cm_arr

        if cm_all is not None:
            label_map = {769: "Left Hand", 770: "Right Hand"}
            display_labels = [label_map.get(lb, str(lb)) for lb in labels]
            path = output_dir / f"confusion_matrix_{exp_name}.png"
            plot_confusion_matrix(cm_all, display_labels, title=f"{exp_name}", save_path=path)
            plt.close()
            generated.append(path)
            logger.info("Generated %s", path)

    # --- 3. ROC curves (from per-subject CV data) ---
    roc_data = {}
    for exp_name, data in experiments.items():
        exp_dir = results_dir / exp_name
        all_y_true = []
        all_y_proba = []
        for cv_json in sorted(exp_dir.glob("cv_sub*.json")):
            with open(cv_json, encoding="utf-8") as fh:
                cv_data = json.load(fh)
            for fold in cv_data.get("folds", []):
                y_true = fold.get("y_true")
                y_proba = fold.get("y_proba")
                if y_true and y_proba:
                    all_y_true.extend(y_true)
                    all_y_proba.extend(y_proba)

        if all_y_true and all_y_proba:
            roc_data[exp_name] = {
                "y_true": np.array(all_y_true),
                "y_proba": np.array(all_y_proba),
            }

    if roc_data:
        path = output_dir / "roc_curves.png"
        plot_roc_curves(roc_data, save_path=path)
        plt.close()
        generated.append(path)
        logger.info("Generated %s", path)

    # --- 4. Subject boxplot ---
    if len(experiments) > 1:
        box_data = {}
        for name, data in experiments.items():
            valid = [s for s in data.get("subjects", []) if not s.get("skipped")]
            if valid:
                box_data[name] = [s["mean_accuracy"] for s in valid]

        if box_data and any(len(v) > 1 for v in box_data.values()):
            path = output_dir / "subject_boxplot.png"
            plot_subject_boxplot(box_data, save_path=path)
            plt.close()
            generated.append(path)
            logger.info("Generated %s", path)

    # --- 5. Comparison table (if multiple methods) ---
    if len(experiments) >= 2:
        try:
            table = compare_methods(results_dir)
            latex_path = output_dir / "comparison_table.tex"
            latex_path.write_text(table.to_latex(), encoding="utf-8")
            generated.append(latex_path)
            logger.info("Generated %s", latex_path)
        except (ValueError, FileNotFoundError) as e:
            logger.warning("Skipping comparison table: %s", e)

    return generated
