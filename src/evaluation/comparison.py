"""Multi-method comparison: load results, build comparison tables, run stats."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.statistics import (
    confidence_interval,
    friedman_nemenyi,
    wilcoxon_test,
    FriedmanResult,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonTable:
    """Subject × method accuracy matrix with statistical analysis."""

    method_names: list[str]
    subject_ids: list[int]
    accuracy_matrix: np.ndarray  # (n_subjects, n_methods)
    kappa_matrix: np.ndarray | None = None

    # Populated by analyze()
    mean_accuracy: dict[str, float] = field(default_factory=dict)
    std_accuracy: dict[str, float] = field(default_factory=dict)
    ci_accuracy: dict[str, tuple[float, float]] = field(default_factory=dict)
    friedman: FriedmanResult | None = None
    pairwise_wilcoxon: dict[str, tuple[float, float]] = field(default_factory=dict)

    def analyze(self, alpha: float = 0.05) -> None:
        """Run all statistical analyses."""
        n_methods = len(self.method_names)

        for j, name in enumerate(self.method_names):
            scores = self.accuracy_matrix[:, j]
            self.mean_accuracy[name] = float(np.mean(scores))
            self.std_accuracy[name] = float(np.std(scores))
            self.ci_accuracy[name] = confidence_interval(scores)

        # Friedman + Nemenyi (need >= 3 methods and >= 3 subjects)
        if n_methods >= 3 and len(self.subject_ids) >= 3:
            self.friedman = friedman_nemenyi(
                self.accuracy_matrix, self.method_names, alpha=alpha,
            )

        # Pairwise Wilcoxon (need >= 5 subjects for meaningful results)
        if len(self.subject_ids) >= 5:
            for i in range(n_methods):
                for j in range(i + 1, n_methods):
                    key = f"{self.method_names[i]} vs {self.method_names[j]}"
                    stat, p = wilcoxon_test(
                        self.accuracy_matrix[:, i],
                        self.accuracy_matrix[:, j],
                    )
                    self.pairwise_wilcoxon[key] = (stat, p)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "method_names": self.method_names,
            "subject_ids": self.subject_ids,
            "per_method": {},
        }
        for name in self.method_names:
            d["per_method"][name] = {
                "mean_accuracy": self.mean_accuracy.get(name),
                "std_accuracy": self.std_accuracy.get(name),
                "ci_95": self.ci_accuracy.get(name),
            }
        if self.friedman:
            d["friedman"] = self.friedman.to_dict()
        if self.pairwise_wilcoxon:
            d["pairwise_wilcoxon"] = {
                k: {"statistic": v[0], "p_value": v[1]}
                for k, v in self.pairwise_wilcoxon.items()
            }
        return d

    def to_latex(self) -> str:
        """Generate a LaTeX table of subject × method accuracy."""
        methods = self.method_names
        header = " & ".join(["Subject"] + methods) + r" \\"
        lines = [
            r"\begin{tabular}{" + "c" * (len(methods) + 1) + "}",
            r"\toprule",
            header,
            r"\midrule",
        ]
        for i, sid in enumerate(self.subject_ids):
            row = self.accuracy_matrix[i]
            best_idx = int(np.argmax(row))
            cells = [str(sid)]
            for j, val in enumerate(row):
                s = f"{val:.1%}"
                if j == best_idx:
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\midrule")
        # Mean ± std row
        cells = ["Mean"]
        for name in methods:
            m = self.mean_accuracy.get(name, 0)
            s = self.std_accuracy.get(name, 0)
            cells.append(f"{m:.1%} ± {s:.1%}")
        lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)


def load_experiment_results(results_dir: str | Path) -> dict[str, dict]:
    """Load all results.json files from results/ subdirectories.

    Returns {experiment_name: results_dict}.
    """
    results_dir = Path(results_dir)
    experiments = {}
    for json_path in sorted(results_dir.glob("*/results.json")):
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        name = data.get("experiment", json_path.parent.name)
        experiments[name] = data
    return experiments


def compare_methods(
    results_dir: str | Path,
    method_order: list[str] | None = None,
) -> ComparisonTable:
    """Build a ComparisonTable from all experiments in results_dir.

    Each experiment must have ``results.json`` with per-subject accuracies.
    Only subjects present in ALL experiments are included.
    """
    experiments = load_experiment_results(results_dir)
    if not experiments:
        raise FileNotFoundError(f"No results.json found in {results_dir}")

    # Build subject → accuracy mapping per experiment
    method_data: dict[str, dict[int, float]] = {}
    method_kappa: dict[str, dict[int, float]] = {}
    for exp_name, data in experiments.items():
        acc_map: dict[int, float] = {}
        kappa_map: dict[int, float] = {}
        for sub in data.get("subjects", []):
            if not sub.get("skipped"):
                acc_map[sub["subject"]] = sub["mean_accuracy"]
                kappa_map[sub["subject"]] = sub.get("mean_kappa", 0)
        method_data[exp_name] = acc_map
        method_kappa[exp_name] = kappa_map

    # Find common subjects
    all_subjects_sets = [set(m.keys()) for m in method_data.values()]
    common_subjects = sorted(set.intersection(*all_subjects_sets)) if all_subjects_sets else []

    if not common_subjects:
        raise ValueError("No common subjects found across experiments")

    if method_order:
        methods = [m for m in method_order if m in method_data]
    else:
        methods = list(method_data.keys())

    n_sub = len(common_subjects)
    n_meth = len(methods)
    acc_matrix = np.zeros((n_sub, n_meth))
    kappa_matrix = np.zeros((n_sub, n_meth))

    for j, mname in enumerate(methods):
        for i, sid in enumerate(common_subjects):
            acc_matrix[i, j] = method_data[mname][sid]
            kappa_matrix[i, j] = method_kappa[mname].get(sid, 0)

    table = ComparisonTable(
        method_names=methods,
        subject_ids=common_subjects,
        accuracy_matrix=acc_matrix,
        kappa_matrix=kappa_matrix,
    )
    table.analyze()
    return table
