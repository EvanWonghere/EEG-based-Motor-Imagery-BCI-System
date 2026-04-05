#!/usr/bin/env python
"""Evaluate and compare multiple experiments.

Usage:
    python scripts/evaluate.py --results-dir results/
    python scripts/evaluate.py --results-dir results/ --latex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.comparison import compare_methods, load_experiment_results
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MI-BCI experiments")
    parser.add_argument(
        "--results-dir", type=str, default="results/",
        help="Directory containing experiment result subdirectories",
    )
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save comparison JSON to this path",
    )
    args = parser.parse_args()

    setup_logging()

    results_dir = Path(args.results_dir)
    experiments = load_experiment_results(results_dir)

    if not experiments:
        print(f"No results.json found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(experiments)} experiment(s): {list(experiments.keys())}\n")

    # Compare
    try:
        table = compare_methods(results_dir)
    except ValueError as e:
        print(f"Cannot compare: {e}")
        sys.exit(1)

    # Print summary
    print("=" * 70)
    print("  Method Comparison")
    print("=" * 70)
    header = f"{'Subject':>8}" + "".join(f"  {m:>14}" for m in table.method_names)
    print(header)
    print("-" * 70)
    for i, sid in enumerate(table.subject_ids):
        row = f"  {sid:>5}"
        for j in range(len(table.method_names)):
            row += f"  {table.accuracy_matrix[i, j]:>13.2%}"
        print(row)

    print("-" * 70)
    row = f"  {'Mean':>5}"
    for name in table.method_names:
        m = table.mean_accuracy[name]
        s = table.std_accuracy[name]
        row += f"  {m:>.1%}±{s:>.1%}".rjust(15)
    print(row)
    print()

    # Friedman test
    if table.friedman:
        fr = table.friedman
        print(f"Friedman test: χ²={fr.statistic:.3f}, p={fr.p_value:.4f}")
        print(f"  Mean ranks: {fr.mean_ranks}")
        print(f"  Critical difference (Nemenyi): {fr.cd:.3f}")
        for pair, sig in fr.pairwise_significant.items():
            marker = "***" if sig else "n.s."
            print(f"  {pair}: {marker}")
        print()

    # Pairwise Wilcoxon
    if table.pairwise_wilcoxon:
        print("Pairwise Wilcoxon signed-rank tests:")
        for pair, (stat, p) in table.pairwise_wilcoxon.items():
            sig = "p<0.05 *" if p < 0.05 else "n.s."
            print(f"  {pair}: W={stat:.1f}, p={p:.4f} {sig}")
        print()

    # LaTeX
    if args.latex:
        print("LaTeX table:")
        print(table.to_latex())
        print()

    # Save JSON
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(table.to_dict(), fh, indent=2, ensure_ascii=False)
        print(f"Comparison saved to {out_path}")


if __name__ == "__main__":
    main()
