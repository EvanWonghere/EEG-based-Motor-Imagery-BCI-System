#!/usr/bin/env python
"""Generate thesis figures and analysis reports.

Usage:
    python scripts/analyze.py --results-dir results/ --output-dir results/figures/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.report import generate_report
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis figures")
    parser.add_argument(
        "--results-dir", type=str, default="results/",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/figures/",
        help="Output directory for generated figures",
    )
    args = parser.parse_args()

    setup_logging()

    generated = generate_report(args.results_dir, args.output_dir)

    if generated:
        print(f"\nGenerated {len(generated)} file(s):")
        for p in generated:
            print(f"  {p}")
    else:
        print("No figures generated. Run training first to produce results.")


if __name__ == "__main__":
    main()
