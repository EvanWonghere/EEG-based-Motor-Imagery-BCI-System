#!/usr/bin/env python
"""Regenerate results.json from per-subject cv_sub*.json files.

Fixes the case where results.json was written during a single-subject run
but per-subject CV JSONs exist for all subjects.

Usage:
    python scripts/regenerate_results.py results/fbcsp_svm_2a
    python scripts/regenerate_results.py results/  # all experiments
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def regenerate_one(exp_dir: Path) -> bool:
    """Rebuild results.json for a single experiment directory."""
    cv_files = sorted(exp_dir.glob("cv_sub*.json"))
    if not cv_files:
        return False

    # Read existing results.json for experiment metadata
    results_path = exp_dir / "results.json"
    meta = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as fh:
            meta = json.load(fh)

    exp_name = meta.get("experiment", exp_dir.name)
    model_type = meta.get("model", "unknown")
    features = meta.get("features", "unknown")

    subjects = []
    for cv_path in cv_files:
        with open(cv_path, encoding="utf-8") as fh:
            cv_data = json.load(fh)

        # Extract subject ID from filename: cv_sub3.json -> 3
        sub_id = int(cv_path.stem.replace("cv_sub", ""))
        mean_metrics = cv_data.get("mean_metrics", {})
        fold_accs = [
            f.get("metrics", {}).get("accuracy", 0)
            for f in cv_data.get("folds", [])
        ]

        subjects.append({
            "subject": sub_id,
            "skipped": False,
            "mean_accuracy": mean_metrics.get("accuracy", 0),
            "std_accuracy": cv_data.get("std_metrics", {}).get("accuracy", 0),
            "mean_kappa": mean_metrics.get("kappa", 0),
            "scores": fold_accs,
            "model_path": str(exp_dir / "models" / f"{exp_name}_sub{sub_id}.pkl"),
        })

    valid = [s for s in subjects if not s.get("skipped")]
    import numpy as np
    mean_acc = float(np.mean([s["mean_accuracy"] for s in valid])) if valid else None
    mean_kappa = float(np.mean([s["mean_kappa"] for s in valid])) if valid else None

    agg = {
        "experiment": exp_name,
        "model": model_type,
        "features": features,
        "subjects": subjects,
        "summary": {
            "mean_accuracy": mean_acc,
            "mean_kappa": mean_kappa,
            "n_subjects": len(valid),
        },
    }

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2, ensure_ascii=False)

    print(f"  {exp_name}: {len(valid)} subjects, mean acc={mean_acc:.1%}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate results.json from cv_sub*.json")
    parser.add_argument("path", type=str, help="Experiment dir or parent results/ dir")
    args = parser.parse_args()

    target = Path(args.path)

    if (target / "cv_sub1.json").exists():
        # Single experiment directory
        dirs = [target]
    else:
        # Parent directory containing multiple experiments
        dirs = sorted(d for d in target.iterdir() if d.is_dir() and list(d.glob("cv_sub*.json")))

    if not dirs:
        print(f"No cv_sub*.json files found in {target}")
        sys.exit(1)

    print(f"Regenerating results.json for {len(dirs)} experiment(s):")
    for d in dirs:
        regenerate_one(d)

    print("Done.")


if __name__ == "__main__":
    main()
