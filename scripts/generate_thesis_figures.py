#!/usr/bin/env python
"""Generate all thesis-specific figures that require raw data access.

This script produces figures that cannot be generated from results JSON alone:
  - CSP spatial pattern topomaps (requires fitting CSP on data)
  - ERD/ERS time-frequency maps (requires epoched EEG data)
  - FBCSP selected frequency band distribution (requires fitting FBCSP)
  - Statistical test summary table (from results)

Usage:
    python scripts/generate_thesis_figures.py
    python scripts/generate_thesis_figures.py --subjects 1 3 8
    python scripts/generate_thesis_figures.py --output figures/thesis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import create_dataset
from src.features.csp import CSPExtractor
from src.features.fbcsp import FBCSPExtractor
from src.preprocessing import PreprocessingPipeline
from src.preprocessing.epochs import epochs_to_arrays
from src.utils.config import load_config
from src.utils.logging import setup_logging, get_logger
from src.visualization.topo import plot_csp_patterns

logger = get_logger(__name__)


def load_subject_data(subject_id: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess one subject's data, return (X, y)."""
    ds = create_dataset(cfg["dataset"]["name"], data_dir=str(ROOT / "data"), subjects=[subject_id])
    sessions = cfg["dataset"].get("train_sessions") or ["session_T"]
    session_letters = [str(s)[-1].upper() for s in sessions]
    ep_cfg = cfg["preprocessing"].get("epoch", {})

    all_X, all_y = [], []
    for session in session_letters:
        raw, events, event_id = ds.load_data(subject_id, session=session)
        pipe = PreprocessingPipeline(cfg["preprocessing"])
        epochs = pipe.run(raw, events, event_id)
        X_s, y_s = epochs_to_arrays(epochs, tmin=ep_cfg.get("train_tmin"), tmax=ep_cfg.get("train_tmax"))
        all_X.append(X_s)
        all_y.append(y_s)

    return np.concatenate(all_X), np.concatenate(all_y)


def load_subject_epochs(subject_id: int, cfg: dict):
    """Load and preprocess one subject, return MNE Epochs (for ERD/ERS)."""
    import mne
    ds = create_dataset(cfg["dataset"]["name"], data_dir=str(ROOT / "data"), subjects=[subject_id])
    sessions = cfg["dataset"].get("train_sessions") or ["session_T"]
    session_letters = [str(s)[-1].upper() for s in sessions]

    all_epochs = []
    for session in session_letters:
        raw, events, event_id = ds.load_data(subject_id, session=session)
        pipe = PreprocessingPipeline(cfg["preprocessing"])
        epochs = pipe.run(raw, events, event_id)
        all_epochs.append(epochs)

    if len(all_epochs) == 1:
        return all_epochs[0]
    return mne.concatenate_epochs(all_epochs)


# ---- Figure generators ----

def generate_csp_topomaps(subjects: list[int], output_dir: Path) -> list[Path]:
    """Generate CSP spatial pattern topomaps for selected subjects."""
    import mne
    cfg = load_config(str(ROOT / "configs/experiments/csp_lda_baseline.yaml"))
    generated = []

    # Channel locations from BCI IV 2a standard montage
    ch_names = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
        "P1", "Pz", "P2", "POz",
    ]
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)

    for sid in subjects:
        logger.info("Generating CSP topomaps for subject %d...", sid)
        X, y = load_subject_data(sid, cfg)

        csp = CSPExtractor(n_components=4)
        csp.fit(X, y)
        patterns = csp.get_patterns()

        path = output_dir / f"csp_patterns_sub{sid}.png"
        plot_csp_patterns(patterns, info, title=f"CSP Spatial Patterns — Subject {sid}", save_path=path)
        plt.close()
        generated.append(path)
        logger.info("  Saved %s", path)

    return generated


def generate_erds_maps(subjects: list[int], output_dir: Path) -> list[Path]:
    """Generate ERD/ERS time-frequency maps for selected subjects."""
    from src.visualization.erds import plot_erds_maps
    cfg = load_config(str(ROOT / "configs/experiments/csp_lda_baseline.yaml"))
    generated = []

    for sid in subjects:
        logger.info("Generating ERD/ERS maps for subject %d...", sid)
        epochs = load_subject_epochs(sid, cfg)

        path = output_dir / f"erds_map_sub{sid}.png"
        plot_erds_maps(
            epochs,
            picks=["C3", "Cz", "C4"],
            title=f"ERD/ERS Time-Frequency Map — Subject {sid}",
            save_path=path,
        )
        plt.close()
        generated.append(path)
        logger.info("  Saved %s", path)

    return generated


def generate_fbcsp_band_distribution(subjects: list[int], output_dir: Path) -> list[Path]:
    """Generate FBCSP selected feature frequency band distribution heatmap."""
    cfg = load_config(str(ROOT / "configs/experiments/fbcsp_svm.yaml"))
    generated = []

    band_labels = ["4-8", "8-12", "12-16", "16-20", "20-24", "24-28", "28-32"]
    n_bands = len(band_labels)

    # Collect band selection counts per subject
    all_band_counts = {}
    all_band_info = {}

    for sid in subjects:
        logger.info("Fitting FBCSP for subject %d to analyze band selection...", sid)
        X, y = load_subject_data(sid, cfg)

        fbcsp = FBCSPExtractor(
            filter_bands=[(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)],
            n_components=4,
            n_select=8,
            sfreq=250.0,
        )
        fbcsp.fit(X, y)
        band_info = fbcsp.get_selected_band_info()
        all_band_info[sid] = band_info

        counts = np.zeros(n_bands, dtype=int)
        for feat in band_info:
            counts[feat["band_index"]] += 1
        all_band_counts[sid] = counts

    # --- Figure 1: Heatmap of band selection across subjects ---
    sub_ids = sorted(all_band_counts.keys())
    matrix = np.array([all_band_counts[s] for s in sub_ids])  # (n_subjects, n_bands)

    fig, ax = plt.subplots(figsize=(10, max(4, len(sub_ids) * 0.6)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=4)
    ax.set_xticks(range(n_bands))
    ax.set_xticklabels([f"{bl} Hz" for bl in band_labels])
    ax.set_yticks(range(len(sub_ids)))
    ax.set_yticklabels([f"Sub {s}" for s in sub_ids])
    ax.set_xlabel("Frequency Sub-band")
    ax.set_ylabel("Subject")
    ax.set_title("FBCSP Selected Features: Sub-band Distribution per Subject")

    # Annotate cells with counts
    for i in range(len(sub_ids)):
        for j in range(n_bands):
            val = matrix[i, j]
            color = "white" if val >= 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="# selected features")
    fig.tight_layout()

    path = output_dir / "fbcsp_band_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    generated.append(path)
    logger.info("Saved %s", path)

    # --- Figure 2: Aggregate bar chart ---
    total_counts = matrix.sum(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(n_bands), total_counts, color="#4a90d9", edgecolor="white")
    ax.set_xticks(range(n_bands))
    ax.set_xticklabels([f"{bl} Hz" for bl in band_labels])
    ax.set_xlabel("Frequency Sub-band")
    ax.set_ylabel("Total Selected Features (across subjects)")
    ax.set_title("FBCSP Feature Selection: Aggregate Sub-band Distribution")

    for bar, val in zip(bars, total_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()

    path = output_dir / "fbcsp_band_barchart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    generated.append(path)
    logger.info("Saved %s", path)

    return generated


def generate_statistical_summary(output_dir: Path) -> list[Path]:
    """Generate statistical test summary from results."""
    from src.evaluation.comparison import compare_methods

    results_dir = ROOT / "results"
    generated = []

    try:
        table = compare_methods(results_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.warning("Cannot generate statistical summary: %s", e)
        return generated

    # Save full analysis as JSON
    analysis = table.to_dict()
    path = output_dir / "statistical_analysis.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    generated.append(path)
    logger.info("Saved %s", path)

    # Save LaTeX table with kappa
    if table.kappa_matrix is not None:
        methods = table.method_names
        lines = [
            r"\begin{tabular}{c" + "cc" * len(methods) + "}",
            r"\toprule",
            r"Subject & " + " & ".join(
                rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in methods
            ) + r" \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            " & " + " & ".join(["Acc", r"$\kappa$"] * len(methods)) + r" \\",
            r"\midrule",
        ]
        for i, sid in enumerate(table.subject_ids):
            row_acc = table.accuracy_matrix[i]
            row_kappa = table.kappa_matrix[i]
            best_j = int(np.argmax(row_acc))
            cells = [str(sid)]
            for j in range(len(methods)):
                acc_s = f"{row_acc[j]:.1%}"
                kap_s = f"{row_kappa[j]:.3f}"
                if j == best_j:
                    acc_s = r"\textbf{" + acc_s + "}"
                cells.extend([acc_s, kap_s])
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\midrule")
        cells = ["Mean"]
        for name in methods:
            m_acc = table.mean_accuracy[name]
            s_acc = table.std_accuracy[name]
            j = methods.index(name)
            m_kap = float(np.mean(table.kappa_matrix[:, j]))
            cells.extend([f"{m_acc:.1%} ± {s_acc:.1%}", f"{m_kap:.3f}"])
        lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        path = output_dir / "comparison_table_full.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        generated.append(path)
        logger.info("Saved %s", path)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-specific figures")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 3, 8],
                        help="Subjects for CSP/ERD/FBCSP figures (default: 1 3 8 = high/mid/low performers)")
    parser.add_argument("--all-subjects", action="store_true",
                        help="Use all 9 subjects for FBCSP band analysis")
    parser.add_argument("--output", type=str, default="results/figures/thesis",
                        help="Output directory for figures")
    parser.add_argument("--skip-erds", action="store_true",
                        help="Skip ERD/ERS maps (slow to compute)")
    args = parser.parse_args()

    setup_logging()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    fbcsp_subjects = list(range(1, 10)) if args.all_subjects else args.subjects
    generated = []

    # 1. CSP topomaps
    logger.info("=== Generating CSP spatial pattern topomaps ===")
    generated.extend(generate_csp_topomaps(args.subjects, output_dir))

    # 2. ERD/ERS maps
    if not args.skip_erds:
        logger.info("=== Generating ERD/ERS time-frequency maps ===")
        generated.extend(generate_erds_maps(args.subjects, output_dir))
    else:
        logger.info("=== Skipping ERD/ERS maps ===")

    # 3. FBCSP band distribution
    logger.info("=== Generating FBCSP band distribution analysis ===")
    generated.extend(generate_fbcsp_band_distribution(fbcsp_subjects, output_dir))

    # 4. Statistical summary
    logger.info("=== Generating statistical analysis summary ===")
    generated.extend(generate_statistical_summary(output_dir))

    print(f"\n{'=' * 60}")
    print(f"  Generated {len(generated)} file(s):")
    print(f"{'=' * 60}")
    for p in generated:
        print(f"  {p}")
    print()


if __name__ == "__main__":
    main()
