#!/usr/bin/env python
"""Generate all thesis figures for multi-dataset comparison.

Produces:
  1. Cross-dataset accuracy grouped bar chart (3 datasets × 3 methods)
  2. Cross-dataset subject boxplot (faceted by dataset)
  3. Per-dataset confusion matrices (best method, representative subject)
  4. Per-dataset ROC curves (all methods, representative subject)
  5. Cross-dataset statistical tests (Friedman + Wilcoxon)
  6. Updated comparison LaTeX tables (per-dataset + cross-dataset summary)
  7. Channel count vs accuracy scatter plot

Usage:
    python scripts/generate_multi_dataset_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "LaTeX-Bachelor" / "data" / "img"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Result loading ───────────────────────────────────────────────

EXPERIMENTS = {
    "BCI IV 2a": {
        "CSP+LDA":   "results/csp_lda_baseline/results.json",
        "FBCSP+SVM": "results/fbcsp_svm_2a/results.json",
        "EEGNet":    "results/eegnet_2a/results.json",
    },
    "BCI IV 2b": {
        "CSP+LDA":   "results/csp_lda_2b/results.json",
        "FBCSP+SVM": "results/fbcsp_svm_2b/results.json",
        "EEGNet":    "results/eegnet_2b/results.json",
    },
    "PhysioNet": {
        "CSP+LDA":   "results/csp_lda_eegbci/results.json",
        "FBCSP+SVM": "results/fbcsp_svm_eegbci/results.json",
        "EEGNet":    "results/eegnet_eegbci/results.json",
    },
}

METHODS = ["CSP+LDA", "FBCSP+SVM", "EEGNet"]
DATASETS = ["BCI IV 2a", "BCI IV 2b", "PhysioNet"]
METHOD_COLORS = {"CSP+LDA": "#4C72B0", "FBCSP+SVM": "#DD8452", "EEGNet": "#55A868"}


def load_all():
    """Return nested dict: data[dataset][method] = results_json."""
    data = {}
    for ds in DATASETS:
        data[ds] = {}
        for method in METHODS:
            path = ROOT / EXPERIMENTS[ds][method]
            with open(path) as f:
                data[ds][method] = json.load(f)
    return data


def get_subject_accs(result_json) -> list[float]:
    return [s["mean_accuracy"] for s in result_json["subjects"]]


def get_subject_kappas(result_json) -> list[float]:
    return [s.get("mean_kappa", 0) for s in result_json["subjects"]]


# ── Figure 1: Cross-dataset grouped bar chart ────────────────────

def fig_cross_dataset_bars(data):
    """3 groups (datasets) × 3 bars (methods), with error bars."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n_ds = len(DATASETS)
    n_m = len(METHODS)
    width = 0.25
    x = np.arange(n_ds)

    for i, method in enumerate(METHODS):
        means, stds = [], []
        for ds in DATASETS:
            accs = get_subject_accs(data[ds][method])
            means.append(np.mean(accs))
            stds.append(np.std(accs))
        bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
                      label=method, color=METHOD_COLORS[method], edgecolor="white")
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{m:.1%}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(DATASETS, fontsize=11)
    ax.set_ylabel("Mean Accuracy", fontsize=11)
    ax.set_title("Classification Accuracy Across Datasets", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance (50%)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = OUTPUT / "cross_dataset_accuracy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [1] {path}")
    return path


# ── Figure 2: Cross-dataset boxplot (faceted) ────────────────────

def fig_cross_dataset_boxplot(data):
    """Faceted boxplot: one panel per dataset, 3 methods each."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, ds in zip(axes, DATASETS):
        box_data = [get_subject_accs(data[ds][m]) for m in METHODS]
        bp = ax.boxplot(box_data, labels=METHODS, patch_artist=True,
                        showmeans=True, meanprops=dict(marker="D", markerfacecolor="red", markersize=5))
        for patch, method in zip(bp["boxes"], METHODS):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.7)
        ax.set_title(ds, fontsize=12, fontweight="bold")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(0.2, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    fig.suptitle("Per-Subject Accuracy Distribution by Dataset", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = OUTPUT / "cross_dataset_boxplot.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [2] {path}")
    return path


# ── Figure 3: Confusion matrices (best method per dataset) ───────

def fig_confusion_matrices(data):
    """3-panel confusion matrix: best method, median-accuracy subject per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = ["Left Hand", "Right Hand"]

    best_methods = {
        "BCI IV 2a": "FBCSP+SVM",
        "BCI IV 2b": "EEGNet",
        "PhysioNet":  "CSP+LDA",
    }

    for ax, ds in zip(axes, DATASETS):
        method = best_methods[ds]
        result = data[ds][method]
        # Pick median-accuracy subject
        subjects = result["subjects"]
        accs = [s["mean_accuracy"] for s in subjects]
        median_idx = np.argsort(accs)[len(accs) // 2]
        sub = subjects[median_idx]
        sid = sub["subject"]

        # Load CV data to build confusion matrix
        exp_name = result["experiment"]
        cv_path = ROOT / f"results/{exp_name}/cv_sub{sid}.json"
        if cv_path.exists():
            with open(cv_path) as f:
                cv = json.load(f)
            y_true_all, y_pred_all = [], []
            for fold in cv["folds"]:
                y_true_all.extend(fold.get("y_true", []))
                y_pred_all.extend(fold.get("y_pred", []))
            y_true = np.array(y_true_all)
            y_pred = np.array(y_pred_all)

            # Build 2×2 CM
            classes = sorted(np.unique(y_true))
            cm = np.zeros((2, 2), dtype=int)
            for i, c_true in enumerate(classes):
                for j, c_pred in enumerate(classes):
                    cm[i, j] = np.sum((y_true == c_true) & (y_pred == c_pred))
        else:
            cm = np.array([[0, 0], [0, 0]])

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["L", "R"])
        ax.set_yticklabels(["L", "R"])
        ax.set_xlabel("Predicted")
        if ax == axes[0]:
            ax.set_ylabel("True")

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=14, fontweight="bold")

        acc = sub["mean_accuracy"]
        ax.set_title(f"{ds}\n{method} — Sub {sid} ({acc:.1%})", fontsize=10)

    fig.suptitle("Confusion Matrices (Median-Accuracy Subject)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = OUTPUT / "cross_dataset_confusion.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [3] {path}")
    return path


# ── Figure 4: ROC curves (per dataset, all methods) ──────────────

def fig_roc_curves(data):
    """3-panel ROC curves: all 3 methods per dataset, best subject."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, ds in zip(axes, DATASETS):
        # Pick a good subject (highest mean across methods)
        all_sids = set()
        method_accs = {}
        for method in METHODS:
            result = data[ds][method]
            method_accs[method] = {s["subject"]: s["mean_accuracy"] for s in result["subjects"]}
            all_sids.update(method_accs[method].keys())
        common_sids = sorted(all_sids)
        mean_per_sub = {s: np.mean([method_accs[m].get(s, 0.5) for m in METHODS]) for s in common_sids}
        # Pick the subject closest to the 75th percentile
        sorted_sids = sorted(mean_per_sub.keys(), key=lambda s: mean_per_sub[s])
        target_idx = int(len(sorted_sids) * 0.75)
        sid = sorted_sids[min(target_idx, len(sorted_sids) - 1)]

        for method in METHODS:
            result = data[ds][method]
            exp_name = result["experiment"]
            cv_path = ROOT / f"results/{exp_name}/cv_sub{sid}.json"
            if not cv_path.exists():
                continue
            with open(cv_path) as f:
                cv = json.load(f)

            y_true_all, y_proba_all = [], []
            for fold in cv["folds"]:
                y_true_all.extend(fold.get("y_true", []))
                proba = fold.get("y_proba")
                if proba is not None:
                    y_proba_all.extend(proba)

            if not y_proba_all:
                continue

            y_true = np.array(y_true_all)
            y_proba = np.array(y_proba_all)
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]

            classes = sorted(np.unique(y_true))
            y_binary = (y_true == classes[1]).astype(int) if len(classes) == 2 else y_true

            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_binary, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=METHOD_COLORS[method], lw=2,
                    label=f"{method} (AUC={roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_title(f"{ds} — Subject {sid}", fontsize=11, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        if ax == axes[0]:
            ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("ROC Curves by Dataset (Representative Subject)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = OUTPUT / "cross_dataset_roc.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [4] {path}")
    return path


# ── Figure 5: Channel count vs accuracy ──────────────────────────

def fig_channel_vs_accuracy(data):
    """Scatter plot: n_channels vs mean accuracy per method."""
    ch_counts = {"BCI IV 2a": 22, "BCI IV 2b": 3, "PhysioNet": 64}

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in METHODS:
        xs, ys, errs = [], [], []
        for ds in DATASETS:
            accs = get_subject_accs(data[ds][method])
            xs.append(ch_counts[ds])
            ys.append(np.mean(accs))
            errs.append(np.std(accs))
        ax.errorbar(xs, ys, yerr=errs, fmt="o-", color=METHOD_COLORS[method],
                    label=method, capsize=5, markersize=8, linewidth=2)

    ax.set_xlabel("Number of EEG Channels", fontsize=11)
    ax.set_ylabel("Mean Accuracy", fontsize=11)
    ax.set_title("Effect of Channel Count on Classification Accuracy", fontsize=12, fontweight="bold")
    ax.set_xticks([3, 22, 64])
    ax.set_xticklabels(["3\n(2b)", "22\n(2a)", "64\n(PhysioNet)"])
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0.35, 1.0)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = OUTPUT / "channel_vs_accuracy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [5] {path}")
    return path


# ── Figure 6: Statistical tests ──────────────────────────────────

def fig_statistical_tests(data):
    """Run Friedman + pairwise Wilcoxon per dataset, save JSON summary."""
    stats = {}

    for ds in DATASETS:
        accs = {}
        for m in METHODS:
            accs[m] = get_subject_accs(data[ds][m])

        # Align subject lists (use minimum length)
        min_len = min(len(v) for v in accs.values())
        aligned = {m: accs[m][:min_len] for m in METHODS}

        ds_stats = {"n_subjects": min_len, "methods": {}}
        for m in METHODS:
            ds_stats["methods"][m] = {
                "mean_accuracy": float(np.mean(aligned[m])),
                "std_accuracy": float(np.std(aligned[m])),
                "mean_kappa": float(np.mean(get_subject_kappas(data[ds][m])[:min_len])),
            }

        # Friedman test
        try:
            stat, p = friedmanchisquare(*[aligned[m] for m in METHODS])
            ds_stats["friedman"] = {"statistic": float(stat), "p_value": float(p)}
        except Exception as e:
            ds_stats["friedman"] = {"error": str(e)}

        # Pairwise Wilcoxon
        pairs = [("CSP+LDA", "FBCSP+SVM"), ("CSP+LDA", "EEGNet"), ("FBCSP+SVM", "EEGNet")]
        ds_stats["wilcoxon_pairwise"] = {}
        for m1, m2 in pairs:
            try:
                stat, p = wilcoxon(aligned[m1], aligned[m2])
                ds_stats["wilcoxon_pairwise"][f"{m1}_vs_{m2}"] = {
                    "statistic": float(stat), "p_value": float(p),
                    "significant_005": bool(p < 0.05),
                }
            except Exception as e:
                ds_stats["wilcoxon_pairwise"][f"{m1}_vs_{m2}"] = {"error": str(e)}

        stats[ds] = ds_stats

    path = OUTPUT.parent / "resource" / "statistical_analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  [6] {path}")

    # Also generate a visual summary
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    col_labels = ["Dataset", "Friedman p", "CSP vs FBCSP p", "CSP vs EEGNet p", "FBCSP vs EEGNet p"]
    table_data = []
    for ds in DATASETS:
        row = [ds]
        fr = stats[ds].get("friedman", {})
        row.append(f"{fr.get('p_value', 'N/A'):.4f}" if "p_value" in fr else "N/A")
        for pair in ["CSP+LDA_vs_FBCSP+SVM", "CSP+LDA_vs_EEGNet", "FBCSP+SVM_vs_EEGNet"]:
            w = stats[ds]["wilcoxon_pairwise"].get(pair, {})
            if "p_value" in w:
                p = w["p_value"]
                sig = "*" if p < 0.05 else ""
                row.append(f"{p:.4f}{sig}")
            else:
                row.append("N/A")
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center",
                     cellLoc="center", colColours=["#f0f0f0"] * 5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Highlight significant cells
    for i, row in enumerate(table_data):
        for j in range(2, 5):
            if "*" in row[j]:
                table[i + 1, j].set_facecolor("#ffe0e0")

    fig.suptitle("Statistical Tests Summary (* = p < 0.05)", fontsize=12, fontweight="bold")
    fig.tight_layout()

    path2 = OUTPUT / "statistical_tests_table.png"
    fig.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [6b] {path2}")

    return stats


# ── Figure 7: Per-dataset comparison LaTeX tables ─────────────────

def generate_latex_tables(data):
    """Generate LaTeX comparison tables for each dataset + cross-dataset summary."""
    tex_dir = OUTPUT.parent / "resource"

    # Per-dataset tables
    for ds in DATASETS:
        ds_short = ds.replace(" ", "_").replace("IV_", "").lower()
        lines = []
        lines.append(r"\begin{tabular}{c" + "cc" * len(METHODS) + "}")
        lines.append(r"\toprule")
        lines.append(r"Subject & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in METHODS) + r" \\")
        cmr = " ".join(rf"\cmidrule(lr){{{2+i*2}-{3+i*2}}}" for i in range(len(METHODS)))
        lines.append(cmr)
        lines.append(" & " + " & ".join(["Acc", r"$\kappa$"] * len(METHODS)) + r" \\")
        lines.append(r"\midrule")

        # Get common subjects
        all_sub_data = {}
        for m in METHODS:
            for s in data[ds][m]["subjects"]:
                sid = s["subject"]
                if sid not in all_sub_data:
                    all_sub_data[sid] = {}
                all_sub_data[sid][m] = s

        for sid in sorted(all_sub_data.keys()):
            accs = [all_sub_data[sid].get(m, {}).get("mean_accuracy", 0) for m in METHODS]
            best_j = int(np.argmax(accs))
            cells = [str(sid)]
            for j, m in enumerate(METHODS):
                s = all_sub_data[sid].get(m, {})
                acc = s.get("mean_accuracy", 0)
                kap = s.get("mean_kappa", 0)
                acc_s = f"{acc:.1%}"
                if j == best_j:
                    acc_s = r"\textbf{" + acc_s + "}"
                cells.extend([acc_s, f"{kap:.3f}"])
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\midrule")
        cells = ["Mean"]
        for m in METHODS:
            sm = data[ds][m]["summary"]
            accs = get_subject_accs(data[ds][m])
            kappas = get_subject_kappas(data[ds][m])
            cells.extend([
                f"{sm['mean_accuracy']:.1%} $\\pm$ {np.std(accs):.1%}",
                f"{np.mean(kappas):.3f}",
            ])
        lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        path = tex_dir / f"comparison_table_{ds_short}.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [7] {path}")

    # Cross-dataset summary table
    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & BCI IV 2a & BCI IV 2b & PhysioNet EEGBCI \\")
    lines.append(r"\midrule")
    for m in METHODS:
        cells = [m]
        for ds in DATASETS:
            sm = data[ds][m]["summary"]
            accs = get_subject_accs(data[ds][m])
            cells.append(f"{sm['mean_accuracy']:.1%} $\\pm$ {np.std(accs):.1%}")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    path = tex_dir / "comparison_table_cross_dataset.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [7] {path}")


# ── Figure 8: Trial count impact (before/after PhysioNet) ────────

def fig_trial_count_impact():
    """Bar chart showing PhysioNet accuracy before (45 trials) and after (90 trials)."""
    before = {"CSP+LDA": 59.63, "FBCSP+SVM": 56.30, "EEGNet": 50.22}
    after = {"CSP+LDA": 61.56, "FBCSP+SVM": 60.17, "EEGNet": 47.89}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(METHODS))
    width = 0.35

    bars1 = ax.bar(x - width/2, [before[m] for m in METHODS], width,
                   label="45 trials (MI only)", color="#aec7e8", edgecolor="white")
    bars2 = ax.bar(x + width/2, [after[m] for m in METHODS], width,
                   label="90 trials (MI + ME)", color="#1f77b4", edgecolor="white")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=11)
    ax.set_title("Effect of Trial Count on PhysioNet EEGBCI Performance", fontsize=12, fontweight="bold")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="Chance (50%)")
    ax.set_ylim(40, 70)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = OUTPUT / "trial_count_impact.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [8] {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("Loading all experiment results...")
    data = load_all()

    print(f"\nGenerating figures to {OUTPUT}/\n")

    fig_cross_dataset_bars(data)
    fig_cross_dataset_boxplot(data)
    fig_confusion_matrices(data)
    fig_roc_curves(data)
    fig_channel_vs_accuracy(data)
    stats = fig_statistical_tests(data)
    generate_latex_tables(data)
    fig_trial_count_impact()

    print(f"\nDone! All figures saved to {OUTPUT}/")


if __name__ == "__main__":
    main()
