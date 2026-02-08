"""
分析 models/ 下所有已训练模型：流水线结构、CSP 与分类器参数、
在 replay 集上的表现，以及在多个数据集上的泛化能力（多数据集评估）。
支持 LDA / SVM / RF 多种流水线。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

from python_backend.utils import get_models_dir, get_project_root
from python_backend.training import load_model

# 分析结果图像输出目录（模型目录下 analysis/）
def _get_analysis_dir():
    d = get_models_dir() / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d

# 多数据集评估用的数据加载（与训练相同预处理）
try:
    from python_backend.train_model import get_epochs_for_eval
except Exception:
    get_epochs_for_eval = None

# 默认用于泛化评估的 (数据集, 受试者) 列表；通道数不一致时该列会标为 N/A
DEFAULT_EVAL_CONFIGS = [
    ("2a", 1), ("2a", 2), ("2a", 3),
    ("2b", 1),
    ("eegbci", 1),
]


def _list_model_files(models_dir: Path):
    """列出 models/ 下所有 .pkl 和 .joblib 文件（不含子目录）。"""
    files = []
    for ext in ("pkl", "joblib"):
        files.extend(models_dir.glob(f"*.{ext}"))
    return sorted(files, key=lambda p: (p.stem.lower(), p.suffix))


def _get_classifier_step(clf: Pipeline):
    """返回最后一个分类器步骤名与实例（LDA / SVM / RF）。"""
    for name in ("LDA", "SVM", "RF"):
        if name in clf.named_steps:
            return name, clf.named_steps[name]
    return None, None


def _model_n_channels(clf: Pipeline) -> int:
    """从 CSP 步骤读取模型期望的通道数；无 CSP 时返回 0。"""
    if "CSP" not in clf.named_steps:
        return 0
    return clf.named_steps["CSP"].filters_.shape[0]


def analyze_pipeline(clf: Pipeline, model_label: str = "") -> None:
    """打印单模型流水线结构、CSP 参数、分类器参数（按 LDA/SVM/RF 分支）。"""
    sep = "=" * 60
    print(sep)
    print(f"Pipeline 结构 {model_label}".strip())
    print(sep)
    for name, step in clf.steps:
        print(f"  {name}: {type(step).__name__}")
    print()

    if "CSP" not in clf.named_steps:
        print("  无 CSP 步骤，跳过 CSP 分析。")
        return

    csp = clf.named_steps["CSP"]
    print("CSP")
    print("-" * 40)
    print(f"  n_components: {csp.n_components}")
    print(f"  log: {getattr(csp, 'log', 'N/A')}")
    print(f"  filters_.shape: {csp.filters_.shape}  (n_channels, n_channels; 前 n_components 有效)")
    n_comp = min(csp.n_components, csp.filters_.shape[0])
    comp_filters = csp.filters_[:n_comp]
    norms = np.linalg.norm(comp_filters, axis=1)
    print(f"  前 n_components 滤波器 L2 范数: {norms.round(4).tolist()}")
    print()

    clf_name, clf_step = _get_classifier_step(clf)
    if clf_name == "LDA":
        print("LDA")
        print("-" * 40)
        print(f"  classes_: {clf_step.classes_.tolist()}")
        print(f"  coef_.shape: {clf_step.coef_.shape}")
        print(f"  intercept_: {clf_step.intercept_.tolist()}")
        print(f"  coef (flat): {np.squeeze(clf_step.coef_).round(4).tolist()}")
    elif clf_name == "SVM":
        print("SVM")
        print("-" * 40)
        print(f"  kernel: {clf_step.kernel}, C: {clf_step.C}")
        print(f"  classes_: {clf_step.classes_.tolist()}")
        print(f"  n_support_: {clf_step.n_support_.tolist()}")
    elif clf_name == "RF":
        print("RandomForest")
        print("-" * 40)
        print(f"  n_estimators: {clf_step.n_estimators}")
        print(f"  classes_: {clf_step.classes_.tolist()}")
        if hasattr(clf_step, "feature_importances_"):
            imp = clf_step.feature_importances_
            print(f"  feature_importances_ (CSP 维度): {imp.round(4).tolist()}")
    print()


def evaluate_on_replay(clf: Pipeline, X: np.ndarray, y: np.ndarray, model_label: str = "") -> None:
    """在 (X,y) 上评估并打印准确率、混淆矩阵、分类报告。"""
    print("=" * 60)
    print(f"Replay 集评估 {model_label}".strip())
    print("=" * 60)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    y_pred = clf.predict(X)
    acc = np.mean(y_pred == y)
    print(f"  Accuracy: {acc:.2%}")
    print()

    cm = confusion_matrix(y, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print()

    clf_name, clf_step = _get_classifier_step(clf)
    labels = clf_step.classes_.tolist() if clf_step is not None else None
    print("Classification report:")
    print(classification_report(y, y_pred, labels=labels, zero_division=0))
    print()


def _save_confusion_matrix_figure(cm: np.ndarray, class_labels: list, model_name: str, out_dir: Path) -> None:
    """将混淆矩阵保存为热力图图像。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels(class_labels or [str(i) for i in range(cm.shape[1])])
    ax.set_yticklabels(class_labels or [str(i) for i in range(cm.shape[0])])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, label="Count")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("/", "_")[:50]
    fig.savefig(out_dir / f"confusion_{safe_name}.png", dpi=150)
    plt.close(fig)
    print(f"  已保存图像: {out_dir / f'confusion_{safe_name}.png'}")


def _save_generalization_bar_figure(generalization_rows: list, col_headers: list, out_dir: Path) -> None:
    """将泛化能力表格保存为柱状图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    models = [r[0][:20] for r in generalization_rows]
    x = np.arange(len(col_headers))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(col_headers) * 1.5), 5))
    for i, (model_name, row) in enumerate(generalization_rows):
        vals = [row.get(h) if row.get(h) is not None else 0 for h in col_headers]
        off = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + off, vals, width, label=model_name[:15])
    ax.set_xticks(x)
    ax.set_xticklabels(col_headers, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Generalization — Accuracy per dataset")
    ax.legend(loc="lower right", fontsize=8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "generalization_bar.png", dpi=150)
    plt.close(fig)
    print(f"  已保存图像: {out_dir / 'generalization_bar.png'}")


def _save_replay_accuracy_bar_figure(model_names: list, accuracies: list, out_dir: Path) -> None:
    """Replay 集上各模型准确率柱状图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 0.8), 4))
    x = np.arange(len(model_names))
    ax.bar(x, accuracies, color="steelblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([n[:18] for n in model_names], rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Replay set — Accuracy per model")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "replay_accuracy_bar.png", dpi=150)
    plt.close(fig)
    print(f"  已保存图像: {out_dir / 'replay_accuracy_bar.png'}")


def run_generalization_eval(clf: Pipeline, model_name: str, eval_configs: list) -> dict:
    """
    在多个 (dataset, subject_id) 上评估模型，得到泛化准确率。
    若某配置数据不存在或通道数不匹配，该位置为 None（显示为 N/A）。
    返回 { "2a-1": 0.85, "2a-2": None, ... }。
    """
    result = {}
    n_ch = _model_n_channels(clf)
    if n_ch <= 0:
        return result

    for dataset, sub_id in eval_configs:
        key = f"{dataset}-{sub_id}"
        if get_epochs_for_eval is None:
            result[key] = None
            continue
        data = get_epochs_for_eval(dataset, sub_id)
        if data is None:
            result[key] = None
            continue
        X, y = data
        if X.shape[1] != n_ch:
            result[key] = None  # 通道数不一致，无法评估
            continue
        acc = np.mean(clf.predict(X) == y)
        result[key] = acc
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="分析已训练模型；支持多数据集泛化评估")
    parser.add_argument("--no-generalization", action="store_true",
                        help="不进行多数据集泛化评估（仅 replay + 结构分析）")
    parser.add_argument("--eval-configs", type=str, default=None,
                        help="泛化评估配置，逗号分隔，如 '2a-1,2a-2,2b-1,eegbci-1'（默认: 2a-1,2a-2,2a-3,2b-1,eegbci-1）")
    parser.add_argument("--no-figures", action="store_true",
                        help="不保存分析图像（仅文本与表格）")
    args = parser.parse_args()

    if args.eval_configs:
        eval_configs = []
        for s in args.eval_configs.split(","):
            s = s.strip()
            if "-" in s:
                ds, id_str = s.split("-", 1)
                try:
                    eval_configs.append((ds.strip(), int(id_str.strip())))
                except ValueError:
                    pass
        if not eval_configs:
            eval_configs = DEFAULT_EVAL_CONFIGS
    else:
        eval_configs = DEFAULT_EVAL_CONFIGS

    models_dir = get_models_dir()
    if not models_dir.exists():
        print(f"目录不存在: {models_dir}. 请先运行 train_model.py。")
        sys.exit(1)

    model_files = _list_model_files(models_dir)
    if not model_files:
        print(f"未在 {models_dir} 下找到 .pkl 或 .joblib 文件。请先运行 train_model.py。")
        sys.exit(1)

    replay_path = models_dir / "replay_data.npz"
    replay_data = None
    if replay_path.exists():
        replay_data = np.load(replay_path)
        X_replay, y_replay = replay_data["X"], replay_data["y"]
        print(f"已加载 replay 集: {replay_path} (X: {X_replay.shape}, y: {y_replay.shape})")
    else:
        print("未找到 replay_data.npz，跳过 replay 集评估。")
    print()

    analysis_dir = _get_analysis_dir() if not args.no_figures else None
    report_dir = _get_analysis_dir()
    report_path = report_dir / "report.txt"
    report_lines = []
    if analysis_dir is not None:
        print(f"分析图像将保存到: {analysis_dir}")
    print(f"文本报告将保存到: {report_path}")
    print()

    generalization_rows = []
    best_models = []
    replay_accuracies = []  # (model_name, acc) for bar chart
    for path in model_files:
        name = path.stem
        try:
            clf = joblib.load(path)
        except Exception as e:
            print(f"加载失败 {path.name}: {e}")
            continue
        if not isinstance(clf, Pipeline):
            print(f"跳过非 Pipeline: {path.name}")
            continue

        analyze_pipeline(clf, f"[{name}]")
        if replay_data is not None:
            n_ch = _model_n_channels(clf)
            if n_ch > 0 and X_replay.shape[1] != n_ch:
                print(f"跳过 Replay 评估 [{name}]: 模型通道数 {n_ch} 与 replay 通道数 {X_replay.shape[1]} 不一致")
            else:
                evaluate_on_replay(clf, X_replay, y_replay, f"[{name}]")
                y_pred = clf.predict(X_replay)
                acc = np.mean(y_pred == y_replay)
                replay_accuracies.append((name, acc))
                if analysis_dir is not None:
                    cm = confusion_matrix(y_replay, y_pred)
                    _, clf_step = _get_classifier_step(clf)
                    class_labels = (clf_step.classes_.tolist() if clf_step is not None else None) or list(map(str, np.unique(y_replay)))
                    _save_confusion_matrix_figure(cm, class_labels, name, analysis_dir)

        if not args.no_generalization and get_epochs_for_eval is not None and "CSP" in clf.named_steps:
            row = run_generalization_eval(clf, name, eval_configs)
            if row:
                generalization_rows.append((name, row))

        if name.startswith("best_model_sub"):
            try:
                rest = name.replace("best_model_sub", "").strip()
                sub_id = int(rest.split("_")[0]) if "_" in rest else int(rest)
                best_models.append((sub_id, path.name))
            except ValueError:
                pass

    if replay_accuracies and analysis_dir is not None:
        _save_replay_accuracy_bar_figure(
            [r[0] for r in replay_accuracies],
            [r[1] for r in replay_accuracies],
            analysis_dir,
        )

    # 泛化能力表格（多数据集评估）
    if generalization_rows:
        print("=" * 60)
        print("泛化能力（多数据集评估）")
        print("=" * 60)
        col_headers = [f"{ds}-{sid}" for ds, sid in eval_configs]
        header = "Model            | " + " | ".join(f"{c:>8}" for c in col_headers)
        print(header)
        print("-" * len(header))
        for model_name, row in generalization_rows:
            cells = []
            for k in col_headers:
                v = row.get(k)
                if v is None:
                    cells.append("   N/A  ")
                else:
                    cells.append(f" {v:.2%}  ")
            line = model_name[:16].ljust(17) + "| " + " | ".join(cells)
            print(line)
        print()
        print("说明: N/A 表示该数据集未找到或通道数与模型不匹配。")
        print()
        if analysis_dir is not None:
            _save_generalization_bar_figure(generalization_rows, col_headers, analysis_dir)

    if best_models:
        print("=" * 60)
        print("最佳模型汇总 (best_model_sub*.pkl，供 Unity 演示)")
        print("=" * 60)
        for sub_id, fname in sorted(best_models):
            print(f"  受试者 {sub_id}: {fname}")
        print()

    # Write text report to models/analysis/report.txt
    report_lines.append("Analysis Report")
    report_lines.append("=" * 60)
    report_lines.append(f"Models dir: {models_dir}")
    report_lines.append(f"Replay: {'loaded' if replay_data is not None else 'not found'}")
    if replay_data is not None:
        report_lines.append(f"  X: {X_replay.shape}, y: {y_replay.shape}")
    report_lines.append("")
    report_lines.append("Replay accuracy (models matching replay channel count)")
    report_lines.append("-" * 50)
    for name, acc in replay_accuracies:
        report_lines.append(f"  {name}: {acc:.2%}")
    report_lines.append("")
    if generalization_rows:
        report_lines.append("Generalization (dataset-subject)")
        report_lines.append("-" * 50)
        col_headers = [f"{ds}-{sid}" for ds, sid in eval_configs]
        col_headers_r = [f"{ds}-{sid}" for ds, sid in eval_configs]
        for model_name, row in generalization_rows:
            report_lines.append(f"  {model_name}:")
            for k in col_headers_r:
                v = row.get(k)
                report_lines.append(f"    {k}: {v:.2%}" if v is not None else f"    {k}: N/A")
        report_lines.append("")
    report_lines.append("Best models (best_model_sub*.pkl)")
    report_lines.append("-" * 50)
    for sub_id, fname in sorted(best_models):
        report_lines.append(f"  Subject {sub_id}: {fname}")
    report_lines.append("")
    report_lines.append("Figures (if generated)")
    report_lines.append("-" * 50)
    report_lines.append(f"  Dir: {report_dir}")
    if analysis_dir is not None:
        report_lines.append("  confusion_<model>.png, replay_accuracy_bar.png, generalization_bar.png")
    else:
        report_lines.append("  (run without --no-figures to generate)")
    try:
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"文本报告已保存: {report_path}")
    except Exception as e:
        print(f"保存文本报告失败: {e}")

    print("分析完成。")


if __name__ == "__main__":
    main()
