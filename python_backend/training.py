"""
多模型 CSP 流水线：CSP 特征 + 多种分类器（LDA / SVM / RandomForest）。
提供工厂函数 create_pipeline(model_type)，训练与交叉验证，按名称保存/加载。
用于 replay_stream 与 analyze_model。
"""
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from pathlib import Path
import joblib
from typing import Tuple, List, Optional

from .utils import get_models_dir


# 支持的模型类型（用于 create_pipeline 与保存命名）
MODEL_TYPES = ("lda", "svm", "rf")


def create_pipeline(model_type: str = "lda", n_components: int = 4) -> Pipeline:
    """
    根据名称返回 CSP + 分类器流水线。
    - lda: CSP -> LinearDiscriminantAnalysis（基准）
    - svm: CSP -> StandardScaler -> SVC(kernel='rbf', C=1.0)
    - rf: CSP -> RandomForestClassifier(n_estimators=100)
    CSP 固定 n_components=4, log=True。
    """
    model_type = model_type.lower().strip()
    csp = CSP(
        n_components=n_components,
        reg=None,
        log=True,
        norm_trace=False,
    )
    if model_type == "lda":
        return Pipeline([("CSP", csp), ("LDA", LinearDiscriminantAnalysis())])
    if model_type == "svm":
        return Pipeline([
            ("CSP", csp),
            ("Scaler", StandardScaler()),
            ("SVM", SVC(kernel="rbf", C=1.0)),
        ])
    if model_type == "rf":
        return Pipeline([
            ("CSP", csp),
            ("RF", RandomForestClassifier(n_estimators=100)),
        ])
    raise ValueError(f"Unknown model_type='{model_type}'. Use one of {MODEL_TYPES}")


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "lda",
    n_splits: int = 10,
    random_state: int = 42,
) -> Tuple[Pipeline, np.ndarray]:
    """
    构建指定类型流水线，做分层 K 折交叉验证并拟合全量数据。
    返回 (拟合后的 pipeline, 各折准确率数组)。
    """
    clf = create_pipeline(model_type=model_type)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
    clf.fit(X, y)
    return clf, scores


def save_model(clf: Pipeline, name: str) -> Path:
    """将拟合后的流水线保存到 models/，扩展名 .pkl（joblib 序列化）。"""
    out_dir = get_models_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pkl"
    joblib.dump(clf, path)
    return path


def load_model(name: str, use_joblib_fallback: bool = True) -> Pipeline:
    """
    从 models/ 按名称加载 pipeline。
    先尝试 {name}.pkl，若不存在且 use_joblib_fallback 则尝试 {name}.joblib（兼容旧版）。
    """
    out_dir = get_models_dir()
    for ext in ("pkl", "joblib"):
        path = out_dir / f"{name}.{ext}"
        if path.exists():
            return joblib.load(path)
    if use_joblib_fallback:
        raise FileNotFoundError(f"Model not found: {out_dir / name}.pkl or .joblib. Run train_model.py first.")
    raise FileNotFoundError(f"Model not found: {out_dir / name}.pkl. Run train_model.py first.")


def run_training(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "lda",
    model_name: str = None,
) -> Tuple[Pipeline, np.ndarray]:
    """
    对给定 (X, y) 训练指定类型模型，做 10 折交叉验证，保存到 model_name。
    model_name 若为 None 则使用 csp_{model_type}。
    返回 (拟合后的 pipeline, 各折准确率)。
    """
    if model_name is None:
        model_name = f"csp_{model_type}"
    clf, scores = train_and_evaluate(X, y, model_type=model_type)
    mean_acc = np.mean(scores)
    print(f"  [{model_type.upper()}] CV accuracy: {mean_acc:.2%}, per-fold: {scores.round(3).tolist()}")
    save_model(clf, model_name)
    return clf, scores


def get_best_model_name_for_subject(subject_id: int, dataset_suffix: Optional[str] = None) -> str:
    """每个受试者最佳模型保存为 best_model_sub{id}.pkl；若提供 dataset_suffix 则加 _2a/_2b/_eegbci。"""
    base = f"best_model_sub{subject_id}"
    return f"{base}_{dataset_suffix}" if dataset_suffix else base
