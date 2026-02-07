"""
CSP feature extraction and LDA classifier training.
Saves pipeline (CSP + LDA) to models/ for use by replay_stream.
"""
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, ShuffleSplit
from pathlib import Path
import joblib
from typing import Tuple, Optional

from .utils import get_models_dir


def build_pipeline(n_components: int = 4) -> Pipeline:
    """Build CSP + LDA pipeline."""
    csp = CSP(
        n_components=n_components,
        reg=None,
        log=True,
        norm_trace=False,
    )
    lda = LinearDiscriminantAnalysis()
    return Pipeline([("CSP", csp), ("LDA", lda)])


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, np.ndarray]:
    """
    Train CSP+LDA and run cross-validation. Returns (fitted pipeline, cv scores).
    """
    clf = build_pipeline()
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
    clf.fit(X, y)
    return clf, scores


def save_model(clf: Pipeline, name: str = "csp_lda") -> Path:
    """Save fitted pipeline to models/."""
    out_dir = get_models_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.joblib"
    joblib.dump(clf, path)
    return path


def load_model(name: str = "csp_lda") -> Pipeline:
    """Load pipeline from models/."""
    path = get_models_dir() / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run train_model.py first.")
    return joblib.load(path)


def run_training(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "csp_lda",
) -> Pipeline:
    """
    Train CSP+LDA on X (n_epochs, n_channels, n_times), evaluate with
    cross-validation, save pipeline to models/, and return fitted pipeline.
    Caller should pass already cropped X if using a custom time window.
    """
    clf, scores = train_and_evaluate(X, y)
    mean_acc = np.mean(scores)
    print(f"Cross-validation accuracy: {mean_acc:.2%}")
    print(f"Per-fold scores: {scores}")
    path = save_model(clf, name=model_name)
    print(f"Model saved to {path}")
    return clf
