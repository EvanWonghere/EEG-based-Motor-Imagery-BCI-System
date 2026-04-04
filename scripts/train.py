#!/usr/bin/env python
"""Training entry point: config-driven training pipeline.

Usage:
    python scripts/train.py --config configs/experiments/csp_lda_baseline.yaml
    python scripts/train.py --config configs/experiments/fbcsp_svm.yaml --subject 1
    python scripts/train.py --config configs/experiments/eegnet.yaml --subject 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import create_dataset
from src.features import create_extractor
from src.models import create_model
from src.models.base import BaseModel
from src.preprocessing import PreprocessingPipeline
from src.preprocessing.epochs import epochs_to_arrays
from src.utils.config import load_config, save_config_snapshot
from src.utils.logging import setup_logging, get_logger
from src.utils.paths import ensure_dir, get_results_dir
from src.utils.reproducibility import set_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# sklearn-compatible wrapper for cross-validation
# ---------------------------------------------------------------------------

class _SklearnWrapper(BaseEstimator, ClassifierMixin):
    """Wrap our extractor + model into an sklearn-style estimator for CV.

    Inherits ``BaseEstimator`` / ``ClassifierMixin`` so ``cross_val_score`` works
    with scikit-learn >= 1.6 (``get_tags`` / ``__sklearn_tags__``).
    """

    def __init__(self, extractor=None, model: Optional[BaseModel] = None):
        self.extractor = extractor
        self.model = model

    def fit(self, X, y):
        if self.model is None:
            raise ValueError("model must be provided")
        self.classes_ = np.unique(y)
        if self.extractor is not None:
            feats = self.extractor.fit_transform(X, y)
        else:
            feats = X
        self.model.fit(feats, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("model must be provided before predict")
        if self.extractor is not None:
            feats = self.extractor.transform(X)
        else:
            feats = X
        return self.model.predict(feats)


# ---------------------------------------------------------------------------
# Core training logic
# ---------------------------------------------------------------------------

def train_subject(
    cfg: dict,
    subject_id: int,
    output_dir: Path,
) -> dict:
    """Train on one subject, return results dict."""
    # --- Data loading ---
    ds_cfg = cfg["dataset"]
    ds = create_dataset(
        ds_cfg["name"],
        data_dir=str(ROOT / "data"),
        subjects=[subject_id],
    )
    ep_cfg = cfg["preprocessing"].get("epoch", {})
    sessions = ds_cfg.get("train_sessions") or ["session_T"]
    session_letters = [str(s)[-1].upper() for s in sessions]
    all_X = []
    all_y = []

    for session in session_letters:
        try:
            raw, events, event_id = ds.load_data(subject_id, session=session)
        except FileNotFoundError:
            logger.warning("Subject %d: session %s not found, skipping", subject_id, session)
            continue

        logger.info("Subject %d session %s: %d events loaded", subject_id, session, len(events))

        # --- Preprocessing ---
        pipe = PreprocessingPipeline(cfg["preprocessing"])
        try:
            epochs = pipe.run(raw, events, event_id)
        except ValueError as e:
            logger.warning(
                "Subject %d session %s: preprocessing/epoching failed (%s), skipping",
                subject_id,
                session,
                e,
            )
            continue

        X_s, y_s = epochs_to_arrays(
            epochs,
            tmin=ep_cfg.get("train_tmin"),
            tmax=ep_cfg.get("train_tmax"),
        )
        all_X.append(X_s)
        all_y.append(y_s)

    if not all_X:
        raise FileNotFoundError(f"Subject {subject_id}: no valid session data found")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    logger.info("Subject %d: X=%s, classes=%s", subject_id, X.shape, np.unique(y).tolist())

    if len(np.unique(y)) < 2:
        logger.warning("Subject %d: fewer than 2 classes, skipping", subject_id)
        return {"subject": subject_id, "skipped": True}

    # --- Feature extractor ---
    sfreq = ds.sfreq
    extractor = create_extractor(cfg["features"], sfreq=sfreq)

    # --- Model ---
    model_cfg = cfg["model"]
    model_kwargs = {}
    if model_cfg["type"] == "eegnet":
        model_kwargs.update(n_channels=X.shape[1], n_times=X.shape[2])

    model = create_model(model_cfg, **model_kwargs)

    # --- Cross-validation ---
    eval_cfg = cfg.get("evaluation", {})
    n_splits = eval_cfg.get("n_splits", 10)

    if model_cfg["type"] == "eegnet":
        # EEGNet uses its own train/val split; do simple K-fold manually
        scores = _kfold_deep(model, X, y, n_splits, cfg)
    else:
        wrapper = _SklearnWrapper(extractor, model)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.get("seed", 42))
        scores = cross_val_score(wrapper, X, y, cv=cv, scoring="accuracy")

    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores))
    logger.info(
        "Subject %d [%s]: %.2f%% ± %.2f%% (per-fold: %s)",
        subject_id, model_cfg["type"].upper(),
        mean_acc * 100, std_acc * 100, np.round(scores, 3).tolist(),
    )

    # --- Fit on full data & save ---
    if extractor is not None:
        X_feat = extractor.fit_transform(X, y)
    else:
        X_feat = X
    model.fit(X_feat, y)

    exp_name = cfg.get("experiment", {}).get("name", "exp")
    model_name = f"{exp_name}_sub{subject_id}"
    model_path = output_dir / "models" / f"{model_name}.pkl"
    model.save(model_path)

    # Save replay data
    if cfg.get("output", {}).get("save_replay_data", False):
        replay_path = output_dir / "models" / f"replay_data_sub{subject_id}.npz"
        np.savez_compressed(replay_path, X=X, y=y)

    return {
        "subject": subject_id,
        "skipped": False,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "scores": scores.tolist(),
        "model_path": str(model_path),
    }


def _kfold_deep(model, X, y, n_splits, cfg):
    """Manual K-fold for deep learning models (need fresh model each fold)."""
    from src.models import create_model

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.get("seed", 42))
    scores = []
    model_cfg = cfg["model"]

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fold_model = create_model(
            model_cfg, n_channels=X.shape[1], n_times=X.shape[2],
        )
        fold_model.fit(X[train_idx], y[train_idx])
        preds = fold_model.predict(X[test_idx])
        acc = np.mean(preds == y[test_idx])
        scores.append(acc)
        logger.info("  Fold %d/%d: %.2f%%", fold + 1, n_splits, acc * 100)

    return np.array(scores)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven MI-BCI training")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML")
    parser.add_argument("--subject", type=int, default=None, help="Train single subject (overrides config)")
    args = parser.parse_args()

    setup_logging()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    exp_name = cfg.get("experiment", {}).get("name", "experiment")
    output_dir = ensure_dir(get_results_dir() / exp_name)
    save_config_snapshot(cfg, output_dir)

    subjects = [args.subject] if args.subject else cfg["dataset"].get("subjects", [1])

    logger.info("Experiment: %s", exp_name)
    logger.info("Subjects: %s", subjects)
    logger.info("Model: %s | Features: %s", cfg["model"]["type"], cfg["features"].get("method"))

    all_results = []
    for sid in subjects:
        try:
            result = train_subject(cfg, sid, output_dir)
            all_results.append(result)
        except FileNotFoundError as e:
            logger.warning("Subject %d skipped: %s", sid, e)
        except Exception as e:
            logger.error("Subject %d failed: %s", sid, e, exc_info=True)

    # --- Summary table ---
    print("\n" + "=" * 55)
    print(f"  {exp_name} — Results")
    print("=" * 55)
    print(f"{'Subject':>8}  {'Accuracy':>10}  {'Std':>8}  {'Status'}")
    print("-" * 55)
    for r in all_results:
        if r.get("skipped"):
            print(f"  {r['subject']:>5}    {'—':>10}  {'—':>8}  skipped")
        else:
            print(f"  {r['subject']:>5}    {r['mean_accuracy']:>9.2%}  {r['std_accuracy']:>7.2%}  ok")

    valid = [r for r in all_results if not r.get("skipped")]
    if valid:
        mean_all = np.mean([r["mean_accuracy"] for r in valid])
        print("-" * 55)
        print(f"  {'Mean':>5}    {mean_all:>9.2%}")
    print()


if __name__ == "__main__":
    main()
