"""Traditional classifiers: LDA, SVM-RBF, RandomForest."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.models.base import BaseModel


class TraditionalModel(BaseModel):
    """Wrapper around sklearn classifiers with a uniform API.

    Parameters
    ----------
    model_type : str
        ``"lda"`` | ``"svm"`` | ``"rf"``
    params : dict
        Hyperparameters forwarded to the sklearn constructor.
        For SVM, ``probability=True`` is forced so that ``predict_proba`` works.
    """

    def __init__(self, model_type: str = "lda", params: dict[str, Any] | None = None):
        self.model_type = model_type.lower()
        self._params = dict(params or {})
        self._scaler: StandardScaler | None = None
        self._clf = self._build_classifier()

    def _build_classifier(self):
        if self.model_type == "lda":
            return LinearDiscriminantAnalysis(**self._params)

        if self.model_type == "svm":
            p = dict(self._params)
            p.setdefault("kernel", "rbf")
            p.setdefault("C", 1.0)
            p["probability"] = True     # required for predict_proba
            self._scaler = StandardScaler()
            return SVC(**p)

        if self.model_type == "rf":
            p = dict(self._params)
            p.setdefault("n_estimators", 100)
            return RandomForestClassifier(**p)

        raise ValueError(
            f"Unknown model_type {self.model_type!r}. Use 'lda', 'svm', or 'rf'."
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)
        self._clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._clf.predict_proba(X)

    @property
    def sklearn_clf(self):
        """Access the underlying sklearn classifier (for introspection)."""
        return self._clf
