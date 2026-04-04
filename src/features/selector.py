"""Feature selection methods."""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif


def mutual_information_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_select: int,
    random_state: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """Select top *n_select* features by mutual information.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    n_select : int
        Number of features to keep.
    random_state : int

    Returns
    -------
    X_selected : np.ndarray, shape (n_samples, n_select)
    selected_indices : list[int]
        Indices of the selected features in the original feature space.
    """
    mi = mutual_info_classif(X, y, random_state=random_state)
    indices = np.argsort(mi)[::-1][:n_select].tolist()
    indices.sort()  # keep original order for reproducibility
    return X[:, indices], indices
