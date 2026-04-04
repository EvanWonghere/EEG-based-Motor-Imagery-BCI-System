"""CSP (Common Spatial Patterns) feature extraction.

Finds spatial filters that maximise variance for one class while minimising
it for the other.  Outputs log-variance features.
"""

from __future__ import annotations

import numpy as np
from mne.decoding import CSP


class CSPExtractor:
    """Thin wrapper around ``mne.decoding.CSP`` with a consistent API.

    Parameters
    ----------
    n_components : int
        Number of CSP components (half from each end of the eigenvalue
        spectrum).  4 is a common default for 2-class MI.
    log : bool
        If ``True``, return log-variance features (recommended).
    reg : str | float | None
        Regularisation for covariance estimation.
    """

    def __init__(self, n_components: int = 4, log: bool = True, reg: float | None = None):
        self.csp = CSP(
            n_components=n_components,
            reg=reg,
            log=log,
            norm_trace=False,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> CSPExtractor:
        """Fit CSP on epochs ``X`` (n_trials, n_channels, n_times)."""
        self.csp.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform epochs to CSP features (n_trials, n_components)."""
        return self.csp.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_patterns(self) -> np.ndarray:
        """Return spatial patterns (n_components, n_channels) for topomap plotting."""
        return self.csp.patterns_
