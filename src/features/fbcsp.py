"""FBCSP (Filter Bank Common Spatial Patterns) feature extraction.

BCI Competition IV 冠军方案的核心算法。

原理：
  1. 将 EEG 信号��过多个子频带带通滤波器 (如 4-8, 8-12, ..., 28-32 Hz)
  2. 在每个子频带上独立执行 CSP
  3. 拼接所有子带的 CSP 特征 (如 7 子带 × 4 组分 = 28 维)
  4. 用互信息特征选择保留最具判别力的子集 (如 28 → 8)

相比普通 CSP 的优势：
  - 自动从多个子带中选择最优频率信息
  - 对被试间的最优频带差异更鲁棒
"""

from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from mne.filter import filter_data

from src.features.selector import mutual_information_selection
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FBCSPExtractor:
    """Filter Bank CSP feature extractor.

    Parameters
    ----------
    filter_bands : list[tuple[float, float]]
        Sub-band definitions, e.g. ``[(4,8), (8,12), ...]``.
    n_components : int
        CSP components per sub-band.
    selection_method : str
        Feature selection method (currently only ``"mutual_info"``).
    n_select : int
        Number of features to keep after selection.
    sfreq : float
        Sampling rate of the input data (Hz).
    """

    def __init__(
        self,
        filter_bands: list[tuple[float, float]] | None = None,
        n_components: int = 4,
        selection_method: str = "mutual_info",
        n_select: int = 8,
        sfreq: float = 250.0,
    ):
        self.filter_bands = filter_bands or [
            (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32),
        ]
        self.n_components = n_components
        self.selection_method = selection_method
        self.n_select = n_select
        self.sfreq = sfreq

        self._csp_list: list[CSP] = []
        self._selected_indices: list[int] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FBCSPExtractor:
        """Fit FBCSP: per-band CSP + feature selection.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)
        y : np.ndarray, shape (n_trials,)
        """
        self._csp_list = []
        all_features = []

        for lo, hi in self.filter_bands:
            X_filt = filter_data(
                X.astype(np.float64), sfreq=self.sfreq,
                l_freq=lo, h_freq=hi, verbose=False,
            )
            csp = CSP(
                n_components=self.n_components,
                reg=None, log=True, norm_trace=False,
            )
            feats = csp.fit_transform(X_filt, y)
            self._csp_list.append(csp)
            all_features.append(feats)

        # Concatenate: (n_trials, n_bands * n_components)
        X_concat = np.hstack(all_features)

        # Feature selection
        _, self._selected_indices = mutual_information_selection(
            X_concat, y, n_select=self.n_select,
        )
        logger.info(
            "FBCSP: %d bands × %d components = %d features → selected %d (indices: %s)",
            len(self.filter_bands), self.n_components, X_concat.shape[1],
            self.n_select, self._selected_indices,
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data through fitted FBCSP.

        Returns
        -------
        features : np.ndarray, shape (n_trials, n_select)
        """
        if self._selected_indices is None:
            raise RuntimeError("FBCSPExtractor must be fit before transform")
        all_features = []
        for (lo, hi), csp in zip(self.filter_bands, self._csp_list):
            X_filt = filter_data(
                X.astype(np.float64), sfreq=self.sfreq,
                l_freq=lo, h_freq=hi, verbose=False,
            )
            all_features.append(csp.transform(X_filt))

        X_concat = np.hstack(all_features)
        return X_concat[:, self._selected_indices]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_selected_band_info(self) -> list[dict]:
        """Return which sub-band each selected feature belongs to.

        Useful for analysis: verify that different subjects rely on different
        frequency ranges.

        Returns
        -------
        list[dict]
            Each entry has keys ``"feature_index"``, ``"band_index"``,
            ``"band_range"``, ``"csp_component"``.
        """
        info = []
        for idx in (self._selected_indices or []):
            band_idx = idx // self.n_components
            comp_idx = idx % self.n_components
            info.append({
                "feature_index": idx,
                "band_index": band_idx,
                "band_range": self.filter_bands[band_idx],
                "csp_component": comp_idx,
            })
        return info
