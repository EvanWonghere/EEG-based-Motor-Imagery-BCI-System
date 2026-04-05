"""CSP spatial pattern topographic maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne


def plot_csp_patterns(
    patterns: np.ndarray,
    info: mne.Info,
    title: str = "CSP Patterns",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot CSP spatial patterns as topographic maps.

    Parameters
    ----------
    patterns : (n_components, n_channels)
        CSP spatial patterns from ``CSPExtractor.get_patterns()``.
    info : mne.Info
        MNE Info with channel locations.
    title : str
        Figure title.
    save_path : path or None
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_components = patterns.shape[0]
    fig, axes = plt.subplots(1, n_components, figsize=(3 * n_components, 3))
    if n_components == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        mne.viz.plot_topomap(
            patterns[idx],
            info,
            axes=ax,
            show=False,
        )
        side = "L" if idx < n_components // 2 else "R"
        ax.set_title(f"CSP {idx + 1} ({side})")

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
