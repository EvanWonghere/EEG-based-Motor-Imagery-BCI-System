"""ERD/ERS (Event-Related Desynchronization/Synchronization) time-frequency maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_multitaper


def plot_erds_maps(
    epochs: mne.Epochs,
    picks: list[str] | None = None,
    freqs: np.ndarray | None = None,
    baseline: tuple[float, float] = (-0.5, 0.0),
    tmin: float | None = None,
    tmax: float | None = None,
    title: str = "ERD/ERS Maps",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot ERD/ERS time-frequency maps per channel and event class.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data with event_id containing class labels.
    picks : list of channel names
        Channels to plot.  Default: C3, Cz, C4 (motor cortex).
    freqs : array
        Frequencies for TFR.  Default: 4–40 Hz in 1 Hz steps.
    baseline : (tmin, tmax)
        Baseline interval for ERD/ERS percent change.
    tmin, tmax : float or None
        Crop time window for display.
    title : str
        Figure title.
    save_path : path or None
        If given, save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if picks is None:
        available = epochs.ch_names
        motor_chs = ["C3", "Cz", "C4"]
        picks = [ch for ch in motor_chs if ch in available]
        if not picks:
            picks = available[:3]

    if freqs is None:
        freqs = np.arange(4, 41, 1.0)

    event_ids = epochs.event_id
    n_classes = len(event_ids)
    n_channels = len(picks)

    fig, axes = plt.subplots(
        n_classes, n_channels,
        figsize=(4 * n_channels, 3 * n_classes),
        squeeze=False,
    )

    for row, (event_name, event_code) in enumerate(sorted(event_ids.items())):
        ep_class = epochs[event_name]
        power = tfr_multitaper(
            ep_class,
            freqs=freqs,
            n_cycles=freqs / 2.0,
            picks=picks,
            return_itc=False,
            average=True,
            verbose=False,
        )
        power.apply_baseline(baseline=baseline, mode="percent")

        for col, ch_name in enumerate(picks):
            ax = axes[row, col]
            ch_idx = power.ch_names.index(ch_name)
            data = power.data[ch_idx]  # (n_freqs, n_times)

            im = ax.imshow(
                data * 100,  # percent
                aspect="auto",
                origin="lower",
                extent=[power.times[0], power.times[-1], freqs[0], freqs[-1]],
                cmap="RdBu_r",
                vmin=-100,
                vmax=100,
            )

            if row == 0:
                ax.set_title(ch_name)
            if col == 0:
                ax.set_ylabel(f"{event_name}\nFreq (Hz)")
            if row == n_classes - 1:
                ax.set_xlabel("Time (s)")

            plt.colorbar(im, ax=ax, label="% change")

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
