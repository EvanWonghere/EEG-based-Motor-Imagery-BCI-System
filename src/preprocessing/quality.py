"""Bad channel detection and data quality reporting."""

from __future__ import annotations

import numpy as np
from mne.io import BaseRaw

from src.utils.logging import get_logger

logger = get_logger(__name__)


def detect_bad_channels(
    raw: BaseRaw,
    zscore_threshold: float = 3.0,
) -> list[str]:
    """Detect bad channels based on abnormal variance.

    Channels whose log-variance is more than *zscore_threshold* standard
    deviations from the median are marked as bad.

    Returns the list of bad channel names (also sets ``raw.info['bads']``).
    """
    data = raw.get_data(picks="eeg")  # (n_channels, n_times)
    variances = np.var(data, axis=1)

    log_var = np.log(variances + 1e-30)
    median = np.median(log_var)
    mad = np.median(np.abs(log_var - median))
    # Robust z-score using MAD
    if mad > 0:
        z = np.abs(log_var - median) / (mad * 1.4826)
    else:
        z = np.zeros_like(log_var)

    eeg_names = [raw.ch_names[i] for i in range(len(raw.ch_names))
                 if raw.get_channel_types([i])[0] == "eeg"]
    bad_chs = [name for name, zi in zip(eeg_names, z) if zi > zscore_threshold]

    if bad_chs:
        raw.info["bads"] = list(set(raw.info["bads"]) | set(bad_chs))
        logger.info("Bad channels detected: %s", bad_chs)

    return bad_chs


def interpolate_bad_channels(raw: BaseRaw) -> BaseRaw:
    """Interpolate channels listed in ``raw.info['bads']``."""
    if raw.info["bads"]:
        logger.info("Interpolating %d bad channel(s): %s", len(raw.info["bads"]), raw.info["bads"])
        raw.interpolate_bads(verbose=False)
    return raw


def generate_quality_report(
    raw: BaseRaw,
    epochs: "mne.Epochs | None" = None,
    ica: "mne.preprocessing.ICA | None" = None,
) -> dict:
    """Generate a summary of data quality metrics.

    Returns a dict with keys like ``n_bad_channels``, ``n_ica_excluded``,
    ``n_epochs_dropped``, etc.
    """
    report: dict = {
        "n_channels": len(raw.ch_names),
        "n_bad_channels": len(raw.info["bads"]),
        "bad_channels": list(raw.info["bads"]),
        "duration_s": raw.times[-1],
        "sfreq": raw.info["sfreq"],
    }

    if ica is not None:
        report["n_ica_components"] = ica.n_components_
        report["n_ica_excluded"] = len(ica.exclude)
        report["ica_excluded"] = list(ica.exclude)

    if epochs is not None:
        report["n_epochs_total"] = len(epochs.events)
        drop_log = epochs.drop_log
        n_dropped = sum(1 for d in drop_log if len(d) > 0)
        report["n_epochs_dropped"] = n_dropped
        report["epochs_drop_rate"] = n_dropped / max(len(drop_log), 1)

    return report
