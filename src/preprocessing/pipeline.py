"""Preprocessing pipeline orchestrator.

Reads a ``preprocessing`` config dict and executes steps in order:
    1. Bad channel detection & interpolation
    2. Common Average Reference (CAR)
    3. Bandpass filter
    4. Notch filter (optional)
    5. ICA artifact removal (optional)
    6. Epoching
    7. Bad epoch rejection
"""

from __future__ import annotations

from typing import Any

import mne
import numpy as np
from mne.io import BaseRaw

from src.preprocessing.artifacts import reject_bad_epochs, run_ica_artifact_removal
from src.preprocessing.epochs import create_epochs
from src.preprocessing.filters import bandpass_filter, notch_filter
from src.preprocessing.quality import (
    detect_bad_channels,
    generate_quality_report,
    interpolate_bad_channels,
)
from src.preprocessing.reference import set_common_average_reference
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline:
    """Config-driven preprocessing pipeline.

    Parameters
    ----------
    config : dict
        The ``preprocessing`` section of an experiment config.  Expected keys::

            reference: "car"          # or null
            bandpass: [8, 30]
            notch_filter: null        # or [50.0]
            bad_channel_detection: false
            artifact_rejection:
              method: null            # or "ica"
              n_components: 15
              eog_channels: null
            epoch:
              tmin: -0.5
              tmax: 3.0
              train_tmin: 0.5
              train_tmax: 2.5
              baseline: null
              reject_threshold: 100.0e-6
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._ica: mne.preprocessing.ICA | None = None
        self._quality_report: dict | None = None

    def run(
        self,
        raw: BaseRaw,
        events: np.ndarray,
        event_id: dict[str, int],
    ) -> mne.Epochs:
        """Execute the full pipeline and return clean Epochs.

        Parameters
        ----------
        raw : BaseRaw
            Raw EEG data (will be modified in-place).
        events : np.ndarray
            Event array from data loader.
        event_id : dict
            Event name → code mapping.

        Returns
        -------
        mne.Epochs
            Preprocessed, epoched data ready for feature extraction.
        """
        cfg = self.config

        # --- 1. Bad channel detection & interpolation ---
        if cfg.get("bad_channel_detection", False):
            logger.info("Step 1/7: Bad channel detection")
            detect_bad_channels(raw)
            interpolate_bad_channels(raw)
        else:
            logger.info("Step 1/7: Bad channel detection — skipped")

        # --- 2. Re-reference ---
        ref = cfg.get("reference")
        if ref == "car":
            logger.info("Step 2/7: Common Average Reference")
            set_common_average_reference(raw)
        else:
            logger.info("Step 2/7: Re-reference — skipped")

        # --- 3. Bandpass filter ---
        bp = cfg.get("bandpass", [8, 30])
        logger.info("Step 3/7: Bandpass filter %s Hz", bp)
        bandpass_filter(raw, l_freq=bp[0], h_freq=bp[1])

        # --- 4. Notch filter ---
        notch = cfg.get("notch_filter")
        if notch:
            logger.info("Step 4/7: Notch filter %s Hz", notch)
            notch_filter(raw, freqs=notch)
        else:
            logger.info("Step 4/7: Notch filter — skipped")

        # --- 5. ICA artifact removal ---
        art_cfg = cfg.get("artifact_rejection", {})
        if art_cfg.get("method") == "ica":
            logger.info("Step 5/7: ICA artifact removal")
            raw, self._ica = run_ica_artifact_removal(
                raw,
                n_components=art_cfg.get("n_components", 15),
                eog_channels=art_cfg.get("eog_channels"),
            )
        else:
            logger.info("Step 5/7: ICA — skipped")

        # --- 6. Epoching ---
        ep_cfg = cfg.get("epoch", {})
        tmin = ep_cfg.get("tmin", -0.5)
        tmax = ep_cfg.get("tmax", 3.0)
        baseline = ep_cfg.get("baseline")
        if isinstance(baseline, list):
            baseline = tuple(baseline)
        logger.info("Step 6/7: Epoching [%.1f, %.1f]s", tmin, tmax)
        epochs = create_epochs(
            raw, events, event_id,
            tmin=tmin, tmax=tmax, baseline=baseline,
        )

        # --- 7. Bad epoch rejection ---
        threshold = ep_cfg.get("reject_threshold")
        if threshold is not None:
            logger.info("Step 7/7: Bad epoch rejection (threshold=%.0f µV)", threshold * 1e6)
            epochs = reject_bad_epochs(epochs, reject_threshold=threshold)
        else:
            logger.info("Step 7/7: Bad epoch rejection — skipped")

        # Quality report
        self._quality_report = generate_quality_report(raw, epochs, self._ica)
        logger.info(
            "Pipeline complete: %d epochs, %d bad channels, %d dropped",
            len(epochs),
            self._quality_report["n_bad_channels"],
            self._quality_report.get("n_epochs_dropped", 0),
        )

        return epochs

    @property
    def ica(self) -> mne.preprocessing.ICA | None:
        """The fitted ICA object (``None`` if ICA was not run)."""
        return self._ica

    @property
    def quality_report(self) -> dict | None:
        """Quality metrics from the last ``run()`` call."""
        return self._quality_report
