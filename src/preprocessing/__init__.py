"""Preprocessing pipeline and individual processing steps."""

from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.epochs import create_epochs, epochs_to_arrays
from src.preprocessing.filters import bandpass_filter, notch_filter
from src.preprocessing.reference import set_common_average_reference
from src.preprocessing.artifacts import run_ica_artifact_removal, reject_bad_epochs
from src.preprocessing.quality import detect_bad_channels, interpolate_bad_channels

__all__ = [
    "PreprocessingPipeline",
    "create_epochs",
    "epochs_to_arrays",
    "bandpass_filter",
    "notch_filter",
    "set_common_average_reference",
    "run_ica_artifact_removal",
    "reject_bad_epochs",
    "detect_bad_channels",
    "interpolate_bad_channels",
]
