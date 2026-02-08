"""
Utilities and constants for the MI BCI pipeline.
"""
from pathlib import Path

# Bandpass filter for Mu (8-13 Hz) and Beta (13-30 Hz) rhythms
BAND_LOW_HZ = 8.0
BAND_HIGH_HZ = 30.0

# BCI Competition IV 2a event IDs (when using .gdf data)
EVENT_ID_BCI_2A = {
    "769": "left_hand",
    "770": "right_hand",
    "771": "feet",
    "772": "tongue",
}

# Epoch time window (seconds relative to event)
EPOCH_TMIN = -0.5
EPOCH_TMAX = 3.0
# Optional crop for training (strongest MI period)
EPOCH_TRAIN_TMIN = 0.5
EPOCH_TRAIN_TMAX = 2.5


def get_project_root() -> Path:
    """Return project root (parent of python_backend)."""
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Return path to data directory."""
    return get_project_root() / "data"


# BCI IV 2a / 2b 数据所在子目录（相对 data/，与 MNE/MOABB 下载结构一致）
# 例如: data/MNE-bnci-data/database/data-sets/001-2014/A01E.mat
DIR_BCI_2A = "MNE-bnci-data/database/data-sets/001-2014"
DIR_BCI_2B = "MNE-bnci-data/database/data-sets/004-2014"


def get_models_dir() -> Path:
    """Return path to models directory."""
    return get_project_root() / "models"
