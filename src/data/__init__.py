"""Data management: dataset registry and factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.base import EEGDataset
from src.data.bci_iv_2a import BCIIV2aDataset
from src.data.bci_iv_2b import BCIIV2bDataset
from src.data.physionet_eegbci import PhysioNetEEGBCIDataset

DATASET_REGISTRY: dict[str, type[EEGDataset]] = {
    "bci_iv_2a": BCIIV2aDataset,
    "bci_iv_2b": BCIIV2bDataset,
    "physionet_eegbci": PhysioNetEEGBCIDataset,
}


def create_dataset(name: str, **kwargs: Any) -> EEGDataset:
    """Instantiate a dataset by registry name.

    Parameters
    ----------
    name : str
        Key in ``DATASET_REGISTRY`` (e.g. ``"bci_iv_2a"``).
    **kwargs
        Forwarded to the dataset constructor (``data_dir``, ``subjects``, etc.).
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset {name!r}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name](**kwargs)


__all__ = [
    "EEGDataset",
    "BCIIV2aDataset",
    "BCIIV2bDataset",
    "PhysioNetEEGBCIDataset",
    "DATASET_REGISTRY",
    "create_dataset",
]
