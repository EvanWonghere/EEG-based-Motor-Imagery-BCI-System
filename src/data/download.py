"""Dataset download utilities.

Supports:
- BCI Competition IV 2a/2b — official GDF zips from bbci.de
- PhysioNet EEG Motor Movement/Imagery — via MNE eegbci
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from src.utils.logging import get_logger
from src.utils.paths import get_data_dir

logger = get_logger(__name__)

BCI_2A_GDF_ZIP = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"
BCI_2B_GDF_ZIP = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip"

EEGBCI_DEFAULT_RUNS = [6, 10, 14]


def _get_mne_data_path(override: Path | None = None) -> Path:
    """Return MNE data root, creating it if needed."""
    if override:
        p = override.resolve()
    else:
        env = os.environ.get("MNE_DATA") or os.environ.get("MNE_DATASETS_SAMPLE_PATH")
        p = Path(os.path.expanduser(env)).resolve() if env else Path.home() / "mne_data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _set_mne_env(path: Path) -> None:
    os.environ["MNE_DATA"] = str(path)
    try:
        import mne
        mne.set_config("MNE_DATA", str(path), set_env=True)
    except Exception:
        pass


def _download_gdf_zip(url: str, out_dir: Path, dataset_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir.parent / f"{dataset_name}_gdf.zip"
    logger.info("Downloading %s ...", url)
    urlretrieve(url, zip_path)
    logger.info("Extracting .gdf files to %s ...", out_dir)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        for gdf in Path(tmp).rglob("*.gdf"):
            shutil.copy2(gdf, out_dir / gdf.name)
    zip_path.unlink(missing_ok=True)
    logger.info("Done: %s", out_dir)


def download_bci_iv_2a(path: Path | None = None) -> Path:
    """Download BCI IV 2a GDF files. Returns data root."""
    target = path or get_data_dir()
    _set_mne_env(target)
    out_dir = target / "MNE-bnci-data" / "database" / "data-sets" / "001-2014"
    _download_gdf_zip(BCI_2A_GDF_ZIP, out_dir, "BCICIV_2a")
    return target


def download_bci_iv_2b(path: Path | None = None) -> Path:
    """Download BCI IV 2b GDF files. Returns data root."""
    target = path or get_data_dir()
    _set_mne_env(target)
    out_dir = target / "MNE-bnci-data" / "database" / "data-sets" / "004-2014"
    _download_gdf_zip(BCI_2B_GDF_ZIP, out_dir, "BCICIV_2b")
    return target


def download_physionet_eegbci(
    path: Path | None = None,
    subjects: list[int] | None = None,
    runs: list[int] | None = None,
) -> Path:
    """Download PhysioNet EEG Motor Movement/Imagery via MNE."""
    from mne.datasets import eegbci

    target = path or get_data_dir()
    _set_mne_env(target)
    subject_list = subjects or list(range(1, 11))
    run_list = runs or EEGBCI_DEFAULT_RUNS
    logger.info("Downloading PhysioNet EEGBCI (subjects %s, runs %s) ...", subject_list, run_list)
    for subj in subject_list:
        eegbci.load_data(subj, run_list, path=str(target))
    logger.info("Done: PhysioNet EEGBCI")
    return target
