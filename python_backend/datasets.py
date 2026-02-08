"""
Dataset download utilities for MI BCI.
- BCI IV 2a/2b: download official GDF zips (bbci.de) and extract to data/.../001-2014 and 004-2014.
- MNE: PhysioNet EEG Motor Movement/Imagery Dataset (eegbci).
Respects MNE_DATA from environment (set via .env or export).
"""
import os
import zipfile
from pathlib import Path
from typing import Optional, List
from urllib.request import urlretrieve

from .utils import get_project_root


# Official BCI Competition IV GDF zip URLs (bbci.de)
BCI_2A_GDF_ZIP = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"
BCI_2B_GDF_ZIP = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip"


def get_mne_data_path() -> Path:
    """
    Return MNE data root directory. Uses MNE_DATA env var if set,
    otherwise default ~/mne_data. Creates directory if missing.
    """
    path = os.environ.get("MNE_DATA") or os.environ.get("MNE_DATASETS_SAMPLE_PATH")
    if path:
        p = Path(os.path.expanduser(path)).resolve()
    else:
        p = Path.home() / "mne_data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _set_mne_data_path(target_path: Path) -> None:
    """Set MNE_DATA in environment and MNE config so MNE/moabb use it."""
    os.environ["MNE_DATA"] = str(target_path)
    try:
        import mne
        mne.set_config("MNE_DATA", str(target_path), set_env=True)
    except Exception:
        pass


def _download_gdf_zip(url: str, out_dir: Path, dataset_name: str) -> None:
    """Download a GDF zip from url and extract all .gdf files into out_dir."""
    import shutil
    import tempfile
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir.parent / (dataset_name + "_gdf.zip")
    print(f"  Downloading {url} ...")
    urlretrieve(url, zip_path)
    print(f"  Extracting .gdf files to {out_dir} ...")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        for gdf_path in Path(tmp).rglob("*.gdf"):
            dest = out_dir / gdf_path.name
            shutil.copy2(gdf_path, dest)
    zip_path.unlink(missing_ok=True)
    print(f"  Done. GDF files in {out_dir}")


def download_bci_iv_2a(
    path: Optional[Path] = None,
    subjects: Optional[List[int]] = None,
) -> Path:
    """
    Download BCI Competition IV 2a in GDF format (official zip from bbci.de).
    Extracts to path/MNE-bnci-data/database/data-sets/001-2014/ (A01T.gdf, A01E.gdf, ...).
    subjects is ignored; zip contains all 9 subjects. Returns the data root path used.
    """
    target = path or get_mne_data_path()
    _set_mne_data_path(target)
    out_dir = target / "MNE-bnci-data" / "database" / "data-sets" / "001-2014"
    print(f"Downloading BCI IV 2a (GDF) to {out_dir} ...")
    _download_gdf_zip(BCI_2A_GDF_ZIP, out_dir, "BCICIV_2a")
    print("BCI IV 2a done.")
    return target


def download_bci_iv_2b(
    path: Optional[Path] = None,
    subjects: Optional[List[int]] = None,
) -> Path:
    """
    Download BCI Competition IV 2b in GDF format (official zip from bbci.de).
    Extracts to path/MNE-bnci-data/database/data-sets/004-2014/ (B01T.gdf, B01E.gdf, ...).
    subjects is ignored; zip contains all subjects. Returns the data root path used.
    """
    target = path or get_mne_data_path()
    _set_mne_data_path(target)
    out_dir = target / "MNE-bnci-data" / "database" / "data-sets" / "004-2014"
    print(f"Downloading BCI IV 2b (GDF) to {out_dir} ...")
    _download_gdf_zip(BCI_2B_GDF_ZIP, out_dir, "BCICIV_2b")
    print("BCI IV 2b done.")
    return target


# Default runs for PhysioNet EEGBCI: motor imagery tasks (hands vs feet commonly used in tutorials)
EEGBCI_DEFAULT_RUNS = [6, 10, 14]  # Motor imagery: left/right hand, both hands/feet, etc.


def download_physionet_eegbci(
    path: Optional[Path] = None,
    subjects: Optional[List[int]] = None,
    runs: Optional[List[int]] = None,
) -> Path:
    """
    Download PhysioNet EEG Motor Movement/Imagery Dataset via MNE (eegbci).
    109 subjects, 14 runs each; runs 6, 10, 14 are motor imagery (hands/feet).
    Data is stored under MNE_DATA (or path). Returns the data root path used.
    """
    from mne.datasets import eegbci

    target = path or get_mne_data_path()
    _set_mne_data_path(target)
    subject_list = subjects if subjects is not None else list(range(1, 11))  # 1-10 by default
    run_list = runs if runs is not None else EEGBCI_DEFAULT_RUNS
    print(f"Downloading PhysioNet EEG Motor Movement/Imagery (eegbci) to {target} (subjects {subject_list}, runs {run_list})...")
    for subj in subject_list:
        eegbci.load_data(subj, run_list, path=str(target))
    print("PhysioNet EEGBCI done.")
    return target


def download_all_bci_datasets(
    path: Optional[Path] = None,
    subjects_2a: Optional[List[int]] = None,
    subjects_2b: Optional[List[int]] = None,
    skip_2b: bool = False,
) -> Path:
    """
    Download BCI IV 2a and optionally 2b to the given path (or MNE_DATA).
    By default downloads subjects 1--9 for both. Set skip_2b=True to only get 2a.
    Returns the data root path used.
    """
    target = path or get_mne_data_path()
    _set_mne_data_path(target)

    sub_2a = subjects_2a if subjects_2a is not None else list(range(1, 10))
    download_bci_iv_2a(path=target, subjects=sub_2a)

    if not skip_2b:
        sub_2b = subjects_2b if subjects_2b is not None else list(range(1, 10))
        download_bci_iv_2b(path=target, subjects=sub_2b)

    return target
