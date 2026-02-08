"""
CLI entry point to download BCI datasets.
- BCI Competition IV 2a/2b via MOABB (BNCI2014_001, BNCI2014_004).
- PhysioNet EEG Motor Movement/Imagery via MNE (eegbci).
Loads .env from project root so MNE_DATA is applied.
Usage:
  python python_backend/download_datasets.py                    # 2a + 2b
  python python_backend/download_datasets.py --2a-only         # 2a only
  python python_backend/download_datasets.py --physionet-eegbci # 2a + 2b + PhysioNet EEGBCI
  python python_backend/download_datasets.py --physionet-eegbci-only  # PhysioNet EEGBCI only
  python python_backend/download_datasets.py --path /custom/path
"""
from pathlib import Path
import sys
import argparse

# Load .env before any MNE/moabb imports
ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python_backend.datasets import (
    get_mne_data_path,
    download_bci_iv_2a,
    download_bci_iv_2b,
    download_all_bci_datasets,
    download_physionet_eegbci,
)
from python_backend.utils import get_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BCI datasets: BCI IV 2a/2b in GDF (official zip), PhysioNet EEGBCI via MNE. Default: project data/."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Target directory (default: MNE_DATA from .env or ~/mne_data)",
    )
    parser.add_argument(
        "--2a-only",
        action="store_true",
        dest="two_a_only",
        help="Only download BCI IV 2a (BNCI2014_001), skip 2b",
    )
    parser.add_argument(
        "--physionet-eegbci",
        action="store_true",
        dest="physionet_eegbci",
        help="Also download PhysioNet EEG Motor Movement/Imagery (MNE eegbci)",
    )
    parser.add_argument(
        "--physionet-eegbci-only",
        action="store_true",
        dest="physionet_eegbci_only",
        help="Download only PhysioNet EEG Motor Movement/Imagery (skip BCI IV 2a/2b)",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs for BCI IV, e.g. 1,2,3 (default: 1-9)",
    )
    parser.add_argument(
        "--eegbci-subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs for PhysioNet EEGBCI only, e.g. 1,2,3 (default: 1-10)",
    )
    args = parser.parse_args()

    path = args.path
    if path is None:
        path = get_data_dir()
        print(f"Using data path: {path} (project data/; use --path to override)")
    else:
        path = path.resolve()
        path.mkdir(parents=True, exist_ok=True)
        print(f"Using data path: {path}")

    subjects = None
    if args.subjects:
        subjects = [int(s.strip()) for s in args.subjects.split(",")]

    eegbci_subjects = None
    if args.eegbci_subjects:
        eegbci_subjects = [int(s.strip()) for s in args.eegbci_subjects.split(",")]

    if args.physionet_eegbci_only:
        print("\nDownloading PhysioNet EEG Motor Movement/Imagery (eegbci) only...")
        download_physionet_eegbci(path=path, subjects=eegbci_subjects)
        print(f"\nDone. Data under: {path}")
        return

    if not args.physionet_eegbci:
        # BCI IV only
        if args.two_a_only:
            print("\nDownloading BCI Competition IV 2a (BNCI2014_001) only...")
            download_bci_iv_2a(path=path, subjects=subjects)
        else:
            print("\nDownloading BCI Competition IV 2a and 2b...")
            download_all_bci_datasets(
                path=path,
                subjects_2a=subjects,
                subjects_2b=subjects,
                skip_2b=False,
            )
        print(f"\nDone. Data under: {path}")
        print("GDF files are in data/MNE-bnci-data/database/data-sets/001-2014 and 004-2014.")
        return

    # With --physionet-eegbci: run BCI IV then PhysioNet EEGBCI
    if args.two_a_only:
        print("\nDownloading BCI Competition IV 2a (BNCI2014_001)...")
        download_bci_iv_2a(path=path, subjects=subjects)
    else:
        print("\nDownloading BCI Competition IV 2a and 2b...")
        download_all_bci_datasets(
            path=path,
            subjects_2a=subjects,
            subjects_2b=subjects,
            skip_2b=False,
        )
    print("\nDownloading PhysioNet EEG Motor Movement/Imagery (eegbci)...")
    download_physionet_eegbci(path=path, subjects=eegbci_subjects)
    print(f"\nDone. Data under: {path}")
    print("BCI IV: data/MNE-bnci-data/database/data-sets/.... PhysioNet EEGBCI: under same path.")


if __name__ == "__main__":
    main()
