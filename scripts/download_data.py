#!/usr/bin/env python
"""CLI entry point for downloading BCI datasets.

Usage:
    python scripts/download_data.py                     # 2a + 2b
    python scripts/download_data.py --2a-only           # 2a only
    python scripts/download_data.py --physionet-eegbci  # also PhysioNet
    python scripts/download_data.py --path /custom/path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logging import setup_logging
from src.utils.paths import get_data_dir
from src.data.download import download_bci_iv_2a, download_bci_iv_2b, download_physionet_eegbci


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Download BCI datasets")
    parser.add_argument("--path", type=Path, default=None, help="Target directory")
    parser.add_argument("--2a-only", action="store_true", dest="two_a_only")
    parser.add_argument("--physionet-eegbci", action="store_true", dest="physionet")
    parser.add_argument("--physionet-only", action="store_true", dest="physionet_only")
    parser.add_argument("--eegbci-subjects", type=str, default=None)
    args = parser.parse_args()

    path = args.path or get_data_dir()
    path.mkdir(parents=True, exist_ok=True)
    print(f"Data path: {path}")

    eegbci_subjects = None
    if args.eegbci_subjects:
        eegbci_subjects = [int(s) for s in args.eegbci_subjects.split(",")]

    if args.physionet_only:
        download_physionet_eegbci(path=path, subjects=eegbci_subjects)
        return

    if not args.two_a_only:
        download_bci_iv_2a(path=path)
        download_bci_iv_2b(path=path)
    else:
        download_bci_iv_2a(path=path)

    if args.physionet:
        download_physionet_eegbci(path=path, subjects=eegbci_subjects)

    print("Done.")


if __name__ == "__main__":
    main()
