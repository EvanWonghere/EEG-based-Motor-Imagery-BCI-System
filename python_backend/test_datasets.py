"""
Test that downloaded datasets can be loaded.
Run from project root:  python python_backend/test_datasets.py
Loads .env so MNE_DATA is used; skips datasets that are not present.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python_backend.datasets import get_mne_data_path, _set_mne_data_path


def test_physionet_eegbci() -> bool:
    """Test PhysioNet EEG Motor Movement/Imagery (MNE eegbci)."""
    print("\n--- PhysioNet EEG Motor Movement/Imagery (eegbci) ---")
    try:
        from mne.datasets import eegbci
        from mne.io import concatenate_raws, read_raw_edf

        path = get_mne_data_path()
        _set_mne_data_path(path)
        # Subject 1, runs 6/10/14 (motor imagery)
        fnames = eegbci.load_data(1, [6, 10, 14], path=str(path))
        if not fnames:
            print("  SKIP: No files found (dataset not downloaded?). Run: python python_backend/download_datasets.py --physionet-eegbci-only")
            return False
        raw = concatenate_raws([read_raw_edf(f, preload=False) for f in fnames])
        eegbci.standardize(raw)
        print(f"  OK: subject 1, runs [6,10,14] -> {len(raw.ch_names)} ch, sfreq={raw.info['sfreq']} Hz, duration={raw.times[-1]:.1f}s")
        if raw.annotations is not None and len(raw.annotations) > 0:
            print(f"  Events: {len(raw.annotations)} annotations, descriptions: {set(raw.annotations.description)}")
        return True
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_bci_iv_2a() -> bool:
    """Test BCI Competition IV 2a (BNCI2014_001) via MOABB."""
    print("\n--- BCI Competition IV 2a (BNCI2014_001) ---")
    try:
        from moabb.datasets import BNCI2014_001

        path = get_mne_data_path()
        _set_mne_data_path(path)
        dataset = BNCI2014_001()
        data = dataset.get_data(subjects=[1])
        if not data or 1 not in data:
            print("  SKIP: Subject 1 not found (dataset not downloaded?). Run: python python_backend/download_datasets.py --2a-only")
            return False
        # data[1] is dict of session_name -> dict of run_name -> Raw
        n_sessions = len(data[1])
        n_runs = sum(len(runs) for runs in data[1].values())
        first_session = next(iter(data[1].values()))
        raw = next(iter(first_session.values()))
        print(f"  OK: subject 1 -> {n_sessions} session(s), {n_runs} run(s), {len(raw.ch_names)} ch, sfreq={raw.info['sfreq']} Hz")
        return True
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_bci_iv_2b() -> bool:
    """Test BCI Competition IV 2b (BNCI2014_004) via MOABB."""
    print("\n--- BCI Competition IV 2b (BNCI2014_004) ---")
    try:
        from moabb.datasets import BNCI2014_004

        path = get_mne_data_path()
        _set_mne_data_path(path)
        dataset = BNCI2014_004()
        data = dataset.get_data(subjects=[1])
        if not data or 1 not in data:
            print("  SKIP: Subject 1 not found (dataset not downloaded?). Run: python python_backend/download_datasets.py")
            return False
        first_session = next(iter(data[1].values()))
        raw = next(iter(first_session.values()))
        print(f"  OK: subject 1 -> {len(raw.ch_names)} ch, sfreq={raw.info['sfreq']} Hz")
        return True
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main() -> None:
    print(f"MNE_DATA path: {get_mne_data_path()}")
    results = []
    results.append(("PhysioNet EEGBCI", test_physionet_eegbci()))
    results.append(("BCI IV 2a", test_bci_iv_2a()))
    results.append(("BCI IV 2b", test_bci_iv_2b()))
    print("\n" + "=" * 50)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {name}: {'OK' if ok else 'SKIP/FAIL'}")
    print(f"Summary: {passed}/{len(results)} datasets loaded successfully.")


if __name__ == "__main__":
    main()
