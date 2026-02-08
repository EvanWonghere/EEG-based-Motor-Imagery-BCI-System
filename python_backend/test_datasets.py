"""
Test that downloaded datasets can be loaded.
Run from project root:  python python_backend/test_datasets.py
2a/2b: loads via train_model (GDF preferred, e.g. A01E.gdf in data/.../001-2014). Same paths as training.
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

# NumPy 2: fromstring(binary) removed; patch before MNE/GDF reads (MNE edf.py uses fromstring(etmode, UINT8))
import numpy as np
_orig_fromstring = getattr(np, "fromstring", None)
if _orig_fromstring is not None:
    def _fromstring_compat(s, dtype=float, count=-1, sep=""):
        if sep == "":
            try:
                return np.frombuffer(s, dtype=dtype, count=count)
            except (TypeError, ValueError):
                if isinstance(s, str):
                    return np.frombuffer(s.encode("latin-1"), dtype=dtype, count=count)
                raise
        return _orig_fromstring(s, dtype=dtype, count=count, sep=sep)
    np.fromstring = _fromstring_compat

from python_backend.utils import get_data_dir
from python_backend.datasets import get_mne_data_path, _set_mne_data_path


def test_physionet_eegbci() -> bool:
    """Test PhysioNet EEG Motor Movement/Imagery (MNE eegbci)."""
    print("\n--- PhysioNet EEG Motor Movement/Imagery (eegbci) ---")
    try:
        from mne.datasets import eegbci
        from mne.io import concatenate_raws, read_raw_edf

        for path in (get_data_dir(), get_mne_data_path()):
            _set_mne_data_path(path)
            fnames = eegbci.load_data(1, [6, 10, 14], path=str(path))
            if fnames:
                break
        if not fnames:
            print("  SKIP: No files found. Run: python python_backend/download_datasets.py --physionet-eegbci-only")
            return False
        raw = concatenate_raws([read_raw_edf(f, preload=False) for f in fnames])
        eegbci.standardize(raw)
        print(f"  OK: subject 1, runs [6,10,14] -> {len(raw.ch_names)} ch, sfreq={raw.info['sfreq']} Hz, duration={raw.times[-1]:.1f}s")
        if raw.annotations is not None and len(raw.annotations) > 0:
            print(f"  Events: {len(raw.annotations)} annotations")
        return True
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_bci_iv_2a() -> bool:
    """Test BCI Competition IV 2a (GDF e.g. A01E.gdf in data/.../001-2014)."""
    print("\n--- BCI Competition IV 2a (GDF) ---")
    try:
        from python_backend.train_model import load_bci_iv_2a

        raw, events, event_id = load_bci_iv_2a(1)
        n_trials = len(events)
        print(f"  OK: subject 1 -> {len(raw.ch_names)} ch, sfreq={raw.info['sfreq']} Hz, {n_trials} trials (event_id: {event_id})")
        return True
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        print("  Run: python python_backend/download_datasets.py --2a-only")
        return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_bci_iv_2b() -> bool:
    """Test BCI Competition IV 2b (GDF e.g. B01E.gdf in data/.../004-2014)."""
    print("\n--- BCI Competition IV 2b (GDF) ---")
    try:
        from python_backend.train_model import load_bci_iv_2b

        raw, events, event_id = load_bci_iv_2b(1)
        n_trials = len(events)
        print(f"  OK: subject 1 -> {len(raw.ch_names)} ch, sfreq={raw.info['sfreq']} Hz, {n_trials} trials (event_id: {event_id})")
        return True
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        print("  Run: python python_backend/download_datasets.py")
        return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main() -> None:
    data_dir = get_data_dir()
    print(f"Data directory: {data_dir}")
    print(f"MNE_DATA (fallback): {get_mne_data_path()}")
    results = []
    results.append(("PhysioNet EEGBCI", test_physionet_eegbci()))
    results.append(("BCI IV 2a (GDF)", test_bci_iv_2a()))
    results.append(("BCI IV 2b (GDF)", test_bci_iv_2b()))
    print("\n" + "=" * 50)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {name}: {'OK' if ok else 'SKIP/FAIL'}")
    print(f"Summary: {passed}/{len(results)} datasets loaded successfully.")


if __name__ == "__main__":
    main()
