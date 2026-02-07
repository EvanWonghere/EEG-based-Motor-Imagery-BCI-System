"""
Entry point for training: load data -> preprocess -> train CSP+LDA -> save model.
Supports PhysioNet EEGBCI (.edf) for quick test; use data/ and BCI IV 2a .gdf for thesis.
"""
from pathlib import Path
import sys

# Load .env from project root first (so MNE_DATA etc. are set before importing mne)
ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from python_backend.utils import (
    BAND_LOW_HZ,
    BAND_HIGH_HZ,
    EPOCH_TMIN,
    EPOCH_TMAX,
    EPOCH_TRAIN_TMIN,
    EPOCH_TRAIN_TMAX,
    get_models_dir,
)
from python_backend.preprocessing import bandpass_filter, get_epochs, epochs_to_arrays
from python_backend.training import run_training


def load_physionet_eegbci(subject: int = 1, runs: list = None) -> tuple:
    """Load PhysioNet EEGBCI data (hands vs feet). Returns (raw, events, event_id)."""
    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    if runs is None:
        runs = [6, 10, 14]
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.annotations.rename(dict(T1="hands", T2="feet"))
    events, event_id = mne.events_from_annotations(raw)
    target_id = {k: v for k, v in event_id.items() if k in ("hands", "feet")}
    return raw, events, target_id


def main() -> None:
    print("Loading data (PhysioNet EEGBCI subject 1, runs 6/10/14)...")
    raw, events, event_id = load_physionet_eegbci()
    print(f"Event IDs: {event_id}")

    print("Filtering 8-30 Hz...")
    bandpass_filter(raw, l_freq=BAND_LOW_HZ, h_freq=BAND_HIGH_HZ)

    print("Epoching...")
    epochs = get_epochs(raw, events, event_id, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX)
    X, y = epochs_to_arrays(
        epochs, tmin=EPOCH_TRAIN_TMIN, tmax=EPOCH_TRAIN_TMAX
    )
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    print("Training CSP + LDA...")
    run_training(X, y, model_name="csp_lda")

    # Save a small replay set for replay_stream.py (optional)
    replay_path = get_models_dir() / "replay_data.npz"
    np.savez_compressed(replay_path, X=X, y=y)
    print(f"Replay data saved to {replay_path}. You can run replay_stream.py next.")


if __name__ == "__main__":
    main()
