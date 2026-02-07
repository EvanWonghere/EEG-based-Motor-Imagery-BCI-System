# EEG-Based Motor Imagery BCI System

> **Undergraduate Graduation Project (Computer Science)**
> Focus: Offline Algorithm Implementation & Interactive Simulation via LSL.

## 📖 Project Overview

This project implements a Brain-Computer Interface (BCI) system based on Motor Imagery (MI). Instead of using live hardware, this project utilizes a **"Data Replay + Simulation"** architecture.

1. **Python Backend**: Handles data loading (BCI Competition IV 2a), preprocessing, CSP feature extraction, model training, and LSL streaming (simulation).
2. **Unity Frontend**: Acts as the LSL receiver, visualizing the classification results (e.g., Left/Right Hand imagery) in a 3D environment.

## 🛠 Tech Stack

### 1. Python Backend (Analysis & Streaming)

* **Python**: 3.8+
* **Libraries**:
  * `mne`: EEG data handling (GDF/FIF), filtering, epoching.
  * `scikit-learn`: CSP (Common Spatial Patterns), LDA/SVM classifiers.
  * `numpy`/`scipy`: Signal processing.
  * `pylsl`: Lab Streaming Layer protocol for simulating real-time streams.

### 2. Unity Frontend (Visualization)

* **Engine**: Unity 2021.3+ (LTS)
* **Language**: C#
* **Plugins**: `LSL4Unity` (for receiving data streams).

## 📂 Directory Structure

```text
Project_Root/
├── data/                   # BCI Competition IV 2a Dataset (.gdf)
├── models/                 # Saved models (.joblib) and replay_data.npz
├── python_backend/         # Python Source Code
│   ├── preprocessing.py    # Filtering, Artifact Removal
│   ├── training.py         # CSP + Classifier Training
│   ├── replay_stream.py    # LSL Replay Script (Core logic for simulation)
│   ├── train_model.py      # Training entry point
│   ├── download_datasets.py # CLI: download BCI IV 2a/2b / PhysioNet EEGBCI
│   ├── test_datasets.py    # Test that downloaded datasets load correctly
│   ├── datasets.py         # Dataset download helpers (MNE_DATA, MOABB)
│   ├── utils.py            # Utilities
│   └── archive/            # Legacy scripts (prototype, test_*)
├── unity_frontend/         # Unity Project
│   └── Assets/
│       ├── Scripts/        # C# Scripts (LSLReceiver.cs, GameController.cs)
│       └── Scenes/         # Visualization Scenes
├── tutorials/              # Optional learning scripts
├── environment.yml        # Conda env thesis (pip deps from requirements.txt)
├── .env.example           # Example env vars (copy to .env, do not commit .env)
├── requirements.txt
└── README.md
```

## 🔄 Workflow Pipeline

1. **Offline Training**:
   * Load `.gdf` data -> Bandpass Filter (8-30Hz) -> Epoching (Event IDs: 769, 770).
   * Fit CSP to extract spatial features.
   * Train LDA classifier and evaluate accuracy.
   * Save the CSP filters and LDA model.

2. **Online Simulation (Pseudo-Online)**:
   * **Sender (Python)**: Loads test data and "replays" it via `pylsl` to mimic a real-time stream.
   * **Receiver (Unity)**: Listens for the LSL stream.
   * **Feedback**: Unity executes actions based on received markers (e.g., "Left" -> Move Object Left).

## ⚠️ Note to AI Assistant (Cursor)

* **No Hardware**: This project does **NOT** involve drivers for physical EEG devices (e.g., OpenBCI). All "real-time" aspects are simulated via data replay.
* **Dataset**: Strictly follows the **BCI Competition IV 2a** format. Focus is on **Left Hand (769)** vs **Right Hand (770)** classification.
* **LSL Configuration**: Python acts as the `StreamOutlet`; Unity acts as the `StreamInlet`.

## 🔧 Environment Variables

Copy `.env.example` to `.env` and set paths as needed (e.g. where MNE should download/store datasets):

```bash
cp .env.example .env
# Edit .env: set MNE_DATA=/path/to/your/mne_data (optional; default is ~/mne_data)
```

Main variables:

* **MNE_DATA** – Root directory for MNE datasets (PhysioNet EEGBCI, sample data, etc.). If unset, MNE uses `~/mne_data`.
* Optional dataset-specific vars (see [MNE config](https://mne.tools/stable/overview/configuration.html)): e.g. `MNE_DATASETS_SAMPLE_PATH`.

Scripts that use MNE or project data load `.env` via `python-dotenv` when run from the project root. Do not commit `.env` (it is in `.gitignore`).

**Git**: Use [Conventional Commits](https://www.conventionalcommits.org/) (e.g. `feat(scope): description`, `docs: ...`, `fix: ...`). See `.cursorrules` (Rule 6) for details.

## 🚀 Quick Start

1. Create and activate the **thesis** conda env: `conda env create -f environment.yml` then `conda activate thesis`. (If the env already exists: `conda activate thesis && conda env update -f environment.yml --prune`.)
2. Copy `.env.example` to `.env` and set `MNE_DATA` (or leave default).
3. In Cursor/VS Code: **Python: Select Interpreter** (Ctrl+Shift+P) and choose the interpreter for conda env `thesis`. New terminals will then auto-activate thesis when you run Python files. (If you use Anaconda instead of Miniconda, edit `.vscode/settings.json` and change `miniconda3` to `anaconda3` in `python.defaultInterpreterPath`.)
4. **(Optional)** Download datasets: `python python_backend/download_datasets.py` (BCI IV 2a+2b to `MNE_DATA`). Add `--2a-only` to skip 2b; `--physionet-eegbci` to also download PhysioNet EEG Motor Movement/Imagery; `--physionet-eegbci-only` for that dataset only; `--path /custom/path` to override.
5. Train the model: `python python_backend/train_model.py`
6. Open Unity Project and play the `MainScene`.
7. Start simulation: `python python_backend/replay_stream.py`
