# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG-based Motor Imagery Brain-Computer Interface (BCI) system for an undergraduate thesis. Binary classification of Left Hand (event 769) vs Right Hand (event 770) using BCI Competition IV 2a dataset. Architecture: Python ML backend → LSL streaming → Unity 3D visualization frontend.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate thesis
# Update: conda env update -f environment.yml --prune
```

Copy `.env.example` to `.env` and set `MNE_DATA` path if needed (defaults to `~/mne_data`).

## Common Commands

```bash
# Download datasets
python python_backend/download_datasets.py              # BCI IV 2a+2b
python python_backend/download_datasets.py --2a-only     # Only 2a
python python_backend/download_datasets.py --physionet-eegbci  # PhysioNet

# Train models
python python_backend/train_model.py --dataset 2a                    # Single dataset
python python_backend/train_model.py --dataset 2a --subjects 1-9     # All subjects
python python_backend/train_model.py --dataset all                   # Everything

# Analyze trained models
python python_backend/analyze_model.py

# Run LSL replay simulation
python python_backend/replay_stream.py
python python_backend/replay_stream.py --model best_model_sub1

# Validate dataset loading
python python_backend/test_datasets.py
```

## Architecture

### ML Pipeline

Raw EEG (.gdf/.edf) → Bandpass filter (8-30 Hz) → Epoching (tmin=-0.5s, tmax=3.0s) → CSP feature extraction (4 components, log variance) → Classifier (LDA/SVM/RandomForest) → Trained model (.pkl) + replay data (.npz)

- **preprocessing.py**: MNE bandpass FIR filter, event detection, epoching
- **training.py**: `create_pipeline(model_type)` factory — supports LDA, SVM (RBF+StandardScaler), RandomForest
- **train_model.py**: Entry point, multi-dataset/multi-subject training, 10-fold stratified CV
- **replay_stream.py**: Loads model + replay_data.npz, streams predictions over LSL (BCI_Control/Markers outlets)
- **analyze_model.py**: Pipeline introspection, CSP parameter visualization, generalization reports
- **utils.py**: Constants (BAND_LOW_HZ=8, BAND_HIGH_HZ=30, event IDs) and path helpers

### LSL Integration

Python backend sends classification results via `pylsl` StreamOutlet → Unity frontend receives via LSL4Unity StreamInlet in Update() polling loop → drives game logic for left/right hand feedback.

### Key Constants

- Frequency band: 8–30 Hz (Mu 8–13, Beta 13–30)
- Event IDs: 769 (Left), 770 (Right), 771 (Feet), 772 (Tongue)
- Epoch window: -0.5s to 3.0s; training crop: 0.5s to 2.5s
- Sampling rate: 250 Hz (BCI IV 2a/2b)

## Conventions

- **Git commits**: Conventional Commits format `<type>(<scope>): <description>` — types: feat, fix, docs, style, refactor, test, chore; scopes: python_backend, tutorials, unity, etc.
- **Python style**: PEP 8, type hints throughout
- **Language**: English in source code comments; 简体中文 for user-facing explanations and docs
- **Model naming**: `csp_{type}_sub{id}[_{dataset_suffix}].pkl` (e.g., `csp_lda_sub1_2a.pkl`)
- **Never commit**: `.env`, `data/MNE-*`, `models/*.pkl`, `models/analysis/`

## Key Dependencies

- **mne**: EEG data loading, filtering, epoching
- **scikit-learn**: CSP, LDA/SVM/RandomForest, cross-validation
- **pylsl**: Lab Streaming Layer streaming
- **joblib**: Model serialization
- **numpy/scipy**: Signal processing
- **Unity 6000+** with LSL4Unity plugin for frontend
