# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG-based Motor Imagery Brain-Computer Interface (BCI) system for an undergraduate thesis. Binary classification of Left Hand (event 769) vs Right Hand (event 770) using BCI Competition IV 2a dataset.

Three classification pipelines: CSP+LDA (baseline), FBCSP+SVM, EEGNet (deep learning). YAML-driven config system with three-layer merge: `configs/default.yaml` → `configs/datasets/*.yaml` → `configs/experiments/*.yaml`.

Architecture: Python ML backend with offline training/evaluation + WebSocket-based online simulation frontend (browser). Unity 3D frontend planned for Phase 6.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate thesis
# Update: conda env update -f environment.yml --prune
```

Copy `.env.example` to `.env` and set `MNE_DATA` path if needed (defaults to `~/mne_data`).

## Common Commands

```bash
# Download BCI Competition IV 2a dataset
python scripts/download_data.py

# Train a single experiment (uses YAML config)
python scripts/train.py --config configs/experiments/fbcsp_svm.yaml
python scripts/train.py --config configs/experiments/fbcsp_svm.yaml --subject 1

# Multi-method comparison (evaluate all results)
python scripts/evaluate.py results/ --latex

# Generate thesis figures from results
python scripts/analyze.py results/ figures/

# Run online BCI simulation (WebSocket backend + browser frontend)
python scripts/run_online.py --model results/fbcsp_svm_2a/models/fbcsp_svm_2a_sub1.pkl

# Run tests
python -m pytest tests/ -v
```

## Architecture

### Source Layout

```
src/
  data/          # Dataset loading (BCI IV 2a via MNE)
  preprocessing/ # Bandpass filtering, epoching, ICA artifact removal
  features/      # CSP, FBCSP feature extractors
  models/        # BaseModel, CSP+LDA, FBCSP+SVM, EEGNet classifiers
  evaluation/    # Metrics, cross-validation, statistical tests, comparison tables
  visualization/ # Result plots, confusion matrices, ROC curves, topomaps, ERD/ERS
  online/        # WebSocket server, online classifier, replay stream
  utils/         # Config loading, logging
scripts/         # CLI entry points (train, evaluate, analyze, run_online, download_data)
configs/         # YAML experiment configs (three-layer merge)
web_frontend/    # Browser-based BCI simulation UI (HTML + JS + CSS)
tests/           # pytest test suite
```

### ML Pipeline

Raw EEG (.gdf) → Bandpass filter (8-30 Hz) → ICA artifact removal (optional) → Epoching → Feature extraction (CSP/FBCSP or raw for EEGNet) → Classifier → 10-fold stratified CV with full metrics

### Evaluation

- **Metrics**: accuracy, Cohen's kappa, F1-weighted, ROC-AUC, confusion matrix
- **Statistics**: Wilcoxon signed-rank, paired permutation test, Friedman + Nemenyi post-hoc
- **Outputs**: per-subject `cv_sub*.json` with fold-level predictions, `results.json` aggregate, LaTeX comparison table

### Online Simulation

Python WebSocket backend (`src/online/server.py`) replays saved trial data through trained model, pushes state updates (cue → imagine → classify → feedback → rest) to browser via WebSocket (port 8765). Separate HTTP server (port 8080) serves the web frontend.

### Key Constants

- Frequency band: 8-30 Hz (Mu 8-13, Beta 13-30)
- Event IDs: 769 (Left), 770 (Right), 771 (Feet), 772 (Tongue)
- Epoch window: -0.5s to 3.0s; training crop: 0.5s to 2.5s
- Sampling rate: 250 Hz (BCI IV 2a)

## Conventions

- **Git commits**: Conventional Commits format `<type>(<scope>): <description>` — types: feat, fix, docs, style, refactor, test, chore
- **Python style**: PEP 8, type hints throughout
- **Language**: English in source code comments; 简体中文 for user-facing explanations and docs
- **Never commit**: `.env`, `data/MNE-*`, `results/`, `*.pkl`, `*.npz`

## Key Dependencies

- **mne**: EEG data loading, filtering, epoching, ICA
- **scikit-learn**: CSP, LDA/SVM, cross-validation utilities
- **torch**: EEGNet deep learning model
- **websockets** (>=14.0,<16.0): Online simulation WebSocket server
- **matplotlib**: Thesis figure generation
- **joblib**: Model/extractor serialization
- **numpy/scipy**: Signal processing, statistics
