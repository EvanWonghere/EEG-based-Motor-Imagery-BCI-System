# EEG-Based Motor Imagery BCI System

> **Undergraduate thesis project** — Computer Science, Shandong Normal University.

A complete Motor Imagery (MI) Brain-Computer Interface system: offline training & evaluation of three classification pipelines on multiple public datasets, plus a WebSocket-based online simulation with browser frontend.

## Overview

Binary classification of **Left Hand (769)** vs **Right Hand (770)** motor imagery EEG signals. Three pipelines are implemented and compared under a unified evaluation framework:

| Pipeline | Category | Description |
|----------|----------|-------------|
| **CSP + LDA** | Baseline | Common Spatial Patterns + Linear Discriminant Analysis |
| **FBCSP + SVM** | Improved | Filter Bank CSP with mutual-information feature selection + SVM |
| **EEGNet** | Deep Learning | Compact CNN designed for EEG (Lawhern et al., 2018) |

Evaluated on three datasets for generalization:

| Dataset | Subjects | Channels | Sampling Rate |
|---------|----------|----------|---------------|
| BCI Competition IV 2a | 9 | 22 | 250 Hz |
| BCI Competition IV 2b | 9 | 3 (C3/Cz/C4) | 250 Hz |
| PhysioNet EEGBCI | 109 (20 used) | 64 | 160 Hz |

## Project Structure

```text
├── src/                        # Core Python package
│   ├── data/                   #   Dataset loaders (2a, 2b, PhysioNet)
│   ├── preprocessing/          #   Bandpass filter, CAR, ICA, epoching
│   ├── features/               #   CSP, FBCSP feature extractors
│   ├── models/                 #   LDA, SVM, EEGNet classifiers
│   ├── evaluation/             #   Metrics, CV, statistical tests
│   ├── visualization/          #   Figures, topomaps, ERD/ERS
│   ├── online/                 #   WebSocket server, replay stream
│   └── utils/                  #   Config loading, logging, paths
├── scripts/                    # CLI entry points
│   ├── train.py                #   Train & cross-validate a pipeline
│   ├── evaluate.py             #   Multi-method comparison & LaTeX tables
│   ├── analyze.py              #   Generate thesis figures
│   ├── run_online.py           #   Launch online BCI simulation
│   └── download_data.py        #   Download datasets via MNE
├── configs/                    # YAML experiment configs
│   ├── default.yaml            #   Global defaults
│   ├── datasets/               #   Per-dataset overrides
│   └── experiments/            #   Per-experiment overrides
├── web_frontend/               # Browser-based online simulation UI
├── tests/                      # pytest test suite
├── docs/                       # Thesis (LaTeX) and references
├── results/                    # Experiment outputs (git-ignored)
├── python_backend/             # Legacy prototype scripts (archived)
├── environment.yml             # Conda environment definition
└── CLAUDE.md                   # AI assistant project instructions
```

### Configuration System

Three-layer YAML merge: `configs/default.yaml` → `configs/datasets/*.yaml` → `configs/experiments/*.yaml`. Each layer overrides the previous, so experiments only specify what differs from defaults.

## Quick Start

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate thesis

# If environment already exists:
conda env update -f environment.yml --prune
```

Copy `.env.example` to `.env` and set `MNE_DATA` if needed (defaults to `~/mne_data`).

### 2. Download Data

```bash
python scripts/download_data.py                    # BCI IV 2a + 2b
python scripts/download_data.py --physionet-eegbci  # Also download PhysioNet
```

### 3. Train & Evaluate

```bash
# Train a single pipeline (config-driven)
python scripts/train.py --config configs/experiments/fbcsp_svm.yaml

# Train for a specific subject
python scripts/train.py --config configs/experiments/fbcsp_svm.yaml --subject 1

# Compare all methods and generate LaTeX tables
python scripts/evaluate.py results/ --latex

# Generate thesis figures
python scripts/analyze.py results/ figures/
```

### 4. Online Simulation

The online system replays saved trial data through a trained model via WebSocket, providing real-time visual feedback in the browser.

```bash
# Launch WebSocket backend (port 8765) + HTTP server (port 8080)
python scripts/run_online.py --model results/fbcsp_svm_2a/models/fbcsp_svm_2a_sub1.pkl
```

Then open `http://localhost:8080` in a browser. The closed-loop pipeline runs: **Cue → Imagine → Classify → Feedback → Rest**.

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

## Key Technical Details

- **Frequency band**: 8–30 Hz (Mu: 8–13 Hz, Beta: 13–30 Hz)
- **Epoch window**: −0.5 s to 3.0 s; training crop: 0.5 s to 2.5 s
- **Evaluation**: 10-fold stratified cross-validation
- **Metrics**: Accuracy, Cohen's κ, weighted F1, ROC-AUC, confusion matrix
- **Statistics**: Friedman test, Wilcoxon signed-rank, paired permutation test

## Key Dependencies

- **mne** — EEG data loading, filtering, epoching, ICA
- **scikit-learn** — CSP, LDA/SVM, cross-validation
- **torch** — EEGNet deep learning model
- **websockets** — Online simulation WebSocket server
- **matplotlib** — Thesis figure generation

## Conventions

- **Git**: [Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`)
- **Python**: PEP 8, type hints throughout
- **Language**: English in code; 简体中文 in thesis and user-facing docs
- **Never commit**: `.env`, `data/MNE-*`, `results/`, `*.pkl`, `*.npz`

## License

This project is developed for academic purposes as part of an undergraduate thesis.
