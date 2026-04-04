"""Project path management.

All path helpers resolve relative to the project root so that scripts work
regardless of the caller's working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at import time (no-op if missing)
load_dotenv()


def get_project_root() -> Path:
    """Return the project root directory (contains ``src/``)."""
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    """Return the data directory, respecting the ``DATA_DIR`` env var."""
    override = os.getenv("DATA_DIR")
    if override:
        return Path(override).resolve()
    return get_project_root() / "data"


def get_models_dir() -> Path:
    """Return the models directory, respecting the ``MODELS_DIR`` env var."""
    override = os.getenv("MODELS_DIR")
    if override:
        return Path(override).resolve()
    return get_project_root() / "results" / "models"


def get_results_dir() -> Path:
    """Return the top-level results directory."""
    return get_project_root() / "results"


def get_figures_dir() -> Path:
    """Return the figures output directory."""
    return get_results_dir() / "figures"


def get_logs_dir() -> Path:
    """Return the logs directory."""
    return get_results_dir() / "logs"


def get_configs_dir() -> Path:
    """Return the configs directory."""
    return get_project_root() / "configs"


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it doesn't exist, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
