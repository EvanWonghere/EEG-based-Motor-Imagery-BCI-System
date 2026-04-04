"""Reproducibility helpers: seed fixing and environment snapshots."""

from __future__ import annotations

import platform
import random
import sys
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy, and (if available) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_environment_snapshot() -> dict[str, Any]:
    """Capture versions of key packages and platform info.

    Useful for logging alongside experiment results so that results can be
    traced back to the exact environment.
    """
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }

    for pkg_name in ("numpy", "scipy", "sklearn", "mne", "torch", "pylsl", "yaml"):
        try:
            mod = __import__(pkg_name)
            snapshot[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            snapshot[pkg_name] = None

    return snapshot
