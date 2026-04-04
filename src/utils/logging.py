"""Unified logging setup for the project.

Usage::

    from src.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Training subject %d", subject_id)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """Configure the root ``src`` logger.

    Call once at program startup (e.g. in a ``scripts/*.py`` entry point).
    Subsequent calls are no-ops.

    Parameters
    ----------
    level : int
        Logging level (default ``INFO``).
    log_file : str | Path | None
        If given, also write log messages to this file.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    root_logger = logging.getLogger("src")
    root_logger.setLevel(level)
    for handler in handlers:
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``src`` namespace."""
    return logging.getLogger(name)
