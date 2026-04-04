"""YAML configuration loading with three-layer merge strategy.

Merge order (later overrides earlier):
    1. ``configs/default.yaml``          — global defaults
    2. ``configs/datasets/<name>.yaml``  — dataset-specific overrides
    3. experiment config file             — experiment-specific overrides
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import get_configs_dir, ensure_dir


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns a new dict)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_config(experiment_path: str | Path) -> dict[str, Any]:
    """Load an experiment config with three-layer merge.

    Parameters
    ----------
    experiment_path : str | Path
        Path to an experiment YAML file.  Can be absolute or relative to the
        project root.

    Returns
    -------
    dict
        The fully merged configuration dictionary.
    """
    configs_dir = get_configs_dir()
    experiment_path = Path(experiment_path)

    # --- Layer 1: default.yaml ---
    default_path = configs_dir / "default.yaml"
    config = _load_yaml(default_path) if default_path.exists() else {}

    # --- Layer 2: dataset-specific config ---
    # Look up dataset.name in the experiment file to find the right overlay
    exp_raw = _load_yaml(experiment_path)
    dataset_name = (exp_raw.get("dataset") or {}).get("name")
    if dataset_name:
        dataset_path = configs_dir / "datasets" / f"{dataset_name}.yaml"
        if dataset_path.exists():
            config = _deep_merge(config, _load_yaml(dataset_path))

    # --- Layer 3: experiment overrides ---
    config = _deep_merge(config, exp_raw)

    return config


def save_config_snapshot(config: dict, output_dir: str | Path) -> Path:
    """Dump the fully-merged config to *output_dir*/config_snapshot.yaml.

    Returns the path of the written file.
    """
    output_dir = ensure_dir(Path(output_dir))
    out_path = output_dir / "config_snapshot.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return out_path
