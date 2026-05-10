"""
config/loader.py
----------------
Loads and validates the YAML configuration file.
Provides a singleton Config object accessible across the project.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the config file. Defaults to config/config.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {path}")
    return cfg


def get(key_path: str, config_path: str | Path | None = None, default: Any = None) -> Any:
    """
    Retrieve a nested config value using dot notation.

    Example:
        get("camera.width") -> 1280
        get("cameras")      -> list of camera dicts

    Args:
        key_path : Dot-separated key path.
        config_path: Optional override for config file path.
        default: Value returned if key is missing.
    """
    cfg = load_config(config_path)
    keys = key_path.split(".")
    value = cfg
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value
