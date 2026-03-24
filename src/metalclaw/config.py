"""Load/save ~/.metalclaw/config.yaml with sensible defaults for Apple Silicon."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

METALCLAW_HOME = Path(os.environ.get("METALCLAW_HOME", Path.home() / ".metalclaw"))
CONFIG_PATH = METALCLAW_HOME / "config.yaml"
MODELS_DIR = METALCLAW_HOME / "models"
STATE_DIR = METALCLAW_HOME / "state"

DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "sandbox": {
        "name": "metalclaw-sandbox",
        "memory_limit": "64g",
        "cpus": 8,
    },
    "gpu": {
        "backend": "vulkan",
        "layers": -1,
    },
    "inference": {
        "model": "qwen2.5-7b",
        "port": 8080,
        "context_size": 8192,
    },
    "agent": {
        "type": "none",
        "command": "",
    },
    "policy": {
        "base": "default",
        "presets": [],
    },
    "machine": {
        "provider": "libkrun",
        "cpus": 8,
        "memory": 61440,  # krunkit max is 61440 MiB (60 GB)
        "disk": 100,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, preferring override values."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def ensure_dirs() -> None:
    """Create metalclaw home directories if they don't exist."""
    for d in (METALCLAW_HOME, MODELS_DIR, STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    """Load config from disk, merged with defaults."""
    ensure_dirs()
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                user_cfg = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError):
            return DEFAULT_CONFIG.copy()
        if not isinstance(user_cfg, dict):
            return DEFAULT_CONFIG.copy()
        return _deep_merge(DEFAULT_CONFIG, user_cfg)
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict[str, Any]) -> None:
    """Write config to disk."""
    ensure_dirs()
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def get(key: str, cfg: dict[str, Any] | None = None) -> Any:
    """Dot-notation access: get('inference.port')."""
    if cfg is None:
        cfg = load_config()
    parts = key.split(".")
    val: Any = cfg
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return None
    return val
