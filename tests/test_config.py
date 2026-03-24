"""Tests for config module."""

from unittest.mock import patch

from metalclaw.config import (
    DEFAULT_CONFIG,
    _deep_merge,
    get,
    load_config,
    save_config,
)


def test_deep_merge_basic():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": {"c": 99, "d": 3}}


def test_deep_merge_adds_new_keys():
    base = {"a": 1}
    override = {"b": 2}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": 2}


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1}}
    override = {"a": {"b": 2}}
    _deep_merge(base, override)
    assert base["a"]["b"] == 1


def test_load_config_returns_defaults(tmp_path):
    with patch("metalclaw.config.METALCLAW_HOME", tmp_path), \
         patch("metalclaw.config.CONFIG_PATH", tmp_path / "config.yaml"), \
         patch("metalclaw.config.MODELS_DIR", tmp_path / "models"), \
         patch("metalclaw.config.STATE_DIR", tmp_path / "state"):
        cfg = load_config()
    assert cfg["version"] == 1
    assert cfg["gpu"]["backend"] == "vulkan"


def test_save_and_load_roundtrip(tmp_path):
    config_path = tmp_path / "config.yaml"
    with patch("metalclaw.config.METALCLAW_HOME", tmp_path):
        with patch("metalclaw.config.CONFIG_PATH", config_path):
            with patch("metalclaw.config.MODELS_DIR", tmp_path / "models"):
                with patch("metalclaw.config.STATE_DIR", tmp_path / "state"):
                    cfg = DEFAULT_CONFIG.copy()
                    cfg["inference"]["model"] = "test-model"
                    save_config(cfg)
                    loaded = load_config()
    assert loaded["inference"]["model"] == "test-model"


def test_get_dot_notation():
    cfg = {"a": {"b": {"c": 42}}}
    assert get("a.b.c", cfg) == 42
    assert get("a.b", cfg) == {"c": 42}
    assert get("x.y.z", cfg) is None
