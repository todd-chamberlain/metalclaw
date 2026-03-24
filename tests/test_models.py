"""Tests for models module."""

from pathlib import Path
from unittest.mock import patch

from metaclaw.models import BUILTIN_MODELS, list_available, get_model_path


def test_builtin_models_have_required_fields():
    required = {"name", "url", "filename", "size_gb", "min_memory_gb",
                "context_window", "description"}
    for key, info in BUILTIN_MODELS.items():
        missing = required - set(info.keys())
        assert not missing, f"Model '{key}' missing fields: {missing}"


def test_list_available_includes_all_builtins():
    available = list_available()
    keys = {m["key"] for m in available}
    assert keys == set(BUILTIN_MODELS.keys())


def test_list_available_marks_download_status():
    with patch("metaclaw.models._load_registry", return_value={}):
        available = list_available()
    for m in available:
        assert m["downloaded"] is False


def test_get_model_path_returns_none_for_unknown():
    with patch("metaclaw.models._load_registry", return_value={}):
        assert get_model_path("nonexistent") is None


def test_get_model_path_returns_path_for_gguf_in_models_dir(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    gguf = models_dir / "test.gguf"
    gguf.write_bytes(b"fake")
    with patch("metaclaw.models.MODELS_DIR", models_dir):
        assert get_model_path(str(gguf)) == gguf.resolve()


def test_get_model_path_rejects_outside_models_dir(tmp_path):
    gguf = tmp_path / "evil.gguf"
    gguf.write_bytes(b"fake")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with patch("metaclaw.models.MODELS_DIR", models_dir):
        assert get_model_path(str(gguf)) is None


def test_all_models_use_q4_k_m():
    """Verify all builtin models use q4_k_m quantization for consistency."""
    for key, info in BUILTIN_MODELS.items():
        fname = info["filename"].lower()
        assert "q4_k_m" in fname, f"Model '{key}' doesn't use q4_k_m: {info['filename']}"
