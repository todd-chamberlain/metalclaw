"""Tests for models module."""

from pathlib import Path
from unittest.mock import patch

from metalclaw.models import BUILTIN_MODELS, list_available, get_model_path


def test_builtin_models_have_required_fields():
    required = {"name", "url", "filename", "size_gb", "min_memory_gb",
                "context_window", "description", "expected_sha256"}
    for key, info in BUILTIN_MODELS.items():
        missing = required - set(info.keys())
        assert not missing, f"Model '{key}' missing fields: {missing}"


def test_list_available_includes_all_builtins():
    available = list_available()
    keys = {m["key"] for m in available}
    assert keys == set(BUILTIN_MODELS.keys())


def test_list_available_marks_download_status():
    with patch("metalclaw.models._load_registry", return_value={}):
        available = list_available()
    for m in available:
        assert m["downloaded"] is False


def test_get_model_path_returns_none_for_unknown():
    with patch("metalclaw.models._load_registry", return_value={}):
        assert get_model_path("nonexistent") is None


def test_get_model_path_returns_path_for_gguf_in_models_dir(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    gguf = models_dir / "test.gguf"
    gguf.write_bytes(b"fake")
    with patch("metalclaw.models.MODELS_DIR", models_dir):
        assert get_model_path(str(gguf)) == gguf.resolve()


def test_get_model_path_rejects_outside_models_dir(tmp_path):
    gguf = tmp_path / "evil.gguf"
    gguf.write_bytes(b"fake")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with patch("metalclaw.models.MODELS_DIR", models_dir):
        assert get_model_path(str(gguf)) is None


def test_get_model_path_rejects_symlinks(tmp_path):
    """Symlinks should be rejected to prevent traversal attacks."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    real_file = tmp_path / "outside.gguf"
    real_file.write_bytes(b"fake")
    link = models_dir / "linked.gguf"
    link.symlink_to(real_file)
    with patch("metalclaw.models.MODELS_DIR", models_dir):
        assert get_model_path(str(link)) is None


def test_all_models_use_q4_k_m():
    """Verify all builtin models use q4_k_m quantization for consistency."""
    for key, info in BUILTIN_MODELS.items():
        fname = info["filename"].lower()
        assert "q4_k_m" in fname, f"Model '{key}' doesn't use q4_k_m: {info['filename']}"


def test_all_models_have_sha256():
    """All builtin models must have a non-empty expected_sha256."""
    for key, info in BUILTIN_MODELS.items():
        sha = info.get("expected_sha256", "")
        assert sha, f"Model '{key}' has no expected_sha256"
        assert len(sha) == 64, f"Model '{key}' sha256 is not 64 chars: {len(sha)}"


def test_registry_handles_corrupt_json(tmp_path):
    """Corrupted registry file should return empty dict, not crash."""
    reg_path = tmp_path / "registry.json"
    reg_path.write_text("not valid json {{{")
    with patch("metalclaw.models.REGISTRY_PATH", reg_path):
        from metalclaw.models import _load_registry
        assert _load_registry() == {}
