"""Tests for metal module (host-side Metal inference server)."""

from pathlib import Path
from unittest.mock import patch

from metalclaw.metal import (
    LLAMA_CPP_DIR,
    LLAMA_SERVER_BIN,
    PID_FILE,
    server_running,
)


def test_server_not_running_when_no_pid_file(tmp_path):
    with patch("metalclaw.metal.PID_FILE", tmp_path / "nonexistent.pid"):
        assert server_running() is False


def test_server_not_running_when_pid_file_stale(tmp_path):
    pid_file = tmp_path / "server.pid"
    pid_file.write_text("999999999")  # PID that doesn't exist
    with patch("metalclaw.metal.PID_FILE", pid_file):
        assert server_running() is False


def test_paths_under_metalclaw_home():
    """All metal paths should be under METALCLAW_HOME."""
    from metalclaw.config import METALCLAW_HOME
    assert str(LLAMA_CPP_DIR).startswith(str(METALCLAW_HOME))
    assert str(LLAMA_SERVER_BIN).startswith(str(METALCLAW_HOME))
    assert str(PID_FILE).startswith(str(METALCLAW_HOME))
