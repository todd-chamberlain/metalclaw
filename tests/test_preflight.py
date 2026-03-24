"""Tests for preflight checks."""

from unittest.mock import patch

from metaclaw.preflight import (
    CheckResult,
    PreflightReport,
    _parse_podman_version,
    check_port,
)


def test_preflight_report_all_passed():
    report = PreflightReport(checks=[
        CheckResult("a", True, "ok"),
        CheckResult("b", True, "ok"),
    ])
    assert report.all_passed is True
    assert report.failed == []


def test_preflight_report_with_failure():
    report = PreflightReport(checks=[
        CheckResult("a", True, "ok"),
        CheckResult("b", False, "bad"),
    ])
    assert report.all_passed is False
    assert len(report.failed) == 1
    assert report.failed[0].name == "b"


def test_parse_podman_version():
    assert _parse_podman_version("podman version 5.2.1") == (5, 2, 1)
    assert _parse_podman_version("podman version 4.9.0") == (4, 9, 0)
    assert _parse_podman_version("garbage") == (0,)


def test_check_port_available():
    # Use a high port unlikely to be in use
    result = check_port(59123)
    assert result.passed is True


def test_check_port_in_use():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.listen(1)
        # Check while socket is still bound
        result = check_port(port)
        assert result.passed is False
