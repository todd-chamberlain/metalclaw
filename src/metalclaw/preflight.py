"""Preflight checks: verify podman, krunkit, disk space, port availability."""

from __future__ import annotations

import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from metalclaw.gpu import detect_apple_silicon

console = Console()


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class PreflightReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def _get_version(cmd: str) -> str | None:
    path = shutil.which(cmd)
    if not path:
        return None
    try:
        out = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip().split("\n")[0]
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _parse_podman_version(version_str: str) -> tuple[int, ...]:
    """Extract version tuple from 'podman version X.Y.Z' string."""
    for part in version_str.split():
        if part and part[0].isdigit():
            try:
                return tuple(int(x) for x in part.split("."))
            except ValueError:
                continue
    return (0,)


def check_podman() -> CheckResult:
    """Check podman >= 5.0 is installed."""
    ver = _get_version("podman")
    if not ver:
        return CheckResult("podman", False, "Not found. Install: brew install podman")
    parsed = _parse_podman_version(ver)
    if parsed < (5, 0):
        return CheckResult(
            "podman", False, f"Found {ver}, need >= 5.0. Run: brew upgrade podman"
        )
    return CheckResult("podman", True, ver)


def check_krunkit() -> CheckResult:
    """Check krunkit is installed (required for libkrun GPU passthrough)."""
    path = shutil.which("krunkit")
    if not path:
        return CheckResult(
            "krunkit", False, "Not found. Install: brew install krunkit"
        )
    return CheckResult("krunkit", True, f"Found at {path}")


def check_apple_silicon() -> CheckResult:
    """Check we're on Apple Silicon."""
    if detect_apple_silicon():
        return CheckResult("apple_silicon", True, "ARM64 detected")
    return CheckResult("apple_silicon", False, "Requires Apple Silicon (M-series)")


def check_disk_space(min_gb: int = 20) -> CheckResult:
    """Check available disk space."""
    home = Path.home()
    try:
        import os

        stat = os.statvfs(str(home))
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        if free_gb < min_gb:
            return CheckResult(
                "disk_space",
                False,
                f"{free_gb:.1f} GB free, need >= {min_gb} GB",
            )
        return CheckResult("disk_space", True, f"{free_gb:.1f} GB free")
    except OSError as e:
        return CheckResult("disk_space", False, f"Could not check: {e}")


def check_port(port: int = 8080) -> CheckResult:
    """Check if the inference port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(("127.0.0.1", port))
            if result == 0:
                return CheckResult(
                    "port", False, f"Port {port} is already in use"
                )
            return CheckResult("port", True, f"Port {port} is available")
    except OSError:
        return CheckResult("port", True, f"Port {port} appears available")


def check_cmake() -> CheckResult:
    """Check cmake is installed (required for building Metal inference server)."""
    path = shutil.which("cmake")
    if not path:
        return CheckResult(
            "cmake", False, "Not found. Install: brew install cmake"
        )
    ver = _get_version("cmake")
    return CheckResult("cmake", True, ver or f"Found at {path}")


def run_preflight(port: int = 8080) -> PreflightReport:
    """Run all preflight checks and return report."""
    report = PreflightReport()
    report.checks.append(check_apple_silicon())
    report.checks.append(check_podman())
    report.checks.append(check_krunkit())
    report.checks.append(check_cmake())
    report.checks.append(check_disk_space())
    report.checks.append(check_port(port))
    return report


def print_report(report: PreflightReport) -> None:
    """Print preflight report to console."""
    for check in report.checks:
        icon = "[green]\u2713[/green]" if check.passed else "[red]\u2717[/red]"
        console.print(f"  {icon} {check.name}: {check.detail}")
