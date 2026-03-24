"""Podman machine lifecycle management with libkrun GPU passthrough."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

from rich.console import Console

from metaclaw.config import load_config

console = Console()

MACHINE_NAME = "metaclaw"

# Force libkrun provider for GPU passthrough on all podman machine commands
_LIBKRUN_ENV = {**os.environ, "CONTAINERS_MACHINE_PROVIDER": "libkrun"}


@dataclass
class MachineStatus:
    exists: bool
    running: bool
    provider: str
    cpus: int
    memory_mb: int
    disk_gb: int


def _podman(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["podman", "machine", *args],
            capture_output=True,
            text=True,
            timeout=300,
            check=check,
            env=_LIBKRUN_ENV,
        )
    except FileNotFoundError:
        raise RuntimeError("podman not found. Install: brew install podman")
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("podman command timed out") from e


def get_status() -> MachineStatus:
    """Get current machine status."""
    try:
        result = _podman("inspect", MACHINE_NAME, check=False)
        if result.returncode != 0:
            return MachineStatus(False, False, "", 0, 0, 0)
        data = json.loads(result.stdout)
        # podman machine inspect returns a list
        info = data[0] if isinstance(data, list) and data else data
        if not isinstance(info, dict):
            return MachineStatus(False, False, "", 0, 0, 0)
        return MachineStatus(
            exists=True,
            running=info.get("State", "") == "running",
            provider=info.get("VMType", ""),
            cpus=info.get("CPUs", 0),
            memory_mb=info.get("Memory", 0),
            disk_gb=info.get("DiskSize", 0),
        )
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError,
            AttributeError, RuntimeError):
        return MachineStatus(False, False, "", 0, 0, 0)


def init_machine(cpus: int | None = None, memory_mb: int | None = None,
                 disk_gb: int | None = None) -> bool:
    """Initialize podman machine with libkrun provider for GPU support."""
    cfg = load_config()
    cpus = cpus or cfg["machine"]["cpus"]
    memory_mb = memory_mb or cfg["machine"]["memory"]
    disk_gb = disk_gb or cfg["machine"]["disk"]

    status = get_status()
    if status.exists:
        if status.provider and status.provider != "libkrun":
            console.print(
                f"[yellow]Machine '{MACHINE_NAME}' exists but uses provider "
                f"'{status.provider}' instead of libkrun.[/yellow]"
            )
            console.print(
                "[yellow]Remove it with: podman machine rm metaclaw[/yellow]"
            )
            return False
        console.print(f"[green]Machine '{MACHINE_NAME}' already exists[/green]")
        return True

    console.print("Initializing podman machine with libkrun (GPU passthrough)...")
    try:
        _podman(
            "init",
            MACHINE_NAME,
            f"--cpus={cpus}",
            f"--memory={memory_mb}",
            f"--disk-size={disk_gb}",
        )
        console.print("[green]Machine initialized[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to init machine: {e.stderr}[/red]")
        return False


def start_machine() -> bool:
    """Start the podman machine."""
    status = get_status()
    if not status.exists:
        console.print("[red]Machine not initialized. Run: metaclaw onboard[/red]")
        return False
    if status.running:
        console.print("[green]Machine already running[/green]")
        return True

    console.print("Starting podman machine...")
    try:
        _podman("start", MACHINE_NAME)
        # Set this machine as the default podman connection
        subprocess.run(
            ["podman", "system", "connection", "default", MACHINE_NAME],
            capture_output=True, text=True, timeout=10, check=False,
        )
        console.print("[green]Machine started[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to start machine: {e.stderr}[/red]")
        return False


def stop_machine() -> bool:
    """Stop the podman machine."""
    status = get_status()
    if not status.exists or not status.running:
        return True

    console.print("Stopping podman machine...")
    try:
        _podman("stop", MACHINE_NAME)
        console.print("[green]Machine stopped[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to stop machine: {e.stderr}[/red]")
        return False


def verify_gpu() -> bool:
    """Check Vulkan support inside the VM via vulkaninfo."""
    try:
        result = subprocess.run(
            ["podman", "machine", "ssh", MACHINE_NAME, "--",
             "vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=30, check=False,
            env=_LIBKRUN_ENV,
        )
        if "Venus" in result.stdout or "Virtio" in result.stdout:
            console.print("[green]  GPU passthrough verified (Venus/Virtio)[/green]")
            return True
        console.print("[yellow]  vulkaninfo ran but no Venus driver detected[/yellow]")
        return False
    except subprocess.SubprocessError:
        console.print("[yellow]  Could not verify GPU (vulkaninfo not available)[/yellow]")
        return False
