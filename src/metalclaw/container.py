"""Container image build and lifecycle management."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from rich.console import Console

from metalclaw.config import load_config
from metalclaw.policy import NetworkPolicy, policy_to_podman_network

console = Console()

IMAGE_NAME = "metalclaw-sandbox"
CONTAINER_DIR = Path(__file__).parent.parent.parent / "container"

# Ensure podman talks to the libkrun machine, not applehv
_LIBKRUN_ENV = {**os.environ, "CONTAINERS_MACHINE_PROVIDER": "libkrun"}


def _podman(*args: str, check: bool = True,
            timeout: int = 600) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["podman", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
            env=_LIBKRUN_ENV,
        )
    except FileNotFoundError:
        raise RuntimeError("podman not found. Install: brew install podman")
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"podman command timed out after {timeout}s") from e


def image_exists() -> bool:
    """Check if the metalclaw sandbox image is built."""
    try:
        result = _podman("image", "exists", IMAGE_NAME, check=False)
        return result.returncode == 0
    except (subprocess.SubprocessError, RuntimeError):
        return False


def build_image() -> bool:
    """Build the sandbox container image from Containerfile."""
    containerfile = CONTAINER_DIR / "Containerfile"
    if not containerfile.exists():
        console.print(f"[red]Containerfile not found at {containerfile}[/red]")
        return False

    console.print("Building container image (this may take several minutes)...")
    try:
        result = _podman(
            "build",
            "-t", IMAGE_NAME,
            "-f", str(containerfile),
            str(CONTAINER_DIR),
            timeout=1800,
            check=False,
        )
        if result.returncode != 0:
            console.print(f"[red]Build failed:\n{result.stderr}[/red]")
            return False
        console.print("[green]Container image built[/green]")
        return True
    except RuntimeError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        return False


def container_exists(name: str | None = None) -> bool:
    """Check if container exists."""
    cfg = load_config()
    name = name or cfg["sandbox"]["name"]
    try:
        result = _podman("container", "exists", name, check=False)
        return result.returncode == 0
    except RuntimeError:
        return False


def container_running(name: str | None = None) -> bool:
    """Check if container is running."""
    cfg = load_config()
    name = name or cfg["sandbox"]["name"]
    try:
        result = _podman(
            "inspect", "--format", "{{.State.Running}}", name, check=False
        )
        return result.stdout.strip() == "true"
    except (subprocess.SubprocessError, RuntimeError):
        return False


def start_container(
    model_path: Path,
    policy: NetworkPolicy | None = None,
    agent_type: str = "none",
    agent_command: str = "",
) -> bool:
    """Create and start the sandbox container."""
    cfg = load_config()
    name = cfg["sandbox"]["name"]
    port = cfg["inference"]["port"]
    ctx_size = cfg["inference"]["context_size"]
    gpu_layers = cfg["gpu"]["layers"]

    # Clean up existing container if present
    if container_running(name):
        _podman("stop", "-t", "10", name, check=False)
    if container_exists(name):
        _podman("rm", name, check=False)

    # Determine network mode from policy
    network = "none"
    if policy:
        network = policy_to_podman_network(policy)

    # Resource limits from config
    mem_limit = cfg["sandbox"]["memory_limit"]
    cpus = cfg["sandbox"]["cpus"]

    # Build run command
    cmd = [
        "run", "-d",
        "--name", name,
        f"--memory={mem_limit}",
        f"--cpus={cpus}",
        "--pids-limit=4096",
        "--device", "/dev/dri",
        "-v", f"{model_path}:/models/model.gguf:ro",
        "-p", f"127.0.0.1:{port}:{port}",
        f"--network={network}",
        "-e", f"MODEL_PATH=/models/model.gguf",
        "-e", f"PORT={port}",
        "-e", f"CTX_SIZE={ctx_size}",
        "-e", f"GPU_LAYERS={gpu_layers}",
        "-e", f"AGENT_TYPE={agent_type}",
    ]

    if agent_command:
        cmd.extend(["-e", f"AGENT_COMMAND={agent_command}"])

    cmd.append(IMAGE_NAME)

    try:
        result = _podman(*cmd, check=False)
        if result.returncode != 0:
            console.print(f"[red]Failed to start container:\n{result.stderr}[/red]")
            return False
        console.print(f"[green]Container '{name}' started[/green]")
        return True
    except (subprocess.SubprocessError, RuntimeError) as e:
        console.print(f"[red]Error starting container: {e}[/red]")
        return False


def stop_container(name: str | None = None) -> bool:
    """Stop and remove the container."""
    cfg = load_config()
    name = name or cfg["sandbox"]["name"]

    if not container_exists(name):
        return True

    try:
        _podman("stop", "-t", "10", name, check=False)
        _podman("rm", name, check=False)
        console.print(f"[green]Container '{name}' stopped and removed[/green]")
        return True
    except (subprocess.SubprocessError, RuntimeError) as e:
        console.print(f"[red]Error stopping container: {e}[/red]")
        return False


def exec_shell(name: str | None = None) -> None:
    """Exec into container with interactive shell (replaces current process)."""
    import os

    cfg = load_config()
    name = name or cfg["sandbox"]["name"]

    if not container_running(name):
        console.print("[red]Container is not running[/red]")
        return

    os.execve(
        "/usr/bin/env",
        ["env", "CONTAINERS_MACHINE_PROVIDER=libkrun",
         "podman", "exec", "-it", name, "/bin/bash"],
        _LIBKRUN_ENV,
    )


def get_logs(name: str | None = None, tail: int = 100) -> str:
    """Get container logs (non-streaming). Use 'metalclaw logs -f' for follow mode."""
    cfg = load_config()
    name = name or cfg["sandbox"]["name"]

    if not container_exists(name):
        return "Container does not exist"

    try:
        result = _podman("logs", "--tail", str(tail), name, check=False, timeout=5)
        return result.stdout + result.stderr
    except RuntimeError:
        return "(log retrieval failed)"
