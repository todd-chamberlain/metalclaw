"""Container image build and lifecycle management."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from rich.console import Console

from metalclaw.config import load_config
from metalclaw.policy import (
    NetworkPolicy,
    SandboxPolicy,
    policy_to_podman_args,
)

console = Console()

IMAGE_NAME = "metalclaw-sandbox"
# Container dir: first check source tree, then fall back to package-relative
_SRC_CONTAINER_DIR = Path(__file__).parent.parent.parent / "container"
_PKG_CONTAINER_DIR = Path(__file__).parent / "container"
CONTAINER_DIR = _SRC_CONTAINER_DIR if _SRC_CONTAINER_DIR.exists() else _PKG_CONTAINER_DIR

# Ensure podman talks to the libkrun machine, not applehv
_LIBKRUN_ENV = {**os.environ, "CONTAINERS_MACHINE_PROVIDER": "libkrun"}

# Maximum log tail to prevent resource exhaustion
MAX_LOG_TAIL = 10000


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


def _resolve_pip_packages(raw: str | list) -> list[str]:
    """Normalize extra_pip_packages config to a list of requirement lines.

    Accepts either a YAML list or a legacy space-separated string.
    """
    if isinstance(raw, list):
        return [s.strip() for s in raw if s and str(s).strip()]
    if isinstance(raw, str) and raw.strip():
        return raw.split()
    return []


def _validate_base_image(value: str) -> str | None:
    """Validate base_image format. Returns error message or None if valid."""
    if not value:
        return "base_image cannot be empty"
    # OCI image references: registry/path:tag or registry/path@digest
    if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9.\-_/:@]+$', value):
        return f"base_image contains invalid characters: {value}"
    return None


def _validate_system_packages(value: str) -> str | None:
    """Validate extra_system_packages for safe characters. Returns error or None."""
    if not value:
        return None
    # Package names: alphanumeric, dash, underscore, dot, plus, space
    if not re.match(r'^[a-zA-Z0-9.\-_+ ]+$', value):
        return f"extra_system_packages contains invalid characters: {value}"
    return None


def _validate_url_scheme(value: str) -> str | None:
    """Validate URL uses https:// only. Returns error or None.

    CA certificates must be fetched over HTTPS to prevent MITM injection.
    """
    if not value:
        return None
    if not value.startswith("https://"):
        return f"ca_cert_url must use https:// (got: {value})"
    return None


# Env var keys that could enable privilege escalation or code injection
# inside the container if overridden by an attacker with config write access.
_BLOCKED_ENV_KEYS = frozenset({
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "LD_DEBUG",
    "PATH", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME",
    "NODE_OPTIONS", "NODE_PATH",
    "BASH_ENV", "ENV", "CDPATH", "GLOBIGNORE",
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
})


def _validate_env_var(key: str, value: str) -> str | None:
    """Validate a deploy.extra_env entry. Returns error or None."""
    if key in _BLOCKED_ENV_KEYS:
        return (
            f"deploy.extra_env key '{key}' is blocked (security-sensitive). "
            "Remove it or use a container-safe alternative."
        )
    if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', key):
        return f"deploy.extra_env key '{key}' is not a valid env var name"
    return None


def _path_has_symlink(p: Path) -> bool:
    """Check if any component of a path is a symlink."""
    for parent in [p, *p.parents]:
        if parent.is_symlink():
            return True
        if parent == parent.parent:
            break
    return False


def _validate_pip_requirement(line: str) -> str | None:
    """Validate a single pip requirement line. Returns error or None."""
    lowered = line.lower().strip()
    if lowered.startswith(("file:", "git+", "hg+", "svn+", "bzr+")):
        return f"extra_pip_packages entry uses disallowed VCS/file source: {line}"
    if "--index-url" in lowered or "--extra-index-url" in lowered:
        return f"extra_pip_packages entry contains disallowed pip option: {line}"
    if "--find-links" in lowered or "--trusted-host" in lowered:
        return f"extra_pip_packages entry contains disallowed pip option: {line}"
    return None


def build_image() -> bool:
    """Build the sandbox container image from Containerfile.

    Reads build.* config keys:
        build.base_image             -> BASE_IMAGE build arg
        build.extra_pip_packages     -> extra-requirements.txt (COPY'd into image)
        build.extra_system_packages  -> EXTRA_SYSTEM_PACKAGES build arg
        build.ca_cert_url            -> CA_CERT_URL build arg (fallback)
        deploy.ca_cert               -> build-ca-cert.pem (COPY'd, preferred)
    """
    containerfile = CONTAINER_DIR / "Containerfile"
    if not containerfile.exists():
        console.print(f"[red]Containerfile not found at {containerfile}[/red]")
        return False

    cfg = load_config()
    build_cfg = cfg.get("build", {})

    # ── Validate build inputs ────────────────────────────────────
    base_image = build_cfg.get("base_image", "registry.fedoraproject.org/fedora:42")
    err = _validate_base_image(base_image)
    if err:
        console.print(f"[red]{err}[/red]")
        return False

    extra_sys = build_cfg.get("extra_system_packages", "")
    err = _validate_system_packages(extra_sys)
    if err:
        console.print(f"[red]{err}[/red]")
        return False

    ca_cert_url = build_cfg.get("ca_cert_url", "")
    err = _validate_url_scheme(ca_cert_url)
    if err:
        console.print(f"[red]{err}[/red]")
        return False

    # ── Generate requirements file in build context ──────────────
    # Standard pip requirements file — no shell expansion, no quoting issues.
    pip_packages = _resolve_pip_packages(build_cfg.get("extra_pip_packages", []))
    for pkg in pip_packages:
        err = _validate_pip_requirement(pkg)
        if err:
            console.print(f"[red]{err}[/red]")
            return False
    req_file = CONTAINER_DIR / "extra-requirements.txt"
    req_file.write_text("\n".join(pip_packages) + "\n" if pip_packages else "")

    # ── CA cert: prefer local file over URL fetch ────────────────
    # If deploy.ca_cert points to a local file, copy it into the build
    # context so the Containerfile can COPY it directly.  This avoids
    # requiring network access to a corporate PKI server during build.
    # Falls back to CA_CERT_URL ARG for CI/CD environments.
    deploy_cfg = cfg.get("deploy", {})
    ca_cert_file = CONTAINER_DIR / "build-ca-cert.pem"
    local_ca = deploy_cfg.get("ca_cert", "")
    if local_ca:
        local_ca_expanded = Path(local_ca).expanduser()
        if _path_has_symlink(local_ca_expanded):
            console.print(f"[yellow]Warning: CA cert path contains a symlink, rejecting: {local_ca}[/yellow]")
            ca_cert_file.write_text("")
        elif local_ca_expanded.resolve().is_file():
            shutil.copy2(local_ca_expanded.resolve(), ca_cert_file)
        else:
            ca_cert_file.write_text("")
    else:
        ca_cert_file.write_text("")

    cmd = [
        "build",
        "-t", IMAGE_NAME,
        "-f", str(containerfile),
        "--build-arg", f"BASE_IMAGE={base_image}",
    ]

    # Build args (system packages and CA cert URL fallback)
    if extra_sys:
        cmd.extend(["--build-arg", f"EXTRA_SYSTEM_PACKAGES={extra_sys}"])
    if ca_cert_url and not ca_cert_file.stat().st_size:
        # Only use URL if no local cert was copied
        cmd.extend(["--build-arg", f"CA_CERT_URL={ca_cert_url}"])

    cmd.append(str(CONTAINER_DIR))

    console.print("Building container image (this may take several minutes)...")
    console.print(f"  Base image: [cyan]{base_image}[/cyan]")
    if pip_packages:
        console.print(f"  Extra pip: [cyan]{', '.join(pip_packages)}[/cyan]")
    if extra_sys:
        console.print(f"  Extra sys: [cyan]{extra_sys}[/cyan]")
    if ca_cert_file.stat().st_size:
        console.print(f"  CA cert:   [cyan]{local_ca} (local)[/cyan]")
    elif ca_cert_url:
        console.print(f"  CA cert:   [cyan]{ca_cert_url} (URL)[/cyan]")

    try:
        result = _podman(*cmd, timeout=1800, check=False)
        if result.returncode != 0:
            console.print(f"[red]Build failed:\n{result.stderr}[/red]")
            return False
        console.print("[green]Container image built[/green]")
        return True
    except RuntimeError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        return False
    finally:
        # Clean up generated files from source tree
        req_file.unlink(missing_ok=True)
        ca_cert_file.unlink(missing_ok=True)


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
    sandbox_policy: SandboxPolicy | None = None,
    network_policy: NetworkPolicy | None = None,
    agent_type: str = "none",
    agent_command: str = "",
    gpu_backend: str = "vulkan",
    inference_url: str = "",
) -> bool:
    """Create and start the sandbox container.

    Args:
        model_path: Path to the GGUF model file.
        sandbox_policy: Full sandbox policy (filesystem + process + network).
        network_policy: Network policy (used if sandbox_policy not provided).
        agent_type: Agent type (none, claude-code, custom).
        agent_command: Custom agent command (validated for safety).
        gpu_backend: GPU backend (metal, vulkan, cpu).
        inference_url: URL of host inference server (metal mode).
    """
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

    # Resolve policies from SandboxPolicy or fallback to legacy args
    if sandbox_policy:
        fs_policy = sandbox_policy.filesystem
        proc_policy = sandbox_policy.process
        net_policy = sandbox_policy.network
    else:
        fs_policy = None
        proc_policy = None
        net_policy = network_policy

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
        "-e", f"GPU_BACKEND={gpu_backend}",
        "-e", f"AGENT_TYPE={agent_type}",
        "-e", f"PORT={port}",
    ]

    # Network isolation from policy
    if net_policy:
        cmd.extend(policy_to_podman_args(net_policy))
    else:
        cmd.append("--network=pasta")

    # Filesystem isolation -- always enforce from policy or defaults
    read_only_root = fs_policy.read_only_root if fs_policy else True
    if read_only_root:
        cmd.append("--read-only")
        cmd.extend(["--tmpfs", "/tmp:rw,nosuid,nodev,size=1g"])  # nosec B108
        cmd.extend(["--tmpfs", "/sandbox:rw,nosuid,nodev,size=10g"])

    # Process isolation -- always default to sandbox user
    run_user = proc_policy.run_as_user if proc_policy else "sandbox"
    if not run_user:
        run_user = "sandbox"
    cmd.extend(["--user", run_user])

    if gpu_backend == "metal" and inference_url:
        # Metal mode: inference runs on host, container is agent/workspace only
        cmd.extend([
            "-e", f"INFERENCE_URL={inference_url}",
        ])
    else:
        # Vulkan/CPU mode: inference runs in container
        cmd.extend([
            "--device", "/dev/dri",
            "-v", f"{model_path}:/models/model.gguf:ro",
            "-p", f"127.0.0.1:{port}:{port}",
            "-e", f"MODEL_PATH=/models/model.gguf",
            "-e", f"CTX_SIZE={ctx_size}",
            "-e", f"GPU_LAYERS={gpu_layers}",
        ])

    # ── Deploy-level: CA cert mount (any agent type) ─────────────
    deploy = cfg.get("deploy", {})
    ca_cert = deploy.get("ca_cert", "")
    if ca_cert:
        ca_expanded = Path(ca_cert).expanduser()
        if _path_has_symlink(ca_expanded):
            console.print(f"[yellow]Warning: CA cert path contains a symlink, rejecting: {ca_cert}[/yellow]")
        elif ca_expanded.resolve().is_file():
            ca_path = ca_expanded.resolve()
            container_cert = "/etc/metalclaw/ca-cert.pem"
            cmd.extend(["-v", f"{ca_path}:{container_cert}:ro"])
            cmd.extend(["-e", f"SSL_CERT_FILE={container_cert}"])
            cmd.extend(["-e", f"REQUESTS_CA_BUNDLE={container_cert}"])
            # Agent-specific cert env vars
            if agent_type == "mattermost":
                cmd.extend(["-e", f"MATTERMOST_CA_CERT={container_cert}"])
        else:
            console.print(f"[yellow]Warning: CA cert not found: {ca_cert}[/yellow]")

    # ── Deploy-level: extra env vars (any agent type) ──────────
    extra_env = deploy.get("extra_env", {})
    if isinstance(extra_env, dict):
        for k, v in extra_env.items():
            err = _validate_env_var(k, v)
            if err:
                console.print(f"[red]{err}[/red]")
                return False
            cmd.extend(["-e", f"{k}={v}"])

    # ── Mattermost agent: inject config env vars ───────────────
    if agent_type == "mattermost":
        mm = cfg.get("mattermost", {})
        mm_env = {
            "MATTERMOST_URL": mm.get("url", ""),
            "MATTERMOST_TOKEN": mm.get("token", ""),
            "MATTERMOST_TEAM": mm.get("team", ""),
            "MATTERMOST_TRIGGER": mm.get("trigger", "@metalclaw"),
            "MATTERMOST_SYSTEM_PROMPT": mm.get("system_prompt", ""),
            "MATTERMOST_MAX_HISTORY": str(mm.get("max_history", 20)),
        }
        for k, v in mm_env.items():
            if v:
                cmd.extend(["-e", f"{k}={v}"])

    # Agent command -- validate before passing to container
    if agent_command:
        _validate_agent_command(agent_command)
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


def _validate_agent_command(command: str) -> None:
    """Validate agent command for shell injection and globbing risks."""
    dangerous = {"$(", "`", "&&", "||", ";", "|", ">", "<", "\n", "*", "?"}
    for pattern in dangerous:
        if pattern in command:
            raise ValueError(
                f"Agent command contains disallowed character: {pattern!r}. "
                "Use a simple command without shell operators or glob patterns."
            )


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

    # Cap tail to prevent resource exhaustion
    tail = min(tail, MAX_LOG_TAIL)

    if not container_exists(name):
        return "Container does not exist"

    try:
        result = _podman("logs", "--tail", str(tail), name, check=False, timeout=5)
        return result.stdout + result.stderr
    except RuntimeError:
        return "(log retrieval failed)"
