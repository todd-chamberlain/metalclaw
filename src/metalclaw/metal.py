"""Host-side Metal inference server management.

In 'metal' mode, llama-server runs natively on macOS with the Metal backend,
giving full Apple GPU acceleration. The container handles agent execution,
workspace isolation, and policy enforcement -- connecting to the host inference
server via the network.
"""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import time
from pathlib import Path

from rich.console import Console

from metalclaw.config import METALCLAW_HOME, STATE_DIR, load_config

console = Console()

LLAMA_CPP_DIR = METALCLAW_HOME / "llama-cpp"
BIN_DIR = METALCLAW_HOME / "bin"
LLAMA_SERVER_BIN = BIN_DIR / "llama-server"
PID_FILE = STATE_DIR / "llama-server.pid"
MODEL_FILE = STATE_DIR / "llama-server.model"  # Track which model is running


def check_build_deps() -> bool:
    """Check cmake and compiler toolchain are available."""
    import shutil

    cmake = shutil.which("cmake")
    if not cmake:
        console.print("[red]cmake not found. Install: brew install cmake[/red]")
        return False

    # Check for a working compiler (clang from Xcode CLT or Xcode)
    clang_check = subprocess.run(
        ["xcrun", "--find", "clang"], capture_output=True, text=True, timeout=10
    )
    if clang_check.returncode != 0:
        console.print(
            "[red]C compiler not found. Install Xcode Command Line Tools:[/red]"
        )
        console.print("[red]  xcode-select --install[/red]")
        return False

    # Verify Metal.framework exists (ships with macOS, not Xcode-specific)
    metal_fw = Path("/System/Library/Frameworks/Metal.framework")
    if not metal_fw.exists():
        console.print("[red]Metal.framework not found. macOS 10.14+ required.[/red]")
        return False

    return True


def build_server(force: bool = False) -> bool:
    """Clone and build llama.cpp with Metal backend on the host."""
    if LLAMA_SERVER_BIN.exists() and not force:
        console.print(f"[green]llama-server already built at {LLAMA_SERVER_BIN}[/green]")
        return True

    if not check_build_deps():
        return False

    BIN_DIR.mkdir(parents=True, exist_ok=True)

    # Clone or update llama.cpp
    if not LLAMA_CPP_DIR.exists():
        console.print("Cloning llama.cpp...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/ggml-org/llama.cpp.git",
                 str(LLAMA_CPP_DIR)],
                check=True, capture_output=True, text=True, timeout=120,
            )
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to clone llama.cpp: {e.stderr}[/red]")
            return False
    else:
        console.print("Updating llama.cpp...")
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=str(LLAMA_CPP_DIR),
            capture_output=True, text=True, timeout=60, check=False,
        )
        if result.returncode != 0:
            console.print(
                f"[yellow]Warning: git pull failed (rc={result.returncode}), "
                f"building with existing source[/yellow]"
            )
            if result.stderr:
                console.print(f"[dim]{result.stderr.strip()[:200]}[/dim]")

    # Build with Metal backend
    console.print("Building llama-server with Metal backend (this may take a few minutes)...")
    build_dir = LLAMA_CPP_DIR / "build"

    try:
        subprocess.run(
            ["cmake", "-B", str(build_dir),
             "-DGGML_METAL=ON",
             "-DCMAKE_BUILD_TYPE=Release",
             "-DBUILD_SHARED_LIBS=OFF"],
            cwd=str(LLAMA_CPP_DIR),
            check=True, capture_output=True, text=True, timeout=120,
        )
    except subprocess.CalledProcessError as e:
        console.print(f"[red]cmake configure failed: {e.stderr}[/red]")
        return False

    try:
        nproc = os.cpu_count() or 4
        subprocess.run(
            ["cmake", "--build", str(build_dir),
             "--config", "Release",
             f"-j{nproc}",
             "--target", "llama-server"],
            cwd=str(LLAMA_CPP_DIR),
            check=True, capture_output=True, text=True, timeout=600,
        )
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e.stderr[:500]}[/red]")
        return False

    # Copy binary to bin dir
    built_bin = build_dir / "bin" / "llama-server"
    if not built_bin.exists():
        console.print("[red]Build succeeded but llama-server binary not found[/red]")
        return False

    import shutil
    shutil.copy2(str(built_bin), str(LLAMA_SERVER_BIN))
    LLAMA_SERVER_BIN.chmod(0o755)

    console.print(f"[green]llama-server built: {LLAMA_SERVER_BIN}[/green]")
    return True


def _read_pid_file() -> int | None:
    """Read PID from file with file locking to prevent TOCTOU races."""
    if not PID_FILE.exists():
        return None
    try:
        with open(PID_FILE) as f:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
            try:
                pid = int(f.read().strip())
                return pid
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except (ValueError, OSError):
        return None


def _write_pid_file(pid: int, model_path: str = "") -> None:
    """Write PID to file with file locking."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PID_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(str(pid))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    # Also record which model is running
    if model_path:
        MODEL_FILE.write_text(model_path)


def server_running() -> bool:
    """Check if host llama-server is running."""
    pid = _read_pid_file()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)  # signal 0 = check existence
        return True
    except OSError:
        PID_FILE.unlink(missing_ok=True)
        MODEL_FILE.unlink(missing_ok=True)
        return False


def _running_model() -> str:
    """Get the model path of the currently running server."""
    if MODEL_FILE.exists():
        return MODEL_FILE.read_text().strip()
    return ""


def start_server(
    model_path: Path,
    port: int | None = None,
    ctx_size: int | None = None,
    gpu_layers: int | None = None,
) -> bool:
    """Start llama-server on the host with Metal backend."""
    cfg = load_config()
    port = port or cfg["inference"]["port"]
    ctx_size = ctx_size or cfg["inference"]["context_size"]
    gpu_layers = gpu_layers if gpu_layers is not None else cfg["gpu"]["layers"]

    if server_running():
        # Check if the running server is using the same model
        running_model = _running_model()
        if running_model and running_model != str(model_path):
            console.print(
                f"[yellow]Server running with different model ({Path(running_model).name}), "
                f"restarting with {model_path.name}...[/yellow]"
            )
            stop_server()
        else:
            console.print("[green]Host llama-server already running[/green]")
            return True

    if not LLAMA_SERVER_BIN.exists():
        console.print("[red]llama-server not built. Run: metalclaw onboard[/red]")
        return False

    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/red]")
        return False

    console.print(f"Starting Metal inference server on port {port}...")

    # Start llama-server as a background process
    cmd = [
        str(LLAMA_SERVER_BIN),
        "--model", str(model_path),
        "--port", str(port),
        "--host", "0.0.0.0",  # nosec B104
        "--ctx-size", str(ctx_size),
        "--n-gpu-layers", str(gpu_layers),
        "--log-disable",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Save PID and model path
        _write_pid_file(proc.pid, str(model_path))

        # Brief check that it didn't crash immediately
        time.sleep(1)
        if proc.poll() is not None:
            console.print("[red]llama-server exited immediately[/red]")
            PID_FILE.unlink(missing_ok=True)
            MODEL_FILE.unlink(missing_ok=True)
            return False

        console.print(f"[green]Metal inference server started (PID {proc.pid})[/green]")
        return True
    except OSError as e:
        console.print(f"[red]Failed to start llama-server: {e}[/red]")
        return False


def stop_server() -> bool:
    """Stop the host llama-server."""
    if not server_running():
        PID_FILE.unlink(missing_ok=True)
        MODEL_FILE.unlink(missing_ok=True)
        return True

    pid = _read_pid_file()
    if pid is None:
        return True

    try:
        console.print(f"Stopping Metal inference server (PID {pid})...")
        os.kill(pid, signal.SIGTERM)
        # Wait up to 10 seconds for graceful shutdown
        for _ in range(20):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except OSError:
                break
        else:
            # Force kill if still running
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

        PID_FILE.unlink(missing_ok=True)
        MODEL_FILE.unlink(missing_ok=True)
        console.print("[green]Metal inference server stopped[/green]")
        return True
    except (ValueError, OSError) as e:
        console.print(f"[red]Error stopping server: {e}[/red]")
        PID_FILE.unlink(missing_ok=True)
        MODEL_FILE.unlink(missing_ok=True)
        return False
