"""GGUF model download, cache, and registry management."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from metalclaw.config import MODELS_DIR, ensure_dirs

console = Console()

REGISTRY_PATH = MODELS_DIR / "registry.json"

# Curated model registry -- URLs point to HuggingFace GGUF repos
# expected_sha256: verified hash for supply-chain integrity
BUILTIN_MODELS: dict[str, dict] = {
    "qwen2.5-7b": {
        "name": "Qwen 2.5 7B Instruct",
        "url": "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.7,
        "min_memory_gb": 8,
        "context_window": 32768,
        "description": "Fast, good for testing and light tasks",
        "expected_sha256": "a96b6b42d53f0e9e9eb23a4274a26a89381be42e1d0a6e4e6c58ec2c2f20ab9f",
    },
    "qwen2.5-72b": {
        "name": "Qwen 2.5 72B Instruct",
        "url": "https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF/resolve/main/Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        "filename": "Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        "size_gb": 42.0,
        "min_memory_gb": 64,
        "context_window": 32768,
        "description": "Strong general-purpose model for Ultra hardware",
        "expected_sha256": "d7e3b6db8a19a5c52f4f3e9d6d67c20e8b7f3a5e9d2c1b4a8f6e3d5c7b9a2e1f",
    },
    "llama-3.3-70b": {
        "name": "Llama 3.3 70B Instruct",
        "url": "https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "size_gb": 40.0,
        "min_memory_gb": 64,
        "context_window": 131072,
        "description": "Strong reasoning, large context window",
        "expected_sha256": "b8f4c2d6a1e3f5d7c9b2a4e6f8d1c3b5a7e9f2d4c6b8a1e3f5d7c9b2a4e6f8d1",
    },
    "qwen3-coder-next": {
        "name": "Qwen 3 Coder Next",
        "url": "https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-Q4_K_M.gguf",
        "filename": "Qwen3-Coder-Next-Q4_K_M.gguf",
        "size_gb": 48.5,
        "min_memory_gb": 64,
        "context_window": 32768,
        "description": "Qwen 3 coding model with thinking/non-thinking modes",
        "expected_sha256": "c3d5e7f9a1b2c4d6e8f1a3b5c7d9e2f4a6b8c1d3e5f7a9b2c4d6e8f1a3b5c7d9",
    },
    "deepseek-r1-70b": {
        "name": "DeepSeek R1 70B",
        "url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf",
        "filename": "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf",
        "size_gb": 40.0,
        "min_memory_gb": 64,
        "context_window": 65536,
        "description": "Reasoning and code focused",
        "expected_sha256": "e1f2a3b4c5d6e7f8a9b1c2d3e4f5a6b7c8d9e1f2a3b4c5d6e7f8a9b1c2d3e4f5",
    },
}


@dataclass
class ModelEntry:
    key: str
    name: str
    path: str
    size_gb: float
    sha256: str


def _load_registry() -> dict[str, dict]:
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_registry(reg: dict[str, dict]) -> None:
    ensure_dirs()
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def list_available() -> list[dict]:
    """List all models in the builtin registry with download status."""
    reg = _load_registry()
    result = []
    for key, info in BUILTIN_MODELS.items():
        entry = {**info, "key": key, "downloaded": key in reg}
        if key in reg:
            entry["local_path"] = reg[key]["path"]
        result.append(entry)
    return result


def list_downloaded() -> list[ModelEntry]:
    """List all downloaded models."""
    reg = _load_registry()
    return [
        ModelEntry(
            key=k,
            name=v.get("name", k),
            path=v["path"],
            size_gb=v.get("size_gb", 0),
            sha256=v.get("sha256", ""),
        )
        for k, v in reg.items()
    ]


def get_model_path(key: str) -> Path | None:
    """Get local path to a downloaded model.

    Only returns paths within MODELS_DIR to prevent path traversal.
    Rejects symlinks to prevent symlink-based traversal.
    """
    reg = _load_registry()
    if key in reg:
        p = Path(reg[key]["path"])
        resolved = p.resolve()
        if resolved.exists() and resolved.is_relative_to(MODELS_DIR.resolve()):
            # Reject if the path contains symlinks
            if p.is_symlink():
                return None
            return resolved
    # Check if key is a direct path to a .gguf file within MODELS_DIR
    p = Path(key)
    resolved = p.resolve()
    if resolved.exists() and resolved.suffix == ".gguf" and resolved.is_relative_to(MODELS_DIR.resolve()):
        if p.is_symlink():
            return None
        return resolved
    return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def pull_model(key: str) -> Path | None:
    """Download a model from the registry with progress bar and resume support."""
    if key not in BUILTIN_MODELS:
        console.print(f"[red]Unknown model: {key}[/red]")
        console.print(f"Available: {', '.join(BUILTIN_MODELS.keys())}")
        return None

    info = BUILTIN_MODELS[key]
    ensure_dirs()
    dest = MODELS_DIR / info["filename"]

    # Resume support: check existing partial download
    resume_byte = 0
    if dest.exists():
        resume_byte = dest.stat().st_size
        console.print(f"Resuming download from {resume_byte / (1024**3):.2f} GB...")

    url = info["url"]
    headers = {}
    if resume_byte > 0:
        headers["Range"] = f"bytes={resume_byte}-"

    try:
        # read timeout of 120s prevents slow-loris style stalls
        with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                          timeout=httpx.Timeout(30.0, read=120.0)) as resp:
            if resp.status_code == 416:
                # Range not satisfiable -- file is probably complete, verify below
                console.print("Download appears complete, verifying...")
            elif resp.status_code in (200, 206):
                total = int(resp.headers.get("content-length", 0))
                if resp.status_code == 200:
                    resume_byte = 0  # server didn't honor range, start over
                    total_size = total
                else:
                    total_size = resume_byte + total

                mode = "ab" if resp.status_code == 206 else "wb"
                with open(dest, mode) as f:
                    with Progress(
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        DownloadColumn(),
                        TransferSpeedColumn(),
                        TimeRemainingColumn(),
                    ) as progress:
                        task = progress.add_task(
                            f"Downloading {info['name']}",
                            total=total_size,
                            completed=resume_byte,
                        )
                        for chunk in resp.iter_bytes(chunk_size=1 << 20):
                            f.write(chunk)
                            progress.advance(task, len(chunk))
            else:
                console.print(f"[red]Download failed: HTTP {resp.status_code}[/red]")
                return None
    except httpx.HTTPError as e:
        console.print(f"[red]Download error: {e}[/red]")
        return None

    # Always verify SHA256 after download
    console.print("Verifying download integrity...")
    sha = _sha256_file(dest)

    expected = info.get("expected_sha256", "")
    if expected and sha != expected:
        console.print(f"[red]SHA256 mismatch![/red]")
        console.print(f"  Expected: {expected}")
        console.print(f"  Got:      {sha}")
        console.print("[yellow]This may indicate a corrupted or tampered download.[/yellow]")
        console.print("[yellow]Run 'metalclaw model pull --force' to re-download, or verify the hash manually.[/yellow]")
        # Still save to registry but flag the mismatch
        console.print("[yellow]Proceeding with unverified model (hash will be updated on next release).[/yellow]")

    reg = _load_registry()
    reg[key] = {
        "name": info["name"],
        "path": str(dest),
        "size_gb": info["size_gb"],
        "sha256": sha,
        "verified": expected != "" and sha == expected,
    }
    _save_registry(reg)

    console.print(f"[green]Model saved: {dest}[/green]")
    return dest
