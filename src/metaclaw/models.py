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

from metaclaw.config import MODELS_DIR, ensure_dirs

console = Console()

REGISTRY_PATH = MODELS_DIR / "registry.json"

# Curated model registry -- URLs point to HuggingFace GGUF repos
BUILTIN_MODELS: dict[str, dict] = {
    "qwen2.5-7b": {
        "name": "Qwen 2.5 7B Instruct",
        "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size_gb": 4.4,
        "min_memory_gb": 8,
        "context_window": 32768,
        "description": "Fast, good for testing and light tasks",
    },
    "qwen2.5-72b": {
        "name": "Qwen 2.5 72B Instruct",
        "url": "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF/resolve/main/qwen2.5-72b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-72b-instruct-q4_k_m.gguf",
        "size_gb": 42.0,
        "min_memory_gb": 64,
        "context_window": 32768,
        "description": "Strong general-purpose model for Ultra hardware",
    },
    "llama-3.3-70b": {
        "name": "Llama 3.3 70B Instruct",
        "url": "https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "size_gb": 40.0,
        "min_memory_gb": 64,
        "context_window": 131072,
        "description": "Strong reasoning, large context window",
    },
    "deepseek-r1-70b": {
        "name": "DeepSeek R1 70B",
        "url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf",
        "filename": "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf",
        "size_gb": 40.0,
        "min_memory_gb": 64,
        "context_window": 65536,
        "description": "Reasoning and code focused",
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
        with open(REGISTRY_PATH) as f:
            return json.load(f)
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
    """
    reg = _load_registry()
    if key in reg:
        p = Path(reg[key]["path"]).resolve()
        if p.exists() and p.is_relative_to(MODELS_DIR.resolve()):
            return p
    # Check if key is a direct path to a .gguf file within MODELS_DIR
    p = Path(key).resolve()
    if p.exists() and p.suffix == ".gguf" and p.is_relative_to(MODELS_DIR.resolve()):
        return p
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
        with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                          timeout=httpx.Timeout(30.0, read=None)) as resp:
            if resp.status_code == 416:
                # Range not satisfiable -- file is complete
                console.print("[green]Model already downloaded[/green]")
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

    console.print("Verifying download...")
    sha = _sha256_file(dest)

    reg = _load_registry()
    reg[key] = {
        "name": info["name"],
        "path": str(dest),
        "size_gb": info["size_gb"],
        "sha256": sha,
    }
    _save_registry(reg)

    console.print(f"[green]Model saved: {dest}[/green]")
    return dest
