"""Apple Silicon GPU detection and capability reporting."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from rich.console import Console

console = Console()


@dataclass
class GPUInfo:
    is_apple_silicon: bool
    chip_name: str
    gpu_cores: int
    unified_memory_gb: int
    recommended_layers: int
    max_model_size_gb: int


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def detect_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    out = _run(["sysctl", "-n", "hw.optional.arm64"])
    return out == "1"


def get_gpu_info() -> GPUInfo:
    """Detect GPU capabilities via system_profiler."""
    if not detect_apple_silicon():
        return GPUInfo(
            is_apple_silicon=False,
            chip_name="unknown",
            gpu_cores=0,
            unified_memory_gb=0,
            recommended_layers=-1,
            max_model_size_gb=0,
        )

    chip_name = "Apple Silicon"
    gpu_cores = 0
    memory_gb = 0

    # Get chip details from SPHardwareDataType
    hw_json = _run(["system_profiler", "SPHardwareDataType", "-json"])
    if hw_json:
        try:
            data = json.loads(hw_json)
            hw = data.get("SPHardwareDataType", [{}])[0]
            chip_name = hw.get("chip_type", chip_name)
            mem_str = hw.get("physical_memory", "0 GB")
            memory_gb = int(mem_str.split()[0]) if mem_str else 0
        except (json.JSONDecodeError, IndexError, ValueError):
            pass

    # Get GPU core count from SPDisplaysDataType
    disp_json = _run(["system_profiler", "SPDisplaysDataType", "-json"])
    if disp_json:
        try:
            data = json.loads(disp_json)
            displays = data.get("SPDisplaysDataType", [])
            for d in displays:
                name = d.get("sppci_model", "")
                if "Apple" in name:
                    cores_str = d.get("sppci_cores", "0")
                    gpu_cores = int(cores_str) if cores_str else 0
                    break
        except (json.JSONDecodeError, ValueError):
            pass

    # Heuristic: reserve ~20% memory for system, rest for model
    max_model_gb = int(memory_gb * 0.75) if memory_gb else 0

    return GPUInfo(
        is_apple_silicon=True,
        chip_name=chip_name,
        gpu_cores=gpu_cores,
        unified_memory_gb=memory_gb,
        recommended_layers=-1,  # all layers on GPU
        max_model_size_gb=max_model_gb,
    )


def print_gpu_report(info: GPUInfo) -> None:
    """Print GPU capability summary."""
    if not info.is_apple_silicon:
        console.print("[red]Not running on Apple Silicon[/red]")
        return

    console.print(f"  Chip: [cyan]{info.chip_name}[/cyan]")
    console.print(f"  GPU cores: [cyan]{info.gpu_cores}[/cyan]")
    console.print(f"  Unified memory: [cyan]{info.unified_memory_gb} GB[/cyan]")
    console.print(f"  Max model size: [cyan]~{info.max_model_size_gb} GB[/cyan]")

    if info.unified_memory_gb >= 128:
        console.print("  Tier: [green]Ultra (70B+ models)[/green]")
    elif info.unified_memory_gb >= 64:
        console.print("  Tier: [green]Pro (70B quantized models)[/green]")
    elif info.unified_memory_gb >= 32:
        console.print("  Tier: [yellow]Standard (13B-34B models)[/yellow]")
    else:
        console.print("  Tier: [red]Limited (7B models only)[/red]")
