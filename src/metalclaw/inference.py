"""llama-server health checks and inference verification."""

from __future__ import annotations

import json
import time

import httpx
from rich.console import Console

from metalclaw.config import load_config

console = Console()

# Absolute maximum time to wait for inference server, regardless of retries
MAX_HEALTH_CHECK_SECONDS = 300  # 5 minutes


def health_check(port: int | None = None, retries: int = 30,
                 interval: float = 2.0) -> bool:
    """Poll /health until llama-server is ready.

    Uses both retry count AND absolute wall-clock timeout to prevent
    infinite loops on perpetual 'loading model' responses.
    """
    cfg = load_config()
    port = port or cfg["inference"]["port"]
    url = f"http://127.0.0.1:{port}/health"

    start = time.monotonic()

    for i in range(retries):
        # Absolute timeout guard
        elapsed = time.monotonic() - start
        if elapsed > MAX_HEALTH_CHECK_SECONDS:
            console.print(
                f"[red]  Health check timed out after {elapsed:.0f}s[/red]"
            )
            return False

        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                status = data.get("status", "ok")
                if status == "ok":
                    console.print("[green]  Inference server is healthy[/green]")
                    return True
                elif status == "loading model":
                    if i % 5 == 0:
                        console.print(
                            f"  Loading model... ({elapsed:.0f}s elapsed)"
                        )
                else:
                    console.print(f"  Server status: {status}")
        except (httpx.HTTPError, json.JSONDecodeError):
            if i % 5 == 0:
                console.print(f"  Waiting for inference server... ({i}/{retries})")
        time.sleep(interval)

    console.print("[red]  Inference server did not become healthy[/red]")
    return False


def verify_model(port: int | None = None) -> str | None:
    """Check /v1/models to confirm model is loaded. Returns model name or None."""
    cfg = load_config()
    port = port or cfg["inference"]["port"]
    url = f"http://127.0.0.1:{port}/v1/models"

    try:
        resp = httpx.get(url, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if models:
                model_id = models[0].get("id", "unknown")
                console.print(f"  Model loaded: [cyan]{model_id}[/cyan]")
                return model_id
    except (httpx.HTTPError, json.JSONDecodeError):
        pass

    console.print("[red]  Could not verify model[/red]")
    return None


def test_inference(port: int | None = None) -> bool:
    """Send a minimal completion request to verify GPU inference works."""
    cfg = load_config()
    port = port or cfg["inference"]["port"]
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10,
        "temperature": 0,
    }

    try:
        start = time.monotonic()
        resp = httpx.post(url, json=payload, timeout=60.0)
        elapsed = time.monotonic() - start

        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                console.print(f"  Test response: [cyan]{text.strip()}[/cyan]")
                console.print(f"  Latency: [cyan]{elapsed:.2f}s[/cyan]")
                return True
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        console.print(f"[red]  Inference test failed: {e}[/red]")

    return False
