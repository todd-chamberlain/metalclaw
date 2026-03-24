"""Click CLI: onboard, run, status, stop, connect, model, policy, logs."""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.table import Table

from metalclaw import __version__

console = Console()


@click.group()
@click.version_option(__version__, prog_name="metalclaw")
def main() -> None:
    """Metalclaw: Sandboxed GPU-accelerated AI agent runtime for Apple Silicon."""
    pass


# ---------------------------------------------------------------------------
# onboard
# ---------------------------------------------------------------------------

@main.command()
@click.option("--model", default=None, help="Model to download (e.g. qwen2.5-7b)")
@click.option("--skip-download", is_flag=True, help="Skip model download step")
def onboard(model: str | None, skip_download: bool) -> None:
    """Set up metalclaw: preflight checks, GPU detection, model download, image build."""
    from metalclaw import config, preflight, gpu, machine, models, container

    console.print(f"\n[bold]Metalclaw Onboard[/bold] v{__version__}\n")

    # Step 1: Preflight
    console.print("[bold]Step 1/3: Preflight checks[/bold]")
    report = preflight.run_preflight()
    preflight.print_report(report)

    if not report.all_passed:
        console.print("\n[red]Preflight failed. Fix the issues above and retry.[/red]")
        for f in report.failed:
            console.print(f"  [red]{f.name}: {f.detail}[/red]")
        sys.exit(1)

    console.print()
    console.print("[bold]GPU Detection[/bold]")
    gpu_info = gpu.get_gpu_info()
    gpu.print_gpu_report(gpu_info)

    # Save GPU-informed defaults to config
    cfg = config.load_config()
    if gpu_info.unified_memory_gb >= 64:
        cfg["inference"]["model"] = model or "qwen2.5-72b"
        cfg["machine"]["memory"] = min(gpu_info.unified_memory_gb * 1024, 61440)
    else:
        cfg["inference"]["model"] = model or "qwen2.5-7b"

    console.print()
    console.print("[bold]Initializing podman machine (libkrun)[/bold]")
    if not machine.init_machine(
        cpus=cfg["machine"]["cpus"],
        memory_mb=cfg["machine"]["memory"],
        disk_gb=cfg["machine"]["disk"],
    ):
        sys.exit(1)

    if not machine.start_machine():
        sys.exit(1)

    machine.verify_gpu()

    # Step 2: Model + Image
    console.print()
    console.print("[bold]Step 2/3: Model & container image[/bold]")

    if not skip_download:
        model_key = cfg["inference"]["model"]
        console.print(f"Selected model: [cyan]{model_key}[/cyan]")

        existing = models.get_model_path(model_key)
        if existing:
            console.print(f"[green]Model already downloaded: {existing}[/green]")
        else:
            if not models.pull_model(model_key):
                console.print("[red]Model download failed[/red]")
                sys.exit(1)

    console.print()
    if not container.image_exists():
        if not container.build_image():
            sys.exit(1)
    else:
        console.print("[green]Container image already built[/green]")

    # Step 3: Save config
    console.print()
    console.print("[bold]Step 3/3: Saving configuration[/bold]")
    config.save_config(cfg)
    console.print(f"[green]Config saved to {config.CONFIG_PATH}[/green]")

    console.print()
    console.print("[bold green]Onboarding complete![/bold green]")
    console.print("Next: [cyan]metalclaw run[/cyan] to start the sandbox")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@main.command()
@click.option("--model", default=None, help="Override model to use")
@click.option("--agent", default=None, type=click.Choice(["none", "claude-code", "custom"]),
              help="Agent type to run inside sandbox")
@click.option("--presets", default=None, help="Comma-separated policy presets")
def run(model: str | None, agent: str | None, presets: str | None) -> None:
    """Start the metalclaw sandbox with inference server."""
    from metalclaw import config, machine, container, models, inference, policy, agent as agent_mod

    cfg = config.load_config()
    model_key = model or cfg["inference"]["model"]
    agent_type = agent or cfg["agent"]["type"]

    console.print(f"\n[bold]Metalclaw Run[/bold]\n")

    # Ensure machine is running
    ms = machine.get_status()
    if not ms.exists:
        console.print("[red]Machine not initialized. Run: metalclaw onboard[/red]")
        sys.exit(1)
    if not ms.running:
        if not machine.start_machine():
            sys.exit(1)

    # Resolve model path
    model_path = models.get_model_path(model_key)
    if not model_path:
        console.print(f"[red]Model '{model_key}' not found. Run: metalclaw model pull {model_key}[/red]")
        sys.exit(1)

    # Build policy
    base_pol = policy.load_policy(cfg["policy"]["base"])
    if not base_pol:
        console.print("[red]Could not load base policy[/red]")
        sys.exit(1)

    # Apply presets from config and CLI
    preset_names = list(cfg["policy"].get("presets", []))
    if presets:
        preset_names.extend(presets.split(","))

    preset_policies = []
    for name in preset_names:
        p = policy.load_policy(name.strip())
        if p:
            preset_policies.append(p)

    # Agent config + its required presets
    agent_cfg = agent_mod.get_agent_config(agent_type, cfg["agent"]["command"])
    final_policy = agent_mod.resolve_policy_with_agent(base_pol, agent_cfg)
    if preset_policies:
        final_policy = policy.merge_policies(final_policy, *preset_policies)

    console.print("[bold]Configuration[/bold]")
    console.print(f"  Model: [cyan]{model_key}[/cyan] ({model_path})")
    console.print(f"  Agent: [cyan]{agent_cfg.agent_type}[/cyan]")
    policy.print_policy(final_policy)

    # Ensure image exists
    if not container.image_exists():
        console.print("[yellow]Container image not found, building...[/yellow]")
        if not container.build_image():
            sys.exit(1)

    # Start container
    console.print()
    console.print("[bold]Starting sandbox[/bold]")
    if not container.start_container(
        model_path=model_path,
        policy=final_policy,
        agent_type=agent_cfg.agent_type,
        agent_command=agent_cfg.command,
    ):
        sys.exit(1)

    # Health check
    console.print()
    console.print("[bold]Waiting for inference server[/bold]")
    port = cfg["inference"]["port"]
    if not inference.health_check(port):
        console.print("[red]Inference server failed to start. Check logs: metalclaw logs[/red]")
        sys.exit(1)

    inference.verify_model(port)

    console.print()
    console.print("[bold green]Sandbox is running![/bold green]")
    console.print(f"  API: [cyan]http://127.0.0.1:{port}/v1[/cyan]")
    console.print(f"  Shell: [cyan]metalclaw connect[/cyan]")
    console.print(f"  Logs: [cyan]metalclaw logs[/cyan]")
    console.print(f"  Stop: [cyan]metalclaw stop[/cyan]")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@main.command()
def status() -> None:
    """Show status of machine, container, and inference server."""
    from metalclaw import machine, container, inference, config

    cfg = config.load_config()
    port = cfg["inference"]["port"]

    console.print(f"\n[bold]Metalclaw Status[/bold]\n")

    # Machine
    ms = machine.get_status()
    if ms.exists:
        state = "[green]running[/green]" if ms.running else "[yellow]stopped[/yellow]"
        console.print(f"  Machine: {state} (provider={ms.provider}, cpus={ms.cpus})")
    else:
        console.print("  Machine: [red]not initialized[/red]")

    # Container
    name = cfg["sandbox"]["name"]
    if container.container_running(name):
        console.print(f"  Container: [green]running[/green] ({name})")
    elif container.container_exists(name):
        console.print(f"  Container: [yellow]stopped[/yellow] ({name})")
    else:
        console.print("  Container: [dim]not created[/dim]")

    # Inference
    try:
        import httpx
        resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=3.0)
        if resp.status_code == 200:
            console.print(f"  Inference: [green]healthy[/green] (port {port})")
        else:
            console.print(f"  Inference: [yellow]unhealthy (HTTP {resp.status_code})[/yellow]")
    except Exception:
        console.print(f"  Inference: [dim]not reachable[/dim] (port {port})")


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------

@main.command()
@click.option("--machine", "stop_machine", is_flag=True, help="Also stop the podman machine")
def stop(stop_machine: bool) -> None:
    """Stop the sandbox container (and optionally the machine)."""
    from metalclaw import container, machine as mach

    console.print("\n[bold]Metalclaw Stop[/bold]\n")

    container.stop_container()

    if stop_machine:
        mach.stop_machine()

    console.print("[green]Done[/green]")


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------

@main.command()
def connect() -> None:
    """Open an interactive shell inside the sandbox container."""
    from metalclaw import container

    container.exec_shell()


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------

@main.command()
@click.option("--tail", default=100, help="Number of lines to show")
@click.option("-f", "--follow", is_flag=True, help="Follow log output")
def logs(tail: int, follow: bool) -> None:
    """Show container logs."""
    from metalclaw import container

    if follow:
        import os
        from metalclaw.config import load_config
        from metalclaw.container import _LIBKRUN_ENV
        cfg = load_config()
        name = cfg["sandbox"]["name"]
        os.execve(
            "/usr/bin/env",
            ["env", "CONTAINERS_MACHINE_PROVIDER=libkrun",
             "podman", "logs", "-f", "--tail", str(tail), name],
            _LIBKRUN_ENV,
        )
    else:
        output = container.get_logs(tail=tail)
        console.print(output)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

@main.group()
def model() -> None:
    """Manage GGUF models for inference."""
    pass


@model.command("list")
def model_list() -> None:
    """List available and downloaded models."""
    from metalclaw.models import list_available

    table = Table(title="Models")
    table.add_column("Key", style="cyan")
    table.add_column("Name")
    table.add_column("Size", justify="right")
    table.add_column("Min RAM", justify="right")
    table.add_column("Status")

    for m in list_available():
        status = "[green]downloaded[/green]" if m["downloaded"] else "[dim]available[/dim]"
        table.add_row(
            m["key"],
            m["name"],
            f"{m['size_gb']:.1f} GB",
            f"{m['min_memory_gb']} GB",
            status,
        )

    console.print(table)


@model.command("pull")
@click.argument("model_key")
def model_pull(model_key: str) -> None:
    """Download a model from the registry."""
    from metalclaw.models import pull_model

    pull_model(model_key)


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------

@main.group()
def policy() -> None:
    """Manage network policies."""
    pass


@policy.command("list")
def policy_list() -> None:
    """List available policy presets."""
    from metalclaw.policy import list_presets

    table = Table(title="Policy Presets")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    for p in list_presets():
        table.add_row(p["name"], p["description"])

    console.print(table)


@policy.command("show")
@click.argument("name")
def policy_show(name: str) -> None:
    """Show details of a policy."""
    from metalclaw.policy import load_policy, print_policy

    p = load_policy(name)
    if p:
        print_policy(p)


if __name__ == "__main__":
    main()
