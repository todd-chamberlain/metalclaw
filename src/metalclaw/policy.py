"""YAML network policy parsing and preset management (NemoClaw-compatible schema)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

console = Console()

# Policies ship with the package
POLICIES_DIR = Path(__file__).parent.parent.parent / "policies"


@dataclass
class EndpointRule:
    host: str
    ports: list[int] = field(default_factory=lambda: [443])
    protocol: str = "tcp"
    direction: str = "outbound"


@dataclass
class NetworkPolicy:
    name: str
    description: str
    default_action: str  # "deny" or "allow"
    allow_localhost: bool
    endpoints: list[EndpointRule] = field(default_factory=list)


def _find_policy_file(name: str) -> Path | None:
    """Locate a policy YAML by name (check presets/ then base dir)."""
    candidates = [
        POLICIES_DIR / "presets" / f"{name}.yaml",
        POLICIES_DIR / f"{name}.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_policy(name: str) -> NetworkPolicy | None:
    """Load and parse a policy YAML file."""
    path = _find_policy_file(name)
    if not path:
        console.print(f"[red]Policy not found: {name}[/red]")
        return None

    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        console.print(f"[red]Failed to read policy {name}: {e}[/red]")
        return None

    if not isinstance(data, dict):
        console.print(f"[red]Invalid policy format in {name}: expected mapping[/red]")
        return None

    net = data.get("network_policies", data)
    endpoints = []
    for ep in net.get("allowed_endpoints", []):
        endpoints.append(
            EndpointRule(
                host=ep.get("host", ""),
                ports=ep.get("ports", [443]),
                protocol=ep.get("protocol", "tcp"),
                direction=ep.get("direction", "outbound"),
            )
        )

    return NetworkPolicy(
        name=data.get("name", name),
        description=data.get("description", ""),
        default_action=net.get("default_action", "deny"),
        allow_localhost=net.get("allow_localhost", True),
        endpoints=endpoints,
    )


def merge_policies(base: NetworkPolicy, *presets: NetworkPolicy) -> NetworkPolicy:
    """Merge preset policies into a base policy."""
    merged_endpoints = list(base.endpoints)
    for preset in presets:
        merged_endpoints.extend(preset.endpoints)

    return NetworkPolicy(
        name=base.name,
        description=base.description,
        default_action=base.default_action,
        allow_localhost=base.allow_localhost,
        endpoints=merged_endpoints,
    )


def policy_to_podman_network(policy: NetworkPolicy) -> str:
    """Determine podman network mode from policy.

    MVP: only supports binary network isolation. 'pasta' mode with
    --dns-forward=none restricts outbound to localhost only when deny-all.
    Full 'pasta' mode grants outbound access for policies with endpoints.

    Note: --network=none cannot be used because it also blocks port
    forwarding (-p), making the inference API unreachable from the host.

    Returns:
        'pasta' in all cases (port forwarding requires it)
    """
    return "pasta"


def list_presets() -> list[dict[str, str]]:
    """List available policy presets."""
    presets_dir = POLICIES_DIR / "presets"
    result = []
    if presets_dir.exists():
        for f in sorted(presets_dir.glob("*.yaml")):
            try:
                with open(f) as fh:
                    data = yaml.safe_load(fh) or {}
                if not isinstance(data, dict):
                    data = {}
            except (yaml.YAMLError, OSError):
                data = {}
            result.append({
                "name": f.stem,
                "description": data.get("description", ""),
            })

    # Include base policies
    for f in sorted(POLICIES_DIR.glob("*.yaml")):
        try:
            with open(f) as fh:
                data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                data = {}
        except (yaml.YAMLError, OSError):
            data = {}
        result.append({
            "name": f.stem,
            "description": data.get("description", "(base policy)"),
        })

    return result


def print_policy(policy: NetworkPolicy) -> None:
    """Print policy summary."""
    console.print(f"  Policy: [cyan]{policy.name}[/cyan]")
    console.print(f"  Default: [cyan]{policy.default_action}[/cyan]")
    console.print(f"  Localhost: {'[green]allowed[/green]' if policy.allow_localhost else '[red]denied[/red]'}")
    if policy.endpoints:
        console.print(f"  Allowed endpoints:")
        for ep in policy.endpoints:
            ports = ",".join(str(p) for p in ep.ports)
            console.print(f"    - {ep.host}:{ports} ({ep.protocol})")
