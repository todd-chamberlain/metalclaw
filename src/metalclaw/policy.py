"""YAML network policy parsing and preset management (NemoClaw-compatible schema)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

console = Console()

# Policies ship inside the package
POLICIES_DIR = Path(__file__).parent / "policies"


@dataclass
class EndpointRule:
    host: str
    ports: list[int] = field(default_factory=lambda: [443])
    protocol: str = "tcp"
    direction: str = "outbound"
    # NemoClaw-compatible fields
    binaries: list[str] = field(default_factory=list)
    access: str = "full"  # "full" or "read-only" (GET/HEAD/OPTIONS only)
    tls: str = "passthrough"  # "terminate" for L7 inspection, "passthrough" for TCP


@dataclass
class FilesystemPolicy:
    """Filesystem isolation policy."""
    read_only_root: bool = True
    read_write: list[str] = field(default_factory=lambda: ["/sandbox", "/tmp"])
    read_only: list[str] = field(default_factory=lambda: ["/usr", "/lib", "/etc"])


@dataclass
class ProcessPolicy:
    """Process isolation policy."""
    run_as_user: str = "sandbox"
    run_as_group: str = "sandbox"


@dataclass
class NetworkPolicy:
    name: str
    description: str
    default_action: str  # "deny" or "allow"
    allow_localhost: bool
    endpoints: list[EndpointRule] = field(default_factory=list)


@dataclass
class SandboxPolicy:
    """Full sandbox policy combining network, filesystem, and process policies."""
    version: int = 1
    network: NetworkPolicy = field(default_factory=lambda: NetworkPolicy(
        name="default", description="", default_action="deny",
        allow_localhost=True,
    ))
    filesystem: FilesystemPolicy = field(default_factory=FilesystemPolicy)
    process: ProcessPolicy = field(default_factory=ProcessPolicy)


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


def _parse_endpoint(ep: dict[str, Any]) -> EndpointRule:
    """Parse a single endpoint rule from YAML dict."""
    return EndpointRule(
        host=ep.get("host", ""),
        ports=ep.get("ports", [443]),
        protocol=ep.get("protocol", "tcp"),
        direction=ep.get("direction", "outbound"),
        binaries=ep.get("binaries", []),
        access=ep.get("access", "full"),
        tls=ep.get("tls", "passthrough"),
    )


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
    endpoints = [_parse_endpoint(ep) for ep in net.get("allowed_endpoints", [])]

    return NetworkPolicy(
        name=data.get("name", name),
        description=data.get("description", ""),
        default_action=net.get("default_action", "deny"),
        allow_localhost=net.get("allow_localhost", True),
        endpoints=endpoints,
    )


def load_sandbox_policy(name: str) -> SandboxPolicy:
    """Load full sandbox policy (network + filesystem + process)."""
    path = _find_policy_file(name)
    if not path:
        return SandboxPolicy()

    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError):
        return SandboxPolicy()

    if not isinstance(data, dict):
        return SandboxPolicy()

    # Parse network
    net_policy = load_policy(name)

    # Parse filesystem
    fs_data = data.get("filesystem_policy", {})
    fs = FilesystemPolicy(
        read_only_root=fs_data.get("read_only_root", True),
        read_write=fs_data.get("read_write", ["/sandbox", "/tmp"]),
        read_only=fs_data.get("read_only", ["/usr", "/lib", "/etc"]),
    )

    # Parse process
    proc_data = data.get("process", {})
    proc = ProcessPolicy(
        run_as_user=proc_data.get("run_as_user", "sandbox"),
        run_as_group=proc_data.get("run_as_group", "sandbox"),
    )

    return SandboxPolicy(
        version=data.get("version", 1),
        network=net_policy or SandboxPolicy().network,
        filesystem=fs,
        process=proc,
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
            access_tag = f" [{ep.access}]" if ep.access != "full" else ""
            binary_tag = f" (binaries: {', '.join(ep.binaries)})" if ep.binaries else ""
            console.print(f"    - {ep.host}:{ports} ({ep.protocol}){access_tag}{binary_tag}")
