"""NemoClaw-compatible sandbox policy parsing, merging, and enforcement.

Schema v1 aligns with NVIDIA NemoClaw (March 2026) using named policy groups,
L7 HTTP method+path rules, binary restrictions, and full CONNECT tunnel support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

console = Console()

# Policies ship inside the package
POLICIES_DIR = Path(__file__).parent / "policies"


# ---------------------------------------------------------------------------
# Data model (NemoClaw-compatible)
# ---------------------------------------------------------------------------

@dataclass
class HttpRule:
    """L7 HTTP method + path allow rule."""
    method: str = "*"  # GET, POST, PUT, PATCH, DELETE, *
    path: str = "/**"  # glob pattern


@dataclass
class EndpointRule:
    """Single endpoint within a named policy group."""
    host: str
    port: int = 443
    protocol: str = "rest"  # "rest" for L7 inspection, "tcp" for passthrough
    enforcement: str = "enforce"  # "enforce" or "monitor"
    tls: str = "terminate"  # "terminate" for L7, "passthrough" for TCP
    access: str = ""  # "full" for CONNECT tunnel (WebSockets), empty = use rules
    rules: list[HttpRule] = field(default_factory=list)


@dataclass
class BinarySpec:
    """Binary permitted to use a policy group's endpoints."""
    path: str  # absolute path, supports globs like /usr/bin/python3*


@dataclass
class PolicyGroup:
    """Named network policy group (NemoClaw format)."""
    name: str
    endpoints: list[EndpointRule] = field(default_factory=list)
    binaries: list[BinarySpec] = field(default_factory=list)


@dataclass
class NetworkPolicy:
    """Full network policy: default action + named groups."""
    name: str
    description: str
    default_action: str = "deny"
    allow_localhost: bool = True
    groups: dict[str, PolicyGroup] = field(default_factory=dict)


@dataclass
class FilesystemPolicy:
    """Filesystem isolation policy."""
    read_only_root: bool = True
    include_workdir: bool = True
    read_write: list[str] = field(default_factory=lambda: ["/sandbox", "/tmp"])  # nosec B108
    read_only: list[str] = field(default_factory=lambda: [
        "/usr", "/lib", "/etc", "/proc", "/dev/urandom",
    ])


@dataclass
class ProcessPolicy:
    """Process isolation policy."""
    run_as_user: str = "sandbox"
    run_as_group: str = "sandbox"


@dataclass
class LandlockPolicy:
    """Landlock / macOS Sandbox stub (best_effort on macOS)."""
    compatibility: str = "best_effort"


@dataclass
class SandboxPolicy:
    """Full sandbox policy combining network, filesystem, process, and landlock."""
    version: int = 1
    network: NetworkPolicy = field(default_factory=lambda: NetworkPolicy(
        name="default", description="Deny-all baseline",
    ))
    filesystem: FilesystemPolicy = field(default_factory=FilesystemPolicy)
    process: ProcessPolicy = field(default_factory=ProcessPolicy)
    landlock: LandlockPolicy = field(default_factory=LandlockPolicy)


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------

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


def _parse_http_rule(raw: dict[str, Any]) -> HttpRule:
    allow = raw.get("allow", raw)
    return HttpRule(
        method=allow.get("method", "*"),
        path=allow.get("path", "/**"),
    )


def _parse_endpoint(ep: dict[str, Any]) -> EndpointRule:
    rules = [_parse_http_rule(r) for r in ep.get("rules", [])]
    return EndpointRule(
        host=ep.get("host", ""),
        port=ep.get("port", 443),
        protocol=ep.get("protocol", "rest"),
        enforcement=ep.get("enforcement", "enforce"),
        tls=ep.get("tls", "terminate"),
        access=ep.get("access", ""),
        rules=rules,
    )


def _parse_binary(b: Any) -> BinarySpec:
    if isinstance(b, dict):
        return BinarySpec(path=b.get("path", ""))
    return BinarySpec(path=str(b))


def _parse_policy_group(key: str, data: dict[str, Any]) -> PolicyGroup:
    endpoints = [_parse_endpoint(ep) for ep in data.get("endpoints", [])]
    binaries = [_parse_binary(b) for b in data.get("binaries", [])]
    return PolicyGroup(
        name=data.get("name", key),
        endpoints=endpoints,
        binaries=binaries,
    )


def _parse_network_policies(raw: Any) -> dict[str, PolicyGroup]:
    """Parse network_policies section -- supports both named-groups and legacy flat format."""
    if not isinstance(raw, dict):
        return {}

    # Legacy flat format detection: has 'allowed_endpoints' key
    if "allowed_endpoints" in raw:
        return _parse_legacy_network(raw)

    # NemoClaw named-groups format: each key is a group name
    groups: dict[str, PolicyGroup] = {}
    for key, val in raw.items():
        # Skip metadata keys that aren't groups
        if key in ("default_action", "allow_localhost"):
            continue
        if isinstance(val, dict):
            groups[key] = _parse_policy_group(key, val)
    return groups


def _parse_legacy_network(raw: dict[str, Any]) -> dict[str, PolicyGroup]:
    """Parse legacy flat allowed_endpoints into a single 'legacy' group."""
    endpoints = [_parse_endpoint(ep) for ep in raw.get("allowed_endpoints", [])]
    if not endpoints:
        return {}
    # Collect all unique binaries across endpoints
    all_bins: list[BinarySpec] = []
    for ep in raw.get("allowed_endpoints", []):
        for b in ep.get("binaries", []):
            spec = _parse_binary(b)
            if spec not in all_bins:
                all_bins.append(spec)
    return {"legacy": PolicyGroup(name="legacy", endpoints=endpoints, binaries=all_bins)}


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

    # Preset metadata wrapper
    preset_meta = data.get("preset", {})
    policy_name = preset_meta.get("name", data.get("name", name))
    description = preset_meta.get("description", data.get("description", ""))

    net_raw = data.get("network_policies", {})
    groups = _parse_network_policies(net_raw)

    # Extract default_action / allow_localhost from either top-level or network section
    default_action = net_raw.get("default_action", data.get("default_action", "deny"))
    allow_localhost = net_raw.get("allow_localhost", data.get("allow_localhost", True))

    return NetworkPolicy(
        name=policy_name,
        description=description,
        default_action=default_action,
        allow_localhost=allow_localhost,
        groups=groups,
    )


def load_sandbox_policy(name: str) -> SandboxPolicy:
    """Load full sandbox policy (network + filesystem + process + landlock)."""
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
        include_workdir=fs_data.get("include_workdir", True),
        read_write=fs_data.get("read_write", ["/sandbox", "/tmp"]),  # nosec B108
        read_only=fs_data.get("read_only", ["/usr", "/lib", "/etc"]),
    )

    # Parse process
    proc_data = data.get("process", {})
    proc = ProcessPolicy(
        run_as_user=proc_data.get("run_as_user", "sandbox"),
        run_as_group=proc_data.get("run_as_group", "sandbox"),
    )

    # Parse landlock
    ll_data = data.get("landlock", {})
    ll = LandlockPolicy(
        compatibility=ll_data.get("compatibility", "best_effort"),
    )

    return SandboxPolicy(
        version=data.get("version", 1),
        network=net_policy or SandboxPolicy().network,
        filesystem=fs,
        process=proc,
        landlock=ll,
    )


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_policies(base: NetworkPolicy, *presets: NetworkPolicy) -> NetworkPolicy:
    """Merge preset policies into a base policy (NemoClaw structured merge).

    Groups are merged by name: preset groups override existing on name collision.
    """
    merged_groups = dict(base.groups)
    for preset in presets:
        for gname, group in preset.groups.items():
            if gname in merged_groups:
                # Merge endpoints and binaries, preset takes precedence
                existing = merged_groups[gname]
                combined_eps = list(existing.endpoints) + list(group.endpoints)
                combined_bins = list(existing.binaries)
                for b in group.binaries:
                    if b not in combined_bins:
                        combined_bins.append(b)
                merged_groups[gname] = PolicyGroup(
                    name=group.name,
                    endpoints=combined_eps,
                    binaries=combined_bins,
                )
            else:
                merged_groups[gname] = group

    return NetworkPolicy(
        name=base.name,
        description=base.description,
        default_action=base.default_action,
        allow_localhost=base.allow_localhost,
        groups=merged_groups,
    )


# ---------------------------------------------------------------------------
# Podman network mode
# ---------------------------------------------------------------------------

def policy_to_podman_args(policy: NetworkPolicy) -> list[str]:
    """Convert policy to podman network arguments.

    Uses pasta mode for all cases (required for port forwarding).
    When deny-all with no endpoints, adds --dns-forward=none to restrict
    outbound to localhost only.
    """
    args = ["--network=pasta"]
    if policy.default_action == "deny" and not policy.groups:
        # True deny-all: block DNS forwarding so only localhost is reachable
        args.append("--dns=none")
    return args


# ---------------------------------------------------------------------------
# Listing and display
# ---------------------------------------------------------------------------

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
            # Support preset metadata wrapper
            preset_meta = data.get("preset", {})
            desc = preset_meta.get("description", data.get("description", ""))
            result.append({
                "name": f.stem,
                "description": desc,
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
    if policy.groups:
        console.print(f"  Policy groups ({len(policy.groups)}):")
        for gname, group in policy.groups.items():
            bins = ", ".join(b.path for b in group.binaries)
            bin_tag = f" (binaries: {bins})" if bins else ""
            console.print(f"    [{gname}]{bin_tag}")
            for ep in group.endpoints:
                access_tag = f" CONNECT" if ep.access == "full" else ""
                rules_tag = ""
                if ep.rules:
                    methods = {r.method for r in ep.rules}
                    rules_tag = f" [{','.join(sorted(methods))}]"
                console.print(
                    f"      - {ep.host}:{ep.port} "
                    f"({ep.protocol}, tls={ep.tls}){access_tag}{rules_tag}"
                )
