"""Agent runtime configuration for the sandbox."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console

from urllib.parse import urlparse

from metalclaw.config import load_config
from metalclaw.policy import (
    BinarySpec,
    EndpointRule,
    NetworkPolicy,
    PolicyGroup,
    load_policy,
    merge_policies,
)

console = Console()

AGENT_TYPES = ("none", "openclaw", "claude-code", "mattermost", "custom")


@dataclass
class AgentConfig:
    agent_type: str
    command: str
    required_presets: list[str]


def get_agent_config(agent_type: str | None = None,
                     command: str | None = None) -> AgentConfig:
    """Build agent configuration from settings."""
    cfg = load_config()
    agent_type = agent_type or cfg["agent"]["type"]
    command = command or cfg["agent"]["command"]

    if agent_type not in AGENT_TYPES:
        console.print(
            f"[red]Unknown agent type: {agent_type}. "
            f"Valid: {', '.join(AGENT_TYPES)}[/red]"
        )
        agent_type = "none"

    required_presets: list[str] = []

    if agent_type == "openclaw":
        # OpenClaw needs outbound access for its channel adapters
        required_presets.append("openclaw")
        if not command:
            command = ""  # handled by agent-wrapper.sh

    elif agent_type == "claude-code":
        required_presets.append("anthropic")
        if not command:
            command = "claude --model openai/local --api-base http://localhost:8080/v1"

    elif agent_type == "mattermost":
        mm_cfg = cfg.get("mattermost", {})
        if not mm_cfg.get("url"):
            console.print("[yellow]Warning: mattermost.url not configured[/yellow]")
        if not mm_cfg.get("token"):
            console.print("[yellow]Warning: mattermost.token not configured[/yellow]")
        required_presets.append("mattermost")
        if not command:
            command = ""  # handled by agent-wrapper.sh

    return AgentConfig(
        agent_type=agent_type,
        command=command,
        required_presets=required_presets,
    )


def resolve_policy_with_agent(base_policy: NetworkPolicy,
                              agent_cfg: AgentConfig) -> NetworkPolicy:
    """Merge agent-required presets into the base policy."""
    presets = []
    for name in agent_cfg.required_presets:
        p = load_policy(name)
        if p:
            presets.append(p)
        else:
            console.print(f"[yellow]Warning: agent preset '{name}' not found[/yellow]")

    if presets:
        merged = merge_policies(base_policy, *presets)
    else:
        merged = base_policy

    # Dynamic policy injection for self-hosted Mattermost
    if agent_cfg.agent_type == "mattermost":
        cfg = load_config()
        mm_url = cfg.get("mattermost", {}).get("url", "")
        if mm_url and not _is_mattermost_cloud(mm_url):
            merged = _inject_self_hosted_mattermost(merged, mm_url)

    return merged


def _is_mattermost_cloud(url: str) -> bool:
    """Check if URL points to Mattermost Cloud."""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return False
    return host.endswith(".mattermost.cloud") or host == "community.mattermost.com"


def _inject_self_hosted_mattermost(policy: NetworkPolicy, url: str) -> NetworkPolicy:
    """Add a policy group for a self-hosted Mattermost domain."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except Exception:
        return policy

    if not host:
        return policy

    group = PolicyGroup(
        name="mattermost_self_hosted",
        endpoints=[
            EndpointRule(
                host=host,
                port=port,
                protocol="rest",
                enforcement="enforce",
                tls="terminate",
                rules=[],
                access="full",
            ),
        ],
        binaries=[
            BinarySpec(path="/usr/bin/python3"),
            BinarySpec(path="/usr/bin/curl"),
        ],
    )
    self_hosted_policy = NetworkPolicy(
        name="mattermost_self_hosted",
        description=f"Self-hosted Mattermost: {host}",
        groups={"mattermost_self_hosted": group},
    )
    return merge_policies(policy, self_hosted_policy)
