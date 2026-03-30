"""Agent runtime configuration for the sandbox."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console

from metalclaw.config import load_config
from metalclaw.policy import load_policy, merge_policies, NetworkPolicy

console = Console()

AGENT_TYPES = ("none", "openclaw", "claude-code", "custom")


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
        return merge_policies(base_policy, *presets)
    return base_policy
