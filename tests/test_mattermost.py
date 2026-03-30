"""Tests for Mattermost agent integration."""

from unittest.mock import patch

import pytest

from metalclaw.agent import (
    AGENT_TYPES,
    get_agent_config,
    resolve_policy_with_agent,
    _is_mattermost_cloud,
)
from metalclaw.config import DEFAULT_CONFIG
from metalclaw.policy import NetworkPolicy


def _mock_config(**overrides):
    """Build a test config with optional overrides."""
    cfg = {**DEFAULT_CONFIG}
    for k, v in overrides.items():
        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Agent type registration
# ---------------------------------------------------------------------------

def test_mattermost_in_agent_types():
    assert "mattermost" in AGENT_TYPES


def test_mattermost_agent_config():
    """Mattermost agent type loads correct presets."""
    cfg = _mock_config(
        mattermost={"url": "https://chat.example.com", "token": "xoxb-test"},
    )
    with patch("metalclaw.agent.load_config", return_value=cfg):
        ac = get_agent_config("mattermost")
    assert ac.agent_type == "mattermost"
    assert "mattermost" in ac.required_presets
    assert ac.command == ""


def test_mattermost_requires_url():
    """Warns when mattermost.url is empty."""
    cfg = _mock_config(mattermost={"url": "", "token": "tok"})
    with patch("metalclaw.agent.load_config", return_value=cfg), \
         patch("metalclaw.agent.console") as mock_console:
        get_agent_config("mattermost")
    calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("mattermost.url" in c for c in calls)


def test_mattermost_requires_token():
    """Warns when mattermost.token is empty."""
    cfg = _mock_config(mattermost={"url": "https://mm.example.com", "token": ""})
    with patch("metalclaw.agent.load_config", return_value=cfg), \
         patch("metalclaw.agent.console") as mock_console:
        get_agent_config("mattermost")
    calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("mattermost.token" in c for c in calls)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_mattermost_config_defaults():
    """Config section exists with all required fields."""
    mm = DEFAULT_CONFIG["mattermost"]
    assert mm["url"] == ""
    assert mm["token"] == ""
    assert mm["team"] == ""
    assert mm["trigger"] == "@metalclaw"
    assert mm["system_prompt"] == ""
    assert mm["max_history"] == 20
    # ca_cert moved to deploy-level, not in mattermost section
    assert "ca_cert" not in mm


def test_build_config_defaults():
    """Build section exists with configurable build args."""
    build = DEFAULT_CONFIG["build"]
    assert build["base_image"] == "registry.fedoraproject.org/fedora:42"
    assert build["extra_pip_packages"] == []
    assert build["extra_system_packages"] == ""
    assert build["ca_cert_url"] == ""


def test_deploy_config_defaults():
    """Deploy section exists with runtime config."""
    deploy = DEFAULT_CONFIG["deploy"]
    assert deploy["ca_cert"] == ""
    assert deploy["extra_env"] == {}


# ---------------------------------------------------------------------------
# Cloud detection
# ---------------------------------------------------------------------------

def test_is_mattermost_cloud_true():
    assert _is_mattermost_cloud("https://myteam.mattermost.cloud") is True
    assert _is_mattermost_cloud("https://community.mattermost.com") is True


def test_is_mattermost_cloud_false():
    assert _is_mattermost_cloud("https://mattermost.example.com") is False
    assert _is_mattermost_cloud("https://chat.internal.corp") is False


def test_is_mattermost_cloud_empty():
    assert _is_mattermost_cloud("") is False


# ---------------------------------------------------------------------------
# Dynamic policy injection
# ---------------------------------------------------------------------------

def test_mattermost_self_hosted_policy_injection():
    """Self-hosted Mattermost URLs get a dynamic endpoint injected."""
    cfg = _mock_config(
        mattermost={"url": "https://mattermost.example.com", "token": "tok"},
    )
    base = NetworkPolicy(name="base", description="", groups={})
    with patch("metalclaw.agent.load_config", return_value=cfg):
        ac = get_agent_config("mattermost")

    with patch("metalclaw.agent.load_config", return_value=cfg), \
         patch("metalclaw.agent.load_policy", return_value=NetworkPolicy(
             name="mattermost", description="", groups={},
         )):
        result = resolve_policy_with_agent(base, ac)

    assert "mattermost_self_hosted" in result.groups
    ep = result.groups["mattermost_self_hosted"].endpoints[0]
    assert ep.host == "mattermost.example.com"
    assert ep.port == 443


def test_mattermost_self_hosted_custom_port():
    """Self-hosted URL with explicit port."""
    cfg = _mock_config(
        mattermost={"url": "https://mm.corp.io:8443", "token": "tok"},
    )
    base = NetworkPolicy(name="base", description="", groups={})
    with patch("metalclaw.agent.load_config", return_value=cfg):
        ac = get_agent_config("mattermost")

    with patch("metalclaw.agent.load_config", return_value=cfg), \
         patch("metalclaw.agent.load_policy", return_value=NetworkPolicy(
             name="mattermost", description="", groups={},
         )):
        result = resolve_policy_with_agent(base, ac)

    assert "mattermost_self_hosted" in result.groups
    ep = result.groups["mattermost_self_hosted"].endpoints[0]
    assert ep.host == "mm.corp.io"
    assert ep.port == 8443


def test_mattermost_cloud_no_extra_policy():
    """Cloud URLs don't get dynamic injection."""
    cfg = _mock_config(
        mattermost={"url": "https://myteam.mattermost.cloud", "token": "tok"},
    )
    base = NetworkPolicy(name="base", description="", groups={})
    with patch("metalclaw.agent.load_config", return_value=cfg):
        ac = get_agent_config("mattermost")

    with patch("metalclaw.agent.load_config", return_value=cfg), \
         patch("metalclaw.agent.load_policy", return_value=NetworkPolicy(
             name="mattermost", description="", groups={},
         )):
        result = resolve_policy_with_agent(base, ac)

    assert "mattermost_self_hosted" not in result.groups
