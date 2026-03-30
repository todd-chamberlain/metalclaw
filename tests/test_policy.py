"""Tests for policy module (NemoClaw-compatible named groups schema)."""

import pytest

from metalclaw.policy import (
    BinarySpec,
    EndpointRule,
    FilesystemPolicy,
    HttpRule,
    LandlockPolicy,
    NetworkPolicy,
    PolicyGroup,
    ProcessPolicy,
    SandboxPolicy,
    merge_policies,
    policy_to_podman_args,
)


# ---------------------------------------------------------------------------
# Network mode / podman args
# ---------------------------------------------------------------------------

def test_deny_all_policy_maps_to_pasta_with_dns_none():
    """Deny-all with no groups should block DNS forwarding."""
    pol = NetworkPolicy(
        name="test", description="", default_action="deny",
        allow_localhost=True, groups={},
    )
    args = policy_to_podman_args(pol)
    assert "--network=pasta" in args
    assert "--dns=none" in args


def test_policy_with_groups_maps_to_pasta_without_dns_block():
    """Policy with endpoint groups should allow DNS (for endpoint resolution)."""
    pol = NetworkPolicy(
        name="test", description="", default_action="deny",
        allow_localhost=True,
        groups={
            "github": PolicyGroup(
                name="github",
                endpoints=[EndpointRule(host="github.com", port=443)],
                binaries=[BinarySpec(path="/usr/bin/git")],
            ),
        },
    )
    args = policy_to_podman_args(pol)
    assert "--network=pasta" in args
    assert "--dns=none" not in args


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def test_merge_policies():
    base = NetworkPolicy(name="base", description="", groups={})
    preset = NetworkPolicy(
        name="github", description="",
        groups={
            "github": PolicyGroup(
                name="github",
                endpoints=[EndpointRule(host="github.com")],
                binaries=[BinarySpec(path="/usr/bin/git")],
            ),
        },
    )
    merged = merge_policies(base, preset)
    assert "github" in merged.groups
    assert merged.groups["github"].endpoints[0].host == "github.com"
    assert merged.default_action == "deny"


def test_merge_multiple_presets():
    base = NetworkPolicy(name="base", description="", groups={})
    p1 = NetworkPolicy(
        name="a", description="",
        groups={"a": PolicyGroup(name="a", endpoints=[EndpointRule(host="a.com")])},
    )
    p2 = NetworkPolicy(
        name="b", description="",
        groups={"b": PolicyGroup(name="b", endpoints=[EndpointRule(host="b.com")])},
    )
    merged = merge_policies(base, p1, p2)
    assert len(merged.groups) == 2
    assert set(merged.groups.keys()) == {"a", "b"}


def test_merge_deduplicates_by_group_name():
    """When merging groups with the same name, endpoints combine."""
    base = NetworkPolicy(
        name="base", description="",
        groups={
            "shared": PolicyGroup(
                name="shared",
                endpoints=[EndpointRule(host="a.com")],
                binaries=[BinarySpec(path="/usr/bin/curl")],
            ),
        },
    )
    preset = NetworkPolicy(
        name="preset", description="",
        groups={
            "shared": PolicyGroup(
                name="shared",
                endpoints=[EndpointRule(host="b.com")],
                binaries=[BinarySpec(path="/usr/bin/wget")],
            ),
        },
    )
    merged = merge_policies(base, preset)
    assert len(merged.groups) == 1
    assert len(merged.groups["shared"].endpoints) == 2
    hosts = {ep.host for ep in merged.groups["shared"].endpoints}
    assert hosts == {"a.com", "b.com"}
    # Binaries from both should be present
    bin_paths = {b.path for b in merged.groups["shared"].binaries}
    assert bin_paths == {"/usr/bin/curl", "/usr/bin/wget"}


# ---------------------------------------------------------------------------
# EndpointRule (NemoClaw fields)
# ---------------------------------------------------------------------------

def test_endpoint_rule_nemoclaw_fields():
    """Endpoint rules support NemoClaw-compatible enforcement/tls/rules fields."""
    ep = EndpointRule(
        host="api.github.com",
        port=443,
        protocol="rest",
        enforcement="enforce",
        tls="terminate",
        rules=[
            HttpRule(method="GET", path="/**"),
            HttpRule(method="POST", path="/repos/**"),
        ],
    )
    assert ep.enforcement == "enforce"
    assert ep.tls == "terminate"
    assert len(ep.rules) == 2
    assert ep.rules[0].method == "GET"
    assert ep.rules[1].path == "/repos/**"


def test_endpoint_rule_connect_tunnel():
    """access='full' enables CONNECT tunnel for WebSocket endpoints."""
    ep = EndpointRule(
        host="gateway.discord.gg",
        port=443,
        access="full",
    )
    assert ep.access == "full"


def test_endpoint_rule_defaults():
    """Default values for NemoClaw fields."""
    ep = EndpointRule(host="example.com")
    assert ep.port == 443
    assert ep.protocol == "rest"
    assert ep.enforcement == "enforce"
    assert ep.tls == "terminate"
    assert ep.access == ""
    assert ep.rules == []


def test_http_rule_defaults():
    rule = HttpRule()
    assert rule.method == "*"
    assert rule.path == "/**"


# ---------------------------------------------------------------------------
# Binary spec
# ---------------------------------------------------------------------------

def test_binary_spec():
    b = BinarySpec(path="/usr/bin/git")
    assert b.path == "/usr/bin/git"


def test_binary_spec_glob():
    b = BinarySpec(path="/usr/bin/python3*")
    assert "*" in b.path


# ---------------------------------------------------------------------------
# PolicyGroup
# ---------------------------------------------------------------------------

def test_policy_group():
    g = PolicyGroup(
        name="github",
        endpoints=[EndpointRule(host="github.com")],
        binaries=[BinarySpec(path="/usr/bin/git")],
    )
    assert g.name == "github"
    assert len(g.endpoints) == 1
    assert len(g.binaries) == 1


# ---------------------------------------------------------------------------
# Filesystem / Process / Landlock
# ---------------------------------------------------------------------------

def test_filesystem_policy_defaults():
    fs = FilesystemPolicy()
    assert fs.read_only_root is True
    assert fs.include_workdir is True
    assert "/sandbox" in fs.read_write
    assert "/tmp" in fs.read_write


def test_process_policy_defaults():
    proc = ProcessPolicy()
    assert proc.run_as_user == "sandbox"
    assert proc.run_as_group == "sandbox"


def test_landlock_policy_defaults():
    ll = LandlockPolicy()
    assert ll.compatibility == "best_effort"


# ---------------------------------------------------------------------------
# SandboxPolicy composition
# ---------------------------------------------------------------------------

def test_sandbox_policy_composition():
    sp = SandboxPolicy(
        network=NetworkPolicy("test", "", groups={}),
        filesystem=FilesystemPolicy(read_only_root=True),
        process=ProcessPolicy(run_as_user="agent"),
        landlock=LandlockPolicy(compatibility="strict"),
    )
    assert sp.network.default_action == "deny"
    assert sp.filesystem.read_only_root is True
    assert sp.process.run_as_user == "agent"
    assert sp.landlock.compatibility == "strict"


def test_sandbox_policy_defaults():
    sp = SandboxPolicy()
    assert sp.version == 1
    assert sp.network.name == "default"
    assert sp.filesystem.read_only_root is True
    assert sp.process.run_as_user == "sandbox"
    assert sp.landlock.compatibility == "best_effort"


# ---------------------------------------------------------------------------
# Preset loading
# ---------------------------------------------------------------------------

def test_mattermost_preset_loads():
    """Verify mattermost.yaml parses correctly."""
    from metalclaw.policy import load_policy

    pol = load_policy("mattermost")
    assert pol is not None
    assert pol.name == "mattermost"
    assert "mattermost" in pol.groups
    group = pol.groups["mattermost"]
    hosts = {ep.host for ep in group.endpoints}
    assert "*.mattermost.cloud" in hosts
    assert "community.mattermost.com" in hosts
    # Should have at least one CONNECT endpoint (access=full)
    full_eps = [ep for ep in group.endpoints if ep.access == "full"]
    assert len(full_eps) >= 1
