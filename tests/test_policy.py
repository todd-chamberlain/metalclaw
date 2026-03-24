"""Tests for policy module."""

from metalclaw.policy import (
    EndpointRule,
    FilesystemPolicy,
    NetworkPolicy,
    ProcessPolicy,
    SandboxPolicy,
    merge_policies,
    policy_to_podman_network,
)


def test_deny_all_policy_maps_to_pasta():
    pol = NetworkPolicy(
        name="test",
        description="",
        default_action="deny",
        allow_localhost=True,
        endpoints=[],
    )
    assert policy_to_podman_network(pol) == "pasta"


def test_policy_with_endpoints_maps_to_pasta():
    pol = NetworkPolicy(
        name="test",
        description="",
        default_action="deny",
        allow_localhost=True,
        endpoints=[
            EndpointRule(host="github.com", ports=[443]),
        ],
    )
    assert policy_to_podman_network(pol) == "pasta"


def test_merge_policies():
    base = NetworkPolicy(
        name="base",
        description="",
        default_action="deny",
        allow_localhost=True,
        endpoints=[],
    )
    preset = NetworkPolicy(
        name="github",
        description="",
        default_action="deny",
        allow_localhost=True,
        endpoints=[
            EndpointRule(host="github.com", ports=[443]),
        ],
    )
    merged = merge_policies(base, preset)
    assert len(merged.endpoints) == 1
    assert merged.endpoints[0].host == "github.com"
    assert merged.default_action == "deny"


def test_merge_multiple_presets():
    base = NetworkPolicy("base", "", "deny", True, [])
    p1 = NetworkPolicy("a", "", "deny", True, [EndpointRule("a.com")])
    p2 = NetworkPolicy("b", "", "deny", True, [EndpointRule("b.com")])
    merged = merge_policies(base, p1, p2)
    assert len(merged.endpoints) == 2
    hosts = {ep.host for ep in merged.endpoints}
    assert hosts == {"a.com", "b.com"}


def test_endpoint_rule_nemoclaw_fields():
    """Endpoint rules support NemoClaw-compatible binary/access/tls fields."""
    ep = EndpointRule(
        host="api.github.com",
        ports=[443],
        binaries=["/usr/bin/git", "/usr/bin/curl"],
        access="read-only",
        tls="terminate",
    )
    assert ep.binaries == ["/usr/bin/git", "/usr/bin/curl"]
    assert ep.access == "read-only"
    assert ep.tls == "terminate"


def test_endpoint_rule_defaults():
    """Default values for NemoClaw fields."""
    ep = EndpointRule(host="example.com")
    assert ep.binaries == []
    assert ep.access == "full"
    assert ep.tls == "passthrough"
    assert ep.ports == [443]
    assert ep.protocol == "tcp"
    assert ep.direction == "outbound"


def test_filesystem_policy_defaults():
    fs = FilesystemPolicy()
    assert fs.read_only_root is True
    assert "/sandbox" in fs.read_write
    assert "/tmp" in fs.read_write


def test_process_policy_defaults():
    proc = ProcessPolicy()
    assert proc.run_as_user == "sandbox"
    assert proc.run_as_group == "sandbox"


def test_sandbox_policy_composition():
    sp = SandboxPolicy(
        network=NetworkPolicy("test", "", "deny", True, []),
        filesystem=FilesystemPolicy(read_only_root=True),
        process=ProcessPolicy(run_as_user="agent"),
    )
    assert sp.network.default_action == "deny"
    assert sp.filesystem.read_only_root is True
    assert sp.process.run_as_user == "agent"
