"""Tests for policy module."""

from metaclaw.policy import (
    EndpointRule,
    NetworkPolicy,
    merge_policies,
    policy_to_podman_network,
)


def test_deny_all_policy_maps_to_network_none():
    pol = NetworkPolicy(
        name="test",
        description="",
        default_action="deny",
        allow_localhost=True,
        endpoints=[],
    )
    assert policy_to_podman_network(pol) == "none"


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
