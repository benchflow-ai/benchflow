"""Behavior tests for the #785 follow-up: wildcard allowlist + LLM-lane.

Driven out one behavior at a time (TDD). Tests exercise public interfaces:
SandboxConfig construction (validation), the egress proxy's host-match, the
egress override builder, and the network-policy resolver.
"""

import pytest
from pydantic import ValidationError

from benchflow.sandbox import _egress_proxy
from benchflow.task.config import AgentConfig, SandboxConfig, VerifierConfig


def test_allowlist_accepts_leading_wildcard():
    cfg = SandboxConfig(network_mode="allowlist", allowed_hosts=["*.example.com"])
    assert "*.example.com" in cfg.allowed_hosts


def test_allowlist_rejects_non_leading_wildcard():
    for bad in ["a.*.com", "*example.com", "example.*", "ex*mple.com"]:
        with pytest.raises(ValidationError):
            SandboxConfig(network_mode="allowlist", allowed_hosts=[bad])


def test_proxy_wildcard_matches_subdomains_not_apex(monkeypatch):
    monkeypatch.setattr(_egress_proxy, "_ALLOWED", ("*.example.com",))
    assert _egress_proxy._host_allowed("a.example.com")
    assert _egress_proxy._host_allowed("x.y.example.com")  # multi-level
    assert not _egress_proxy._host_allowed("example.com")  # apex excluded
    assert not _egress_proxy._host_allowed("evil.com")


def test_proxy_bare_host_matches_apex_and_subdomain(monkeypatch):
    monkeypatch.setattr(_egress_proxy, "_ALLOWED", ("example.com",))
    assert _egress_proxy._host_allowed("example.com")  # apex
    assert _egress_proxy._host_allowed("api.example.com")  # subdomain
    assert not _egress_proxy._host_allowed("notexample.com")


def test_allow_model_endpoint_defaults_true_and_settable():
    assert SandboxConfig().allow_model_endpoint is True
    assert SandboxConfig(allow_model_endpoint=False).allow_model_endpoint is False


def test_proxy_lane_host_allowed_even_with_empty_allowlist(monkeypatch):
    monkeypatch.setattr(_egress_proxy, "_ALLOWED", ())
    monkeypatch.setattr(_egress_proxy, "_LANE", "host.docker.internal", raising=False)
    assert _egress_proxy._host_allowed("host.docker.internal")  # the model lane
    assert not _egress_proxy._host_allowed("evil.com")


def test_egress_override_wires_hostname_model_lane(tmp_path):
    import json

    from benchflow.sandbox._egress import build_egress_override

    # A hostname lane (macOS host.docker.internal) needs the host-gateway route
    # mapped so the sidecar can reach the host-side model proxy.
    path = build_egress_override(
        [], out_dir=tmp_path, model_lane="host.docker.internal"
    )
    proxy = json.loads(path.read_text())["services"]["bf-egress"]
    assert proxy["environment"]["BENCHFLOW_EGRESS_LANE_HOST"] == "host.docker.internal"
    assert "host.docker.internal:host-gateway" in proxy["extra_hosts"]


def test_egress_override_ip_model_lane_needs_no_extra_hosts(tmp_path):
    import json

    from benchflow.sandbox._egress import build_egress_override

    # A bridge-gateway IP lane (Linux) is already routable from the sidecar — no
    # extra_hosts mapping, just the always-allow lane host.
    path = build_egress_override([], out_dir=tmp_path, model_lane="172.17.0.1")
    proxy = json.loads(path.read_text())["services"]["bf-egress"]
    assert proxy["environment"]["BENCHFLOW_EGRESS_LANE_HOST"] == "172.17.0.1"
    assert "extra_hosts" not in proxy


def test_resolve_sets_model_lane_for_restrictive():
    from benchflow.sandbox.network_policy import resolve_network_decision

    assert (
        resolve_network_decision(
            SandboxConfig(network_mode="no-network"), "docker"
        ).model_lane
        is True
    )
    assert (
        resolve_network_decision(
            SandboxConfig(network_mode="allowlist", allowed_hosts=["x.com"]), "docker"
        ).model_lane
        is True
    )
    assert (
        resolve_network_decision(SandboxConfig(), "docker").model_lane is False
    )  # public


def test_lift_agent_network_skips_docker_and_daytona():
    # Docker (model lane) and daytona (fail-closed, no lane) preserve the policy.
    from benchflow.sandbox.setup import _lift_agent_network_to_public

    cfg = SandboxConfig(network_mode="no-network")
    for sb in ("docker", "daytona", "daytona-dind"):
        out = _lift_agent_network_to_public(cfg, sb)
        assert out is cfg  # untouched, not copied
        assert out.network_mode == "no-network"


def test_lift_agent_network_public_for_modal():
    from benchflow.sandbox.setup import _lift_agent_network_to_public

    cfg = SandboxConfig(network_mode="no-network")
    out = _lift_agent_network_to_public(cfg, "modal")
    assert out is not cfg  # copied, original never mutated
    assert out.network_mode == "public"
    assert cfg.network_mode == "no-network"


def test_resolve_model_lane_opt_out():
    from benchflow.sandbox.network_policy import resolve_network_decision

    d = resolve_network_decision(
        SandboxConfig(network_mode="no-network", allow_model_endpoint=False), "docker"
    )
    assert d.model_lane is False


def test_effective_shared_policy_honors_agent_and_verifier_allowlists():
    """Guards PR #785: role network policies must reach the shared sandbox."""
    from benchflow.sandbox.network_policy import effective_shared_network_config

    out = effective_shared_network_config(
        SandboxConfig(),
        AgentConfig(network_mode="allowlist", allowed_hosts=["api.example.com"]),
        VerifierConfig(network_mode="allowlist", allowed_hosts=["verify.example.com"]),
    )

    assert out.network_mode == "allowlist"
    assert out.allowed_hosts == ["api.example.com", "verify.example.com"]


def test_effective_shared_policy_no_network_wins_over_public_environment():
    """Guards PR #785: terminal agent/verifier no-network must not run open."""
    from benchflow.sandbox.network_policy import effective_shared_network_config

    out = effective_shared_network_config(
        SandboxConfig(),
        AgentConfig(network_mode="no-network"),
        VerifierConfig(),
    )

    assert out.network_mode == "no-network"
    assert out.allowed_hosts is None
    assert out.allow_internet is False
