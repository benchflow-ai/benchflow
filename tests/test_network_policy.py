"""network_mode enforcement: policy resolution, egress override, capability gate.

The end-to-end egress proof (allowed host reachable, others blocked, direct
egress blocked) runs as a Docker integration test on a host with docker; these
unit tests pin the resolution logic, the generated compose override shape, the
proxy host-matching, and the preflight capability gate.
"""

import json

from benchflow.sandbox import _egress_proxy
from benchflow.sandbox._egress import build_egress_override
from benchflow.sandbox.network_policy import (
    EffectivePolicy,
    resolve_network_decision,
    resolve_network_mode,
    sandbox_supports_allowlist,
)
from benchflow.task.config import NetworkMode, SandboxConfig


def _cfg(**kw) -> SandboxConfig:
    return SandboxConfig(**kw)


class TestResolveMode:
    def test_public_default(self):
        assert resolve_network_mode(_cfg()) is NetworkMode.PUBLIC

    def test_no_network_explicit(self):
        assert (
            resolve_network_mode(_cfg(network_mode="no-network"))
            is NetworkMode.NO_NETWORK
        )

    def test_allowlist(self):
        c = _cfg(network_mode="allowlist", allowed_hosts=["example.com"])
        assert resolve_network_mode(c) is NetworkMode.ALLOWLIST

    def test_deprecated_allow_internet_false_forces_no_network(self):
        # config reconciliation already does this for the public default; the
        # resolver must agree even if a caller mutates allow_internet post-init.
        c = _cfg()
        c.allow_internet = False
        assert resolve_network_mode(c) is NetworkMode.NO_NETWORK


class TestResolveDecision:
    def test_docker_allowlist_enforced(self):
        c = _cfg(network_mode="allowlist", allowed_hosts=["a.com", "b.com"])
        d = resolve_network_decision(c, "docker")
        assert d.policy is EffectivePolicy.ALLOWLIST
        assert d.allowed_hosts == ("a.com", "b.com")
        assert d.downgraded_from is None

    def test_non_docker_allowlist_fails_closed(self):
        c = _cfg(network_mode="allowlist", allowed_hosts=["a.com"])
        for sb in ("daytona", "modal"):
            d = resolve_network_decision(c, sb)
            assert d.policy is EffectivePolicy.BLOCK_ALL
            assert d.downgraded_from is NetworkMode.ALLOWLIST  # never silently open

    def test_no_network(self):
        assert (
            resolve_network_decision(_cfg(network_mode="no-network"), "docker").policy
            is EffectivePolicy.BLOCK_ALL
        )

    def test_public_open(self):
        assert resolve_network_decision(_cfg(), "docker").policy is EffectivePolicy.OPEN

    def test_capability_predicate(self):
        assert sandbox_supports_allowlist("docker")
        assert not sandbox_supports_allowlist("daytona")
        assert not sandbox_supports_allowlist("modal")
        assert not sandbox_supports_allowlist(None)


class TestEgressOverride:
    def test_structure(self, tmp_path):
        path = build_egress_override(["example.com", "api.test.org"], out_dir=tmp_path)
        doc = json.loads(path.read_text())
        main = doc["services"]["main"]
        proxy = doc["services"]["bf-egress"]
        # main detached from default bridge → only the internal network
        assert main["networks"] == ["bf_egress_internal"]
        assert main["environment"]["HTTPS_PROXY"].startswith("http://bf-egress:")
        assert main["depends_on"] == ["bf-egress"]
        # proxy bridges internal + external and carries the allowlist
        assert set(proxy["networks"]) == {"bf_egress_internal", "bf_egress_external"}
        assert proxy["environment"]["ALLOWED_HOSTS"] == "example.com,api.test.org"
        assert doc["networks"]["bf_egress_internal"]["internal"] is True
        assert "internal" not in doc["networks"]["bf_egress_external"]
        # the proxy script is staged next to the override for bind-mounting
        assert (tmp_path / "egress_proxy.py").exists()


class TestProxyHostMatching:
    def test_match(self, monkeypatch):
        monkeypatch.setattr(_egress_proxy, "_ALLOWED", ("example.com", "test.org"))
        assert _egress_proxy._host_allowed("example.com")
        assert _egress_proxy._host_allowed("api.example.com")  # subdomain
        assert _egress_proxy._host_allowed("EXAMPLE.COM")  # case-insensitive
        assert _egress_proxy._host_allowed("example.com.")  # trailing dot
        assert not _egress_proxy._host_allowed("notexample.com")  # not a subdomain
        assert not _egress_proxy._host_allowed("evil.com")
        assert not _egress_proxy._host_allowed("example.com.evil.com")

    def test_empty_allowlist_denies_all(self, monkeypatch):
        monkeypatch.setattr(_egress_proxy, "_ALLOWED", ())
        assert not _egress_proxy._host_allowed("example.com")


class TestCapabilityGate:
    def test_allowlist_rejected_off_docker_accepted_on_docker(self):
        from benchflow.task.config import TaskConfig
        from benchflow.task.runtime_capabilities import validate_task_runtime_support

        cfg = TaskConfig(
            schema_version="1.3",
            environment={"network_mode": "allowlist", "allowed_hosts": ["example.com"]},
        )
        docker_issues = [
            i
            for i in validate_task_runtime_support(cfg, sandbox="docker")
            if "network_mode" in i.path
        ]
        daytona_issues = [
            i
            for i in validate_task_runtime_support(cfg, sandbox="daytona")
            if "network_mode" in i.path
        ]
        assert docker_issues == []  # enforced on docker
        assert daytona_issues  # rejected at preflight elsewhere
