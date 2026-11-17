"""No names under no-network (EN-609, RCI on b8b278706): Docker's embedded
resolver (127.0.0.11, on main's loopback) is answered by dockerd from outside
the namespace and forwards what it cannot answer, so a no-network phase that
admits loopback still resolved any name, a channel out in the names asked.
The rules of every no-network hold refuse it above the loopback ACCEPT, and
count what they refuse. A separate verifier sandbox takes the verifier's own
``[verifier] network_mode``, so a no-network one starts with no network."""

from __future__ import annotations

import os
import subprocess

import pytest

from benchflow.sandbox import verifier_network as vn
from benchflow.task.config import NetworkMode, TaskConfig


def test_every_no_network_hold_refuses_dockers_resolver_above_loopback():
    v4, v6, _ = vn.everyone_rules(
        "no-network", subnets=("172.30.0.0/16",), gateways=("172.30.0.1",)
    )
    assert v4.index(vn.DOCKER_DNS_REFUSAL) < v4.index("-o lo -j ACCEPT")
    assert vn.DOCKER_DNS_REFUSAL not in v6  # an IPv4 address
    assert v4[-1] == v6[-1] == "-j REJECT"


@pytest.mark.parametrize("mode", ["allowlist", "denylist"])
def test_the_host_list_policies_keep_their_dns(mode):
    """An allowlist's names go to the proxy's filter (nat); a denylist lists
    what is blocked, and names resolve as before."""
    v4, _, _ = vn.everyone_rules(mode, subnets=("172.30.0.0/16",))
    assert vn.DOCKER_DNS_REFUSAL not in v4


def test_a_main_with_no_route_refuses_it_too():
    assert vn.NO_ROUTE_RULES[0][0] == vn.DOCKER_DNS_REFUSAL


def test_the_count_includes_what_the_resolver_refusal_refused(tmp_path):
    fake = tmp_path / "iptables"
    fake.write_text(
        "#!/bin/sh\ncat <<'EOF'\n"
        "-P OUTPUT ACCEPT -c 0 0\n"
        "-A OUTPUT -d 127.0.0.11/32 -m comment --comment bf-docker-dns -c 4 228 "
        "-j REJECT --reject-with icmp-port-unreachable\n"
        "-A OUTPUT -o lo -c 8 680 -j ACCEPT\n"
        "-A OUTPUT -d 172.30.0.1/32 -c 9 9 -j REJECT --reject-with icmp-port-unreachable\n"
        "-A OUTPUT -c 3 180 -j REJECT --reject-with icmp-port-unreachable\n"
        "EOF\n"
    )
    fake.chmod(0o755)
    done = subprocess.run(
        ["bash", "-c", vn.count_cmd()],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{tmp_path}:/usr/bin:/bin"},
        timeout=60,
    )
    # The last rule's 3 and the resolver refusal's 4; the gateway's own
    # REJECT is not the policy's count (as before).
    assert "BF verifier-network refused 7 status=0" in done.stdout, done


def test_verifier_service_rules_refuse_it_in_the_verifier_phase():
    from benchflow.sandbox.service_cover import service_rules

    kw = {"metadata": True, "subnets": ("172.30.0.0/16",), "gateways": ("172.30.0.1",)}
    v4, _ = service_rules(egress="closed", refuse_docker_dns=True, **kw)
    assert v4.index(vn.DOCKER_DNS_REFUSAL) < v4.index("-o lo -j ACCEPT")
    start, _ = service_rules(egress="closed", **kw)
    assert vn.DOCKER_DNS_REFUSAL not in start
    opened, _ = service_rules(egress="open", refuse_docker_dns=True, **kw)
    assert vn.DOCKER_DNS_REFUSAL not in opened


def _config(toml: str) -> TaskConfig:
    return TaskConfig.model_validate_toml('version = "1.0"\n' + toml)


def test_a_separate_verifier_sandbox_takes_the_verifiers_own_network():
    """RCI: ``[verifier] network_mode`` with no ``[verifier.sandbox]`` reached
    the in-sandbox verifier only; the separate one copied ``[sandbox]``."""
    from benchflow.task.verifier_sandbox import _verifier_sandbox_config

    config = _config(
        "[environment]\nallow_internet = true\n"
        '[verifier]\nsandbox_mode = "separate"\nnetwork_mode = "no-network"\n'
    )
    sandbox = _verifier_sandbox_config(config)
    assert sandbox.network_mode == NetworkMode.NO_NETWORK
    assert sandbox.allow_internet is False
    assert config.sandbox.allow_internet is True  # the agent's is untouched


def test_an_allowlist_verifier_sandbox_takes_its_own_hosts():
    from benchflow.task.verifier_sandbox import _verifier_sandbox_config

    config = _config(
        '[environment]\nnetwork_mode = "denylist"\nblocked_hosts = ["example.org"]\n'
        '[verifier]\nsandbox_mode = "separate"\nnetwork_mode = "allowlist"\n'
        'allowed_hosts = ["pypi.org"]\n'
    )
    sandbox = _verifier_sandbox_config(config)
    assert sandbox.network_mode == NetworkMode.ALLOWLIST
    assert sandbox.allowed_hosts == ["pypi.org"]
    assert sandbox.blocked_hosts == []


@pytest.mark.parametrize(
    "toml",
    [
        # A declared [verifier.sandbox] is the verifier sandbox's own.
        '[environment]\nallow_internet = true\n[verifier]\nsandbox_mode = "separate"\n'
        'network_mode = "no-network"\n[verifier.sandbox]\nallow_internet = true\n',
        # No verifier network: a fresh copy of [sandbox], as before.
        '[environment]\nallow_internet = true\n[verifier]\nsandbox_mode = "separate"\n',
    ],
    ids=["declared", "unset"],
)
def test_otherwise_the_verifier_sandbox_is_as_before(toml):
    from benchflow.task.verifier_sandbox import _verifier_sandbox_config

    assert _verifier_sandbox_config(_config(toml)).allow_internet is True
