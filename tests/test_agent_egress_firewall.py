"""Execute the firewall setup against Linux's real zero-size procfs interface.

Guards against the skipped IPv6 rules present in commit 4196c4a. Firewall
binaries are recording stubs; these tests never alter the host firewall.
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox.egress_denylist import EgressDenylist, denylist_agent_env
from benchflow.sandbox.lockdown import (
    _agent_egress_firewall_cmd,
    enforce_agent_egress_firewall,
)

pytestmark = pytest.mark.skipif(
    not Path("/proc/net/if_inet6").exists(),
    reason="requires the Linux IPv6 procfs interface",
)


@pytest.fixture
def firewall(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "calls"
    commands = {
        "id": "printf '1000\\n'\n",
        "iptables": 'printf "ipv4 %s\\n" "$*" >> "$FIREWALL_LOG"\n[ "$1" != -C ]\n',
        "ip6tables": 'printf "ipv6 %s\\n" "$*" >> "$FIREWALL_LOG"\n[ "$1" != -C ]\n',
    }
    for name, body in commands.items():
        executable = binaries / name
        executable.write_text("#!/bin/sh\n" + body)
        executable.chmod(0o755)
    return binaries, log


def run_firewall(binaries, log, **options):
    return subprocess.run(
        ["/bin/sh", "-c", _agent_egress_firewall_cmd("agent", **options)],
        env={**os.environ, "PATH": str(binaries), "FIREWALL_LOG": str(log)},
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def test_zero_size_procfs_still_installs_ipv6_rules(firewall):
    binaries, log = firewall
    assert Path("/proc/net/if_inet6").stat().st_size == 0

    result = run_firewall(binaries, log)

    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    for family in ("ipv4", "ipv6"):
        assert (
            f"{family} -I OUTPUT 1 -o lo -m owner --uid-owner 1000 -j ACCEPT" in calls
        )
        assert f"{family} -A OUTPUT -m owner --uid-owner 1000 -j REJECT" in calls


def test_ipv6_stack_without_ip6tables_fails_closed(firewall):
    binaries, log = firewall
    (binaries / "ip6tables").unlink()

    result = run_firewall(binaries, log)

    assert result.returncode == 86
    assert "IPv6 enabled but ip6tables unavailable" in result.stderr


def test_ipv6_rule_installation_failure_is_not_ignored(firewall):
    binaries, log = firewall
    (binaries / "ip6tables").write_text("#!/bin/sh\nexit 1\n")

    result = run_firewall(binaries, log)

    assert result.returncode != 0


def _chain(calls, family):
    """The OUTPUT chain the recorded -I/-A calls leave, top first."""
    chain = []
    for call in calls:
        fam, _, args = call.partition(" ")
        if fam != family:
            continue
        if args.startswith("-I OUTPUT 1 "):
            chain.insert(0, args.removeprefix("-I OUTPUT 1 "))
        elif args.startswith("-A OUTPUT "):
            chain.append(args.removeprefix("-A OUTPUT "))
    return chain


def test_an_offline_agents_dns_on_loopback_is_refused_before_loopback_is_accepted(
    firewall,
):
    """Fails on c0d0122e, where an offline agent resolved any name through Docker's resolver at 127.0.0.11."""
    binaries, log = firewall
    assert "--dport 53" not in _agent_egress_firewall_cmd("agent")

    result = run_firewall(binaries, log, refuse_dns=True)

    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    owner = "-m owner --uid-owner 1000"
    v4 = _chain(calls, "ipv4")
    accept = v4.index(f"-o lo {owner} -j ACCEPT")
    for rule in (
        f"-d 127.0.0.11 {owner} -j REJECT",
        f"-o lo -p udp --dport 53 {owner} -j REJECT",
        f"-o lo -p tcp --dport 53 {owner} -j REJECT",
    ):
        assert v4.index(rule) < accept, v4
    v6 = _chain(calls, "ipv6")
    accept6 = v6.index(f"-o lo {owner} -j ACCEPT")
    for proto in ("udp", "tcp"):
        assert v6.index(f"-o lo -p {proto} --dport 53 {owner} -j REJECT") < accept6, v6


@pytest.mark.asyncio
async def test_only_an_offline_agents_firewall_refuses_dns():
    """No-web agents get the DNS refusal (none on c0d0122e); a denylist agent's route is unchanged."""
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
    await enforce_agent_egress_firewall(
        env,
        "agent",
        {
            "BENCHFLOW_DISALLOW_WEB_TOOLS": "1",
            "BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:4000",
        },
    )
    assert "-d 127.0.0.11" in env.exec.await_args.args[0]
    env.exec.reset_mock()
    policy = EgressDenylist((), ("blocked.bench.invalid",))
    await enforce_agent_egress_firewall(env, "agent", denylist_agent_env({}, policy))
    assert "-d 127.0.0.11" not in env.exec.await_args.args[0]
