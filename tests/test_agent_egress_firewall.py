"""Execute the firewall setup against Linux's real zero-size procfs interface.

Guards against the skipped IPv6 rules present in commit 4196c4a. Firewall
binaries are recording stubs; these tests never alter the host firewall.
"""

import os
import subprocess
from pathlib import Path

import pytest

from benchflow.sandbox.lockdown import _agent_egress_firewall_cmd

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


def run_firewall(binaries, log):
    return subprocess.run(
        ["/bin/sh", "-c", _agent_egress_firewall_cmd("agent")],
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
