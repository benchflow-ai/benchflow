"""The firewall rules that confine the oracle's processes (REPORT B10).

They match the oracle only (its supplementary group, or a non-root task
user's uid), so root's other processes keep their network: the egress proxy,
BenchFlow's installs and the verifier.
"""

from __future__ import annotations

import re

import pytest

from benchflow.sandbox.oracle_egress import (
    ORACLE_EGRESS_GID,
    arm_cmd,
    disarm_cmd,
    oracle_firewall_rules,
    owner_match,
)


def _specs(cmd: str) -> list[str]:
    return re.findall(r"-C OUTPUT (.*?) 2>/dev/null", cmd)


@pytest.mark.parametrize("allow", [None, ("10.1.0.0/16", "2001:db8::/32")])
@pytest.mark.parametrize(
    "match", [owner_match(gid=ORACLE_EGRESS_GID), owner_match(uid=1001)]
)
def test_every_rule_names_the_oracle_and_never_root(allow, match):
    """Root's other processes (the egress proxy, BenchFlow's installs, the
    verifier) are never matched: no rule without the oracle's owner match."""
    v4, v6 = oracle_firewall_rules(match, allow)
    for rule in [*v4, *v6]:
        assert match in rule.spec, rule
        assert "--uid-owner 0 " not in rule.spec + " ", rule
    cmd = arm_cmd(match, allow, gid=ORACLE_EGRESS_GID)
    specs = _specs(cmd)
    assert specs and all(match in spec for spec in specs)


@pytest.mark.parametrize("allow", [None, ("10.1.0.0/16",)])
def test_disarm_removes_every_rule_arm_adds(allow):
    match = owner_match(gid=ORACLE_EGRESS_GID)
    armed = sorted(_specs(arm_cmd(match, allow, gid=ORACLE_EGRESS_GID)))
    removed = sorted(
        re.findall(
            r"while \S+ -t \S+ -C OUTPUT (.*?) 2>/dev/null", disarm_cmd(match, allow)
        )
    )
    assert armed == removed


def test_an_allowlist_redirects_the_oracles_dns_and_admits_listed_ranges():
    match = owner_match(gid=ORACLE_EGRESS_GID)
    cmd = arm_cmd(match, ("10.1.0.0/16",), gid=ORACLE_EGRESS_GID)
    assert f"--dport 53 {match} -j REDIRECT" in cmd
    assert f"-d 10.1.0.0/16 {match} -j ACCEPT" in cmd
    assert f"-d 127.0.0.11 {match} -j REJECT" in cmd
    # A denylist leaves DNS alone: the proxy refuses by name.
    assert "REDIRECT" not in arm_cmd(match, None, gid=ORACLE_EGRESS_GID)


def test_arming_refuses_an_image_that_already_uses_the_group():
    cmd = arm_cmd(owner_match(gid=ORACLE_EGRESS_GID), None, gid=ORACLE_EGRESS_GID)
    assert f":{ORACLE_EGRESS_GID}:' /etc/group" in cmd


def test_owner_match_takes_exactly_one_of_group_and_uid():
    with pytest.raises(ValueError):
        owner_match()
    with pytest.raises(ValueError):
        owner_match(gid=1, uid=2)


def test_installing_iptables_leaves_apt_lists_as_found():
    cmd = arm_cmd(owner_match(gid=ORACLE_EGRESS_GID), None, gid=ORACLE_EGRESS_GID)
    install = cmd.index("apt-get install -y -qq iptables")
    restore = cmd.index('[ "$bf_lists" -gt 0 ] || rm -rf /var/lib/apt/lists/*')
    assert cmd.index("bf_lists=$(ls /var/lib/apt/lists") < install < restore
