"""The rule ledger: what BenchFlow put in each namespace, for its re-proof (EN-609)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from benchflow.sandbox import rule_ledger as rl
from tests.test_helper_report_lines import (  # noqa: F401
    MAIN,
    ShellDocker,
    _helper,
    stubs,
)


def _rules(**kw):
    base = {
        "where": "main",
        "phase": "verifier",
        "v4": ["-d 192.0.2.0/24 -j ACCEPT", "-j REJECT"],
        "v6": ["-j REJECT"],
        "nat": ["-p udp --dport 53 -j REDIRECT --to-ports 5353"],
    }
    return rl.InstalledRules(**{**base, **kw})


def test_record_replaces_by_namespace_and_phase_and_retire_drops_from_expected():
    env = SimpleNamespace()
    rl.record(env, _rules(phase="agent"))
    rl.record(env, _rules(phase="verifier"))
    rl.record(env, _rules(phase="verifier", v4=["-j REJECT"]))
    assert [(r.phase, len(r.v4)) for r in rl.entries(env)] == [
        ("agent", 2),
        ("verifier", 1),
    ]
    rl.retire(env, "main", "agent")
    assert [r.phase for r in rl.expected(env)] == ["verifier"]
    assert all(r.proven_at for r in rl.entries(env))


def test_the_check_script_is_valid_sh():
    assert subprocess.run(["sh", "-n", "-c", rl.check_cmd(_rules())]).returncode == 0


@pytest.mark.asyncio
async def test_the_check_runs_through_the_helper_and_names_what_is_missing(stubs):  # noqa: F811
    """Real script, stub iptables whose -C fails (the stub finds no rule):
    every rule reads missing, and the report line serves the helper."""
    helper = _helper(ShellDocker(stubs))
    rules = _rules()
    outcome = await helper.installed(MAIN, rl.check_cmd(rules))
    assert outcome is not None, helper.unavailable
    out = outcome[0].stdout
    assert "BF rule-check status=0 missing=4" in out
    assert rl.missing_in(out, rules) == [
        "v4: -d 192.0.2.0/24 -j ACCEPT",
        "v4: -j REJECT",
        "v6: -j REJECT",
        "nat: -p udp --dport 53 -j REDIRECT --to-ports 5353",
    ]


def test_present_rules_are_not_missing():
    out = "BF rule-check present v4 0\nBF rule-check present v4 1\nBF rule-check status=0 missing=0\n"
    assert rl.missing_in(out, _rules()) == []
