"""The network-policy pages say what the code holds (the 0.10.0 candidates said Daytona enforces a denylist while
``NETWORK_POLICY_REFUSALS`` refuses every restricted policy there). Each backend list is derived from the providers'
capability flags minus the refusal table, and compared with the table in docs/tasks/network-and-secrets.md and the
remedies in tools/docs_errors_fixes.tsv (the source of docs/reference/errors.md)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from benchflow._utils import sandbox_providers as sp
from benchflow.task import runtime_capabilities as rc

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "docs" / "tasks" / "network-and-secrets.md"
FIXES = ROOT / "tools" / "docs_errors_fixes.tsv"
SHOWN = {
    "docker": "Docker",
    "remote-docker": "remote Docker",
    "daytona": "Daytona",
    "modal": "Modal",
    "apple-container": "Apple Container",
    "agentcore": "AgentCore",
}
#: network_mode -> (the provider flag, the refusal-table policy for a model agent)
POLICIES = {
    "denylist": ("enforces_denylist", rc.DENYLIST_POLICY),
    "allowlist": ("enforces_allowlist", rc.ALLOWLIST_POLICY),
    "no-network": ("enforces_no_network", rc.NO_NETWORK_AGENT_POLICY),
}


def held(mode: str) -> list[str]:
    """The providers that run a model agent's task under *mode*, in the providers' order."""
    flag, policy = POLICIES[mode]
    return [
        p.name
        for p in sp._PROVIDERS
        if getattr(p, flag) and policy not in rc.NETWORK_POLICY_REFUSALS.get(p.name, ())
    ]


def _table_cell(mode: str) -> str:
    for line in TABLE.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] == f"`{mode}`":
            return cells[2]
    raise AssertionError(f"no row for `{mode}` in {TABLE.name}")


@pytest.mark.parametrize("mode", sorted(POLICIES))
def test_the_table_names_exactly_the_backends_that_hold_the_policy(mode):
    first = _table_cell(mode).split(";")[0]
    named = [
        name
        for name, shown in SHOWN.items()
        if re.search(rf"(?<![A-Za-z]){re.escape(shown)}(?![A-Za-z])", first)
    ]
    assert sorted(named) == sorted(held(mode)), (mode, first)


@pytest.mark.parametrize("mode", sorted(POLICIES))
def test_the_error_fix_names_exactly_the_backends_that_hold_the_policy(mode):
    rows = [
        line.split("\t")
        for line in FIXES.read_text().splitlines()
        if f"network_mode='{mode}' is not enforced by" in line
    ]
    assert len(rows) == 1, mode
    fix = rows[0][-1].split(" which enforce")[0]
    named = re.findall(r"`(?:--sandbox )?([a-z-]+)`", fix)
    assert sorted(named) == sorted(held(mode)), (mode, fix)


def test_daytona_and_modal_refuse_every_restricted_policy():
    """The fact the docs state in words (known issues, network-and-secrets.md)."""
    for backend in ("daytona", "modal"):
        refused = rc.NETWORK_POLICY_REFUSALS[backend]
        assert {
            rc.ALLOWLIST_POLICY,
            rc.DENYLIST_POLICY,
            rc.NO_NETWORK_AGENT_POLICY,
        } <= refused
