"""Gated live guard for the pinned claude-agent-acp config-option contract.

Skipped by default. Run with ``RUN_ACP_DEP_GUARD=1`` (needs ``npm`` + ``node`` +
network):

    RUN_ACP_DEP_GUARD=1 uv run --extra dev python -m pytest \
        tests/test_acp_pinned_protocol_guard.py -q

It installs the exact package selected by ``benchflow.agents.registry``, starts
it over ACP stdio, and proves the complete Fable model + effort path works:
``initialize``, ``session/new``, and both ``session/set_config_option`` calls.
The adapter exposes and accepts these options without auth, so no credentials
are needed. Re-run when bumping the ``@agentclientprotocol`` pin.
"""

import asyncio
import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow.agents.registry import _CLAUDE_AGENT_ACP_PACKAGE

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ACP_DEP_GUARD") != "1",
    reason="gated live ACP guard; set RUN_ACP_DEP_GUARD=1 (needs npm + node + network)",
)

EXPECTED_OPTION_IDS = {"model", "effort"}
FABLE_MODEL = "claude-fable-5-1"
FABLE_EFFORT = "xhigh"


def _tool_or_skip(name: str) -> str:
    path = shutil.which(name)
    if not path:
        pytest.skip(f"{name} not available")
    return path


async def _exercise_config_options(entry: Path) -> tuple[set[str], dict[str, str]]:
    from benchflow.acp.client import ACPClient
    from benchflow.acp.transport import StdioTransport

    client = ACPClient(StdioTransport("node", [str(entry)], env={}, cwd="/tmp"))
    try:
        await client.connect()
        await asyncio.wait_for(client.initialize(), timeout=60)
        await asyncio.wait_for(client.session_new(cwd="/tmp"), timeout=90)
        opts = client.session.config_options or []
        ids = {
            o["id"]
            for o in opts
            if isinstance(o, dict) and isinstance(o.get("id"), str)
        }
        if ids >= EXPECTED_OPTION_IDS:
            await asyncio.wait_for(
                client.set_config_option("model", FABLE_MODEL), timeout=60
            )
            await asyncio.wait_for(
                client.set_config_option("effort", FABLE_EFFORT), timeout=60
            )
        current = {
            o["id"]: o["currentValue"]
            for o in client.session.config_options or []
            if isinstance(o, dict)
            and o.get("id") in EXPECTED_OPTION_IDS
            and isinstance(o.get("currentValue"), str)
        }
        return ids, current
    finally:
        with contextlib.suppress(Exception):
            await client.close()


def test_pinned_claude_acp_supports_fable_model_and_effort(tmp_path):
    """Guards PR #1086's Fable-compatible adapter and ACP config contract."""
    npm = _tool_or_skip("npm")
    _tool_or_skip("node")
    prefix = tmp_path / "claude"
    prefix.mkdir()
    subprocess.run(
        [npm, "install", "--prefix", str(prefix), _CLAUDE_AGENT_ACP_PACKAGE],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    entry = (
        prefix
        / "node_modules"
        / "@agentclientprotocol"
        / "claude-agent-acp"
        / "dist"
        / "index.js"
    )
    assert entry.is_file(), f"pinned agent entry not found: {entry}"

    ids, current = asyncio.run(_exercise_config_options(entry))
    missing = EXPECTED_OPTION_IDS - ids
    assert not missing, (
        f"pinned {_CLAUDE_AGENT_ACP_PACKAGE} no longer advertises config option(s) "
        f"{sorted(missing)!r} (advertised: {sorted(ids)!r}); the registry "
        f"model/effort wiring is stale — re-verify acp_model_config_id / "
        f"acp_effort_config_id"
    )
    assert current.get("model", "").split("[", 1)[0] == FABLE_MODEL, current
    assert current.get("effort") == FABLE_EFFORT, current
