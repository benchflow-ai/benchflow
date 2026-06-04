"""Gated live guard: the pinned claude-agent-acp advertises the config option
ids the registry wires up.

Skipped by default. Run with ``RUN_ACP_DEP_GUARD=1`` (needs ``npm`` + ``node`` +
network):

    RUN_ACP_DEP_GUARD=1 uv run --extra dev python -m pytest \
        tests/test_acp_pinned_protocol_guard.py -q

It installs the pinned ``claude-agent-acp@0.40.0``, starts it over ACP stdio,
runs ``initialize`` + ``session/new``, and asserts the advertised config option
ids include ``{"model", "effort"}`` — the ids ``benchflow.agents.registry``
hard-codes for model and reasoning-effort selection. If a future pin keeps
``session/set_config_option`` but renames an id, this fails (a plain SDK method
grep would not). ``session/new`` advertises the options without auth, so no
credentials are needed. Re-run when bumping the ``@agentclientprotocol`` pin.
"""

import asyncio
import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ACP_DEP_GUARD") != "1",
    reason="gated live ACP guard; set RUN_ACP_DEP_GUARD=1 (needs npm + node + network)",
)

PINNED_CLAUDE = "@agentclientprotocol/claude-agent-acp@0.40.0"
EXPECTED_OPTION_IDS = {"model", "effort"}


def _tool_or_skip(name: str) -> str:
    path = shutil.which(name)
    if not path:
        pytest.skip(f"{name} not available")
    return path


async def _advertised_option_ids(entry: Path) -> set[str]:
    from benchflow.acp.client import ACPClient
    from benchflow.acp.transport import StdioTransport

    client = ACPClient(StdioTransport("node", [str(entry)], env={}, cwd="/tmp"))
    try:
        await client.connect()
        await asyncio.wait_for(client.initialize(), timeout=60)
        await asyncio.wait_for(client.session_new(cwd="/tmp"), timeout=90)
        opts = client.session.config_options or []
        return {
            o["id"]
            for o in opts
            if isinstance(o, dict) and isinstance(o.get("id"), str)
        }
    finally:
        with contextlib.suppress(Exception):
            await client.close()


def test_pinned_claude_acp_advertises_model_and_effort_options(tmp_path):
    npm = _tool_or_skip("npm")
    _tool_or_skip("node")
    prefix = tmp_path / "claude"
    prefix.mkdir()
    subprocess.run(
        [npm, "install", "--prefix", str(prefix), PINNED_CLAUDE],
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

    ids = asyncio.run(_advertised_option_ids(entry))
    missing = EXPECTED_OPTION_IDS - ids
    assert not missing, (
        f"pinned {PINNED_CLAUDE} no longer advertises config option(s) "
        f"{sorted(missing)!r} (advertised: {sorted(ids)!r}); the registry "
        f"model/effort wiring is stale — re-verify acp_model_config_id / "
        f"acp_effort_config_id"
    )
