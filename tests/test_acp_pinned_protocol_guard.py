"""Gated live guards for the pinned ``@agentclientprotocol`` adapters.

Skipped by default. Run with ``RUN_ACP_DEP_GUARD=1`` (needs ``npm`` + ``node`` +
network):

    RUN_ACP_DEP_GUARD=1 uv run --extra dev python -m pytest \
        tests/test_acp_pinned_protocol_guard.py -q

Each guard installs the exact package selected by ``benchflow.agents.registry``
and starts it over ACP stdio. claude-agent-acp: the complete Fable model +
effort path works (``initialize``, ``session/new``, both
``session/set_config_option`` calls). codex-acp: the bundled Codex catalog
lists the current model, so ``session/new`` advertises its ``model[effort]``
ids and ``session/set_model`` accepts the one BenchFlow selects. Neither
adapter needs credentials for this. Re-run when bumping an
``@agentclientprotocol`` pin.
"""

import asyncio
import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow.agents.registry import _CLAUDE_AGENT_ACP_PACKAGE, _CODEX_ACP_PACKAGE

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ACP_DEP_GUARD") != "1",
    reason="gated live ACP guard; set RUN_ACP_DEP_GUARD=1 (needs npm + node + network)",
)

EXPECTED_OPTION_IDS = {"model", "effort"}
FABLE_MODEL = "claude-fable-5-1"
FABLE_EFFORT = "xhigh"
CODEX_MODEL = "gpt-6.1-sol"
CODEX_EFFORT = "max"


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


async def _exercise_codex_model_selection(
    entry: Path, work: Path
) -> tuple[list[str], str]:
    from benchflow.acp.client import ACPClient
    from benchflow.acp.runtime import _select_acp_model_id
    from benchflow.acp.transport import StdioTransport
    from benchflow.agents.codex_config import apply_codex_provider_config

    home = work / "codex-home"
    home.mkdir()
    # The LiteLLM provider shape BenchFlow launches Codex with. Nothing listens
    # on the base URL: session/new and set_model read only the bundled catalog.
    env = {"OPENAI_API_KEY": "guard-placeholder", "CODEX_HOME": str(home)}
    apply_codex_provider_config(
        env,
        base_url="http://127.0.0.1:9/v1",
        model=CODEX_MODEL,
        provider_name="litellm",
    )
    client = ACPClient(StdioTransport("node", [str(entry)], env=env, cwd=str(work)))
    try:
        await client.connect()
        await asyncio.wait_for(client.initialize(), timeout=60)
        session = await asyncio.wait_for(client.session_new(cwd=str(work)), timeout=120)
        available = (session.model_state or {}).get("availableModels") or []
        ids = [
            m["modelId"]
            for m in available
            if isinstance(m, dict) and isinstance(m.get("modelId"), str)
        ]
        model_id = _select_acp_model_id(CODEX_MODEL, "codex-acp", session, CODEX_EFFORT)
        if model_id in ids:
            # An id the catalog lacks is rejected (-32603); the caller reports it.
            await asyncio.wait_for(client.set_model(model_id), timeout=60)
        return ids, model_id
    finally:
        with contextlib.suppress(Exception):
            await client.close()


def test_pinned_codex_acp_knows_the_current_model(tmp_path):
    """Guards the codex-acp 2.0.1 pin: codex 0.156.1 (1.13.1) has no gpt-6.1-sol,
    so it ran on fallback metadata and set_model rejected gpt-6.1-sol[max]."""
    npm = _tool_or_skip("npm")
    _tool_or_skip("node")
    prefix = tmp_path / "codex"
    prefix.mkdir()
    subprocess.run(
        [npm, "install", "--prefix", str(prefix), _CODEX_ACP_PACKAGE],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    entry = (
        prefix
        / "node_modules"
        / "@agentclientprotocol"
        / "codex-acp"
        / "dist"
        / "index.js"
    )
    assert entry.is_file(), f"pinned agent entry not found: {entry}"

    ids, model_id = asyncio.run(_exercise_codex_model_selection(entry, tmp_path))
    wanted = f"{CODEX_MODEL}[{CODEX_EFFORT}]"
    assert wanted in ids, (
        f"pinned {_CODEX_ACP_PACKAGE} does not advertise {wanted!r}; its Codex "
        f"catalog lacks {CODEX_MODEL} (advertised: {sorted(ids)!r})"
    )
    assert model_id == wanted
