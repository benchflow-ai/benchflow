"""Branch children can resume the parent's agent conversation.

Every child used to start a fresh agent session,
so a child prompt such as "now finish the task" reached an agent that never
saw the parent's conversation. Agents that keep their session on disk (Claude
Code under ``~/.claude/projects``) have it captured in the sandbox snapshot;
``branch(resume_session=True)`` hands each child the parent's ACP session id
and ``connect()`` opens it with ``session/load`` instead of ``session/new``.
The replayed history is the shared prefix, not the child's continuation, so
it is left out of the child's trajectory. An agent that does not advertise
``loadSession`` is refused rather than silently given a fresh session.

Unit tests against mocks and fakes; no Docker, Daytona or credentials.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.acp.client import ACPClient
from tests.test_branch_isolated import IsoRollout, _root, _tree
from tests.test_branch_restore_parent import _fork, _rollout


def _acp_mock(load_session: bool) -> AsyncMock:
    init = MagicMock()
    init.agent_info = None
    init.agent_capabilities = SimpleNamespace(load_session=load_session)
    client = AsyncMock(spec=ACPClient)
    client.initialize = AsyncMock(return_value=init)
    client.session_new = AsyncMock(return_value=MagicMock(session_id="new"))
    client.session_load = AsyncMock(return_value=MagicMock(session_id="parent-s"))
    return client


async def _connect(tmp_path, client, **kwargs):
    from benchflow.acp.runtime import connect_acp

    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=client),
        patch("benchflow.acp.runtime._configure_acp_session", new_callable=AsyncMock),
        patch(
            "benchflow.acp.runtime.enforce_agent_egress_firewall",
            new_callable=AsyncMock,
        ),
    ):
        return await connect_acp(
            env=AsyncMock(),
            agent="claude-agent-acp",
            agent_launch="claude-agent-acp",
            agent_env={},
            sandbox_user=None,
            model=None,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
            **kwargs,
        )


async def test_connect_loads_the_named_session(tmp_path):
    client = _acp_mock(load_session=True)
    _, session, _, _ = await _connect(tmp_path, client, resume_session_id="parent-s")
    client.session_load.assert_awaited_once()
    assert client.session_load.await_args.args[0] == "parent-s"
    client.session_new.assert_not_awaited()
    assert session.session_id == "parent-s"
    assert isinstance(session.replayed_prefix, tuple)


async def test_an_agent_without_load_session_is_refused(tmp_path):
    client = _acp_mock(load_session=False)
    with pytest.raises(RuntimeError, match="loadSession"):
        await _connect(tmp_path, client, resume_session_id="parent-s")
    client.session_new.assert_not_awaited()


async def test_without_a_resume_id_a_new_session_is_opened(tmp_path):
    client = _acp_mock(load_session=True)
    await _connect(tmp_path, client)
    client.session_new.assert_awaited_once()
    client.session_load.assert_not_awaited()


# ── the branch engine ────────────────────────────────────────────────


async def test_in_place_children_get_the_parent_session_id(tmp_path):
    rollout, _sandbox = _rollout(tmp_path)
    rollout._session = SimpleNamespace(session_id="parent-s")
    seen = []

    async def child(_node):
        seen.append(rollout._resume_session_id)
        return 1.0

    await rollout.branch(2, child, snapshot_layers={"sandbox"}, resume_session=True)
    assert seen == ["parent-s", "parent-s"]
    assert rollout._resume_session_id is None  # the parent opens a new one
    snapshot = _fork(rollout)["snapshot"]
    assert snapshot["agent_session"] == "resumed"
    assert "agent_session" not in snapshot["excluded"]


async def test_isolated_children_get_the_parent_session_id(tmp_path):
    root = await _root(tmp_path)
    root._session = SimpleNamespace(session_id="parent-s")
    seen = []

    async def run(node, *, child):
        seen.append(child.rollout._resume_session_id)
        return 1.0

    await root.branch(
        2, run, snapshot_layers={"sandbox"}, isolate_children=True, resume_session=True
    )
    assert seen == ["parent-s", "parent-s"]
    assert _tree(root)["forks"][0]["snapshot"]["agent_session"] == "resumed"
    assert all(isinstance(sub, IsoRollout) for sub in IsoRollout.all)


async def test_resume_needs_a_parent_session(tmp_path):
    rollout, sandbox = _rollout(tmp_path)

    async def child(_node):
        return 1.0

    with pytest.raises(ValueError, match="no agent session"):
        await rollout.branch(2, child, snapshot_layers={"sandbox"}, resume_session=True)
    assert sandbox.snapshots == []


async def test_default_fork_records_a_fresh_session(tmp_path):
    rollout, _sandbox = _rollout(tmp_path)

    async def child(_node):
        return 1.0

    await rollout.branch(2, child, snapshot_layers={"sandbox"})
    snapshot = _fork(rollout)["snapshot"]
    assert snapshot["agent_session"] == "fresh"
    assert "agent_session" in snapshot["excluded"]
    json.dumps(snapshot)
