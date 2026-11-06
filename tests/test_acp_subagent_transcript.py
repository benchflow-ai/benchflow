"""Transcript regressions grounded in claude-agent-acp 0.73.0 (unchanged in 0.81.2).

The legacy transcript extension uses one ACP session and explicit
_meta.claudeCode.parentToolUseId. It does not enable native child sessions.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.runtime import connect_acp
from benchflow.acp.session import ACPSession
from benchflow.acp.transport import Transport
from benchflow.acp.types import ACP_PROTOCOL_VERSION
from benchflow.agents.registry import AGENTS, _acpx_wrap
from benchflow.trajectories._capture import (
    TrajectoryWriter,
    _capture_session_trajectory,
    _snapshot_session_trajectory,
)


def chunk(text, parent=None, thought=False):
    update = {
        "sessionUpdate": "agent_thought_chunk" if thought else "agent_message_chunk",
        "content": {"type": "text", "text": text},
    }
    if parent:
        update["_meta"] = {"claudeCode": {"parentToolUseId": parent}}
    return update


@pytest.mark.parametrize("thought", [False, True])
def test_interleaved_child_text_is_attributed_not_merged_into_parent(thought, tmp_path):
    """Extends the child attribution fix: producer child text must not contaminate root answer."""
    session = ACPSession("root")
    for text, parent in [
        ("root-a", None),
        ("child-a", "spawn-1"),
        ("child-b", "spawn-2"),
        ("-tail", "spawn-2"),
        ("root-b", None),
    ]:
        session.handle_update(chunk(text, parent, thought))
    live = _snapshot_session_trajectory(session)
    assert [e["text"] for e in live] == ["root-a", "child-a", "child-b-tail", "root-b"]
    assert [e.get("parent_tool_call_id") for e in live] == [
        None,
        "spawn-1",
        "spawn-2",
        None,
    ]
    assert (session.full_thought if thought else session.full_message) == "root-aroot-b"
    # All five observations still count for the existing idle watchdog.
    assert len(session.thought_chunks if thought else session.message_chunks) == 5
    final = _capture_session_trajectory(session)
    assert live == final == _snapshot_session_trajectory(session)
    writer = TrajectoryWriter(tmp_path / "trace.jsonl")
    writer.write_events(final)
    assert [json.loads(s) for s in writer.path.read_text().splitlines()] == final


@pytest.mark.parametrize("enabled", [False, True])
async def test_initialize_advertises_only_legacy_transcript_extension(enabled):
    """Extends the child attribution fix: exact 0.73 initialize gate, never native subagents."""
    client = ACPClient(None, subagent_transcript=enabled)
    captured = {}

    async def request(method, params):
        captured.update(params)
        return {"protocolVersion": ACP_PROTOCOL_VERSION, "agentCapabilities": {}}

    client._send_request = request
    await client.initialize()
    caps = captured["clientCapabilities"]
    assert caps.get("_meta", {}).get("subagent-transcript", False) is enabled
    assert "subagents" not in caps
    assert not caps["terminal"]


def test_transcript_capability_is_direct_claude_only():
    """Extends the child attribution fix: unverified harnesses/wrappers must not opt in."""
    assert AGENTS["claude-agent-acp"].acp_subagent_transcript is True
    assert AGENTS["codex-acp"].acp_subagent_transcript is False
    assert _acpx_wrap(AGENTS["claude-agent-acp"]).acp_subagent_transcript is False


async def test_session_ids_isolate_notifications_but_accept_legacy_missing_id():
    """Extends the child attribution fix: same-session extension must not mix other sessions."""
    client = ACPClient(None)
    client._session = ACPSession("root")
    for session_id, text in [("foreign", "drop"), ("root", "keep"), (None, "legacy")]:
        params = {"update": chunk(text, "spawn")}
        if session_id is not None:
            params["sessionId"] = session_id
        await client._handle_notification(
            {"method": "session/update", "params": params}
        )
    assert [e["text"] for e in _capture_session_trajectory(client._session)] == [
        "keeplegacy"
    ]


@pytest.mark.parametrize("load", [False, True])
async def test_session_handshake_preserves_matching_history_after_id_resolved(load):
    """Extends the child attribution fix: history can arrive before session/new/load response."""
    client = ACPClient(None)
    old = ACPSession("old")
    client._session = old

    async def request(method, params):
        for session_id in ["old", "new"]:
            await client._handle_notification(
                {
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": chunk(session_id, "spawn"),
                    },
                }
            )
        return {"sessionId": "new"}

    client._send_request = request
    session = await (client.session_load("new") if load else client.session_new())
    assert [e["text"] for e in _capture_session_trajectory(session)] == ["new"]
    assert not old.message_chunks


@pytest.mark.parametrize("load", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_handshake_discards_history_without_changing_old_session(
    load, cancel
):
    """Extends the child attribution fix: failed/cancelled replay has no cross-session side effects."""
    client = ACPClient(None)
    old = ACPSession("old")
    old.message_chunks.append("legacy")
    client._session = old

    async def request(method, params):
        await client._handle_notification(
            {
                "method": "session/update",
                "params": {"sessionId": "new", "update": chunk("unadmitted", "spawn")},
            }
        )
        if cancel:
            raise asyncio.CancelledError
        raise ConnectionError("synthetic")

    client._send_request = request
    with pytest.raises(asyncio.CancelledError if cancel else ConnectionError):
        await (client.session_load("new") if load else client.session_new())
    assert client._opening_session_updates is None
    assert client._session is old
    assert old.full_message == "legacy"
    await client._handle_notification(
        {
            "method": "session/update",
            "params": {"sessionId": "old", "update": chunk("-after")},
        }
    )
    assert old.full_message == "legacy-after"


def test_child_text_keeps_legacy_root_chunks_and_redacts_disk(tmp_path, monkeypatch):
    """Extends the child attribution fix: preserve legacy/root text and host receipts, redact children."""
    session = ACPSession("root")
    session.message_chunks.append("legacy root")
    session.handle_update({"sessionUpdate": "tool_call", "toolCallId": "spawn"})
    assert session.full_message == "legacy root"
    clock = iter([100, 101])
    monkeypatch.setattr("benchflow.acp.session.time.time_ns", lambda: next(clock))
    session.handle_update(chunk("ANTHROPIC_API_KEY=child-secret-value", "spawn"))
    session.handle_update(chunk(" trailing", "spawn"))
    assert session.full_message == "legacy root"
    writer = TrajectoryWriter(tmp_path / "trace.jsonl")
    writer.flush(session)
    text = writer.path.read_text()
    assert "child-secret-value" not in text
    child = json.loads(text.splitlines()[-1])
    assert child["parent_tool_call_id"] == "spawn"
    assert child["receipt"]["first_observed_ns"] == "100"
    assert child["receipt"]["last_observed_ns"] == "101"
    assert "usage" not in child


@pytest.mark.parametrize(
    "agent,enabled", [("claude-agent-acp", True), ("codex-acp", False)]
)
async def test_runtime_passes_harness_transcript_capability(agent, enabled, tmp_path):
    """Extends the child attribution fix: actual runtime must apply the per-harness setting."""
    acp = AsyncMock(spec=ACPClient)
    acp.initialize.return_value = SimpleNamespace(agent_info=None)
    acp.session_new.return_value = ACPSession("root")
    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=acp) as factory,
    ):
        await connect_acp(
            env=AsyncMock(),
            agent=agent,
            agent_launch=agent,
            agent_env={},
            sandbox_user=None,
            model=None,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )
    assert factory.call_args.kwargs["subagent_transcript"] is enabled


@pytest.mark.parametrize("load", [False, True])
async def test_handshake_receipts_preserve_arrival_time_and_late_tool_updates(
    load, monkeypatch
):
    """Buffered history is observed on arrival, not when its session ID resolves."""
    clock = [100]
    monkeypatch.setattr("benchflow.acp.session.time.time_ns", lambda: clock[0])
    client = ACPClient(None)

    async def notify(update):
        update = {**update, "receipt": {"first_observed_ns": "forged-update"}}
        await client._handle_notification(
            {
                "method": "session/update",
                "params": {"sessionId": "new", "update": update},
                # Remote receipt claims cannot override local host observation.
                "receipt": {"first_observed_ns": "forged"},
            }
        )

    async def request(*args):
        await notify(chunk("early", "spawn"))
        clock[0] = 101
        await notify({"sessionUpdate": "tool_call", "toolCallId": "tool"})
        clock[0] = 999
        return {"sessionId": "new"}

    client._send_request = request
    session = await (client.session_load("new") if load else client.session_new())
    first = _snapshot_session_trajectory(session)
    assert [event["receipt"]["first_observed_ns"] for event in first] == ["100", "101"]
    assert [event["receipt"]["last_observed_ns"] for event in first] == ["100", "101"]
    clock[0] = 1000
    await notify(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tool",
            "status": "completed",
        }
    )
    final = _capture_session_trajectory(session)
    assert final[0]["receipt"] == first[0]["receipt"]
    assert final[1]["receipt"]["first_observed_ns"] == "101"
    assert final[1]["receipt"]["last_observed_ns"] == "1000"
    assert final == _snapshot_session_trajectory(session)


class _NoSessionIdAgent(Transport):
    """In-memory ACP agent whose ``session/new`` response omits ``sessionId``.

    Its notifications carry the agent's own session ID, which the client never
    learned, so the client runs under its ``"default"`` placeholder.
    """

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[dict] = asyncio.Queue()

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def receive(self) -> dict:
        return await self._inbox.get()

    def _reply(self, request_id, result) -> None:
        self._inbox.put_nowait({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _notify(self, update) -> None:
        self._inbox.put_nowait(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": "agent-internal-7", "update": update},
            }
        )

    async def send(self, message) -> None:
        method, request_id = message.get("method"), message.get("id")
        if method == "initialize":
            self._reply(
                request_id,
                {"protocolVersion": ACP_PROTOCOL_VERSION, "agentCapabilities": {}},
            )
        elif method == "session/new":
            self._notify(chunk("early "))
            self._reply(request_id, {})
        elif method == "session/prompt":
            self._notify(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "t1",
                    "kind": "execute",
                    "title": "ls",
                    "status": "completed",
                }
            )
            self._notify(chunk("answer"))
            self._reply(request_id, {"stopReason": "end_turn"})


async def test_updates_are_kept_when_agent_omits_session_id():
    """Guards the child transcript filter against dropping a whole session.

    With no ``sessionId`` in the session/new response the client falls back to
    a ``"default"`` placeholder; the child transcript filter then dropped every real update as
    "for a different session", leaving an empty trajectory.
    """
    client = ACPClient(_NoSessionIdAgent())
    await client.connect()
    await client.initialize()
    session = await client.session_new()
    session.record_user_prompt("go")
    await client.prompt("go")
    session.mark_prompt_end()
    trajectory = _capture_session_trajectory(session)
    assert [event["type"] for event in trajectory] == [
        "agent_message",
        "user_message",
        "tool_call",
        "agent_message",
    ]
    assert trajectory[2]["tool_call_id"] == "t1"
    assert session.full_message == "early answer"
