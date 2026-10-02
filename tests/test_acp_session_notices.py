"""ACP session notices: what the adapter composed is not what the agent said.

codex-acp 2.x and claude-agent-acp 0.81+ send what they compose themselves (a
Codex warning, a model fallback or reroute) as ``notice`` session updates to
a client that declares ``clientCapabilities.session.notices``, and write it
into the agent's reply for any other client. BenchFlow declared neither, so
on a model missing from Codex's catalog a provider that never answered still
left one agent message ("Warning: Model metadata for `gpt-5.4` not found
..."), and the trial was scored 0 as a timeout instead of being an
integration failure.
"""

from __future__ import annotations

from typing import Any

from benchflow.acp.client import ACPClient
from benchflow.acp.session import ACPSession
from benchflow.acp.types import ACP_PROTOCOL_VERSION
from benchflow.integration_health import diagnose
from benchflow.trajectories._capture import (
    _capture_session_trajectory,
    _snapshot_session_trajectory,
)
from benchflow.trajectories.export_atif import acp_events_to_atif_steps
from benchflow.trajectories.viewer.payload import _normalize_steps

CODEX_WARNING = (
    "Model metadata for `gpt-5.4` not found. Defaulting to fallback metadata; "
    "this can degrade performance and cause issues."
)
PROMPT = {"type": "user_message", "text": "Do the task."}
TIMEOUT = {
    "type": "agent_timeout",
    "reason": "idle",
    "timeout_sec": 45.0,
    "pending_tool_call_ids": [],
    "terminal_trajectory_complete": True,
}


def _notice(title: str = CODEX_WARNING, **fields: Any) -> dict[str, Any]:
    return {"sessionUpdate": "notice", "severity": "warning", "title": title, **fields}


def _text(kind: str, text: str) -> dict[str, Any]:
    return {"sessionUpdate": kind, "content": {"type": "text", "text": text}}


async def test_initialize_takes_session_notices():
    client = ACPClient(None)
    captured: dict[str, Any] = {}

    async def request(method, params):
        captured.update(params)
        return {"protocolVersion": ACP_PROTOCOL_VERSION, "agentCapabilities": {}}

    client._send_request = request  # type: ignore[method-assign]
    await client.initialize()
    # Both adapters check for an object here (not true, not a list).
    assert captured["clientCapabilities"]["session"] == {"notices": {}}


def test_a_notice_is_its_own_event_not_the_agents_words():
    session = ACPSession("s")
    session.record_user_prompt("Do the task.")
    session.handle_update(_text("agent_thought_chunk", "Reading the task."))
    session.handle_update(_notice())
    session.handle_update(_text("agent_message_chunk", "Done."))
    assert session.full_message == "Done."
    assert session.message_chunks == ["Done."]
    assert session.thought_chunks == ["Reading the task."]
    live = _snapshot_session_trajectory(session)
    final = _capture_session_trajectory(session)
    assert final == live
    assert [e["type"] for e in final] == [
        "user_message",
        "agent_thought",
        "notice",
        "agent_message",
    ]
    notice = final[2]
    assert {k: v for k, v in notice.items() if k not in ("receipt", "ts")} == {
        "type": "notice",
        "severity": "warning",
        "title": CODEX_WARNING,
    }
    # Timed by its host receipt, like the text events around it.
    assert notice["ts"] and notice["receipt"]["source"] == "benchflow_host"


def test_a_notice_keeps_its_description_and_one_without_a_title_is_dropped():
    session = ACPSession("s")
    session.handle_update({"sessionUpdate": "notice", "severity": "warning"})
    session.handle_update(_notice("  "))
    session.handle_update(
        _notice(
            "Model fallback",
            severity="info",
            description="claude-opus-5-5 declined; retried with claude-sonnet-4-6.",
        )
    )
    events = _capture_session_trajectory(session)
    assert [
        {k: v for k, v in e.items() if k not in ("receipt", "ts")} for e in events
    ] == [
        {
            "type": "notice",
            "severity": "info",
            "title": "Model fallback",
            "description": "claude-opus-5-5 declined; retried with claude-sonnet-4-6.",
        }
    ]


def test_a_silent_provider_behind_a_notice_is_an_integration_failure():
    """The codex-slow-model trajectory on a model missing from Codex's catalog."""
    finding = diagnose(
        [
            PROMPT,
            {"type": "notice", "severity": "warning", "title": CODEX_WARNING},
            TIMEOUT,
        ],
        agent="codex-acp",
        n_tool_calls=0,
        output_tokens=None,
        logs={},
        agent_seconds=45.3,
    )
    assert finding is not None
    assert finding.cause == "no_activity"
    assert finding.evidence == (
        "0 tool calls, 0 messages, 0 thoughts after the prompt in 45.3 s; "
        f"the adapter's notice: {CODEX_WARNING}"
    )


def test_the_same_warning_as_agent_text_reads_as_work():
    """What BenchFlow recorded before it took notices: older results keep it."""
    reply = {"type": "agent_message", "text": f"Warning: {CODEX_WARNING}\n\n"}
    assert (
        diagnose(
            [PROMPT, reply, TIMEOUT],
            agent="codex-acp",
            n_tool_calls=0,
            output_tokens=None,
            logs={},
            agent_seconds=45.3,
        )
        is None
    )


def test_a_notice_can_name_the_cause():
    notice = {
        "type": "notice",
        "severity": "error",
        "title": "Authentication failed",
        "description": "401 Unauthorized: check the API key.",
    }
    finding = diagnose(
        [PROMPT, notice],
        agent="codex-acp",
        n_tool_calls=0,
        output_tokens=None,
        logs={},
        agent_seconds=3.0,
    )
    assert finding is not None
    assert finding.cause == "agent_auth"
    assert finding.evidence_source == "trajectory notice"


def test_atif_exports_a_notice_as_a_system_step():
    steps = acp_events_to_atif_steps(
        [
            PROMPT,
            {"type": "agent_thought", "text": "Reading the task."},
            {"type": "notice", "severity": "warning", "title": CODEX_WARNING},
            {"type": "notice", "severity": "info", "title": " "},
            {"type": "agent_message", "text": "Done."},
        ]
    )
    assert steps == [
        {"step_id": 1, "source": "user", "message": "Do the task."},
        {
            "step_id": 2,
            "source": "agent",
            "message": "",
            "reasoning_content": "Reading the task.",
        },
        {
            "step_id": 3,
            "source": "system",
            "message": CODEX_WARNING,
            "extra": {"notice": {"severity": "warning"}},
        },
        {"step_id": 4, "source": "agent", "message": "Done."},
    ]


def test_the_viewer_shows_a_notice_apart_from_the_agents_words():
    steps = _normalize_steps(
        [
            PROMPT,
            {
                "type": "notice",
                "severity": "warning",
                "title": "Model fallback",
                "description": "Retried with claude-sonnet-4-6.",
                "ts": "2026-10-02T09:00:00+00:00",
            },
        ],
        None,
    )
    payload = steps[1].to_payload()
    assert {k: v for k, v in payload.items() if k != "t"} == {
        "i": 2,
        "kind": "notice",
        "severity": "warning",
        "title": "Model fallback",
        "text": "Model fallback\n\nRetried with claude-sonnet-4-6.",
    }
    assert isinstance(payload["t"], float)
