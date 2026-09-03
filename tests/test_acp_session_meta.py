"""ACP ``session/new`` extension metadata (``_meta``).

Guards the fix for empty Claude thoughts: the Claude Agent SDK defaults
adaptive thinking to ``display: "omitted"``, so ``agent_thought_chunk``
arrives with empty text and a trajectory records that reasoning happened
without being able to show any of it. BenchFlow opts into summaries through
``session/new``'s ``_meta``, which is registry data rather than a runtime
special case.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchflow.acp.client import ACPClient
from benchflow.agents.registry import AGENTS


class _RecordingTransport:
    """Captures the ``session/new`` payload instead of speaking to an agent."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def start(self) -> None:  # pragma: no cover - not exercised
        return None

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def close(self) -> None:  # pragma: no cover - not exercised
        return None


async def _session_new_payload(**kwargs: Any) -> dict[str, Any]:
    transport = _RecordingTransport()
    client = ACPClient(transport)  # type: ignore[arg-type]

    async def _fake_send_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        transport.sent.append({"method": method, "params": params})
        return {"sessionId": "s-1"}

    client._send_request = _fake_send_request  # type: ignore[assignment]
    await client.session_new(**kwargs)
    return transport.sent[0]["params"]


async def test_session_new_omits_meta_when_the_agent_declares_none() -> None:
    params = await _session_new_payload(cwd="/app")

    assert params["cwd"] == "/app"
    # An absent _meta must stay absent — an empty dict is a different wire
    # message and some agents validate it.
    assert "_meta" not in params


async def test_session_new_forwards_session_meta_verbatim() -> None:
    meta = {"claudeCode": {"options": {"thinking": {"type": "adaptive"}}}}

    params = await _session_new_payload(cwd="/app", session_meta=meta)

    assert params["_meta"] == meta


@pytest.mark.parametrize("agent", ["claude-agent-acp"])
def test_claude_asks_for_summarized_thinking(agent: str) -> None:
    """Without this the trajectory's agent_thought events carry empty text.

    `effort` controls thinking depth and MAX_THINKING_TOKENS its budget;
    neither restores the text, so the display setting is the one that matters.
    """
    thinking = AGENTS[agent].acp_session_meta["claudeCode"]["options"]["thinking"]

    assert thinking["display"] == "summarized"
    # Valid ThinkingConfig variants in @anthropic-ai/claude-agent-sdk 0.3.160.
    assert thinking["type"] in {"adaptive", "enabled"}


def test_agents_without_session_meta_send_nothing() -> None:
    # The field is opt-in per agent; a blanket default would put Claude-shaped
    # options on every harness's session/new.
    assert AGENTS["codex-acp"].acp_session_meta == {}


def _chunk(kind: str, text: str) -> dict[str, Any]:
    return {"sessionUpdate": kind, "content": {"type": "text", "text": text}}


@pytest.mark.parametrize(
    ("kind", "event_type"),
    [
        ("agent_thought_chunk", "agent_thought"),
        ("agent_message_chunk", "agent_message"),
    ],
)
def test_empty_content_chunks_do_not_become_events(kind: str, event_type: str) -> None:
    """A content block opens with an empty chunk before its deltas arrive.

    Recording it produced contentless agent_thought/agent_message events that
    inflated `steps` and `agent_thought_steps` in the run's
    trajectory_summary — a run could report 2 thinking steps with 0 characters
    of thinking.
    """
    from benchflow.acp.session import ACPSession

    session = ACPSession("s-1")
    session.handle_update(_chunk(kind, ""))
    session.handle_update(_chunk(kind, "real "))
    session.handle_update(_chunk(kind, "content"))
    session._flush_agent_text()

    events = [e for e in session.events if e.get("type") == event_type]
    assert len(events) == 1
    assert events[0]["text"] == "real content"
