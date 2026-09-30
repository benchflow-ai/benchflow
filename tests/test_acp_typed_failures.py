"""The ACP client's error mapping: a spent subscription is a UsageLimitError.

Guards the dx/errors fix for usage limits that surfaced as a generic, retried
``acp_error``. claude-agent-acp 0.81.2 reports a typed failure when the client
advertises ``_meta.jetbrains.air.capabilities = ["sessionFailure"]``: the turn
ends ``end_turn`` with ``_meta.jetbrains.air.sessionFailure`` (see its
src/session-failure-extension.ts; the record shapes below are built the way
``sessionFailureMeta`` and ``turnOutcome`` build them). codex-acp 1.13 and 2.0
answer a spent ChatGPT plan with a -32603 whose ``data`` carries
``codexErrorInfo: "usageLimitExceeded"`` (its src/CodexEventHandler.ts).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import pytest

from benchflow._utils.scoring import classify_error
from benchflow.acp.client import ACPClient, ACPError, acp_error
from benchflow.acp.session import ACPSession
from benchflow.acp.transport import Transport
from benchflow.acp.types import ACP_PROTOCOL_VERSION
from benchflow.agents.errors import UsageLimitError
from benchflow.agents.registry import AGENTS, _acpx_wrap

WEEKLY = "You've hit your weekly limit · resets Oct 2, 4pm (UTC)"


def _failure_response(
    category: str, actions: list[str], title: str, **extra: Any
) -> dict[str, Any]:
    """A prompt response as claude-agent-acp 0.81.2 settles a failed turn."""
    return {
        "stopReason": "end_turn",
        "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
        "_meta": {
            "quota": {"token_count": {"totalTokens": 0}, "model_usage": []},
            "jetbrains": {
                "air": {
                    "version": 1,
                    "sessionFailure": {
                        "id": "prompt-1:error",
                        "revision": 1,
                        "category": category,
                        "severity": "error",
                        "title": title,
                        "actions": actions,
                        **extra,
                    },
                }
            },
        },
    }


class _ScriptedTransport(Transport):
    """Answers each request with the next scripted message; records what was sent."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.sent: list[dict[str, Any]] = []
        self._replies = list(replies)
        self._inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def start(self) -> None:
        pass

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)
        if "id" in message and "method" in message and self._replies:
            reply = self._replies.pop(0)
            for note in reply.pop("_notifications", []):
                await self._inbox.put(note)
            await self._inbox.put({"jsonrpc": "2.0", "id": message["id"], **reply})

    async def receive(self) -> dict[str, Any]:
        return await self._inbox.get()

    async def close(self) -> None:
        pass


async def _client_with_session(replies: list[dict[str, Any]]) -> ACPClient:
    client = ACPClient(_ScriptedTransport(replies), typed_failures=True)
    client._session = ACPSession("s-1")
    return client


@pytest.mark.parametrize(
    ("typed", "transcript"),
    [(False, False), (True, False), (True, True), (False, True)],
)
async def test_initialize_asks_for_typed_failures_only_when_told(typed, transcript):
    client = ACPClient(None, subagent_transcript=transcript, typed_failures=typed)
    captured: dict[str, Any] = {}

    async def request(method, params):
        captured.update(params)
        return {"protocolVersion": ACP_PROTOCOL_VERSION, "agentCapabilities": {}}

    client._send_request = request  # type: ignore[method-assign]
    await client.initialize()
    meta = captured["clientCapabilities"].get("_meta") or {}
    air = meta.get("jetbrains", {}).get("air")
    if typed:
        assert air == {"version": 1, "capabilities": ["sessionFailure"]}
    else:
        assert air is None
    assert meta.get("subagent-transcript", False) is transcript


def test_only_the_verified_adapter_asks_for_typed_failures():
    assert AGENTS["claude-agent-acp"].acp_typed_failures is True
    assert AGENTS["codex-acp"].acp_typed_failures is False
    assert _acpx_wrap(AGENTS["claude-agent-acp"]).acp_typed_failures is False


async def test_a_spent_subscription_raises_a_usage_limit_error():
    client = await _client_with_session(
        [{"result": _failure_response("limit", [], WEEKLY)}]
    )
    with pytest.raises(UsageLimitError) as caught:
        await client.prompt("hi")
    err = caught.value
    assert err.window == "7-day"
    assert err.resets_at is not None
    assert (err.resets_at.month, err.resets_at.day, err.resets_at.hour) == (10, 2, 16)
    assert classify_error(str(err)) == "usage_limit"


async def test_a_limit_record_with_no_words_is_still_a_usage_limit():
    # A spent quota the SDK names only by its category (e.g. billing_error)
    # carries the adapter's fallback title.
    title = "The Claude account has no available quota."
    client = await _client_with_session(
        [{"result": _failure_response("limit", [], title)}]
    )
    with pytest.raises(UsageLimitError) as caught:
        await client.prompt("hi")
    assert caught.value.detail == title
    assert caught.value.window is None and caught.value.resets_at is None


@pytest.mark.parametrize(
    ("category", "actions", "title", "expected_category"),
    [
        # The agent process died: the same retried acp_error as before.
        (
            "connection",
            ["new_session"],
            "The connection to Claude was lost.",
            "acp_error",
        ),
        # A transient rate limit keeps its provider_rate_limit category.
        (
            "limit",
            ["retry"],
            "API Error: Request rejected (429) · rate limit exceeded",
            "provider_rate_limit",
        ),
        (
            "service",
            ["retry", "new_session"],
            "Claude Agent encountered an internal error.",
            "acp_error",
        ),
        (
            "limit",
            ["new_session"],
            "This Claude turn reached its configured limit.",
            "acp_error",
        ),
    ],
)
async def test_other_failures_raise_the_acp_error_the_agent_raised_before(
    category, actions, title, expected_category
):
    client = await _client_with_session(
        [{"result": _failure_response(category, actions, title)}]
    )
    with pytest.raises(ACPError) as caught:
        await client.prompt("hi")
    err = caught.value
    assert not isinstance(err, UsageLimitError)
    assert err.code == -32603
    assert str(err).startswith(f"ACP error -32603: Internal error: {title.rstrip('.')}")
    assert classify_error(str(err)) == expected_category


async def test_a_lost_agent_process_says_it_was_the_agents_own():
    """Before the opt-in, a killed Claude Code CLI read 'The Claude Agent process
    exited unexpectedly'; the typed record's title only says the connection
    was lost, so the message says whose connection and where to look."""
    title = "The connection to Claude was lost."
    client = await _client_with_session(
        [{"result": _failure_response("connection", ["new_session"], title)}]
    )
    with pytest.raises(ACPError) as caught:
        await client.prompt("hi")
    assert str(caught.value) == (
        "ACP error -32603: Internal error: The connection to Claude was lost: "
        "the agent's own process ended or lost its stream (not BenchFlow's "
        "connection to the sandbox); the agent log in the trial's agent/ folder "
        "says which"
    )


async def test_a_finished_turn_without_a_failure_is_returned():
    ok = {"stopReason": "end_turn", "_meta": {"quota": {"token_count": {}}}}
    client = await _client_with_session([{"result": ok}])
    result = await client.prompt("hi")
    assert result.stop_reason == "end_turn"


async def test_the_json_rpc_usage_limit_is_typed_too():
    # What claude-agent-acp answers a client that did not opt in.
    message = "Internal error: You've hit your weekly limit · resets Oct 3, 7pm (UTC)"
    client = await _client_with_session(
        [
            {
                "error": {
                    "code": -32603,
                    "message": message,
                    "data": {"errorKind": "rate_limit"},
                }
            }
        ]
    )
    with pytest.raises(UsageLimitError) as caught:
        await client.prompt("hi")
    assert caught.value.window == "7-day"


def test_codex_usage_limit_data_is_typed():
    data = {
        "message": (
            "You\u2019ve hit your usage limit. Upgrade to Plus to continue using Codex "
            "(https://chatgpt.com/explore/plus), or try again at Oct 3rd, 2026 5:05 PM."
        ),
        "codexErrorInfo": "usageLimitExceeded",
    }
    err = acp_error(-32603, "Internal error", data)
    assert isinstance(err, UsageLimitError)
    assert err.resets_at == datetime(2026, 10, 3, 17, 5, tzinfo=UTC)
    assert err.detail.startswith("You've hit your usage limit.")
    other = acp_error(
        -32603, "Internal error", {"message": "boom", "codexErrorInfo": "other"}
    )
    assert type(other) is ACPError
    assert str(other) == "ACP error -32603: Internal error"


async def test_a_retry_notice_is_logged_and_an_advisory_stays_in_the_transcript(caplog):
    client = await _client_with_session([])
    session = client.session
    assert session is not None

    def notice(category: str, severity: str, title: str) -> dict[str, Any]:
        return {
            "method": "session/update",
            "params": {
                "sessionId": "s-1",
                "update": {
                    "sessionUpdate": "session_info_update",
                    "_meta": {
                        "jetbrains": {
                            "air": {
                                "version": 1,
                                "sessionFailure": {
                                    "id": "n",
                                    "revision": 1,
                                    "category": category,
                                    "severity": severity,
                                    "title": title,
                                    "actions": [],
                                },
                            }
                        }
                    },
                },
            },
        }

    with caplog.at_level(logging.INFO, logger="benchflow.acp.client"):
        await client._handle_notification(
            notice("service", "warning", "Retrying Claude, attempt 2 of 10.")
        )
        await client._handle_notification(
            notice(
                "unknown",
                "warning",
                "claude-opus-5-5 declined this request; retried with claude-sonnet-4-6.",
            )
        )
    assert "Agent notice: Retrying Claude, attempt 2 of 10." in caplog.text
    assert session.full_message == (
        "**Notice:** claude-opus-5-5 declined this request; retried with "
        "claude-sonnet-4-6.\n\n"
    )
