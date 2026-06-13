"""Guards #382: session/request_permission must dispatch through Session.on_ask_user.

Before the fix, ``ACPClient._handle_agent_request`` handled
``session/request_permission`` internally — picking the most permissive
option and sending the response itself. The ``ACPSessionAdapter`` stored
the handler the kernel passed to ``Session.on_ask_user`` but never wired
it into the live request path, so rollout-branching policies and
NudgeBench-style "should the agent ask?" rewards saw no events.

These tests pin the dispatch path end-to-end:

* the architecture-level handler runs when the agent issues
  ``session/request_permission`` and its return value drives the wire
  response;
* the no-handler fallback still auto-approves with the
  most-permissive-option policy the benchmark relied on before #382;
* the bridge surfaces the option IDs to the handler so it can branch on
  them.
"""

import asyncio
import json

import pytest

from benchflow.acp.client import ACPClient, _auto_approve_option_id
from benchflow.agents.protocol import ACPSessionAdapter, AskUserRequest


class _FakeTransport:
    """Minimal stand-in for ACP transport — captures outbound messages."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, msg: dict) -> None:
        self.sent.append(msg)


def _make_client() -> ACPClient:
    client = ACPClient.__new__(ACPClient)
    client._transport = _FakeTransport()
    client._ask_user_handler = None
    return client


def _request(req_id: int, options: list[dict]) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "session/request_permission",
        "params": {"sessionId": "s1", "options": options},
    }


@pytest.mark.asyncio
async def test_request_permission_invokes_session_on_ask_user_handler():
    """A handler registered through ``Session.on_ask_user`` runs and drives the response."""
    client = _make_client()
    adapter = ACPSessionAdapter(client)

    received: list[AskUserRequest] = []

    async def handler(request: AskUserRequest) -> str:
        received.append(request)
        return "deny"

    adapter.on_ask_user(handler)

    await client._handle_agent_request(
        _request(
            123,
            [
                {"optionId": "allow_once", "kind": "allow_once"},
                {"optionId": "deny", "kind": "reject"},
                {"optionId": "bypassPermissions", "kind": "allow_always"},
            ],
        )
    )

    # The handler ran, was given the enumerated options as the branchable set,
    # and its return value drove the outcome.optionId on the wire.
    assert len(received) == 1
    assert received[0].options == ["allow_once", "deny", "bypassPermissions"]
    assert received[0].option_kinds == {
        "allow_once": "allow_once",
        "deny": "reject",
        "bypassPermissions": "allow_always",
    }

    sent = client._transport.sent
    assert len(sent) == 1
    assert sent[0]["id"] == 123
    assert sent[0]["result"] == {"outcome": {"outcome": "selected", "optionId": "deny"}}


@pytest.mark.asyncio
async def test_request_permission_falls_back_to_auto_approve_when_no_handler():
    """No registered handler ⇒ keep the benchmark-mode auto-approve default."""
    client = _make_client()

    await client._handle_agent_request(
        _request(
            124,
            [
                {"optionId": "allow_once", "kind": "allow_once"},
                {"optionId": "bypassPermissions", "kind": "allow_always"},
            ],
        )
    )

    sent = client._transport.sent
    assert len(sent) == 1
    # The hard-coded "most permissive option" policy still wins when no
    # handler is registered — pre-#382 callers must keep working.
    assert sent[0]["result"]["outcome"]["optionId"] == "bypassPermissions"


@pytest.mark.asyncio
async def test_request_permission_handler_exception_falls_back_to_auto_approve():
    """A misbehaving handler must not deadlock the agent — fall back, log."""
    client = _make_client()
    adapter = ACPSessionAdapter(client)

    async def handler(_request: AskUserRequest) -> str:
        raise RuntimeError("policy error")

    adapter.on_ask_user(handler)

    await client._handle_agent_request(
        _request(
            125,
            [{"optionId": "allow_once", "kind": "allow_once"}],
        )
    )

    sent = client._transport.sent
    assert len(sent) == 1
    # Falls back to the auto-approve policy so the rollout keeps moving.
    assert sent[0]["result"]["outcome"]["optionId"] == "allow_once"


@pytest.mark.asyncio
async def test_acp_client_on_ask_user_overrides_default_policy():
    """Registering directly on the ACP client (lower-level path) also works."""
    client = _make_client()

    async def handler(params: dict) -> str:
        # The lower-level handler gets the raw ACP params dict.
        assert params["sessionId"] == "s1"
        return "allow_once"

    client.on_ask_user(handler)

    await client._handle_agent_request(
        _request(
            126,
            [
                {"optionId": "allow_once", "kind": "allow_once"},
                {"optionId": "bypassPermissions", "kind": "allow_always"},
            ],
        )
    )

    sent = client._transport.sent
    assert sent[0]["result"]["outcome"]["optionId"] == "allow_once"


@pytest.mark.asyncio
async def test_acp_client_on_ask_user_can_be_cleared():
    """Passing ``None`` clears the handler and restores the default policy."""
    client = _make_client()

    async def handler(_params: dict) -> str:
        return "deny"

    client.on_ask_user(handler)
    client.on_ask_user(None)

    await client._handle_agent_request(
        _request(
            127,
            [
                {"optionId": "allow_once", "kind": "allow_once"},
                {"optionId": "bypassPermissions", "kind": "allow_always"},
            ],
        )
    )

    sent = client._transport.sent
    # After clearing, the most-permissive policy is back in charge.
    assert sent[0]["result"]["outcome"]["optionId"] == "bypassPermissions"


def test_auto_approve_option_id_preserves_legacy_priority_order():
    """The default policy is the same one the client used pre-#382.

    Priority: bypassPermissions ▸ allow_always ▸ allow_once ▸ first option.
    Locking the rule in a unit test guards against accidental drift while
    we route the live path through ``on_ask_user``.
    """
    # bypassPermissions wins outright.
    assert (
        _auto_approve_option_id(
            [
                {"optionId": "allow_once", "kind": "allow_once"},
                {"optionId": "bypassPermissions", "kind": "allow_always"},
            ]
        )
        == "bypassPermissions"
    )
    # allow_always wins over allow_once.
    assert (
        _auto_approve_option_id(
            [
                {"optionId": "ao", "kind": "allow_once"},
                {"optionId": "aa", "kind": "allow_always"},
            ]
        )
        == "aa"
    )
    # allow_once wins over a no-kind first option.
    assert (
        _auto_approve_option_id(
            [
                {"optionId": "first"},
                {"optionId": "second", "kind": "allow_once"},
            ]
        )
        == "second"
    )
    # No recognised kinds — fall back to first option's id.
    assert _auto_approve_option_id([{"optionId": "only", "kind": "reject"}]) == "only"
    # Empty options — sentinel.
    assert _auto_approve_option_id([]) == "default"


@pytest.mark.asyncio
async def test_repro_from_issue_382():
    """Pin the exact reproduction from the issue body — handler must be called."""
    client = _make_client()
    adapter = ACPSessionAdapter(client)

    calls: list[AskUserRequest] = []

    async def handler(req: AskUserRequest) -> str:
        calls.append(req)
        return "deny"

    adapter.on_ask_user(handler)

    await client._handle_agent_request(
        {
            "jsonrpc": "2.0",
            "id": 123,
            "method": "session/request_permission",
            "params": {
                "options": [
                    {"optionId": "allow_once", "kind": "allow_once"},
                    {"optionId": "bypassPermissions", "kind": "allow_always"},
                ]
            },
        }
    )

    # Pre-#382: handler_calls == 0 and option_id == "bypassPermissions".
    # Post-#382: handler runs and option_id reflects its return.
    assert len(calls) == 1
    sent = client._transport.sent
    assert sent[0]["result"]["outcome"]["optionId"] == "deny"
    # Sanity: the response is valid JSON-RPC and correlates with the request id.
    assert json.loads(json.dumps(sent[0]))["id"] == 123


def test_synchronous_dispatch_with_event_loop():
    """Same dispatch under ``asyncio.run`` — guards against loop-binding regressions."""
    client = _make_client()
    adapter = ACPSessionAdapter(client)

    async def handler(_req: AskUserRequest) -> str:
        return "allow_once"

    adapter.on_ask_user(handler)

    asyncio.run(
        client._handle_agent_request(
            _request(
                200,
                [{"optionId": "allow_once", "kind": "allow_once"}],
            )
        )
    )

    assert client._transport.sent[0]["result"]["outcome"]["optionId"] == "allow_once"
