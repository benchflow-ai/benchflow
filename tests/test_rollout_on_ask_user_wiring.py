"""Guards #382 follow-up: ACPSessionAdapter must reach the live runtime.

PR #382 made ``ACPSessionAdapter.on_ask_user`` forward into the ACP client,
but never instantiated the adapter inside ``rollout.py`` / ``connect_acp``.
The correctness reviewer found that ``connect_acp`` returned the raw
``ACPSession`` (no adapter) — so the kernel had nowhere to register a
handler against the live wire path. Every rollout therefore kept running
the unconditional auto-approve policy that pre-#382 callers shipped.

These tests pin two boundaries:

* ``connect_acp`` constructs and returns an :class:`ACPSessionAdapter`
  bound to the live :class:`ACPClient`. The :class:`Rollout` stores it as
  ``_session_adapter`` and exposes :meth:`Rollout.on_ask_user`.
* A handler registered through that path actually fires when an agent
  emits ``session/request_permission`` during a real prompt round-trip
  (driven through the existing mock-ACP-agent stdio fixture so the wire
  is real, not a stub).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.transport import StdioTransport
from benchflow.acp.types import StopReason
from benchflow.agents.protocol import ACPSessionAdapter, AskUserRequest
from benchflow.rollout import Rollout

# Re-uses the existing mock agent that emits ``session/request_permission``
# mid-prompt — the same one ``test_acp.py`` drives for the interleaved
# notification + request scenario.
MOCK_AGENT_INTERLEAVED = str(
    Path(__file__).parent / "fixtures" / "mock_acp_agent_interleaved.py"
)


def _bare_rollout() -> Rollout:
    """Build a ``Rollout`` instance without going through ``__init__``.

    The real ``__init__`` requires a sandbox config and resolves agent
    launchers — none of which the on_ask_user wiring touches. ``__new__``
    plus the handful of fields the wiring reads keeps the test focused.
    """
    rollout = Rollout.__new__(Rollout)
    rollout._acp_client = None
    rollout._session = None
    rollout._session_adapter = None
    rollout._ask_user_handler = None
    rollout._ask_user_handler_set = False
    return rollout


@pytest.mark.asyncio
async def test_rollout_on_ask_user_forwards_to_live_adapter() -> None:
    """Registering through ``Rollout.on_ask_user`` reaches the client."""
    rollout = _bare_rollout()

    # Stand-in client + adapter — what ``connect_acp`` would produce.
    client = ACPClient.__new__(ACPClient)
    client._transport = MagicMock()
    client._ask_user_handler = None
    rollout._acp_client = client
    rollout._session_adapter = ACPSessionAdapter(client)

    received: list[AskUserRequest] = []

    async def handler(request: AskUserRequest) -> str:
        received.append(request)
        return "deny"

    rollout.on_ask_user(handler)

    # The bridge installed by ACPSessionAdapter is now on the client.
    assert client._ask_user_handler is not None

    # Drive a permission request through the bridge.
    bridge = client._ask_user_handler
    option_id = await bridge(
        {
            "sessionId": "s1",
            "options": [
                {"optionId": "allow_once", "kind": "allow_once"},
                {"optionId": "deny", "kind": "reject"},
            ],
            "toolCall": {"title": "rm -rf /etc"},
        }
    )

    assert option_id == "deny"
    assert len(received) == 1
    assert received[0].options == ["allow_once", "deny"]
    assert received[0].option_kinds == {
        "allow_once": "allow_once",
        "deny": "reject",
    }
    assert received[0].prompt == "rm -rf /etc"


@pytest.mark.asyncio
async def test_rollout_on_ask_user_handler_persists_across_reconnect() -> None:
    """A handler registered before the first connect must survive reconnects.

    The kernel rebuilds the adapter every time it attaches an agent process
    (``connect()`` / ``_reconnect_for_role()``). Without a sticky handler
    on the rollout, the registered policy would silently revert to
    auto-approve after the first scene boundary.
    """
    rollout = _bare_rollout()

    async def handler(_request: AskUserRequest) -> str:
        return "allow_once"

    # Register before any adapter exists — handler is stored on the rollout.
    rollout.on_ask_user(handler)
    assert rollout._ask_user_handler is handler

    # First "connect" — adapter is built and the handler is reapplied.
    client_a = ACPClient.__new__(ACPClient)
    client_a._transport = MagicMock()
    client_a._ask_user_handler = None
    rollout._acp_client = client_a
    rollout._session_adapter = ACPSessionAdapter(client_a)
    rollout._reapply_ask_user_handler()
    assert client_a._ask_user_handler is not None

    # Second "connect" (reconnect for new role) — fresh client + adapter,
    # the same handler must rebind without the caller doing anything.
    client_b = ACPClient.__new__(ACPClient)
    client_b._transport = MagicMock()
    client_b._ask_user_handler = None
    rollout._acp_client = client_b
    rollout._session_adapter = ACPSessionAdapter(client_b)
    rollout._reapply_ask_user_handler()
    assert client_b._ask_user_handler is not None


@pytest.mark.asyncio
async def test_reapply_is_noop_when_handler_never_registered() -> None:
    """The connect() hot path must not touch the client when no handler is set.

    Without this, reconnect() would synchronously call ``client.on_ask_user(None)``
    on every attach — harmless for real clients but it broke test mocks (the
    benchmark-mode default already wins on the client side, no clearing
    needed).
    """
    rollout = _bare_rollout()
    client = MagicMock()
    adapter = MagicMock()
    rollout._acp_client = client
    rollout._session_adapter = adapter

    rollout._reapply_ask_user_handler()

    client.on_ask_user.assert_not_called()
    adapter.on_ask_user.assert_not_called()


@pytest.mark.asyncio
async def test_rollout_on_ask_user_none_clears_client_handler() -> None:
    """Passing ``None`` to ``on_ask_user`` restores the auto-approve policy."""
    rollout = _bare_rollout()
    client = ACPClient.__new__(ACPClient)
    client._transport = MagicMock()
    client._ask_user_handler = None
    rollout._acp_client = client
    rollout._session_adapter = ACPSessionAdapter(client)

    async def handler(_request: AskUserRequest) -> str:
        return "deny"

    rollout.on_ask_user(handler)
    assert client._ask_user_handler is not None

    rollout.on_ask_user(None)
    assert client._ask_user_handler is None


@pytest.mark.asyncio
async def test_connect_acp_returns_session_adapter_bound_to_client(tmp_path) -> None:
    """``connect_acp`` hands back an adapter wrapping the live ACPClient.

    Without this, every kernel/SDK caller hitting ``Rollout.on_ask_user``
    would still bypass the wire path — exactly the gap PR #382's reviewer
    flagged.
    """
    from benchflow.acp.runtime import connect_acp

    # Mock the transport-level dependencies — we only need to verify the
    # return shape and that the adapter forwards into the returned client.
    mock_session = MagicMock()
    mock_session.session_id = "s1"
    mock_init = MagicMock()
    mock_init.agent_info = None

    mock_acp = AsyncMock(spec=ACPClient)
    mock_acp.connect = AsyncMock()
    mock_acp.initialize = AsyncMock(return_value=mock_init)
    mock_acp.session_new = AsyncMock(return_value=mock_session)
    mock_acp.set_model = AsyncMock()
    mock_acp.close = AsyncMock()

    captured: dict[str, Any] = {}

    def _capture_handler(handler: Any) -> None:
        captured["handler"] = handler

    mock_acp.on_ask_user = MagicMock(side_effect=_capture_handler)

    from unittest.mock import patch

    mock_env = AsyncMock()
    with (
        patch(
            "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
            return_value=MagicMock(),
        ),
        patch(
            "benchflow.acp.runtime.ContainerTransport",
            return_value=MagicMock(),
        ),
        patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
    ):
        result = await connect_acp(
            env=mock_env,
            agent="test-agent",
            agent_launch="test-agent",
            agent_env={},
            sandbox_user=None,
            model=None,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )

    # The new fourth tuple element is the adapter — wired to the live client.
    assert len(result) == 4
    client, session, adapter, agent_name = result
    assert client is mock_acp
    assert session is mock_session
    assert isinstance(adapter, ACPSessionAdapter)
    assert agent_name == "test-agent"

    # Registering a handler through the adapter must reach the client.
    async def handler(_req: AskUserRequest) -> str:
        return "allow_once"

    adapter.on_ask_user(handler)
    assert "handler" in captured
    assert captured["handler"] is not None


@pytest.mark.asyncio
async def test_handler_fires_on_full_rollout_prompt_roundtrip() -> None:
    """Drives a real prompt through the mock ACP agent fixture end-to-end.

    The mock subprocess emits ``session/update`` → ``session/request_permission``
    → ``session/update`` → final ``session/prompt`` response, exactly the
    sequence a real ACP agent would produce when asking for tool approval
    mid-turn. Wiring is what PR #382's follow-up adds: ``connect_acp``
    style construction of the adapter, then ``Rollout.on_ask_user``
    registration. A handler is registered through that path, and we assert
    it fires *during* the prompt — not via an out-of-band stub.
    """
    client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT_INTERLEAVED]))
    received: list[AskUserRequest] = []

    async def handler(request: AskUserRequest) -> str:
        received.append(request)
        # The handler's return value drives the optionId sent back on the
        # wire — this is the contract pre-fix was silently bypassing.
        return "allow_once"

    try:
        await client.connect()
        await client.initialize()
        await client.session_new()

        # Production wiring — construct the adapter the way connect_acp now
        # does, then register through the Rollout-facing path.
        adapter = ACPSessionAdapter(client)
        rollout = _bare_rollout()
        rollout._acp_client = client
        rollout._session_adapter = adapter
        rollout.on_ask_user(handler)

        # Driving prompt() makes the mock agent emit the permission request
        # mid-turn; the client must dispatch it through our handler before
        # returning the final stop reason.
        result = await client.prompt("Go!")
        assert StopReason(result.stop_reason) == StopReason.END_TURN
    finally:
        await client.close()

    # The handler ran exactly once with the agent-emitted options.
    assert len(received) == 1
    assert received[0].options == ["allow_once"]
