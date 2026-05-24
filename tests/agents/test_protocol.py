"""Tests for the Agent plane contract Protocols — ``Agent`` and ``Session``."""

import asyncio

from benchflow.agents.protocol import (
    ACPSessionAdapter,
    Agent,
    AgentCapabilities,
    AskUserRequest,
    Session,
)


def test_session_is_runtime_checkable_protocol():
    """Tracer bullet — a trivial class structurally satisfies ``Session``."""

    class Dummy:
        async def prompt(self, text): ...
        async def cancel(self): ...
        def on_ask_user(self, handler): ...

        @property
        def steps(self): ...

    assert isinstance(Dummy(), Session)


def test_incomplete_class_is_not_a_session():
    """A class missing part of the live surface is not a ``Session``."""

    class Partial:
        async def prompt(self, text): ...

    assert not isinstance(Partial(), Session)


def test_agent_is_runtime_checkable_protocol():
    """A class with ``connect`` + ``capabilities`` structurally satisfies ``Agent``."""

    class Dummy:
        async def connect(self, sandbox, role): ...
        def capabilities(self): ...

    assert isinstance(Dummy(), Agent)


def test_class_missing_connect_is_not_an_agent():
    """``capabilities`` alone is not an ``Agent`` — ``connect`` is the live edge."""

    class Partial:
        def capabilities(self): ...

    assert not isinstance(Partial(), Agent)


def test_agent_capabilities_defaults_to_acp():
    """An ``AgentCapabilities`` with no args describes a default ACP agent."""
    caps = AgentCapabilities()
    assert caps.protocol == "acp"
    assert caps.nudges is True
    assert caps.ask_user is False
    assert caps.token_logprobs is False


def test_agent_capabilities_carries_overrides():
    """Capability flags are explicit, frozen, and round-trip what was set."""
    caps = AgentCapabilities(ask_user=True, token_logprobs=True)
    assert caps.ask_user is True
    assert caps.token_logprobs is True


def test_ask_user_request_carries_enumerated_options():
    """``AskUserRequest`` is the branchable primitive — finite options."""
    req = AskUserRequest(
        prompt="Which branch?",
        options=["main", "dev"],
        request_id="r1",
    )
    assert req.prompt == "Which branch?"
    assert req.options == ["main", "dev"]
    assert req.request_id == "r1"


def test_ask_user_request_options_default_empty():
    """A free-text ask carries no enumerated options."""
    req = AskUserRequest(prompt="What now?")
    assert req.options == []


def test_acp_session_adapter_satisfies_session_protocol():
    """The real ACP stack honours the ``Session`` contract via the adapter.

    The architecture's ``Session`` surface (prompt / cancel / on_ask_user /
    steps) is split across the real classes: ``ACPClient`` owns the live
    verbs, ``ACPSession`` owns accumulated state. ``ACPSessionAdapter`` is
    the thin documented bridge that exposes one ``Session``.
    """
    from benchflow.acp.client import ACPClient
    from benchflow.acp.session import ACPSession

    client = ACPClient.__new__(ACPClient)
    client._session = ACPSession("s1")
    adapter = ACPSessionAdapter(client)

    assert isinstance(adapter, Session)


def test_adapter_prompt_returns_stop_reason():
    """``prompt`` unwraps the ACP ``PromptResult`` to a bare ``StopReason``."""
    from benchflow.acp.client import ACPClient
    from benchflow.acp.types import PromptResult, StopReason

    class FakeClient:
        async def prompt(self, text):
            assert text == "do the task"
            return PromptResult(stopReason=StopReason.END_TURN)

    adapter = ACPSessionAdapter(FakeClient())  # type: ignore[arg-type]
    result = asyncio.run(adapter.prompt("do the task"))
    assert result is StopReason.END_TURN
    # Sanity: a real ACPClient really does expose a coroutine ``prompt``.
    assert asyncio.iscoroutinefunction(ACPClient.prompt)


def test_adapter_steps_reflects_acp_session_events():
    """``steps`` surfaces ``ACPSession.events`` — the session's trajectory."""
    from benchflow.acp.client import ACPClient
    from benchflow.acp.session import ACPSession

    client = ACPClient.__new__(ACPClient)
    session = ACPSession("s1")
    session.record_user_prompt("first instruction")
    client._session = session

    adapter = ACPSessionAdapter(client)
    assert adapter.steps == [{"type": "user_message", "text": "first instruction"}]


def test_adapter_steps_empty_without_session():
    """Before ``session_new``, the adapter has no steps rather than crashing."""
    from benchflow.acp.client import ACPClient

    client = ACPClient.__new__(ACPClient)
    client._session = None
    adapter = ACPSessionAdapter(client)
    assert adapter.steps == []


def test_adapter_on_ask_user_stores_and_forwards_handler():
    """The agent-initiated hook is registered on the adapter and forwarded.

    Forwarding to ``ACPClient.on_ask_user`` is the load-bearing fix for
    #382: without it the handler was stored but never invoked, and the ACP
    transport silently auto-approved every ``session/request_permission``.
    """

    class FakeClient:
        def __init__(self):
            self.bridge = None

        def on_ask_user(self, handler):
            self.bridge = handler

    client = FakeClient()
    adapter = ACPSessionAdapter(client)  # type: ignore[arg-type]

    async def handler(request):
        return "answer"

    adapter.on_ask_user(handler)
    assert adapter._ask_user_handler is handler
    # The adapter wires a bridge callable into the client; without this
    # forwarding the live ACP request path can't invoke the handler.
    assert client.bridge is not None


def test_acp_client_is_not_yet_an_agent():
    """Documents the known gap: ``ACPClient`` is not an ``Agent`` factory.

    The architecture's ``Agent`` is a *factory*: ``connect(sandbox, role)
    -> Session``. ``ACPClient.connect()`` takes no args and returns
    ``None`` (it just starts the transport). No class in the ACP stack
    plays the ``Agent`` role today — the SDK orchestration in
    ``rollout.py`` does the connect-and-handshake inline. The ``Agent``
    Protocol is defined to the architecture's shape so that work has a
    contract to converge on; this test pins the gap until then.
    """
    from benchflow.acp.client import ACPClient

    client = ACPClient.__new__(ACPClient)
    assert not isinstance(client, Agent)
