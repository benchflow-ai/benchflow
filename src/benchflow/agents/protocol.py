"""The Agent contract — the second of BenchFlow's four planes.

The kernel depends only on these Protocols (architecture.md, "The four
contracts"). The Agent plane answers *who acts*: the agent under test
(eval) or the policy under training.

Two Protocols, one per altitude of the plane:

* ``Agent`` — the *factory*. Declared once, it connects to a sandbox in a
  given role and hands back a live ``Session``.
* ``Session`` — the **real behavioural surface of the plane**: the live
  agent conversation. ``prompt`` carries the task instruction and every
  nudge; ``on_ask_user`` is the agent-initiated branch hook; ``steps`` is
  the session's contribution to the rollout trajectory.

The ACP classes (``ACPClient`` / ``ACPSession``) are the first concrete
implementation; ``ACPSessionAdapter`` below is the thin, documented
adapter that makes ``ACPClient`` honour the ``Session`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

# The ACP ``StopReason`` enum already names exactly why an agent stops after
# a prompt (end_turn / max_tokens / refusal / cancelled / …); the Agent
# plane reuses it rather than minting a parallel one.
from benchflow.acp.types import StopReason

if TYPE_CHECKING:
    from benchflow.acp.client import ACPClient

__all__ = [
    "ACPSessionAdapter",
    "Agent",
    "AgentCapabilities",
    "AskUserHandler",
    "AskUserRequest",
    "Session",
    "StopReason",
]


@dataclass(frozen=True)
class AgentCapabilities:
    """What an ``Agent`` declares about itself before it ever connects.

    Returned by :meth:`Agent.capabilities`. Kept deliberately small — the
    kernel only needs enough to route and gate, not the full ACP
    ``initialize`` payload.

    Attributes:
        protocol:    Wire protocol the agent speaks — ``"acp"`` by default.
        nudges:      Whether the agent accepts a follow-up ``prompt`` on a
                     live session (multi-turn / nudge support).
        ask_user:    Whether the agent can initiate an ``ask_user`` request
                     back to the client — the branchable interaction
                     primitive.
        token_logprobs: Whether the agent surfaces token-ids + logprobs
                     (token-in/token-out). Best-effort for ACP agents.
    """

    protocol: str = "acp"
    nudges: bool = True
    ask_user: bool = False
    token_logprobs: bool = False


@dataclass
class AskUserRequest:
    """An agent-initiated request for input — the branchable primitive.

    Surfaced through :meth:`Session.on_ask_user`. Carries an optional set
    of enumerated ``options``; a finite option set makes the interaction a
    finite, scoreable tree (the tree itself is the platform layer).

    Attributes:
        prompt:  The question the agent is asking the client/user.
        options: Enumerated answers, when the agent offers a choice.
        request_id: Correlates the request with the answer the handler
                    returns.
    """

    prompt: str
    options: list[str] = field(default_factory=list)
    request_id: str = ""


# An ``on_ask_user`` handler receives the agent's request and returns the
# answer text the client should send back.
class AskUserHandler(Protocol):
    """Callable the client registers to answer agent-initiated questions."""

    async def __call__(self, request: AskUserRequest) -> str: ...


@runtime_checkable
class Session(Protocol):
    """The live agent session — the Agent plane's real surface.

    A ``Session`` is one open conversation with a connected agent. The
    kernel drives the rollout entirely through it: it ``prompt``s with the
    task instruction (and every later nudge), can ``cancel`` an in-flight
    turn, registers an ``on_ask_user`` handler for agent-initiated
    branches, and reads ``steps`` for the session's contribution to the
    trajectory.
    """

    async def prompt(self, text: str) -> StopReason:
        """Send the task instruction or a nudge; block until the turn ends.

        Returns the :class:`StopReason` for why the agent stopped.
        """
        ...

    async def cancel(self) -> None:
        """Abort the in-flight turn (ACP ``session/cancel``)."""
        ...

    def on_ask_user(self, handler: AskUserHandler) -> None:
        """Register the handler for agent-initiated ``ask_user`` requests.

        The agent-initiated channel; the hook a branch is taken on.
        """
        ...

    @property
    def steps(self) -> list[Any]:
        """This session's ordered steps — its contribution to the rollout."""
        ...


@runtime_checkable
class Agent(Protocol):
    """The agent factory — *who acts* in a rollout.

    Declared once (the registry stores agent declarations as data); the
    kernel calls :meth:`connect` to open a live :class:`Session` against a
    sandbox in a given role, and :meth:`capabilities` to learn what the
    agent supports before connecting.
    """

    async def connect(self, sandbox: Any, role: str) -> Session:
        """Connect to ``sandbox`` in ``role`` and return a live session."""
        ...

    def capabilities(self) -> AgentCapabilities:
        """Declare what this agent supports, before any connection."""
        ...


class ACPSessionAdapter:
    """Bridges the real ACP stack onto the :class:`Session` contract.

    The architecture's ``Session`` is one object with the live verbs
    *and* the accumulated state. The current ACP implementation splits
    that surface across two classes:

    * :class:`~benchflow.acp.client.ACPClient` — owns the live verbs
      (``prompt``, ``cancel``) because they are JSON-RPC calls over the
      transport.
    * :class:`~benchflow.acp.session.ACPSession` — owns the accumulated
      state (tool calls, message chunks, events).

    This adapter is the thin, documented seam that re-unifies them into a
    single ``Session``. It is **not** an ACP rewrite — it delegates every
    call straight through. Two gaps with the architecture's shape, kept
    explicit here rather than papered over:

    * ``ACPClient`` exposes no ``on_ask_user`` hook. ACP's agent-initiated
      channel is ``session/request_permission``, which the client today
      auto-approves in ``_handle_agent_request``. The adapter stores the
      handler so the contract is honest; wiring it into the client's
      permission path is a follow-up (NudgeBench forces it).
    * ``steps`` maps onto ``ACPSession.events`` — the chronological event
      log. The kernel's ``Step`` noun is not yet a typed object, so the
      adapter surfaces the raw event dicts.
    """

    def __init__(self, client: ACPClient) -> None:
        self._client = client
        self._ask_user_handler: AskUserHandler | None = None

    async def prompt(self, text: str) -> StopReason:
        """Send the task instruction or a nudge; return the stop reason.

        ``PromptResult.stop_reason`` arrives as the bare ``StrEnum`` value;
        coerce it back to a :class:`StopReason` member so the contract's
        return type is literally honoured.
        """
        result = await self._client.prompt(text)
        return StopReason(result.stop_reason)

    async def cancel(self) -> None:
        """Abort the in-flight turn (ACP ``session/cancel``)."""
        await self._client.cancel()

    def on_ask_user(self, handler: AskUserHandler) -> None:
        """Register the agent-initiated ``ask_user`` handler.

        Stored on the adapter; see the class docstring for the gap with
        ``ACPClient``'s current auto-approve permission path.
        """
        self._ask_user_handler = handler

    @property
    def steps(self) -> list[Any]:
        """This session's ordered events — its contribution to the rollout."""
        session = self._client.session
        return list(session.events) if session is not None else []
