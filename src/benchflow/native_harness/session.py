"""The native harness's :class:`~benchflow.agents.protocol.Session`.

The Agent plane's ``Session`` is one live conversation: ``prompt``, ``cancel``,
``on_ask_user``, ``steps`` and the ``on_change`` hook. :class:`NativeSession`
is its second implementation, next to ``ACPSessionAdapter``: it delegates to
:class:`~benchflow.native_harness.client.NativeCLIClient` the way the adapter
delegates to ``ACPClient``, and its ``steps`` are the same
:class:`~benchflow.acp.session.ACPSession` events, so a kernel, a branch
engine or an SDK caller holding a ``Session`` cannot tell the harnesses apart
except through :attr:`capabilities`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from benchflow.agents.protocol import (
    AgentCapabilities,
    AskUserHandler,
    Session,
    StopReason,
)
from benchflow.native_harness.client import NativeCLIClient


class NativeSession:
    """A native CLI conversation behind the ``Session`` contract."""

    def __init__(self, client: NativeCLIClient) -> None:
        self._client = client
        self._on_change_handler: Callable[[Session], None] | None = None

    @property
    def capabilities(self) -> AgentCapabilities:
        """What this harness supports (multi-turn yes; ask_user no)."""
        return self._client.harness.capabilities

    @property
    def session_id(self) -> str | None:
        """The CLI's session id: what a resumed turn or branch child continues."""
        return self._client.cli_session_id

    @property
    def on_change(self) -> Callable[[Session], None] | None:
        """Assignable streaming hook, bridged onto the underlying session state."""
        return self._on_change_handler

    @on_change.setter
    def on_change(self, handler: Callable[[Session], None] | None) -> None:
        self._on_change_handler = handler
        if handler is None:
            self._client.session.on_change = None
            return

        def _bridge(_session: Any) -> None:
            handler(self)

        self._client.session.on_change = _bridge

    async def prompt(self, text: str) -> StopReason:
        """Run one turn of the CLI; return why it stopped."""
        result = await self._client.prompt(text)
        return StopReason(result.stop_reason)

    async def cancel(self) -> None:
        """Stop the running turn (the CLI's process group is killed)."""
        await self._client.cancel()

    def on_ask_user(self, handler: AskUserHandler) -> None:
        """Refused: the CLIs run without a permission channel (see the client)."""
        self._client.on_ask_user(handler)

    @property
    def steps(self) -> list[Any]:
        """This session's ordered events, the same records the ACP path keeps."""
        return list(self._client.session.events)
