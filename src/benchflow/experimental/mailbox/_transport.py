"""EXPERIMENTAL SURFACE — may change or be removed in any minor version.

Message transport for experimental multi-agent runners. The Mailbox transport
is an in-memory per-role queue; alternative transports (HTTP, MCP, pub-sub)
would land as siblings in this subpackage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    id: str
    sender: str
    recipient: str
    content: str
    turn: int
    kind: str = "direct"
    ts: str = field(default_factory=lambda: datetime.now().isoformat())


class MessageTransport(ABC):
    @abstractmethod
    async def send(self, msg: Message) -> None: ...

    @abstractmethod
    async def receive(self, role_name: str) -> Message | None: ...

    @abstractmethod
    async def list_pending(self, role_name: str) -> list[Message]: ...


class MailboxTransport(MessageTransport):
    """In-memory per-role message queues."""

    def __init__(self) -> None:
        self._queues: dict[str, list[Message]] = {}

    async def send(self, msg: Message) -> None:
        self._queues.setdefault(msg.recipient, []).append(msg)

    async def receive(self, role_name: str) -> Message | None:
        q = self._queues.get(role_name, [])
        return q.pop(0) if q else None

    async def list_pending(self, role_name: str) -> list[Message]:
        return list(self._queues.get(role_name, []))
