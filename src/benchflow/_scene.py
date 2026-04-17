"""Multi-agent scene runtime — turn-based scheduler for N roles.

A Scene orchestrates multiple ACP agents in a shared sandbox. Each Role
runs one at a time; a role yields control when it calls send_message()
or exits. The scheduler injects received messages into the next role's
prompt and continues until max_rounds or an explicit done signal.

Transport is pluggable via MessageTransport. The default MailboxTransport
is an in-memory queue — no HTTP server, no sidecar.

Message passing between agents uses a file-based convention: each agent
writes to /tmp/outbox/{recipient}.json to send a message. The scheduler
reads the outbox after each agent exits, routes through the transport,
and injects into the next agent's prompt.

0.3 scope: exactly 2 roles, sequential execution, mailbox transport only.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass
class Role:
    name: str
    agent: str
    model: str
    instruction: str
    tools: list[str] = field(default_factory=list)


@dataclass
class Message:
    id: str
    sender: str
    recipient: str
    content: str
    turn: int
    kind: str = "direct"
    ts: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Transport abstraction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


class Scene:
    """Turn-based multi-agent scene.

    Scheduler rule (0.3):
      - exactly 2 roles (enforced at init)
      - exactly 1 active role at a time
      - a role yields when it sends a message or exits
      - scene stops on max_rounds or explicit done
    """

    def __init__(
        self,
        roles: dict[str, Role],
        transport: MessageTransport | None = None,
        max_rounds: int = 10,
    ) -> None:
        if len(roles) != 2:
            raise ValueError(f"0.3 scene requires exactly 2 roles, got {len(roles)}")
        self.roles = roles
        self.transport = transport or MailboxTransport()
        self.max_rounds = max_rounds
        self.trajectory: list[Message] = []
        self._round = 0
        self._done = False

    @property
    def role_names(self) -> list[str]:
        return list(self.roles.keys())

    def next_active_role(self, current: str) -> str:
        names = self.role_names
        idx = names.index(current)
        return names[(idx + 1) % len(names)]

    async def send_message(self, sender: str, recipient: str, content: str) -> str:
        """Called by the runtime when an agent invokes the send_message tool."""
        if recipient not in self.roles:
            return f"Error: unknown recipient '{recipient}'"
        self._round += 1
        msg = Message(
            id=str(uuid.uuid4())[:8],
            sender=sender,
            recipient=recipient,
            content=content,
            turn=self._round,
        )
        self.trajectory.append(msg)
        await self.transport.send(msg)
        logger.info(
            f"[Scene] round={self._round} {sender} → {recipient}: {content[:80]}..."
            if len(content) > 80
            else f"[Scene] round={self._round} {sender} → {recipient}: {content}"
        )
        return f"Message delivered to {recipient} (round {self._round})"

    async def end_scene(self, sender: str, reward: float | None = None) -> None:
        """Called when a role signals the scene is complete."""
        self._done = True
        logger.info(f"[Scene] ended by {sender}, reward={reward}")

    @property
    def is_done(self) -> bool:
        return self._done or self._round >= self.max_rounds

    def build_prompt_for_role(self, role: Role, inbox: list[Message]) -> str:
        """Build the prompt for a role, injecting any pending messages."""
        parts = [role.instruction]
        if inbox:
            parts.append("\n---\nYou have received the following messages:\n")
            for msg in inbox:
                parts.append(f"**From {msg.sender} (round {msg.turn}):** {msg.content}\n")
        parts.append(
            f"\nYou can send a message to another agent using the send_message tool. "
            f"Available recipients: {', '.join(n for n in self.roles if n != role.name)}."
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Outbox convention: agents write /tmp/outbox/{recipient}.json
    # ------------------------------------------------------------------

    _OUTBOX_DIR = "/app/.outbox"

    async def _setup_outbox(self, env: Any) -> None:
        await env.exec(f"rm -rf {self._OUTBOX_DIR} && mkdir -p {self._OUTBOX_DIR}")

    async def _read_outbox(self, env: Any, sender: str) -> list[Message]:
        """Read all messages left by sender in /tmp/outbox/ and clear them."""
        result = await env.exec(f"ls {self._OUTBOX_DIR}/*.json 2>/dev/null || true")
        files = [f.strip() for f in (result.stdout or "").strip().splitlines() if f.strip()]
        messages = []
        for fpath in files:
            cat_result = await env.exec(f"cat {fpath}")
            try:
                data = json.loads(cat_result.stdout or "{}")
                recipient = data.get("to", "")
                content = data.get("content", "")
                if recipient and content:
                    self._round += 1
                    msg = Message(
                        id=str(uuid.uuid4())[:8],
                        sender=sender,
                        recipient=recipient,
                        content=content,
                        turn=self._round,
                    )
                    self.trajectory.append(msg)
                    await self.transport.send(msg)
                    logger.info(f"[Scene] round={self._round} {sender} → {recipient}: {content[:80]}")
                    messages.append(msg)
            except json.JSONDecodeError:
                logger.warning(f"[Scene] invalid JSON in outbox file: {fpath}")
            await env.exec(f"rm -f {fpath}")
        return messages

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    RoleRunner = Callable[..., Any]

    async def run(
        self,
        env: Any,
        role_runner: RoleRunner,
    ) -> list[Message]:
        """Run the 2-role scene to completion.

        Args:
            env: the sandbox environment (started, files uploaded)
            role_runner: async callback(env, role, prompt) -> None
                         Connects ACP, sends prompt, waits for agent exit.
                         Does NOT manage env lifecycle.

        Returns: the full message trajectory.
        """
        await self._setup_outbox(env)
        active = self.role_names[0]

        while not self.is_done:
            role = self.roles[active]
            inbox = []
            while True:
                msg = await self.transport.receive(active)
                if msg is None:
                    break
                inbox.append(msg)

            prompt = self.build_prompt_for_role(role, inbox)
            logger.info(f"[Scene] running {active} (round {self._round})")

            await role_runner(env, role, prompt)

            outbox_msgs = await self._read_outbox(env, sender=active)
            if not outbox_msgs:
                logger.info(f"[Scene] {active} exited without sending a message — scene ends")
                break

            active = self.next_active_role(active)

        logger.info(f"[Scene] complete: {len(self.trajectory)} messages, {self._round} rounds")
        return self.trajectory

    def save_trajectory(self, path: Path) -> None:
        """Write the inter-agent message trajectory as JSONL."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {
                    "id": m.id,
                    "sender": m.sender,
                    "recipient": m.recipient,
                    "content": m.content,
                    "turn": m.turn,
                    "kind": m.kind,
                    "ts": m.ts,
                },
                default=str,
            )
            for m in self.trajectory
        ]
        path.write_text("\n".join(lines) + "\n" if lines else "")
        logger.info(f"Scene trajectory saved: {len(self.trajectory)} messages → {path}")
