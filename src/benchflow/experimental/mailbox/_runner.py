"""EXPERIMENTAL SURFACE — may change or be removed in any minor version.

MailboxRunner — a scheduler policy that drives a two-role, outbox-routed
conversation. Not a Scene: ``Scene`` is the dataclass in
``benchflow.contracts.trial_config``; runners are scheduler policies
that execute a configuration. The graduated reference policy is
``benchflow.trial.Trial._run_scene`` (TurnLoop: iterate ``scene.turns``
in order). MailboxRunner is an alternative policy and does not consume
a ``Scene`` dataclass today — it owns its own ``Role`` shape.

Scheduler rule (0.3):
  - exactly 2 roles (enforced at init)
  - exactly 1 active role at a time
  - a role yields when it sends a message or exits
  - run stops on max_rounds or explicit done

Message passing between agents uses a file-based convention: each agent
writes to /app/.outbox/{recipient}.json to send a message. The runner
reads the outbox after each agent exits, routes through the transport,
and injects into the next agent's prompt.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow.experimental.mailbox._transport import (
    MailboxTransport,
    Message,
    MessageTransport,
)

logger = logging.getLogger(__name__)


@dataclass
class MailboxRole:
    """Participant in a MailboxRunner session.

    Note: this is the runner's internal Role type, distinct from
    ``benchflow.contracts.trial_config.Role``. Kept separate because the
    runner needs ``instruction`` and ``tools`` for its per-turn prompt
    construction, which the graduated Role dataclass does not carry.
    """

    name: str
    agent: str
    model: str
    instruction: str
    tools: list[str] = field(default_factory=list)


class MailboxRunner:
    """Turn-based multi-agent runner using Mailbox transport + outbox files."""

    def __init__(
        self,
        roles: dict[str, MailboxRole],
        transport: MessageTransport | None = None,
        max_rounds: int = 10,
    ) -> None:
        if len(roles) != 2:
            raise ValueError(
                f"0.3 MailboxRunner requires exactly 2 roles, got {len(roles)}"
            )
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
            f"[MailboxRunner] round={self._round} {sender} → {recipient}: "
            f"{content[:80]}..."
            if len(content) > 80
            else f"[MailboxRunner] round={self._round} {sender} → {recipient}: {content}"
        )
        return f"Message delivered to {recipient} (round {self._round})"

    async def end_scene(self, sender: str, reward: float | None = None) -> None:
        """Called when a role signals the interaction is complete."""
        self._done = True
        logger.info(f"[MailboxRunner] ended by {sender}, reward={reward}")

    @property
    def is_done(self) -> bool:
        return self._done or self._round >= self.max_rounds

    def build_prompt_for_role(self, role: MailboxRole, inbox: list[Message]) -> str:
        """Build the prompt for a role, injecting any pending messages."""
        parts = [role.instruction]
        if inbox:
            parts.append("\n---\nYou have received the following messages:\n")
            for msg in inbox:
                parts.append(
                    f"**From {msg.sender} (round {msg.turn}):** {msg.content}\n"
                )
        parts.append(
            f"\nYou can send a message to another agent using the send_message tool. "
            f"Available recipients: {', '.join(n for n in self.roles if n != role.name)}."
        )
        return "\n".join(parts)

    _OUTBOX_DIR = "/app/.outbox"

    async def _setup_outbox(self, env: Any) -> None:
        await env.exec(f"rm -rf {self._OUTBOX_DIR} && mkdir -p {self._OUTBOX_DIR}")

    async def _read_outbox(self, env: Any, sender: str) -> list[Message]:
        """Read all messages left by sender in /app/.outbox/ and clear them."""
        result = await env.exec(f"ls {self._OUTBOX_DIR}/*.json 2>/dev/null || true")
        files = [
            f.strip() for f in (result.stdout or "").strip().splitlines() if f.strip()
        ]
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
                    logger.info(
                        f"[MailboxRunner] round={self._round} {sender} → "
                        f"{recipient}: {content[:80]}"
                    )
                    messages.append(msg)
            except json.JSONDecodeError:
                logger.warning(f"[MailboxRunner] invalid JSON in outbox file: {fpath}")
            await env.exec(f"rm -f {fpath}")
        return messages

    RoleRunner = Callable[..., Any]

    async def run(
        self,
        env: Any,
        role_runner: RoleRunner,
    ) -> list[Message]:
        """Run the 2-role interaction to completion.

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
            logger.info(f"[MailboxRunner] running {active} (round {self._round})")

            await role_runner(env, role, prompt)

            outbox_msgs = await self._read_outbox(env, sender=active)
            if not outbox_msgs:
                logger.info(
                    f"[MailboxRunner] {active} exited without sending a message — "
                    "run ends"
                )
                break

            active = self.next_active_role(active)

        logger.info(
            f"[MailboxRunner] complete: {len(self.trajectory)} messages, "
            f"{self._round} rounds"
        )
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
        logger.info(
            f"MailboxRunner trajectory saved: {len(self.trajectory)} messages → {path}"
        )
