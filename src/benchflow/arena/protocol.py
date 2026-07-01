"""Turn-poll contract for inter-agent concurrent (arena) runs — Seam 3.

A shared-environment service that hosts N concurrent seats exposes exactly two
calls per seat: ``observe`` (poll for your turn) and ``act`` (submit one legal
action). This is the one genuinely new schema the arena runtime needs; it lives
here as a small, additive contract — no change to the verifier-facing
``Environment`` protocol.

A real ``SeatClient`` is an HTTP client to the co-tenant service; tests use an
in-memory fake. A real ``SeatPolicy`` wraps an ACP agent; tests script it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["SeatStatus", "Observation", "SeatClient", "SeatPolicy"]


class SeatStatus(enum.StrEnum):
    WAITING = "waiting"            # seated/queued, the table has not formed yet
    NOT_YOUR_TURN = "not_your_turn"
    YOUR_TURN = "your_turn"
    DONE = "done"


@dataclass
class Observation:
    """One seat's view, as returned by ``observe``."""

    status: SeatStatus
    request_id: str | None = None
    public: dict[str, Any] = field(default_factory=dict)
    private: dict[str, Any] = field(default_factory=dict)
    legal_actions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.status is SeatStatus.DONE

    @classmethod
    def from_payload(cls, p: dict[str, Any]) -> Observation:
        obs = p.get("observation") or {}
        return cls(
            status=SeatStatus(p.get("status", "done")),
            request_id=p.get("request_id"),
            public=dict(obs.get("public", {})),
            private=dict(obs.get("private", {})),
            legal_actions=list(p.get("legal_actions", [])),
        )


@runtime_checkable
class SeatClient(Protocol):
    """The shared-env service contract: poll a turn, submit an action."""

    async def observe(self, seat_id: str) -> dict[str, Any]: ...

    async def act(
        self, seat_id: str, request_id: str, action: dict[str, Any]
    ) -> dict[str, Any]: ...


@runtime_checkable
class SeatPolicy(Protocol):
    """A seat's brain: choose one legal action for a ``your_turn`` observation."""

    async def act(self, obs: Observation) -> dict[str, Any]: ...
