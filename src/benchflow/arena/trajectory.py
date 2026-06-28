"""Per-seat trajectory capture for arena runs.

Each seat's turn — observation, chosen action, and (when the seat is LLM-backed)
the raw model request/response/usage — is appended as one JSON line to a per-seat
file. This is the bench's own per-agent decision trajectory; it complements the
LiteLLM proxy's ``llm_trajectory.jsonl`` (the raw model calls, captured whenever a
seat routes its LLM through ``BENCHFLOW_PROVIDER_BASE_URL``).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["TurnRecord", "SeatTrajectory"]


@dataclass
class TurnRecord:
    seat: str
    turn: int
    status: str
    request_id: str | None = None
    observation: dict[str, Any] = field(default_factory=dict)
    legal_actions: list[Any] = field(default_factory=list)
    action: dict[str, Any] | None = None
    llm: dict[str, Any] | None = None  # {model, messages, response, usage}
    t: float = 0.0


class SeatTrajectory:
    """Append-only per-seat trajectory writer under one rollout dir."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._turns: dict[str, int] = {}

    def path(self, seat: str) -> Path:
        return self.root / f"{seat}.trajectory.jsonl"

    def record(
        self,
        seat: str,
        *,
        status: Any,
        observation: dict[str, Any] | None = None,
        legal_actions: list[Any] | None = None,
        action: dict[str, Any] | None = None,
        llm: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> TurnRecord:
        turn = self._turns.get(seat, 0) + 1
        self._turns[seat] = turn
        rec = TurnRecord(
            seat=seat, turn=turn, status=str(status), request_id=request_id,
            observation=dict(observation or {}), legal_actions=list(legal_actions or []),
            action=action, llm=llm, t=time.time(),
        )
        with self.path(seat).open("a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        return rec
