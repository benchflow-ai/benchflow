"""Dense reward events emitted during execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class RewardEvent:
    """A single reward signal emitted during or after a trial.

    Attributes:
        type:   "terminal" (end-of-trial verifier), "process" (verifier
                subprocess), or "dense" (mid-execution signal).
        reward: Scalar reward value for this event.
        source: Name of the RewardFunc that produced this event.
        step:   Tool-call index for dense rewards (None for terminal).
        ts:     ISO-8601 timestamp; auto-filled if omitted.
    """

    type: str
    reward: float
    source: str
    step: int | None = None
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
