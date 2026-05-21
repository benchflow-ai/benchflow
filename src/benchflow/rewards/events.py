"""Dense reward events emitted during execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class RewardEvent:
    """A single reward signal emitted during or after a trial.

    The tagged reward record of the architecture: ``(space, granularity,
    value)`` — ``space``/``granularity`` are the tag, ``reward`` the value.

    Attributes:
        type:        "terminal" (end-of-trial verifier), "process" (verifier
                     subprocess), or "dense" (mid-execution signal).
        reward:      Scalar reward value for this event.
        source:      Name of the RewardFunc / scorer that produced this event.
        step:        Tool-call index for dense rewards (None for terminal).
        space:       Evaluation space — "output" (did it finish the job?),
                     "action", "reasoning", or "memory". Defaults to "output".
        granularity: "terminal" (whole trajectory) or "step" (one edge).
        ts:          ISO-8601 timestamp; auto-filled if omitted.
    """

    type: str
    reward: float
    source: str
    step: int | None = None
    space: str = "output"
    granularity: str = "terminal"
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
