"""Reward event schema and current verifier-dict conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RewardEvent:
    """One terminal, process, or dense reward signal."""

    type: str
    value: float
    source: str
    tag: str = "reward"
    step_index: int | None = None
    ts: datetime = field(default_factory=datetime.now)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return the persisted JSONL event shape."""

        return {
            "ts": self.ts.isoformat(),
            "type": self.type,
            "source": self.source,
            "value": self.value,
            "tag": self.tag,
            "step_index": self.step_index,
            "meta": self.meta,
        }


def rewards_from_verifier_dict(
    rewards: dict[str, Any] | None,
    *,
    finished_at: datetime | None = None,
) -> list[RewardEvent]:
    """Convert the current verifier reward dict into reward events.

    This preserves the existing rewards.jsonl semantics:
    rubric entries become ``process`` events and the scalar ``reward`` becomes
    the terminal event.
    """

    if not rewards:
        return []
    ts = finished_at or datetime.now()
    events: list[RewardEvent] = []
    rubric = rewards.get("rubric")
    if isinstance(rubric, list):
        for i, item in enumerate(rubric):
            if not isinstance(item, dict):
                continue
            score = float(item.get("score", 0.0))
            name = item.get("name", f"rubric_{i}")
            events.append(
                RewardEvent(
                    ts=ts,
                    type="process",
                    source="verifier_rubric",
                    value=score,
                    tag=str(name),
                    step_index=i,
                    meta={k: v for k, v in item.items() if k not in ("score", "name")},
                )
            )
    scalar = rewards.get("reward")
    if scalar is not None:
        events.append(
            RewardEvent(
                ts=ts,
                type="terminal",
                source="verifier",
                value=float(scalar),
                tag="reward",
                step_index=None,
                meta={
                    k: v
                    for k, v in rewards.items()
                    if k not in {"reward", "rubric"}
                },
            )
        )
    return events
