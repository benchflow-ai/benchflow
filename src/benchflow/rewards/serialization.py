"""Reward event persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow.rewards.events import RewardEvent, rewards_from_verifier_dict


def write_rewards_jsonl(
    rollout_dir: Path,
    rewards: dict[str, Any] | None = None,
    finished_at: datetime | None = None,
    *,
    events: list[RewardEvent] | None = None,
) -> None:
    """Write reward events to ``rewards.jsonl``.

    ``rewards`` is the current verifier-produced dict shape. New rubric code can
    pass explicit ``events`` once scoring is fully native.
    """

    reward_events = list(events or [])
    reward_events.extend(
        rewards_from_verifier_dict(rewards, finished_at=finished_at)
        if rewards is not None
        else []
    )
    if not reward_events:
        return

    path = rollout_dir / "rewards.jsonl"
    path.write_text(
        "\n".join(json.dumps(event.to_json(), default=str) for event in reward_events)
        + "\n"
    )
