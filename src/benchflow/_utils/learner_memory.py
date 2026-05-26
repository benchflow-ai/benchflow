"""Helpers for sequential-shared Memory-space scoring and artifacts."""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow._utils.reward_events import (
    reward_event_to_dict,
    reward_event_to_jsonl_record,
)
from benchflow.learner_skills import capture_skills, skill_memory_delta
from benchflow.learner_store import LearnerState
from benchflow.models import RolloutResult
from benchflow.rewards.memory_scorer import MEMORY_STATE_KEY, MemoryScorer
from benchflow.task.config import TaskConfig
from benchflow.trajectories.tree import RolloutNode

logger = logging.getLogger(__name__)


def expected_skills_for_task(task_dir: Path) -> list[str] | None:
    """Read the task-authored Memory-space answer key, if any."""
    config_path = task_dir / "task.toml"
    return TaskConfig.model_validate_toml(config_path.read_text()).expected_skills


def evolved_skills_for_result(
    result: RolloutResult, export_dir: Path
) -> dict[str, Any]:
    """Return skills captured from a rollout, preferring the result payload.

    When the rollout's skill export failed (``result.export_error``), the
    ``export_dir`` is half-written and re-reading it would re-introduce the
    same partial state the rollout already signalled as unsafe. Return ``{}``
    in that case so callers don't accidentally feed broken artifacts into
    the LearnerStore (#389 follow-up).
    """
    if result.export_error is not None:
        return {}
    if result.evolved_skills is not None:
        return dict(result.evolved_skills)
    return capture_skills(export_dir)


def memory_delta_from_skills(
    *,
    before_state: LearnerState,
    evolved_skills: dict[str, Any],
    expected_skills: list[str] | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build the fixture-safe Memory-space delta and normalized skill state."""
    before_skills = {k: str(v) for k, v in before_state.skills.items()}
    after_skills = {k: str(v) for k, v in evolved_skills.items()}
    delta = skill_memory_delta(
        before=before_skills,
        after=after_skills,
        expected=expected_skills,
    )
    return after_skills, delta


async def attach_memory_score(
    *,
    result: RolloutResult,
    node: RolloutNode,
    delta: dict[str, Any],
    result_path: Path | None = None,
) -> float:
    """Score a Memory-space delta, attach it to the result, and persist it."""
    node.state[MEMORY_STATE_KEY] = delta
    memory_event = await MemoryScorer().score(node)
    result.reward_events = [*(result.reward_events or []), memory_event]
    memory_score = float(memory_event.reward)
    if result_path is not None:
        patch_rollout_memory_artifact(
            result_path,
            memory_score=memory_score,
            reward_events=result.reward_events,
        )
    return memory_score


def patch_rollout_memory_artifact(
    result_path: Path, *, memory_score: float, reward_events: list[Any]
) -> None:
    """Patch rollout artifacts with additive Memory-space reward events."""
    if not result_path.is_file():
        return
    memory_events = [
        event for event in reward_events if getattr(event, "space", None) == "memory"
    ]
    data = json.loads(result_path.read_text())
    data["memory_score"] = memory_score
    data["reward_events"] = [reward_event_to_dict(event) for event in memory_events]
    result_path.write_text(json.dumps(data, indent=2))

    if memory_events:
        append_rollout_reward_events_jsonl(
            result_path.with_name("rewards.jsonl"), memory_events
        )


def append_rollout_reward_events_jsonl(path: Path, events: list[Any]) -> None:
    """Append additive reward-space events to the rollout JSONL artifact."""
    prior = path.read_text() if path.is_file() else ""
    ts = _last_reward_jsonl_ts(prior) or datetime.now().isoformat()
    lines = [json.dumps(reward_event_to_jsonl_record(event, ts=ts)) for event in events]
    prefix = prior
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    path.write_text(prefix + "\n".join(lines) + "\n")


def _last_reward_jsonl_ts(text: str) -> str | None:
    for line in reversed([line for line in text.splitlines() if line.strip()]):
        with contextlib.suppress(Exception):
            raw = json.loads(line)
            ts = raw.get("ts")
            if isinstance(ts, str):
                return ts
    return None


def patch_learner_generation_artifact(
    result_path: Path,
    *,
    inherited_from: int,
    produced: int | None,
    committed: bool | None,
) -> None:
    """Stamp continual-learning generation metadata onto a result artifact.

    Records which store generation the rollout inherited from and (if it
    committed) which generation it produced. ``committed`` is ``None`` for an
    unscored rollout, ``True`` for an accepted commit, ``False`` when
    :meth:`LearnerStore.commit_or_revert` rejected a regression.

    Additive — readers that do not know about the ``learner_generation`` block
    just skip it. Guards issue #394: a resumed run can audit per-rollout
    which generation it consumed and produced.
    """
    if not result_path.is_file():
        return
    try:
        data = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(f"Could not stamp generation metadata on {result_path}: {e}")
        return
    data["learner_generation"] = {
        "inherited_from": inherited_from,
        "produced": produced,
        "committed": committed,
    }
    try:
        result_path.write_text(json.dumps(data, indent=2))
    except OSError as e:
        logger.debug(f"Could not stamp generation metadata on {result_path}: {e}")
