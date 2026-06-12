"""ORS-episode reward evidence parsing and normalization.

Extracted from ``benchflow.task.verifier`` as a pure leaf cluster. Loads
declared ORS episode reward evidence (JSON / JSONL) and normalizes it into a
``VerifyResult`` carrying the terminal reward, per-item scores, and events.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from benchflow.rewards.events import Granularity, RewardEvent, Space
from benchflow.rewards.protocol import VerifyResult
from benchflow.rewards.validation import is_valid_reward_number
from benchflow.task.verifier_document import VerifierStrategy
from benchflow.task.verifier_errors import (
    ORSEpisodeInputError,
    VerifierOutputParseError,
)


def _load_ors_episode_records(
    path: Path,
    *,
    strategy: VerifierStrategy,
    declared_path: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ORSEpisodeInputError(
            f"ors-episode input {declared_path!r} must resolve to a regular file"
        )
    text = path.read_text(errors="replace")
    try:
        if path.suffix == ".jsonl":
            records: list[dict[str, Any]] = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                loaded = json.loads(line)
                if not isinstance(loaded, dict):
                    raise ORSEpisodeInputError(
                        f"ors-episode input {declared_path!r} line {line_number} "
                        "must be a JSON object"
                    )
                records.append(cast(dict[str, Any], loaded))
            if not records:
                raise ORSEpisodeInputError(
                    f"ors-episode input {declared_path!r} has no JSON records"
                )
            return records

        loaded = json.loads(text)
    except json.JSONDecodeError as e:
        raise ORSEpisodeInputError(
            f"ors-episode input {declared_path!r} is not valid JSON"
        ) from e

    if isinstance(loaded, dict):
        return [loaded]
    if isinstance(loaded, list) and all(isinstance(item, dict) for item in loaded):
        if not loaded:
            raise ORSEpisodeInputError(
                f"ors-episode input {declared_path!r} has no JSON records"
            )
        return cast(list[dict[str, Any]], loaded)
    raise ORSEpisodeInputError(
        f"ors-episode input {declared_path!r} must be a JSON object or list of objects"
    )


def _count_ors_episode_records(
    path: Path,
    *,
    strategy: VerifierStrategy,
    declared_path: str,
) -> int:
    return len(
        _load_ors_episode_records(
            path,
            strategy=strategy,
            declared_path=declared_path,
        )
    )


def _ors_records_to_verify_result(
    records: list[dict[str, Any]],
    *,
    strategy: VerifierStrategy,
) -> VerifyResult:
    events: list[RewardEvent] = []
    items: dict[str, float] = {}
    reward: float | None = None
    error: str | None = None
    headline_space: Space = "output"
    headline_granularity: Granularity = "terminal"

    for index, record in enumerate(records):
        path = f"records[{index}]"
        if _is_ors_response(record):
            if record.get("is_valid") is False:
                metadata = record.get("metadata")
                message = metadata.get("error") if isinstance(metadata, dict) else None
                raise VerifierOutputParseError(
                    f"ors-episode strategy {strategy.name!r} contains invalid "
                    f"ORS response at {path}: {message or 'is_valid=false'}"
                )
            reward = _bounded_ors_reward(record.get("reward"), path=f"{path}.reward")
            metadata = record.get("metadata", {})
            if isinstance(metadata, dict):
                items.update(
                    _ors_items(metadata.get("items"), path=f"{path}.metadata.items")
                )
                events.extend(
                    _ors_events(
                        metadata.get("events"),
                        strategy=strategy,
                        path=f"{path}.metadata.events",
                    )
                )
                raw_error = metadata.get("error")
                if isinstance(raw_error, str) and raw_error:
                    error = raw_error
                if isinstance(metadata.get("space"), str):
                    headline_space = _ors_space(
                        metadata["space"],
                        path=f"{path}.metadata.space",
                    )
                if isinstance(metadata.get("granularity"), str):
                    headline_granularity = _ors_granularity(
                        metadata["granularity"],
                        path=f"{path}.metadata.granularity",
                    )
            continue

        if "events" in record:
            events.extend(
                _ors_events(
                    record.get("events"),
                    strategy=strategy,
                    path=f"{path}.events",
                )
            )
            if "reward" in record:
                reward = _bounded_ors_reward(
                    record.get("reward"),
                    path=f"{path}.reward",
                )
            items.update(_ors_items(record.get("items"), path=f"{path}.items"))
            metadata = record.get("metadata", {})
            if isinstance(metadata, dict):
                items.update(
                    _ors_items(metadata.get("items"), path=f"{path}.metadata.items")
                )
            continue

        event = _ors_event(record, strategy=strategy, path=path)
        events.append(event)

    # Headline selection from the plain-event stream. Only a *genuinely*
    # terminal event (explicit ``type == "terminal"`` or
    # ``granularity == "terminal"``) may set the headline reward — a
    # ``finished``/``done`` marker on a step/dense record must never demote an
    # earlier real terminal (the last-write-wins bug). Conflicting genuine
    # terminals fail closed rather than silently picking the last.
    if reward is None:
        terminal_events = [
            event
            for event in events
            if event.type == "terminal" or event.granularity == "terminal"
        ]
        distinct_rewards = {event.reward for event in terminal_events}
        if len(distinct_rewards) > 1:
            raise VerifierOutputParseError(
                f"ors-episode strategy {strategy.name!r} has conflicting terminal "
                f"rewards {sorted(distinct_rewards)}; expected exactly one headline"
            )
        if terminal_events:
            chosen = terminal_events[0]
            reward = chosen.reward
            headline_space = chosen.space
            headline_granularity = chosen.granularity
    if reward is None:
        raise VerifierOutputParseError(
            f"ors-episode strategy {strategy.name!r} did not include a terminal reward"
        )
    if not items:
        items[strategy.name] = reward

    return VerifyResult(
        reward=reward,
        items=items,
        events=events,
        error=error,
        space=headline_space,
        granularity=headline_granularity,
    )


def _is_ors_response(record: dict[str, Any]) -> bool:
    return "reward" in record and (
        "is_valid" in record
        or (
            isinstance(record.get("metadata"), dict)
            and (
                "items" in record["metadata"]
                or "events" in record["metadata"]
                or "space" in record["metadata"]
            )
        )
    )


def _ors_events(
    value: Any,
    *,
    strategy: VerifierStrategy,
    path: str,
) -> list[RewardEvent]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise VerifierOutputParseError(f"{path} must be a list")
    events: list[RewardEvent] = []
    for index, raw_event in enumerate(value):
        if not isinstance(raw_event, dict):
            raise VerifierOutputParseError(f"{path}[{index}] must be an object")
        events.append(
            _ors_event(
                cast(dict[str, Any], raw_event),
                strategy=strategy,
                path=f"{path}[{index}]",
            )
        )
    return events


def _ors_event(
    record: dict[str, Any],
    *,
    strategy: VerifierStrategy,
    path: str,
) -> RewardEvent:
    reward = _bounded_ors_reward(record.get("reward"), path=f"{path}.reward")
    raw_step = record.get("step")
    if raw_step is None:
        step = None
    elif isinstance(raw_step, int):
        step = raw_step
    else:
        raise VerifierOutputParseError(f"{path}.step must be an integer or null")
    return RewardEvent(
        type=str(record.get("type") or "terminal"),
        reward=reward,
        source=str(record.get("source") or strategy.name),
        step=step,
        space=_ors_space(record.get("space", "output"), path=f"{path}.space"),
        granularity=_ors_granularity(
            record.get("granularity", "terminal"),
            path=f"{path}.granularity",
        ),
        ts=str(record.get("timestamp") or record.get("ts") or ""),
    )


def _bounded_ors_reward(value: Any, *, path: str) -> float:
    try:
        reward = float(value)
    except (TypeError, ValueError) as e:
        raise VerifierOutputParseError(f"{path} must be a numeric reward") from e
    if not is_valid_reward_number(reward):
        raise VerifierOutputParseError(
            f"{path} must be a finite numeric reward between 0.0 and 1.0"
        )
    return reward


def _ors_items(value: Any, *, path: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise VerifierOutputParseError(f"{path} must be a mapping")
    items: dict[str, float] = {}
    for name, raw_score in value.items():
        items[str(name)] = _bounded_ors_reward(raw_score, path=f"{path}.{name}")
    return items


def _ors_space(value: Any, *, path: str) -> Space:
    if value not in {"output", "action", "reasoning", "memory", "latent"}:
        raise VerifierOutputParseError(f"{path} must be a valid reward space")
    return cast(Space, value)


def _ors_granularity(value: Any, *, path: str) -> Granularity:
    if value not in {"terminal", "step"}:
        raise VerifierOutputParseError(f"{path} must be terminal or step granularity")
    return cast(Granularity, value)
