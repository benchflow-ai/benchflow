"""Pure helpers for serialized reward events and additive score summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_EVENT_FIELDS = ("type", "reward", "source", "space", "granularity", "step")
_JSONL_META_FIELDS = ("space", "granularity")


def reward_event_to_dict(event: Any) -> dict[str, Any]:
    """Serialize one reward event through the public, fixture-safe allowlist."""
    out: dict[str, Any] = {}
    for field in _EVENT_FIELDS:
        value = (
            event.get(field)
            if isinstance(event, Mapping)
            else getattr(event, field, None)
        )
        if value is None and field == "step":
            continue
        out[field] = value
    return out


def reward_event_to_jsonl_record(event: Any, *, ts: str) -> dict[str, Any]:
    """Serialize one reward event into the rollout ``rewards.jsonl`` shape."""
    data = reward_event_to_dict(event)
    meta = {field: data[field] for field in _JSONL_META_FIELDS if data.get(field)}
    return {
        "ts": ts,
        "type": data.get("type"),
        "source": data.get("source"),
        "value": data.get("reward"),
        "tag": data.get("space") or data.get("source") or "reward",
        "step_index": data.get("step"),
        "meta": meta,
    }


def memory_score_from_events(events: list[Any] | None) -> float | None:
    """Return the latest Memory-space score from serialized or object events."""
    if not events:
        return None
    for event in reversed(events):
        space = (
            event.get("space")
            if isinstance(event, Mapping)
            else getattr(event, "space", None)
        )
        if space != "memory":
            continue
        reward = (
            event.get("reward")
            if isinstance(event, Mapping)
            else getattr(event, "reward", None)
        )
        if isinstance(reward, int | float):
            return float(reward)
    return None


def memory_score_from_result(result: Mapping[str, Any]) -> float | None:
    """Extract a Memory-space score from a persisted result.json shape."""
    raw = result.get("memory_score")
    if isinstance(raw, int | float):
        return float(raw)
    events = result.get("reward_events")
    return memory_score_from_events(events if isinstance(events, list) else None)


def memory_summary(
    results: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Aggregate optional per-task Memory-space scores."""
    scores = {
        task: score
        for task, data in results.items()
        if (score := memory_score_from_result(data)) is not None
    }
    if not scores:
        return {"scored": 0, "avg_score": None, "score": None}, {}
    avg = sum(scores.values()) / len(scores)
    return {
        "scored": len(scores),
        "avg_score": avg,
        "score": f"{avg:.1%}",
    }, scores
