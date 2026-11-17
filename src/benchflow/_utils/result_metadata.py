"""Shared result metadata derived from usage telemetry and ACP trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: The token split result.json's agent_result carries, repeated in
#: ATIF's ``final_metrics.extra``.
TOKEN_SPLIT_FIELDS = (
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_read_tokens",
    "n_cache_write_tokens",
    "n_cache_write_5m_tokens",
    "n_cache_write_1h_tokens",
)


def _cache_writes(agent_result: Mapping[str, Any]) -> Any:
    written = agent_result.get("n_cache_write_tokens")
    return (
        written if written is not None else agent_result.get("n_cache_creation_tokens")
    )


def incomplete_usage_fields(source: Any) -> dict[str, Any]:
    """``usage_incomplete: true`` and ``usage_incomplete_reason`` when a
    trial's tokens are those of some of its calls only (a turn cut off
    before its report that no log counted, a log copy that had not caught
    up), and ``cost_incomplete: true`` when a cost was priced from them:
    the figures are what was counted, never the whole. Present only then.
    *source* is an agent_result (a mapping) or a result object."""

    def get(key: str) -> Any:
        if isinstance(source, Mapping):
            return source.get(key)
        return getattr(source, key, None)

    if get("usage_incomplete") is not True:
        return {}
    fields: dict[str, Any] = {
        "usage_incomplete": True,
        "usage_incomplete_reason": get("usage_incomplete_reason"),
    }
    if get("cost_usd") is not None:
        fields["cost_incomplete"] = True
    return fields


def floor_note(partial: int, costed: int) -> str:
    """What is a floor for *partial* trials whose count missed calls,
    *costed* of them with a cost: their tokens, and their cost where one was
    priced (an unknown cost is unknown, not a floor)."""
    if costed >= partial:
        return "their tokens and cost are a floor"
    if not costed:
        return "their tokens are a floor (no cost is known)"
    return f"their tokens are a floor, and so is the cost of {costed} of them"


def cost_provenance(agent_result: Mapping[str, Any]) -> dict[str, Any]:
    """Where a result's tokens and cost came from: ``usage_source``,
    ``cost_source`` and ``price_table_date``, as result.json's ``agent_result``
    has them, with its token split (:data:`TOKEN_SPLIT_FIELDS`). ATIF's
    ``final_metrics.extra`` carries the same."""
    from benchflow.usage_tracking import result_usage_source

    split = {field: agent_result.get(field) for field in TOKEN_SPLIT_FIELDS}
    split["n_cache_write_tokens"] = _cache_writes(agent_result)
    return {
        "usage_source": result_usage_source(agent_result.get("usage_source")),
        "cost_source": agent_result.get("cost_source"),
        "price_table_date": agent_result.get("price_table_date"),
        "cost_usd_provider": agent_result.get("cost_usd_provider"),
        "n_calls_cost_estimated": agent_result.get("n_calls_cost_estimated"),
        **split,
        **incomplete_usage_fields(agent_result),
    }


def final_metrics_from_agent_result(
    agent_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return final token/cost metrics in the ATIF ``final_metrics`` shape.

    ATIF's fields keep ATIF's meaning: ``total_prompt_tokens`` counts
    the whole prompt, cache included (uncached input + cache reads + cache
    writes), and ``total_cached_tokens`` the cache reads. BenchFlow's own
    split, and where it came from, is in ``extra`` (:func:`cost_provenance`).
    Unknown counts are None, never 0.
    """
    fresh = agent_result.get("n_input_tokens")
    read = agent_result.get("n_cache_read_tokens")
    written = _cache_writes(agent_result)
    return {
        "total_prompt_tokens": None
        if fresh is None
        else fresh + (read or 0) + (written or 0),
        "total_completion_tokens": agent_result.get("n_output_tokens"),
        "total_cached_tokens": read,
        "total_cost_usd": agent_result.get("cost_usd"),
        "extra": cost_provenance(agent_result),
    }


def final_metrics_from_rollout(result: Any) -> dict[str, Any]:
    """Build ATIF-shaped final metrics from a RolloutResult-like object."""
    return final_metrics_from_agent_result(
        {
            field: getattr(result, field, None)
            for field in (
                *TOKEN_SPLIT_FIELDS,
                "n_cache_creation_tokens",
                "cost_usd",
                "usage_source",
                "cost_source",
                "price_table_date",
                "cost_usd_provider",
                "n_calls_cost_estimated",
                "usage_incomplete",
                "usage_incomplete_reason",
            )
        }
    )


def trajectory_summary_from_events(
    trajectory: Sequence[Mapping[str, Any]] | None,
    *,
    partial_trajectory: bool,
    trajectory_source: str | None,
) -> dict[str, Any]:
    """Summarize ACP trajectory steps as event, tool-call and status counts."""
    event_type_counts: dict[str, int] = {}
    tool_call_status_counts: dict[str, int] = {}

    for event in trajectory or ():
        event_type = str(event.get("type") or "unknown")
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        if event_type == "tool_call":
            status = str(event.get("status") or "unknown")
            tool_call_status_counts[status] = tool_call_status_counts.get(status, 0) + 1

    steps = sum(event_type_counts.values())
    tool_call_steps = event_type_counts.get("tool_call", 0)
    known_non_tool_steps = sum(
        event_type_counts.get(event_type, 0)
        for event_type in ("user_message", "agent_message", "agent_thought")
    )

    return {
        "steps": steps,
        "tool_call_steps": tool_call_steps,
        "user_message_steps": event_type_counts.get("user_message", 0),
        "agent_message_steps": event_type_counts.get("agent_message", 0),
        "agent_thought_steps": event_type_counts.get("agent_thought", 0),
        "other_steps": steps - tool_call_steps - known_non_tool_steps,
        "event_type_counts": event_type_counts,
        "tool_call_status_counts": tool_call_status_counts,
        "partial_trajectory": partial_trajectory,
        "trajectory_source": trajectory_source,
    }
