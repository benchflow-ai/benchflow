"""Shared result metadata derived from usage telemetry and ACP trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def final_metrics_from_agent_result(
    agent_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return Harbor-compatible final token/cost metrics.

    Harbor's OpenHands runner exposes these four fields under
    ``trajectory.json.final_metrics``. BenchFlow keeps richer provenance in
    ``agent_result``; this helper adds the familiar Harbor shape without
    dropping BenchFlow-specific telemetry such as cache-creation tokens and
    source labels.
    """
    return {
        "total_prompt_tokens": agent_result.get("n_input_tokens"),
        "total_completion_tokens": agent_result.get("n_output_tokens"),
        "total_cached_tokens": agent_result.get("n_cache_read_tokens"),
        "total_cost_usd": agent_result.get("cost_usd"),
    }


def final_metrics_from_rollout(result: Any) -> dict[str, Any]:
    """Build Harbor-compatible final metrics from a RolloutResult-like object."""
    return final_metrics_from_agent_result(
        {
            "n_input_tokens": getattr(result, "n_input_tokens", None),
            "n_output_tokens": getattr(result, "n_output_tokens", None),
            "n_cache_read_tokens": getattr(result, "n_cache_read_tokens", None),
            "cost_usd": getattr(result, "cost_usd", None),
        }
    )


def trajectory_summary_from_events(
    trajectory: Sequence[Mapping[str, Any]] | None,
    *,
    partial_trajectory: bool,
    trajectory_source: str | None,
) -> dict[str, Any]:
    """Summarize ACP trajectory steps using the same counts Harbor reports."""
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
