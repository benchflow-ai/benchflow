"""Helpers for serializing evaluation rollout results."""

from pathlib import Path
from typing import Any

from benchflow._utils.benchmark_repos import task_source_provenance
from benchflow._utils.reward_events import (
    memory_score_from_events,
    reward_event_to_dict,
)
from benchflow.models import RolloutResult
from benchflow.trajectories.metrics import (
    count_skill_invocations,
    result_skill_invocations,
)


def agent_result_from_rollout(result: RolloutResult) -> dict[str, Any]:
    """Return the serialized agent_result block for an in-memory rollout result."""
    n_skill_invocations = result.n_skill_invocations or count_skill_invocations(
        result.trajectory
    )
    return {
        "n_tool_calls": result.n_tool_calls,
        "n_skill_invocations": n_skill_invocations,
        "n_prompts": result.n_prompts,
        "n_input_tokens": result.n_input_tokens,
        "n_output_tokens": result.n_output_tokens,
        "n_cache_read_tokens": result.n_cache_read_tokens,
        "n_cache_creation_tokens": result.n_cache_creation_tokens,
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
        "usage_source": result.usage_source,
        "price_source": result.price_source,
    }


def rollout_result_payload(
    result: RolloutResult,
    *,
    source_provenance: dict[str, Any] | None,
    tasks_dir: Path,
    task_name: str,
) -> dict[str, Any]:
    """Normalize an in-memory rollout result to the persisted result shape."""
    reward_events = result.reward_events or []
    memory_score = memory_score_from_events(reward_events)
    task_source = result.source_provenance or task_source_provenance(
        source_provenance, tasks_dir / task_name
    )
    n_skill_invocations = result.n_skill_invocations or count_skill_invocations(
        result.trajectory
    )
    return {
        "task_name": result.task_name,
        "rewards": result.rewards,
        "error": result.error,
        "verifier_error": result.verifier_error,
        "export_error": result.export_error,
        "n_tool_calls": result.n_tool_calls,
        "n_skill_invocations": n_skill_invocations,
        "agent_result": agent_result_from_rollout(result),
        **(
            {"reward_events": [reward_event_to_dict(event) for event in reward_events]}
            if reward_events
            else {}
        ),
        **({"memory_score": memory_score} if memory_score is not None else {}),
        **({"source": task_source} if task_source else {}),
    }


def usage_summary(results: dict[str, dict]) -> dict[str, Any]:
    """Aggregate provider telemetry fields for summary.json."""
    completed = [
        r
        for r in results.values()
        if r.get("rewards") is not None
        and not r.get("error")
        and not r.get("verifier_error")
    ]
    covered = [
        r
        for r in completed
        if (r.get("agent_result") or {}).get("usage_source") == "provider_response"
    ]

    def total(field: str) -> int:
        return sum((r.get("agent_result") or {}).get(field) or 0 for r in covered)

    total_cost = round(
        sum((r.get("agent_result") or {}).get("cost_usd") or 0.0 for r in covered),
        10,
    )
    return {
        "total_input_tokens": total("n_input_tokens"),
        "total_output_tokens": total("n_output_tokens"),
        "total_cache_read_tokens": total("n_cache_read_tokens"),
        "total_cache_creation_tokens": total("n_cache_creation_tokens"),
        "total_tokens": total("total_tokens"),
        "total_cost_usd": total_cost,
        "avg_cost_per_trial_usd": (
            round(total_cost / len(covered), 10) if covered else None
        ),
        "telemetry_coverage": (len(covered) / len(completed) if completed else 0.0),
    }


def skill_invocation_summary(results: dict[str, dict]) -> dict[str, Any]:
    """Aggregate structured skill invocation counts for summary.json."""
    total = sum(result_skill_invocations(result) for result in results.values())
    return {
        "total_skill_invocations": total,
        "avg_skill_invocations": (
            round(total / len(results), 1) if results else 0.0
        ),
    }
