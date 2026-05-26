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

# Phase keys produced by Rollout (see rollout.py — environment_setup,
# agent_setup, agent_execution, verifier, total). Kept here so summary
# aggregation stays in lockstep with the rollout writer.
_TIMING_PHASES: tuple[str, ...] = (
    "environment_setup",
    "agent_setup",
    "agent_execution",
    "verifier",
    "total",
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
        "error_category": result.error_category,
        "verifier_error": result.verifier_error,
        "verifier_error_category": result.verifier_error_category,
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
        "avg_skill_invocations": (round(total / len(results), 1) if results else 0.0),
    }


def _round_secs(value: float) -> float:
    """Round a duration in seconds to the same precision rollout.py uses."""
    return round(float(value), 1)


def tool_call_summary(results: dict[str, dict]) -> dict[str, Any]:
    """Aggregate per-rollout ``n_tool_calls`` across every result in the job.

    Unlike ``usage_summary``, this counts EVERY rollout — including errored
    and verifier-errored ones — because tool-call cost is paid regardless of
    whether verification succeeded. Reviewers asking "how many tool calls did
    this job consume?" want the literal sum, not a success-filtered one.
    """
    counts = [int(r.get("n_tool_calls") or 0) for r in results.values()]
    total = sum(counts)
    return {
        "total_tool_calls": total,
        "avg_tool_calls_per_task": (total / len(counts)) if counts else 0.0,
        "max_tool_calls_per_task": max(counts) if counts else 0,
    }


def phase_timing_summary(results: dict[str, dict]) -> dict[str, Any]:
    """Aggregate per-phase wall-clock timing across every rollout.

    Sums and averages cover any rollout that recorded a ``timing`` block, so
    reviewers can answer "how much time went to agent vs verifier?" without
    inspecting each ``result.json``. Phase keys follow ``rollout.py``:
    ``environment_setup``, ``agent_setup``, ``agent_execution``, ``verifier``,
    ``total``. The ``timing_coverage`` ratio surfaces when phase data is
    incomplete (e.g. mocked test runs that don't persist ``timing``).
    """
    timings: list[dict[str, float]] = []
    for r in results.values():
        t = r.get("timing")
        if isinstance(t, dict) and t:
            timings.append(t)

    out: dict[str, Any] = {
        "timing_coverage": (len(timings) / len(results)) if results else 0.0,
    }
    for phase in _TIMING_PHASES:
        values = [
            float(t[phase]) for t in timings if isinstance(t.get(phase), (int, float))
        ]
        # Phases with no data get a 0.0 sum + null avg/max so downstream
        # readers can distinguish "ran but cost nothing" from "no data".
        out[f"{phase}_time_sec"] = _round_secs(sum(values)) if values else 0.0
        out[f"avg_{phase}_time_sec"] = (
            _round_secs(sum(values) / len(values)) if values else None
        )
        out[f"max_{phase}_time_sec"] = _round_secs(max(values)) if values else None
    return out
