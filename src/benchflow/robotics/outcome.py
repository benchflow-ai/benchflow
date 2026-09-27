"""Honest outcome states for embodied trials: execution is not assessment.

A physical run that ends cleanly has only *executed*. Whether the task was
achieved is decided later by a reviewer looking at the recorded evidence, so
every trial carries two independent states:

``execution.status``
    ``completed``    the agent session ended on its own, nothing halted it
    ``no_motion``    ended on its own with a healthy pipeline and zero motion
                     dispatched: a refusal or a decision not to act; only the
                     transcript tells which
    ``halted``       the bridge stopped accepting commands (``halt_reason``)
    ``timed_out``    the agent's time budget ran out
    ``cancelled``    interrupted by the operator or the process
    ``agent_error``  the agent or its provider failed
    ``infrastructure_error``  the harness around the agent failed
    ``unfinalized``  the runner never recorded a terminal state

``execution.pipeline`` separates a stall from a dead controller: ``healthy``,
``controller_failure``, ``capture_failure`` or ``unknown``.

``assessment.status``
    ``pending``       nobody has assessed the evidence yet
    ``verified``      assessed with admissible evidence; the task succeeded
    ``failed``        assessed with admissible evidence; it did not
    ``unassessable``  evidence cannot support a verdict, or not a scored trial

Only ``verified`` and ``failed`` carry a reward. BenchFlow's shared scoring
(:mod:`benchflow._utils.scoring`) withholds the score of every other state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EXECUTION_STATUSES = frozenset(
    {
        "completed",
        "no_motion",
        "halted",
        "timed_out",
        "cancelled",
        "agent_error",
        "infrastructure_error",
        "unfinalized",
    }
)
_CONTROLLER_HALTS = frozenset({"uncertain_outcome", "hardware_abort"})
_SCORED_KIND = "physical_trial"


def _pipeline(manifest: Mapping[str, Any], actions: Mapping[str, Any] | None) -> str:
    halt = manifest.get("bridge_halt_reason")
    if halt == "recording_lost" or manifest.get("footage_complete") is False:
        return "capture_failure"
    if (
        halt in _CONTROLLER_HALTS
        or manifest.get("final_observation_error")
        or (
            actions
            and (actions.get("uncertain_outcomes") or actions.get("transport_failures"))
        )
    ):
        return "controller_failure"
    if manifest.get("footage_complete") is True:
        return "healthy"
    return "unknown"


def execution_outcome(
    manifest: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any] | None = None,
    actions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify how a trial's execution ended, from its recorded artifacts."""
    status = manifest.get("status")
    metrics = metrics or {}
    halt = manifest.get("bridge_halt_reason")
    pipeline = _pipeline(manifest, actions)
    error = str(metrics.get("error") or "")
    if status in {None, "preparing", "running"} or "finished_utc_epoch" not in manifest:
        execution = "unfinalized"
    elif status == "interrupted" or "operator stopped" in error.lower():
        execution = "cancelled"
    elif status in {"infrastructure_error", "smoke_failed"}:
        execution = "infrastructure_error"
    elif halt:
        execution = "halted"
    elif status == "agent_error":
        timed_out = metrics.get("error_category") in {"timeout", "idle_timeout"}
        execution = (
            "timed_out"
            if timed_out or "time budget exhausted" in error.lower()
            else "agent_error"
        )
    elif (
        actions is not None
        and pipeline == "healthy"
        and actions.get("motion_dispatched") == 0
        and actions.get("observations_ok", 0) > 0
    ):
        execution = "no_motion"
    else:
        execution = "completed"
    return {
        "status": execution,
        "recorded_status": status,
        "halt_reason": halt,
        "pipeline": pipeline,
        "motion_dispatched": actions.get("motion_dispatched") if actions else None,
        "error": metrics.get("error") or manifest.get("error_type"),
        "error_category": metrics.get("error_category"),
    }


def assessment_outcome(
    manifest: Mapping[str, Any],
    review: Mapping[str, Any] | None,
    *,
    execution: str,
) -> dict[str, Any]:
    """The assessment state and, only when assessed, the reward."""
    if manifest.get("kind") != _SCORED_KIND:
        return {
            "status": "unassessable",
            "reason": "not_a_scored_trial",
            "reward": None,
        }
    if execution == "unfinalized":
        return {
            "status": "unassessable",
            "reason": "execution_not_finalized",
            "reward": None,
        }
    if review is None:
        return {"status": "pending", "reason": "awaiting_reviewer", "reward": None}
    base = {
        "reviewer": review.get("reviewer"),
        "evidence": review.get("evidence"),
        "interventions": review.get("interventions"),
        "scored_utc_epoch": review.get("scored_utc_epoch"),
    }
    if not review.get("benchmark_valid"):
        return {
            **base,
            "status": "unassessable",
            "reason": "evidence_not_admissible",
            # Older assessments carry only the boolean.
            "invalid_reasons": review.get("invalid_reasons"),
            "reward": None,
        }
    success = bool(review.get("autonomous_success"))
    return {
        **base,
        "status": "verified" if success else "failed",
        "reason": None,
        "reward": 1.0 if success else 0.0,
    }
