"""Scoped child observations, adapted from JeremyJC67 PR #1046.

A branch-child record is intentionally distinct from a terminal rollout result:
shared sandbox continuations do not independently produce all rollout telemetry.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from benchflow.diagnostics import RolloutDiagnostics
from benchflow.review.persistence import write_json_atomic
from benchflow.trajectories.types import redact_trajectory_obj

# Rollout state written by connect/execute/verify/disconnect, _build_result and
# verifier recovery. Never deepcopy live sessions, provider runtimes, or
# sandbox/plane objects.
RESULT_STATE_FIELDS = (
    "_timing",
    "_verifier_error",
    "_diagnostics",
    "_native_usage_metrics",
    "_native_usage_checkpoint",
    "_error",
    "_export_error",
    "_evolved_skills",
    "_usage_metrics",
    "_scoring",
    "_completed_result",
    "_solver_execution_complete",
    "_solver_completion_result",
    "_terminal_timeout",
    "_bare_timeout",
    "_provider_failure_cached",
    "_provider_auth_status_cached",
    "_api_failure_summary_cached",
    "_user_rounds_log",
    "_agent_env",
    "_branch_child_active",
    "_branch_cleanup_unquiesced",
    "_agent_name",
    "_active_role",
    "_rollout_name",
    "_started_at",
)


def scope_child_result_state(rollout: Any) -> None:
    for name in (
        "_rewards",
        "_verifier_error",
        "_error",
        "_export_error",
        "_evolved_skills",
        "_native_usage_checkpoint",
        "_scoring",
        "_completed_result",
        "_solver_completion_result",
        "_provider_failure_cached",
        "_provider_auth_status_cached",
        "_api_failure_summary_cached",
    ):
        if hasattr(rollout, name):
            setattr(rollout, name, None)
    for name in (
        "_terminal_timeout",
        "_bare_timeout",
        "_solver_execution_complete",
        "_capture_over_limit",
    ):
        if hasattr(rollout, name):
            setattr(rollout, name, False)
    rollout._branch_cleanup_unquiesced = False
    rollout._timing = {}
    rollout._user_rounds_log = []
    # Deferred: benchflow.rollout imports this module.
    from benchflow.rollout import _zero_native_acp_usage_metrics

    rollout._diagnostics = RolloutDiagnostics()
    rollout._native_usage_metrics = _zero_native_acp_usage_metrics()
    rollout._usage_metrics = _zero_native_acp_usage_metrics()


def write_child_observation(
    rollout: Any,
    child: Any,
    child_dir: Path,
    *,
    error: BaseException | None,
    trajectory_start: int,
    lineage: dict[str, Any] | None = None,
) -> None:
    """Persist genuine observations, including failed/unscored children.

    ``lineage`` (parent rollout, fork id, parent node, index, label) keeps a
    child archive locatable in the tree when it is copied out of the run.
    """
    child_dir.mkdir(parents=True, exist_ok=True)
    observation = {
        "kind": "branch-child",
        "node_id": child.id,
        "lineage": lineage,
        "reward": child.state.get("reward"),
        "error": str(error) if error is not None else rollout._error,
        "verifier_error": rollout._verifier_error,
        "export_error": rollout._export_error,
        "timing": copy.deepcopy(rollout._timing),
        "native_usage": copy.deepcopy(rollout._native_usage_metrics),
        "trajectory": copy.deepcopy(rollout._trajectory[trajectory_start:]),
    }
    write_json_atomic(
        child_dir / "observation.json",
        redact_trajectory_obj(json.loads(json.dumps(observation, default=str))),
    )
