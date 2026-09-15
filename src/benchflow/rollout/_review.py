"""Terminal review lifecycle hooks, separate from solver execution."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from benchflow._utils.text import describe_exception
from benchflow.agents.credentials import credential_evidence_overrides
from benchflow.review.automatic import finish_review, prepare_review
from benchflow.review.evidence import capture_task_evidence
from benchflow.review.evidence_runtime import ensure_evidence_python
from benchflow.review.outcome import scoring_error
from benchflow.review.persistence import commit_scoring_result, scoring_lock

if TYPE_CHECKING:
    from benchflow.models import RolloutResult
    from benchflow.rollout import Rollout

logger = logging.getLogger(__name__)


def prepare_terminal_review(rollout: Rollout) -> None:
    """Preflight only ordinary scored tasks; review children never recurse."""
    cfg = rollout._config
    if cfg.purpose == "reviewer" or cfg.skip_verify:
        return
    rollout._review_plan = prepare_review(cfg.task_path, cfg.reviewer)
    if rollout._review_plan is not None:
        digest = rollout._review_plan.task_digest
        if cfg.task_digest is not None and cfg.task_digest != digest:
            raise ValueError(
                "Task digest differs from the task selected for automatic review"
            )
        cfg.task_digest = digest


async def prepare_capture_runtime(rollout: Rollout) -> None:
    """Ensure required capture dependencies exist before solver execution."""
    if rollout._review_plan is not None or rollout._config.purpose == "reviewer":
        await ensure_evidence_python(
            rollout._env, timeout_sec=rollout._config.sandbox_setup_timeout
        )


async def capture_terminal_workspace(rollout: Rollout) -> None:
    """Freeze solver or reviewer files before verifier hardening mutates them."""
    if rollout._review_plan is None and rollout._config.purpose != "reviewer":
        return
    try:
        await rollout.disconnect()
        if rollout._config.sandbox_user:
            await rollout._planes.quiesce_agent(
                rollout._env, rollout._config.sandbox_user
            )
        await capture_task_evidence(
            rollout._env,
            rollout._agent_cwd,
            rollout._require_rollout_dir() / "evidence",
            artifacts=rollout._task.config.artifacts,
            excluded_paths=credential_evidence_overrides(
                rollout._agent_env,
                workspace=rollout._agent_cwd,
                cred_home=(
                    f"/home/{rollout._config.sandbox_user}"
                    if rollout._config.sandbox_user
                    else "/root"
                ),
            ),
        )
    except Exception as exc:
        # Tests may still provide useful diagnostics. A capture error prevents
        # final rubric scoring and must not become a solver capability error.
        rollout._export_error = f"Workspace evidence capture failed: {exc}"
        logger.exception("Workspace evidence capture failed")


async def finish_terminal_review(
    rollout: Rollout, *, result: RolloutResult | None = None
) -> RolloutResult:
    """Review after cleanup finalized telemetry and released the solver VM."""
    assert rollout._review_plan is not None
    rollout._phase = "reviewing"
    # This phase record cannot be mistaken for a completed parent result.
    if result is None:
        result = rollout._build_result(result_filename="solver.json")
    try:
        with scoring_lock(rollout._require_rollout_dir()):
            scoring = await finish_review(
                rollout._review_plan, rollout._require_rollout_dir()
            )
            payload = commit_scoring_result(rollout._require_rollout_dir(), scoring)
    except Exception as exc:
        # A failed score write must not escape as a retryable solver crash.
        # solver.json remains available for review-only recovery even if the
        # filesystem cannot accept the final result at this moment.
        logger.exception("Could not commit automatic scoring")
        scoring = scoring_error("Scoring commit failed: " + describe_exception(exc))
        rollout._scoring = result.scoring = scoring
        rollout._rewards = result.rewards = None
        rollout._verifier_error = result.verifier_error = scoring.error
        rollout._phase = "cleaned"
        return result
    rollout._scoring = result.scoring = scoring
    rollout._rewards = result.rewards = scoring.numeric_rewards()
    rollout._verifier_error = result.verifier_error = payload.get("verifier_error")
    result.verifier_error_category = payload.get("verifier_error_category")
    result.finished_at = datetime.fromisoformat(payload["finished_at"])
    rollout._timing = payload["timing"]
    rollout._phase = "cleaned"
    return result
