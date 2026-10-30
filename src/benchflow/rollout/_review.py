"""Terminal review lifecycle hooks, separate from solver execution."""

from __future__ import annotations

import logging
import shlex
from datetime import datetime
from typing import TYPE_CHECKING

from benchflow._utils.text import describe_exception
from benchflow.agents.credentials import credential_evidence_overrides
from benchflow.review.automatic import finish_review, prepare_review
from benchflow.review.evidence import capture_task_evidence
from benchflow.review.evidence_runtime import ensure_evidence_python
from benchflow.review.outcome import scoring_error
from benchflow.review.persistence import commit_scoring_result, scoring_lock
from benchflow.rollout._separate_verifier import scan_outside_workspace_writes
from benchflow.task.config import ArtifactConfig
from benchflow.task.verifier_sandbox import separate_verifier_requested

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


def _freeze_requested(rollout: Rollout) -> bool:
    """Whether the terminal workspace must be frozen into ``evidence/``.

    A rubric review grades it, a reviewer rollout is it, and a separate
    verifier sandbox receives nothing else from the agent's sandbox.
    """
    return (
        rollout._review_plan is not None
        or rollout._config.purpose == "reviewer"
        or separate_verifier_requested(getattr(rollout._task, "config", None))
    )


async def prepare_capture_runtime(rollout: Rollout) -> None:
    """Ensure required capture dependencies exist before solver execution."""
    if _freeze_requested(rollout):
        await ensure_evidence_python(
            rollout._env, timeout_sec=rollout._config.sandbox_setup_timeout
        )


def _separate(rollout: Rollout) -> bool:
    return separate_verifier_requested(getattr(rollout._task, "config", None))


async def mark_agent_phase_start(rollout: Rollout) -> None:
    """Read the sandbox clock as harness setup ends and the agent phase begins.

    Only for a separate verifier: files written after this instant outside
    everything that crosses are listed when the workspace is frozen.
    """
    if not _separate(rollout):
        return
    try:
        result = await rollout._env.exec(
            shlex.join(
                ["python3", "-I", "-c", "import time; print(repr(time.time()))"]
            ),
            user="root",
            timeout_sec=30,
        )
        rollout._agent_phase_started = float((result.stdout or "").split()[-1])
    except Exception:
        logger.warning(
            "Could not read the sandbox clock; files written outside the "
            "workspace will not be listed",
            exc_info=True,
        )


async def _list_outside_workspace_writes(rollout: Rollout) -> None:
    since = getattr(rollout, "_agent_phase_started", None)
    if since is None or not _separate(rollout):
        return
    cfg = rollout._config
    keep = [rollout._agent_cwd, *cfg.separate_verifier_include]
    if cfg.sandbox_user:
        keep.append(f"/home/{cfg.sandbox_user}")
    report = await scan_outside_workspace_writes(rollout._env, since=since, keep=keep)
    rollout._outside_workspace_writes = report
    if report and report.get("count"):
        logger.warning(
            "%d file(s) written outside the workspace %s during the agent phase "
            "will not reach the separate verifier (by directory: %s; e.g. %s). "
            "If the verifier reads them, pass --separate-verifier-include for "
            "their directory.",
            report["count"],
            rollout._agent_cwd,
            report.get("by_directory"),
            ", ".join(report.get("sample", [])[:5]),
        )


async def capture_terminal_workspace(rollout: Rollout) -> None:
    """Freeze solver or reviewer files before verifier hardening mutates them."""
    if not _freeze_requested(rollout):
        return
    includes: list[ArtifactConfig] = []
    if _separate(rollout):
        # Captured beside the workspace and carried into the verifier sandbox
        # at the same paths; the review sees them as outputs outside it.
        includes = [
            ArtifactConfig(source=path)
            for path in rollout._config.separate_verifier_include
        ]
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
            artifacts=[*rollout._task.config.artifacts, *includes],
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
        return
    await _list_outside_workspace_writes(rollout)


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
