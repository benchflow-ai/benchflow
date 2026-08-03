"""Post-verifier rubric-review engine using a separate reviewer sandbox."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow.agents.errors import AgentProtocolError
from benchflow.review.config import (
    REVIEW_RUBRIC_FILENAME,
    ReviewCriterion,
    ReviewParams,
    ReviewRubric,
    ReviewRubricError,
    find_review_rubric,
    load_review_rubric,
)
from benchflow.review.prompts import render_retry_prompt, render_review_prompt
from benchflow.review.runtime import (
    IsolatedReviewerRuntime,
    capture_evidence_snapshot,
)
from benchflow.review.scoring import (
    STATUS_COMPROMISED,
    STATUS_CONFIG_ERROR,
    STATUS_ERROR,
    STATUS_SCORED,
    CriterionVerdict,
    ReviewOutcome,
    aggregate,
    parse_reviewer_message,
)

if TYPE_CHECKING:  # pragma: no cover
    from benchflow.rollout import Rollout

logger = logging.getLogger(__name__)

_MAX_PARSE_RETRIES = 2
_MAX_TRANSPORT_RETRIES = 2


def resolve_review_params(config_review: ReviewParams | None) -> ReviewParams:
    return config_review if config_review is not None else ReviewParams()


async def run_review_engine(rollout: Rollout) -> None:
    """Run one rubric review and always leave canonical status metadata."""

    params = resolve_review_params(getattr(rollout._config, "review", None))
    if params.enabled is False:
        return
    task = getattr(rollout, "_task", None)
    if task is None:
        return
    verifier_dir = task.paths.tests_dir
    rubric_path = find_review_rubric(verifier_dir)
    if rubric_path is None and params.enabled is None:
        return

    started_at = datetime.now()
    review_dir = rollout._require_rollout_dir() / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    rubric: ReviewRubric | None = None
    harness: str | None = None
    model: str | None = None
    mode: str | None = None
    timeout_sec: float | None = None
    reasoning_effort: str | None = None
    reviewer_meta: dict[str, Any] = {}
    outcome = ReviewOutcome(
        status=STATUS_ERROR,
        error="review aborted before completion",
    )
    try:
        if rubric_path is None:
            candidate = verifier_dir / REVIEW_RUBRIC_FILENAME
            if candidate.is_file():
                load_review_rubric(candidate)
            raise ReviewRubricError(
                "--review was requested but no versioned review rubric.json "
                f"exists under {verifier_dir}"
            )
        rubric = load_review_rubric(rubric_path)
        (
            harness,
            model,
            timeout_sec,
            mode,
            reasoning_effort,
        ) = _resolve_reviewer(params, rubric)
        outcome, reviewer_meta = await _run_isolated_review(
            rollout,
            rubric,
            rubric_path=rubric_path,
            harness=harness,
            model=model,
            timeout_sec=timeout_sec,
            mode=mode,
            reasoning_effort=reasoning_effort,
            review_dir=review_dir,
        )
    except ReviewRubricError as exc:
        logger.error("Rubric review configuration error: %s", exc)
        outcome = ReviewOutcome(status=STATUS_CONFIG_ERROR, error=str(exc))
    except Exception as exc:  # review infrastructure never changes execution reward
        logger.error("Rubric review failed", exc_info=True)
        outcome = ReviewOutcome(status=STATUS_ERROR, error=str(exc))
    finally:
        finished_at = datetime.now()
        duration = (finished_at - started_at).total_seconds()
        rollout._timing["review"] = duration
        updates = outcome.reward_updates(rubric)
        if updates and rollout._rewards is not None:
            rollout._rewards = {**rollout._rewards, **updates}
        details = _review_details(
            outcome=outcome,
            rubric=rubric,
            rubric_path=rubric_path,
            task_path=rollout._config.task_path,
            harness=harness,
            model=model,
            mode=mode,
            timeout_sec=timeout_sec,
            reviewer_meta=reviewer_meta,
            started_at=started_at,
            finished_at=finished_at,
        )
        details_path = review_dir / "review-details.json"
        details_path.write_text(
            json.dumps(details, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        rollout._review_metadata = {
            "status": outcome.status,
            "status_reason": outcome.error,
            "plan": outcome.plan,
            "plan_passed": outcome.passed,
            "reviewer_harness": harness,
            "reviewer_model": model,
            "rubric_sha256": details["rubric"]["sha256"],
            "details_path": "review/review-details.json",
            "isolation": details["isolation"],
        }

    if outcome.status == STATUS_SCORED:
        logger.info(
            "Plan review scored %.4f (passed=%s, gates_failed=%s)",
            outcome.plan,
            outcome.passed,
            outcome.failed_gates or "none",
        )
    else:
        logger.warning("Plan review %s: %s", outcome.status, outcome.error)


def _resolve_reviewer(
    params: ReviewParams,
    rubric: ReviewRubric,
) -> tuple[str, str | None, float, str, str | None]:
    """Resolve CLI overrides over rubric defaults; a harness is mandatory."""

    harness = params.harness or rubric.reviewer.harness
    if harness is None:
        raise ReviewRubricError(
            "reviewer harness is required: set reviewer.harness in rubric.json "
            "or pass --reviewer-harness"
        )
    model = params.model or rubric.reviewer.model
    if model is None:
        from benchflow.evaluation import effective_model

        model = effective_model(harness, None)
    timeout_sec = params.timeout_sec or rubric.reviewer.timeout_sec
    mode = params.mode or rubric.reviewer.mode
    return harness, model, timeout_sec, mode, params.reasoning_effort


def _integrity_criterion(token: str) -> ReviewCriterion:
    return ReviewCriterion(
        id="review-integrity-control",
        criterion=(
            "The read-only file /review/control/integrity-token.txt contains "
            f"the exact token {token}."
        ),
        criterion_type="failure-check",
        weight=0.0,
    )


async def _run_isolated_review(
    rollout: Rollout,
    rubric: ReviewRubric,
    *,
    rubric_path: Path,
    harness: str,
    model: str | None,
    timeout_sec: float,
    mode: str,
    reasoning_effort: str | None,
    review_dir: Path,
) -> tuple[ReviewOutcome, dict[str, Any]]:
    snapshot = await capture_evidence_snapshot(rollout)
    runtime = IsolatedReviewerRuntime(
        rollout,
        harness=harness,
        model=model,
        timeout_sec=timeout_sec,
        review_dir=review_dir,
        reasoning_effort=reasoning_effort,
    )
    reviewer_meta: dict[str, Any] = {}
    try:
        await runtime.start(snapshot, rubric_path)
        control = _integrity_criterion(snapshot.control_token)
        if mode == "batched":
            batches = [[*rubric.criteria, control]]
        else:
            batches = [[control], *[[criterion] for criterion in rubric.criteria]]

        verdicts: list[CriterionVerdict] = []
        control_verdict: CriterionVerdict | None = None
        for index, batch in enumerate(batches):
            if index > 0:
                await runtime.fresh_session()
            prompt = render_review_prompt(
                batch,
                task_prompt=(getattr(rollout._task, "instruction", "") or "").strip(),
                trajectory_files=snapshot.trajectory_files,
                first_batch=True,
            )
            batch_verdicts = await _grade_batch(runtime, batch, prompt)
            for verdict in batch_verdicts:
                if verdict.criterion_id == control.id:
                    control_verdict = verdict
                else:
                    verdicts.append(verdict)

        if (
            control_verdict is None
            or control_verdict.score is None
            or control_verdict.criterion_met is not True
        ):
            return (
                ReviewOutcome(
                    status=STATUS_COMPROMISED,
                    verdicts=verdicts,
                    error=(
                        "review integrity control failed; discarded all plan "
                        "verdicts as potentially compromised"
                    ),
                ),
                reviewer_meta,
            )
        return aggregate(rubric, verdicts), reviewer_meta
    finally:
        try:
            reviewer_meta.update(await runtime.close())
            reviewer_meta["trajectory_snapshot_files"] = snapshot.trajectory_files
        except Exception as exc:
            logger.error("Isolated reviewer cleanup failed", exc_info=True)
            reviewer_meta["cleanup_error"] = str(exc)
        snapshot.cleanup()


async def _grade_batch(
    runtime: IsolatedReviewerRuntime,
    batch: list[ReviewCriterion],
    prompt: str,
) -> list[CriterionVerdict]:
    error: str | None = None
    evidence_trace = ""
    for attempt in range(_MAX_PARSE_RETRIES + 1):
        message = prompt if attempt == 0 else render_retry_prompt(error or "")
        turn = await _prompt_with_transport_retries(runtime, message)
        evidence_trace += "\n" + turn.evidence_trace
        verdicts, error = parse_reviewer_message(
            turn.reply,
            batch,
            evidence_trace=evidence_trace,
        )
        if error is None:
            return verdicts
    return [
        CriterionVerdict(
            criterion_id=criterion.id,
            criterion_met=None,
            unscored_reason=f"reviewer reply could not be parsed: {error}",
        )
        for criterion in batch
    ]


async def _prompt_with_transport_retries(
    runtime: IsolatedReviewerRuntime,
    message: str,
):
    """Retry a reviewer prompt in a fresh session after transient ACP loss."""

    for attempt in range(_MAX_TRANSPORT_RETRIES + 1):
        try:
            return await runtime.prompt(message)
        except AgentProtocolError:
            if attempt == _MAX_TRANSPORT_RETRIES:
                raise
            logger.warning(
                "Reviewer ACP prompt failed on attempt %d/%d; retrying in a fresh session",
                attempt + 1,
                _MAX_TRANSPORT_RETRIES + 1,
                exc_info=True,
            )
            await runtime.fresh_session()
    raise AssertionError("unreachable")


def _review_details(
    *,
    outcome: ReviewOutcome,
    rubric: ReviewRubric | None,
    rubric_path: Path | None,
    task_path: Path,
    harness: str | None,
    model: str | None,
    mode: str | None,
    timeout_sec: float | None,
    reviewer_meta: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    criteria_by_id = {
        criterion.id: criterion for criterion in (rubric.criteria if rubric else ())
    }
    relative_rubric: str | None = None
    if rubric_path is not None:
        try:
            relative_rubric = str(rubric_path.relative_to(task_path))
        except ValueError:
            relative_rubric = rubric_path.name
    digest = (
        hashlib.sha256(rubric_path.read_bytes()).hexdigest()
        if rubric_path is not None and rubric_path.is_file()
        else None
    )
    return {
        "status": outcome.status,
        "plan": outcome.plan,
        "plan_passed": outcome.passed,
        "pass_threshold": rubric.pass_threshold if rubric else None,
        "failed_gates": outcome.failed_gates,
        "error": outcome.error,
        "reviewer": {
            "harness": harness,
            "model": model,
            "mode": mode,
            "timeout_sec": timeout_sec,
            **reviewer_meta,
        },
        "isolation": {
            "environment": "separate-sandbox",
            "sandbox_user": "reviewer",
            "network": "no-network",
            "evidence": "sanitized-read-only-copy",
            "oracle_mounted": False,
            "verifier_mounted": False,
        },
        "rubric": {
            "path": relative_rubric,
            "sha256": digest,
            "n_criteria": len(rubric.criteria) if rubric else 0,
        },
        "timing": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
        },
        "verdicts": [
            {
                "id": verdict.criterion_id,
                "explanation": verdict.explanation,
                "evidence": list(verdict.evidence),
                "criterion_met": verdict.criterion_met,
                "unscored_reason": verdict.unscored_reason,
                "criterion_type": (
                    criteria_by_id[verdict.criterion_id].criterion_type
                    if verdict.criterion_id in criteria_by_id
                    else None
                ),
                "gating": (
                    criteria_by_id[verdict.criterion_id].gating
                    if verdict.criterion_id in criteria_by_id
                    else None
                ),
                "weight": (
                    criteria_by_id[verdict.criterion_id].weight
                    if verdict.criterion_id in criteria_by_id
                    else None
                ),
            }
            for verdict in outcome.verdicts
        ],
    }
