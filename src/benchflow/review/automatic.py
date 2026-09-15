"""Automatic terminal rubric review: planning, execution, and provenance.

The solver lifecycle owns evidence capture. This module consumes a finalized
solver record, reuses the ordinary reviewer runtime, and returns one typed
scoring outcome. It never launches or reruns the solver.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import shutil
import tempfile
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path

from benchflow._utils.text import describe_exception
from benchflow.review.config import Rubric, find_task_rubrics, load_rubric_snapshot
from benchflow.review.options import ReviewerConfig
from benchflow.review.outcome import ScoringResult, complete_scoring, scoring_error
from benchflow.review.persistence import write_json_atomic

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedReview:
    """Validated task identity and reviewer settings, including private auth."""

    task_path: Path
    task_digest: str
    rubric: Rubric
    rubric_path: Path
    rubric_sha256: str
    config: ReviewerConfig

    def metadata(self) -> dict:
        return {
            "rubric_path": str(self.rubric_path.relative_to(self.task_path)),
            "rubric_sha256": self.rubric_sha256,
            "contract": self.rubric.contract,
            "reviewer": self.config.to_config_artifact(),
        }


def prepare_review(task_path: Path, reviewer: ReviewerConfig) -> PreparedReview | None:
    """Detect the task rubric and reject missing reviewer auth before solving."""
    from benchflow._utils.task_authoring import task_digest
    from benchflow.agents.env import resolve_agent_env
    from benchflow.evaluation import effective_model
    from benchflow.review.preflight import validate_reviewer_backend

    task_path = Path(task_path).resolve()
    candidates = find_task_rubrics(task_path)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(
            "Multiple task review rubrics are ambiguous: "
            + ", ".join(map(str, candidates))
        )
    rubric_path = candidates[0]
    rubric, rubric_contents = load_rubric_snapshot(rubric_path)
    if not rubric.is_weighted:
        raise ValueError(
            f"Automatic review requires a v0.2 weighted rubric: {rubric_path}. "
            "Add explicit blocker and weight fields to every criterion."
        )
    try:
        model = effective_model(reviewer.agent, reviewer.model)
        if not model:
            raise ValueError("specify --reviewer-model for this harness")
        resolved_env = resolve_agent_env(reviewer.agent, model, reviewer.agent_env)
    except ValueError as exc:
        raise ValueError(f"Reviewer preflight for {rubric_path}: {exc}") from exc
    config = reviewer.model_copy(update={"model": model, "agent_env": resolved_env})
    validate_reviewer_backend(config)
    return PreparedReview(
        task_path=task_path,
        task_digest=task_digest(task_path),
        rubric=rubric,
        rubric_path=rubric_path,
        rubric_sha256=hashlib.sha256(rubric_contents).hexdigest(),
        config=config,
    )


def _stage_review_task(plan: PreparedReview, out_dir: Path) -> Path:
    """Bind the reviewer to a durable task copy before it waits for capacity.

    Hash the copied regular files, not just the live source tree. A concurrent
    edit can therefore either preserve the original identity or reject this
    attempt, but never silently substitute different task evidence.
    """
    from benchflow._utils.task_authoring import task_digest

    target = out_dir / "task-input"
    with tempfile.TemporaryDirectory(prefix=".task-input-", dir=out_dir) as temporary:
        staged = Path(temporary) / "task"
        shutil.copytree(plan.task_path, staged, symlinks=True)
        if task_digest(staged) != plan.task_digest:
            raise ValueError("Task contents changed after reviewer preflight")
        copied_rubric = staged / plan.rubric_path.relative_to(plan.task_path)
        rubric, contents = load_rubric_snapshot(copied_rubric)
        if (
            hashlib.sha256(contents).hexdigest() != plan.rubric_sha256
            or rubric != plan.rubric
        ):
            raise ValueError("Rubric contents changed after reviewer preflight")
        # The original rubric is retained on the host for reproducibility;
        # the wrapper still uploads only its guidance and response contract.
        (out_dir / "rubric.json").write_bytes(contents)
        staged.rename(target)
    return target


_LIMITS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _review_limit(config: ReviewerConfig) -> asyncio.Semaphore:
    """Share reviewer capacity across trials without leaking event loops."""
    loop = asyncio.get_running_loop()
    limits = _LIMITS.setdefault(loop, {})
    key = (config.environment, config.concurrency)
    return limits.setdefault(key, asyncio.Semaphore(config.concurrency))


async def finish_review(plan: PreparedReview, rollout_dir: Path) -> ScoringResult:
    """Judge immutable solver evidence and retain every attempt's full output."""
    from benchflow.review.evidence import EvidenceManifest, validate_workspace
    from benchflow.review.runner import run_review

    source = json.loads((rollout_dir / "solver.json").read_text())
    raw_reward = (source.get("rewards") or {}).get("reward")
    verifier_reward = (
        float(raw_reward)
        if isinstance(raw_reward, int | float)
        and not isinstance(raw_reward, bool)
        and math.isfinite(raw_reward)
        else None
    )
    tests_pass = verifier_reward == 1.0 if verifier_reward is not None else None
    attempt = uuid.uuid4().hex
    out_dir = rollout_dir / "reviews" / attempt
    out_dir.mkdir(parents=True, exist_ok=True)
    outcome: ScoringResult
    details: dict = {"attempt": attempt, **plan.metadata()}
    try:
        if source.get("task_digest") != plan.task_digest:
            raise ValueError("Solver and reviewer task digests do not match")
        if source.get("verifier_error") or verifier_reward is None:
            raise ValueError(
                source.get("verifier_error")
                or "Deterministic verifier produced no reward"
            )
        if source.get("partial_trajectory"):
            raise ValueError(
                "Solver trajectory is incomplete; review cannot establish a complete score"
            )
        if source.get("export_error"):
            raise ValueError(source["export_error"])
        bundle = rollout_dir / "evidence"
        manifest = EvidenceManifest.model_validate_json(
            (bundle / "manifest.json").read_text()
        )
        await asyncio.to_thread(validate_workspace, bundle / "workspace", manifest)
        task_snapshot = await asyncio.to_thread(_stage_review_task, plan, out_dir)
        details.update(
            task_digest=plan.task_digest,
            task_input=str(task_snapshot.relative_to(rollout_dir)),
            rubric_snapshot=str((out_dir / "rubric.json").relative_to(rollout_dir)),
            workspace_archive_sha256=manifest.archive_sha256,
        )
        async with _review_limit(plan.config):
            trial = await run_review(
                rollout_dir=rollout_dir,
                task_dir=task_snapshot,
                rubric=plan.rubric,
                rubric_path=plan.rubric_path,
                config=plan.config,
                out_dir=out_dir,
                deterministic_pass=bool(tests_pass),
                workspace_bundle=bundle,
            )
        details.update(
            {
                "checks": trial.checks,
                "summary": trial.summary,
                "review_valid": trial.review_valid,
            }
        )
        reviewer_run = (
            str(Path(trial.reviewer_rollout).relative_to(rollout_dir))
            if trial.reviewer_rollout
            else None
        )
        if not trial.review_valid or trial.scoring is None or reviewer_run is None:
            outcome = scoring_error(
                trial.error or "Reviewer did not produce a complete valid review",
                tests_pass=tests_pass,
                verifier_reward=verifier_reward,
                reviewer_run=reviewer_run,
            )
        else:
            outcome = complete_scoring(
                trial.scoring,
                verifier_reward=verifier_reward,
                reviewer_run=reviewer_run,
            )
    except Exception as exc:
        logger.exception("Automatic review failed for %s", rollout_dir.name)
        outcome = scoring_error(
            describe_exception(exc),
            tests_pass=tests_pass,
            verifier_reward=verifier_reward,
        )
    outcome = outcome.model_copy(update={"revision": f"scoring/{attempt}.json"})
    details["scoring"] = outcome.to_dict()
    write_json_atomic(out_dir / "review_report.json", details)
    write_json_atomic(rollout_dir / "scoring" / f"{attempt}.json", details)
    return outcome
