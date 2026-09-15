"""Shared reviewer execution and detached review reporting.

``run_reviews`` takes a path that is either one rollout directory or a job
directory containing many, assembles one wrapper task per rollout (see
:mod:`benchflow.review.wrapper`), runs every wrapper as an ordinary rollout
on the selected sandbox backend, and writes ``review_report.json``.

Both entry points copy evidence and retain complete reviewer child runs.
``run_reviews`` is report-only; automatic terminal scoring calls ``run_review``
with an explicit deterministic verdict before the parent result is committed.  A wrapper rollout's own reward means only
"the reviewer produced a structurally valid result file"; the graded
outcomes live in the report.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from benchflow.review.config import (
    REVIEW_RESULT_FILENAME,
    ReviewRubricError,
    Rubric,
    build_review_response_model,
    find_task_rubric,
    load_rubric,
)
from benchflow.review.options import ReviewerConfig
from benchflow.review.outcome import deterministic_pass as source_deterministic_pass
from benchflow.review.scoring import ReviewScoring, score_weighted_review
from benchflow.review.wrapper import (
    REVIEWER_AGENT_TIMEOUT_SEC,
    REVIEWER_ARTIFACT_PACKAGES,
    REVIEWER_IMAGE,
    assemble_review_task,
    remove_review_task,
)

logger = logging.getLogger(__name__)

REVIEW_REPORT_FILENAME = "review_report.json"

_OUTCOME_KEYS = ("pass", "fail", "not_applicable")
_TASK_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# Report models


class ReviewRunError(ValueError):
    """Raised when the review input path or filters are unusable."""


@dataclass
class TrialReview:
    """One reviewed rollout."""

    trial_name: str
    source_rollout: str
    review_valid: bool = False
    summary: str | None = None
    checks: dict[str, dict[str, Any]] | None = None
    error: str | None = None
    reviewer_rollout: str | None = None
    rubric_path: str | None = None
    criteria: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rubric_contract: str | None = None
    criterion_metadata: list[dict[str, str | int | None]] = field(default_factory=list)
    scoring: ReviewScoring | None = None

    def outcome_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {key: 0 for key in _OUTCOME_KEYS}
        for check in (self.checks or {}).values():
            outcome = check.get("outcome")
            if isinstance(outcome, str) and outcome in counts:
                counts[outcome] += 1
        return counts


@dataclass
class ReviewReport:
    """Everything one ``bench review`` invocation produced."""

    path: str
    rubric_path: str
    criteria: list[str]
    agent: str
    model: str | None
    environment: str
    network: str = "no-internet"
    job_summary: str | None = None
    trials: list[TrialReview] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rubric": {
                "path": self.rubric_path,
                "criteria": self.criteria,
                "contracts": list(
                    dict.fromkeys(
                        trial.rubric_contract
                        for trial in self.trials
                        if trial.rubric_contract is not None
                    )
                ),
            },
            "reviewer": {
                "agent": self.agent,
                "model": self.model,
                "environment": self.environment,
                "network": self.network,
            },
            "job_summary": self.job_summary,
            "trials": [
                {
                    "trial_name": trial.trial_name,
                    "source_rollout": trial.source_rollout,
                    "review_valid": trial.review_valid,
                    "summary": trial.summary,
                    "checks": trial.checks,
                    "error": trial.error,
                    "reviewer_rollout": trial.reviewer_rollout,
                    "rubric_path": trial.rubric_path,
                    "rubric_contract": trial.rubric_contract,
                    "criteria": trial.criteria,
                    "criterion_metadata": trial.criterion_metadata,
                    "scoring": trial.scoring.to_dict() if trial.scoring else None,
                    "notes": trial.notes,
                }
                for trial in self.trials
            ],
        }


# Source and rubric discovery


def _is_rollout_dir(path: Path) -> bool:
    return (path / "config.json").is_file() or (path / "result.json").is_file()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _is_passing(rollout_dir: Path) -> bool:
    """Read integrated gates; keep detached review's strict legacy error rule."""

    from benchflow._utils.scoring import classify_score_outcome

    result = _read_json(rollout_dir / "result.json")
    if result is None:
        return False
    if result.get("scoring") is not None:
        return classify_score_outcome(result) == "passed"
    return source_deterministic_pass(result)


def discover_rollouts(
    path: Path,
    *,
    filter_passing: bool | None = None,
) -> list[Path]:
    """Resolve ``path`` to the rollout directories to review."""

    if not path.exists():
        raise ReviewRunError(f"path does not exist: {path}")
    if _is_rollout_dir(path):
        rollouts = [path]
    else:
        rollouts = sorted(
            child
            for child in path.iterdir()
            if child.is_dir() and _is_rollout_dir(child)
        )
        if not rollouts:
            raise ReviewRunError(
                f"{path} is neither a rollout directory (config.json/result.json) "
                "nor a job directory containing rollout directories"
            )
    if filter_passing is True:
        rollouts = [rollout for rollout in rollouts if _is_passing(rollout)]
    elif filter_passing is False:
        rollouts = [rollout for rollout in rollouts if not _is_passing(rollout)]
    if not rollouts:
        qualifier = (
            "passing "
            if filter_passing is True
            else ("failing " if filter_passing is False else "")
        )
        raise ReviewRunError(f"no {qualifier}rollout directories found in {path}")
    return rollouts


def _source_task_dir(
    rollout_dir: Path,
    *,
    tasks_root: Path | None,
) -> tuple[Path | None, str | None]:
    """Resolve the reviewed task's directory, or explain why it was skipped.

    ``config.json`` is rollout-authored data, not a trusted input: a
    downloaded rollout can name **any** host directory (``~/.ssh``, a
    checkout, a directory holding ``.env``) and this code would otherwise
    copy it wholesale into the reviewer sandbox. So the recorded path is
    only ever used as a *name to look up beneath an operator-supplied
    root* (``--tasks-root``): the basename must resolve inside that root,
    and no path outside it is ever read. Without ``--tasks-root`` the task
    copy is skipped entirely and the reviewer works from run records alone.
    """

    config = _read_json(rollout_dir / "config.json") or {}
    recorded = config.get("task_path")
    if not isinstance(recorded, str) or not recorded:
        return None, None
    name = PurePosixPath(recorded.replace("\\", "/")).name
    if not name:
        return None, None
    if tasks_root is None:
        return None, (
            f"task evidence skipped: rollout names task {name!r} but no "
            "--tasks-root was given (a path recorded inside a rollout is "
            "untrusted and is never read directly)"
        )
    root = tasks_root.resolve()
    candidate = (root / name).resolve()
    if root not in candidate.parents or not candidate.is_dir():
        return None, (
            f"task evidence skipped: {name!r} does not resolve to a "
            f"directory inside --tasks-root {root}"
        )
    return candidate, None


def _task_digest_issue(rollout_dir: Path, task_dir: Path) -> str | None:
    """Compare the rollout's recorded task digest against the task on disk.

    Task evidence is admitted only when result/config provenance identifies
    one valid digest and the trusted on-disk task matches it.
    """

    recorded_by_source: dict[str, str] = {}
    for filename in ("result.json", "config.json"):
        document = _read_json(rollout_dir / filename)
        if document is None or "task_digest" not in document:
            continue
        recorded = document["task_digest"]
        if not isinstance(recorded, str) or not _TASK_DIGEST_RE.fullmatch(recorded):
            return f"{filename} carries an invalid task_digest"
        recorded_by_source[filename] = recorded
    if not recorded_by_source:
        return "task digest is missing from both result.json and config.json"
    recorded_values = set(recorded_by_source.values())
    if len(recorded_values) != 1:
        return "result.json and config.json carry conflicting task digests"
    recorded = next(iter(recorded_values))
    try:
        from benchflow._utils.task_authoring import task_digest

        actual = task_digest(task_dir)
    except Exception as exc:
        # Fail closed: a rollout that RECORDED a digest is claiming a
        # specific task; if that claim cannot be verified, the task must
        # not be admitted as evidence.
        return f"task digest could not be verified ({exc!r})"
    if actual != recorded:
        return (
            f"task digest mismatch: rollout recorded {recorded}, "
            f"--tasks-root copy is {actual}"
        )
    return None


def _resolve_rubric(
    explicit_rubric: tuple[Rubric, Path] | None,
    task_dir: Path | None,
) -> tuple[Rubric, Path]:
    """Resolution order: explicit ``-r`` > the task's own rubric > default."""

    if explicit_rubric is not None:
        return explicit_rubric
    if task_dir is not None:
        shipped = find_task_rubric(task_dir)
        if shipped is not None:
            return load_rubric(shipped), shipped
    from benchflow.review.config import DEFAULT_RUBRIC_PATH

    return load_rubric(None), DEFAULT_RUBRIC_PATH


# Reviewer artifact ingestion


def _coerce_summary(value: Any) -> str | None:
    """Reviewer output is untrusted; only a string survives as the summary."""

    if value is None or isinstance(value, str):
        return value
    return str(value)


def _coerce_checks(value: Any) -> dict[str, dict[str, Any]] | None:
    """Normalize invalid reviewer checks for diagnostic display only.

    Even invalid reviews are retained for diagnostics, so a criterion whose
    value is a list (or anything else non-dict) must not raise later in
    table rendering. Weighted reviews cross the typed boundary in
    :func:`_validate_review_payload` and never use this lossy representation
    for scoring; legacy reviews retain their wrapper-authoritative behavior.
    """

    if not isinstance(value, dict):
        return None
    normalized: dict[str, dict[str, Any]] = {}
    for name, check in value.items():
        if isinstance(check, dict):
            normalized_check: dict[str, Any] = {
                "explanation": _coerce_summary(check.get("explanation"))
            }
            if "outcome" in check:
                normalized_check["outcome"] = check.get("outcome")
            if "score" in check:
                normalized_check["score"] = check.get("score")
            normalized[str(name)] = normalized_check
        else:
            normalized[str(name)] = {"explanation": str(check)}
    return normalized


def _validate_review_payload(
    rubric: Rubric,
    review: dict[str, Any],
    *,
    expected_trial_name: str,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Validate and canonicalize one artifact before it can be scored."""

    response = build_review_response_model(rubric).model_validate(review)
    payload = response.model_dump(mode="json")
    actual_trial_name = payload["trial_name"]
    if actual_trial_name != expected_trial_name:
        raise ValueError(
            f"trial_name must be {expected_trial_name!r}, got {actual_trial_name!r}"
        )
    return payload["summary"], payload["checks"]


def _reviewer_rollout_leaf(runtime_dir: Path) -> Path | None:
    """Return the single rollout directory this invocation produced.

    ``runtime_dir`` is unique per (invocation, source rollout), so exactly one
    reviewer rollout may exist beneath it. Anything else — zero because the
    run died before creating it, more than one because something unexpected
    wrote into the directory — is treated as no result rather than guessed
    at. Artifacts are then read from that exact leaf; there is deliberately
    no recursive artifact discovery that could pick up stale files.
    """

    leaves = sorted(
        candidate.parent for candidate in runtime_dir.glob("*/*/config.json")
    )
    return leaves[0] if len(leaves) == 1 else None


def _leaf_review_result(leaf: Path) -> dict[str, Any] | None:
    for relative in (
        Path("verifier") / REVIEW_RESULT_FILENAME,
        Path("artifacts") / REVIEW_RESULT_FILENAME,
    ):
        data = _read_json(leaf / relative)
        if data is not None:
            return data
    return None


def _leaf_reward(leaf: Path) -> float | None:
    result = _read_json(leaf / "result.json")
    if result is None:
        return None
    rewards = result.get("rewards")
    if isinstance(rewards, dict):
        reward = rewards.get("reward")
        if isinstance(reward, int | float) and not isinstance(reward, bool):
            return float(reward)
    return None


def _reviewer_completion_error(leaf: Path) -> str | None:
    """A valid JSON verdict does not make an interrupted reviewer complete."""
    result = _read_json(leaf / "result.json")
    if result is None:
        return "reviewer result is missing or unreadable"
    for error_field in ("error", "verifier_error", "export_error"):
        if result.get(error_field):
            return f"reviewer {error_field}: {result[error_field]}"
    if result.get("partial_trajectory"):
        return "reviewer trajectory is incomplete"
    return None


# Review orchestration and aggregation


async def _review_one(
    rollout_dir: Path,
    *,
    explicit_rubric: tuple[Rubric, Path] | None,
    template: str | None,
    config: ReviewerConfig,
    tasks_root: Path | None,
    out_dir: Path,
) -> TrialReview:
    trial = TrialReview(
        trial_name=rollout_dir.name,
        source_rollout=str(rollout_dir),
    )
    task_dir, skip_reason = _source_task_dir(rollout_dir, tasks_root=tasks_root)
    if skip_reason:
        trial.notes.append(skip_reason)
        logger.info("%s: %s", rollout_dir.name, skip_reason)
    if task_dir is not None:
        digest_issue = await asyncio.to_thread(
            _task_digest_issue,
            rollout_dir,
            task_dir,
        )
        if digest_issue:
            # Enforced, not merely noted: reviewing an old rollout against
            # changed task content silently misattributes findings, so the
            # mismatched task is dropped from evidence entirely.
            trial.notes.append(digest_issue + " — task evidence excluded")
            logger.warning("%s: %s", rollout_dir.name, digest_issue)
            task_dir = None
    try:
        rubric, resolved_rubric = _resolve_rubric(explicit_rubric, task_dir)
    except ReviewRubricError as exc:
        trial.error = str(exc)
        return trial
    reviewed = await run_review(
        rollout_dir,
        task_dir,
        rubric,
        resolved_rubric,
        config,
        out_dir,
        deterministic_pass=source_deterministic_pass(
            _read_json(rollout_dir / "result.json") or {}
        ),
        template=template,
    )
    reviewed.notes[:0] = trial.notes
    return reviewed


async def run_review(
    rollout_dir: Path,
    task_dir: Path | None,
    rubric: Rubric,
    rubric_path: Path,
    config: ReviewerConfig,
    out_dir: Path,
    deterministic_pass: bool,
    workspace_bundle: Path | None = None,
    *,
    template: str | None = None,
) -> TrialReview:
    """Run one reviewer over admitted evidence, independently of parent persistence.

    Automatic scoring passes the deterministic verdict explicitly and can call
    this before its final result exists. Detached review resolves untrusted task
    provenance before calling the same operation. Child artifacts are durable;
    only the generated wrapper and its evidence upload copies are temporary.
    """
    from benchflow import run as run_rollout
    from benchflow.rollout import RolloutConfig

    trial = TrialReview(trial_name=rollout_dir.name, source_rollout=str(rollout_dir))
    wrapper_root = Path(tempfile.mkdtemp(prefix="benchflow-review-"))
    trial.rubric_path = str(rubric_path)
    trial.rubric_contract = rubric.contract
    trial.criteria = [criterion.name for criterion in rubric.criteria]
    trial.criterion_metadata = [criterion.metadata() for criterion in rubric.criteria]

    wrapper_dir = wrapper_root / f"review-{rollout_dir.name}"
    # Unique per invocation: reusing --out-dir must never let this run see a
    # previous run's reviewer artifacts.
    runtime_dir = out_dir / "runtime" / rollout_dir.name / uuid.uuid4().hex[:12]
    try:
        _, uploads = await asyncio.to_thread(
            assemble_review_task,
            rollout_dir,
            task_dir,
            rubric,
            wrapper_dir,
            template=template,
            image=config.image,
            agent_timeout_sec=config.timeout_sec,
            open_network=config.open_network,
            net_admin_overlay=(
                config.environment == "docker" and not config.open_network
            ),
            workspace_bundle=workspace_bundle,
        )
        from benchflow.review.persistence import write_json_atomic

        write_json_atomic(
            runtime_dir / "reviewer-environment.json",
            {
                "image": config.image,
                "artifact_packages": list(REVIEWER_ARTIFACT_PACKAGES)
                if config.image == REVIEWER_IMAGE
                else [],
                "open_network": config.open_network,
            },
        )
        shutil.copyfile(wrapper_dir / "task.md", runtime_dir / "reviewer-task.md")
        # The wrapper task declares allow_internet: false, which engages the
        # no-web pipeline end to end: web tools disabled, the model proxy
        # forced sandbox-local, and the agent-UID egress firewall scoped to
        # that loopback gateway.
        hooks = [_lock_review_evidence]
        if workspace_bundle is not None:
            from benchflow.review.evidence import (
                EvidenceManifest,
                install_review_evidence,
            )

            manifest = EvidenceManifest.model_validate_json(
                (workspace_bundle / "manifest.json").read_text()
            )
            hooks.insert(0, partial(install_review_evidence, manifest=manifest))
        rollout_config = RolloutConfig(
            task_path=wrapper_dir,
            agent=config.agent,
            model=config.model,
            agent_env=dict(config.agent_env),
            environment=config.environment,
            reasoning_effort=config.reasoning_effort,
            purpose="reviewer",
            parent_rollout=rollout_dir.name,
            jobs_dir=runtime_dir,
            timeout=config.timeout_sec,
            uploads=uploads,
            pre_agent_hooks=hooks,
        )
        result = await run_rollout(rollout_config)
        leaf = _reviewer_rollout_leaf(runtime_dir)
        trial.reviewer_rollout = str(leaf) if leaf else None
        reward = _leaf_reward(leaf) if leaf else None
        completion_error = _reviewer_completion_error(leaf) if leaf else None
        if result.error and completion_error is None:
            completion_error = f"reviewer error: {result.error}"
        trial.review_valid = reward == 1.0 and completion_error is None
        review = _leaf_review_result(leaf) if leaf else None
        if review is None:
            trial.error = (
                f"reviewer did not produce a readable {REVIEW_RESULT_FILENAME} "
                f"(agent error: {result.error})"
                if result.error
                else f"reviewer did not produce a readable {REVIEW_RESULT_FILENAME}"
            )
            return trial
        trial.summary = _coerce_summary(review.get("summary"))
        trial.checks = _coerce_checks(review.get("checks"))
        if not trial.review_valid:
            trial.error = (
                completion_error or "reviewer output failed structural validation"
            )
        elif rubric.is_weighted:
            try:
                trial.summary, trial.checks = _validate_review_payload(
                    rubric,
                    review,
                    expected_trial_name=rollout_dir.name,
                )
            except (ValidationError, ValueError) as exc:
                trial.review_valid = False
                trial.error = (
                    f"reviewer output failed host-side structural validation: {exc}"
                )
        if trial.review_valid and rubric.is_weighted and trial.checks is not None:
            try:
                trial.scoring = score_weighted_review(
                    rubric,
                    trial.checks,
                    deterministic_pass=deterministic_pass,
                )
            except ValueError as exc:
                # Defense in depth for an internal aggregation invariant.
                trial.review_valid = False
                trial.error = (
                    f"reviewer output failed host-side structural validation: {exc}"
                )
        logger.info(
            "Reviewed %s with rubric %s (valid=%s)",
            rollout_dir.name,
            rubric_path,
            trial.review_valid,
        )
    except Exception as exc:
        logger.error("Review failed for %s", rollout_dir.name, exc_info=True)
        trial.error = str(exc)
        leaf = _reviewer_rollout_leaf(runtime_dir)
        if leaf is not None:
            trial.reviewer_rollout = str(leaf)
    finally:
        remove_review_task(wrapper_root)
    return trial


async def _lock_review_evidence(sandbox: Any) -> None:
    """Make uploaded evidence readable but immutable before the agent starts."""

    result = await sandbox.exec(
        "chown -R 0:0 /evidence && chmod -R a-w,a+rX /evidence",
        user="root",
        timeout_sec=120,
    )
    if result.return_code != 0:
        raise RuntimeError(
            "failed to lock reviewer evidence read-only: "
            f"{(result.stderr or result.stdout or '')[:300]}"
        )


def _summarize_job(trials: list[TrialReview]) -> str | None:
    """Deterministic cross-run aggregation.

    Intentionally not an LLM call: a host-side model call would bypass the
    sandbox backend, the reviewer egress policy, ``agent_env``, and normal
    telemetry. Consumers wanting prose synthesis can run it over the report.
    """

    # Only structurally valid reviews contribute verdict counts; rejected
    # output stays in the report as diagnostics but must not move the
    # job-level numbers.
    reviewed = [trial for trial in trials if trial.checks and trial.review_valid]
    if len(reviewed) < 2:
        return None
    total = {key: 0 for key in _OUTCOME_KEYS}
    per_criterion: dict[str, dict[str, int]] = {}
    for trial in reviewed:
        for name, check in (trial.checks or {}).items():
            outcome = check.get("outcome")
            if isinstance(outcome, str) and outcome in total:
                total[outcome] += 1
                bucket = per_criterion.setdefault(
                    name, {key: 0 for key in _OUTCOME_KEYS}
                )
                bucket[outcome] += 1
    weighted = [trial.scoring for trial in reviewed if trial.scoring is not None]
    if weighted:
        lines = [
            f"{len(reviewed)} of {len(trials)} runs reviewed (valid reviews only).",
            "Binary judgments (legacy criteria and weighted blockers only): "
            f"{total['pass']} pass, {total['fail']} fail, "
            f"{total['not_applicable']} not applicable.",
        ]
    else:
        lines = [
            f"{len(reviewed)} of {len(trials)} runs reviewed (valid reviews only): "
            f"{total['pass']} pass, {total['fail']} fail, "
            f"{total['not_applicable']} not applicable across all criteria."
        ]
    for name in sorted(per_criterion):
        bucket = per_criterion[name]
        lines.append(
            f"{name}: {bucket['pass']} pass / {bucket['fail']} fail / "
            f"{bucket['not_applicable']} n-a"
        )
    if weighted:
        decision_counts: dict[str, int] = {}
        for scoring in weighted:
            key = scoring.decision.value
            decision_counts[key] = decision_counts.get(key, 0) + 1
        average_quality = sum(item.raw_quality for item in weighted) / len(weighted)
        decisions = ", ".join(
            f"{name}={decision_counts[name]}" for name in sorted(decision_counts)
        )
        lines.append(
            f"weighted reviews: average raw quality {average_quality:.3f}; {decisions}."
        )
    failures = [trial.trial_name for trial in trials if trial.error]
    if failures:
        lines.append("reviews with errors: " + ", ".join(sorted(failures)))
    return "\n".join(lines)


# Public API


async def run_reviews(
    path: Path,
    *,
    agent: str,
    model: str | None = None,
    environment: str = "docker",
    rubric_path: Path | None = None,
    prompt_path: Path | None = None,
    agent_env: dict[str, str] | None = None,
    concurrency: int = 4,
    timeout_sec: int = REVIEWER_AGENT_TIMEOUT_SEC,
    image: str = REVIEWER_IMAGE,
    open_network: bool = False,
    tasks_root: Path | None = None,
    filter_passing: bool | None = None,
    out_dir: Path | None = None,
) -> tuple[ReviewReport, Path]:
    """Review rollout(s) at ``path`` and return the report plus its location."""

    path = Path(path).resolve()
    rollouts = discover_rollouts(path, filter_passing=filter_passing)
    template = prompt_path.read_text(encoding="utf-8") if prompt_path else None
    explicit_rubric = (
        (load_rubric(rubric_path), rubric_path) if rubric_path is not None else None
    )
    if model is None:
        from benchflow.evaluation import effective_model

        try:
            model = effective_model(agent, None) or None
        except ValueError as exc:
            raise ReviewRunError(
                f"agent {agent!r} has no registry default model ({exc}); pass "
                "--model with a gateway model id such as "
                "'gemini/gemini-2.5-flash'"
            ) from None
        if model is None:
            raise ReviewRunError(
                f"agent {agent!r} has no registry default model; pass --model "
                "with a gateway model id such as 'gemini/gemini-2.5-flash'"
            )

    if out_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        out_dir = Path("jobs") / f"review-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    reviewer = ReviewerConfig(
        agent=agent,
        model=model,
        environment=environment,
        agent_env=agent_env or {},
        timeout_sec=timeout_sec,
        image=image,
        open_network=open_network,
        concurrency=max(1, concurrency),
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def bounded(rollout_dir: Path) -> TrialReview:
        async with semaphore:
            return await _review_one(
                rollout_dir,
                explicit_rubric=explicit_rubric,
                template=template,
                config=reviewer,
                tasks_root=tasks_root,
                out_dir=out_dir,
            )

    trials = await asyncio.gather(*(bounded(rollout) for rollout in rollouts))

    trials.sort(key=lambda trial: trial.trial_name)
    rubric_for_report = rubric_path or Path("<per-task or default>")
    if explicit_rubric is not None:
        criteria_names = [criterion.name for criterion in explicit_rubric[0].criteria]
    else:
        # Jobs can contain tasks with different rubrics. The report-level list
        # is a deterministic union; each trial remains the authoritative
        # mapping for its own checks.
        criteria_names = list(
            dict.fromkeys(name for trial in trials for name in trial.criteria)
        )
    report = ReviewReport(
        path=str(path),
        rubric_path=str(rubric_for_report),
        criteria=criteria_names,
        agent=agent,
        model=model,
        environment=environment,
        network="open (explicit --allow-open-network)"
        if open_network
        else "no-internet",
        trials=trials,
    )
    report.job_summary = _summarize_job(trials)

    report_path = out_dir / REVIEW_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return report, report_path
