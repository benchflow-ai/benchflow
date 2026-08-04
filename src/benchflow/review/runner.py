"""Detached rubric review over finished rollout directories.

``run_reviews`` takes a path that is either one rollout directory or a job
directory containing many, assembles one wrapper task per rollout (see
:mod:`benchflow.review.wrapper`), runs every wrapper as an ordinary rollout
on the selected sandbox backend, and writes ``review_report.json``.

Reviews never touch the reviewed rollouts: evidence is copied, results live
under the review output directory, and the source ``result.json`` /
``rewards`` are read-only inputs.  A wrapper rollout's own reward means only
"the reviewer produced a structurally valid result file"; the graded
outcomes live in the report.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow.review.config import (
    REVIEW_RESULT_FILENAME,
    ReviewRubricError,
    Rubric,
    find_task_rubric,
    load_rubric,
)
from benchflow.review.wrapper import (
    REVIEWER_AGENT_TIMEOUT_SEC,
    REVIEWER_IMAGE,
    assemble_review_task,
)

logger = logging.getLogger(__name__)

REVIEW_REPORT_FILENAME = "review_report.json"

_OUTCOME_KEYS = ("pass", "fail", "not_applicable")


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

    def outcome_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {key: 0 for key in _OUTCOME_KEYS}
        for check in (self.checks or {}).values():
            outcome = check.get("outcome")
            if outcome in counts:
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
            "rubric": {"path": self.rubric_path, "criteria": self.criteria},
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
                    "criteria": trial.criteria,
                    "notes": trial.notes,
                }
                for trial in self.trials
            ],
        }


def _is_rollout_dir(path: Path) -> bool:
    return (path / "config.json").is_file() or (path / "result.json").is_file()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _is_passing(rollout_dir: Path) -> bool:
    """A rollout passes when it earned reward 1.0 and recorded no error.

    Anything unreadable counts as failing, so ``--failing`` sweeps in runs
    that crashed before writing a result.
    """

    result = _read_json(rollout_dir / "result.json")
    if result is None:
        return False
    rewards = result.get("rewards")
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    return reward == 1.0 and result.get("error") is None


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


def _task_digest_mismatch(rollout_dir: Path, task_dir: Path) -> str | None:
    """Compare the rollout's recorded task digest against the task on disk.

    A digest mismatch means the run was executed against different task
    content than the reviewer would see; that is reported per trial rather
    than silently reviewed.
    """

    result = _read_json(rollout_dir / "result.json") or {}
    recorded = result.get("task_digest")
    if not isinstance(recorded, str) or not recorded:
        return None
    try:
        from benchflow._utils.task_authoring import task_digest

        actual = task_digest(task_dir)
    except Exception:
        return None
    if actual and actual != recorded:
        return (
            f"task digest mismatch: rollout recorded {recorded}, "
            f"--tasks-root copy is {actual}"
        )
    return None


def _resolve_rubric(
    rubric_path: Path | None,
    task_dir: Path | None,
) -> tuple[Rubric, Path]:
    """Resolution order: explicit ``-r`` > the task's own rubric > default."""

    if rubric_path is not None:
        return load_rubric(rubric_path), rubric_path
    if task_dir is not None:
        shipped = find_task_rubric(task_dir)
        if shipped is not None:
            return load_rubric(shipped), shipped
    from benchflow.review.config import DEFAULT_RUBRIC_PATH

    return load_rubric(None), DEFAULT_RUBRIC_PATH


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
        candidate.parent
        for candidate in runtime_dir.rglob("config.json")
        if candidate.parent != runtime_dir
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
    if isinstance(rewards, dict) and "reward" in rewards:
        try:
            return float(rewards["reward"])
        except (TypeError, ValueError):
            return None
    return None


async def _review_one(
    rollout_dir: Path,
    *,
    rubric_path: Path | None,
    template: str | None,
    agent: str,
    model: str | None,
    environment: str,
    agent_env: dict[str, str],
    timeout_sec: int,
    image: str,
    open_network: bool,
    tasks_root: Path | None,
    out_dir: Path,
    workdir: Path,
) -> TrialReview:
    from benchflow import run as run_rollout
    from benchflow.rollout import RolloutConfig

    trial = TrialReview(
        trial_name=rollout_dir.name,
        source_rollout=str(rollout_dir),
    )
    task_dir, skip_reason = _source_task_dir(rollout_dir, tasks_root=tasks_root)
    if skip_reason:
        trial.notes.append(skip_reason)
        logger.info("%s: %s", rollout_dir.name, skip_reason)
    if task_dir is not None:
        mismatch = _task_digest_mismatch(rollout_dir, task_dir)
        if mismatch:
            # Enforced, not merely noted: reviewing an old rollout against
            # changed task content silently misattributes findings, so the
            # mismatched task is dropped from evidence entirely.
            trial.notes.append(mismatch + " — task evidence excluded")
            logger.warning("%s: %s", rollout_dir.name, mismatch)
            task_dir = None
    try:
        rubric, resolved_rubric = _resolve_rubric(rubric_path, task_dir)
    except ReviewRubricError as exc:
        trial.error = str(exc)
        return trial
    trial.rubric_path = str(resolved_rubric)
    trial.criteria = [criterion.name for criterion in rubric.criteria]

    wrapper_dir = workdir / f"review-{rollout_dir.name}"
    # Unique per invocation: reusing --out-dir must never let this run see a
    # previous run's reviewer artifacts.
    runtime_dir = out_dir / "runtime" / rollout_dir.name / uuid.uuid4().hex[:12]
    try:
        _, uploads = assemble_review_task(
            rollout_dir,
            task_dir,
            rubric,
            wrapper_dir,
            template=template,
            image=image,
            agent_timeout_sec=timeout_sec,
            open_network=open_network,
            net_admin_overlay=(environment == "docker" and not open_network),
        )
        # The wrapper task declares allow_internet: false, which engages the
        # no-web pipeline end to end: web tools disabled, the model proxy
        # forced sandbox-local, and the agent-UID egress firewall scoped to
        # that loopback gateway.
        config = RolloutConfig(
            task_path=wrapper_dir,
            agent=agent,
            model=model,
            agent_env=dict(agent_env),
            environment=environment,
            jobs_dir=runtime_dir,
            timeout=timeout_sec,
            uploads=uploads,
        )
        result = await run_rollout(config)
        leaf = _reviewer_rollout_leaf(runtime_dir)
        trial.reviewer_rollout = str(leaf) if leaf else str(runtime_dir)
        reward = _leaf_reward(leaf) if leaf else None
        trial.review_valid = reward == 1.0
        review = _leaf_review_result(leaf) if leaf else None
        if review is None:
            trial.error = (
                f"reviewer did not produce a readable {REVIEW_RESULT_FILENAME} "
                f"(agent error: {result.error})"
                if result.error
                else f"reviewer did not produce a readable {REVIEW_RESULT_FILENAME}"
            )
            return trial
        trial.summary = review.get("summary")
        checks = review.get("checks")
        trial.checks = checks if isinstance(checks, dict) else None
        if not trial.review_valid:
            trial.error = "reviewer output failed structural validation"
        logger.info(
            "Reviewed %s with rubric %s (valid=%s)",
            rollout_dir.name,
            resolved_rubric,
            trial.review_valid,
        )
    except Exception as exc:
        logger.error("Review failed for %s", rollout_dir.name, exc_info=True)
        trial.error = str(exc)
    finally:
        shutil.rmtree(wrapper_dir, ignore_errors=True)
    return trial


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
            if outcome in total:
                total[outcome] += 1
                bucket = per_criterion.setdefault(
                    name, {key: 0 for key in _OUTCOME_KEYS}
                )
                bucket[outcome] += 1
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
    failures = [trial.trial_name for trial in trials if trial.error]
    if failures:
        lines.append("reviews with errors: " + ", ".join(sorted(failures)))
    return "\n".join(lines)


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
    if rubric_path is not None:
        load_rubric(rubric_path)  # fail fast on a bad -r before any sandbox spend
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

    semaphore = asyncio.Semaphore(max(1, concurrency))
    workdir = Path(tempfile.mkdtemp(prefix="benchflow-review-"))

    async def bounded(rollout_dir: Path) -> TrialReview:
        async with semaphore:
            return await _review_one(
                rollout_dir,
                rubric_path=rubric_path,
                template=template,
                agent=agent,
                model=model,
                environment=environment,
                agent_env=agent_env or {},
                timeout_sec=timeout_sec,
                image=image,
                open_network=open_network,
                tasks_root=tasks_root,
                out_dir=out_dir,
                workdir=workdir,
            )

    try:
        trials = list(await asyncio.gather(*(bounded(rollout) for rollout in rollouts)))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    trials.sort(key=lambda trial: trial.trial_name)
    rubric_for_report = rubric_path or Path("<per-task or default>")
    criteria_names: list[str] = []
    if rubric_path is not None:
        criteria_names = [
            criterion.name for criterion in load_rubric(rubric_path).criteria
        ]
    elif trials:
        first = next((trial.criteria for trial in trials if trial.criteria), [])
        criteria_names = list(first)
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
