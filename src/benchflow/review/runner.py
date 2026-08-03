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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow.review.config import (
    REVIEW_RESULT_FILENAME,
    ReviewRubricError,
    Rubric,
    find_task_rubric,
    load_rubric,
)
from benchflow.review.prompts import render_job_summary_prompt
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


def _source_task_dir(rollout_dir: Path) -> Path | None:
    config = _read_json(rollout_dir / "config.json") or {}
    task_path = config.get("task_path")
    if isinstance(task_path, str) and task_path:
        candidate = Path(task_path)
        if candidate.is_dir():
            return candidate
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


def _find_review_result(runtime_dir: Path) -> dict[str, Any] | None:
    for candidate in sorted(runtime_dir.rglob(REVIEW_RESULT_FILENAME)):
        data = _read_json(candidate)
        if data is not None:
            return data
    return None


def _reviewer_reward(runtime_dir: Path) -> float | None:
    for candidate in sorted(runtime_dir.rglob("result.json")):
        result = _read_json(candidate)
        if result is None:
            continue
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
    out_dir: Path,
    workdir: Path,
) -> TrialReview:
    from benchflow import run as run_rollout
    from benchflow.rollout import RolloutConfig

    trial = TrialReview(
        trial_name=rollout_dir.name,
        source_rollout=str(rollout_dir),
    )
    task_dir = _source_task_dir(rollout_dir)
    try:
        rubric, resolved_rubric = _resolve_rubric(rubric_path, task_dir)
    except ReviewRubricError as exc:
        trial.error = str(exc)
        return trial

    wrapper_dir = workdir / f"review-{rollout_dir.name}"
    runtime_dir = out_dir / "runtime" / rollout_dir.name
    try:
        _, uploads = assemble_review_task(
            rollout_dir,
            task_dir,
            rubric,
            wrapper_dir,
            template=template,
            image=image,
            agent_timeout_sec=timeout_sec,
        )
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
        trial.reviewer_rollout = str(runtime_dir)
        reward = _reviewer_reward(runtime_dir)
        trial.review_valid = reward == 1.0
        review = _find_review_result(runtime_dir)
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


async def _summarize_job(
    trials: list[TrialReview],
    *,
    agent: str,
    model: str | None,
) -> str | None:
    """Aggregate per-trial reviews into one prose summary via a plain LLM call."""

    reviewed = [trial for trial in trials if trial.checks]
    if len(reviewed) < 2:
        return None
    from benchflow.evaluation import effective_model

    summary_model = effective_model(agent, model)
    if summary_model is None:
        return None
    blocks = []
    for trial in reviewed:
        lines = [f"Run: {trial.trial_name}", f"  Summary: {trial.summary or ''}"]
        for name, check in (trial.checks or {}).items():
            lines.append(
                f"  {name}: {check.get('outcome')} — {check.get('explanation', '')}"
            )
        blocks.append("\n".join(lines))
    prompt = render_job_summary_prompt(blocks)
    try:
        from litellm import acompletion

        response = await acompletion(
            model=summary_model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content  # type: ignore[union-attr]
        return content.strip() if isinstance(content, str) else None
    except Exception:
        logger.warning("Job-level review summary failed", exc_info=True)
        return None


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

        model = effective_model(agent, None) or None
        if model is None:
            raise ReviewRunError(
                f"agent {agent!r} has no registry default model; pass --model "
                "(for opencode use models.dev provider/model ids, e.g. "
                "'google/gemini-2.5-flash')"
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
    report = ReviewReport(
        path=str(path),
        rubric_path=str(rubric_for_report),
        criteria=criteria_names,
        agent=agent,
        model=model,
        environment=environment,
        trials=trials,
    )
    report.job_summary = await _summarize_job(trials, agent=agent, model=model)

    report_path = out_dir / REVIEW_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return report, report_path
