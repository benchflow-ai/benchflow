"""Discover task results without counting nested review/evidence runs."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchflow._utils.scoring import (
    ENDING_KEEPS_REWARD,
    assessment_status,
    assessment_withholds_score,
    classify_error,
    retry_category,
)

logger = logging.getLogger(__name__)

# Files an Evaluation (or branch) job writes in its folder: evaluation.json
# when it starts, summary.json when it ends (the only one in older jobs).
_JOB_RECORDS = ("evaluation.json", "summary.json")

# Where an Evaluation puts the trials of a job that a rerun with other settings
# replaced (evaluation._check_resume_mismatch): kept for the record, never read
# as the job's trials. A reader pointed at the folder itself still reads them.
SET_ASIDE_DIR = ".set-aside"
# result.json's mark on a trial that was stopped (Ctrl-C, SIGTERM, a budget)
# before it finished: the file keeps what the trial spent, but the
# trial is no finished attempt. Readers treat it as interrupted, as a folder
# with no result.json, and a resume runs the task again.
INTERRUPTED_MARK = "interrupted"


def is_interrupted_result(result: Any) -> bool:
    """Whether a result.json payload is a stopped trial's spend record."""
    return isinstance(result, dict) and result.get(INTERRUPTED_MARK) is True


def _reads_interrupted(path: Path) -> bool:
    try:
        return is_interrupted_result(json.loads(path.read_text()))
    except (OSError, ValueError):
        return False


def is_set_aside(path: Path, root: Path) -> bool:
    """Whether ``path`` lies in a set-aside folder below ``root``."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    return SET_ASIDE_DIR in parts


def holds_attempts(folder: Path) -> bool:
    """Whether results of one task in ``folder`` are attempts of one trial.

    An Evaluation job runs each task once and retries it, on an
    infrastructure error, in the same folder (a resume re-runs it there
    too), so its results for one task are attempts, and one of them is the
    trial's result. Any other folder, such as a ``bf.run_batch`` job or
    ``bf.run`` calls sharing a ``job_name``, holds independent rollouts:
    each is a sample of its own.
    """
    return any((folder / name).is_file() for name in _JOB_RECORDS)


def with_placeholder_settings(trial_dir: Path, result: Any) -> Any:
    """*result* with the agent and model of a trial that left no rollout folder.

    The job writes such a trial a folder (its result.json marked
    ``no_rollout_folder``), and a job written before that result.json
    recorded them has the agent and model only in the folder's config.json.
    Without them the trial is no attempt of its task's other trials, and
    each reader that groups attempts by task, agent and model counts it as
    one more trial. A script agent is named as a rollout's result.json names
    it, by its file (config.json keeps its host path). Any other result is
    returned as it is.
    """
    if not isinstance(result, dict) or result.get("no_rollout_folder") is not True:
        return result
    from benchflow.agents.scripts import script_label

    filled = dict(result)
    if not (result.get("agent") and result.get("model")):
        try:
            config = json.loads((trial_dir / "config.json").read_text())
        except (OSError, ValueError):
            config = None
        if isinstance(config, dict):
            for key in ("agent", "model"):
                if config.get(key) and not result.get(key):
                    filled[key] = config[key]
    if isinstance(filled.get("agent"), str):
        filled["agent"] = script_label(filled["agent"])
    return filled


def attempt_rank(result_path: Path, *, scored: bool) -> tuple[bool, float, str]:
    """How the attempts of one trial are ordered in a job with no
    ``attempt-decision.json`` records; the highest is the trial's result.

    A scored attempt wins, then the newest. Where the run recorded its
    decisions, they decide instead
    (:func:`benchflow._utils.attempt_decision.final_attempt`), for every
    reader: this order is only the rule for a job from before those records.
    """
    try:
        mtime = result_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (scored, mtime, str(result_path))


def is_job_summary(path: Path) -> bool:
    """Whether *path* is a job's own ``result.json`` (a task.toml-package
    runner writes one beside its trial folders), not a trial's: read as a
    trial, it hid every trial below it."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and "n_total_trials" in data
        and "stats" in data
        and "trial_name" not in data
    )


def iter_task_result_paths(root: Path) -> list[Path]:
    """Return result files at trial boundaries, excluding reviewer children.

    A result below a trial's result or solver checkpoint is an artifact of that
    parent, not an additional trial. In particular, captured files must remain
    excluded while the parent awaits review and has no final result yet. The
    explicit role also covers orphaned reviewer runs. Trials a rerun set aside
    (:data:`SET_ASIDE_DIR`) are not the job's. Corrupt task files are left to
    callers' existing error handling.
    """
    paths = [
        p
        for p in sorted(root.rglob("result.json"))
        if not is_set_aside(p, root) and not is_job_summary(p)
    ]
    trial_dirs = {path.parent for path in paths}
    trial_dirs.update(path.parent for path in root.rglob("solver.json"))
    selected: list[Path] = []
    for path in paths:
        if any(parent in trial_dirs for parent in path.parent.parents):
            continue
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            result = None
        if isinstance(result, dict) and result.get("purpose") == "reviewer":
            continue
        if is_interrupted_result(result):
            continue  # a stopped trial: interrupted, not a finished attempt
        selected.append(path)
    return selected


def interrupted_trial_results(root: Path) -> dict[str, dict[str, Any]]:
    """The spend records of the trials stopped before they finished
    (:data:`INTERRUPTED_MARK`), by trial folder name: summary.json's usage
    counts what they spent."""
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("result.json")):
        if is_set_aside(path, root) or is_job_summary(path):
            continue
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if is_interrupted_result(result):
            found[path.parent.name] = result
    return found


def kept_ending_is_final(result: Mapping[str, Any]) -> bool:
    """Whether a resume keeps *result* because the agent's turn had ended the
    trial before a later step failed (``post_turn_failure_info``).

    It does when the trial is scored under that ending (the agent's time
    limit, the host deadline, the trial budget, an output-cap ending, a model
    refusal): the later failure lost the reward, and running the trial again
    would pay for the agent again (``bench regrade`` re-runs the verifier
    only). An ending that leaves the trial unscored anyway does not keep it: a
    resume runs it again, as it runs that trial with nothing failing after
    it, whether or not the run retries the ending (it retries a provider
    error, not an agent that did nothing). Which is which is
    :data:`~benchflow._utils.scoring.ENDING_KEEPS_REWARD`; a category it does
    not list (a result from another version) is final.
    """
    if not result.get("post_turn_failure_info"):
        return False
    error = result.get("error")
    category = retry_category(
        error, result.get("error_category") or classify_error(error)
    )
    return category not in ENDING_KEEPS_REWARD or ENDING_KEEPS_REWARD[category]


def load_task_results(
    root: Path, *, include_terminal_errors: bool = False
) -> dict[str, dict[str, Any]]:
    """Load one durable result per task: the attempt the run kept.

    Evaluation resume and scoring-only summary refresh must select identical
    records. The attempt the run marked final
    (:mod:`benchflow._utils.attempt_decision`) is the task's result, and an
    attempt it retried never is, whatever reward it carries; in a job from
    before those records a scored attempt wins, then the newer. Malformed
    files remain logged and skipped as in evaluation resume; a malformed
    reward envelope remains visible as an errored trial. A result that
    declares an outcome assessment finished executing and is durable even
    while unscored: resume must never re-run an episode awaiting assessment.
    """
    from benchflow._utils.attempt_decision import FINAL, RETRIED, read_decision

    best: dict[str, tuple[tuple[bool, float, str], dict[str, Any]]] = {}
    # The attempt the run kept, by task: its order among the task's kept
    # attempts, and its result when a resume counts that result as finished.
    kept: dict[str, tuple[tuple[int, str], dict[str, Any] | None]] = {}
    for path in iter_task_result_paths(root):
        try:
            result = json.loads(path.read_text())
            task = result["task_name"]
            decision = read_decision(path.parent)
            if decision is not None and decision["decision"] == RETRIED:
                continue
            rewards = result.get("rewards")
            later = result.get("post_turn_failure_info")
            final_ending = kept_ending_is_final(result)
            verifier_error = result.get("verifier_error")
            if (
                isinstance(later, dict)
                and later.get("kind") == "verifier_crash"
                and not final_ending
            ):
                # The verify step's crash after an ending the trial is
                # unscored under anyway (kept as [solver-preserved]) is no
                # verdict: the trial runs again, as without the crash.
                verifier_error = None
            finished = not (
                rewards is None
                and not verifier_error
                and result.get("scoring") is None
                and assessment_status(result) is None
                # A step that failed after the agent's turn had ended the
                # trial: an ending the trial is scored under is final, and a
                # resume that ran the trial again would pay for the agent
                # again (regrade re-runs the verifier only).
                and not final_ending
                and not (include_terminal_errors and result.get("error"))
                # The model's refusal is its answer: a resume that ran the
                # trial again would re-roll it.
                and not result.get("model_refusal")
            )
            if finished and rewards is not None and not isinstance(rewards, dict):
                logger.warning(
                    "Malformed rewards field in %s for task %r: "
                    "expected dict or null, got %s %r — "
                    "treating as no reward (task will count as errored)",
                    path,
                    task,
                    type(rewards).__name__,
                    rewards,
                )
            if decision is not None and decision["decision"] == FINAL:
                order = (decision["sequence"], str(path))
                if task not in kept or order >= kept[task][0]:
                    kept[task] = (order, result if finished else None)
                continue
            if not finished:
                continue
            scored = rewards is not None and not assessment_withholds_score(result)
            rank = (scored, path.stat().st_mtime, str(path))
            previous = best.get(task)
            if previous is None or rank >= previous[0]:
                best[task] = (rank, result)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.debug("Skipping corrupt result file %s: %s", path, exc)
    results = {task: result for task, (_, result) in best.items() if task not in kept}
    results.update(
        {task: result for task, (_, result) in kept.items() if result is not None}
    )
    return results
