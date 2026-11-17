"""The run's decision about each attempt of a trial, kept with the attempt.

An Evaluation job runs a task and, on a retryable error, runs it again: its
results for one task are attempts, and one of them is the trial's result.
Which one was worked out again by every reader (the run, a resume,
``bf.load_job``, regrade), each by its own rule, so one job gave different
scores: an attempt the run retried because its agent went idle could still
carry a reward, and a reader that preferred "a scored attempt" turned the
errored trial into a passed one.

The run's retry decision is the record. When the run retries an attempt it
writes ``retried`` into that attempt's folder; the attempt it keeps gets
``final``. An attempt the run retried is never the trial's result. A resume
that runs the task again writes a later ``final`` (a higher ``sequence``),
which supersedes the earlier one. A job from before these records has none,
and its readers fall back to their earlier rule. A reader of an attempt that
has no folder to read a record from (a ``results.jsonl`` row) supplies the
label its row carries instead (:func:`row_decision`).
"""

from __future__ import annotations

import glob
import json
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DECISION_FILE = "attempt-decision.json"
FINAL = "final"
RETRIED = "retried"
#: An attempt that is not its task's result though the run did not retry it:
#: a later attempt was kept in its place (a resume ran the task again).
SUPERSEDED = "superseded"
#: The key a job-level export row carries what its attempt is under.
ATTEMPT_DECISION_FIELD = "attempt_decision"


def read_decision(trial_dir: Path) -> dict[str, Any] | None:
    """The decision recorded in *trial_dir*, or None: no record, or one that
    cannot be read (then the attempt is as undecided as in an older job)."""
    try:
        record = json.loads((trial_dir / DECISION_FILE).read_text())
    except (OSError, ValueError):
        return None
    if (
        not isinstance(record, dict)
        or record.get("decision") not in (FINAL, RETRIED)
        or not isinstance(record.get("sequence"), int)
        or isinstance(record.get("sequence"), bool)
    ):
        return None
    return record


def carries_decisions(folder: Path) -> bool:
    """Whether a rollout folder in *folder* holds a readable decision record:
    the run that wrote it retried and kept attempts there."""
    return any(
        read_decision(path.parent) is not None
        for path in folder.glob(f"*/{DECISION_FILE}")
    )


def groups_attempts(folder: Path) -> bool:
    """Whether the rollouts of one task in *folder* are attempts of one trial.

    They are in an Evaluation job folder
    (:func:`benchflow._utils.result_paths.holds_attempts`), and in a folder
    that carries the run's records (:func:`carries_decisions`): a job folder
    copied without its evaluation.json and summary.json still holds the
    attempts the run retried and the one it kept. Any other folder
    (``bf.run_batch``, ``bf.run`` calls sharing a job name) holds samples,
    each a trial of its own.
    """
    from benchflow._utils.result_paths import holds_attempts

    return holds_attempts(folder) or carries_decisions(folder)


def _next_sequence(job_dir: Path, task: str) -> int:
    """One more than the highest sequence among *task*'s decisions in the job."""
    highest = 0
    for path in job_dir.glob(f"{glob.escape(task)}__*/{DECISION_FILE}"):
        record = read_decision(path.parent)
        if record is not None and record.get("task") == task:
            highest = max(highest, record["sequence"])
    return highest + 1


def record_decision(trial_dir: Path, decision: str, *, task: str, attempt: int) -> None:
    """Write the run's *decision* (``final`` or ``retried``) for the attempt
    in *trial_dir*. ``attempt`` counts this run's tries of the task from 1;
    ``sequence`` orders every decision the job ever made for the task."""
    from benchflow._utils.disk_space import write_text_atomic

    if decision not in (FINAL, RETRIED):
        raise ValueError(f"not an attempt decision: {decision!r}")
    record = {
        "schema_version": 1,
        "decision": decision,
        "task": task,
        "attempt": attempt,
        "sequence": _next_sequence(trial_dir.parent, task),
        "decided_at": datetime.now(UTC).isoformat(),
    }
    write_text_atomic(trial_dir / DECISION_FILE, json.dumps(record, indent=2) + "\n")


def decision_of(trial_dir: Path) -> str | None:
    """``final`` or ``retried`` as the run recorded it in *trial_dir*; None
    with no record."""
    record = read_decision(trial_dir)
    return None if record is None else record["decision"]


def row_decision(row: Any) -> dict[str, Any] | None:
    """The decision a job-level export row carries (its
    :data:`ATTEMPT_DECISION_FIELD`) as a record the picker reads; None when
    the row has no label (a job from before these records).

    A row carries the label alone, never a ``sequence``: the export gives
    ``final`` to the kept attempt of a task and to no other one
    (:func:`chain_decisions`), so a ``final`` row has no earlier ``final`` to
    be ordered against, and a ``retried`` or ``superseded`` row is never the
    trial's result.
    """
    label = row.get(ATTEMPT_DECISION_FIELD) if isinstance(row, dict) else None
    if label not in (FINAL, RETRIED, SUPERSEDED):
        return None
    return {"decision": label, "sequence": 0}


def _reader[T](
    folder: Callable[[T], Path], record: Callable[[T], dict[str, Any] | None] | None
) -> Callable[[T], dict[str, Any] | None]:
    return (
        record if record is not None else lambda attempt: read_decision(folder(attempt))
    )


def kept_attempt[T](
    attempts: Sequence[T],
    *,
    folder: Callable[[T], Path],
    record: Callable[[T], dict[str, Any] | None] | None = None,
) -> T | None:
    """The attempt the run kept among the *attempts* of one task: of those
    marked ``final``, the latest. None when none is marked.

    Each attempt's decision is read from its *folder*; *record*, when given,
    supplies it instead, for attempts that have no folder of their own
    (:func:`row_decision` for a ``results.jsonl`` row)."""
    recorded = _reader(folder, record)
    kept = []
    for index, attempt in enumerate(attempts):
        decision = recorded(attempt)
        if decision is not None and decision["decision"] == FINAL:
            kept.append((decision["sequence"], str(folder(attempt)), index))
    return attempts[max(kept)[2]] if kept else None


def not_retried[T](
    attempts: Sequence[T],
    *,
    folder: Callable[[T], Path],
    record: Callable[[T], dict[str, Any] | None] | None = None,
) -> list[T]:
    """The *attempts* the run did not retry, nor replace by a later attempt
    it kept (a ``superseded`` row): what a reader's earlier rule picks among
    when the run kept none. *record* as for :func:`kept_attempt`."""
    recorded = _reader(folder, record)
    return [
        a
        for a in attempts
        if (decision := recorded(a)) is None
        or decision["decision"] not in (RETRIED, SUPERSEDED)
    ]


def final_attempt[T](
    attempts: Sequence[T],
    *,
    folder: Callable[[T], Path],
    earlier_rule: Callable[[Sequence[T]], T | None],
    record: Callable[[T], dict[str, Any] | None] | None = None,
) -> T | None:
    """The trial's result among the *attempts* of one task, or None.

    The attempt the run kept (:func:`kept_attempt`). With none marked (a job
    from before these records, or a run that stopped before it decided),
    *earlier_rule* picks among the attempts the run did not retry
    (:func:`not_retried`). None when every attempt was retried and none kept:
    the trial has no result yet, and its task is interrupted, not finished.
    *record* supplies the decision of an attempt that has no folder to read
    it from (:func:`kept_attempt`).
    """
    kept = kept_attempt(attempts, folder=folder, record=record)
    if kept is not None:
        return kept
    undecided = not_retried(attempts, folder=folder, record=record)
    return earlier_rule(undecided) if undecided else None


def _task_of(trial_dir: Path) -> str:
    """The task an attempt folder ran: its result's, else its record's, else
    the folder's name before its ``__<id>`` suffix."""
    try:
        result = json.loads((trial_dir / "result.json").read_text())
    except (OSError, ValueError):
        result = None
    name = result.get("task_name") if isinstance(result, dict) else None
    if not isinstance(name, str) or not name:
        record = read_decision(trial_dir)
        name = record.get("task") if record is not None else None
    if not isinstance(name, str) or not name:
        name = trial_dir.name.rsplit("__", 1)[0]
    return name


def chain_decisions(trial_dirs: Iterable[Path]) -> dict[Path, str]:
    """What each attempt folder of one job folder is among the attempts of
    its task, for a reader that sees one attempt at a time (an export row).

    ``final`` for the attempt the run kept, and only that one: a task has one
    result. ``retried`` for an attempt the run retried. ``superseded`` for any
    other attempt of a task that has a kept one: an earlier ``final`` a resume
    replaced, or an attempt with no record. A folder is left out when its
    task has no kept attempt and the run did not retry it (a job from before
    these records).
    """
    chains: dict[str, list[Path]] = {}
    for trial_dir in trial_dirs:
        chains.setdefault(_task_of(trial_dir), []).append(trial_dir)
    labels: dict[Path, str] = {}
    for chain in chains.values():
        kept = kept_attempt(chain, folder=lambda trial_dir: trial_dir)
        for trial_dir in chain:
            if kept is not None and trial_dir == kept:
                labels[trial_dir] = FINAL
            elif decision_of(trial_dir) == RETRIED:
                labels[trial_dir] = RETRIED
            elif kept is not None:
                labels[trial_dir] = SUPERSEDED
    return labels


def job_decisions(job_dir: Path, trial_dirs: Iterable[Path]) -> dict[Path, str]:
    """:func:`chain_decisions` for an export of the job folder *job_dir*
    whose rows come from the attempt folders *trial_dirs*, worked out among
    every attempt of the job and not only those with a row.

    A kept attempt can have no export file: a trial that failed before its
    rollout folder existed gets only the result.json the job writes for it.
    It is still the task's result, so an earlier ``final`` (a resume ran the
    task again) is ``superseded``, never a second result.
    """
    folders = set(trial_dirs)
    folders.update(path.parent for path in job_dir.glob("*/result.json"))
    folders.update(path.parent for path in job_dir.glob(f"*/{DECISION_FILE}"))
    return chain_decisions(sorted(folders))
