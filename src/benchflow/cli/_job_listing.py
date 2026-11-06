"""The rows of ``bench eval list``: one per job, never one per folder.

A jobs folder holds jobs (``<jobs>/<job>/``), folders of jobs
(``<jobs>/<experiment>/<job>/``, as ``--jobs-dir jobs/<name>`` makes), and the
backward-compatible copy of the newest job's summary.json at its root. Each
job is listed by its own summary.json (a partial one when the run was
stopped), or, when it has none (interrupted) or a live process still runs it,
by its evaluation.json and the trials it finished (DF-p3-eval-runner-6, -7;
DFR-p3-10).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow._utils.command_hints import command_arg

# Job folders are looked for this many levels below the listed folder: a
# --matrix or --trials run keeps them in <jobs>/<alias>/trial-NN/<job>.
_DEPTH = 4


@dataclass(frozen=True)
class JobRow:
    label: str
    tasks: str
    summary: str
    memory: float | None = None
    # A dim line after the table: how to finish an unfinished job.
    next_step: str | None = None


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _subdirs(folder: Path) -> list[Path]:
    try:
        return sorted(d for d in folder.iterdir() if d.is_dir())
    except OSError:
        return []


def _is_trial(folder: Path) -> bool:
    if (folder / "result.json").is_file():
        return True
    # An attempt that never finished: its config.json, in a task__<id> folder
    # (a job folder can hold a config.json of its own).
    return (folder / "config.json").is_file() and "__" in folder.name


def _trials(job: Path) -> list[Path]:
    return [d for d in _subdirs(job) if _is_trial(d)]


def _child_jobs(folder: Path) -> bool:
    return any(
        (d / "evaluation.json").is_file()
        or (d / "summary.json").is_file()
        or _trials(d)
        for d in _subdirs(folder)
    )


def _is_job(folder: Path) -> bool:
    if (folder / "evaluation.json").is_file() or _trials(folder):
        return True
    # A summary.json with jobs below it is the jobs folder's copy of the
    # newest job's summary, not a job.
    return (folder / "summary.json").is_file() and not _child_jobs(folder)


def _summary_row(label: str, data: Any, planned: int | None) -> JobRow:
    if not isinstance(data, dict):
        return JobRow(label, "?", "[dim]corrupt summary[/dim]")
    total = data.get("total", "?")
    # The scoring rule's passes (summary.json's "passed" is the audit count,
    # which names a verifier error first even when the trial was scored).
    scoring = data.get("scoring") if isinstance(data.get("scoring"), dict) else {}
    passed = scoring.get("passed", data.get("passed", "?"))
    text = f"{passed}/{total} ({data.get('score', '?')})"
    settings = scoring.get("settings") if isinstance(scoring, dict) else None
    if (isinstance(settings, dict) and settings.get("mixed")) or isinstance(
        data.get("restarted_after_change"), dict
    ):
        text += " [yellow]mixed settings[/yellow]"
    # summary.json's own plan, else evaluation.json's.
    planned = data.get("planned", planned)
    tasks = str(total)
    if isinstance(planned, int) and isinstance(total, int) and planned > total:
        # A budget or usage-limit stop: score is over what ran; the plan's
        # figure goes beside it.
        of_plan = data.get("score_of_plan")
        if not isinstance(of_plan, int | float):
            of_plan = passed / planned if isinstance(passed, int) and planned else 0.0
        left = f"{planned - total} not run"
        if data.get("stopped") is True:
            # A partial summary, written when the run was stopped (DFR-p3-11).
            left = (
                f"stopped: {data.get('interrupted', 0)} interrupted, "
                f"{data.get('not_started', 0)} not started"
            )
        text = (
            f"{passed}/{total} run ({data.get('score', '?')}), {passed}/{planned} "
            f"({of_plan:.1%}) of the plan; {left}"
        )
        tasks = str(planned)
    memory = data.get("memory_score")
    return JobRow(
        label,
        tasks,
        text,
        memory if isinstance(memory, int | float) else None,
    )


def _task_name(trial: Path) -> str:
    """The task a trial folder ran, as ``bench eval inspect`` names it: its
    result.json's task_name, else the attempt's config.json task_path."""
    from benchflow.jobs import _attempt_task_name

    result = _read(trial / "result.json")
    if isinstance(result, dict) and isinstance(result.get("task_name"), str):
        return result["task_name"]
    return _attempt_task_name(trial)


def _row(label: str, job: Path) -> JobRow:
    record = _read(job / "evaluation.json")
    planned = record.get("planned_tasks") if isinstance(record, dict) else None
    planned_n = len(planned) if isinstance(planned, list) else None
    from benchflow._utils.job_lock import live_holder

    pid = live_holder(job)
    if (job / "summary.json").is_file() and pid is None:
        data = _read(job / "summary.json")
        row = _summary_row(label, data, planned_n)
        if isinstance(data, dict) and data.get("stopped") is True:
            resume = f"`bench eval resume {command_arg(job, '<job dir>')}` finishes it"
            return JobRow(row.label, row.tasks, row.summary, row.memory, resume)
        return row
    trials = _trials(job)
    done = {_task_name(d) for d in trials if (d / "result.json").is_file()}
    if pid is not None:
        # A live process holds the job's lock (DFR-p3-10): nothing to resume.
        of = f" of {planned_n}" if planned_n is not None else ""
        text = f"[dim]running: {len(done)}{of} tasks have a result[/dim]"
        return JobRow(label, str(planned_n or len(trials)), text)
    if planned_n is not None:
        text = (
            f"[dim]not finished: {len(done)} of {planned_n} tasks have a result[/dim]"
        )
        tasks = str(planned_n)
    else:
        text = f"[dim]no summary ({len(done)} tasks have a result)[/dim]"
        tasks = str(len({_task_name(d) for d in trials}))
    next_step = (
        f"unfinished; `bench eval resume {command_arg(job, '<job dir>')}` finishes it"
        if isinstance(record, dict)
        else None
    )
    return JobRow(label, tasks, text, next_step=next_step)


def job_rows(jobs_dir: Path) -> list[JobRow]:
    """One row per job at or below ``jobs_dir`` (see the module docstring)."""
    if _is_job(jobs_dir):
        return [_row(jobs_dir.name, jobs_dir)]
    rows: list[JobRow] = []

    def walk(folder: Path, label: str, depth: int) -> bool:
        if _is_job(folder):
            rows.append(_row(label, folder))
            return True
        if depth >= _DEPTH:
            return False
        found = False
        for child in _subdirs(folder):
            found = walk(child, f"{label}/{child.name}", depth + 1) or found
        return found

    for child in _subdirs(jobs_dir):
        if not walk(child, child.name, 1):
            # A folder with no job in it: listed, as before, so nothing is
            # hidden.
            rows.append(
                JobRow(child.name, str(len(_subdirs(child))), "[dim]no job[/dim]")
            )
    return rows
