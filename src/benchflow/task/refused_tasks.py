"""Tasks BenchFlow refuses before the agent's turn, unscored.

A task here has a reference solution that fails under one of BenchFlow's
always-on protections, so a 0 for it would be BenchFlow's doing, not the
agent's: seta-env 1243's solution starts postfix during the agent phase, and
under BenchFlow's root-exec limit the daemons it starts cannot use the setgid
``postdrop``, so mail delivery fails and test.sh's check hangs. Such a task is
refused before the turn for every agent and is never scored.

The list ships in the package (``refused_tasks.json``, read with
:mod:`importlib.resources`); a task or a workspace cannot supply or change it.
Each entry is keyed by benchmark, task id and BenchFlow's own task digest
(:func:`benchflow._utils.task_authoring.task_digest`, over every file of the
task folder), so an edited task does not inherit its entry: one changed byte
gives another digest, and that task runs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from benchflow._utils.scoring import TASK_REFUSED, TASK_REFUSED_HEADER

__all__ = [
    "RefusedTask",
    "TaskRefusedError",
    "refusal_for",
    "refuse_listed_task",
    "refused_tasks",
    "refused_tasks_summary",
]


@dataclass(frozen=True)
class RefusedTask:
    """One entry: the task it names and why BenchFlow will not score it."""

    benchmark: str
    task: str
    digest: str
    reason: str


class TaskRefusedError(RuntimeError):
    """A listed task, refused before the turn: unscored, never retried."""

    def __init__(self, entry: RefusedTask) -> None:
        self.entry = entry
        super().__init__(
            f"{TASK_REFUSED_HEADER}{entry.benchmark} {entry.task}: {entry.reason}"
        )


@cache
def refused_tasks() -> tuple[RefusedTask, ...]:
    """The package's list, read once."""
    text = (
        resources.files("benchflow.task")
        .joinpath("refused_tasks.json")
        .read_text(encoding="utf-8")
    )
    fields = ("benchmark", "task", "digest", "reason")
    return tuple(
        RefusedTask(**{name: str(entry[name]) for name in fields})
        for entry in json.loads(text)["tasks"]
    )


def refusal_for(task_path: str | Path) -> RefusedTask | None:
    """The entry for the task folder at ``task_path``, or None.

    The task id is the folder's name; only a folder whose name an entry
    names is hashed, and its digest must be the entry's.
    """
    path = Path(task_path).resolve()
    entries = [entry for entry in refused_tasks() if entry.task == path.name]
    if not entries:
        return None
    from benchflow._utils.task_authoring import task_digest

    digest = task_digest(path)
    return next((entry for entry in entries if entry.digest == digest), None)


def refuse_listed_task(task_path: str | Path) -> None:
    """Raise :class:`TaskRefusedError` when the task is on the list."""
    entry = refusal_for(task_path)
    if entry is not None:
        raise TaskRefusedError(entry)


def refused_tasks_summary(
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """summary.json's list of the trials refused before the turn, each with
    its reason; {} when none. They stay in the task count, unscored."""
    rows = [
        {"task": str(name), "reason": str(result.get("error") or "")}
        for name, result in sorted(results.items())
        if result.get("error_category") == TASK_REFUSED
    ]
    if not rows:
        return {}
    return {"refused_before_turn": {"count": len(rows), "trials": rows}}
