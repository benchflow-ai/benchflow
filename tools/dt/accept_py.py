"""Would Python BenchFlow launch these tasks? A check that starts no sandbox.

Each task goes through the loader ``bench eval run`` uses
(``task/formats.py`` ``materialize_task_dir``, line 185, which for a task.md
draft 2 package calls ``taskmd/materialize.py:192`` ``load_plan``; a
``TaskMdError`` there is ``refused_at_load``), then through the launch gate's
own check (``sandbox/setup.py`` ``_validate_task_runtime_for_launch``:
``validate_task_runtime_support(task, sandbox=..., task_dir=...)``,
``task/runtime_capabilities.py:77-109``). One JSON line per task:
``accepted``, ``refused`` (the unsupported features with their code) or
``refused_at_load`` (the planner's reasons), or ``unreadable``.

Unlike ``normalise.py`` and ``compare.py`` this imports benchflow on purpose:
it runs inside an arm's own environment, on the build VM only:

    <arm venv>/bin/python -I tools/dt/accept_py.py --sandbox docker <task>...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def accept(task_dir: Path, *, sandbox: str = "docker") -> dict[str, Any]:
    from benchflow.task import Task
    from benchflow.task.config import TaskConfig
    from benchflow.task.document import TaskDocument
    from benchflow.task.formats import materialize_task_dir
    from benchflow.task.runtime_capabilities import validate_task_runtime_support
    from benchflow.task.runtime_view import TaskRuntimeView
    from benchflow.taskmd import TaskMdError

    row: dict[str, Any] = {"task": task_dir.name, "sandbox": sandbox}
    try:
        native = materialize_task_dir(task_dir)
    except TaskMdError as exc:
        return {
            **row,
            "verdict": "refused_at_load",
            "reasons": list(exc.reasons),
            "message": str(exc),
        }
    try:
        task = Task(native)
    except Exception as exc:  # the loader's own refusal of a malformed folder
        return {
            **row,
            "verdict": "unreadable",
            "message": f"{type(exc).__name__}: {exc}",
        }
    document = getattr(task, "document", None)
    config = getattr(task, "config", None)
    if isinstance(document, TaskDocument):
        runtime_task: TaskDocument | TaskConfig = document
    elif isinstance(config, TaskConfig):
        runtime_task = config
    else:
        return {**row, "verdict": "unreadable", "message": "no task document or config"}
    view = TaskRuntimeView.from_task(task, task_dir=native)
    features = validate_task_runtime_support(
        runtime_task, sandbox=sandbox, task_dir=view.task_dir
    )
    if features:
        return {
            **row,
            "verdict": "refused",
            "features": [
                {
                    "path": f.path,
                    "reason": f.reason,
                    "code": getattr(f, "code", None),
                }
                for f in features
            ],
        }
    return {**row, "verdict": "accepted"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("tasks", nargs="+", type=Path)
    parser.add_argument("--sandbox", default="docker")
    args = parser.parse_args(argv)
    # The format cache (materialized task.md packages) goes to a folder of
    # this run's own, never the caller's ~/.cache.
    os.environ.setdefault(
        "BENCHFLOW_TASK_FORMAT_CACHE", tempfile.mkdtemp(prefix="dt-accept-")
    )
    for task_dir in args.tasks:
        print(json.dumps(accept(task_dir, sandbox=args.sandbox), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
