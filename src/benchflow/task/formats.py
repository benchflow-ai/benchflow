"""Task formats: task folders that BenchFlow runs natively after a load-time build step.

Some benchmarks keep their tasks in a compact source form and generate the rest of a native package
(``environment/``, ``verifier/``, ``oracle/``) from a shared runtime. A robouse task, for example, is a
``task.md`` with a ``robouse:`` block; its agent image, trusted simulator service and verifier are the same
for every task and are generated from the installed ``robouse`` package.

A *task format* claims such folders and materializes them as ordinary native task packages. Every entry
point that loads a task (``bench eval run --tasks-dir``, ``bench tasks check``, ``RolloutConfig`` in the
Python API) passes the folder through :func:`materialize_task_dir` first, so the rest of BenchFlow
(sandboxes, agents, verifier, trial layout, job tooling) only ever sees native packages.

Formats register through the ``benchflow.task_formats`` entry-point group (the value is an object, or a
zero-argument class, with ``name``, ``detect`` and ``materialize``)::

    [project.entry-points."benchflow.task_formats"]
    robouse = "robouse.benchflow_format:RobouseTaskFormat"

or in-process with :func:`register_task_format`.

``materialize`` must be deterministic and idempotent: it writes the native package under ``out_root``
(``$BENCHFLOW_TASK_FORMAT_CACHE/<format>``, default ``~/.cache/benchflow/task-formats/<format>``) and
returns its path. The returned folder's name is the task name BenchFlow uses in trial names and results.
The materialized package must not itself be claimed by any format.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "benchflow.task_formats"
CACHE_ENV = "BENCHFLOW_TASK_FORMAT_CACHE"


@runtime_checkable
class TaskFormat(Protocol):
    """A task source format that BenchFlow converts into a native task package on load."""

    name: str

    def detect(self, task_dir: Path) -> bool:
        """Whether *task_dir* is a task in this format. Must be cheap and must not raise."""
        ...

    def materialize(self, task_dir: Path, out_root: Path) -> Path:
        """Write the native task package for *task_dir* under *out_root* and return its folder."""
        ...


_registered: list[TaskFormat] = []
_entry_point_formats: list[TaskFormat] | None = None


def register_task_format(fmt: TaskFormat) -> None:
    """Register a task format in this process (entry points are loaded automatically)."""
    if not isinstance(fmt, TaskFormat):
        raise TypeError(
            f"{fmt!r} does not implement TaskFormat (name, detect, materialize)"
        )
    if all(existing.name != fmt.name for existing in _registered):
        _registered.append(fmt)


def _load_entry_points() -> list[TaskFormat]:
    global _entry_point_formats
    if _entry_point_formats is not None:
        return _entry_point_formats
    from importlib.metadata import entry_points

    loaded: list[TaskFormat] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
            fmt = obj() if isinstance(obj, type) else obj
        except Exception as exc:  # a broken plugin must not break native tasks
            logger.warning("task format %r failed to load: %s", ep.name, exc)
            continue
        if not isinstance(fmt, TaskFormat):
            logger.warning(
                "task format %r does not implement TaskFormat; ignored", ep.name
            )
            continue
        loaded.append(fmt)
    _entry_point_formats = loaded
    return loaded


def task_formats() -> list[TaskFormat]:
    """All known task formats: in-process registrations first, then entry points."""
    names = {fmt.name for fmt in _registered}
    return [
        *_registered,
        *(fmt for fmt in _load_entry_points() if fmt.name not in names),
    ]


def detect_task_format(task_dir: Path) -> TaskFormat | None:
    """The format that claims *task_dir*, or None for a native task (or a non-task)."""
    try:
        if not task_dir.is_dir():
            return None
    except OSError:
        return None
    for fmt in task_formats():
        try:
            if fmt.detect(task_dir):
                return fmt
        except Exception as exc:
            logger.warning(
                "task format %r failed to inspect %s: %s", fmt.name, task_dir, exc
            )
    return None


def task_format_cache_root() -> Path:
    env = os.environ.get(CACHE_ENV)
    return (
        Path(env).expanduser()
        if env
        else Path.home() / ".cache" / "benchflow" / "task-formats"
    )


def materialize_task_dir(task_dir: str | Path) -> Path:
    """Return the native task package for *task_dir*.

    A folder that no format claims is returned unchanged. A claimed folder is materialized under the
    format cache and the materialized folder is returned.
    """
    path = task_dir if isinstance(task_dir, Path) else Path(task_dir)
    fmt = detect_task_format(path)
    if fmt is None:
        return path  # unchanged, same object
    out_root = task_format_cache_root() / fmt.name
    out_root.mkdir(parents=True, exist_ok=True)
    native = Path(fmt.materialize(path.resolve(), out_root))
    if detect_task_format(native) is not None:
        raise RuntimeError(
            f"task format {fmt.name!r} materialized {path} as {native}, which is still claimed by a task format"
        )
    logger.debug("task format %s: %s -> %s", fmt.name, path, native)
    return native
