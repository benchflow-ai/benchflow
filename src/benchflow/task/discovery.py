"""Task-directory discovery helpers shared by eval and artifact writers."""

from __future__ import annotations

from pathlib import Path


def is_task_dir(path: Path) -> bool:
    """Return whether *path* is a BenchFlow task package directory."""

    return (path / "task.md").is_file() or (path / "task.toml").is_file()


def contains_immediate_task(path: Path) -> bool:
    """Return whether *path* directly contains one or more task directories."""

    if not path.is_dir():
        return False
    return any(child.is_dir() and is_task_dir(child) for child in path.iterdir())


def resolve_task_collection_root(path: str | Path) -> Path:
    """Return the directory whose immediate children are task dirs.

    Besides the historical inputs (a single task dir or a parent containing task
    dirs), this accepts dataset snapshots that wrap tasks under ``tasks/``:

    ``snapshot_root/tasks/<task_id>/{instruction.md, task.toml, ...}``
    """

    root = Path(path)
    if is_task_dir(root) or contains_immediate_task(root):
        return root
    nested_tasks = root / "tasks"
    if contains_immediate_task(nested_tasks):
        return nested_tasks
    return root
