"""Shared task package output-format helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

TaskOutputFormat = Literal["legacy", "task-md"]
TASK_OUTPUT_FORMATS: tuple[TaskOutputFormat, ...] = ("legacy", "task-md")


def validate_task_output_format(value: str) -> TaskOutputFormat:
    """Return a supported task output format or raise a clear ValueError."""
    if value not in TASK_OUTPUT_FORMATS:
        joined = ", ".join(TASK_OUTPUT_FORMATS)
        raise ValueError(f"task_format must be one of: {joined}")
    return cast(TaskOutputFormat, value)


def task_entrypoint_name(task_format: TaskOutputFormat) -> str:
    return "task.md" if task_format == "task-md" else "task.toml"


def conflicting_task_entrypoint_name(task_format: TaskOutputFormat) -> str:
    return "task.toml" if task_format == "task-md" else "task.md"


def verifier_dir_name(task_format: TaskOutputFormat) -> str:
    return "verifier" if task_format == "task-md" else "tests"


def oracle_dir_name(task_format: TaskOutputFormat) -> str:
    return "oracle" if task_format == "task-md" else "solution"


def ensure_existing_task_output_format(
    task_dir: Path,
    task_format: TaskOutputFormat,
) -> None:
    """Reject reusing a task directory created with the other output format."""
    expected = task_entrypoint_name(task_format)
    unexpected = conflicting_task_entrypoint_name(task_format)
    if (task_dir / unexpected).exists():
        raise ValueError(
            f"{task_dir} already contains {unexpected}; pass --overwrite or use "
            f"a separate output directory for {task_format} output."
        )
    if not (task_dir / expected).exists():
        raise ValueError(
            f"{task_dir} already exists but is missing {expected}; pass "
            "--overwrite or use a fresh output directory."
        )
