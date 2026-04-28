"""Task loader.

The declarative task shapes (``TaskConfig`` and friends, ``EnvironmentConfig``)
live in ``benchflow.contracts.task_config``. This module owns the IO side:
reading ``task.toml`` + ``instruction.md`` off disk and exposing a ``TaskPaths``
view onto the canonical directory layout.
"""

from __future__ import annotations

import re
from pathlib import Path

from benchflow.contracts.task_config import TaskConfig

_CANARY_LINE_RE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)


def strip_canary(text: str) -> str:
    """Strip canary marker lines from the start of instruction text."""
    lines = text.split("\n")
    idx = 0
    while idx < len(lines) and _CANARY_LINE_RE.match(lines[idx].strip()):
        idx += 1
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    return "\n".join(lines[idx:])


class TaskPaths:
    """File paths for a task with the canonical directory structure."""

    CONFIG_FILENAME = "task.toml"

    def __init__(self, task_dir: Path | str):
        self.task_dir = Path(task_dir).resolve()

    @property
    def instruction_path(self) -> Path:
        return self.task_dir / "instruction.md"

    @property
    def readme_path(self) -> Path:
        return self.task_dir / "README.md"

    @property
    def gitignore_path(self) -> Path:
        return self.task_dir / ".gitignore"

    @property
    def config_path(self) -> Path:
        return self.task_dir / self.CONFIG_FILENAME

    @property
    def environment_dir(self) -> Path:
        return self.task_dir / "environment"

    @property
    def solution_dir(self) -> Path:
        return self.task_dir / "solution"

    @property
    def solve_path(self) -> Path:
        return self.solution_dir / "solve.sh"

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / "tests"

    @property
    def test_path(self) -> Path:
        return self.tests_dir / "test.sh"

    def is_valid(self, disable_verification: bool = False) -> bool:
        return (
            self.config_path.exists()
            and self.environment_dir.exists()
            and self.instruction_path.exists()
            and (disable_verification or self.test_path.exists())
        )


class Task:
    """Composes a TaskPaths + TaskConfig + instruction text."""

    def __init__(self, task_dir: Path | str):
        self._task_dir = Path(task_dir).resolve()
        self.paths = TaskPaths(self._task_dir)
        self.instruction = strip_canary(self.paths.instruction_path.read_text())
        self.config = TaskConfig.model_validate_toml(self.paths.config_path.read_text())
        if self.config.task is not None:
            self.name = self.config.task.name
        else:
            self.name = self.paths.task_dir.name

    @property
    def task_dir(self) -> Path:
        return self._task_dir


__all__ = ["Task", "TaskPaths", "strip_canary"]
