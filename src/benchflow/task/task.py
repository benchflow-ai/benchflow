"""Task ($T$) - the problem specification an agent solves."""

from __future__ import annotations

from pathlib import Path

from benchflow.task.aliases import alias_dir_collision_issues
from benchflow.task.config import TaskConfig
from benchflow.task.document import TaskDocument
from benchflow.task.package import TaskRuntimeView
from benchflow.task.paths import TaskPaths
from benchflow.task.verifier_document import (
    VerifierDocument,
    load_verifier_document,
)


class Task:
    """Represents a task with the standard directory structure.

    ::

        task_dir/
        ├── task.md              # native unified format, or:
        ├── instruction.md        # legacy split format
        ├── task.toml             # legacy split format
        ├── environment/
        │   ├── Dockerfile
        │   └── ...
        ├── solution/
        │   ├── solve.sh
        │   └── ...
        └── tests/
            ├── test.sh
            └── ...
    """

    def __init__(self, task_dir: Path | str) -> None:
        self._task_dir = Path(task_dir).resolve()
        self.paths = TaskPaths(self._task_dir)
        alias_issues = alias_dir_collision_issues(self.paths)
        if alias_issues:
            raise ValueError("; ".join(alias_issues))
        self.document: TaskDocument | None = None
        if self.paths.task_document_path.exists():
            self.document = TaskDocument.from_path(self.paths.task_document_path)
            self.instruction = self.document.instruction
            self.config = self.document.config
            self.scenes = self.document.scenes
        else:
            self.instruction = self.paths.instruction_path.read_text()
            self.config = TaskConfig.model_validate_toml(
                self.paths.config_path.read_text()
            )
            self.scenes = []
        if self.config.task is not None:
            self.name = self.config.task.name
        else:
            self.name = self.paths.task_dir.name

        benchflow = self.document.benchflow if self.document is not None else {}
        self.verifier_document: VerifierDocument | None = load_verifier_document(
            self._task_dir,
            benchflow,
        )

    @property
    def task_dir(self) -> Path:
        return self._task_dir

    @property
    def runtime_view(self) -> TaskRuntimeView:
        """Selected executable interpretation for rollout and validation."""
        return TaskRuntimeView.from_task(self)

    def __repr__(self) -> str:
        return f"Task({self.name!r}, dir={self._task_dir})"
