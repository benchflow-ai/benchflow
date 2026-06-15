"""Task ($T$) - the problem specification an agent solves."""

from __future__ import annotations

from pathlib import Path

from benchflow.task.config import TaskConfig
from benchflow.task.document import TaskDocument
from benchflow.task.paths import TaskPaths


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
        self.document: TaskDocument | None = None
        if self.paths.task_document_path.exists():
            self.document = TaskDocument.from_path(self.paths.task_document_path)
            self.instruction = self.document.instruction
            self.config = self.document.config
            self.scenes = self.document.scenes
        else:
            if not self.paths.instruction_path.exists():
                # Neither task.md (native) nor a legacy instruction.md is present.
                # Name the formats the author can actually create rather than
                # leaking the internal fallback filename, which otherwise
                # misdirects users toward the deprecated split format.
                raise FileNotFoundError(
                    f"no task document in {self._task_dir}: expected task.md "
                    "(native), or legacy task.toml + instruction.md"
                )
            self.instruction = self.paths.instruction_path.read_text()
            self.config = TaskConfig.model_validate_toml(
                self.paths.config_path.read_text()
            )
            self.scenes = []
        if self.config.task is not None:
            self.name = self.config.task.name
        else:
            self.name = self.paths.task_dir.name

    @property
    def task_dir(self) -> Path:
        return self._task_dir

    def __repr__(self) -> str:
        return f"Task({self.name!r}, dir={self._task_dir})"
