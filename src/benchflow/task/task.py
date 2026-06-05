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
    VerifierDocumentParseError,
    resolve_verifier_spec_path,
    verifier_document_issues,
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

        self.verifier_document: VerifierDocument | None = None
        benchflow = self.document.benchflow if self.document is not None else {}
        if isinstance(benchflow, dict):
            benchflow_verifier = benchflow.get("verifier")
            if isinstance(benchflow_verifier, dict):
                spec_issues = verifier_document_issues(
                    self._task_dir,
                    benchflow_verifier=benchflow_verifier,
                )
                if spec_issues:
                    raise ValueError("; ".join(spec_issues))
                spec = benchflow_verifier.get("spec")
                if isinstance(spec, str) and spec.strip():
                    spec_path = resolve_verifier_spec_path(self._task_dir, spec.strip())
                    try:
                        self.verifier_document = VerifierDocument.from_path(spec_path)
                    except VerifierDocumentParseError as exc:
                        raise ValueError(f"{spec} parse error: {exc}") from exc

    @property
    def task_dir(self) -> Path:
        return self._task_dir

    @property
    def runtime_view(self) -> TaskRuntimeView:
        """Selected executable interpretation for rollout and validation."""
        return TaskRuntimeView.from_task(self)

    def __repr__(self) -> str:
        return f"Task({self.name!r}, dir={self._task_dir})"
