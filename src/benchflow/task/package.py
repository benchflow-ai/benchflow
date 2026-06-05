"""Task package and runtime view for BenchFlow-native task directories.

``TaskPackage`` is the on-disk authoring layout. ``TaskRuntimeView`` is the
selected executable interpretation rollout, verifier, and validation code consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from benchflow._types import Scene
from benchflow.task.aliases import alias_dir_collision_issues, normalized_tree_map
from benchflow.task.config import TaskConfig
from benchflow.task.document import TaskDocument
from benchflow.task.paths import TaskPaths
from benchflow.task.verifier_document import (
    VerifierDocument,
    load_verifier_document,
)

if TYPE_CHECKING:
    from benchflow.task.task import Task

TaskEntrypoint = Literal["task-md", "legacy-split"]
VerifierDirKind = Literal["native", "legacy"]
OracleDirKind = Literal["native", "legacy"]


@dataclass(frozen=True)
class AliasCollisionStatus:
    """Native/legacy directory alias equivalence diagnostics."""

    issues: tuple[str, ...]

    @property
    def has_collisions(self) -> bool:
        return bool(self.issues)


@dataclass(frozen=True)
class TaskPackage:
    """Authoring package rooted at a task directory."""

    task_dir: Path
    paths: TaskPaths

    @classmethod
    def load(cls, task_dir: Path | str) -> TaskPackage:
        root = Path(task_dir).resolve()
        return cls(task_dir=root, paths=TaskPaths(root))


@dataclass(frozen=True)
class TaskRuntimeView:
    """Selected executable interpretation of a task package."""

    package: TaskPackage
    entrypoint: TaskEntrypoint
    instruction_text: str
    config: TaskConfig
    document: TaskDocument | None
    scenes: tuple[Scene, ...]
    verifier_dir_kind: VerifierDirKind
    oracle_dir_kind: OracleDirKind
    alias_collisions: AliasCollisionStatus
    has_legacy_split_files: bool
    benchflow: dict[str, Any]
    verifier_document: VerifierDocument | None = None

    @property
    def task_dir(self) -> Path:
        return self.package.task_dir

    @property
    def paths(self) -> TaskPaths:
        return self.package.paths

    @property
    def uses_native_verifier_dir(self) -> bool:
        return self.verifier_dir_kind == "native"

    @property
    def uses_native_oracle_dir(self) -> bool:
        return self.oracle_dir_kind == "native"

    @property
    def verifier_dir(self) -> Path:
        return self.paths.tests_dir

    @property
    def oracle_dir(self) -> Path:
        return self.paths.solution_dir

    @property
    def scene_names(self) -> tuple[str, ...]:
        return tuple(scene.name for scene in self.scenes)

    @classmethod
    def from_task_dir(
        cls,
        task_dir: Path | str,
        *,
        fail_on_alias_collision: bool = True,
    ) -> TaskRuntimeView:
        package = TaskPackage.load(task_dir)
        alias_issues = alias_dir_collision_issues(package.paths)
        if fail_on_alias_collision and alias_issues:
            raise ValueError("; ".join(alias_issues))
        return cls._build(package, alias_issues)

    @classmethod
    def from_task(cls, task: Task) -> TaskRuntimeView:
        package = TaskPackage(task_dir=task.task_dir, paths=task.paths)
        return cls._build(
            package,
            alias_dir_collision_issues(package.paths),
            document=task.document,
            config=task.config,
            instruction_text=task.instruction.strip(),
            scenes=tuple(task.scenes),
            verifier_document=task.verifier_document,
        )

    @classmethod
    def _build(
        cls,
        package: TaskPackage,
        alias_issues: list[str],
        *,
        document: TaskDocument | None = None,
        config: TaskConfig | None = None,
        instruction_text: str | None = None,
        scenes: tuple[Scene, ...] | None = None,
        verifier_document: VerifierDocument | None = None,
    ) -> TaskRuntimeView:
        paths = package.paths
        has_task_md = paths.task_document_path.exists()
        has_legacy_split = (
            paths.config_path.exists() and paths.instruction_path.exists()
        )
        has_instruction_only = (
            paths.instruction_path.exists()
            and not paths.config_path.exists()
            and not has_task_md
        )

        if has_task_md:
            entrypoint: TaskEntrypoint = "task-md"
            if document is None:
                document = TaskDocument.from_path(paths.task_document_path)
            if config is None:
                config = document.config
            if instruction_text is None:
                instruction_text = document.instruction.strip()
            if scenes is None:
                scenes = tuple(document.scenes)
            benchflow = dict(document.benchflow)
        elif has_legacy_split or has_instruction_only:
            entrypoint = "legacy-split"
            if config is None:
                if has_legacy_split:
                    config = TaskConfig.model_validate_toml(
                        paths.config_path.read_text()
                    )
                else:
                    config = TaskConfig()
            if instruction_text is None:
                instruction_text = paths.instruction_path.read_text().strip()
            if scenes is None:
                scenes = ()
            benchflow = {}
        else:
            raise FileNotFoundError(
                f"Task missing task.md or legacy task.toml + instruction.md: "
                f"{package.task_dir}"
            )

        verifier_dir_kind: VerifierDirKind = (
            "native" if paths.uses_native_verifier_dir else "legacy"
        )
        oracle_dir_kind: OracleDirKind = (
            "native" if paths.uses_native_oracle_dir else "legacy"
        )

        if verifier_document is None and entrypoint == "task-md":
            verifier_document = load_verifier_document(package.task_dir, benchflow)

        return cls(
            package=package,
            entrypoint=entrypoint,
            instruction_text=instruction_text,
            config=config,
            document=document,
            scenes=scenes,
            verifier_dir_kind=verifier_dir_kind,
            oracle_dir_kind=oracle_dir_kind,
            alias_collisions=AliasCollisionStatus(issues=tuple(alias_issues)),
            has_legacy_split_files=has_legacy_split,
            benchflow=benchflow,
            verifier_document=verifier_document,
        )

    def materialize_instruction_md(self) -> str:
        """Return prompt text for the ``/instruction.md`` compatibility upload."""
        return self.instruction_text

    def selected_verifier_tree_map(self) -> dict[str, bytes]:
        """Normalized file map for the selected verifier directory."""
        return normalized_tree_map(self.verifier_dir)

    def selected_oracle_tree_map(self) -> dict[str, bytes]:
        """Normalized file map for the selected oracle directory."""
        return normalized_tree_map(self.oracle_dir)


__all__ = [
    "AliasCollisionStatus",
    "OracleDirKind",
    "TaskEntrypoint",
    "TaskPackage",
    "TaskRuntimeView",
    "VerifierDirKind",
]
