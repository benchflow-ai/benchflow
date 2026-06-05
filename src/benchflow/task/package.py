"""Task package and runtime view for BenchFlow-native task directories.

``TaskPackage`` is the on-disk authoring layout. ``TaskRuntimeView`` is the
selected executable interpretation rollout, verifier, and validation code consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from benchflow._types import Scene
from benchflow.skill_policy import SKILL_MODE_SELF_GEN
from benchflow.task.aliases import alias_dir_collision_issues, normalized_tree_map
from benchflow.task.config import TaskConfig
from benchflow.task.document import TaskDocument
from benchflow.task.paths import TaskPaths
from benchflow.task.prompt_composition import (
    compose_task_prompt,
    prompt_composition_settings,
)
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
        settings = prompt_composition_settings(self.benchflow)
        return compose_task_prompt(
            self.instruction_text,
            None,
            None,
            None,
            composition=settings.composition,
            order=settings.order,
        )

    def compose_turn_prompt(
        self,
        scene_name: str,
        role_name: str,
        turn_prompt: str | None = None,
        *,
        explicit_turn: bool = False,
    ) -> str:
        """Compose one scene turn prompt using ``benchflow.prompt`` settings."""
        settings = prompt_composition_settings(self.benchflow)
        role_prompt = (
            self.document.role_prompts.get(role_name) if self.document else None
        )
        scene_prompt = (
            self.document.scene_prompts.get(scene_name) if self.document else None
        )
        return compose_task_prompt(
            self.instruction_text,
            role_prompt,
            scene_prompt,
            turn_prompt,
            composition=settings.composition,
            order=settings.order,
            explicit_turn=explicit_turn,
        )

    def to_rollout_scenes(
        self,
        *,
        prompts: list[str | None] | None = None,
        skill_mode: str = "no-skill",
    ) -> list[Scene]:
        """Return document scenes for rollout when no explicit override applies."""
        if prompts is not None or skill_mode == SKILL_MODE_SELF_GEN:
            return []
        return list(self.scenes)

    def document_runtime_summary(self) -> dict[str, Any]:
        """Return a compact runtime summary for logging and debug."""
        settings = prompt_composition_settings(self.benchflow)
        summary: dict[str, Any] = {
            "task_dir": str(self.task_dir),
            "entrypoint": self.entrypoint,
            "instruction_chars": len(self.instruction_text),
            "scene_names": list(self.scene_names),
            "verifier_dir": str(self.verifier_dir),
            "verifier_dir_kind": self.verifier_dir_kind,
            "oracle_dir": str(self.oracle_dir),
            "oracle_dir_kind": self.oracle_dir_kind,
            "has_legacy_split_files": self.has_legacy_split_files,
            "alias_collisions": list(self.alias_collisions.issues),
            "prompt_composition": settings.composition,
            "prompt_order": list(settings.order),
        }
        if self.document is not None:
            summary["role_names"] = sorted(self.document.roles)
            summary["role_prompt_sections"] = sorted(self.document.role_prompts)
            summary["scene_prompt_sections"] = sorted(self.document.scene_prompts)
        if self.benchflow:
            summary["benchflow_keys"] = sorted(self.benchflow)
        if self.verifier_document is not None:
            summary["verifier_document"] = {
                "name": self.verifier_document.name,
                "default_strategy": self.verifier_document.default_strategy,
            }
        return summary

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
