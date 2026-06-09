"""Executable task view selected from native or compatibility task files."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchflow._types import Scene
from benchflow.task.config import TaskConfig
from benchflow.task.document import TaskDocument
from benchflow.task.paths import TaskPaths
from benchflow.task.verifier_document import (
    VerifierDocument,
    load_verifier_document,
)

TaskEntrypoint = Literal["task.md", "legacy"]


@dataclass(frozen=True)
class TaskRuntimeCompatibility:
    """Compatibility metadata for the selected task layout."""

    has_task_md: bool
    has_legacy_definition: bool
    selected_entrypoint: TaskEntrypoint
    uses_native_oracle_dir: bool
    uses_native_verifier_dir: bool
    has_legacy_solution_alias: bool
    has_legacy_tests_alias: bool


@dataclass(frozen=True)
class TaskRuntimeView:
    """Single runtime-facing interpretation of a task package.

    The authoring layer can parse both native ``task.md`` packages and legacy
    Harbor/Pier split packages. Runtime callers should consult this view instead
    of re-deciding path precedence at every launch or verifier boundary.
    """

    task_dir: Path
    entrypoint: TaskEntrypoint
    config: TaskConfig
    prompt: str
    verifier_dir: Path
    verifier_document: VerifierDocument | None
    oracle_dir: Path
    environment_dir: Path
    scenes: tuple[Scene, ...]
    source_hashes: Mapping[str, str]
    compatibility: TaskRuntimeCompatibility

    @classmethod
    def from_task_dir(cls, task_dir: str | Path) -> TaskRuntimeView:
        """Build a view from files on disk."""

        paths = TaskPaths(task_dir)
        if paths.task_document_path.exists():
            document = TaskDocument.from_path(paths.task_document_path)
            return cls.from_parsed(
                paths.task_dir,
                document=document,
                config=document.config,
                prompt=document.instruction,
                scenes=document.scenes,
            )

        config = TaskConfig.model_validate_toml(paths.config_path.read_text())
        return cls.from_parsed(
            paths.task_dir,
            document=None,
            config=config,
            prompt=paths.instruction_path.read_text(),
            scenes=[],
        )

    @classmethod
    def from_task(cls, task: Any, *, task_dir: str | Path | None = None) -> TaskRuntimeView:
        """Build a view from a ``Task``-like object without reparsing."""

        root = Path(task_dir) if task_dir is not None else Path(task.task_dir)
        document = getattr(task, "document", None)
        scenes = tuple(getattr(task, "scenes", ()))
        return cls.from_parsed(
            root,
            document=document if isinstance(document, TaskDocument) else None,
            config=task.config,
            prompt=task.instruction,
            scenes=scenes,
        )

    @classmethod
    def from_parsed(
        cls,
        task_dir: str | Path,
        *,
        document: TaskDocument | None,
        config: TaskConfig,
        prompt: str,
        scenes: list[Scene] | tuple[Scene, ...],
    ) -> TaskRuntimeView:
        """Build a view from already parsed task components."""

        paths = TaskPaths(task_dir)
        entrypoint: TaskEntrypoint = (
            "task.md" if document is not None or paths.task_document_path.exists() else "legacy"
        )
        return cls(
            task_dir=paths.task_dir,
            entrypoint=entrypoint,
            config=config,
            prompt=prompt,
            verifier_dir=paths.tests_dir,
            verifier_document=load_verifier_document(paths.tests_dir),
            oracle_dir=paths.solution_dir,
            environment_dir=paths.environment_dir,
            scenes=tuple(scenes),
            source_hashes=_source_hashes(paths),
            compatibility=TaskRuntimeCompatibility(
                has_task_md=paths.task_document_path.exists(),
                has_legacy_definition=(
                    paths.config_path.exists() and paths.instruction_path.exists()
                ),
                selected_entrypoint=entrypoint,
                uses_native_oracle_dir=paths.uses_native_oracle_dir,
                uses_native_verifier_dir=paths.uses_native_verifier_dir,
                has_legacy_solution_alias=paths.legacy_solution_dir.exists(),
                has_legacy_tests_alias=paths.legacy_tests_dir.exists(),
            ),
        )


def _source_hashes(paths: TaskPaths) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for root in (
        paths.task_document_path,
        paths.config_path,
        paths.instruction_path,
        paths.environment_dir,
        paths.oracle_dir,
        paths.legacy_solution_dir,
        paths.verifier_source_dir,
        paths.legacy_tests_dir,
    ):
        if root.exists():
            _add_hashes(hashes, paths.task_dir, root)
    return hashes


def _add_hashes(hashes: dict[str, str], task_dir: Path, path: Path) -> None:
    if path.is_file():
        hashes[path.relative_to(task_dir).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        return
    if not path.is_dir():
        return
    for child in sorted(path.rglob("*")):
        if child.is_file():
            hashes[child.relative_to(task_dir).as_posix()] = hashlib.sha256(
                child.read_bytes()
            ).hexdigest()
