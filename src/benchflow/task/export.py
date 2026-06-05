"""Native task.md export to Harbor/Pier split layout with loss reporting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from benchflow.task.aliases import alias_dir_collision_issues, normalized_tree_map
from benchflow.task.document import TaskDocument, TaskDocumentParseError
from benchflow.task.paths import TaskPaths

ExportTarget = Literal["harbor", "pier"]
ExportMode = Literal["full", "degraded"]


@dataclass(frozen=True)
class ExportLoss:
    """A native concept that the selected export target cannot express."""

    concept: str
    reason: str


@dataclass(frozen=True)
class ExportResult:
    """In-memory Harbor/Pier split export plus an honest loss report."""

    target: ExportTarget
    mode: ExportMode
    files: dict[str, str]
    losses: tuple[ExportLoss, ...]
    selected_definition: str
    selected_oracle_dir: str
    selected_verifier_dir: str
    exported_oracle_dir: str
    exported_verifier_dir: str
    ignored_aliases: tuple[str, ...]
    input_hashes: dict[str, str]
    output_hashes: dict[str, str]


def export_task_package(
    task_dir: str | Path,
    *,
    target: ExportTarget = "harbor",
) -> ExportResult:
    """Export a native ``task.md`` package to Harbor/Pier split layout.

    The exporter emits ``task.toml``, ``instruction.md``, ``solution/``, and
    ``tests/`` for Harbor-compatible fields and records every native-only
    concept that cannot be represented in the target format.
    """
    paths = TaskPaths(task_dir)
    if not paths.task_document_path.is_file():
        raise FileNotFoundError(
            f"Native export requires {TaskPaths.DOCUMENT_FILENAME}: "
            f"{paths.task_document_path}"
        )

    document = TaskDocument.from_path(paths.task_document_path)
    losses = _collect_semantic_losses(document, paths)
    ignored_aliases = _ignored_alias_notes(paths)
    for issue in alias_dir_collision_issues(paths):
        losses.append(
            ExportLoss(
                concept="alias.collision",
                reason=issue,
            )
        )

    files: dict[str, str] = {}
    files["task.toml"] = document.config.model_dump_toml()
    files["instruction.md"] = _format_instruction(document.instruction)

    _copy_tree(
        files,
        source_dir=paths.environment_dir,
        target_prefix="environment",
    )
    _copy_tree(
        files,
        source_dir=paths.oracle_dir,
        target_prefix=TaskPaths.LEGACY_SOLUTION_DIRNAME,
    )
    _copy_tree(
        files,
        source_dir=paths.verifier_source_dir,
        target_prefix=TaskPaths.LEGACY_TESTS_DIRNAME,
    )

    input_hashes = _hash_task_inputs(paths)
    output_hashes = _hash_export_files(files)
    mode: ExportMode = "degraded" if losses else "full"

    return ExportResult(
        target=target,
        mode=mode,
        files=files,
        losses=tuple(losses),
        selected_definition=TaskPaths.DOCUMENT_FILENAME,
        selected_oracle_dir=f"{TaskPaths.NATIVE_ORACLE_DIRNAME}/",
        selected_verifier_dir=f"{TaskPaths.NATIVE_VERIFIER_DIRNAME}/",
        exported_oracle_dir=f"{TaskPaths.LEGACY_SOLUTION_DIRNAME}/",
        exported_verifier_dir=f"{TaskPaths.LEGACY_TESTS_DIRNAME}/",
        ignored_aliases=ignored_aliases,
        input_hashes=input_hashes,
        output_hashes=output_hashes,
    )


def _collect_semantic_losses(
    document: TaskDocument,
    paths: TaskPaths,
) -> list[ExportLoss]:
    losses: list[ExportLoss] = []

    if document.frontmatter.get("agents"):
        losses.append(
            ExportLoss(
                concept="agents",
                reason="Harbor/Pier split layout has no native agents.roles surface",
            )
        )
    if document.scenes:
        losses.append(
            ExportLoss(
                concept="scenes",
                reason="Harbor/Pier split layout cannot express interaction scenes",
            )
        )
    if document.user:
        losses.append(
            ExportLoss(
                concept="user",
                reason="Harbor/Pier split layout cannot express simulated-user loops",
            )
        )

    for key in sorted(document.benchflow):
        losses.append(
            ExportLoss(
                concept=f"benchflow.{key}",
                reason="BenchFlow document namespace is not exported to Harbor/Pier",
            )
        )

    for role_name in sorted(document.role_prompts):
        losses.append(
            ExportLoss(
                concept=f"prompt.role:{role_name}",
                reason="Role prompts are not represented in instruction.md",
            )
        )
    for scene_name in sorted(document.scene_prompts):
        losses.append(
            ExportLoss(
                concept=f"prompt.scene:{scene_name}",
                reason="Scene prompts are not represented in instruction.md",
            )
        )
    if document.user_persona:
        losses.append(
            ExportLoss(
                concept="prompt.user-persona",
                reason="Simulated-user persona is not represented in instruction.md",
            )
        )

    verifier_md = paths.verifier_source_dir / "verifier.md"
    if verifier_md.is_file():
        losses.append(
            ExportLoss(
                concept="verifier.verifier_md",
                reason="Harbor/Pier tests/ has no verifier document surface",
            )
        )
    rubrics_dir = paths.verifier_source_dir / "rubrics"
    if rubrics_dir.is_dir() and any(rubrics_dir.iterdir()):
        losses.append(
            ExportLoss(
                concept="verifier.rubrics",
                reason="Harbor/Pier tests/ has no native rubric package surface",
            )
        )

    return losses


def _ignored_alias_notes(paths: TaskPaths) -> tuple[str, ...]:
    notes: list[str] = []
    if paths.uses_native_oracle_dir and paths.legacy_solution_dir.is_dir():
        notes.append(
            f"{TaskPaths.LEGACY_SOLUTION_DIRNAME}/ ignored; "
            f"{TaskPaths.NATIVE_ORACLE_DIRNAME}/ selected"
        )
    if paths.uses_native_verifier_dir and paths.legacy_tests_dir.is_dir():
        notes.append(
            f"{TaskPaths.LEGACY_TESTS_DIRNAME}/ ignored; "
            f"{TaskPaths.NATIVE_VERIFIER_DIRNAME}/ selected"
        )
    return tuple(notes)


def _format_instruction(instruction: str) -> str:
    text = instruction.strip()
    if not text:
        return ""
    if not text.endswith("\n"):
        text += "\n"
    return text


def _copy_tree(
    files: dict[str, str],
    *,
    source_dir: Path,
    target_prefix: str,
) -> None:
    if not source_dir.is_dir():
        return
    for rel_path, content in normalized_tree_map(source_dir).items():
        files[f"{target_prefix}/{rel_path}"] = _decode_bytes(content)


def _decode_bytes(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        raise TaskDocumentParseError(
            "Native export only supports UTF-8 text files in copied subtrees"
        ) from None


def _hash_task_inputs(paths: TaskPaths) -> dict[str, str]:
    hashes: dict[str, str] = {}
    _record_file_hash(hashes, paths.task_document_path)
    for subtree in (
        paths.environment_dir,
        paths.oracle_dir,
        paths.verifier_source_dir,
    ):
        if not subtree.is_dir():
            continue
        for rel_path, content in normalized_tree_map(subtree).items():
            hashes[f"{subtree.name}/{rel_path}"] = _sha256_bytes(content)
    return hashes


def _hash_export_files(files: dict[str, str]) -> dict[str, str]:
    return {
        rel_path: _sha256_text(content) for rel_path, content in sorted(files.items())
    }


def _record_file_hash(hashes: dict[str, str], path: Path) -> None:
    if path.is_file():
        hashes[path.name] = _sha256_bytes(path.read_bytes())


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_text(content: str) -> str:
    return _sha256_bytes(content.encode("utf-8"))


__all__ = [
    "ExportLoss",
    "ExportMode",
    "ExportResult",
    "ExportTarget",
    "export_task_package",
]
