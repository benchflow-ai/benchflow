"""Native task.md export to Harbor/Pier split layout with loss reporting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchflow.task.aliases import alias_dir_collision_issues, normalized_tree_map
from benchflow.task.config import TaskConfig
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


def export_report_json(result: ExportResult) -> dict[str, object]:
    """Serialize an export result to the compatibility export-report schema."""
    return {
        "target": result.target,
        "mode": result.mode,
        "losses": [
            {"concept": loss.concept, "reason": loss.reason}
            for loss in result.losses
        ],
        "input_hashes": dict(result.input_hashes),
        "output_hashes": dict(result.output_hashes),
        "selected_definition": result.selected_definition,
        "selected_oracle_dir": result.selected_oracle_dir,
        "selected_verifier_dir": result.selected_verifier_dir,
        "exported_oracle_dir": result.exported_oracle_dir,
        "exported_verifier_dir": result.exported_verifier_dir,
        "ignored_aliases": list(result.ignored_aliases),
    }


EXPORT_REPORT_REL_PATH = "compatibility/export-report.json"


def write_export_report(
    output_dir: str | Path,
    result: ExportResult,
) -> Path:
    """Write ``compatibility/export-report.json`` for an export result."""
    report_path = Path(output_dir) / EXPORT_REPORT_REL_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(export_report_json(result), indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def materialize_export_result(
    result: ExportResult,
    output_dir: str | Path,
    *,
    write_report: bool = True,
) -> Path:
    """Write an in-memory export result to disk as a Harbor/Pier split package."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for rel_path, content in result.files.items():
        dest = root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    if write_report:
        write_export_report(root, result)
    return root


@dataclass(frozen=True)
class NativeComparison:
    """Semantic comparison of a split import against co-located native ``task.md``."""

    has_native_document: bool
    config_equal: bool | None
    instruction_equal: bool | None
    oracle_hash_equal: bool | None
    verifier_hash_equal: bool | None
    environment_hash_equal: bool | None
    config_native_dump: dict[str, Any] | None
    config_import_dump: dict[str, Any] | None
    instruction_native_normalized: str | None
    instruction_import_normalized: str | None
    oracle_native_hashes: dict[str, str] | None
    oracle_import_hashes: dict[str, str] | None
    verifier_native_hashes: dict[str, str] | None
    verifier_import_hashes: dict[str, str] | None
    environment_native_hashes: dict[str, str] | None
    environment_import_hashes: dict[str, str] | None


@dataclass(frozen=True)
class ImportResult:
    """Parsed Harbor/Pier split package plus optional native comparison."""

    task_dir: Path
    config: TaskConfig
    instruction: str
    instruction_normalized: str
    oracle_hashes: dict[str, str]
    verifier_hashes: dict[str, str]
    environment_hashes: dict[str, str]
    native_comparison: NativeComparison | None


def import_split_task_package(task_dir: str | Path) -> ImportResult:
    """Read a Harbor/Pier split package and compare against native ``task.md`` when present."""
    root = Path(task_dir).resolve()
    paths = TaskPaths(root)
    if not paths.config_path.is_file():
        raise FileNotFoundError(
            f"Split import requires {TaskPaths.CONFIG_FILENAME}: {paths.config_path}"
        )
    if not paths.instruction_path.is_file():
        raise FileNotFoundError(
            f"Split import requires instruction.md: {paths.instruction_path}"
        )

    config = TaskConfig.model_validate_toml(paths.config_path.read_text(encoding="utf-8"))
    instruction = paths.instruction_path.read_text(encoding="utf-8")
    instruction_normalized = _format_instruction(instruction)

    oracle_hashes = _subtree_hash_map(
        paths.legacy_solution_dir,
        TaskPaths.LEGACY_SOLUTION_DIRNAME,
    )
    verifier_hashes = _subtree_hash_map(
        paths.legacy_tests_dir,
        TaskPaths.LEGACY_TESTS_DIRNAME,
    )
    environment_hashes = _subtree_hash_map(paths.environment_dir, "environment")

    native_comparison = _compare_split_against_native(paths)
    return ImportResult(
        task_dir=root,
        config=config,
        instruction=instruction,
        instruction_normalized=instruction_normalized,
        oracle_hashes=oracle_hashes,
        verifier_hashes=verifier_hashes,
        environment_hashes=environment_hashes,
        native_comparison=native_comparison,
    )


def validate_export_round_trip(
    native_dir: str | Path,
    exported_dir: str | Path,
) -> list[str]:
    """Validate Harbor-compatible semantic parity between native and exported packages."""
    native_root = Path(native_dir).resolve()
    exported_root = Path(exported_dir).resolve()
    native_paths = TaskPaths(native_root)
    if not native_paths.task_document_path.is_file():
        return [
            f"Native round-trip requires {TaskPaths.DOCUMENT_FILENAME}: "
            f"{native_paths.task_document_path}"
        ]

    document = TaskDocument.from_path(native_paths.task_document_path)
    try:
        imported = import_split_task_package(exported_root)
    except FileNotFoundError as exc:
        return [str(exc)]

    issues: list[str] = []
    native_config_dump = document.config.model_dump()
    import_config_dump = imported.config.model_dump()
    if native_config_dump != import_config_dump:
        issues.append("Config drift: canonical TaskConfig dumps differ")

    native_instruction = _format_instruction(document.instruction)
    if native_instruction != imported.instruction_normalized:
        issues.append("Prompt drift: normalized instruction text differs")

    native_oracle_hashes = _subtree_hash_map(
        native_paths.oracle_dir,
        TaskPaths.NATIVE_ORACLE_DIRNAME,
    )
    if native_oracle_hashes != imported.oracle_hashes:
        issues.extend(
            _hash_map_diff_issues(
                "Oracle",
                native_oracle_hashes,
                imported.oracle_hashes,
                native_label=TaskPaths.NATIVE_ORACLE_DIRNAME,
                import_label=TaskPaths.LEGACY_SOLUTION_DIRNAME,
            )
        )

    native_verifier_hashes = _subtree_hash_map(
        native_paths.verifier_source_dir,
        TaskPaths.NATIVE_VERIFIER_DIRNAME,
    )
    if native_verifier_hashes != imported.verifier_hashes:
        issues.extend(
            _hash_map_diff_issues(
                "Verifier",
                native_verifier_hashes,
                imported.verifier_hashes,
                native_label=TaskPaths.NATIVE_VERIFIER_DIRNAME,
                import_label=TaskPaths.LEGACY_TESTS_DIRNAME,
            )
        )

    native_environment_hashes = _subtree_hash_map(
        native_paths.environment_dir,
        "environment",
    )
    if native_environment_hashes != imported.environment_hashes:
        issues.extend(
            _hash_map_diff_issues(
                "Environment",
                native_environment_hashes,
                imported.environment_hashes,
                native_label="environment",
                import_label="environment",
            )
        )

    return issues


def _compare_split_against_native(paths: TaskPaths) -> NativeComparison | None:
    if not paths.task_document_path.is_file():
        return None

    document = TaskDocument.from_path(paths.task_document_path)
    native_config_dump = document.config.model_dump()
    import_config_dump = TaskConfig.model_validate_toml(
        paths.config_path.read_text(encoding="utf-8")
    ).model_dump()
    native_instruction = _format_instruction(document.instruction)
    import_instruction = _format_instruction(
        paths.instruction_path.read_text(encoding="utf-8")
    )

    native_oracle_hashes = _subtree_hash_map(
        paths.oracle_dir,
        TaskPaths.NATIVE_ORACLE_DIRNAME,
    )
    import_oracle_hashes = _subtree_hash_map(
        paths.legacy_solution_dir,
        TaskPaths.LEGACY_SOLUTION_DIRNAME,
    )
    native_verifier_hashes = _subtree_hash_map(
        paths.verifier_source_dir,
        TaskPaths.NATIVE_VERIFIER_DIRNAME,
    )
    import_verifier_hashes = _subtree_hash_map(
        paths.legacy_tests_dir,
        TaskPaths.LEGACY_TESTS_DIRNAME,
    )
    native_environment_hashes = _subtree_hash_map(paths.environment_dir, "environment")
    import_environment_hashes = _subtree_hash_map(paths.environment_dir, "environment")

    return NativeComparison(
        has_native_document=True,
        config_equal=native_config_dump == import_config_dump,
        instruction_equal=native_instruction == import_instruction,
        oracle_hash_equal=native_oracle_hashes == import_oracle_hashes,
        verifier_hash_equal=native_verifier_hashes == import_verifier_hashes,
        environment_hash_equal=native_environment_hashes == import_environment_hashes,
        config_native_dump=native_config_dump,
        config_import_dump=import_config_dump,
        instruction_native_normalized=native_instruction,
        instruction_import_normalized=import_instruction,
        oracle_native_hashes=native_oracle_hashes,
        oracle_import_hashes=import_oracle_hashes,
        verifier_native_hashes=native_verifier_hashes,
        verifier_import_hashes=import_verifier_hashes,
        environment_native_hashes=native_environment_hashes,
        environment_import_hashes=import_environment_hashes,
    )


def _subtree_hash_map(root: Path, prefix: str) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {
        f"{prefix}/{rel_path}": _sha256_bytes(content)
        for rel_path, content in normalized_tree_map(root).items()
    }


def _hash_map_diff_issues(
    label: str,
    native_hashes: dict[str, str],
    import_hashes: dict[str, str],
    *,
    native_label: str,
    import_label: str,
) -> list[str]:
    issues: list[str] = []
    native_rel = {path.removeprefix(f"{native_label}/"): digest for path, digest in native_hashes.items()}
    import_rel = {path.removeprefix(f"{import_label}/"): digest for path, digest in import_hashes.items()}

    for rel_path in sorted(set(native_rel) | set(import_rel)):
        native_digest = native_rel.get(rel_path)
        import_digest = import_rel.get(rel_path)
        if native_digest == import_digest:
            continue
        if native_digest is None:
            issues.append(
                f"{label} drift: {import_label}/{rel_path} missing from native {native_label}/"
            )
        elif import_digest is None:
            issues.append(
                f"{label} drift: {native_label}/{rel_path} missing from exported {import_label}/"
            )
        else:
            issues.append(
                f"{label} drift: {rel_path} hash differs between "
                f"{native_label}/ and {import_label}/"
            )
    return issues


__all__ = [
    "EXPORT_REPORT_REL_PATH",
    "ExportLoss",
    "ExportMode",
    "ExportResult",
    "ExportTarget",
    "ImportResult",
    "NativeComparison",
    "export_report_json",
    "export_task_package",
    "import_split_task_package",
    "materialize_export_result",
    "validate_export_round_trip",
    "write_export_report",
]
