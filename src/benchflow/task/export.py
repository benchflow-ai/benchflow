"""Compatibility exports from native task packages to split layouts."""

from __future__ import annotations

import json
import shutil
import tempfile
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import tomli_w

from benchflow.task.document import TaskDocument, render_task_md_from_legacy
from benchflow.task.imports import import_task_config_toml, merge_compat_extra
from benchflow.task.package import TaskPackage
from benchflow.task.paths import TaskPaths
from benchflow.task.runtime_view import TaskRuntimeView

CompatibilityTarget = Literal["harbor"]


@dataclass(frozen=True)
class CompatibilityExportLoss:
    """One native concept that a target split layout cannot express."""

    path: str
    reason: str
    severity: Literal["loss", "warning"] = "loss"


@dataclass(frozen=True)
class CompatibilityExportReport:
    """Machine-readable report for a compatibility export."""

    target: CompatibilityTarget
    status: Literal["lossless", "degraded"]
    source_task_dir: str
    selected_entrypoint: str
    selected_prompt: str
    selected_oracle_dir: str | None
    selected_verifier_dir: str | None
    emitted_files: list[str]
    input_hashes: dict[str, str]
    output_hashes: dict[str, str]
    restored_extension_paths: list[str] = field(default_factory=list)
    losses: list[CompatibilityExportLoss] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class HarborRoundTripMismatch:
    """One failed comparison in split -> task.md -> split conformance."""

    path: str
    reason: str


@dataclass(frozen=True)
class HarborRoundTripConformanceReport:
    """Machine-readable Harbor split round-trip conformance result."""

    target: CompatibilityTarget
    status: Literal["lossless", "drift"]
    source_task_dir: str
    config_equal: bool
    prompt_equal: bool
    environment_file_map_equal: bool
    solution_file_map_equal: bool
    tests_file_map_equal: bool
    restored_extension_paths: list[str]
    mismatches: list[HarborRoundTripMismatch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_harbor_roundtrip_conformance_report(
    task_dir: str | Path,
    *,
    target: CompatibilityTarget = "harbor",
) -> HarborRoundTripConformanceReport:
    """Compare a split task against its migrated task.md split export.

    This is the first pure conformance check for Harbor parity. It does
    not execute the task. Instead it proves that the supported compatibility
    surface survives the native authoring hop: canonical ``task.toml`` config,
    normalized ``instruction.md`` prompt, and file maps for environment,
    solution/oracle, and tests/verifier.
    """

    source = Path(task_dir)
    mismatches: list[HarborRoundTripMismatch] = []

    with tempfile.TemporaryDirectory(prefix="benchflow-harbor-roundtrip-") as raw_tmp:
        tmp_root = Path(raw_tmp)
        migrated = tmp_root / "migrated"
        exported = tmp_root / "exported"
        shutil.copytree(source, migrated)
        (migrated / TaskPaths.DOCUMENT_FILENAME).write_text(
            render_task_md_from_legacy(migrated)
        )
        (migrated / TaskPaths.CONFIG_FILENAME).unlink()
        (migrated / "instruction.md").unlink()

        export_report = export_task_to_split_layout(
            migrated,
            exported,
            target=target,
        )

        config_equal = _imported_config_equal(
            source / TaskPaths.CONFIG_FILENAME,
            exported / TaskPaths.CONFIG_FILENAME,
            source=target,
        )
        if not config_equal:
            mismatches.append(
                HarborRoundTripMismatch(
                    path="task.toml",
                    reason="canonical TaskConfig differs after migrate/export",
                )
            )

        prompt_equal = _normalize_prompt((source / "instruction.md").read_text()) == (
            _normalize_prompt((exported / "instruction.md").read_text())
        )
        if not prompt_equal:
            mismatches.append(
                HarborRoundTripMismatch(
                    path="instruction.md",
                    reason="normalized prompt text differs after migrate/export",
                )
            )

        environment_equal = _file_map(source / "environment") == _file_map(
            exported / "environment"
        )
        if not environment_equal:
            mismatches.append(
                HarborRoundTripMismatch(
                    path="environment/",
                    reason="environment file hashes differ after migrate/export",
                )
            )

        solution_equal = _file_map(source / "solution") == _file_map(
            exported / "solution"
        )
        if not solution_equal:
            mismatches.append(
                HarborRoundTripMismatch(
                    path="solution/",
                    reason="solution file hashes differ after migrate/export",
                )
            )

        tests_equal = _file_map(source / "tests") == _file_map(exported / "tests")
        if not tests_equal:
            mismatches.append(
                HarborRoundTripMismatch(
                    path="tests/",
                    reason="tests file hashes differ after migrate/export",
                )
            )

        restored_extension_paths = list(export_report.restored_extension_paths)

    return HarborRoundTripConformanceReport(
        target=target,
        status="drift" if mismatches else "lossless",
        source_task_dir=str(source),
        config_equal=config_equal,
        prompt_equal=prompt_equal,
        environment_file_map_equal=environment_equal,
        solution_file_map_equal=solution_equal,
        tests_file_map_equal=tests_equal,
        restored_extension_paths=restored_extension_paths,
        mismatches=mismatches,
    )


def build_compatibility_export_report(
    task_dir: str | Path,
    *,
    target: CompatibilityTarget = "harbor",
    output_dir: str | Path | None = None,
) -> CompatibilityExportReport:
    """Build a report without writing export files."""

    package = TaskPackage.from_task_dir(task_dir)
    view = package.view
    losses = _detect_losses(package)
    output_hashes = _hash_export_output(Path(output_dir)) if output_dir else {}
    emitted_files = sorted(output_hashes)
    return CompatibilityExportReport(
        target=target,
        status="degraded" if losses else "lossless",
        source_task_dir=str(view.task_dir),
        selected_entrypoint=view.entrypoint,
        selected_prompt="instruction.md",
        selected_oracle_dir=_rel_or_none(view.task_dir, view.oracle_dir),
        selected_verifier_dir=_rel_or_none(view.task_dir, view.verifier_dir),
        emitted_files=emitted_files,
        input_hashes=dict(view.source_hashes),
        output_hashes=output_hashes,
        restored_extension_paths=_compat_extra_paths(package.document),
        losses=losses,
    )


def export_task_to_split_layout(
    task_dir: str | Path,
    output_dir: str | Path,
    *,
    target: CompatibilityTarget = "harbor",
    overwrite: bool = False,
) -> CompatibilityExportReport:
    """Materialize a Harbor-compatible split task layout.

    The export preserves supported split-layout surfaces:
    ``task.toml``, ``instruction.md``, ``environment/``, ``solution/``, and
    ``tests/``. Native-only semantics are not hidden; they are reported in
    ``compatibility/export-report.json``.
    """

    package = TaskPackage.from_task_dir(task_dir)
    view = package.view
    dest = Path(output_dir)

    # Reject a destination that overlaps the source task directory in EITHER
    # direction. ``TaskPackage`` only loads config/prompt/document into memory;
    # the environment/oracle/verifier trees are copied from disk *after* the
    # ``rmtree`` below, so exporting onto (or inside, or above) the source would
    # delete those files before they are copied — a silent destructive export.
    src = Path(task_dir).resolve()
    dst = dest.resolve()
    if src == dst or dst.is_relative_to(src) or src.is_relative_to(dst):
        raise ValueError(
            f"Export destination {dest} overlaps the source task directory "
            f"{task_dir}; choose a separate output directory"
        )

    if dest.exists() and not overwrite:
        raise FileExistsError(f"Export destination already exists: {dest}")

    # Build the export in a sibling temp dir and swap it into place only once it
    # is fully materialized, so any abort/collision can never leave a
    # half-deleted destination (and, with the guard above, never the source).
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="benchflow-split-export-", dir=dest.parent
    ) as staging_root:
        staged = Path(staging_root) / "export"
        staged.mkdir()

        (staged / "task.toml").write_text(_export_task_toml(view))
        (staged / "instruction.md").write_text(view.prompt.strip() + "\n")

        _copy_tree_if_exists(view.environment_dir, staged / "environment")
        _copy_tree_if_exists(view.oracle_dir, staged / "solution")
        _copy_tree_if_exists(view.verifier_dir, staged / "tests")

        report = build_compatibility_export_report(
            view.task_dir,
            target=target,
            output_dir=staged,
        )
        report_dir = staged / "compatibility"
        report_dir.mkdir(exist_ok=True)
        (report_dir / "export-report.json").write_text(report.to_json())

        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(staged), str(dest))
    return report


def _detect_losses(package: TaskPackage) -> list[CompatibilityExportLoss]:
    losses: list[CompatibilityExportLoss] = []
    view = package.view
    document = package.document
    if document is not None:
        if document.roles:
            losses.append(
                CompatibilityExportLoss(
                    path="agents.roles",
                    reason="split layout has no native verifier-visible role table",
                )
            )
        if document.scenes:
            losses.append(
                CompatibilityExportLoss(
                    path="scenes",
                    reason="split layout instruction.md has no scene/handoff model",
                )
            )
        if document.role_prompts:
            losses.append(
                CompatibilityExportLoss(
                    path="## role:*",
                    reason="role-scoped prompts are not represented in instruction.md",
                )
            )
        if document.scene_prompts:
            losses.append(
                CompatibilityExportLoss(
                    path="## scene:*",
                    reason="scene-scoped prompts are not represented in instruction.md",
                )
            )
        if document.user:
            losses.append(
                CompatibilityExportLoss(
                    path="user",
                    reason="split layout has no simulated-user contract",
                )
            )
        if document.user_persona:
            losses.append(
                CompatibilityExportLoss(
                    path="## user-persona",
                    reason="user persona is not represented in instruction.md",
                )
            )
        for key in sorted(document.benchflow):
            if key == "compat" and _compat_extra(document):
                continue
            losses.append(
                CompatibilityExportLoss(
                    path=f"benchflow.{key}",
                    reason="benchflow-native metadata is preserved only in the export report",
                    severity="warning",
                )
            )

    if view.verifier_document is not None:
        losses.append(
            CompatibilityExportLoss(
                path="verifier.verifier_md",
                reason="verifier/verifier.md is copied as a file but target runtimes do not execute its strategy model",
            )
        )

    losses.extend(_alias_collision_losses(view.task_dir))
    return losses


def _load_document(view: TaskRuntimeView) -> TaskDocument | None:
    path = view.task_dir / TaskPaths.DOCUMENT_FILENAME
    if not path.exists():
        return None
    return TaskDocument.from_path(path)


def _export_task_toml(view: TaskRuntimeView) -> str:
    data = tomllib.loads(view.config.model_dump_toml())
    document = _load_document(view)
    extra = _compat_extra(document)
    if extra:
        data = merge_compat_extra(data, extra)
    return tomli_w.dumps(data)


def _compat_extra(document: TaskDocument | None) -> dict[str, Any]:
    if document is None:
        return {}
    compat = document.benchflow.get("compat")
    if not isinstance(compat, dict):
        return {}
    extra = compat.get("extra")
    return extra if isinstance(extra, dict) else {}


def _compat_extra_paths(document: TaskDocument | None) -> list[str]:
    if document is None:
        return []
    compat = document.benchflow.get("compat")
    if isinstance(compat, dict):
        raw_paths = compat.get("extra_paths")
        if isinstance(raw_paths, list) and all(
            isinstance(path, str) for path in raw_paths
        ):
            return sorted(raw_paths)
    return sorted(_format_path(path) for path in _leaf_paths(_compat_extra(document)))


def _leaf_paths(
    data: dict[str, Any] | list[Any],
    prefix: tuple[str | int, ...] = (),
) -> list[tuple[str | int, ...]]:
    paths: list[tuple[str | int, ...]] = []
    items = enumerate(data) if isinstance(data, list) else data.items()
    for key, value in items:
        path = (*prefix, key)
        if isinstance(value, dict | list):
            child_paths = _leaf_paths(value, path)
            paths.extend(child_paths or [path])
        elif value is not None:
            paths.append(path)
    return paths


def _format_path(path: tuple[str | int, ...]) -> str:
    rendered = ""
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif rendered:
            rendered += f".{part}"
        else:
            rendered = part
    return rendered


def _alias_collision_losses(task_dir: Path) -> list[CompatibilityExportLoss]:
    paths = TaskPaths(task_dir)
    losses: list[CompatibilityExportLoss] = []
    if (
        paths.oracle_dir.exists()
        and paths.legacy_solution_dir.exists()
        and _file_map(paths.oracle_dir) != _file_map(paths.legacy_solution_dir)
    ):
        losses.append(
            CompatibilityExportLoss(
                path="oracle|solution",
                reason="native oracle/ and legacy solution/ both exist and differ",
            )
        )
    if (
        paths.verifier_source_dir.exists()
        and paths.legacy_tests_dir.exists()
        and _file_map(paths.verifier_source_dir) != _file_map(paths.legacy_tests_dir)
    ):
        losses.append(
            CompatibilityExportLoss(
                path="verifier|tests",
                reason="native verifier/ and legacy tests/ both exist and differ",
            )
        )
    if paths.task_document_path.exists() and paths.config_path.exists():
        try:
            document = TaskDocument.from_path(paths.task_document_path)
            legacy = import_task_config_toml(
                paths.config_path.read_text(),
                source="legacy",
            ).config
        except Exception as exc:
            losses.append(
                CompatibilityExportLoss(
                    path="task.md|task.toml",
                    reason=f"cannot prove native and legacy definitions equivalent: {exc}",
                )
            )
        else:
            if document.config.model_dump() != legacy.model_dump():
                losses.append(
                    CompatibilityExportLoss(
                        path="task.md|task.toml",
                        reason="native task.md config and legacy task.toml config differ",
                    )
                )
            if paths.instruction_path.exists() and _normalize_prompt(
                document.instruction
            ) != _normalize_prompt(paths.instruction_path.read_text()):
                losses.append(
                    CompatibilityExportLoss(
                        path="task.md|instruction.md",
                        reason="native task.md prompt and legacy instruction.md prompt differ",
                    )
                )
    return losses


def _copy_tree_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if source.is_dir():
        shutil.copytree(source, destination)
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _hash_export_output(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return _file_map(root, skip_prefixes={"compatibility/"})


def _file_map(root: Path, *, skip_prefixes: set[str] | None = None) -> dict[str, str]:
    import hashlib

    skip_prefixes = skip_prefixes or set()
    hashes: dict[str, str] = {}
    if not root.is_dir():
        return hashes
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(rel.startswith(prefix) for prefix in skip_prefixes):
            continue
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _rel_or_none(root: Path, path: Path) -> str | None:
    if not path.exists():
        return None
    return path.relative_to(root).as_posix()


def _normalize_prompt(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _imported_config_equal(left: Path, right: Path, *, source: str) -> bool:
    left_config = import_task_config_toml(left.read_text(), source=source).config
    right_config = import_task_config_toml(right.read_text(), source=source).config
    return left_config.model_dump() == right_config.model_dump()
