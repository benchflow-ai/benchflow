"""Tests for native task.md export to Harbor/Pier split layout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.task import TaskConfig, TaskDocument
from benchflow.task.export import (
    EXPORT_REPORT_REL_PATH,
    ExportLoss,
    export_report_json,
    export_task_package,
    import_split_task_package,
    materialize_export_result,
    validate_export_round_trip,
)

COMPAT_EXPORT_EXAMPLE = Path(
    "docs/examples/task-standard/benchflow-wanted-features/compat-export-loss-reports"
)


def test_export_compat_example_emits_split_layout_and_loss_report() -> None:
    """Guards F1/F6 compat-export-loss-reports dogfood acceptance."""
    document = TaskDocument.from_path(COMPAT_EXPORT_EXAMPLE / "task.md")
    result = export_task_package(COMPAT_EXPORT_EXAMPLE)

    assert result.target == "harbor"
    assert result.mode == "degraded"
    assert result.selected_definition == "task.md"
    assert result.selected_oracle_dir == "oracle/"
    assert result.selected_verifier_dir == "verifier/"
    assert result.exported_oracle_dir == "solution/"
    assert result.exported_verifier_dir == "tests/"

    assert "task.toml" in result.files
    assert "instruction.md" in result.files
    assert any(path.startswith("solution/") for path in result.files)
    assert any(path.startswith("tests/") for path in result.files)
    assert "environment/Dockerfile" in result.files

    exported_config = TaskConfig.model_validate_toml(result.files["task.toml"])
    assert exported_config.model_dump() == document.config.model_dump()
    assert (
        result.files["instruction.md"].strip()
        == document.instruction.strip()
    )

    concepts = {loss.concept for loss in result.losses}
    assert "agents" in concepts
    assert "scenes" in concepts
    assert "benchflow.document_version" in concepts
    assert "benchflow.compatibility" in concepts
    assert "benchflow.traceability" in concepts
    assert "benchflow.verifier" in concepts
    assert "prompt.role:adapter_engineer" in concepts
    assert "prompt.role:compatibility_reviewer" in concepts
    assert "verifier.verifier_md" in concepts
    assert "verifier.rubrics" in concepts

    assert result.input_hashes["task.md"]
    assert result.output_hashes["task.toml"]
    assert result.output_hashes["instruction.md"]
    assert result.output_hashes["tests/test.sh"]
    assert result.output_hashes["solution/solve.md"]


def test_export_requires_task_md() -> None:
    """Guards native export against legacy-only packages."""
    with pytest.raises(FileNotFoundError, match=r"task\.md"):
        export_task_package("src/benchflow/demo_task")


def test_export_loss_is_typed() -> None:
    """Guards F6 requirement for typed export losses."""
    result = export_task_package(COMPAT_EXPORT_EXAMPLE)
    assert result.losses
    assert all(isinstance(loss, ExportLoss) for loss in result.losses)
    assert all(loss.concept and loss.reason for loss in result.losses)


def test_export_report_json_serializes_losses_hashes_mode_and_target() -> None:
    """Guards on-disk compatibility/export-report.json schema for bench tasks export."""
    result = export_task_package(COMPAT_EXPORT_EXAMPLE, target="pier")
    report = export_report_json(result)

    assert report["target"] == "pier"
    assert report["mode"] == "degraded"
    assert report["losses"] == [
        {"concept": loss.concept, "reason": loss.reason} for loss in result.losses
    ]
    assert report["input_hashes"] == result.input_hashes
    assert report["output_hashes"] == result.output_hashes
    assert report["selected_definition"] == "task.md"
    assert report["selected_oracle_dir"] == "oracle/"
    assert report["selected_verifier_dir"] == "verifier/"
    assert report["exported_oracle_dir"] == "solution/"
    assert report["exported_verifier_dir"] == "tests/"
    assert report["ignored_aliases"] == list(result.ignored_aliases)

    # Round-trip through JSON to mirror CLI write path.
    parsed = json.loads(json.dumps(report))
    assert parsed["target"] == "pier"
    assert parsed["mode"] == "degraded"
    assert len(parsed["losses"]) == len(result.losses)
    assert parsed["input_hashes"]["task.md"]
    assert parsed["output_hashes"]["task.toml"]
    assert EXPORT_REPORT_REL_PATH == "compatibility/export-report.json"


def test_export_round_trip_preserves_harbor_compatible_fields(tmp_path: Path) -> None:
    """Guards F1 compat-export-loss-reports export→materialize→import round-trip parity."""
    result = export_task_package(COMPAT_EXPORT_EXAMPLE)
    exported_dir = materialize_export_result(result, tmp_path / "exported")

    assert (exported_dir / "task.toml").is_file()
    assert (exported_dir / "instruction.md").is_file()
    assert (exported_dir / EXPORT_REPORT_REL_PATH).is_file()

    issues = validate_export_round_trip(COMPAT_EXPORT_EXAMPLE, exported_dir)
    assert issues == []

    imported = import_split_task_package(exported_dir)
    document = TaskDocument.from_path(COMPAT_EXPORT_EXAMPLE / "task.md")

    assert imported.config.model_dump() == document.config.model_dump()
    assert imported.instruction_normalized == document.instruction.strip() + "\n"
    assert imported.oracle_hashes["solution/solve.md"] == result.output_hashes["solution/solve.md"]
    assert imported.verifier_hashes["tests/test.sh"] == result.output_hashes["tests/test.sh"]
    assert imported.environment_hashes["environment/Dockerfile"]
    assert imported.native_comparison is None


def test_import_split_task_package_requires_split_layout(tmp_path: Path) -> None:
    """Guards split import against native-only packages."""
    native_only = tmp_path / "native-only"
    native_only.mkdir()
    (native_only / "task.md").write_text("---\n---\n\n## prompt\n\nHi.\n")

    with pytest.raises(FileNotFoundError, match=r"task\.toml"):
        import_split_task_package(native_only)


def test_validate_export_round_trip_reports_config_drift(tmp_path: Path) -> None:
    """Guards round-trip validation surfaces semantic drift."""
    result = export_task_package(COMPAT_EXPORT_EXAMPLE)
    exported_dir = materialize_export_result(result, tmp_path / "exported")
    (exported_dir / "task.toml").write_text(
        (exported_dir / "task.toml").read_text().replace("7200", "3600")
    )

    issues = validate_export_round_trip(COMPAT_EXPORT_EXAMPLE, exported_dir)
    assert issues == ["Config drift: canonical TaskConfig dumps differ"]
