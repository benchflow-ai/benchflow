"""Tests for native task.md export to Harbor/Pier split layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.task import TaskConfig, TaskDocument
from benchflow.task.export import ExportLoss, export_task_package

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
