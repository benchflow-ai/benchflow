"""Dogfood acceptance tests for task-standard benchflow-wanted-features packages."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow._utils.task_authoring import check_task
from benchflow.task import Task, TaskRuntimeView
from benchflow.task.export import ExportLoss, export_task_package
from benchflow.task.runtime_capabilities import validate_task_runtime_support

DOGFOOD_ROOT = Path(
    "docs/examples/task-standard/benchflow-wanted-features"
)
COMPAT_EXPORT_TASK = DOGFOOD_ROOT / "compat-export-loss-reports"
RUNTIME_GATE_TASK = DOGFOOD_ROOT / "runtime-capability-gate"

_EXPECTED_EXPORT_LOSS_CONCEPTS = frozenset(
    {
        "agents",
        "scenes",
        "benchflow.document_version",
        "benchflow.compatibility",
        "benchflow.traceability",
        "benchflow.verifier",
        "prompt.role:adapter_engineer",
        "prompt.role:compatibility_reviewer",
        "verifier.verifier_md",
        "verifier.rubrics",
    }
)


def _dogfood_task_dirs() -> list[Path]:
    return sorted(
        path
        for path in DOGFOOD_ROOT.iterdir()
        if path.is_dir() and (path / "task.md").is_file()
    )


@pytest.mark.parametrize("task_dir", _dogfood_task_dirs(), ids=lambda p: p.name)
def test_dogfood_check_task_passes(task_dir: Path) -> None:
    """Guards bench tasks check acceptance for every wanted-features dogfood package."""
    assert check_task(task_dir) == []


@pytest.mark.parametrize("task_dir", _dogfood_task_dirs(), ids=lambda p: p.name)
def test_dogfood_task_loads(task_dir: Path) -> None:
    """Guards Task() loading for every wanted-features dogfood package."""
    task = Task(task_dir)
    assert task.runtime_view.entrypoint == "task-md"


@pytest.mark.parametrize("task_dir", _dogfood_task_dirs(), ids=lambda p: p.name)
def test_dogfood_runtime_view_from_task_dir(task_dir: Path) -> None:
    """Guards TaskRuntimeView.from_task_dir for every wanted-features dogfood package."""
    view = TaskRuntimeView.from_task_dir(task_dir)
    assert view.task_dir == task_dir.resolve()
    assert view.entrypoint == "task-md"
    assert view.instruction_text.strip()


def test_compat_export_loss_reports_degraded_export() -> None:
    """Guards compat-export-loss-reports degraded Harbor export and loss concepts."""
    result = export_task_package(COMPAT_EXPORT_TASK)

    assert result.target == "harbor"
    assert result.mode == "degraded"
    assert result.selected_definition == "task.md"
    assert result.selected_oracle_dir == "oracle/"
    assert result.selected_verifier_dir == "verifier/"
    assert result.exported_oracle_dir == "solution/"
    assert result.exported_verifier_dir == "tests/"
    assert result.losses
    assert all(isinstance(loss, ExportLoss) for loss in result.losses)

    concepts = {loss.concept for loss in result.losses}
    assert concepts == _EXPECTED_EXPORT_LOSS_CONCEPTS


def test_runtime_capability_gate_fails_on_docker() -> None:
    """Guards runtime-capability-gate fail-closed validation before sandbox creation."""
    task = Task(RUNTIME_GATE_TASK)
    issues = validate_task_runtime_support(task, "docker", RUNTIME_GATE_TASK)

    assert issues
    paths = {issue.path for issue in issues}
    assert "environment.workdir" in paths
    assert all(issue.sandbox_type == "docker" for issue in issues)
