"""Tests for TaskPackage and TaskRuntimeView."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.rollout import _read_task_instruction
from benchflow.task import Task, TaskPackage, TaskRuntimeView
from benchflow.task.paths import TaskPaths


def _write_legacy_task(
    task: Path,
    *,
    instruction: str = "Legacy prompt",
    verifier_dirname: str = TaskPaths.LEGACY_TESTS_DIRNAME,
    oracle_dirname: str = TaskPaths.LEGACY_SOLUTION_DIRNAME,
) -> None:
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text(
        'version = "1.0"\n[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 120\n'
    )
    (task / "instruction.md").write_text(instruction)
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    verifier = task / verifier_dirname
    verifier.mkdir()
    (verifier / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    if oracle_dirname:
        oracle = task / oracle_dirname
        oracle.mkdir()
        (oracle / "solve.sh").write_text("#!/bin/bash\necho ok\n")


def _write_task_md(
    task: Path,
    *,
    prompt: str = "Native prompt",
    scenes_yaml: str = "",
    benchflow_yaml: str = "",
) -> None:
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.md").write_text(
        f"""---
version: "1.0"
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
environment:
  cpus: 1
  memory_mb: 2048
agents:
  roles:
    solver:
      agent: claude-agent-acp
{scenes_yaml}{benchflow_yaml}---
## prompt

{prompt}
"""
    )
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    verifier = task / TaskPaths.NATIVE_VERIFIER_DIRNAME
    verifier.mkdir()
    (verifier / "test.sh").write_text("#!/bin/bash\nexit 0\n")


def test_task_package_loads_task_dir() -> None:
    """Guards P0 TaskPackage boundary for on-disk task roots."""
    task_dir = Path("src/benchflow/demo_task")
    package = TaskPackage.load(task_dir)

    resolved = task_dir.resolve()
    assert package.task_dir == resolved
    assert package.paths.config_path == resolved / "task.toml"


def test_runtime_view_selects_legacy_split_entrypoint(tmp_path: Path) -> None:
    """Guards P0 authoritative entrypoint selection for Harbor split layout."""
    task = tmp_path / "legacy"
    _write_legacy_task(task, instruction="Do the legacy thing")

    view = TaskRuntimeView.from_task_dir(task)

    assert view.entrypoint == "legacy-split"
    assert view.instruction_text == "Do the legacy thing"
    assert view.document is None
    assert view.scenes == ()
    assert view.verifier_dir_kind == "legacy"
    assert view.oracle_dir_kind == "legacy"
    assert view.verifier_dir == task / "tests"
    assert view.oracle_dir == task / "solution"
    assert view.uses_native_verifier_dir is False
    assert view.uses_native_oracle_dir is False
    assert view.has_legacy_split_files is True
    assert view.alias_collisions.has_collisions is False


def test_runtime_view_selects_task_md_entrypoint(tmp_path: Path) -> None:
    """Guards P0 authoritative entrypoint selection for native task.md."""
    task = tmp_path / "native"
    _write_task_md(
        task,
        prompt="Solve from task.md",
        scenes_yaml="""scenes:
  - name: solve
    turns:
      - role: solver
""",
        benchflow_yaml="""benchflow:
  document_version: "0.3"
""",
    )

    view = TaskRuntimeView.from_task_dir(task)

    assert view.entrypoint == "task-md"
    assert view.instruction_text == "Solve from task.md"
    assert view.document is not None
    assert view.scene_names == ("solve",)
    assert len(view.scenes) == 1
    assert view.scenes[0].turns[0].role == "solver"
    assert view.verifier_dir_kind == "native"
    assert view.oracle_dir_kind == "legacy"
    assert view.verifier_dir == task / "verifier"
    assert view.benchflow["document_version"] == "0.3"
    assert view.has_legacy_split_files is False


def test_runtime_view_prefers_native_verifier_and_oracle_dirs(tmp_path: Path) -> None:
    """Guards P0 native-vs-legacy directory selection."""
    task = tmp_path / "native-dirs"
    _write_task_md(task)
    (task / "oracle").mkdir()
    (task / "oracle" / "solve.sh").write_text("#!/bin/bash\necho native\n")

    view = TaskRuntimeView.from_task_dir(task)

    assert view.verifier_dir_kind == "native"
    assert view.oracle_dir_kind == "native"
    assert view.verifier_dir == task / "verifier"
    assert view.oracle_dir == task / "oracle"
    assert "test.sh" in view.selected_verifier_tree_map()
    assert "solve.sh" in view.selected_oracle_tree_map()


def test_runtime_view_reports_alias_collisions(tmp_path: Path) -> None:
    """Guards P0 alias collision diagnostics without silent native preference."""
    task = tmp_path / "collision"
    _write_legacy_task(
        task,
        verifier_dirname=TaskPaths.NATIVE_VERIFIER_DIRNAME,
        oracle_dirname=TaskPaths.NATIVE_ORACLE_DIRNAME,
    )
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho legacy\n")
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\necho legacy\n")

    with pytest.raises(ValueError, match="verifier/"):
        TaskRuntimeView.from_task_dir(task)

    view = TaskRuntimeView.from_task_dir(task, fail_on_alias_collision=False)
    assert view.alias_collisions.has_collisions is True
    assert any("verifier/" in issue for issue in view.alias_collisions.issues)
    assert any("tests/" in issue for issue in view.alias_collisions.issues)


def test_runtime_view_accepts_byte_identical_alias_trees(tmp_path: Path) -> None:
    """Equivalent native and legacy alias trees should not report collisions."""
    task = tmp_path / "equivalent"
    _write_legacy_task(
        task,
        verifier_dirname=TaskPaths.NATIVE_VERIFIER_DIRNAME,
        oracle_dirname=TaskPaths.NATIVE_ORACLE_DIRNAME,
    )
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\necho ok\n")

    view = TaskRuntimeView.from_task_dir(task)

    assert view.alias_collisions.has_collisions is False


def test_task_runtime_view_property_matches_from_task_dir(tmp_path: Path) -> None:
    """Guards Task.runtime_view integration without duplicating parse logic."""
    task = tmp_path / "task"
    _write_task_md(task, prompt="Through Task class")

    loaded = Task(task)
    assert loaded.runtime_view.entrypoint == "task-md"
    assert loaded.runtime_view.instruction_text == "Through Task class"
    assert loaded.runtime_view.scene_names == ()


def test_read_task_instruction_uses_runtime_view(tmp_path: Path) -> None:
    """Guards rollout instruction materialization through TaskRuntimeView."""
    task = tmp_path / "task-md"
    _write_task_md(task, prompt="Materialized for /instruction.md")

    assert _read_task_instruction(task) == "Materialized for /instruction.md"


def test_runtime_view_detects_legacy_files_alongside_task_md(tmp_path: Path) -> None:
    """Guards visibility of split files when task.md is authoritative."""
    task = tmp_path / "mixed"
    _write_task_md(task, prompt="Authoritative prompt")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Legacy prompt\n")

    view = TaskRuntimeView.from_task_dir(task)

    assert view.entrypoint == "task-md"
    assert view.instruction_text == "Authoritative prompt"
    assert view.has_legacy_split_files is True
