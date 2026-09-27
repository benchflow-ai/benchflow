"""Harbor multi-step tasks are refused by the runtime capability check.

A Harbor multi-step task keeps its prompts under ``steps/<name>/instruction.md``
and has no root ``instruction.md``. The documented behaviour (docs/
running-benchmarks.md, "Refused before launch") is the runtime capability
refusal naming ``steps``. Before this fix the loader raised "no task document
... expected task.md (native), or legacy task.toml + instruction.md" first, so
the refusal was unreachable for any real multi-step task: ``bench eval run``
recorded a per-trial FileNotFoundError and ``bench tasks check`` told the
author to add a root instruction.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.task import Task
from benchflow.task.runtime_capabilities import (
    UnsupportedTaskFeatureError,
    raise_for_task_runtime_support,
    validate_task_runtime_support,
)
from benchflow.task.runtime_view import TaskRuntimeView

MULTI_STEP_TOML = """\
version = "1.0"

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048

[[steps]]
name = "create-file"

[steps.agent]
timeout_sec = 30.0

[[steps]]
name = "append-content"

[steps.agent]
timeout_sec = 30.0
"""


def _multi_step_task(root: Path) -> Path:
    task = root / "multi-step"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "task.toml").write_text(MULTI_STEP_TOML)
    for name, text in (("create-file", "Create a.txt."), ("append-content", "Append.")):
        step = task / "steps" / name
        (step / "tests").mkdir(parents=True)
        (step / "instruction.md").write_text(text + "\n")
        (step / "tests" / "test.sh").write_text(
            "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
        )
    return task


def test_multi_step_task_loads_and_is_refused_on_steps(tmp_path: Path) -> None:
    task_dir = _multi_step_task(tmp_path)

    task = Task(task_dir)

    assert len(task.config.steps) == 2
    issues = validate_task_runtime_support(
        task.config, sandbox="daytona", task_dir=task_dir
    )
    assert [issue.path for issue in issues] == ["steps"]
    with pytest.raises(UnsupportedTaskFeatureError, match="steps"):
        raise_for_task_runtime_support(task.config, sandbox="docker", task_dir=task_dir)


def test_multi_step_runtime_view_reads_from_disk(tmp_path: Path) -> None:
    view = TaskRuntimeView.from_task_dir(_multi_step_task(tmp_path))

    assert len(view.config.steps) == 2
    assert view.prompt == ""


def test_single_step_task_without_instruction_still_names_the_formats(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "no-prompt"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text('version = "1.0"\n')

    with pytest.raises(FileNotFoundError, match=r"expected task\.md"):
        Task(task_dir)


def test_tasks_check_reports_the_steps_refusal_not_a_missing_instruction(
    tmp_path: Path,
) -> None:
    from benchflow.cli.main import app

    task_dir = _multi_step_task(tmp_path)

    result = CliRunner().invoke(
        app, ["tasks", "check", str(task_dir), "--sandbox", "daytona"]
    )

    assert result.exit_code == 1
    out = " ".join(result.output.split())
    assert "Unsupported runtime feature: steps" in out
    assert "Missing required file: instruction.md" not in out
    assert "runtime capability parse error" not in out


def test_rollout_prompt_resolution_defers_to_the_launch_gate(tmp_path: Path) -> None:
    """The rollout reads the prompt before the launch gate runs, so it must not
    fail first on the absent root instruction.md of a multi-step task."""
    from benchflow.rollout._setup import _resolve_prompts

    assert _resolve_prompts(_multi_step_task(tmp_path), None) == [""]
