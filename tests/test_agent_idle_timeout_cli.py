"""Tests for the ``--agent-idle-timeout`` CLI flag on ``bench eval create``.

Closes #338: long-running optimization tasks need to extend or disable the
ACP idle watchdog without patching the installed source.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import typer
from typer.testing import CliRunner

from benchflow.cli.main import app, eval_create
from benchflow.evaluation import EvaluationResult


def _make_tasks_dir(tmp_path: Path) -> Path:
    """Build a minimal directory with two task subdirs for batch dispatch."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for name in ("alpha", "beta"):
        d = tasks / name
        d.mkdir()
        (d / "task.toml").write_text('schema_version = "1.1"\n')
        (d / "instruction.md").write_text("solve\n")
    return tasks


def _stub_evaluation_run(captured: dict):
    """Patch ``Evaluation.run`` to capture the constructed config and short-circuit."""

    async def fake_run(self):
        captured["config"] = self._config
        return EvaluationResult(
            job_name="test",
            config=self._config,
            total=0,
            passed=0,
            errored=0,
        )

    return patch("benchflow.evaluation.Evaluation.run", new=fake_run)


def test_eval_create_integer_idle_timeout_lands_in_config(tmp_path: Path):
    """Integer arg propagates verbatim into EvaluationConfig.agent_idle_timeout."""
    tasks = _make_tasks_dir(tmp_path)
    captured: dict = {}

    with _stub_evaluation_run(captured):
        eval_create(
            tasks_dir=tasks,
            agent_idle_timeout="300",
            jobs_dir=str(tmp_path / "jobs"),
        )

    assert captured["config"].agent_idle_timeout == 300


def test_eval_create_none_string_disables_idle_watchdog(tmp_path: Path):
    """`--agent-idle-timeout none` maps to None so the watchdog is disabled."""
    tasks = _make_tasks_dir(tmp_path)
    captured: dict = {}

    with _stub_evaluation_run(captured):
        eval_create(
            tasks_dir=tasks,
            agent_idle_timeout="none",
            jobs_dir=str(tmp_path / "jobs"),
        )

    assert captured["config"].agent_idle_timeout is None


def test_eval_create_zero_string_disables_idle_watchdog(tmp_path: Path):
    """`--agent-idle-timeout 0` also disables (parity with the YAML contract)."""
    tasks = _make_tasks_dir(tmp_path)
    captured: dict = {}

    with _stub_evaluation_run(captured):
        eval_create(
            tasks_dir=tasks,
            agent_idle_timeout="0",
            jobs_dir=str(tmp_path / "jobs"),
        )

    assert captured["config"].agent_idle_timeout is None


def test_eval_create_default_idle_timeout_is_600(tmp_path: Path):
    """No flag keeps the 600s default that the issue body asked us to preserve."""
    tasks = _make_tasks_dir(tmp_path)
    captured: dict = {}

    with _stub_evaluation_run(captured):
        eval_create(
            tasks_dir=tasks,
            jobs_dir=str(tmp_path / "jobs"),
        )

    assert captured["config"].agent_idle_timeout == 600


def test_eval_create_rejects_invalid_idle_timeout(tmp_path: Path):
    """Non-numeric, non-sentinel strings raise typer.Exit with a friendly message."""
    tasks = _make_tasks_dir(tmp_path)

    try:
        eval_create(
            tasks_dir=tasks,
            agent_idle_timeout="forever",
            jobs_dir=str(tmp_path / "jobs"),
        )
    except typer.Exit as exc:
        assert exc.exit_code == 1
    else:
        raise AssertionError("expected typer.Exit for invalid idle timeout")


def test_eval_create_idle_timeout_overrides_yaml_config(tmp_path: Path):
    """Explicit CLI flag overrides any ``agent_idle_timeout`` set in the YAML."""
    tasks = _make_tasks_dir(tmp_path)
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "tasks_dir: " + str(tasks) + "\n"
        "agent: claude-agent-acp\n"
        "agent_idle_timeout: 1800\n"
    )
    captured: dict = {}

    with _stub_evaluation_run(captured):
        eval_create(
            config_file=yaml_path,
            agent_idle_timeout="none",
            jobs_dir=str(tmp_path / "jobs"),
        )

    # CLI explicit "none" wins over YAML's 1800
    assert captured["config"].agent_idle_timeout is None


def test_cli_runner_help_documents_none_sentinel():
    """Help text should describe both integer and 'none' usage."""
    import re

    result = CliRunner().invoke(app, ["eval", "create", "--help"])
    assert result.exit_code == 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--agent-idle-timeout" in plain
    assert "none" in plain
