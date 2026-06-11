"""Tests for the ``--context-root`` CLI flag / YAML key on ``bench eval create``.

Guards the fix for BF-7 (CLAWSBENCH-COMPAT.md): ``context_root`` /
``stage_dockerfile_deps`` existed end-to-end in benchflow
(``EvaluationConfig.context_root`` → ``RolloutConfig`` →
``stage_dockerfile_deps``) but neither operator entry point could set it —
``eval create`` had no ``--context-root`` flag and ``Evaluation.from_yaml``
had no key — so task Dockerfiles that COPY repo-root paths could not build
without a bespoke pre-staging script.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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


def test_eval_create_context_root_lands_in_config(tmp_path: Path):
    """--context-root propagates into EvaluationConfig.context_root."""
    tasks = _make_tasks_dir(tmp_path)
    captured: dict = {}

    with _stub_evaluation_run(captured):
        eval_create(
            tasks_dir=tasks,
            context_root=tmp_path,
            jobs_dir=str(tmp_path / "jobs"),
        )

    assert captured["config"].context_root == str(tmp_path)


def test_eval_create_default_context_root_is_none(tmp_path: Path):
    """No flag keeps the existing default (no Dockerfile dep staging)."""
    tasks = _make_tasks_dir(tmp_path)
    captured: dict = {}

    with _stub_evaluation_run(captured):
        eval_create(
            tasks_dir=tasks,
            jobs_dir=str(tmp_path / "jobs"),
        )

    assert captured["config"].context_root is None


def test_eval_create_yaml_context_root_key_parses(tmp_path: Path):
    """The native-YAML ``context_root`` key reaches EvaluationConfig."""
    tasks = _make_tasks_dir(tmp_path)
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "tasks_dir: " + str(tasks) + "\n"
        "agent: claude-agent-acp\n"
        "context_root: " + str(tmp_path) + "\n"
    )
    captured: dict = {}

    with _stub_evaluation_run(captured):
        eval_create(
            config_file=yaml_path,
            jobs_dir=str(tmp_path / "jobs"),
        )

    assert captured["config"].context_root == str(tmp_path)


def test_eval_create_context_root_overrides_yaml_config(tmp_path: Path):
    """Explicit CLI flag overrides any ``context_root`` set in the YAML."""
    tasks = _make_tasks_dir(tmp_path)
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "tasks_dir: " + str(tasks) + "\n"
        "agent: claude-agent-acp\n"
        "context_root: /somewhere/else\n"
    )
    captured: dict = {}

    with _stub_evaluation_run(captured):
        eval_create(
            config_file=yaml_path,
            context_root=tmp_path,
            jobs_dir=str(tmp_path / "jobs"),
        )

    assert captured["config"].context_root == str(tmp_path)


def test_eval_create_yaml_without_key_keeps_default_none(tmp_path: Path):
    """A YAML config without the key keeps the None default."""
    tasks = _make_tasks_dir(tmp_path)
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("tasks_dir: " + str(tasks) + "\nagent: claude-agent-acp\n")
    captured: dict = {}

    with _stub_evaluation_run(captured):
        eval_create(
            config_file=yaml_path,
            jobs_dir=str(tmp_path / "jobs"),
        )

    assert captured["config"].context_root is None


def test_cli_runner_help_documents_context_root():
    """Help text should document the flag (full name only, per ENG-74)."""
    import re

    result = CliRunner().invoke(app, ["eval", "create", "--help"])
    assert result.exit_code == 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--context-root" in plain
