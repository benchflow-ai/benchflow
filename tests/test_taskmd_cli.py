"""``bench tasks check`` on a task.md draft 2 folder prints the reference checker's report and BenchFlow's."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from tests._taskmd_helpers import EXAMPLES


@pytest.fixture(autouse=True)
def format_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "cache"))


def test_a_runnable_package_passes_both_checks() -> None:
    result = CliRunner().invoke(app, ["tasks", "check", str(EXAMPLES / "hello-world")])
    assert result.exit_code == 0, result.output
    assert "reference checker (task-md tools/taskmd.py check):" in result.output
    assert "grading needs a model: no" in result.output
    assert "BenchFlow:" in result.output and "honored: [agent] timeout" in result.output
    assert "taskmd task format: checking the materialized package" in result.output
    assert "valid (structural)" in result.output


def test_a_refused_package_fails_and_names_the_fields() -> None:
    result = CliRunner().invoke(app, ["tasks", "check", str(EXAMPLES / "calc-quarterly")])
    assert result.exit_code == 1
    assert "refused: [sandbox] clock" in result.output
    assert "refused: [world]" in result.output
    assert "materialized package" not in result.output


def test_a_package_refused_only_for_agents_passes_with_a_note(monkeypatch) -> None:
    from benchflow.taskmd import family
    from tests.test_taskmd_format import _host_docker

    monkeypatch.setattr(family, "_docker", _host_docker)
    monkeypatch.setattr(family.shutil, "which", lambda name: "/usr/bin/docker")
    from benchflow.taskmd import check

    monkeypatch.setattr(check.shutil, "which", lambda name: "/usr/bin/docker")
    result = CliRunner().invoke(app, ["tasks", "check", str(EXAMPLES / "sql-family")])
    assert result.exit_code == 0, result.output
    assert "refused when an agent runs" in result.output and "[agent] budget" in result.output
    assert "materialized seed" in result.output
