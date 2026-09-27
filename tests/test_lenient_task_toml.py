"""Lenient task.toml on the run path.

Harbor lets its task schema grow and 88 Harbor adapters write task.toml files
from newer schemas; BenchFlow refused any unknown key at run time
(``extra="forbid"``), so one new upstream key stopped a whole Harbor task. A
legacy task.toml now loads with unknown keys ignored and a warning (and a
``bench tasks check`` warning, which does not fail the check). Keys whose
semantics BenchFlow cannot honour stay refused through the runtime
capability check: Harbor 0.23's ``[[verifier.collect]]`` hooks. The native
``TaskConfig.model_validate_toml`` and ``task.md`` stay strict.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.task import Task, TaskConfig
from benchflow.task.runtime_capabilities import validate_task_runtime_support
from benchflow.task.runtime_view import TaskRuntimeView

# Harbor's examples/tasks/hello-world/task.toml (harbor-framework/harbor
# 6cb9ff31, Apache-2.0), verbatim.
HARBOR_HELLO_WORLD = """\
schema_version = "1.4"

[task]
name = "harbor/hello-world"
version = "1.0.0"
authors = []
keywords = []

[metadata]
author_name = "Alex Shaw"
author_email = "author@example.com"
difficulty = "easy"
category = "programming"
tags = [ "trivial",]

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 120.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
storage_mb = 10240
gpus = 0
mcp_servers = []

[verifier.env]

[solution.env]
"""

# A top-level key goes before the first table; a new table goes at the end.
TOP = 'future_top_level = "from a newer Harbor schema"\n'
EXTRA_TABLE = """
[agent.future_knob]
enabled = true
"""
WITH_EXTRA = TOP + HARBOR_HELLO_WORLD + EXTRA_TABLE


def _task(root: Path, toml: str) -> Path:
    d = root / "hello-world"
    (d / "environment").mkdir(parents=True)
    (d / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (d / "tests").mkdir()
    (d / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (d / "solution").mkdir()
    (d / "solution" / "solve.sh").write_text("#!/bin/bash\necho hi\n")
    (d / "instruction.md").write_text("Write hello.txt.\n")
    (d / "task.toml").write_text(toml)
    return d


def test_unknown_keys_load_with_a_warning(tmp_path: Path, caplog) -> None:
    d = _task(tmp_path, WITH_EXTRA)
    with caplog.at_level(logging.WARNING):
        task = Task(d)
    assert task.name == "harbor/hello-world"
    assert task.config.sandbox.memory_mb == 2048
    assert task.config.ignored_keys == (
        "agent.future_knob.enabled",
        "future_top_level",
    )
    text = caplog.text
    assert "future_top_level" in text and "agent.future_knob.enabled" in text
    assert "ignored" in text
    # Unknown keys without semantics BenchFlow knows of are not runtime issues.
    assert validate_task_runtime_support(task.config, sandbox="docker") == []


def test_the_native_parser_stays_strict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        TaskConfig.model_validate_toml(WITH_EXTRA)


def test_a_wrong_type_still_fails(tmp_path: Path) -> None:
    d = _task(tmp_path, HARBOR_HELLO_WORLD.replace("cpus = 1", 'cpus = "one"'))
    with pytest.raises(ValueError):
        Task(d)


def test_verifier_collect_is_refused_at_run_time(tmp_path: Path) -> None:
    toml = HARBOR_HELLO_WORLD + (
        '\n[[verifier.collect]]\nservice = "main"\ncommand = "cat /tmp/x"\n'
    )
    task = Task(_task(tmp_path, toml))
    issues = validate_task_runtime_support(task.config, sandbox="daytona")
    assert [i.path for i in issues] == ["verifier.collect"]
    assert "not executed" in issues[0].reason


def test_runtime_view_reads_the_same_config(tmp_path: Path) -> None:
    view = TaskRuntimeView.from_task_dir(_task(tmp_path, WITH_EXTRA))
    assert view.config.ignored_keys == ("agent.future_knob.enabled", "future_top_level")


def test_tasks_check_warns_but_passes(tmp_path: Path) -> None:
    from benchflow._utils.task_authoring import check_task, check_task_warnings
    from benchflow.cli.main import app

    d = _task(tmp_path, WITH_EXTRA)
    assert check_task(d) == []
    warnings = check_task_warnings(d)
    assert any("future_top_level" in w and "ignored" in w for w in warnings)
    result = CliRunner().invoke(app, ["tasks", "check", str(d)])
    assert result.exit_code == 0, result.output
    assert "warning" in result.output.lower() and "future_top_level" in result.output


def test_tasks_check_names_the_refused_key(tmp_path: Path) -> None:
    from benchflow._utils.task_authoring import check_task, check_task_warnings

    toml = HARBOR_HELLO_WORLD + '\n[[verifier.collect]]\ncommand = "true"\n'
    d = _task(tmp_path, toml)
    assert any(
        "verifier.collect" in w and "refused" in w for w in check_task_warnings(d)
    )
    issues = check_task(d, sandbox_type="docker")
    assert any("verifier.collect" in i for i in issues)


def test_the_warning_is_logged_once_per_file(tmp_path: Path, caplog) -> None:
    """A run loads a task more than once; the warning was printed each time."""
    d = _task(tmp_path, WITH_EXTRA)
    with caplog.at_level(logging.WARNING):
        Task(d)
        Task(d)
        TaskRuntimeView.from_task_dir(d)
    assert caplog.text.count("future_top_level") == 1
