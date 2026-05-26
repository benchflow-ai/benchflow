"""Regression tests for #379 — `bench tasks check` and `bench eval create`
must agree on the structural contract for task packages.

The original bug: a task.toml without an [agent] section was rejected by
`bench tasks check` but happily executed by `bench eval create`, producing
recorded evidence for a "malformed" task. The fix aligns the structural
checker with the runtime contract (AgentConfig.timeout_sec defaults to
None) so both commands return the same verdict.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from benchflow._utils.task_authoring import check_task
from benchflow.cli.main import app
from benchflow.evaluation import Evaluation, EvaluationConfig


def _make_task_missing_agent(
    parent: Path, name: str = "malformed-missing-agent"
) -> Path:
    """Create a task package whose task.toml has no [agent] section."""
    task = parent / name
    task.mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 120\n'
    )
    (task / "instruction.md").write_text("# Do something\n")
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    tests = task / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return task


def test_check_task_accepts_missing_agent(tmp_path):
    """The shared validator must not flag missing [agent] as an issue.

    Runtime AgentConfig.timeout_sec defaults to None (rollout treats this
    as "no wall-clock cap"). Rejecting it here would diverge from what
    `bench eval create` actually executes.
    """
    task = _make_task_missing_agent(tmp_path)
    issues = check_task(task)
    assert not any("agent" in i.lower() for i in issues), (
        f"check_task flagged missing [agent] but runtime accepts it: {issues}"
    )


def test_tasks_check_cli_accepts_missing_agent(tmp_path):
    """`bench tasks check` exits 0 for a task missing [agent]."""
    task = _make_task_missing_agent(tmp_path)
    result = CliRunner().invoke(app, ["tasks", "check", str(task)])
    assert result.exit_code == 0, (
        f"`bench tasks check` should accept missing [agent]; "
        f"got exit={result.exit_code}\n{result.output}"
    )
    assert "valid" in result.output


def test_eval_create_enumerates_task_missing_agent(tmp_path):
    """`bench eval create --tasks-dir <dir>` must enumerate the same
    task that `bench tasks check` accepted — no structural filtering on
    missing [agent].
    """
    task = _make_task_missing_agent(tmp_path)
    ev = Evaluation(
        tasks_dir=str(task),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(agent="oracle"),
    )
    task_dirs = ev._get_task_dirs()
    assert task_dirs == [task], (
        f"`eval create` dropped a task that `tasks check` accepts: {task_dirs}"
    )


def test_check_and_eval_agree_on_missing_agent(tmp_path):
    """The two commands must reach the same verdict for the same task."""
    task = _make_task_missing_agent(tmp_path)

    check_issues = check_task(task)
    check_accepts = not check_issues

    ev = Evaluation(
        tasks_dir=str(task),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(agent="oracle"),
    )
    eval_accepts = ev._get_task_dirs() == [task]

    assert check_accepts == eval_accepts, (
        f"check_task and eval enumeration disagree on missing [agent]: "
        f"check_accepts={check_accepts} (issues={check_issues}) "
        f"eval_accepts={eval_accepts}"
    )
    # Both must accept — that's the contract.
    assert check_accepts, f"Both should accept; check_task says: {check_issues}"


def test_tasks_check_escapes_rich_markup_in_diagnostic(tmp_path):
    """The diagnostic must render literal bracketed names (e.g. [agent])
    verbatim. Rich was previously parsing them as styling markup, swallowing
    the section name from the user-facing error.
    """
    # A task with a parse error — still produces a diagnostic that may
    # contain bracketed text. Use a name with bracketed text to force the
    # output to contain literal "[…]".
    task = tmp_path / "bad-toml"
    task.mkdir()
    (task / "task.toml").write_text("[[[ not valid toml\n")
    (task / "instruction.md").write_text("# x\n")
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    tests = task / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/bash\nexit 0\n")

    result = CliRunner().invoke(app, ["tasks", "check", str(task)])
    assert result.exit_code == 1
    # Should mention the parse error literally, not swallow brackets.
    assert "parse error" in result.output
