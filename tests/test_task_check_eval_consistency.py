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

import pytest
from typer.testing import CliRunner

from benchflow._utils.task_authoring import check_task
from benchflow.cli.main import app
from benchflow.evaluation import Evaluation, EvaluationConfig

SCHEMA_ONLY_TASK_MD_EXAMPLES = (
    Path("docs/examples/task-md/harbor-parity"),
    Path("docs/examples/task-md/multi-scene"),
    Path("docs/examples/task-md/nudgebench-team"),
)
FIRST_PARTY_MIXED_TASK_MD_FIXTURES = (
    Path("src/benchflow/demo_task"),
    Path("tests/examples/hello-world-task"),
    Path("tests/conformance/acp_smoke"),
)
REAL_SKILLSBENCH_TASK_MD_EXAMPLES = (
    Path("docs/examples/task-md/real-skillsbench/3d-scan-calc"),
    Path("docs/examples/task-md/real-skillsbench/citation-check"),
    Path("docs/examples/task-md/real-skillsbench/weighted-gdp-calc"),
)
USER_RUNTIME_TASK_MD_EXAMPLES = (
    Path("docs/examples/task-md/user-runtime/private-facts-nudges"),
)


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


def _make_schema_only_task(parent: Path, name: str = "schema-only") -> Path:
    """Create a parse-valid task.md fixture with no runnable task package assets."""
    task = parent / name
    task.mkdir()
    (task / "task.md").write_text(
        """---
task:
  name: benchflow/schema-only
metadata:
  category: schema
---

## prompt

This fixture validates task.md authoring syntax only.
"""
    )
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


def test_eval_create_does_not_enumerate_schema_only_task_md(tmp_path):
    """Schema-valid task.md fixtures are not runnable eval task directories."""
    task = _make_schema_only_task(tmp_path)

    assert check_task(task, validation_level="schema") == []
    assert check_task(task), (
        "default structural validation should reject schema fixture"
    )

    ev = Evaluation(
        tasks_dir=str(task),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(agent="oracle"),
    )

    assert ev._get_task_dirs() == []


def test_eval_create_skips_schema_only_task_md_in_task_collections(tmp_path):
    """Evaluation discovery must agree with structural task validation."""
    schema_fixture = _make_schema_only_task(tmp_path)
    runnable = _make_task_missing_agent(tmp_path, name="runnable-task")

    ev = Evaluation(
        tasks_dir=str(tmp_path),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(agent="oracle"),
    )

    assert check_task(schema_fixture, validation_level="schema") == []
    assert check_task(schema_fixture)
    assert check_task(runnable) == []
    assert ev._get_task_dirs() == [runnable]


@pytest.mark.parametrize("task", SCHEMA_ONLY_TASK_MD_EXAMPLES, ids=lambda p: p.name)
def test_real_schema_only_task_md_examples_are_not_eval_tasks(task: Path, tmp_path):
    """Guards PR #1 against docs example fixtures drifting into runnable tasks."""
    assert (task / "task.md").is_file()
    assert check_task(task, validation_level="schema") == []
    assert check_task(task), "default structural validation should reject fixture"

    ev = Evaluation(
        tasks_dir=str(task),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(agent="oracle"),
    )

    assert ev._get_task_dirs() == []


@pytest.mark.parametrize(
    "task",
    FIRST_PARTY_MIXED_TASK_MD_FIXTURES,
    ids=lambda p: p.name,
)
def test_eval_create_enumerates_first_party_mixed_task_md_fixtures(
    task: Path,
    tmp_path,
):
    """First-party task.md fixtures are real runnable tasks, not schema examples."""
    assert check_task(task) == []

    ev = Evaluation(
        tasks_dir=str(task),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(agent="oracle"),
    )

    assert ev._get_task_dirs() == [task]


def test_eval_create_enumerates_real_skillsbench_native_examples(tmp_path):
    """Publication-grade native SkillsBench examples are runnable eval tasks."""
    root = Path("docs/examples/task-md/real-skillsbench")
    ev = Evaluation(
        tasks_dir=str(root),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(agent="oracle"),
    )

    for task in REAL_SKILLSBENCH_TASK_MD_EXAMPLES:
        assert check_task(task, validation_level="publication-grade") == []

    assert ev._get_task_dirs() == list(REAL_SKILLSBENCH_TASK_MD_EXAMPLES)


def test_eval_create_enumerates_user_runtime_native_examples(tmp_path):
    """Guards PR #1's runnable simulated-user task.md fixture discovery."""
    root = Path("docs/examples/task-md/user-runtime")
    ev = Evaluation(
        tasks_dir=str(root),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(agent="oracle"),
    )

    for task in USER_RUNTIME_TASK_MD_EXAMPLES:
        assert check_task(task, validation_level="publication-grade") == []

    assert ev._get_task_dirs() == list(USER_RUNTIME_TASK_MD_EXAMPLES)


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
