"""DX lane A, round 2: four more first-run errors and outputs.

From DF-p1-first-run-7, -9, -13 and
DF-p4-upgrader-8: each message says what happened, what to do and the
file or path it is about, with no traceback or Python repr. The
end-to-end runs are re-run by the lane's repro script on the VM.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils.scoring import classify_error
from benchflow.cli._errors import run_cli
from benchflow.evaluation import RetryConfig
from benchflow.failures import cause_of, describe


def _cli(argv: list[str]) -> int:
    from benchflow.cli.main import app

    with pytest.raises(SystemExit) as caught:
        run_cli(app, argv)
    return int(caught.value.code or 0)


# --- DF-p1-first-run-7: a misspelled --source-repo --------------------------


def test_a_repository_github_does_not_show_is_named_without_a_prompt(
    tmp_path, monkeypatch
):
    from benchflow._utils import benchmark_repos

    real = subprocess.run
    seen: dict = {}

    def run(cmd, *args, **kwargs):
        if list(cmd[:2]) == ["git", "clone"]:
            seen["env"] = kwargs.get("env") or {}
            raise subprocess.CalledProcessError(
                128,
                cmd,
                stderr="fatal: could not read Username for 'https://github.com': "
                "terminal prompts disabled\n",
            )
        return real(cmd, *args, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark_repos.subprocess, "run", run)
    with pytest.raises(benchmark_repos.RepositoryNotFoundError) as caught:
        benchmark_repos._clone_repo("benchflow-ai", "skilsbench", None, "tasks/x")
    assert seen["env"].get("GIT_TERMINAL_PROMPT") == "0"
    assert str(caught.value) == (
        "repository benchflow-ai/skilsbench was not found on GitHub, or it is private"
    )
    assert "https://github.com/benchflow-ai/skilsbench" in caught.value.hint
    assert "gh auth login" in caught.value.hint


def test_the_cli_prints_the_repository_line_and_its_fix_not_a_repr(
    tmp_path, monkeypatch, capsys
):
    from benchflow._utils import benchmark_repos

    def missing(*args, **kwargs):
        raise benchmark_repos.RepositoryNotFoundError("benchflow-ai", "skilsbench")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark_repos, "resolve_source_with_metadata", missing)
    code = _cli(
        [
            "eval",
            "run",
            "--source-repo",
            "benchflow-ai/skilsbench",
            "--source-path",
            "tasks/citation-check",
            "--agent",
            "oracle",
        ]
    )
    err = capsys.readouterr().err
    assert code == 1
    assert (
        "repository benchflow-ai/skilsbench was not found on GitHub, or it is private"
        in err
    )
    assert "next: check the name (https://github.com/benchflow-ai/skilsbench)" in err
    assert "Command '[" not in err and "returned non-zero exit status" not in err
    assert "Traceback" not in err


def test_another_clone_failure_is_gits_last_line_not_a_repr(tmp_path, monkeypatch):
    from benchflow._utils import benchmark_repos

    real = subprocess.run

    def run(cmd, *args, **kwargs):
        if list(cmd[:2]) == ["git", "clone"]:
            raise subprocess.CalledProcessError(
                128,
                cmd,
                stderr="warning: x\nfatal: Remote branch nope not found in upstream origin\n",
            )
        return real(cmd, *args, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark_repos.subprocess, "run", run)
    with pytest.raises(RuntimeError) as caught:
        benchmark_repos._clone_repo("benchflow-ai", "skillsbench", "nope", None)
    assert str(caught.value) == (
        "git could not clone benchflow-ai/skillsbench: fatal: Remote branch "
        "nope not found in upstream origin"
    )


# --- DF-p1-first-run-9: where BenchFlow looked ----------------------------


@pytest.mark.parametrize(
    ("argv", "says", "next_step"),
    [
        (
            ["eval", "list", "jobs/"],
            "No such jobs directory: jobs (looked in {cwd}/jobs)",
            "next: pass a job folder",
        ),
        (
            ["eval", "metrics", "jobs/oracle"],
            "Not a directory: jobs/oracle (looked in {cwd}/jobs/oracle)",
            "next: pass a job folder",
        ),
        (
            ["eval", "run", "--tasks-dir", "tasks/hello", "--agent", "oracle"],
            "--tasks-dir not found: tasks/hello (looked in {cwd}/tasks/hello)",
            "next: pass a local task folder; a task on GitHub goes in --source-repo and --source-path",
        ),
        (
            ["eval", "run", "--config", "run.yaml"],
            "--config not found: run.yaml (looked in {cwd}/run.yaml)",
            "next: pass the path of a run config YAML file",
        ),
    ],
    ids=["list", "metrics", "tasks-dir", "config"],
)
def test_a_wrong_path_says_where_benchflow_looked(
    argv, says, next_step, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    assert _cli(argv) != 0
    captured = capsys.readouterr()
    text = " ".join((captured.out + captured.err).split())
    assert " ".join(says.format(cwd=tmp_path).split()) in text
    assert next_step in text
    assert "Traceback" not in text


def test_no_jobs_yet_says_where_it_looked(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert _cli(["eval", "list"]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert f"No jobs yet in jobs (looked in {tmp_path}/jobs)" in out


def test_the_viewer_says_where_it_looked_and_what_to_pass(
    tmp_path, monkeypatch, capsys
):
    from benchflow.trajectories.viewer import server

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as caught:
        server.serve("jobs/oracle")
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert (
        f"Not a file or directory: jobs/oracle (looked in {tmp_path}/jobs/oracle)"
        in err
    )
    assert "next: pass a job folder, a trial folder or a trajectory .jsonl file" in err


# --- DF-p1-first-run-13: the nop control's expected 0 -----------------------


def _result(agent: str, passed: int, total: int):
    from benchflow.evaluation import TaskFailure

    return SimpleNamespace(
        config=SimpleNamespace(agent=agent),
        total=total,
        passed=passed,
        errored=0,
        verifier_errored=0,
        score=passed / total,
        mean_reward=passed / total,
        task_failures=[
            TaskFailure(task_name="hello", rewards={"reward": 0.0}, verifier_error=None)
        ]
        if passed < total
        else [],
        results={},
    )


def test_a_nop_controls_zero_is_said_to_be_expected(capsys):
    from benchflow.cli._shared import _report_eval_result

    _report_eval_result(_result("nop", 0, 1))
    out = capsys.readouterr().out
    assert (
        "nop control: 0/1, as expected: doing nothing passes none of the tasks" in out
    )
    assert "✗ Score" not in out
    assert "✗ hello" not in out


def test_a_nop_control_that_passes_is_flagged(capsys):
    from benchflow.cli._shared import _report_eval_result

    _report_eval_result(_result("nop", 1, 1))
    out = capsys.readouterr().out
    assert "nop control: 1/1 passed with nothing done" in out
    assert "can be passed without solving them" in out


def test_an_agents_zero_is_still_a_failure(capsys):
    from benchflow.cli._shared import _report_eval_result

    _report_eval_result(_result("claude-agent-acp", 0, 1))
    out = capsys.readouterr().out
    assert "✗ Score: 0/1" in out
    assert "control" not in out


def test_an_empty_trajectorys_reward_note_is_a_debug_line(caplog):
    from benchflow.trajectories.export_adp import trajectory_to_adp_record

    with caplog.at_level(logging.DEBUG, logger="benchflow"):
        trajectory_to_adp_record(trajectory_id="hello__1", events=[], reward=0.0)
    notes = [r for r in caplog.records if "no action to carry it" in r.getMessage()]
    assert notes and all(r.levelno == logging.DEBUG for r in notes)


# --- DF-p4-upgrader-8: a task.toml typo inside a job ----------------------

TYPO = (
    "TaskConfigKeyError: /w/tasks/legacy/legacy-split/task.toml: - unknown "
    "task.toml key 'agent.timout_sec' in [agent] — did you mean "
    "'timeout_sec'? Fix the typo, or set BENCHFLOW_TASK_TOML_ALLOW_UNKNOWN_KEYS=1 "
    "to load unknown keys anyway."
)


def test_a_task_file_that_does_not_load_is_the_tasks_with_its_fix():
    assert classify_error(TYPO) == "task_config"
    assert not RetryConfig().should_retry(TYPO)
    cause = cause_of(
        {"task_name": "legacy-split", "error": TYPO},
        task_dir=Path("/w/tasks/legacy/legacy-split"),
    )
    assert cause is not None and cause.fault == "task"
    assert cause.label == (
        "a task file does not load: task.toml: unknown task.toml key "
        "'agent.timout_sec' in [agent] — did you mean 'timeout_sec'? Fix the "
        "typo, or set BENCHFLOW_TASK_TOML_ALLOW_UNKNOWN_KEYS=1 to load unknown "
        "keys anyway."
    )
    assert cause.next_step.startswith(
        "`bench check /w/tasks/legacy/legacy-split` names the problem"
    )
    assert cause.headline().endswith("(task problem)")


def test_a_relative_task_folder_still_gives_the_file_under_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    error = TYPO.replace("/w/tasks/", f"{tmp_path}/tasks/")
    cause = cause_of(
        {"task_name": "legacy-split", "error": error},
        task_dir=Path("tasks/legacy/legacy-split"),
    )
    assert cause is not None
    assert cause.label.startswith("a task file does not load: task.toml: unknown")


def test_every_task_file_error_shares_the_category():
    for name in (
        "TaskDocumentParseError",
        "VerifierDocumentParseError",
        "MalformedTaskError",
        "ReviewRubricError",
    ):
        assert classify_error(f"{name}: task.md: front matter") == "task_config"
    # The class must open the error: an agent's words cannot claim it.
    assert classify_error("ACP error -32603: TaskConfigKeyError: x") == "acp_error"
    page = describe("task_config")
    assert page is not None and "bench check" in page.next_step


def test_the_task_config_page_explains_it(capsys):
    assert _cli(["explain", "task_config"]) == 0
    out = capsys.readouterr().out
    assert "a task file does not load" in out
    assert "task.toml" in out
    json.dumps(out)  # plain text
