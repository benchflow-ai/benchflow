"""DX lane A: one error taxonomy for the failures a task author meets first.

Each failure class gets its category, one line saying what happened, one
saying what to do, and the file or line it belongs to, with no traceback at
the default log level, and no retry when another attempt would repeat it
(DF-p5-1, -5, -7, -8, -10, -17;
DF-p2-5, -11). The end-to-end runs are in
tests/test_dx_a_container.py.
"""

from __future__ import annotations

import errno
import json
import logging
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import typer

from benchflow._utils.scoring import classify_error, classify_verifier_error
from benchflow.cli._errors import run_cli
from benchflow.evaluation import RetryConfig
from benchflow.failures import cause_of, describe

# BuildKit's own words, as `compose build` failed on 66471e95 (the dogfood
# fixtures e-build, missing-hub and missing-image), cut to what matters.
_COMPOSE = (
    "Docker compose command failed for environment e-build. Command: docker "
    "--config /tmp/benchflow-docker-config-x compose --project-name e-build "
    "-f docker-compose-base.yaml -f docker-compose-build.yaml build. Return "
    "code: 1. Stdout: "
)
RUN_FAILS = (
    _COMPOSE + "#6 [3/4] RUN apt-get install -y no-such-package-xyz\n"
    "#6 0.268 Reading package lists...\n"
    "#6 0.281 Building dependency tree...\n"
    "#6 0.282 E: Unable to locate package no-such-package-xyz\n"
    '#6 ERROR: process "/bin/sh -c apt-get install -y no-such-package-xyz" '
    "did not complete successfully: exit code: 100\n"
    ". Stderr: Dockerfile:5\n\n--------------------\n\n"
    "   4 |     COPY input.txt /app/data/input.txt\n\n"
    "   5 | >>> RUN apt-get install -y no-such-package-xyz\n\n"
    "   6 |     \n\n--------------------\n\n"
    'failed to solve: process "/bin/sh -c apt-get install -y '
    'no-such-package-xyz" did not complete successfully: exit code: 100\n\n.'
)
HUB_MISSING = (
    _COMPOSE
    + "#3 [internal] load metadata for docker.io/library/ubuntu:99.99-p5-nope\n"
    "#3 ERROR: docker.io/library/ubuntu:99.99-p5-nope: not found\n"
    ". Stderr: Dockerfile:2\n\n--------------------\n\n"
    "   1 |     # canary\n\n   2 | >>> FROM ubuntu:99.99-p5-nope\n\n"
    "--------------------\n\nfailed to solve: ubuntu:99.99-p5-nope: failed to "
    "resolve source metadata for docker.io/library/ubuntu:99.99-p5-nope: "
    "docker.io/library/ubuntu:99.99-p5-nope: not found\n\n."
)
GHCR_PRIVATE = (
    _COMPOSE + ". Stderr: Dockerfile:2\n\n--------------------\n\n"
    "   2 | >>> FROM ghcr.io/benchflow-ai/does-not-exist-p5:1.0\n\n"
    "--------------------\n\nfailed to solve: failed to fetch anonymous token: "
    "unexpected status from GET request to https://ghcr.io/token?scope=x: "
    "403 Forbidden\n\n."
)
REGISTRY_HICCUP = (
    _COMPOSE + ". Stderr: Dockerfile:2\n\n   2 | >>> FROM ubuntu:24.04\n\n"
    "failed to solve: ubuntu:24.04: failed to resolve source metadata for "
    "docker.io/library/ubuntu:24.04: failed to do request: Head "
    '"https://registry-1.docker.io/v2/": dial tcp: i/o timeout\n\n.'
)


# --- DF-p5-8, DF-p2-5: a task image that does not build ----------------------


@pytest.mark.parametrize(
    ("output", "where", "says"),
    [
        (
            RUN_FAILS,
            "environment/Dockerfile:5 `RUN apt-get install -y no-such-package-xyz`",
            "exited with code 100 (E: Unable to locate package no-such-package-xyz)",
        ),
        (
            HUB_MISSING,
            "environment/Dockerfile:2 `FROM ubuntu:99.99-p5-nope`",
            "base image ubuntu:99.99-p5-nope is not found or not public (not found)",
        ),
        (
            GHCR_PRIVATE,
            "environment/Dockerfile:2 `FROM ghcr.io/benchflow-ai/does-not-exist-p5:1.0`",
            "is not found or not public (403 Forbidden)",
        ),
    ],
    ids=["run-step-fails", "hub-image-missing", "ghcr-image-private"],
)
def test_a_build_another_attempt_repeats_is_the_tasks_at_its_dockerfile_line(
    output, where, says
):
    from benchflow._utils.text import describe_exception
    from benchflow.sandbox.image_build import ImageBuildError, image_build_error

    error = image_build_error(output)
    assert isinstance(error, ImageBuildError)
    assert error.fault == "task"
    assert str(error).startswith(f"the task's image did not build: {where} ")
    assert says in str(error)
    assert "docker-compose-base.yaml" not in str(error)  # no compose command line
    assert "docker build <task>/environment" in error.hint
    recorded = describe_exception(error)  # what result.json keeps
    assert classify_error(recorded) == "image_build"
    assert not RetryConfig().should_retry(recorded)
    assert not RetryConfig().reruns_on_resume(recorded)
    cause = cause_of(
        {"task_name": "e-build", "error": recorded, "error_category": "image_build"},
        task_dir=Path("tasks/e-build"),
    )
    assert cause is not None and cause.fault == "task"
    assert cause.label.startswith(f"the task's image did not build: {where}")
    assert cause.next_step == (
        "fix that line in the task's environment/Dockerfile; "
        "`docker build tasks/e-build/environment` prints the whole build log"
    )


def test_a_registry_hiccup_while_building_is_still_retried():
    from benchflow.sandbox.image_build import image_build_error

    assert image_build_error(REGISTRY_HICCUP) is None
    assert classify_error(f"RuntimeError: {REGISTRY_HICCUP}") == "sandbox_setup"
    assert RetryConfig().should_retry(f"RuntimeError: {REGISTRY_HICCUP}")


def test_an_agents_words_cannot_claim_the_image_build_category():
    quoted = "ACP error -32603: the task's image did not build: try again"
    assert classify_error(quoted) == "acp_error"


def test_the_image_build_page_explains_it(capsys):
    from benchflow.cli.main import app

    with pytest.raises(SystemExit) as caught:
        run_cli(app, ["explain", "image_build"])
    assert caught.value.code == 0
    out = capsys.readouterr().out
    assert "the task's image did not build" in out
    assert "Dockerfile" in out


# --- DF-p5-1: the oracle out of time ------------------------------------------


def test_the_oracle_out_of_its_agent_timeout_is_a_timeout_never_retried():
    import asyncio

    from benchflow.rollout._setup import OracleTimeoutError, _run_oracle
    from benchflow.sandbox.protocol import ExecTimeoutError

    class Env:
        async def exec(self, command, **kwargs):
            if kwargs.get("timeout_sec") == 5:
                raise ExecTimeoutError(5)

            class Done:
                return_code, stdout, stderr = 0, "", ""

            return Done()

    root, task = _oracle_task()
    with pytest.raises(OracleTimeoutError) as caught:
        asyncio.run(_run_oracle(Env(), root, 5, task=task))
    text = str(caught.value)
    assert text == "oracle/solve.sh timed out after 5s, the task's agent timeout"
    assert classify_error(text) == "timeout"
    assert not RetryConfig().should_retry(text)
    cause = describe("timeout")
    assert cause is not None and cause.fault == "agent"
    assert "agent.timeout_sec" in cause.next_step


def _oracle_task():
    import tempfile

    from benchflow.task import Task

    root = Path(tempfile.mkdtemp()) / "agent-timeout"
    (root / "oracle").mkdir(parents=True)
    (root / "oracle" / "solve.sh").write_text("sleep 60\n")
    (root / "verifier").mkdir()
    (root / "verifier" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    (root / "environment").mkdir()
    (root / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (root / "task.md").write_text(
        '---\nschema_version: "1.3"\nagent:\n  timeout_sec: 5\n---\n'
        "# agent-timeout\n\n## prompt\n\nWrite hello.\n"
    )
    return root, Task(root)


def test_every_backends_exec_timeout_keeps_its_message():
    from benchflow.sandbox.protocol import ExecTimeoutError

    error = ExecTimeoutError(5)
    assert isinstance(error, RuntimeError)
    assert str(error) == "Command timed out after 5 seconds"
    # A transport's own timeout is still the sandbox's, and retried.
    assert classify_error(f"RuntimeError: {error}") == "infra_failure"


# --- DF-p5-17: a verifier timeout is not retried --------------------------


def test_a_verifier_timeout_never_reruns_the_agent():
    """Neither the run nor a resume repeats the trial."""
    timed_out = "verifier timed out after 5.0s"
    assert classify_verifier_error(timed_out) == "verifier_timeout"
    assert not RetryConfig().should_retry_verifier_error(timed_out)
    assert not RetryConfig().reruns_verifier_error_on_resume(timed_out)
    # A verifier that never started is the sandbox's: still retried.
    wedged = "verifier_wedge: test.sh never started"
    assert RetryConfig().should_retry_verifier_error(wedged)


# --- DF-p2-11: a crashed verifier -----------------------------------------


def _empty_reward(tmp_path: Path, return_code: int):
    from benchflow.task import verifier_core
    from benchflow.task.verifier_errors import RewardFileEmptyError

    reward = tmp_path / "reward.txt"
    reward.write_text("")
    named = verifier_core.Verifier._empty_reward_named(None, "test.sh", return_code)
    with pytest.raises(RewardFileEmptyError) as caught, named:
        raise RewardFileEmptyError(
            "verifier/reward.txt is empty", file="verifier/reward.txt", path=reward
        )
    return str(caught.value)


def test_an_empty_reward_names_test_sh_its_exit_and_the_file(tmp_path):
    assert (
        _empty_reward(tmp_path, 1)
        == "test.sh exited 1 and wrote an empty verifier/reward.txt"
    )
    assert _empty_reward(tmp_path, 0) == "test.sh wrote an empty verifier/reward.txt"
    error = "verifier crashed: RewardFileEmptyError: " + _empty_reward(tmp_path, 1)
    assert classify_verifier_error(error) == "verifier_failure"
    assert "<reachable:" not in error


def test_a_crashed_verifiers_next_step_is_its_output_not_a_passing_check():
    cause = describe("verifier_failure")
    assert cause is not None
    assert "bench tasks check" not in cause.next_step
    assert "verifier/test-stdout.txt" in cause.next_step


def test_the_summary_quotes_the_crashed_verifiers_last_lines(tmp_path, capsys):
    from benchflow.cli._shared import _report_outcomes

    job = tmp_path / "job"
    trial = job / "b1-crash__1"
    (trial / "verifier").mkdir(parents=True)
    (trial / "verifier" / "test-stdout.txt").write_text(
        "Traceback (most recent call last):\n"
        '  File "/tests/check.py", line 9, in <module>\n'
        "    open(REWARD, 'w').write(str(score()))\n"
        "NameError: name 'score' is not defined\x1b[31m\n"
    )
    result = {
        "task_name": "b1-crash",
        "rollout_name": "b1-crash__1",
        "verifier_error": "verifier crashed: RewardFileEmptyError: test.sh exited "
        "1 and wrote an empty verifier/reward.txt",
        "verifier_error_category": "verifier_failure",
        "rewards": None,
    }

    from types import SimpleNamespace

    _report_outcomes(SimpleNamespace(results={"b1-crash": result}), job)
    out = capsys.readouterr().out
    assert "1 unscored: the verifier crashed (task problem): b1-crash" in out
    assert f"test.sh's last lines ({trial}/verifier/test-stdout.txt):" in out
    assert "| NameError: name 'score' is not defined?[31m" in out  # no escape
    assert "\x1b" not in out
    assert "next: fix the verifier from its output" in out
    assert out.index("NameError") < out.index("next:")


def test_accepting_a_reward_is_said_only_once_it_has_parsed():
    source = (
        Path(__file__).parents[1] / "src/benchflow/task/verifier_core.py"
    ).read_text()
    assert "but produced reward output; accepting reward" not in source
    assert "exited with rc=%d after writing its reward" in source


# --- DF-p5-5: a full disk --------------------------------------------------


def test_a_full_disk_is_said_as_such_with_a_next_step_and_no_traceback(
    capsys, tmp_path, monkeypatch
):
    monkeypatch.setenv("BENCHFLOW_LOG_DIR", str(tmp_path / "logs"))
    app = typer.Typer()

    @app.command()
    def go() -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    @app.command()
    def other() -> None:
        pass

    with pytest.raises(SystemExit) as caught:
        run_cli(app, ["go"])
    assert caught.value.code == 1
    err = capsys.readouterr().err
    # No file named: the places it could be (review L-A2).
    assert "a disk BenchFlow writes to is full (the jobs folder, the cache" in err
    assert "next: free space on that disk, or pass another --jobs-dir" in err
    assert "bug in BenchFlow" not in err
    assert "Traceback" not in err


def test_summary_json_is_whole_or_untouched_on_a_full_disk(tmp_path, monkeypatch):
    import os

    from benchflow._utils.disk_space import DiskFullError, write_text_atomic

    summary = tmp_path / "summary.json"
    summary.write_text('{"old": true}')

    def full(fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "fsync", full)
    with pytest.raises(DiskFullError) as caught:
        write_text_atomic(summary, json.dumps({"new": True}))
    assert summary.read_text() == '{"old": true}'
    assert [p.name for p in tmp_path.iterdir()] == ["summary.json"]  # no temp left
    assert str(caught.value).startswith("the disk that holds the job folder is full")
    assert f"writing {summary}" in str(caught.value)
    assert caught.value.hint == "free space on that disk, or pass another --jobs-dir"


def test_a_job_writes_its_summary_atomically():
    source = (Path(__file__).parents[1] / "src/benchflow/evaluation.py").read_text()
    assert '(job_dir / "summary.json").write_text' not in source
    assert 'write_text_atomic(job_dir / "summary.json", summary_text)' in source


def test_a_full_disk_is_the_users_never_retried_and_resumable(tmp_path):
    error = "Unexpected: [Errno 28] No space left on device"
    assert classify_error(error) == "disk_full"
    assert not RetryConfig().should_retry(error)
    assert RetryConfig().reruns_on_resume(error)
    # The sandbox's own disk filling up is not the host's.
    assert classify_error("ACP error -32603: No space left on device") == "acp_error"
    cause = describe("disk_full")
    assert cause is not None and cause.fault == "setup"
    lost = "verifier crashed: RewardFileEmptyError: " + _empty_reward_on_full_disk(
        tmp_path
    )
    assert classify_verifier_error(lost) == "disk_full"
    assert RetryConfig().reruns_verifier_error_on_resume(lost)


def _empty_reward_on_full_disk(tmp_path: Path) -> str:
    from benchflow._utils import disk_space

    original = disk_space.free_bytes
    disk_space.free_bytes = lambda where: 24 * 1024
    try:
        text = _empty_reward(tmp_path, 1)
    finally:
        disk_space.free_bytes = original
    assert text == (
        "verifier/reward.txt is empty: the disk that holds the job folder is "
        "full (24 KB free), so test.sh could not write its reward"
    )
    return text


# --- DF-p5-7: one line per failure, the traceback only with -v --------------


def test_the_err_line_says_the_classified_reason(caplog, tmp_path):
    from benchflow.evaluation import Evaluation, RunResult

    result = RunResult(
        task_name="e-build",
        error="ImageBuildError: the task's image did not build: "
        "environment/Dockerfile:5 `RUN false` exited with code 1",
    )
    evaluation = Evaluation.__new__(Evaluation)
    evaluation._on_result = None
    with caplog.at_level(logging.INFO):
        evaluation._log_and_report(tmp_path / "e-build", result)
    line = next(r.getMessage() for r in caplog.records if "[ERR]" in r.getMessage())
    assert "(the task's image did not build:" in line
    assert "ImageBuildError" not in line


def test_dash_v_prints_debug_logs(monkeypatch):
    from benchflow.cli.main import app

    root = logging.getLogger()
    monkeypatch.setattr(root, "level", logging.INFO)
    with pytest.raises(SystemExit):
        run_cli(app, ["-v", "explain", "image_build"])
    assert root.level == logging.DEBUG


def test_a_failed_trial_logs_one_line_and_its_traceback_only_at_debug():
    source = (
        Path(__file__).parents[1] / "src/benchflow/rollout/__init__.py"
    ).read_text()
    assert 'logger.error("Run failed", exc_info=True)' not in source
    assert 'logger.debug("Run failed", exc_info=True)' in source


# --- DF-p5-10: Ctrl-C ----------------------------------------------------------

_INTERRUPTED = """
import asyncio, os, signal, sys
from pathlib import Path
from benchflow.cli._termination import run_until_terminated

marker = Path(sys.argv[1])

async def job():
    try:
        print("ready", flush=True)
        await asyncio.sleep(60)
    finally:
        await asyncio.sleep(0.2)
        marker.write_text("cleaned")

run_until_terminated(job())
"""


def test_ctrl_c_cleans_up_says_so_and_exits_130(tmp_path):
    marker = tmp_path / "cleanup.txt"
    proc = subprocess.Popen(
        [sys.executable, "-c", _INTERRUPTED, str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # A terminal's Ctrl-C reaches a process whose SIGINT is the default.
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )
    assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
    proc.send_signal(signal.SIGINT)
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 130, err
    assert marker.read_text() == "cleaned"
    assert "Interrupted (Ctrl-C): the running trials were stopped" in err
    assert "Traceback" not in err


def test_a_job_stopped_before_any_result_is_a_job_to_resume(tmp_path, capsys):
    from benchflow.cli.main import app

    job = tmp_path / "jobs-int" / "int4"
    trial = job / "slow__021ed28e"
    trial.mkdir(parents=True)
    (job / "evaluation.json").write_text(json.dumps({"tasks_dir": "tasks"}))
    (trial / "config.json").write_text(json.dumps({"agent": "oracle"}))
    (trial / "budget-cancelled.json").write_text(
        json.dumps({"version": 1, "reason": "interrupted", "timing": {}})
    )
    with pytest.raises(SystemExit) as caught:
        run_cli(app, ["results", str(job)])
    assert caught.value.code == 1
    err = capsys.readouterr().err
    assert f"{job} is a job that was stopped before any trial finished" in err
    assert "1 trial started, none wrote result.json" in err
    assert f"next: `bench eval resume {job}` runs them" in err
    assert "pass a job directory" not in err


def test_the_live_displays_last_frame_says_stopped():
    from rich.console import Console

    from benchflow.cli._live_progress import LiveEvalProgress

    console = Console(record=True, width=100, force_terminal=True)
    live = LiveEvalProgress(
        console, label="slow", agent="oracle", model=None, sandbox="docker"
    )
    with pytest.raises(KeyboardInterrupt), live:
        live.on_plan(1, 0, 1)
        live.on_task_start("slow")
        raise KeyboardInterrupt
    text = console.export_text()
    last = text[text.rindex("benchflow") :]
    assert "stopped" in last
    assert "starting…" not in last


def test_a_cancelled_trials_receipt_says_why(tmp_path):
    import asyncio

    from benchflow.budget import CANCELLED_BUDGET_RECORD, cancelled_attempt_receipt

    async def trial():
        with cancelled_attempt_receipt(lambda: tmp_path):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(trial())
    receipt = json.loads((tmp_path / CANCELLED_BUDGET_RECORD).read_text())
    assert receipt["reason"] == "interrupted"
