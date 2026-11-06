"""DX lane A, the round-1 review and the decision on verifier timeouts.

- A-1: a network or registry failure inside a Dockerfile RUN step is the
  sandbox's, retried; a registry login failure is the user's setup.
- A-2: summary.json keeps the mode a plain write gives it (umask-based).
- Timeouts: a verifier timeout never re-runs the agent; a preserved
  trial gets one verifier-only re-run, in the run or on resume, and a second
  timeout is final.
- L-A1..L-A5: the lows.
"""

from __future__ import annotations

import contextlib
import errno
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.scoring import classify_error, classify_verifier_error
from benchflow.evaluation import RetryConfig
from benchflow.failures import cause_of
from tests.test_verifier_recovery import _save_solver
from tests.test_verifier_recovery import child as child
from tests.test_verifier_recovery import trial as trial

# --- A-1: BuildKit output of a RUN step that failed on the network ----------

_COMPOSE = (
    "Docker compose command failed for environment t. Command: docker compose "
    "build. Return code: 1. Stdout: "
)


def _run_step(lines: list[str], step: str, code: int) -> str:
    out = "".join(f"#7 0.{i}12 {line}\n" for i, line in enumerate(lines))
    return (
        _COMPOSE
        + f"#7 [3/4] {step}\n"
        + out
        + f'#7 ERROR: process "/bin/sh -c {step[4:]}" did not complete '
        f"successfully: exit code: {code}\n. Stderr: Dockerfile:4\n\n"
        f"   4 | >>> {step}\n\n"
        f'failed to solve: process "/bin/sh -c {step[4:]}" did not complete '
        f"successfully: exit code: {code}\n\n."
    )


# Real apt, pip and curl wording for a mirror or index that failed.
APT_DNS = _run_step(
    [
        "Err:1 http://deb.debian.org/debian bookworm InRelease",
        "  Temporary failure resolving 'deb.debian.org'",
        "W: Failed to fetch http://deb.debian.org/debian/dists/bookworm/InRelease  "
        "Temporary failure resolving 'deb.debian.org'",
        "E: Unable to locate package curl",
    ],
    "RUN apt-get update && apt-get install -y curl",
    100,
)
APT_HASH = _run_step(
    [
        "E: Failed to fetch http://deb.debian.org/debian/pool/main/c/curl/curl.deb  "
        "Hash Sum mismatch",
        "E: Unable to fetch some archives, maybe run apt-get update",
    ],
    "RUN apt-get install -y curl",
    100,
)
PIP_503 = _run_step(
    [
        "ERROR: Could not install packages due to an OSError: HTTPSConnectionPool"
        "(host='files.pythonhosted.org', port=443): Read timed out.",
        "ERROR: HTTP error 503 while getting https://files.pythonhosted.org/x.whl",
    ],
    "RUN pip install -r requirements.txt",
    1,
)
PIP_RATE = _run_step(
    ["ERROR: 429 Client Error: Too Many Requests for url: https://pypi.org/simple/x/"],
    "RUN pip install x",
    1,
)
CURL_NET = _run_step(
    ["curl: (6) Could not resolve host: github.com"],
    "RUN curl -fsSL https://github.com/x/y/archive/main.tar.gz | tar -xz",
    6,
)
# A RUN step that is the task's: no network marker in its output.
APT_MISSING = _run_step(
    ["E: Unable to locate package no-such-package-xyz"],
    "RUN apt-get install -y no-such-package-xyz",
    100,
)
# A FROM pull the registry refused this machine's login for.
FROM_LOGIN = (
    _COMPOSE
    + ". Stderr: Dockerfile:1\n\n   1 | >>> FROM registry.example.com/team/base:1\n\n"
    "failed to solve: registry.example.com/team/base:1: failed to resolve source "
    "metadata for registry.example.com/team/base:1: failed to authorize: failed "
    "to fetch oauth token: unexpected status from POST request to "
    "https://registry.example.com/token: 401 Unauthorized\n\n."
)
FROM_DENIED = (
    _COMPOSE + ". Stderr: Dockerfile:1\n\n   1 | >>> FROM private/base:1\n\n"
    "failed to solve: private/base:1: pull access denied, repository does not "
    "exist or may require authorization: server message: insufficient_scope: "
    "authorization failed\n\n."
)


@pytest.mark.parametrize(
    "output",
    [APT_DNS, APT_HASH, PIP_503, PIP_RATE, CURL_NET],
    ids=["apt-dns", "apt-hash-sum", "pip-read-timeout-503", "pip-429", "curl-dns"],
)
def test_a_network_failure_inside_a_run_step_is_retried_infrastructure(output):
    """A-1, fails on 7a1a4773: these were image_build (task), never retried."""
    from benchflow._utils.text import describe_exception
    from benchflow.sandbox.image_build import image_build_error

    error = image_build_error(output)
    assert error is not None
    recorded = describe_exception(error)
    assert classify_error(recorded) == "sandbox_setup", recorded
    assert RetryConfig().should_retry(recorded)
    assert "network or registry failure while the image built" in recorded
    cause = cause_of({"task_name": "t", "error": recorded})
    assert cause is not None and cause.fault == "infrastructure"


def test_the_run_steps_own_error_is_still_the_tasks():
    from benchflow._utils.text import describe_exception
    from benchflow.sandbox.image_build import image_build_error

    recorded = describe_exception(image_build_error(APT_MISSING))
    assert classify_error(recorded) == "image_build"
    assert not RetryConfig().should_retry(recorded)


@pytest.mark.parametrize("output", [FROM_LOGIN, FROM_DENIED], ids=["401", "denied"])
def test_a_registry_login_failure_is_the_users_setup(output):
    """A-1, fails on 7a1a4773: told to fix the task's Dockerfile."""
    from benchflow._utils.text import describe_exception
    from benchflow.sandbox.image_build import image_build_error

    recorded = describe_exception(image_build_error(output))
    assert classify_error(recorded) == "registry_auth", recorded
    assert not RetryConfig().should_retry(recorded)
    cause = cause_of({"task_name": "t", "error": recorded})
    assert cause is not None and cause.fault == "setup"
    assert "docker login" in cause.next_step
    assert "fix that line" not in cause.next_step


# --- A-2: summary.json's mode --------------------------------------------


def test_summary_json_gets_the_mode_a_plain_write_gives(tmp_path):
    """A-2, fails on 7a1a4773: mkstemp left it 0600."""
    from benchflow._utils.disk_space import write_text_atomic

    old = os.umask(0o022)
    try:
        write_text_atomic(tmp_path / "summary.json", "{}")
        (tmp_path / "plain.json").write_text("{}")
    finally:
        os.umask(old)
    mode = (tmp_path / "summary.json").stat().st_mode & 0o777
    assert mode == (tmp_path / "plain.json").stat().st_mode & 0o777 == 0o644


def test_a_replaced_summary_keeps_its_own_mode(tmp_path):
    from benchflow._utils.disk_space import write_text_atomic

    target = tmp_path / "summary.json"
    target.write_text("{}")
    target.chmod(0o640)
    write_text_atomic(target, '{"n": 1}')
    assert target.stat().st_mode & 0o777 == 0o640


# --- One verifier-only re-run on the preserved output ---------

TIMED_OUT = "verifier timed out after 5.0s"


@pytest.mark.asyncio
async def test_a_preserved_trials_verifier_timeout_gets_one_verifier_only_rerun(
    trial,
    child,
    monkeypatch,
):
    """Fails on 7a1a4773: a verifier timeout was not recovered at all."""
    from benchflow.rollout import _verifier_recovery as recovery

    trial._verifier_error = TIMED_OUT
    assert recovery.needs_verifier_recovery(TIMED_OUT)
    verify = AsyncMock(return_value=({"reward": 1.0}, None, None))
    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify)
    rewards, error = await recovery.recover_verifier(trial)
    assert (rewards, error) == ({"reward": 1.0}, None)
    child.run.assert_not_awaited()  # the agent never runs again
    assert verify.await_count == 1


@pytest.mark.asyncio
async def test_a_second_verifier_timeout_is_final_in_the_run_and_on_resume(
    trial,
    child,
    monkeypatch,
):
    from benchflow.review.resume import resume_review
    from benchflow.rollout import _verifier_recovery as recovery

    trial._verifier_error = TIMED_OUT
    _save_solver(trial)
    verify = AsyncMock(return_value=(None, TIMED_OUT, None))
    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify)
    rewards, final = await recovery.recover_verifier(trial)
    assert rewards is None
    assert final.startswith(
        "[solver-preserved] verifier recovery failed: verifier timed out"
    )
    assert classify_verifier_error(final) == "verifier_timeout"
    assert not recovery.needs_verifier_recovery(final)
    assert not RetryConfig().should_retry_verifier_error(final)
    assert not RetryConfig().reruns_verifier_error_on_resume(final)
    root = trial._require_rollout_dir()
    # A resume reads the failed re-run and runs nothing more.
    with contextlib.suppress(Exception):
        await resume_review(root, tasks_root=trial._config.task_path)
    assert verify.await_count == 1
    source = recovery.verification_source(root)
    assert source["verifier_error_category"] == "verifier_timeout"
    child.run.assert_not_awaited()


def test_a_timeout_without_a_preserved_workspace_is_never_rerun():
    assert not RetryConfig().should_retry_verifier_error(TIMED_OUT)
    assert not RetryConfig().reruns_verifier_error_on_resume(TIMED_OUT)


# --- L-A2: which disk is full -------------------------------------------------


@pytest.mark.parametrize(
    ("where", "says"),
    [
        (
            "/home/me/.cache/benchflow/datasets-v2/x",
            "the disk that holds /home/me/.cache/benchflow/datasets-v2/x is full",
        ),
        (
            "/tmp/benchflow-stage-1/x",
            "the disk that holds /tmp/benchflow-stage-1/x is full",
        ),
    ],
)
def test_a_full_disk_names_the_file_it_could_not_write(
    where, says, capsys, tmp_path, monkeypatch
):
    """L-A2, fails on 7a1a4773: always "the job folder" and "--jobs-dir"."""
    import typer

    from benchflow.cli._errors import run_cli

    monkeypatch.setenv("BENCHFLOW_LOG_DIR", str(tmp_path / "logs"))
    app = typer.Typer()

    @app.command()
    def go() -> None:
        raise OSError(errno.ENOSPC, "No space left on device", where)

    @app.command()
    def other() -> None:
        pass

    with pytest.raises(SystemExit):
        run_cli(app, ["go"])
    err = " ".join(capsys.readouterr().err.split())
    assert says in err
    assert "--jobs-dir" not in err
    assert "next: free space on that disk" in err


# --- L-A3: a BenchFlow bug's traceback is kept with the trial ------------------


def test_a_trials_unexpected_error_keeps_its_traceback_in_the_trial(tmp_path):
    """L-A3, fails on 7a1a4773: kept nowhere without -v."""
    from benchflow.rollout import write_error_traceback

    try:
        raise ValueError("boom")
    except ValueError as exc:
        write_error_traceback(tmp_path, exc)
    text = (tmp_path / "error-traceback.txt").read_text()
    assert "Traceback (most recent call last)" in text
    assert "ValueError: boom" in text


def test_the_run_failed_line_names_the_saved_traceback():
    source = (
        Path(__file__).parents[1] / "src/benchflow/rollout/__init__.py"
    ).read_text()
    assert "write_error_traceback(" in source


# --- L-A4, L-A5: every Ctrl-C path says how to resume --------------------------


def test_the_sharded_path_says_one_thing_and_how_to_resume():
    source = (Path(__file__).parents[1] / "src/benchflow/cli/main.py").read_text()
    start = source.index("def run_batch_eval(")
    body = source[start : source.index("\ndef ", start + 10)]
    # Both run_until_terminated calls leave the line to _report_stopped.
    assert body.count("announce=False") == 2


def test_run_config_and_results_resume_say_how_to_resume():
    root = Path(__file__).parents[1] / "src/benchflow/cli"
    main = root.joinpath("main.py").read_text()
    start = main.index("def _run_config_file_eval(")
    body = main[start : main.index("\ndef ", start + 10)]
    assert "report_stopped(" in body and "announce=False" in body
    results = root.joinpath("results.py").read_text()
    assert "report_stopped(" in results and "announce=False" in results


_STOP_SCRIPT = """
import asyncio, sys, time
from pathlib import Path
from benchflow.cli._shared import report_stopped
from benchflow.cli._termination import RunStopped, run_until_terminated

job = Path(sys.argv[1])
started = time.time()
(job / "t__1").mkdir(parents=True)
(job / "evaluation.json").write_text("{}")
(job / "t__1" / "config.json").write_text('{"agent": "oracle"}')

async def run():
    print("ready", flush=True)
    await asyncio.sleep(60)

try:
    run_until_terminated(run(), announce=False)
except RunStopped as stop:
    report_stopped(stop, job, since=started)
    raise
"""


def test_report_stopped_counts_only_this_runs_trials(tmp_path):
    """Over-counting (review note): an earlier run's leftover folder is not
    "stopped" again."""
    job = tmp_path / "job"
    old = job / "old__0"
    old.mkdir(parents=True)
    (old / "config.json").write_text('{"agent": "oracle"}')
    os.utime(old / "config.json", (1, 1))
    proc = subprocess.Popen(
        [sys.executable, "-c", _STOP_SCRIPT, str(job)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )
    assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
    proc.send_signal(signal.SIGINT)
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 130, err
    assert "stopped 1 running trial and removed its sandbox" in err
    assert "Resume with: bench eval resume" in err


# --- L-A1: the oracle's budget on a stalled exec ---------------------------------


def test_the_docs_say_a_stalled_exec_at_the_oracle_budget_is_a_timeout():
    docs = (Path(__file__).parents[1] / "docs/jobs/when-a-run-fails.md").read_text()
    assert "stalled" in docs and "oracle's budget" in docs
