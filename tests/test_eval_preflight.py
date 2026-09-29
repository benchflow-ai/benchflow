"""`bench eval run` checks the sandbox before it creates a job.

Regression test: with the Docker daemon
stopped, `bench eval run --sandbox docker` printed a traceback embedding the
whole `docker compose` command and published a 0/1 job, although
`bench doctor` already reported the daemon clearly. The preflight reuses
doctor's `check_docker`; the suite opts out through BENCHFLOW_SKIP_PREFLIGHT
(tests/conftest.py), so these tests remove that variable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow import doctor as doctor_mod
from benchflow.cli import main as cli_main
from benchflow.cli.main import app
from benchflow.doctor import Check


@pytest.fixture(autouse=True)
def _daytona_key_ok(monkeypatch):
    """These tests are about other checks; the live Daytona key check passes."""
    monkeypatch.setattr(
        doctor_mod,
        "check_daytona",
        lambda probes, *, required, offline: Check(
            "daytona", "sandbox", "daytona", "ok", "stubbed"
        ),
    )


def _write_task(task_dir: Path) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    return task_dir


_DAEMON_DOWN = Check(
    "docker",
    "sandbox",
    "docker",
    "fail",
    "daemon unreachable (context DOCKER_HOST=unix:///tmp/none.sock): "
    "failed to connect to the docker API",
    "Start Docker Desktop / OrbStack, or `colima start`",
)
_DAEMON_UP = Check("docker", "sandbox", "docker", "pass", "Docker 29.5.2 via colima")


@pytest.fixture
def preflight_on(monkeypatch):
    monkeypatch.delenv("BENCHFLOW_SKIP_PREFLIGHT", raising=False)


@pytest.fixture
def batch_calls(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        cli_main, "run_batch_eval", lambda *args, **kwargs: calls.append(args)
    )
    return calls


def _docker_checks(monkeypatch, checks: list[Check]) -> list[bool]:
    seen: list[bool] = []

    def fake_check_docker(probes, *, required):
        seen.append(required)
        return checks

    monkeypatch.setattr(doctor_mod, "check_docker", fake_check_docker)
    return seen


def _run(tmp_path: Path, *extra: str):
    task = _write_task(tmp_path / "task")
    return CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--agent",
            "oracle",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            *extra,
        ],
    )


def test_unreachable_docker_daemon_stops_the_run_before_a_job_exists(
    tmp_path, monkeypatch, preflight_on, batch_calls
):
    seen = _docker_checks(monkeypatch, [_DAEMON_DOWN])

    result = _run(tmp_path, "--sandbox", "docker")

    assert result.exit_code == 1
    assert seen == [True]
    assert batch_calls == []
    assert not (tmp_path / "jobs").exists()
    assert "no job was created" in result.stderr
    assert "daemon unreachable" in result.stderr
    assert "colima start" in result.stderr
    assert "bench doctor" in result.stderr
    assert "BENCHFLOW_SKIP_PREFLIGHT=1" in result.stderr
    assert "Traceback" not in result.output


def test_default_sandbox_is_docker_and_is_checked(
    tmp_path, monkeypatch, preflight_on, batch_calls
):
    seen = _docker_checks(monkeypatch, [_DAEMON_DOWN])

    result = _run(tmp_path)

    assert result.exit_code == 1
    assert seen == [True]
    assert batch_calls == []


def test_ready_docker_lets_the_run_start(
    tmp_path, monkeypatch, preflight_on, batch_calls
):
    _docker_checks(monkeypatch, [_DAEMON_UP])

    result = _run(tmp_path, "--sandbox", "docker")

    assert result.exit_code == 0, result.output
    assert len(batch_calls) == 1


def test_opt_out_variable_skips_the_check(tmp_path, monkeypatch, batch_calls):
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    seen = _docker_checks(monkeypatch, [_DAEMON_DOWN])

    result = _run(tmp_path, "--sandbox", "docker")

    assert result.exit_code == 0, result.output
    assert seen == []
    assert len(batch_calls) == 1


def test_other_sandboxes_do_not_probe_docker(
    tmp_path, monkeypatch, preflight_on, batch_calls
):
    seen = _docker_checks(monkeypatch, [_DAEMON_DOWN])
    monkeypatch.setattr(
        "benchflow.eval_plan.sandbox_sdk_missing", lambda _environment: False
    )

    result = _run(tmp_path, "--sandbox", "daytona")

    assert result.exit_code == 0, result.output
    assert seen == []
    assert len(batch_calls) == 1


def test_run_config_file_is_checked_against_its_own_sandbox(
    tmp_path, monkeypatch, preflight_on
):
    seen = _docker_checks(monkeypatch, [_DAEMON_DOWN])
    ran: list[bool] = []

    async def fake_run(self):
        ran.append(True)

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_run)
    task = _write_task(tmp_path / "tasks" / "task")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"tasks_dir: {task.parent}\nagent: oracle\nenvironment: docker\n"
        f"jobs_dir: {tmp_path / 'jobs'}\n"
    )

    result = CliRunner().invoke(app, ["eval", "run", "--config", str(config)])

    assert result.exit_code == 1
    assert seen == [True]
    assert ran == []
    assert "no job was created" in result.stderr


def test_misspelt_agent_stops_the_run_before_a_job_exists(tmp_path, batch_calls):
    """`bench eval run --agent
    claude-agnet-acp --sandbox daytona` only logged "Will attempt to use as raw
    command", started a Daytona sandbox and failed there, although bf.run and
    Evaluation.run refuse the same name. The name check runs even with
    BENCHFLOW_SKIP_PREFLIGHT=1 (set by the suite), as in the SDK."""
    task = _write_task(tmp_path / "task")
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--agent",
            "claude-agnet-acp",
            "--sandbox",
            "daytona",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )

    assert result.exit_code == 1
    assert batch_calls == []
    assert not (tmp_path / "jobs").exists()
    assert "did you mean 'claude-agent-acp'" in result.stderr
