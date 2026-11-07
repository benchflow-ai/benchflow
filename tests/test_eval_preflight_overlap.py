"""`bench eval run`'s Docker preflight runs while the engine imports.

The preflight's Docker check (``docker context show``, ``docker info``) used
to run after the engine's imports, and also ran ``docker buildx version``,
whose result can only warn and which the preflight drops (it keeps failures).
It now starts in a thread before the imports and runs no buildx check; its
verdict stops a run exactly as before (tests/test_eval_preflight.py).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow import doctor as doctor_mod
from benchflow.cli import main as cli_main
from benchflow.cli.main import app
from benchflow.doctor import Check, CommandResult, DoctorProbes

_INFO = json.dumps(
    {
        "ServerVersion": "29.1.3",
        "NCPU": 32,
        "MemTotal": 64 * 2**30,
        "DockerRootDir": "/var/lib/docker",
        "OperatingSystem": "Ubuntu 24.04",
    }
)


@pytest.fixture
def preflight_on(monkeypatch):
    monkeypatch.delenv("BENCHFLOW_SKIP_PREFLIGHT", raising=False)
    monkeypatch.delenv("DOCKER_HOST", raising=False)


@pytest.fixture
def batch_calls(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        cli_main, "run_batch_eval", lambda *args, **kwargs: calls.append(args)
    )
    return calls


@pytest.fixture
def probe_runs(monkeypatch, tmp_path):
    """Doctor probes on a fake host; records each command and its thread."""
    runs: list[tuple[tuple[str, ...], str]] = []

    def run(argv, timeout):
        runs.append((tuple(argv), threading.current_thread().name))
        if tuple(argv[:2]) == ("docker", "info"):
            return CommandResult(0, _INFO, "")
        if tuple(argv[:3]) == ("docker", "context", "show"):
            return CommandResult(0, "default\n", "")
        return CommandResult(0, "", "")

    def from_host(cls):
        return DoctorProbes(
            environ={},
            origins={},
            home=tmp_path,
            which=lambda name: f"/usr/bin/{name}",
            run=run,
            disk_free=lambda path: 100 * 2**30,
            system="Linux",
        )

    monkeypatch.setattr(DoctorProbes, "from_host", classmethod(from_host))
    return runs


def _run(tmp_path: Path, *extra: str):
    task = tmp_path / "task"
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
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


def test_the_docker_check_runs_in_its_own_thread_without_buildx(
    tmp_path, preflight_on, batch_calls, probe_runs
):
    result = _run(tmp_path, "--sandbox", "docker")

    assert result.exit_code == 0, result.output
    assert len(batch_calls) == 1
    commands = [argv for argv, _ in probe_runs]
    assert ("docker", "info", "--format", "{{json .}}") in commands
    assert not any(argv[:2] == ("docker", "buildx") for argv in commands)
    assert {thread for _, thread in probe_runs} == {"bench-docker-preflight"}


def test_the_default_sandbox_is_checked_in_its_thread(
    tmp_path, preflight_on, batch_calls, probe_runs
):
    result = _run(tmp_path)

    assert result.exit_code == 0, result.output
    assert {thread for _, thread in probe_runs} == {"bench-docker-preflight"}


def test_a_failed_check_from_the_thread_stops_the_run(
    tmp_path, monkeypatch, preflight_on, batch_calls
):
    seen: list[str] = []

    def down(probes, *, required, buildx=True):
        seen.append(threading.current_thread().name)
        return [Check("docker", "sandbox", "docker", "fail", "daemon unreachable")]

    monkeypatch.setattr(doctor_mod, "check_docker", down)

    result = _run(tmp_path)

    assert result.exit_code == 1
    assert seen == ["bench-docker-preflight"]
    assert batch_calls == []
    assert not (tmp_path / "jobs").exists()
    assert "daemon unreachable" in result.stderr


def test_an_error_in_the_thread_reaches_the_command(
    tmp_path, monkeypatch, preflight_on, batch_calls
):
    def broken(probes, *, required, buildx=True):
        raise RuntimeError("doctor broke")

    monkeypatch.setattr(doctor_mod, "check_docker", broken)

    result = _run(tmp_path)

    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "doctor broke"
    assert batch_calls == []


def test_a_remote_docker_host_is_checked_inline(
    tmp_path, monkeypatch, preflight_on, batch_calls
):
    """DOCKER_HOST's check imports the remote sandbox module: never in a thread."""
    seen: list[str] = []

    def up(probes, *, required, buildx=True):
        seen.append(threading.current_thread().name)
        return [Check("docker", "sandbox", "docker", "pass", "Docker ready")]

    monkeypatch.setattr(doctor_mod, "check_docker", up)
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/other.sock")

    result = _run(tmp_path)

    assert result.exit_code == 0, result.output
    assert seen == [threading.main_thread().name]


@pytest.mark.parametrize(
    ("extra", "env"),
    [
        (("--sandbox", "daytona"), {}),
        (("--dry-run",), {}),
        ((), {"BENCHFLOW_SKIP_PREFLIGHT": "1"}),
    ],
    ids=["daytona", "dry-run", "opted-out"],
)
def test_no_docker_check_starts_when_the_run_makes_none(
    tmp_path, monkeypatch, preflight_on, batch_calls, extra, env
):
    seen: list[str] = []

    def up(probes, *, required, buildx=True):
        seen.append(threading.current_thread().name)
        return [Check("docker", "sandbox", "docker", "pass", "Docker ready")]

    monkeypatch.setattr(doctor_mod, "check_docker", up)
    monkeypatch.setattr(
        doctor_mod,
        "check_daytona",
        lambda probes, *, required, offline, for_run=False: Check(
            "daytona", "sandbox", "daytona", "ok", "stubbed"
        ),
    )
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    _run(tmp_path, *extra)

    assert seen == []
