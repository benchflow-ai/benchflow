"""``remote-docker`` is wired like the local Docker provider where it can be.

Registry facts, the sandbox factory, runtime capability checks (separate
verifier, compose services, network modes), the pre-job host check and the
reviewer preflight.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox.providers import PROVIDERS_BY_NAME, ModelProxyLocation
from benchflow.sandbox.remote_docker import (
    REMOTE_DOCKER_HOST_ENV,
    RemoteDockerConfigError,
    RemoteDockerSandbox,
)
from tests import _fake_docker_cli

SSH_URL = "ssh://builder@remote.example.test"


@pytest.fixture(autouse=True)
def _clean_docker_env(monkeypatch):
    for name in (REMOTE_DOCKER_HOST_ENV, "DOCKER_HOST", "DOCKER_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("BENCHFLOW_SKIP_PREFLIGHT", raising=False)


def test_registry_entry():
    provider = PROVIDERS_BY_NAME["remote-docker"]
    assert provider.extra is None
    # The remote container cannot reach a proxy on the caller's machine.
    assert provider.model_proxy is ModelProxyLocation.SANDBOX
    assert provider.supports_compose
    assert provider.enforces_no_network
    assert provider.enforces_allowlist
    assert provider.enforces_denylist


def _task(tmp_path: Path, toml: str = "") -> Path:
    task = tmp_path / "remote-task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM alpine:3.20\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "instruction.md").write_text("Say hi.\n")
    (task / "task.toml").write_text('version = "1.0"\n' + toml)
    return task


def test_factory_builds_a_remote_sandbox(tmp_path, monkeypatch):
    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task import Task
    from benchflow.task.paths import RolloutPaths

    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    task_dir = _task(tmp_path)
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    env = _create_sandbox_environment(
        "remote-docker", Task(task_dir), task_dir, "remote-task__1", paths
    )
    assert isinstance(env, RemoteDockerSandbox)
    assert env.docker_host.url == SSH_URL


def test_factory_refuses_without_a_host(tmp_path):
    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task import Task
    from benchflow.task.paths import RolloutPaths

    task_dir = _task(tmp_path)
    with pytest.raises(RemoteDockerConfigError):
        _create_sandbox_environment(
            "remote-docker",
            Task(task_dir),
            task_dir,
            "remote-task__1",
            RolloutPaths(rollout_dir=tmp_path / "run"),
        )


@pytest.mark.parametrize(
    "toml",
    [
        '[verifier]\nsandbox_mode = "separate"\n',
        '[environment]\nnetwork_mode = "no-network"\n',
        '[environment]\nnetwork_mode = "allowlist"\nallowed_hosts = ["example.com"]\n',
    ],
)
def test_runtime_capabilities_accept_what_local_docker_accepts(tmp_path, toml):
    from benchflow.task import Task
    from benchflow.task.runtime_capabilities import raise_for_task_runtime_support

    task_dir = _task(tmp_path, toml)
    task = Task(task_dir)
    raise_for_task_runtime_support(task.config, sandbox="docker", task_dir=task_dir)
    raise_for_task_runtime_support(
        task.config, sandbox="remote-docker", task_dir=task_dir
    )


def test_separate_verifier_and_artifacts_sets_include_remote_docker():
    from benchflow.task.artifacts import ARTIFACT_SANDBOXES
    from benchflow.task.verifier_sandbox import SEPARATE_VERIFIER_SANDBOXES

    assert "remote-docker" in SEPARATE_VERIFIER_SANDBOXES
    assert "remote-docker" in ARTIFACT_SANDBOXES


def test_acp_transport_is_docker_stdio():
    from benchflow.acp.selection import selected_acp_transport

    assert (
        selected_acp_transport(agent="claude-agent-acp", environment="remote-docker")
        == "docker-stdio"
    )


def _config(environment: str) -> SimpleNamespace:
    return SimpleNamespace(
        environment=environment, agent="oracle", model=None, agent_env={}, scenes=[]
    )


def test_pre_job_check_refuses_an_unreachable_host(tmp_path, monkeypatch):
    from benchflow import runtime

    fake = _fake_docker_cli.install(tmp_path, monkeypatch)
    fake.set(fail={"info": "Cannot connect to the Docker daemon at tcp://x:2376."})
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    with pytest.raises(RuntimeError) as info:
        runtime.check_host([_config("remote-docker")])
    message = str(info.value)
    assert "nothing was started" in message
    assert "Remote Docker host unreachable" in message
    assert "builder@" not in message


def test_pre_job_check_refuses_a_missing_host(tmp_path, monkeypatch):
    from benchflow import runtime

    _fake_docker_cli.install(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match=REMOTE_DOCKER_HOST_ENV):
        runtime.check_host([_config("remote-docker")])


def test_pre_job_check_passes_a_reachable_host(tmp_path, monkeypatch):
    from benchflow import runtime

    fake = _fake_docker_cli.install(tmp_path, monkeypatch)
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    runtime.check_host([_config("remote-docker")])
    assert [c["argv"][0] for c in fake.calls()] == ["info"]


def test_reviewer_preflight_checks_the_remote_host(tmp_path, monkeypatch):
    from benchflow.review.preflight import validate_reviewer_backend

    fake = _fake_docker_cli.install(tmp_path, monkeypatch)
    fake.set(fail={"info": "Cannot connect to the Docker daemon at tcp://x:2376."})
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    with pytest.raises(ValueError, match="Remote Docker host unreachable"):
        validate_reviewer_backend(SimpleNamespace(environment="remote-docker"))


def _cli_run(tmp_path: Path):
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    task = _task(tmp_path)
    return CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--agent",
            "oracle",
            "--sandbox",
            "remote-docker",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )


def test_cli_refuses_an_unreachable_host_before_a_job_exists(tmp_path, monkeypatch):
    """Regression test: the CLI used to create the job and then fail the trial."""
    from benchflow.cli import main as cli_main

    calls: list[tuple] = []
    monkeypatch.setattr(cli_main, "run_batch_eval", lambda *a, **k: calls.append(a))
    fake = _fake_docker_cli.install(tmp_path, monkeypatch)
    fake.set(
        fail={
            "info": "error during connect: command [ssh -l builder -- "
            "remote.example.test docker system dial-stdio] has exited with exit "
            "status 255: stderr=builder@remote.example.test: Permission denied "
            "(publickey)."
        }
    )
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    result = _cli_run(tmp_path)
    assert result.exit_code == 1
    assert calls == []
    assert not (tmp_path / "jobs").exists()
    assert "no job was created" in result.stderr
    assert "Remote Docker host unreachable" in result.stderr
    assert "Permission denied (publickey)" in result.stderr
    assert "builder@" not in result.stderr
    assert "Traceback" not in result.output


def test_cli_refuses_a_missing_host_before_a_job_exists(tmp_path, monkeypatch):
    from benchflow.cli import main as cli_main

    calls: list[tuple] = []
    monkeypatch.setattr(cli_main, "run_batch_eval", lambda *a, **k: calls.append(a))
    _fake_docker_cli.install(tmp_path, monkeypatch)
    result = _cli_run(tmp_path)
    assert result.exit_code == 1
    assert calls == []
    assert REMOTE_DOCKER_HOST_ENV in result.stderr
