"""Each DockerSandbox owns its Compose project, labelled with its process.

Guards the fix for concurrent Docker runs sharing Compose projects: the
project name was the rollout name, and branch children (``n<k>``), regrade
(``verifier``) and SDK callers fix that name, so two
such sandboxes on one daemon shared a project and each one's ``compose down``
deleted the other's containers. It also guards the claim that keeps the
leftover sweep (tests/test_docker_sweep.py) off a live sandbox's resources.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._docker_sweep import (
    PROCESS_ENV,
    live_projects,
    process_token,
    release_project,
)
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.process.docker import DockerProcess
from benchflow.sandbox.protocol import ExecResult
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths


def _sandbox(tmp_path: Path, session_id: str, **kwargs) -> DockerSandbox:
    environment = tmp_path / session_id / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    paths = RolloutPaths(rollout_dir=tmp_path / session_id / "run")
    paths.mkdir()
    return DockerSandbox(
        environment_dir=environment,
        environment_name="task",
        session_id=session_id,
        rollout_paths=paths,
        task_env_config=SandboxConfig(),
        **kwargs,
    )


def test_two_sandboxes_with_one_session_id_get_their_own_projects(tmp_path):
    """Two branch children named ``n3`` (two jobs, or two processes) must not
    share a project: ``start()`` runs ``compose down --remove-orphans`` for it."""
    first = _sandbox(tmp_path / "a", "n3")
    second = _sandbox(tmp_path / "b", "n3")
    assert first.compose_project_name != second.compose_project_name
    for sandbox in (first, second):
        assert sandbox.compose_project_name.startswith("n3-")
        assert sandbox.compose_project_name == sandbox.compose_project_name


async def test_every_compose_call_and_the_live_process_name_the_one_project(
    tmp_path, monkeypatch
):
    sandbox = _sandbox(tmp_path, "Task.Name__1234")
    seen: list[list[str]] = []

    async def fake_exec(*argv, **kwargs):
        seen.append(list(argv))
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        return proc

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    await sandbox._run_docker_compose_command(["ps"])
    await sandbox._run_docker_compose_command(["ps"])
    projects = {argv[argv.index("--project-name") + 1] for argv in seen}
    assert projects == {sandbox.compose_project_name}
    assert DockerProcess.from_sandbox_env(sandbox)._project_name == (
        sandbox.compose_project_name
    )


def test_compose_env_names_this_process_even_over_a_task_variable(tmp_path):
    """The ``benchflow.process`` label decides whether the sweep may remove a
    container, so no task environment variable may set it."""
    sandbox = _sandbox(tmp_path, "t__1", persistent_env={PROCESS_ENV: "forged:1::"})
    assert sandbox._docker_compose_env()[PROCESS_ENV] == process_token()


@pytest.mark.parametrize("keep", [False, True])
async def test_start_claims_the_project_and_stop_releases_it(tmp_path, keep):
    sandbox = _sandbox(tmp_path, f"t__{int(keep)}", keep_containers=keep)
    project = sandbox.compose_project_name
    during: list[bool] = []

    async def compose(command, check=True, timeout_sec=None):
        during.append(project in live_projects())
        return ExecResult(stdout="", stderr="", return_code=0)

    sandbox._run_docker_compose_command = compose
    sandbox._prepare_log_dirs_after_up = AsyncMock()
    sandbox._chown_to_host_user = AsyncMock()
    assert project not in live_projects()
    await sandbox.start(force_build=False)
    assert during and all(during)
    assert project in live_projects()
    await sandbox.stop(delete=True)
    # Kept containers stay claimed: the sweep leaves them for the user.
    assert (project in live_projects()) is keep
    release_project(project)
