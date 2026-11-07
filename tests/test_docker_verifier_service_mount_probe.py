"""No false bind-mount warning for a task whose verifier runs in another service (REPORT B14).

robohub's tasks name ``verifier.service: simulator`` and drop BenchFlow's
verifier-log mount from ``main`` with Compose's ``!override``, so the mount
probe never sees it there. That is the task's design, not a remote daemon, and
the "check DOCKER_HOST" warning printed on every robohub trial was false.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import docker as docker_module
from benchflow.sandbox import setup as sandbox_setup
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.setup import _create_sandbox_environment
from benchflow.task import RolloutPaths, Task


@pytest.fixture(autouse=True)
def isolate_mount_probe(monkeypatch):
    """Guards the 85b87bc9 merge: no daemon is used by this output-policy fake."""
    monkeypatch.setattr(sandbox_setup, "_freeze", lambda sandbox: None)


def _task(root: Path, verifier_service: str | None) -> Path:
    task_dir = root / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Do it.\n")
    service = f'service = "{verifier_service}"\n' if verifier_service else ""
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n'
        "[agent]\ntimeout_sec = 1\n"
        f"[verifier]\ntimeout_sec = 1\n{service}"
    )
    environment = task_dir / "environment"
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (environment / "docker-compose.yaml").write_text(
        "services:\n  main: {}\n  simulator:\n    image: ubuntu:24.04\n"
    )
    return task_dir


async def _probe_unmounted(tmp_path: Path, verifier_service: str | None, caplog):
    task_dir = _task(tmp_path, verifier_service)
    rollout_paths = RolloutPaths(rollout_dir=tmp_path / "jobs" / "rollout")
    rollout_paths.mkdir()
    sandbox = _create_sandbox_environment(
        "docker",
        Task(task_dir),
        task_dir,
        "rollout",
        rollout_paths,
    )
    # The daemon does not show the probe file in main.
    sandbox.exec = AsyncMock(
        return_value=ExecResult(stdout="", stderr="", return_code=1)
    )
    with caplog.at_level(logging.DEBUG, logger="benchflow"):
        await sandbox._probe_verifier_log_mount()
    return sandbox, [r for r in caplog.records if "copied back" in r.message]


@pytest.mark.asyncio
async def test_a_task_naming_verifier_service_gets_no_mount_warning(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(docker_module, "_DOCKER_RUNS_IN_VM", False)
    sandbox, logs = await _probe_unmounted(tmp_path, "simulator", caplog)

    # Still copied back from the container: the mount is really absent.
    assert sandbox.is_mounted is False
    assert [r.levelno for r in logs] == [logging.DEBUG]
    assert "simulator" in logs[0].getMessage()
    assert not any("DOCKER_HOST" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_main_verifier_task_still_warns(tmp_path, monkeypatch, caplog):
    """The warning stays where it means something: the verifier runs in main."""
    monkeypatch.setattr(docker_module, "_DOCKER_RUNS_IN_VM", False)
    sandbox, logs = await _probe_unmounted(tmp_path, None, caplog)

    assert sandbox.is_mounted is False
    assert [r.levelno for r in logs] == [logging.WARNING]
    assert "DOCKER_HOST" in logs[0].getMessage()
