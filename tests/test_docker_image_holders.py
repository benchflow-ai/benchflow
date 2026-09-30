"""A task image is removed only by the last rollout holding it (sandbox/_image_holders.py).

Two rollouts of one task on one daemon used to race: the first teardown's
``compose down --rmi all`` deleted ``bf__<task>`` while the second was
between build and ``compose up`` (5 lost rollouts in
150 on a shared VM).
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._image_holders import HOLDER_DIR_ENV, ImageHold
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.config import SandboxConfig


@pytest.fixture(autouse=True)
def holder_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(HOLDER_DIR_ENV, str(tmp_path / "holders"))
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    return tmp_path / "holders"


def test_only_the_last_holder_removes_the_image():
    a, b = ImageHold("bf__sql-1"), ImageHold("bf__sql-1")
    a.acquire()
    b.acquire()
    assert a.begin_release() is False
    a.end_release()
    assert b.begin_release() is True
    b.end_release()
    assert [p for p in a.folder.iterdir() if not p.name.startswith(".")] == []


def test_a_holder_of_a_dead_process_does_not_count():
    from benchflow.sandbox.leases import lease_token

    me = ImageHold("bf__sql-1")
    me.acquire()
    host, _pid, boot, start = lease_token().split(":")
    dead = me.folder / f"{host}~99999999~{boot}~{start}~1"  # no such pid
    dead.write_text("bf__sql-1")
    assert me.begin_release() is True
    me.end_release()
    assert not dead.exists()


def test_a_live_holder_of_another_machine_is_kept():
    me = ImageHold("bf__sql-1")
    me.acquire()
    other = me.folder / "otherhost~123~abcd~42~1"  # state "unknown": counts
    other.write_text("bf__sql-1")
    assert me.begin_release() is False
    me.end_release()
    assert other.exists()


def test_images_and_daemons_are_held_apart(monkeypatch):
    a, b = ImageHold("bf__sql-1"), ImageHold("bf__sql-2")
    assert a.folder != b.folder
    monkeypatch.setenv("DOCKER_HOST", "tcp://10.0.0.2:2375")
    assert ImageHold("bf__sql-1").folder != a.folder


def test_a_new_holder_waits_for_an_image_removal_in_progress():
    last = ImageHold("bf__sql-1")
    last.acquire()
    assert last.begin_release() is True  # lock held while the image is removed
    newcomer = ImageHold("bf__sql-1")
    done = threading.Event()
    thread = threading.Thread(target=lambda: (newcomer.acquire(), done.set()))
    thread.start()
    time.sleep(0.3)
    assert not done.is_set(), "registered while the image was being removed"
    last.end_release()
    thread.join(timeout=10)
    assert done.is_set() and newcomer.path.exists()


def _sandbox(tmp_path, name: str) -> DockerSandbox:
    task = tmp_path / "task"
    task.mkdir(exist_ok=True)
    (task / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    sandbox = DockerSandbox(task, "sql-1", name, None, SandboxConfig())
    sandbox._run_docker_compose_command = AsyncMock(
        return_value=SimpleNamespace(stdout="", stderr="", return_code=0)
    )
    sandbox._run_docker_compose_build = AsyncMock()
    sandbox._run_docker_compose_up = AsyncMock()
    sandbox._run_pre_compose_hook = AsyncMock()
    sandbox._prepare_log_dirs_after_up = AsyncMock()
    sandbox._chown_to_host_user = AsyncMock()
    return sandbox


def _removed_images(sandbox: DockerSandbox) -> bool:
    return any(
        "--rmi" in call.args[0]
        for call in sandbox._run_docker_compose_command.await_args_list
    )


@pytest.mark.asyncio
async def test_two_rollouts_of_one_task_remove_the_image_once(tmp_path):
    first, second = _sandbox(tmp_path, "r0"), _sandbox(tmp_path, "r1")
    await first.start(force_build=False)
    await second.start(force_build=False)
    await first.stop(delete=True)
    assert not _removed_images(first), "removed the image a sibling still uses"
    await second.stop(delete=True)
    assert _removed_images(second)
    assert first._image_hold is None and second._image_hold is None
