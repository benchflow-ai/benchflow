"""A task image is removed only by the last rollout holding it (sandbox/_image_holders.py).

Two rollouts of one task on one daemon used to race: the first teardown's
``compose down --rmi all`` deleted ``bf__<task>`` while the second was
between build and ``compose up`` (5 lost rollouts in
150 on a shared VM).
"""

from __future__ import annotations

import asyncio
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
    assert a.try_acquire() and b.try_acquire()
    assert a.begin_release() is False
    a.end_release()
    assert b.begin_release() is True
    b.end_release()
    assert [p for p in a.folder.iterdir() if not p.name.startswith(".")] == []


def test_a_holder_of_a_dead_process_does_not_count():
    from benchflow.sandbox.leases import lease_token

    me = ImageHold("bf__sql-1")
    assert me.try_acquire()
    host, _pid, boot, start = lease_token().split(":")
    dead = me.folder / f"{host}~99999999~{boot}~{start}~1"  # no such pid
    dead.write_text("bf__sql-1")
    assert me.begin_release() is True
    me.end_release()
    assert not dead.exists()


def test_a_live_holder_of_another_machine_is_kept():
    me = ImageHold("bf__sql-1")
    assert me.try_acquire()
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


async def test_a_new_holder_waits_for_an_image_removal_in_progress():
    last = ImageHold("bf__sql-1")
    assert last.try_acquire()
    assert last.begin_release() is True  # marks the removal
    newcomer = ImageHold("bf__sql-1")
    assert newcomer.try_acquire() is False
    waiting = asyncio.create_task(newcomer.acquire())
    await asyncio.sleep(1.5)
    assert not waiting.done(), "registered while the image was being removed"
    # Nobody else may start a second removal meanwhile.
    other = ImageHold("bf__sql-1")
    assert other.begin_release() is False
    last.end_release()
    await asyncio.wait_for(waiting, 5)
    assert newcomer.path.exists()


def test_a_removal_marker_of_a_dead_process_is_ignored():
    from benchflow.sandbox.leases import lease_token

    hold = ImageHold("bf__sql-1")
    host, _pid, boot, start = lease_token().split(":")
    hold.folder.mkdir(parents=True)
    (hold.folder / ".removing").write_text(f"{host}:99999999:{boot}:{start}\nx")
    assert hold.try_acquire()


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


@pytest.mark.asyncio
async def test_concurrent_teardowns_remove_the_image_exactly_once(tmp_path):
    sandboxes = [_sandbox(tmp_path, f"r{i}") for i in range(4)]
    await asyncio.gather(*(sb.start(force_build=False) for sb in sandboxes))
    await asyncio.gather(*(sb.stop(delete=True) for sb in sandboxes))
    assert sum(_removed_images(sb) for sb in sandboxes) == 1


@pytest.mark.asyncio
async def test_a_start_cancelled_while_waiting_leaves_no_holder(tmp_path):
    remover = ImageHold("bf__sql-1")
    assert remover.try_acquire() and remover.begin_release()
    sandbox = _sandbox(tmp_path, "r0")
    starting = asyncio.create_task(sandbox.start(force_build=False))
    await asyncio.sleep(1.5)
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    await sandbox.stop(delete=True)
    remover.end_release()
    assert [p for p in remover.folder.iterdir() if not p.name.startswith(".")] == []
    assert not _removed_images(sandbox), "removed an image during another removal"


@pytest.mark.asyncio
async def test_a_failed_start_does_not_remove_a_siblings_image(tmp_path):
    sibling = _sandbox(tmp_path, "r0")
    await sibling.start(force_build=False)
    # Its start failed before registering a holder (for example in the
    # recovery-baseline check), so teardown finds no hold of its own.
    failed = _sandbox(tmp_path, "r1")
    await failed.stop(delete=True)
    assert not _removed_images(failed)
    await sibling.stop(delete=True)
    assert _removed_images(sibling)
