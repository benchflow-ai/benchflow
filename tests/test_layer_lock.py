"""The host lock of a cached layer key (sandbox/_layer_lock.py, PY8 U13).

The lock file is checked and taken here directly: its folder and file rules,
the non-blocking ``flock`` with backoff, cancellation, a killed holder, the
per-daemon names and the per-event-loop process locks.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchflow.sandbox import _layer_lock
from benchflow.sandbox._layer_lock import (
    HostLock,
    LockTimeout,
    LockUnavailable,
    lock_dir,
    lock_name,
    process_lock,
)

KEY = "a" * 64
OTHER_KEY = "b" * 64
DAEMON = "daemon-1"


@pytest.fixture
def locks(tmp_path_factory, monkeypatch) -> Path:
    folder = tmp_path_factory.mktemp("held") / "layer-locks"
    monkeypatch.setenv(_layer_lock.LOCK_DIR_ENV, str(folder))
    return folder


def _open_fds() -> set[str]:
    return set(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else set()


class TestNames:
    def test_the_default_folder_is_per_user_in_the_temp_dir(self, monkeypatch):
        monkeypatch.delenv(_layer_lock.LOCK_DIR_ENV, raising=False)
        assert lock_dir().name == f"benchflow-layer-locks-{os.geteuid()}"

    def test_one_file_per_daemon_kind_and_full_key(self):
        name = lock_name(DAEMON, "agent", KEY)
        assert name.endswith(f"-agent-{KEY}.lock") and "/" not in name
        assert lock_name("daemon-2", "agent", KEY) != name
        assert lock_name(DAEMON, "proxy", KEY) != name
        assert lock_name(DAEMON, "agent", OTHER_KEY) != name

    @pytest.mark.parametrize("key", ["../x", "a/b", ""])
    def test_a_key_that_is_no_name_is_refused(self, key):
        with pytest.raises(ValueError):
            lock_name(DAEMON, "agent", key)


class TestTakeAndRelease:
    async def test_the_lock_is_taken_waited_for_and_released(self, locks):
        holder = HostLock(DAEMON, "agent", KEY)
        assert holder.try_acquire()
        waiter = HostLock(DAEMON, "agent", KEY)
        assert not waiter.try_acquire()
        task = asyncio.create_task(waiter.acquire(time.monotonic() + 10))
        await asyncio.sleep(0.2)
        assert not task.done()
        holder.release()
        waited = await asyncio.wait_for(task, 5)
        assert waited > 0 and waiter.held
        waiter.release()
        # The file stays (a stable inode for the next holder).
        assert (locks / lock_name(DAEMON, "agent", KEY)).is_file()

    async def test_past_the_deadline_it_times_out_and_closes_its_file(self, locks):
        holder = HostLock(DAEMON, "agent", KEY)
        assert holder.try_acquire()
        before = _open_fds()
        waiter = HostLock(DAEMON, "agent", KEY)
        with pytest.raises(LockTimeout):
            await waiter.acquire(time.monotonic() + 0.2)
        assert not waiter.held
        assert _open_fds() == before
        holder.release()

    async def test_a_cancelled_waiter_leaves_the_lock_free_and_no_fd_open(self, locks):
        holder = HostLock(DAEMON, "agent", KEY)
        assert holder.try_acquire()
        before = _open_fds()
        waiter = HostLock(DAEMON, "agent", KEY)
        task = asyncio.create_task(waiter.acquire(time.monotonic() + 30))
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not waiter.held
        assert _open_fds() == before
        holder.release()
        # Free: anyone takes it at once.
        again = HostLock(DAEMON, "agent", KEY)
        assert again.try_acquire()
        again.release()

    async def test_different_daemons_do_not_wait_on_each_other(self, locks):
        one = HostLock("daemon-1", "agent", KEY)
        two = HostLock("daemon-2", "agent", KEY)
        assert one.try_acquire()
        assert two.try_acquire()
        one.release()
        two.release()

    async def test_two_endpoints_of_one_daemon_share_its_lock(self, locks):
        # DOCKER_HOST=unix:///var/run/docker.sock and the default context both
        # report the same `docker info` ID: the lock is named by that ID.
        one = HostLock("same-daemon-id", "proxy", KEY)
        two = HostLock("same-daemon-id", "proxy", KEY)
        assert one.try_acquire()
        assert not two.try_acquire()
        one.release()

    def test_a_killed_holder_frees_the_lock_within_one_backoff(self, locks):
        """SIGKILL: the kernel drops the holder's flock; the waiter needs no
        file removal, and takes the lock within the next poll (1 s at most)
        plus a scheduling allowance measured on this machine."""
        locks.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = locks / lock_name(DAEMON, "agent", KEY)
        script = (
            "import fcntl, os, sys, time\n"
            "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "print('held', flush=True)\n"
            "time.sleep(60)\n"
        )
        holder = subprocess.Popen(
            [sys.executable, "-c", script, str(path)],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == "held"
            inode = path.stat().st_ino

            async def wait_for_it() -> tuple[float, float]:
                waiter = HostLock(DAEMON, "agent", KEY)
                task = asyncio.create_task(waiter.acquire(time.monotonic() + 30))
                # Let the waiter back off to its longest poll first.
                await asyncio.sleep(2.0)
                assert not task.done()
                # Scheduling allowance: how late a 0.05 s sleep wakes here.
                t0 = time.monotonic()
                await asyncio.sleep(0.05)
                allowance = time.monotonic() - t0
                killed = time.monotonic()
                os.kill(holder.pid, signal.SIGKILL)
                await task
                took = time.monotonic() - killed
                waiter.release()
                return took, allowance

            took, allowance = asyncio.run(wait_for_it())
            assert took <= _layer_lock.POLL_MAX_SEC + 0.5 + 10 * allowance, took
            assert path.stat().st_ino == inode
        finally:
            holder.kill()
            holder.wait()


class TestUnusableFolder:
    @pytest.mark.parametrize("mode", [0o755, 0o775, 0o1777])
    def test_existing_shared_directory_is_never_chmodded(self, tmp_path, mode):
        """Guard 399d398a: matching UID is not permission to chmod a folder."""
        shared = tmp_path / "shared-owner-folder"
        shared.mkdir()
        shared.chmod(mode)
        lock = HostLock(DAEMON, "agent", KEY, directory=shared)
        try:
            with pytest.raises(LockUnavailable, match="private"):
                lock.try_acquire()
        finally:
            lock.release()
        assert shared.stat().st_mode & 0o7777 == mode
        assert list(shared.iterdir()) == []

    def test_a_file_where_the_folder_should_be(self, tmp_path, monkeypatch):
        target = tmp_path / "not-a-folder"
        target.write_text("x")
        monkeypatch.setenv(_layer_lock.LOCK_DIR_ENV, str(target))
        with pytest.raises(LockUnavailable, match="not a folder"):
            HostLock(DAEMON, "agent", KEY).try_acquire()

    def test_a_planted_symlink_for_the_folder(self, tmp_path, monkeypatch):
        real = tmp_path / "elsewhere"
        real.mkdir(mode=0o700)
        link = tmp_path / "layer-locks"
        link.symlink_to(real)
        monkeypatch.setenv(_layer_lock.LOCK_DIR_ENV, str(link))
        with pytest.raises(LockUnavailable, match="symlink"):
            HostLock(DAEMON, "agent", KEY).try_acquire()
        assert list(real.iterdir()) == []

    def test_a_symlinked_parent(self, tmp_path, monkeypatch):
        real = tmp_path / "real-parent"
        real.mkdir(mode=0o700)
        (tmp_path / "parent").symlink_to(real)
        monkeypatch.setenv(_layer_lock.LOCK_DIR_ENV, str(tmp_path / "parent" / "locks"))
        with pytest.raises(LockUnavailable, match="symlink"):
            HostLock(DAEMON, "agent", KEY).try_acquire()
        assert list(real.iterdir()) == []

    def test_a_folder_of_another_user(self, locks, monkeypatch):
        locks.mkdir(mode=0o700, parents=True)
        uid = os.geteuid()
        monkeypatch.setattr(_layer_lock, "_uid", lambda: uid + 1)
        with pytest.raises(LockUnavailable, match="belongs to uid"):
            HostLock(DAEMON, "agent", KEY).try_acquire()

    def test_a_read_only_file_system(self, locks, monkeypatch):
        def read_only(*_args, **_kwargs):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr(_layer_lock.os, "mkdir", read_only)
        monkeypatch.setattr(Path, "mkdir", read_only)
        with pytest.raises(LockUnavailable, match="Read-only"):
            HostLock(DAEMON, "agent", KEY).try_acquire()

    def test_a_folder_others_can_write_without_sticky_bit(self, tmp_path, monkeypatch):
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(0o777)
        monkeypatch.setenv(_layer_lock.LOCK_DIR_ENV, str(shared / "locks"))
        with pytest.raises(LockUnavailable, match="not sticky"):
            HostLock(DAEMON, "agent", KEY).try_acquire()

    def test_no_fcntl(self, locks, monkeypatch):
        monkeypatch.setattr(_layer_lock, "fcntl", None)
        with pytest.raises(LockUnavailable, match="fcntl"):
            HostLock(DAEMON, "agent", KEY).try_acquire()


class TestUnusableFile:
    def _path(self, locks: Path) -> Path:
        locks.mkdir(mode=0o700, parents=True, exist_ok=True)
        return locks / lock_name(DAEMON, "agent", KEY)

    def test_a_symlinked_lock_file_is_not_followed(self, locks, tmp_path):
        victim = tmp_path / "victim-a"
        victim.write_text("keep me")
        self._path(locks).symlink_to(victim)
        with pytest.raises(LockUnavailable):
            HostLock(DAEMON, "agent", KEY).try_acquire()
        assert victim.read_text() == "keep me"

    def test_a_lock_file_with_a_second_link(self, locks, tmp_path):
        path = self._path(locks)
        path.write_text("")
        os.link(path, locks / "second-name")
        with pytest.raises(LockUnavailable, match="2 links"):
            HostLock(DAEMON, "agent", KEY).try_acquire()

    def test_a_lock_file_that_is_not_regular(self, locks):
        os.mkfifo(self._path(locks))
        with pytest.raises(LockUnavailable, match="not a regular file"):
            HostLock(DAEMON, "agent", KEY).try_acquire()

    def test_a_lock_file_of_another_user(self, locks, monkeypatch):
        self._path(locks).write_text("")
        real_fstat = os.fstat
        uid = os.geteuid()

        class _Stat:
            def __init__(self, st):
                self._st = st

            def __getattr__(self, name):
                if name == "st_uid":
                    return uid + 1
                return getattr(self._st, name)

        def fstat(fd):
            st = real_fstat(fd)
            # The folder is ours; the file another user's.
            return _Stat(st) if not os.path.isdir(f"/proc/self/fd/{fd}") else st

        monkeypatch.setattr(_layer_lock.os, "fstat", fstat)
        with pytest.raises(LockUnavailable, match="belongs to uid"):
            HostLock(DAEMON, "agent", KEY).try_acquire()

    def test_the_lock_file_is_never_truncated(self, locks):
        path = self._path(locks)
        path.write_text("written by an earlier holder")
        lock = HostLock(DAEMON, "agent", KEY)
        assert lock.try_acquire()
        lock.release()
        assert path.read_text() == "written by an earlier holder"


class TestProcessLocks:
    def test_repeated_event_loops_get_their_own_locks(self):
        async def get() -> asyncio.Lock:
            return process_lock(DAEMON, "agent", KEY)

        async def held_across_loops() -> None:
            lock = process_lock(DAEMON, "agent", KEY)
            async with lock:
                await asyncio.sleep(0)

        first = asyncio.run(get())
        second = asyncio.run(get())
        assert first is not second
        # A lock made by one loop is never waited on by another.
        asyncio.run(held_across_loops())
        asyncio.run(held_across_loops())

    async def test_one_loop_shares_one_lock_per_key(self):
        assert process_lock(DAEMON, "agent", KEY) is process_lock(DAEMON, "agent", KEY)
        assert process_lock(DAEMON, "agent", KEY) is not process_lock(
            "daemon-2", "agent", KEY
        )
