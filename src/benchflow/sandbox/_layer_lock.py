"""One build per cached layer key among BenchFlow processes on one host.

:class:`~benchflow.sandbox.env_layers.DockerLayers` builds a layer the first
time a key misses. Its old lock was an ``asyncio.Lock`` per process, so two
``bench`` processes (a sharded run, two jobs on one VM) that missed the same
key at the same moment both built it, and a prune in one process could remove
a layer another process had just found and was about to tag. This module adds
the host half: an advisory ``flock`` per (Docker daemon, layer kind, full
key), taken after the process lock and held across the hit check, the build
and the trial's tag, and try-locked by the prune before it removes a key's
image or build records.

Where the lock files live: ``$BENCHFLOW_LAYER_LOCK_DIR``, or
``<temp dir>/benchflow-layer-locks-<uid>``; the folder is this user's alone
(mode 0700, owned by this uid, no symlink this user could have planted on the
way). A file is ``<daemon hash>-<kind>-<key>.lock``. The daemon is named by
``docker info``'s ``ID``, so two endpoints or contexts reaching the same
daemon share a lock and two daemons never wait on each other. Files are
opened with ``O_NOFOLLOW`` and must be regular files of this uid with one
link; they are never truncated and never removed (an unlinked lock file
would let the next process lock a new inode while the holder keeps the old
one). A killed holder's lock is released by the kernel.

The lock is taken with a non-blocking ``flock`` polled with backoff
(0.05 s doubling to 1 s), so cancelling a waiter never leaves a thread
blocked in ``flock``. When the folder or file cannot be used, or ``fcntl``
is missing (Windows), :class:`LockUnavailable` says why; the caller then
leaves the shared cache alone (a process lock cannot keep another process's
prune off a layer) and records ``lock: process``.

What it does not cover, said so the record is read right: processes on
other hosts sharing a remote daemon, or this daemon's users with another
lock folder, are not serialised.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib
import os
import stat
import tempfile
import time
import weakref
from pathlib import Path
from typing import Any


def _load_fcntl() -> Any:
    """``fcntl``, or None where the platform has none (Windows)."""
    try:
        return importlib.import_module("fcntl")
    except ImportError:  # pragma: no cover - Windows
        return None


fcntl: Any = _load_fcntl()

#: Where the lock files go (default: the temp dir, per user).
LOCK_DIR_ENV = "BENCHFLOW_LAYER_LOCK_DIR"
#: Lock modes a layer record names (``config.json``'s ``layers.<kind>.lock``).
HOST = "host"
PROCESS = "process"
TIMEOUT = "timeout"
#: Poll backoff of the non-blocking ``flock``: first sleep, then doubling to.
POLL_START_SEC = 0.05
POLL_MAX_SEC = 1.0


class LockUnavailable(Exception):
    """The host lock cannot be used here; the message says why."""


class LockTimeout(Exception):
    """The host lock was not free before the deadline."""


def lock_dir() -> Path:
    """The folder of this user's layer lock files (not created here)."""
    configured = os.environ.get(LOCK_DIR_ENV, "").strip()
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / f"benchflow-layer-locks-{_uid()}"


def _uid() -> int:
    getuid = getattr(os, "geteuid", None)
    return getuid() if getuid is not None else 0


def daemon_hash(daemon_id: str) -> str:
    return hashlib.sha256(daemon_id.encode("utf-8")).hexdigest()[:16]


def lock_name(daemon_id: str, kind: str, key: str) -> str:
    """The lock file's name for (daemon, kind, key): no path separator in it."""
    for part, what in ((kind, "kind"), (key, "key")):
        if not part or not all(c.isalnum() or c in "-_" for c in part):
            raise ValueError(f"not a layer {what}: {part!r}")
    return f"{daemon_hash(daemon_id)}-{kind}-{key}.lock"


def _check_ancestors(path: Path) -> None:
    """Refuse a lock folder reached through a symlink this user could plant.

    Every existing component is looked at with ``lstat``. A symlink is
    accepted only when root owns it and this process is not root (a system
    link such as macOS's ``/var``); any other, including one this user owns,
    is refused, since whoever can write the folder holding it could point it
    elsewhere. A folder others may write must be sticky (``/tmp``), or
    another user could rename the lock folder away.
    """
    me = _uid()
    current = Path(path.anchor or "/")
    for part in path.parts[1:]:
        current = current / part
        try:
            st = os.lstat(current)
        except FileNotFoundError:
            return
        except OSError as e:
            raise LockUnavailable(f"cannot read {current}: {e.strerror}") from e
        if stat.S_ISLNK(st.st_mode):
            if st.st_uid != 0 or me == 0:
                raise LockUnavailable(f"{current} is a symlink")
            continue
        if not stat.S_ISDIR(st.st_mode):
            raise LockUnavailable(f"{current} is not a folder")
        if current == path:
            return
        if st.st_mode & stat.S_IWOTH and not st.st_mode & stat.S_ISVTX:
            raise LockUnavailable(f"{current} is writable by others and not sticky")


def _open_dir(path: Path) -> int:
    """An fd of the lock folder, made if missing, checked to be this user's."""
    if not path.is_absolute():
        path = Path.cwd() / path
    _check_ancestors(path)
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    except OSError as e:
        raise LockUnavailable(f"cannot make {path}: {e.strerror}") from e
    _check_ancestors(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as e:
        raise LockUnavailable(f"cannot open {path}: {e.strerror}") from e
    try:
        st = os.fstat(fd)
        if not stat.S_ISDIR(st.st_mode):
            raise LockUnavailable(f"{path} is not a folder")
        if st.st_uid != _uid():
            raise LockUnavailable(f"{path} belongs to uid {st.st_uid}, not {_uid()}")
        if stat.S_IMODE(st.st_mode) & 0o077:
            # Matching uid does not prove we created this directory. In
            # particular root may have been pointed at /tmp or a shared
            # system folder. Refuse it without changing its permissions.
            raise LockUnavailable(f"{path} is not private to this user")
    except LockUnavailable:
        os.close(fd)
        raise
    except OSError as e:
        os.close(fd)
        raise LockUnavailable(f"cannot check {path}: {e.strerror}") from e
    return fd


def _open_lock_file(directory: Path, name: str) -> int:
    """An fd of the lock file ``name``: a regular file of this uid, one link.

    Created if missing (mode 0600); never truncated, never removed.
    """
    if fcntl is None:
        raise LockUnavailable("this platform has no fcntl.flock")
    dir_fd = _open_dir(directory)
    try:
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            fd = os.open(name, flags, 0o600, dir_fd=dir_fd)
        except OSError as e:
            raise LockUnavailable(
                f"cannot open lock file {directory / name}: {e.strerror}"
            ) from e
    finally:
        os.close(dir_fd)
    try:
        st = os.fstat(fd)
        problem = None
        if not stat.S_ISREG(st.st_mode):
            problem = "is not a regular file"
        elif st.st_uid != _uid():
            problem = f"belongs to uid {st.st_uid}"
        elif st.st_nlink != 1:
            problem = f"has {st.st_nlink} links"
        if problem is not None:
            raise LockUnavailable(f"lock file {directory / name} {problem}")
    except BaseException:
        os.close(fd)
        raise
    return fd


class HostLock:
    """The host lock of one (daemon, kind, key); one holder per instance."""

    def __init__(
        self,
        daemon_id: str,
        kind: str,
        key: str,
        *,
        directory: Path | None = None,
    ) -> None:
        self.name = lock_name(daemon_id, kind, key)
        self.directory = directory if directory is not None else lock_dir()
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def _try_once(self, fd: int) -> bool:
        assert fcntl is not None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        except OSError as e:
            raise LockUnavailable(f"flock failed: {e.strerror}") from e
        return True

    def try_acquire(self) -> bool:
        """Take the lock if it is free now (the prune's way); never waits.

        Raises :class:`LockUnavailable` when the lock cannot be used at all.
        """
        if self._fd is not None:
            raise RuntimeError("this HostLock is already held")
        fd = _open_lock_file(self.directory, self.name)
        try:
            got = self._try_once(fd)
        except BaseException:
            os.close(fd)
            raise
        if not got:
            os.close(fd)
            return False
        self._fd = fd
        return True

    async def acquire(self, deadline: float) -> float:
        """Wait for the lock until ``deadline`` (``time.monotonic()``).

        Returns the seconds waited. Raises :class:`LockTimeout` past the
        deadline and :class:`LockUnavailable` when the lock cannot be used;
        a cancelled wait closes its file, so nothing stays open or locked.
        """
        if self._fd is not None:
            raise RuntimeError("this HostLock is already held")
        started = time.monotonic()
        fd = _open_lock_file(self.directory, self.name)
        delay = POLL_START_SEC
        try:
            while not self._try_once(fd):
                left = deadline - time.monotonic()
                if left <= 0:
                    raise LockTimeout(
                        f"layer lock {self.name} still held after "
                        f"{time.monotonic() - started:.1f}s"
                    )
                await asyncio.sleep(min(delay, left))
                delay = min(delay * 2, POLL_MAX_SEC)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return time.monotonic() - started

    def release(self) -> None:
        """Let the next holder in; the file stays (stable inode)."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if fcntl is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# The process half: one asyncio.Lock per (event loop, daemon, kind, key). An
# asyncio.Lock belongs to the loop it first waited on, so a second loop in
# the same process (asyncio.run twice, a test) gets locks of its own; the
# flock still keeps the two loops apart, as it does two processes.
_process_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[str, str, str], asyncio.Lock]
] = weakref.WeakKeyDictionary()


def process_lock(daemon_id: str, kind: str, key: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    locks = _process_locks.setdefault(loop, {})
    return locks.setdefault((daemon_id, kind, key), asyncio.Lock())


def clear_process_locks() -> None:
    """Forget every process lock (tests)."""
    _process_locks.clear()
