"""Stable OS ownership for a job's human-readable process claim."""

from __future__ import annotations

import contextlib
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


class JobLockedError(RuntimeError):
    """Another process runs this job now."""

    hint = (
        "wait for that run to finish, or pass --fresh or another --jobs-dir "
        "to start a new job"
    )


@dataclass
class JobLock:
    path: Path
    fd: int

    @classmethod
    def acquire(cls, path: Path) -> JobLock:
        import fcntl

        path = path.resolve()
        # Keep the inode outside the job: a refused empty job may be removed,
        # but a lock inode must never be replaced while another opener holds it.
        name = hashlib.sha256(path.parent.name.encode()).hexdigest()
        # A regular hidden file cannot be mistaken for another job by callers
        # enumerating the jobs directory. It still has one permanent inode.
        guard = path.parent.parent / f".evaluation-lock-{name}"
        fd = os.open(guard, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise JobLockedError(
                f"Job {path.parent} is already running; wait for its lock"
            ) from None
        except BaseException:
            os.close(fd)
            raise
        return cls(path, fd)

    def close(self) -> None:
        os.close(self.fd)

    def release(self) -> None:
        try:
            with contextlib.suppress(OSError):
                self.path.unlink()
            with contextlib.suppress(OSError):
                self.path.parent.rmdir()
        finally:
            self.close()


def live_holder(job_dir: Path) -> int | None:
    """The pid of the live process on this host that runs ``job_dir``, else None.

    Reads the job's ``.evaluation.lock`` (its holder's pid and host) and
    writes nothing. A lock left by a process that is gone, or held on another
    host, gives None: readers then report the job as stopped."""
    import json
    import socket

    try:
        holder = json.loads((job_dir / ".evaluation.lock").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(holder, dict) or holder.get("host") != socket.gethostname():
        return None
    pid = holder.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except PermissionError:
        return pid
    except (OSError, OverflowError):
        return None
    return pid
