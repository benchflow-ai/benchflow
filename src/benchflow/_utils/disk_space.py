"""A full disk on the host: the user's to fix, said as such.

A job folder on a full disk fails in three places: the sandbox's write of
the reward into the mounted ``verifier/`` folder (the file is left empty),
BenchFlow's own writes of result.json and summary.json (``OSError`` errno
28), and anything after. Each says the same thing: which disk, how much is
free, and what to do (:data:`DISK_FULL_STEP`), and the trial is filed under
``disk_full`` (setup), never retried.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any

from benchflow._utils.scoring import DISK_FULL_MARKER
from benchflow.errors import UserError

# Below this much free space an empty reward file is the disk's doing.
LOW_SPACE_BYTES = 1024 * 1024
DISK_FULL_STEP = "free space on that disk, or pass another --jobs-dir"
_FULL_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})


def is_disk_full(exc: BaseException) -> bool:
    """Whether ``exc`` is the host's disk (or quota) being full."""
    return isinstance(exc, OSError) and exc.errno in _FULL_ERRNOS


def free_bytes(where: Any) -> int | None:
    """Free bytes on the disk holding ``where`` (a path, or an open file), or None."""
    try:
        if hasattr(where, "fileno"):
            stat = os.fstatvfs(where.fileno())
        else:
            stat = os.statvfs(Path(where))
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return stat.f_bavail * stat.f_frsize


def size_text(count: int) -> str:
    """``24 KB``, ``3.1 MB``: a byte count for a person."""
    if count < 1024 * 1024:
        return f"{count // 1024} KB"
    if count < 1024**3:
        return f"{count / 1024**2:.1f} MB"
    return f"{count / 1024**3:.1f} GB"


DISK_STEP = "free space on that disk"
# With no file named: the places BenchFlow writes to (review L-A2).
DISK_FULL_SOMEWHERE = (
    "a disk BenchFlow writes to is full (the jobs folder, the cache under "
    "~/.cache, or the temporary folder {tmp})"
)


def disk_full_text(path: Path | None = None, *, job_file: bool = True) -> str:
    """Which disk is full, how much is free, and what BenchFlow was writing.

    ``the disk that holds the job folder is full (24 KB free) writing <path>``
    for a job's own file; ``the disk that holds <path> is full (...)`` for any
    other file (the cache, the temporary folder: review L-A2); with no file
    named, the places it could be.
    """
    if path is None:
        if not job_file:
            import tempfile

            return DISK_FULL_SOMEWHERE.format(tmp=tempfile.gettempdir())
        return DISK_FULL_MARKER
    free = free_bytes(path.parent)
    room = f" ({size_text(free)} free)" if free is not None else ""
    if job_file:
        return f"{DISK_FULL_MARKER}{room} writing {path}"
    return f"the disk that holds {path} is full{room}"


def _umask() -> int:
    """The process umask, read without changing it where the OS allows."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError):
        pass
    current = os.umask(0o022)
    os.umask(current)
    return current


def _file_mode(path: Path) -> int:
    """The mode ``path`` has, or the one a new file there gets."""
    try:
        return path.stat().st_mode & 0o7777
    except OSError:
        return 0o666 & ~_umask()


def write_text_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` whole, or leave it as it was.

    The text goes to a temporary file beside it, is flushed to disk, then
    renamed over it: a reader never sees a part of it, and a full disk
    leaves the old file (or none), never a truncated one (DF-p5-cli-craft-5).
    A full disk raises :class:`DiskFullError`.
    """
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # mkstemp makes the file 0600; a plain write makes it 0666 less the
        # umask, and a replaced file keeps its own mode (review A-2).
        os.chmod(temporary, _file_mode(path))
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        if is_disk_full(exc):
            raise DiskFullError(path) from exc
        raise
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


class DiskFullError(OSError, UserError):
    """BenchFlow could not write a job file: the host's disk is full.

    An expected error (the CLI prints it and its next step, no traceback):
    the ``setup`` fault. Raised from the ``OSError`` the write failed with.
    """

    fault = "setup"

    def __init__(self, path: Path | None = None) -> None:
        super().__init__(errno.ENOSPC, disk_full_text(path))
        self.filename = str(path) if path is not None else None
        self.hint = DISK_FULL_STEP

    def __str__(self) -> str:
        return self.strerror or DISK_FULL_MARKER
