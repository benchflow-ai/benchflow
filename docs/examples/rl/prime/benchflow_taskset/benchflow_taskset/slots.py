"""A machine-wide cap on live BenchFlow sandboxes.

prime-rl runs episodes in several env-server worker processes, so a per-process
semaphore cannot cap the sandboxes they hold together. Each live sandbox holds an
exclusive ``flock`` on one of ``count`` slot files; the kernel releases it when
the holder closes the file or dies, so a crashed worker never leaks a slot.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
from collections.abc import AsyncIterator
from pathlib import Path


def _try_lock(directory: Path, count: int) -> tuple[int, int] | None:
    for index in range(count):
        fd = os.open(directory / f"slot-{index:03d}", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        except BaseException:
            os.close(fd)
            raise
        return index, fd
    return None


@contextlib.asynccontextmanager
async def sandbox_slot(directory: str | Path, count: int, *, poll_sec: float = 0.5) -> AsyncIterator[int]:
    """Hold one of ``count`` machine-wide slots while the body runs; yields its index."""
    if count < 1:
        raise ValueError("count must be at least 1")
    path = Path(directory).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    while True:
        held = _try_lock(path, count)
        if held is not None:
            break
        await asyncio.sleep(poll_sec)
    index, fd = held
    try:
        yield index
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
