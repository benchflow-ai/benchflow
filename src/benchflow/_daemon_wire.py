"""Private, bounded JSON transport for the opt-in local runtime (no pickle)."""

from __future__ import annotations

import array
import contextlib
import json
import os
import socket
import stat
import struct
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

LIMIT = 8 * 1024 * 1024


def peer_uid(sock: socket.socket) -> int:
    """Fail closed on platforms without authenticated Unix peers."""
    if hasattr(socket, "SO_PEERCRED"):
        return struct.unpack(
            "3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )[1]
    raise OSError("The local runtime requires Unix peer credentials (Linux).")


def private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise OSError("The daemon folder must be owned by this uid with mode 0700.")


def connect(path: Path) -> socket.socket:
    private_dir(path.parent)
    info = path.lstat()
    if (
        not stat.S_ISSOCK(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise OSError("The daemon socket must be owned by this uid with mode 0600.")
    sock = socket.socket(socket.AF_UNIX)
    try:
        sock.settimeout(5)
        sock.connect(str(path))
        if peer_uid(sock) != os.getuid():
            raise OSError("Daemon uid differs from the caller's uid.")
        return sock
    except BaseException:
        sock.close()
        raise


def listen(path: Path) -> socket.socket:
    private_dir(path.parent)
    sock = socket.socket(socket.AF_UNIX)
    try:
        # Do not remove an existing path: another daemon may own it.
        sock.bind(str(path))
        path.chmod(0o600)
        sock.listen(32)
        return sock
    except BaseException:
        sock.close()
        raise


def send(sock: socket.socket, value: Any) -> None:
    data = json.dumps(value).encode()
    if len(data) > LIMIT:
        raise ValueError("Daemon request exceeds the limit.")
    sock.sendall(struct.pack("!I", len(data)) + data)


def _read(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        block = sock.recv(size - len(chunks))
        if not block:
            raise EOFError("Daemon connection closed.")
        chunks.extend(block)
    return bytes(chunks)


def receive(sock: socket.socket) -> Any:
    size = struct.unpack("!I", _read(sock, 4))[0]
    if size > LIMIT:
        raise ValueError("Daemon request exceeds the limit.")
    return json.loads(_read(sock, size))


def send_fds(sock: socket.socket, fds: list[int]) -> None:
    sock.sendmsg(
        [b"F"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))]
    )


def receive_fds(sock: socket.socket, count: int) -> list[int]:
    item = array.array("i")
    data, ancillary, flags, _ = sock.recvmsg(
        1, socket.CMSG_SPACE(count * item.itemsize)
    )
    for level, kind, payload in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            item.frombytes(payload[: len(payload) - len(payload) % item.itemsize])
    if data != b"F" or flags & socket.MSG_CTRUNC or len(item) != count:
        for fd in item:
            os.close(fd)
        raise ValueError("Malformed daemon descriptor handoff.")
    return list(item)


# Modules between a draw and the code that asked for it.
_RANDOM_PLUMBING = frozenset({"os", "random", "secrets", "uuid"})


@contextlib.contextmanager
def recording_randomness() -> Iterator[set[str]]:
    """Record the modules that draw OS randomness while the block runs.

    A value drawn at import in a pre-imported parent is the same in every
    fork, where each cold process draws its own; the preloads require every
    such module to be on their redraw list. ``secrets`` and ``SystemRandom``
    draw through ``random._urandom``, ``uuid4`` through ``os.urandom``.
    """
    import random

    callers: set[str] = set()
    modules = (cast("Any", os), cast("Any", random))
    originals = (modules[0].urandom, modules[1]._urandom)

    def urandom(size: int, /) -> bytes:
        frame = sys._getframe(1)
        while frame is not None and frame.f_globals.get("__name__") in _RANDOM_PLUMBING:
            frame = frame.f_back
        callers.add(frame.f_globals.get("__name__", "?") if frame is not None else "?")
        return originals[0](size)

    modules[0].urandom = modules[1]._urandom = urandom
    try:
        yield callers
    finally:
        modules[0].urandom, modules[1]._urandom = originals
