"""Find the bfx binary for a sandbox's architecture and check it is the
pinned one before it is used.

The binaries ship as package data (``bin/bfx-<arch>``); a development
checkout builds them with ``make bfx`` on a Linux build machine.
:func:`verified_binary` hashes the file and compares the hash with
:data:`~benchflow._bfx.pins.SHA256`; any other file is refused. A check is
remembered for the same file (device, inode, size, change time), so a
trial pays for hashing once per process.
"""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from pathlib import Path

from benchflow._bfx import pins

#: Where bfx lives inside a sandbox: a folder the agent cannot write
#: (a read-only bind or volume, or root's 0555 folder on upload backends).
SANDBOX_DIR = "/.bfx"
SANDBOX_PATH = f"{SANDBOX_DIR}/bfx"

BIN_DIR = Path(__file__).resolve().parent / "bin"

_ARCH_NAMES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "linux/amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "linux/arm64": "aarch64",
    "linux/arm64/v8": "aarch64",
}


class BfxUnavailable(RuntimeError):
    """No verified bfx binary exists for the architecture."""


def normalize_arch(arch: str) -> str:
    """``x86_64`` or ``aarch64`` for the names Docker, ``uname -m`` and
    Python use; raises :class:`BfxUnavailable` for anything else."""
    name = _ARCH_NAMES.get(arch.strip().lower())
    if name is None:
        raise BfxUnavailable(f"bfx is built for x86_64 and aarch64, not {arch!r}")
    return name


def binary_path(arch: str) -> Path:
    return BIN_DIR / f"bfx-{normalize_arch(arch)}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


_checked: dict[tuple[str, int, int, int, int], str] = {}
_lock = threading.Lock()


def verified_binary(arch: str) -> Path:
    """The path of the pinned bfx for *arch*, after checking its hash.

    Refuses a missing file, a link, anything that is not a regular file, and
    a file whose SHA-256 is not the pin (:class:`BfxUnavailable`).
    """
    name = normalize_arch(arch)
    path = BIN_DIR / f"bfx-{name}"
    want = pins.SHA256.get(name, "")
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise BfxUnavailable(
            f"{path} is missing: this checkout has no bfx binaries "
            "(run `make bfx` on a Linux build machine, or install a release wheel)"
        ) from None
    if not stat.S_ISREG(st.st_mode):
        raise BfxUnavailable(f"{path} is not a regular file")
    key = (name, st.st_dev, st.st_ino, st.st_size, st.st_ctime_ns)
    with _lock:
        if _checked.get(key) == want:
            return path
    got = sha256_of(path)
    if got != want:
        raise BfxUnavailable(
            f"{path} has SHA-256 {got}, not the pinned {want} "
            "(rebuild with `make bfx` from this checkout)"
        )
    with _lock:
        _checked[key] = want
    return path


def available(arch: str) -> bool:
    try:
        verified_binary(arch)
    except BfxUnavailable:
        return False
    return True
