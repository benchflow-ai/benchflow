"""Disarm what the cloud metadata block's delegate inventory counts, for the
agent's turn, and put it back before grading (option A).

The inventory (``lockdown._cloud_metadata_delegate_check_cmd``) counts every
set-ID file and file capability outside a few narrow distribution helpers:
any of them would let a process outside the agent's no_new_privs launch (a
login service, a cron job) gain root or NET_ADMIN and remove the block. Every
stock Debian and Ubuntu image ships ``su`` setuid root, so refusing them all
is not workable. Instead, before the agent starts, root:

1. records each counted file's path, mode, owner, device and inode, link
   count, size, sha256 and ``security.capability`` value in a root-only
   record (:data:`RECORD_DIR`), one entry per inode (hard links share it);
2. takes the set-ID bits off (``chmod u-s,g-s``) and removes the capability;
3. inventories again: nothing may count any more.

Before grading, after the kill of the agent's processes, root puts the
recorded bits and capability back only on a file whose content and standing
are unchanged: reached at its recorded path without a link on any part of
it, with the same owner uid and gid, size, sha256 and link count. A file the
agent changed keeps nothing back: the sha256 shows the content is the same,
the owner that nobody re-owned it, and the link count that no hard link was
added to the inode. The inode number is recorded but not compared: it adds
nothing once the content is proven, and a sandbox restored from a snapshot
has new inodes by construction. Each step verifies itself with ``fstat``
(and the capability by reading it back). Every failure to disarm refuses the
start; a file that could not be restored stays disarmed through grading and
is named in the trial's records.

The record lives in the sandbox, so a snapshot carries it. Every connect's
inventory says whether one is there (``BF metadata-delegates-record``), and
the block then has root list it (``recorded`` lines, :func:`read_strip`), so
a sandbox restored from a snapshot (a branch's child or parent) has its files
put back before its own grading too.

The work is one Python program (:data:`DISARM_PROGRAM`) that opens each file
one path component at a time with ``O_NOFOLLOW``, so it never acts through a
link, and changes it through that descriptor. It runs under root's own Python:
PT's protected kit Python, else BenchFlow's managed CPython, else the staged
CPython below. Where the image has neither, BenchFlow stages its pinned
python-build-standalone CPython (the build its uv pin installs as the
managed Python) from the host's cache into a root-only folder, so no stock
image is refused for lacking tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import re
import shlex
import tempfile
import threading
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow._root_only import root_only_tree_check
from benchflow._runtime_tree import RUNTIME_ROOT
from benchflow._uv_pin import PYTHON_INSTALL_DIR, PYTHON_VERSION
from benchflow.task.verifier_errors import VerifierInfrastructureError

logger = logging.getLogger(__name__)

#: The root-only folder that holds the record of what was disarmed.
RECORD_DIR = f"{RUNTIME_ROOT}/delegates"
#: Where BenchFlow stages its own CPython when the image has no Python.
STAGED_ROOT = f"{RUNTIME_ROOT}/staged-python"

# python-build-standalone's CPython, the build uv 0.13.4 (benchflow._uv_pin)
# installs as BenchFlow's managed Python. Fetched 2026-10-28; each value is
# the release's own SHA256SUMS entry. The musl builds use the image's musl
# loader and the gnu builds its glibc: the C library is the one part of the
# image this interpreter uses, as every program root runs there does.
STAGED_PYTHON_RELEASE = "20261013"
STAGED_PYTHON_VERSION = PYTHON_VERSION
STAGED_PYTHON_SHA256: Mapping[tuple[str, str], str] = {
    ("x86_64", "gnu"): (
        "178d3d83b6e8e99285a65d3fce9c16b87952cc5efdb4d1a0e3ff3be18ff0fc9c"
    ),
    ("aarch64", "gnu"): (
        "9467a99bacc7d90b64b02de7fa9ecfc99b72905e64e74f7ff6b460ad95cfed5a"
    ),
    ("x86_64", "musl"): (
        "99e53f010e58b62dd5bc849cc4822bde9fff7171c6d020fbe4afed2d861ccc77"
    ),
    ("aarch64", "musl"): (
        "efd6e0900431214a9b559803071939421a8a8489375b9be063221a8cfe991c60"
    ),
}
# Astral's mirror first (the host uv itself downloads from), then GitHub.
STAGED_PYTHON_BASES: tuple[str, ...] = (
    "https://releases.astral.sh/github/python-build-standalone/releases/download/"
    + STAGED_PYTHON_RELEASE,
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    + STAGED_PYTHON_RELEASE,
)
_DOWNLOAD_TIMEOUT_SEC = 120
_STAGE_EXEC_SEC = 300
_RESTORE_EXEC_SEC = 120


class StagedPythonUnavailable(RuntimeError):
    """BenchFlow could not fetch or stage its own CPython."""


def staged_python_file(arch: str, libc: str) -> str:
    """The release file name for *arch* (``x86_64``, ``aarch64``) and *libc*
    (``gnu``, ``musl``)."""
    return (
        f"cpython-{STAGED_PYTHON_VERSION}+{STAGED_PYTHON_RELEASE}-{arch}-"
        f"unknown-linux-{libc}-install_only_stripped.tar.gz"
    )


def staged_python_dir(arch: str, libc: str) -> str:
    """The sandbox folder the build for *arch* and *libc* is unpacked into."""
    return f"{STAGED_ROOT}/{STAGED_PYTHON_SHA256[(arch, libc)]}"


def staged_python_bin(arch: str, libc: str) -> str:
    return f"{staged_python_dir(arch, libc)}/python/bin/python3.12"


def _cache_dir() -> Path:
    base = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    root = Path(base) if base else Path.home() / ".cache"
    return root / "benchflow" / "staged-python"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


_verified: dict[tuple[str, int, int, int], str] = {}
_cache_lock = threading.Lock()


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "benchflow"})
    with (
        urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SEC) as answer,
        open(dest, "wb") as out,
    ):
        while chunk := answer.read(1 << 20):
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())


def host_tarball(
    arch: str, libc: str, *, fetch: Callable[[str, Path], None] = _download
) -> Path:
    """The pinned build for *arch* and *libc* in the host's cache, fetched and
    checked against :data:`STAGED_PYTHON_SHA256` the first time. A file whose
    sha256 is not the pin is never returned."""
    want = STAGED_PYTHON_SHA256.get((arch, libc))
    if want is None:
        raise StagedPythonUnavailable(f"no staged CPython for {arch}-{libc}")
    cache = _cache_dir()
    path = cache / f"{want}.tar.gz"
    with _cache_lock:
        if path.is_file():
            st = path.stat()
            key = (str(path), st.st_size, st.st_mtime_ns, st.st_ino)
            if _verified.get(key) == want or _sha256_file(path) == want:
                _verified[key] = want
                return path
        cache.mkdir(parents=True, exist_ok=True)
        name = staged_python_file(arch, libc).replace("+", "%2B")
        errors = []
        for base in STAGED_PYTHON_BASES:
            fd, tmp = tempfile.mkstemp(dir=cache, prefix=".partial-")
            os.close(fd)
            try:
                fetch(f"{base}/{name}", Path(tmp))
                got = _sha256_file(Path(tmp))
                if got != want:
                    raise StagedPythonUnavailable(
                        f"sha256 {got}, not the pinned {want}"
                    )
                os.replace(tmp, path)
                st = path.stat()
                _verified[(str(path), st.st_size, st.st_mtime_ns, st.st_ino)] = want
                return path
            except Exception as exc:  # the next mirror, then a clear refusal
                errors.append(f"{base}: {type(exc).__name__}: {exc}"[:300])
            finally:
                Path(tmp).unlink(missing_ok=True)
        raise StagedPythonUnavailable(
            "could not fetch BenchFlow's pinned CPython "
            f"({staged_python_file(arch, libc)}): " + "; ".join(errors)
        )


# --- the program root runs ----------------------------------------------------

# The extended-attribute calls root's programs use, for any CPython 3.5 or
# later. CPython leaves os.getxattr and its kin out where it was built without
# linux/limits.h, as python-build-standalone's musl builds are (the staged
# CPython on Alpine, and BenchFlow's managed CPython there); the C library
# has them, through ctypes. ``lgetxattr(path, name)`` never follows a link;
# the others take a descriptor. Each raises OSError with the call's errno, and
# is None where neither route exists, which root's programs refuse.
XATTR_SHIM = r"""
def _c_xattr():
    import ctypes as c
    libc = c.CDLL(None, use_errno=True)
    calls = {}
    for name, args, res in (
        ("lgetxattr", (c.c_char_p, c.c_char_p, c.c_void_p, c.c_size_t), c.c_ssize_t),
        ("fgetxattr", (c.c_int, c.c_char_p, c.c_void_p, c.c_size_t), c.c_ssize_t),
        ("fsetxattr", (c.c_int, c.c_char_p, c.c_char_p, c.c_size_t, c.c_int), c.c_int),
        ("fremovexattr", (c.c_int, c.c_char_p), c.c_int),
    ):
        f = getattr(libc, name)
        f.argtypes = args
        f.restype = res
        calls[name] = f

    def check(r):
        if r < 0:
            e = c.get_errno()
            raise OSError(e, os.strerror(e))
        return r

    def get(f, target, name):
        # 64 KiB is the kernel's largest attribute value (XATTR_SIZE_MAX).
        for size in (256, 65536):
            buf = c.create_string_buffer(size)
            n = f(target, name.encode(), buf, size)
            if n >= 0:
                return buf.raw[:n]
            if c.get_errno() != errno.ERANGE:
                break
        check(-1)

    return (
        lambda path, name: get(calls["lgetxattr"], path, name),
        lambda fd, name: get(calls["fgetxattr"], fd, name),
        lambda fd, name, value: check(
            calls["fsetxattr"](fd, name.encode(), value, len(value), 0)
        ),
        lambda fd, name: check(calls["fremovexattr"](fd, name.encode())),
    )


if hasattr(os, "getxattr"):
    def lgetxattr(path, name):
        return os.getxattr(path, name, follow_symlinks=False)
    fgetxattr = os.getxattr
    fsetxattr = getattr(os, "setxattr", None)
    fremovexattr = getattr(os, "removexattr", None)
else:
    try:
        lgetxattr, fgetxattr, fsetxattr, fremovexattr = _c_xattr()
    except Exception:
        lgetxattr = fgetxattr = fsetxattr = fremovexattr = None
"""

# Python 3.5 or later (the image's Python may be old): no f-strings. argv:
# ``strip RECORD_DIR TRUSTED_UIDS``, with the paths to strip on stdin (one
# hex path a line), or ``restore RECORD_DIR TRUSTED_UIDS [PROC]``. Every
# folder on the way to the record, and the record's folder itself, must be a
# trusted uid's and not a link; the record's folder is made root's alone
# (0700). Output lines start ``BF metadata-delegates-``. A strip also lists,
# as ``recorded`` lines, what the record held already (disarmed by an earlier
# connect, before a snapshot): with nothing to strip it only lists.
#
# A restore puts back nothing while a process that is not root runs without
# no_new_privs (``live`` lines and ``kept ... live-non-nnp-process``). Else,
# for every name of every entry: ``restored <hex> <mode>:exact`` when the
# file at that path, reached without a link, has the recorded owner, size,
# sha256 and link count and still the mode the strip left; ``kept`` with why
# when it is not that file any more (``changed:<fields>``, ``changed:mode``,
# ``changed:type``, ``gone-or-linked:<errno>``); ``restore-failed`` with the
# errno when root could not do it (BenchFlow's failure, not the agent's).
# What is not put back stays in the record.
_DISARM_SOURCE = r"""
import errno, hashlib, json, os, stat, sys
@XATTR@
os.umask(0o077)
mode = sys.argv[1]
record_dir = os.fsencode(sys.argv[2])
trusted = set(int(u) for u in sys.argv[3].split(","))
out = sys.stdout.buffer
SETID = stat.S_ISUID | stat.S_ISGID
# What a restore compares, besides the path reached without a link.
IDENTITY = ("uid", "gid", "size", "sha256", "links")
# A path that names nothing, or has a link on it, now: the file is not there.
GONE = (errno.ENOENT, errno.ENOTDIR, errno.ELOOP)
CAP = "security.capability"
NOCAP = (errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP)
DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def say(*parts):
    out.write(b" ".join(parts) + b"\n")


def ename(e):
    if isinstance(e, OSError) and e.errno:
        return errno.errorcode.get(e.errno, "E%d" % e.errno).encode()
    return type(e).__name__.encode()


def parts_of(path):
    parts = [p for p in path.split(b"/") if p]
    if not path.startswith(b"/") or not parts or b".." in parts or b"." in parts:
        raise OSError(errno.EINVAL, "not an absolute path")
    return parts


def open_file(path):
    # One component at a time: a link anywhere on the path fails (ELOOP or
    # ENOTDIR), so nothing is ever done through a link.
    parts = parts_of(path)
    fd = os.open(b"/", DIR)
    try:
        for p in parts[:-1]:
            nfd = os.open(p, DIR, dir_fd=fd)
            os.close(fd)
            fd = nfd
        return os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
        )
    finally:
        os.close(fd)


def record_fd(create):
    parts = parts_of(record_dir)
    fd = os.open(b"/", DIR)
    for i, p in enumerate(parts):
        last = i == len(parts) - 1
        try:
            nfd = os.open(p, DIR, dir_fd=fd)
        except OSError as e:
            if e.errno != errno.ENOENT or not create:
                os.close(fd)
                raise
            os.mkdir(p, 0o700, dir_fd=fd)
            nfd = os.open(p, DIR, dir_fd=fd)
            # A folder above the record keeps the usual 0755 (umask aside).
            os.fchmod(nfd, 0o700 if last else 0o755)
        os.close(fd)
        fd = nfd
        st = os.fstat(fd)
        if st.st_uid not in trusted:
            os.close(fd)
            raise OSError(errno.EPERM, "a folder on the record's path is not root's")
        if last:
            if stat.S_IMODE(st.st_mode) & 0o077:
                os.fchmod(fd, 0o700)
        elif st.st_mode & 0o022 and not st.st_mode & stat.S_ISVTX:
            os.close(fd)
            raise OSError(errno.EPERM, "others may write a folder on the record's path")
    return fd


def digest(fd):
    h = hashlib.sha256()
    os.lseek(fd, 0, 0)
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            return h.hexdigest()
        h.update(chunk)


def caps_of(fd):
    try:
        return fgetxattr(fd, CAP)
    except OSError as e:
        if e.errno in NOCAP:
            return None
        raise


class NotRegular(Exception):
    pass


def facts(fd):
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        raise NotRegular()
    caps = caps_of(fd)
    return {
        "dev": st.st_dev,
        "ino": st.st_ino,
        "links": st.st_nlink,
        "size": st.st_size,
        "uid": st.st_uid,
        "gid": st.st_gid,
        "mode": stat.S_IMODE(st.st_mode),
        "sha256": digest(fd),
        "caps": caps.hex() if caps is not None else None,
    }


def load():
    try:
        dfd = record_fd(False)
    except OSError as e:
        if e.errno == errno.ENOENT:
            return []
        raise
    try:
        fd = os.open(b"record.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
    except OSError as e:
        os.close(dfd)
        if e.errno == errno.ENOENT:
            return []
        raise
    data = b""
    try:
        while True:
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                break
            data += chunk
    finally:
        os.close(fd)
        os.close(dfd)
    # A list: entries of different snapshot generations can share a device
    # and inode number, and each is its own.
    return list(json.loads(data.decode())["entries"])


def save(entries):
    dfd = record_fd(True)
    try:
        if not entries:
            try:
                os.unlink(b"record.json", dir_fd=dfd)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise
            return
        data = json.dumps(
            {"version": 1, "entries": list(entries.values())}, sort_keys=True
        ).encode()
        tmp = b"record.json.%d" % os.getpid()
        fd = os.open(
            tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dfd
        )
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.rename(tmp, b"record.json", src_dir_fd=dfd, dst_dir_fd=dfd)
        os.fsync(dfd)
    finally:
        os.close(dfd)


def line(kind, path, f):
    say(
        b"BF metadata-delegates-" + kind,
        path.hex().encode(),
        b"%o" % f["mode"],
        b"%d" % f["uid"],
        b"%d" % f["gid"],
        b"%d" % f["ino"],
        b"%d" % f["links"],
        b"%d" % f["size"],
        f["sha256"].encode(),
        (f["caps"] or "-").encode(),
    )


def strip(targets):
    # What the record held already is keyed apart from this run's files:
    # after a snapshot restore its device and inode numbers name nothing.
    entries = dict((("earlier", i), e) for i, e in enumerate(load()))
    earlier = sorted(entries)
    failed = 0
    for path in targets:
        try:
            fd = open_file(path)
        except Exception as e:
            say(b"BF metadata-delegates-strip-failed", path.hex().encode(), ename(e))
            failed += 1
            continue
        try:
            f = facts(fd)
            key = (f["dev"], f["ino"])
            privileged = f["mode"] & SETID or f["caps"] is not None
            if key in entries and not privileged:
                # A second name (hard link) of a file disarmed already.
                if path.hex() not in entries[key]["paths"]:
                    entries[key]["paths"].append(path.hex())
                line(b"stripped", path, entries[key])
                continue
            paths = entries[key]["paths"] if key in entries else []
            if path.hex() not in paths:
                paths = paths + [path.hex()]
            entries[key] = dict(f, paths=paths)
            # Recorded before anything changes: a later failure leaves the
            # record able to restore what was already done.
            save(entries)
            if f["mode"] & SETID:
                os.fchmod(fd, f["mode"] & ~SETID)
            if f["caps"] is not None:
                fremovexattr(fd, CAP)
            now = os.fstat(fd)
            if now.st_mode & SETID or caps_of(fd) is not None:
                raise OSError(errno.EPERM, "still privileged after the strip")
            # The link count and mode the restore compares are the stripped
            # file's: an overlay copy-up of a hard-linked file leaves it its
            # own inode.
            entries[key]["links"] = f["links"] = now.st_nlink
            entries[key]["stripped_mode"] = stat.S_IMODE(now.st_mode)
            line(b"stripped", path, f)
        except Exception as e:
            say(b"BF metadata-delegates-strip-failed", path.hex().encode(), ename(e))
            failed += 1
        finally:
            os.close(fd)
    save(entries)
    for key in earlier:
        for h in entries[key]["paths"]:
            line(b"recorded", bytes.fromhex(h), entries[key])
    say(b"BF metadata-delegates-strip-done", b"%d" % len(entries), b"%d" % failed)
    return 7 if failed else 0


def live_non_nnp(proc):
    # Processes that are not root (no real, effective or saved uid 0, so
    # they cannot become root by themselves) and run without no_new_privs:
    # a set-ID file or a file capability put back would work for them. The
    # agent's processes run under no_new_privs; these came from elsewhere
    # (a login, a cron job, a service of the image).
    found = []
    for name in os.listdir(proc):
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(proc, name, "status"), "rb") as f:
                text = f.read().decode("utf-8", "replace")
        except OSError as e:
            if e.errno in (errno.ENOENT, errno.ESRCH):
                continue
            raise
        fields = {}
        for row in text.splitlines():
            k, _, v = row.partition(":")
            fields[k] = v.strip()
        if fields.get("State", "")[:1] in ("Z", "X"):
            continue
        uids = fields.get("Uid", "").split()
        if len(uids) < 3 or "0" in uids[:3]:
            continue
        if fields.get("NoNewPrivs") == "1":
            continue
        found.append((int(name), uids[0], fields.get("Name", "")))
    return sorted(found)


def restore(proc):
    entries = load()
    counts = {b"restored": 0, b"kept": 0, b"restore-failed": 0}

    def verdict(kind, h, why):
        counts[kind] += 1
        say(b"BF metadata-delegates-" + kind, h.encode(), why)

    live = live_non_nnp(proc)
    if live:
        # Nothing goes back while such a process could use it.
        for pid, uid, name in live[:20]:
            say(
                b"BF metadata-delegates-live", b"%d" % pid, uid.encode(),
                name.encode("utf-8", "replace").hex().encode() or b"-",
            )
        for e in entries:
            for h in e["paths"]:
                verdict(b"kept", h, b"live-non-nnp-process")
        save(dict(enumerate(entries)))
    else:
        left = []
        for e in entries:
            want = e["mode"]
            want_caps = bytes.fromhex(e["caps"]) if e["caps"] else None
            stripped = e.get("stripped_mode", want & ~SETID)
            unrestored = []
            # Every name: on overlayfs a strip can leave two names of one
            # file two files.
            for h in e["paths"]:
                try:
                    fd = open_file(bytes.fromhex(h))
                except OSError as er:
                    unrestored.append(h)
                    if er.errno in GONE:
                        verdict(b"kept", h, b"gone-or-linked:" + ename(er))
                    else:
                        verdict(b"restore-failed", h, ename(er))
                    continue
                try:
                    f = facts(fd)
                    differs = [k for k in IDENTITY if f[k] != e.get(k)]
                    caps = caps_of(fd)
                    if differs:
                        unrestored.append(h)
                        verdict(b"kept", h, ("changed:" + ",".join(differs)).encode())
                    elif f["mode"] == want and caps == want_caps:
                        # Put back already, through another name of it.
                        verdict(b"restored", h, b"%o:exact" % want)
                    elif f["mode"] != stripped:
                        unrestored.append(h)
                        verdict(b"kept", h, b"changed:mode")
                    else:
                        os.fchmod(fd, want)
                        if want_caps is not None and caps_of(fd) != want_caps:
                            fsetxattr(fd, CAP, want_caps)
                        now = stat.S_IMODE(os.fstat(fd).st_mode)
                        if now != want or caps_of(fd) != want_caps:
                            unrestored.append(h)
                            verdict(b"restore-failed", h, b"verify")
                        else:
                            verdict(b"restored", h, b"%o:exact" % now)
                except NotRegular:
                    unrestored.append(h)
                    verdict(b"kept", h, b"changed:type")
                except Exception as er:
                    unrestored.append(h)
                    verdict(b"restore-failed", h, ename(er))
                finally:
                    os.close(fd)
            if unrestored:
                left.append(dict(e, paths=unrestored))
        # What was not put back stays recorded, for a later grading.
        save(dict(enumerate(left)))
    say(
        b"BF metadata-delegates-restore-done",
        b"%d" % counts[b"restored"],
        b"%d" % counts[b"kept"],
        b"%d" % counts[b"restore-failed"],
    )
    return 7 if counts[b"restore-failed"] else 0


try:
    if None in (fgetxattr, fsetxattr, fremovexattr):
        # Without them a capability could be neither read nor put back.
        say(b"BF metadata-delegates-disarm-error xattr")
        code = 7
    elif mode == "strip":
        # The files to strip come on stdin, one hex path a line: no argument
        # grows with how many there are.
        code = strip(
            [bytes.fromhex(h) for h in sys.stdin.buffer.read().decode().split()]
        )
    elif mode == "restore":
        code = restore(sys.argv[4] if len(sys.argv) > 4 else "/proc")
    else:
        raise ValueError(mode)
except Exception as e:
    say(b"BF metadata-delegates-disarm-error", ename(e))
    code = 7
out.flush()
sys.exit(code)
"""
DISARM_PROGRAM = _DISARM_SOURCE.replace("@XATTR@", XATTR_SHIM)


def _trusted_uids() -> str:
    from benchflow.sandbox import lockdown

    return ",".join(lockdown._CLOUD_METADATA_TRUSTED_UIDS)


def disarm_cmd(
    python: str,
    action: str,
    *,
    tree: str | None = None,
    targets: tuple[bytes, ...] = (),
    record_dir: str = RECORD_DIR,
    proc_root: str = "/proc",
) -> str:
    """Root's command running :data:`DISARM_PROGRAM` (*action* ``strip`` with
    *targets*, or ``restore``) under *python*, after checking that *python*
    (and its whole *tree*, for a Python BenchFlow installed or staged) is
    root's alone. Prints ``BF metadata-delegates-disarm-error python`` and
    exits 7 when it is not.

    The targets go on the program's stdin (a here-document of hex lines), so
    its arguments stay the same size however many files there are; a
    command over the transport's bound travels as data
    (:mod:`benchflow.sandbox._transport`). *proc_root* is for tests."""
    from benchflow.sandbox.lockdown import _cloud_metadata_root_prelude

    py = shlex.quote(python)
    custody = f'walk {py} && [ -f "$W" ] && [ -x "$W" ]'
    if tree:
        custody += " && " + root_only_tree_check(
            tree, "the delegate inventory's Python"
        )
    args = " ".join(shlex.quote(a) for a in (action, record_dir, _trusted_uids()))
    if action == "strip":
        # Hex lines can never be the delimiter.
        stdin = (
            "<<'BF_DELEGATE_TARGETS'\n"
            + "".join(f"{t.hex()}\n" for t in targets)
            + "BF_DELEGATE_TARGETS\n"
        )
    else:
        if proc_root != "/proc":
            args += " " + shlex.quote(proc_root)
        stdin = "</dev/null\n"
    return (
        "( " + _cloud_metadata_root_prelude() + f"if ! {{ {custody}; }}; then "
        "echo 'BF metadata-delegates-disarm-error python'; exit 7; fi; "
        f"{py} -I -S -B -c {shlex.quote(DISARM_PROGRAM)} {args} {stdin})"
    )


# mode uid gid inode links size sha256 capability
_FIELDS = (
    r"([0-7]+) ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+) ([0-9a-f]{64}) "
    r"([0-9a-f]+|-)"
)
# ``stripped`` now, or ``recorded``: disarmed by an earlier connect (before
# the snapshot this sandbox was restored from), still in the record.
_STRIPPED = re.compile(
    rf"BF metadata-delegates-(stripped|recorded) ((?:[0-9a-f]{{2}})+) {_FIELDS}"
)
_STRIP_FAILED = re.compile(
    r"BF metadata-delegates-strip-failed ((?:[0-9a-f]{2})+) (E?[A-Za-z0-9_]+)"
)
_STRIP_DONE = re.compile(r"BF metadata-delegates-strip-done ([0-9]+) ([0-9]+)")
_RESTORE_LINE = re.compile(
    r"BF metadata-delegates-(restored|kept|restore-failed) "
    r"((?:[0-9a-f]{2})+|-) (\S+)"
)
_RESTORE_DONE = re.compile(
    r"BF metadata-delegates-restore-done ([0-9]+) ([0-9]+) ([0-9]+)"
)
_DISARM_ERROR = re.compile(r"BF metadata-delegates-disarm-error (\S+)")
_DISARM_ERRORS = {
    "python": "its Python failed root's check (gone, or not root's alone)",
    "xattr": "its Python has no way to change file attributes (no os.setxattr, "
    "and ctypes could not reach the C library's)",
}


def _disarm_error(text: str) -> str:
    """Why root's program stopped before its report, from its ``disarm-error``
    line, or ``""``."""
    m = _DISARM_ERROR.search(text)
    return _DISARM_ERRORS.get(m[1], f"it stopped ({m[1]})") if m else ""


def _shown(hex_path: str) -> str:
    from benchflow.sandbox.lockdown import _shown_path

    try:
        return _shown_path(bytes.fromhex(hex_path))
    except ValueError:
        return hex_path


@dataclass
class StripReport:
    """What one strip did: ``stripped`` entries (path, mode, owner, inode,
    link count, size, sha256, capability; ``earlier`` for one the record
    held already), ``failures`` (path, why), and ``problems`` when the report
    itself cannot be trusted."""

    stripped: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures and not self.problems


def read_strip(out: Any, code: Any, failed: str = "") -> StripReport:
    report = StripReport()
    if failed:
        report.problems.append(f"its exec failed ({failed})")
        return report
    text = out if isinstance(out, str) else ""
    done = None
    for raw in text.splitlines():
        if m := _STRIPPED.fullmatch(raw):
            report.stripped.append(
                {
                    "path": _shown(m[2]),
                    "mode": m[3],
                    "uid": int(m[4]),
                    "gid": int(m[5]),
                    "inode": int(m[6]),
                    "links": int(m[7]),
                    "size": int(m[8]),
                    "sha256": m[9],
                    **({"capability": m[10]} if m[10] != "-" else {}),
                    **({"earlier": True} if m[1] == "recorded" else {}),
                }
            )
        elif m := _STRIP_FAILED.fullmatch(raw):
            report.failures.append({"path": _shown(m[1]), "why": m[2]})
        elif m := _STRIP_DONE.fullmatch(raw):
            done = m
        elif raw.strip() and not _DISARM_ERROR.fullmatch(raw):
            report.problems.append(f"it printed a line it never prints ({raw[:120]!r})")
    if done is None:
        report.problems.append(_disarm_error(text) or "it did not report")
    expected = 7 if report.failures or report.problems else 0
    if not report.problems and (type(code) is not int or code != expected):
        report.problems.append(f"it ended with status {code!r}, not {expected}")
    return report


_LIVE_LINE = re.compile(
    r"BF metadata-delegates-live ([0-9]+) ([0-9]+) ((?:[0-9a-f]{2})+|-)"
)


@dataclass
class RestoreReport:
    """What one restore did: ``rows``, each file's outcome (``restored``,
    ``kept_stripped`` with why, or ``failed``: BenchFlow could not do it);
    ``live``, the processes outside no_new_privs that withheld the whole
    restore (pid, uid, comm)."""

    rows: list[dict[str, str]] = field(default_factory=list)
    live: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failed(self) -> list[dict[str, str]]:
        return [r for r in self.rows if r["outcome"] == "failed"]


def read_restore(out: Any, code: Any, failed: str = "") -> RestoreReport:
    """Read one restore (see :data:`DISARM_PROGRAM`). A failed exec, a report
    that did not finish and ``restore-failed`` rows are ``failed``:
    BenchFlow's own failure, never the agent's change."""
    report = RestoreReport()
    if failed:
        report.rows.append(
            {"path": "-", "outcome": "failed", "why": f"its exec failed ({failed})"}
        )
        return report
    text = out if isinstance(out, str) else ""
    done = None
    for raw in text.splitlines():
        if m := _RESTORE_LINE.fullmatch(raw):
            outcome = {
                "restored": "restored",
                "kept": "kept_stripped",
                "restore-failed": "failed",
            }[m[1]]
            report.rows.append(
                {
                    "path": _shown(m[2]) if m[2] != "-" else "-",
                    "outcome": outcome,
                    "why": m[3],
                }
            )
        elif m := _LIVE_LINE.fullmatch(raw):
            report.live.append(
                {
                    "pid": int(m[1]),
                    "uid": int(m[2]),
                    "comm": bytes.fromhex(m[3]).decode("utf-8", "replace")
                    if m[3] != "-"
                    else "",
                }
            )
        elif m := _RESTORE_DONE.fullmatch(raw):
            done = m
    if done is None or type(code) is not int or code not in (0, 7):
        stopped = _disarm_error(text)
        report.rows.append(
            {
                "path": "-",
                "outcome": "failed",
                "why": f"the restore did not report (status {code!r}"
                + (f"; {stopped})" if stopped else ")"),
            }
        )
    return report


# --- choosing root's Python -----------------------------------------------------

_PROBE = re.compile(
    r"BF delegate-python managed=((?:[0-9a-f]{2})+|-) staged=((?:[0-9a-f]{2})+|-) "
    r"arch=(x86_64|aarch64|other) libc=(gnu|musl|none) tar=([01])"
)


def probe_cmd() -> str:
    """Root's look for a Python BenchFlow put in the sandbox: its managed
    CPython (the model proxy's runtime) or one staged earlier, each only when
    its whole tree is root's alone (the staged tree must also match its pin);
    and, for staging, the machine, the C
    library the image's programs use, and whether it has tar. Prints ``BF
    delegate-python managed=<hex|-> staged=<hex|-> arch= libc= tar=``."""
    from benchflow.sandbox._staged_python_layer import tree_check
    from benchflow.sandbox.lockdown import _cloud_metadata_root_prelude

    managed = (
        f"for p in {shlex.quote(PYTHON_INSTALL_DIR)}/cpython-{PYTHON_VERSION}-linux-*"
        "/bin/python3.12; do "
        '[ -x "$p" ] || continue; '
        f"{{ {root_only_tree_check(PYTHON_INSTALL_DIR, 'managed Python')} ; }} "
        '2>/dev/null && { m=$(hx "$p"); break; }; done; '
    )
    staged = "".join(
        f'if [ "$s" = - ] && [ -x {shlex.quote(staged_python_bin(a, c))} ] && '
        f"{{ {tree_check(a, c)} ; }} "
        f"2>/dev/null; then s=$(hx {shlex.quote(staged_python_bin(a, c))}); fi; "
        for a, c in STAGED_PYTHON_SHA256
    )
    return (
        "( "
        + _cloud_metadata_root_prelude()
        + "hx() { printf '%s' \"$1\" | od -An -v -tx1 | tr -d ' \\n'; }; "
        + "m=-; s=-; "
        + managed
        + staged
        + "u=$(uname -m 2>/dev/null); case $u in x86_64|amd64) a=x86_64;; "
        "aarch64|arm64) a=aarch64;; *) a=other;; esac; "
        'c=none; [ -e "/lib/ld-musl-$a.so.1" ] && c=musl; '
        'if [ "$c" = none ]; then for l in /lib64/ld-linux-x86-64.so.2 '
        "/lib/ld-linux-x86-64.so.2 /lib/ld-linux-aarch64.so.1 "
        '/lib64/ld-linux-aarch64.so.1; do [ -e "$l" ] && c=gnu; done; fi; '
        "t=0; tool tar >/dev/null 2>&1 && t=1; "
        'printf "BF delegate-python managed=%s staged=%s arch=%s libc=%s tar=%s\\n" '
        '"$m" "$s" "$a" "$c" "$t" )'
    )


def stage_prepare_cmd(arch: str, libc: str) -> str:
    """Root makes the staging folder for *arch* and *libc* root's alone
    (0700, every folder above root's) and empty, before the upload."""
    from benchflow.sandbox.lockdown import _cloud_metadata_root_prelude

    d = shlex.quote(staged_python_dir(arch, libc))
    root = shlex.quote(STAGED_ROOT)
    return (
        "( "
        + _cloud_metadata_root_prelude()
        + f"mkdir -p {shlex.quote(RUNTIME_ROOT)} && chmod go-w {shlex.quote(RUNTIME_ROOT)} "
        f"&& mkdir -p {root} && chown 0:0 {root} && chmod 700 {root} "
        f"&& rm -rf {d} && mkdir -m 700 {d} && chown 0:0 {d} "
        f"&& walk {d} && [ -d \"$W\" ] && echo 'BF delegate-python prepared' )"
    )


def stage_unpack_cmd(arch: str, libc: str) -> str:
    """Root checks the uploaded build against its pin (where the image has a
    sha256 tool; the host checked it already), unpacks it with the image's
    custody-checked tar, makes every file root's and unwritable by others,
    removes the archive and checks the tree is root's alone."""
    from benchflow.sandbox.lockdown import _cloud_metadata_root_prelude

    d = staged_python_dir(arch, libc)
    q = shlex.quote(d)
    archive = shlex.quote(f"{d}/python.tar.gz")
    want = STAGED_PYTHON_SHA256[(arch, libc)]
    return (
        "( "
        + _cloud_metadata_root_prelude()
        + "bf_tar=$(tool tar) || { echo 'BF delegate-python error=tar'; exit 7; }; "
        "if bf_sum=$(tool sha256sum); then "
        f'got=$("$bf_sum" {archive} | cut -c1-64) || exit 7; '
        f'[ "$got" = {want} ] || {{ echo "BF delegate-python error=sha256 $got"; exit 7; }}; fi; '
        f'"$bf_tar" -xzf {archive} -C {q} || {{ echo "BF delegate-python error=unpack"; exit 7; }}; '
        f"rm -f {archive} && chown -R 0:0 {q}/python && chmod -R go-w {q}/python && "
        + root_only_tree_check(d, "BenchFlow's staged Python")
        + f" && [ -x {q}/python/bin/python3.12 ] && echo 'BF delegate-python staged' )"
    )


@dataclass
class DelegatePython:
    """Root's Python for the inventory, strip and restore: *path*, the
    *tree* to check before each use (None for PT's kit Python, which PT
    checks), and where it came from (``kit``, ``managed``, ``staged``)."""

    path: str
    source: str
    tree: str | None = None


async def find_python(env: Any) -> tuple[DelegatePython | None, str]:
    """Root's Python for the delegate inventory: PT's protected kit Python,
    else BenchFlow's managed CPython, else one staged earlier, else
    BenchFlow's pinned CPython staged now from the host's cache. ``(None,
    why)`` when none can be had."""
    from benchflow.sandbox.firewall_power import _protected_python
    from benchflow.sandbox.lockdown import _cloud_metadata_exec

    kit = await _protected_python(env)
    if kit:
        return DelegatePython(kit, "kit"), ""
    result, failed = await _cloud_metadata_exec(
        env, probe_cmd(), user="root", timeout_sec=120
    )
    out = getattr(result, "stdout", "")
    found = _PROBE.search(out if isinstance(out, str) else "")
    if failed or found is None:
        return None, f"the look for BenchFlow's Python failed ({failed or 'no report'})"
    if found[1] != "-":
        return DelegatePython(
            bytes.fromhex(found[1]).decode(), "managed", PYTHON_INSTALL_DIR
        ), ""
    if found[2] != "-":
        path = bytes.fromhex(found[2]).decode()
        return DelegatePython(path, "staged", path.rsplit("/python/bin/", 1)[0]), ""
    arch, libc, has_tar = found[3], found[4], found[5] == "1"
    if arch == "other" or libc == "none":
        return None, (
            f"the image's machine ({arch}) or C library ({libc}) has no build of "
            "BenchFlow's CPython"
        )
    if not has_tar:
        return None, "the image has no tar to unpack BenchFlow's CPython with"
    try:
        tarball = await asyncio.to_thread(host_tarball, arch, libc)
    except StagedPythonUnavailable as exc:
        return None, str(exc)
    return await _stage(env, tarball, arch, libc)


async def _stage(
    env: Any, tarball: Path, arch: str, libc: str
) -> tuple[DelegatePython | None, str]:
    from benchflow.sandbox.lockdown import _cloud_metadata_exec

    result, failed = await _cloud_metadata_exec(
        env, stage_prepare_cmd(arch, libc), user="root", timeout_sec=60
    )
    if failed or "BF delegate-python prepared" not in str(
        getattr(result, "stdout", "")
    ):
        return None, f"its staging folder could not be made ({failed or 'no report'})"
    from benchflow.trust.state import TRUSTED_TRANSFER, admit

    try:
        await env.upload_file(
            tarball,
            f"{staged_python_dir(arch, libc)}/python.tar.gz",
            admission=admit(
                env, "main", TRUSTED_TRANSFER, reason="BenchFlow's pinned CPython"
            ),
        )
    except Exception as exc:
        return None, f"its upload failed ({type(exc).__name__}: {exc})"[:300]
    result, failed = await _cloud_metadata_exec(
        env, stage_unpack_cmd(arch, libc), user="root", timeout_sec=_STAGE_EXEC_SEC
    )
    out = str(getattr(result, "stdout", "") or "")
    if failed or "BF delegate-python staged" not in out:
        reason = re.search(r"BF delegate-python error=(\S+)", out)
        return None, (
            f"it could not be unpacked ({failed or (reason[1] if reason else 'no report')})"
        )
    return DelegatePython(
        staged_python_bin(arch, libc), "staged", staged_python_dir(arch, libc)
    ), ""


# --- the host's memory of what was disarmed -------------------------------------

#: On the sandbox object: the Python, its tree and what one strip recorded,
#: so the restore before grading uses the same checked Python. In memory, so
#: nothing in the sandbox can change it.
DISARMED_ATTR = "_benchflow_delegates_disarmed"
#: On the sandbox object: the restore's outcome, for the trial's records.
RESTORED_ATTR = "_benchflow_delegates_restored"


def remember(env: Any, python: DelegatePython, stripped: list[dict[str, Any]]) -> None:
    state = getattr(env, DISARMED_ATTR, None)
    known = list(state["stripped"]) if isinstance(state, dict) else []
    seen = {(s["path"], s["inode"]) for s in known}
    known += [s for s in stripped if (s["path"], s["inode"]) not in seen]
    with contextlib.suppress(AttributeError, TypeError):
        setattr(
            env,
            DISARMED_ATTR,
            {"python": python.path, "tree": python.tree, "stripped": known},
        )


class DelegateRestoreError(VerifierInfrastructureError):
    """BenchFlow could not put back what the cloud metadata block disarmed:
    the trial is unscored (never 0), since grading would run on an image the
    agent did not change but BenchFlow left changed."""


async def restore_disarmed_delegates(env: Any) -> RestoreReport | None:
    """Before grading, after the kill of the agent's processes: put back what
    the delegate inventory took, on files whose content and standing are
    unchanged (see :data:`DISARM_PROGRAM`). Returns what it did, also kept on
    *env* (:data:`RESTORED_ATTR`) for the trial's records; None when nothing
    was disarmed.

    A file the agent changed, and every file while a process outside
    no_new_privs that is not root is alive, stays disarmed through grading,
    and the records name it. When BenchFlow itself fails (a timeout, a failed
    exec, its Python's custody, an I/O error) it raises
    :class:`DelegateRestoreError` after keeping the report on *env*: the
    verifier does not run and the trial is unscored.

    The host's memory (:data:`DISARMED_ATTR`) outlives the restore: a later
    grading in this sandbox (after a snapshot restore put the disarmed files
    and the record back, as a branch does) runs it again, and the record in
    the sandbox decides what is left to put back (nothing, once restored)."""
    from benchflow.sandbox.lockdown import _cloud_metadata_exec

    state = getattr(env, DISARMED_ATTR, None)
    if not isinstance(state, dict) or not state.get("stripped"):
        return None
    result, failed = await _cloud_metadata_exec(
        env,
        disarm_cmd(state["python"], "restore", tree=state.get("tree")),
        user="root",
        timeout_sec=_RESTORE_EXEC_SEC,
    )
    report = read_restore(
        getattr(result, "stdout", ""), getattr(result, "return_code", None), failed
    )
    with contextlib.suppress(AttributeError, TypeError):
        setattr(env, RESTORED_ATTR, report)
    if report.live:
        logger.warning(
            "Cloud metadata: nothing disarmed was put back before grading: "
            "processes outside no_new_privs are alive (%s); the files stay "
            "without their set-ID bits or capability while the verifier runs",
            ", ".join(
                f"pid {p['pid']} uid {p['uid']} {p['comm']!r}" for p in report.live
            ),
        )
    for row in report.rows:
        if row["outcome"] == "kept_stripped" and not report.live:
            logger.warning(
                "Cloud metadata: %s was not restored before grading (%s); it "
                "stays without its set-ID bits or capability while the verifier runs",
                row["path"],
                row["why"],
            )
    if report.failed:
        raise DelegateRestoreError(
            "BenchFlow could not put back the set-ID bits or file capabilities "
            "the cloud metadata block took for the agent's turn: "
            + "; ".join(f"{r['path']} ({r['why']})" for r in report.failed[:5])
        )
    return report
