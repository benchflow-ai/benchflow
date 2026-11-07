"""The ``taskmd`` verifier strategy: how a materialized draft 2 package is graded.

A materialized package's ``verifier/verifier.md`` selects this strategy
(``benchflow.taskmd.materialize``). ``Verifier.verify()`` calls
:func:`verify_taskmd`, which follows docs/runtime/judging.md's order:

1. **Offline.** When the verifier's network is ``"none"``, it runs offline: a
   separate verifier sandbox is started without a network, and a shared
   verifier's container is taken offline (``iptables``) before ``test.sh`` runs
   when the agent's run left it online, or the trial is not scored.
2. **Kept copy.** With model judges, the saved outputs (``[sandbox] outputs``,
   or the working folder) are copied to the host before any script runs:
   what the judges read, and ``submission_tree``.
3. **Script verifier.** ``test.sh`` runs in the task's working folder (not in
   ``/verifier``). Without a rubric its ``reward.txt``/``reward.json`` is the
   reward. With one, any reward file it writes is set aside
   (``script-reward.*``) and ``/logs/verifier/ctrf.json`` is read.
4. **Model judges** (``benchflow.taskmd.judging``), after the tests, skipping
   model-judged criteria once a test gate failed (``lazy``).
5. **Review.** ``review.json`` (review-1), and ``reward.json``/``reward.txt``
   with the headline reward, ``strict``, and ``partial``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import errno
import glob
import json
import logging
import os
import posixpath
import re
import shlex
import shutil
import stat
import tarfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tarfile import InvalidHeaderError, ReadError, TruncatedHeaderError
from typing import Any, Self

from benchflow._utils.reachable_write import (
    PrivateStorageError,
    ReachablePath,
    ReachableWriteError,
    UnsafeDownloadEntryError,
    _private_lookup_io,
    _remove_beneath,
    create_private_file,
    destination,
    private_output,
    private_stat,
    private_storage_io,
    private_temporary_directory,
    staging_for,
)
from benchflow._verifier_env import pin_verifier_env
from benchflow.sandbox._base import system_shell_exec
from benchflow.sandbox._download import RemoteFileAbsentError
from benchflow.sandbox._exec_output import diagnostic_result, exec_raw
from benchflow.sandbox.protocol import ExecTimeoutError
from benchflow.task.verifier_errors import (
    VerifierExecutionError,
    VerifierInfrastructureError,
    VerifierOutputParseError,
    verifier_failure_reason,
    verifier_infrastructure_failure,
)
from benchflow.taskmd import grading, judging
from benchflow.taskmd import trajectory as traj
from benchflow.taskmd._util import listed, table
from benchflow.taskmd._vendor import judgeprompt as jp
from benchflow.taskmd._vendor import taskmd as ref
from benchflow.taskmd.materialize import (
    judge_package_dir,
    shared_rubrics,
    taskmd_metadata,
)

logger = logging.getLogger(__name__)

JUDGE_DIR = "taskmd-judge"
# One line per judge session (every model call, its tools and their results) and one per
# model-judged criterion (its samples and verdict), beside review.json.
JUDGMENTS = "judgments.jsonl"
RUNNER_UID = 65534
KEPT_CAPS = {"bytes": 104_857_600, "files": 10_000, "depth": 32}


class TaskMdVerifierError(RuntimeError):
    """Grading could not finish; the scoring boundary decides the outcome."""


class TaskMdEvidenceError(TaskMdVerifierError, VerifierOutputParseError):
    """Verifier-writable output cannot be read or published safely."""


class TaskMdInfrastructureError(TaskMdVerifierError, VerifierInfrastructureError):
    """The judge's controller, backend or provider failed."""


class TaskMdJudgeInfrastructureError(TaskMdInfrastructureError, judging.JudgeError):
    """A judge failure, retaining the direct caller's JudgeError contract."""


# The first line of every shell BenchFlow's taskmd verifier runs as root. The
# image's PATH can start with a folder inside a kept path (``ENV
# PATH=/app/.venv/bin:$PATH`` with /app kept), so a submission could leave a
# ``chown`` or ``tar`` there that root would run; root's shell looks only in the
# system folders (benchflow.sandbox._base.SYSTEM_PATH).
ROOT_PATH = "PATH=/usr/sbin:/usr/bin:/sbin:/bin; export PATH"


def _as_root(script: str) -> str:
    """``script`` with :data:`ROOT_PATH` as its first line."""
    return f"{ROOT_PATH}\n{script}"


def system_shell(sandbox: Any) -> bool:
    """Whether the sandbox's ``exec`` can start root's shell by absolute path (``system_shell=True``)."""
    return getattr(sandbox, "supports_system_shell", False) is True


async def _await_through_cancellation(task: asyncio.Task[Any]) -> Any:
    """``task``'s result once it ends, however often the caller is cancelled
    meanwhile; the task's own cancellation still propagates."""
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


async def root_exec(
    sandbox: Any, script: str, *, timeout_sec: int, env: dict[str, str] | None = None
) -> Any:
    """Run ``script`` (which starts with :data:`ROOT_PATH`) as root.

    The shell itself is looked up, and its libraries loaded, before any line of
    the script runs, so the backend must start the shell as ``/bin/sh``
    by absolute path, started with the image's loader variables set empty, with
    PATH set to the system folders first (``exec(..., system_shell=True)``).
    Refuse unsupported backends before any probe, capture or cleanup command;
    sanitizing inside the script is too late to protect shell startup.
    """
    if not system_shell(sandbox):
        raise TaskMdInfrastructureError(
            "judge capture requires protected root shell execution; "
            "use a sandbox backend that supports system_shell"
        )
    kwargs: dict[str, Any] = {}
    if env is not None:
        kwargs["env"] = env
    try:
        return await exec_raw(
            system_shell_exec,
            sandbox,
            script,
            what="The task.md verifier",
            timeout_sec=timeout_sec,
            **kwargs,
        )
    except (TimeoutError, ExecTimeoutError):
        raise
    except Exception as exc:
        raise VerifierExecutionError(
            "Sandbox judge exec failed: " + verifier_failure_reason(exc)
        ) from exc


# Where the offline check tries to connect when iptables cannot take the
# container offline.
OFFLINE_PROBES = ("1.1.1.1/443", "8.8.8.8/53")


def _offline_script(probes: tuple[str, ...] = OFFLINE_PROBES) -> str:
    """Take this container offline unless it already is. Prints the outcome; exits nonzero when it cannot.

    The network check's bash starts with BASH_ENV empty: this runs as root
    after the agent's turn, and the image's BASH_ENV can name a file the
    agent wrote, which would run as root and could fail the check on purpose
    (an online container reported offline).
    """
    reach = " || ".join(
        f"BASH_ENV= ENV= bash -c 'exec 3<>/dev/tcp/{probe}' 2>/dev/null"
        for probe in probes
    )
    return rf"""
if [ -z "$(awk 'NR > 2 {{ sub(/:.*/, "", $1); if ($1 != "lo") print $1 }}' /proc/net/dev 2>/dev/null)" ]; then
  echo "offline: no network interface but lo"; exit 0
fi
if command -v iptables >/dev/null 2>&1 && iptables -I OUTPUT 1 ! -o lo -j REJECT 2>/dev/null; then
  command -v ip6tables >/dev/null 2>&1 && ip6tables -I OUTPUT 1 ! -o lo -j REJECT 2>/dev/null
  echo "offline: iptables rejects every packet leaving by an interface but lo"; exit 0
fi
if command -v bash >/dev/null 2>&1; then
  if {reach}; then
    echo "online: the container reaches the internet and iptables is not available"; exit 1
  fi
  echo "offline: the platform blocks egress"; exit 0
fi
echo "unknown: no iptables and no bash to check the network"; exit 1
"""


async def take_offline(sandbox: Any, probes: tuple[str, ...] = OFFLINE_PROBES) -> str:
    """Make sure the verifier runs without a network; return how, or raise."""
    from benchflow.sandbox.lockdown import _exec_return_code

    result = await root_exec(
        sandbox,
        _as_root(_offline_script(probes)),
        timeout_sec=60,
        env={"BASH_ENV": "", "ENV": ""},
    )
    output = (getattr(result, "stdout", "") or "").strip()
    if _exec_return_code(result) != 0:
        raise TaskMdVerifierError(
            f'[verifier] network is "none", and BenchFlow could not take the verifier offline: {output}'
        )
    return output


def kept_paths(config: dict[str, Any]) -> list[str]:
    """What the runtime saves: each declared output's path, or the working folder."""
    sandbox = table(config.get("sandbox"))
    paths = [
        p for p in (jp.restored_path(o) for o in listed(sandbox.get("outputs"))) if p
    ]
    if paths:
        return paths
    workdir = sandbox.get("workdir")
    return [jp.normalize_path(workdir)] if isinstance(workdir, str) else []


def _discard_copy(path: ReachablePath) -> None:
    """Remove an entry only through its retained, no-follow parent."""
    with path._folders(private=True) as held, _private_lookup_io("capture discard"):
        parent = held[-1]
        if isinstance(parent, Path):
            entry = parent / path.name
            try:
                info = entry.lstat()
            except FileNotFoundError:
                return
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                # Python's Windows rmtree removes junctions without descending.
                shutil.rmtree(entry)
            else:
                entry.unlink()
        else:
            _remove_beneath(parent, path.name, where=path.details_message())


def _link_kept(source: Path, dest: Path) -> None:
    """Link only the retained regular inode, refusing a concurrent name swap."""
    original, target = destination(source), destination(dest)
    with (
        original.open("rb") as stream,
        original._folders() as src,
        target._folders(private=True) as dst,
    ):
        expected = os.fstat(stream.fileno())
        with private_storage_io("capture hard link create"):
            if isinstance(src[-1], Path):
                os.link(
                    src[-1] / original.name,
                    dst[-1] / target.name,
                    follow_symlinks=False,
                )
            else:
                os.link(
                    original.name,
                    target.name,
                    src_dir_fd=src[-1],
                    dst_dir_fd=dst[-1],
                    follow_symlinks=False,
                )
        actual = private_stat(target)
        if not stat.S_ISREG(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        ):
            target.unlink(missing_ok=True, private=True)
            raise OSError("saved output hard-link source changed during publication")


def _within_caps(root: Path | ReachablePath) -> str | None:
    root = destination(root)
    total, files, depth = 0, 0, 0
    if root.is_file():
        return None if root.stat().st_size <= KEPT_CAPS["bytes"] else "over 100 MB"
    for path in root.rglob("*"):
        if path.is_file():
            files += 1
            total += path.stat().st_size
            depth = max(depth, len(path.relative_to(root).parts))
    if total > KEPT_CAPS["bytes"]:
        return f"{total} bytes, over the 100 MB cap"
    if files > KEPT_CAPS["files"]:
        return f"{files} files, over the 10,000 cap"
    if depth > KEPT_CAPS["depth"]:
        return f"a depth of {depth}, over the cap of 32"
    return None


# The most lines of the sandbox's mode record copy_kept reads per kept path (folders and files).
MODE_RECORD_LINES = 2 * KEPT_CAPS["files"] + 2


def _probe_script(path: str) -> str:
    """What ``path`` is, then each folder's and regular file's mode as the sandbox has it.

    The record lines are ``d <octal mode> <path>`` and ``f <octal mode> <path>``
    (``find`` and ``stat -c``, which GNU and busybox both have; a link is not
    followed). A download keeps neither: the Daytona SDK writes every file with
    the host's umask (0644) and skips a folder that holds no file.
    """
    q = shlex.quote(path)
    return (
        f"if [ -L {q} ]; then echo link; elif [ -d {q} ]; then echo dir; "
        f"elif [ -f {q} ]; then echo file; elif [ -e {q} ]; then echo special; else echo none; fi\n"
        f"if [ -d {q} ] && [ ! -L {q} ]; then "
        f"{{ find {q} -type d -exec stat -c 'd %a %n' {{}} + ; find {q} -type f -exec stat -c 'f %a %n' {{}} + ; }} "
        f"2>/dev/null | head -n {MODE_RECORD_LINES}; "
        f"elif [ -f {q} ] && [ ! -L {q} ]; then stat -c 'f %a %n' {q} 2>/dev/null; fi\n"
    )


def parse_mode_record(path: str, lines: Iterable[str]) -> dict[str, tuple[str, int]]:
    """``{absolute path: ("d" or "f", mode)}`` from the probe's record lines, for ``path`` and what is under it.

    A line that does not parse, or names a path outside ``path`` (a name with
    a newline can split into such a line), is skipped; the mode keeps only the
    permission bits (no setuid, setgid or sticky bit).
    """
    root = PurePosixPath(path)
    found: dict[str, tuple[str, int]] = {}
    for line in lines:
        kind, _, rest = line.partition(" ")
        mode, _, name = rest.partition(" ")
        if kind not in ("d", "f") or not re.fullmatch(r"[0-7]{1,6}", mode):
            continue
        entry = PurePosixPath(name)
        if (
            not name.startswith("/")
            or str(entry) != name.rstrip("/")
            or ".." in entry.parts
        ):
            continue
        if entry != root and root not in entry.parents:
            continue
        found[str(entry)] = (kind, int(mode, 8) & 0o777)
    return found


def _os_error_text(exc: OSError) -> str:
    """``strerror: filename``, or only what the error has of the two."""
    text = exc.strerror or str(exc)
    return f"{text}: {exc.filename}" if exc.filename is not None else text


# Where root in the sandbox writes its tar of a kept folder for copy_kept (removed after).
COPY_DIR = "/run/taskmd-judge-copy"
COPY_TAR = f"{COPY_DIR}/copy.tar"
# The largest such tar BenchFlow takes: the byte cap plus room for every header the file
# cap allows (a GNU long name is another header and its blocks). Past it, the folder is over the caps.
COPY_TAR_LIMIT = KEPT_CAPS["bytes"] + MODE_RECORD_LINES * 2048


def _copy_script(path: str, exclude: tuple[str, ...] = ()) -> str:
    """Root's tar of the kept folder ``path`` at :data:`COPY_TAR`; prints ``BF copy <bytes> <tar's exit code>``.

    The tar stops at :data:`COPY_TAR_LIMIT` bytes, so an oversized folder
    never fills the sandbox's disk.
    """
    rel = shlex.quote("./" + path.strip("/"))
    root = glob.escape("./" + path.strip("/"))
    patterns = []
    for rule in exclude:
        # Match relative paths and basenames, like ordinary artifact capture.
        # Anchor patterns below this output so an ancestor cannot be excluded.
        patterns.append(root + "/" + rule)
        if "/" not in rule:
            patterns.append(root + "/*/" + rule)
    options = " ".join(shlex.quote("--exclude=" + p) for p in patterns)
    if patterns:
        # GNU tar needs explicit anchoring; BusyBox matches the full path for
        # these slash-containing patterns and has no --anchored option.
        options = "$bf_tar_flags " + options + " "
    flags = (
        'bf_tar_flags=""\ncase "$(tar --help 2>&1)" in *--anchored*) bf_tar_flags="--wildcards --anchored";; esac\n'
        if patterns
        else ""
    )
    return f"""
rm -rf {COPY_DIR} && mkdir -m 700 {COPY_DIR} || exit 1
{flags}\
{{ tar -C / {options}-cf - {rel} 2>/dev/null; echo $? > {COPY_DIR}/rc; }} | head -c {COPY_TAR_LIMIT} > {COPY_TAR}
echo "BF copy $(( $(wc -c < {COPY_TAR}) )) $(cat {COPY_DIR}/rc)"
"""


# What BenchFlow reads of the sandbox's tar before tarfile parses it (:func:`_scan_tar_headers`):
# the most bytes one GNU long name or link name, or one PAX header, may declare; all of
# them together (the header room in COPY_TAR_LIMIT); and the most 512-byte headers.
TAR_EXTENDED_BYTES = 65_536
TAR_EXTENDED_TOTAL = COPY_TAR_LIMIT - KEPT_CAPS["bytes"]
TAR_HEADERS = 4 * MODE_RECORD_LINES  # 80,008 headers in total, all types included.
# The first bytes of a gzip, bzip2, xz or zstd stream: root's ``tar -cf`` writes none of them.
COMPRESSED = {
    b"\x1f\x8b": "gzip",
    b"BZh": "bzip2",
    b"\xfd7zXZ\x00": "xz",
    b"\x28\xb5\x2f\xfd": "zstd",
}


def _tar_number(field: bytes) -> int | None:
    """A header's number field as tarfile reads it (octal, or GNU base-256); None if it does not parse."""
    if field[:1] in (b"\x80", b"\xff"):
        n = int.from_bytes(field[1:], "big")
        return n - 256 ** (len(field) - 1) if field[:1] == b"\xff" else n
    try:
        return int(field.split(b"\0", 1)[0].decode("ascii").strip() or "0", 8)
    except (UnicodeDecodeError, ValueError):
        return None


def _tar_blocks(size: int) -> int:
    return -(-size // 512) * 512


def _scan_tar_headers(archive: Path, path: str) -> None:
    """Check the sandbox's tar header by header, before tarfile parses it, reading no file's data.

    tarfile reads a GNU long name or a PAX header whole before it returns the
    member, so a header that declares gigabytes is read into memory before
    any name or cap is checked. Each such header's declared size, their
    total and the number of headers are capped here
    (:data:`TAR_EXTENDED_BYTES`, :data:`TAR_EXTENDED_TOTAL`,
    :data:`TAR_HEADERS`), and a header tarfile would read differently from
    what this scan skips (a sparse file, an old-style folder, a PAX ``size``
    or ``GNU.sparse`` record) is refused. Global PAX headers are refused too:
    tarfile accumulates their keys and copies them into every later member,
    so the encoded-byte caps would not bound memory. Over a cap or malformed stops the judging.
    """

    def stop(why: str) -> TaskMdVerifierError:
        return TaskMdEvidenceError(
            f"the sandbox's archive of the saved output {path} {why}"
        )

    end = archive.stat().st_size
    headers = extended = 0
    with destination(archive).open("rb") as tar:
        while True:
            block = tar.read(512)
            if len(block) < 512:
                raise stop(f"is cut short at byte {tar.tell() - len(block)}")
            if block.count(0) == 512:
                # A genuine end-of-archive block, not a missing or partial header.
                return
            headers += 1
            if headers > TAR_HEADERS:
                raise stop(f"has more than {TAR_HEADERS} headers")
            checksum = _tar_number(block[148:156])
            if checksum not in (
                256 + sum(block[:148]) + sum(block[156:]),
                256 + sum(b - 256 if b > 127 else b for b in block[:148] + block[156:]),
            ):
                raise stop(f"has a malformed header at byte {tar.tell() - 512}")
            size = _tar_number(block[124:136])
            if size is None or size < 0:
                raise stop(f"has a malformed size at byte {tar.tell() - 512}")
            kind = block[156:157]
            if kind == b"g":
                raise stop("has a global PAX header, which BenchFlow does not read")
            if kind == b"\0" and block[:100].split(b"\0", 1)[0].endswith(b"/"):
                # An old-style folder: tarfile reads it as a folder or a file depending
                # on its version and the header before it, so where its data ends is unclear.
                raise stop(
                    "has an old-style folder header, which BenchFlow does not read"
                )
            if kind == b"S":
                raise stop("has a sparse file, which BenchFlow does not read")
            if kind in (b"1", b"2", b"3", b"4", b"5", b"6"):
                continue  # tarfile reads no data after a link, folder or special file
            extended_header = kind in (b"L", b"K", b"x", b"X")
            if extended_header:
                if size > TAR_EXTENDED_BYTES:
                    raise stop(
                        f"has an extended header of {size} bytes, over the cap of {TAR_EXTENDED_BYTES}"
                    )
                extended += size
                if extended > TAR_EXTENDED_TOTAL:
                    raise stop(
                        f"has more than {TAR_EXTENDED_TOTAL} bytes of extended headers"
                    )
            if tar.tell() + _tar_blocks(size) > end:
                raise stop(f"is cut short at byte {tar.tell() - 512}")
            if not extended_header:
                # A regular file's data, or that of a type tarfile does not know, which it skips too.
                tar.seek(_tar_blocks(size), os.SEEK_CUR)
                continue
            payload = tar.read(_tar_blocks(size))
            # Matched anywhere in the header, so a name holding these bytes is refused too.
            if kind not in (b"L", b"K") and (
                b" size=" in payload or b"GNU.sparse" in payload
            ):
                raise stop(
                    "has a PAX size or sparse record, which BenchFlow does not read"
                )


def _kept_name(parts: tuple[str, ...]) -> None:
    """Reject submitted names the capture filesystem cannot represent."""
    for part in parts:
        try:
            size = len(os.fsencode(part))
        except UnicodeError as exc:
            raise TaskMdEvidenceError("unrepresentable saved output filename") from exc
        # These archives are portable across controllers. POSIX filesystems
        # commonly limit each encoded component to 255 bytes; Win32's bound
        # is no larger in UTF-16 code units for names accepted here.
        if size > 255:
            raise TaskMdEvidenceError("saved output filename exceeds 255 bytes")


def _real_folders(fs_root: Path, parts: tuple[str, ...], path: str) -> Path:
    """``fs_root`` joined with ``parts``, each folder on the way made (0700) or checked to be a real folder, not a link."""
    _kept_name(parts)
    here = fs_root
    capability = destination(fs_root / ".anchor").parent
    assert isinstance(capability, ReachablePath)
    capability.mkdir(parents=True, exist_ok=True, private=True)
    for part in parts:
        here = here / part
        capability = capability / part
        try:
            capability.mkdir(parents=True, exist_ok=True, private=True)
        except OSError as exc:
            if exc.errno == errno.ENAMETOOLONG:
                raise TaskMdEvidenceError(
                    "unrepresentable saved output folder name"
                ) from exc
            if not isinstance(
                exc, (ReachableWriteError, FileExistsError, NotADirectoryError)
            ) or verifier_infrastructure_failure(exc):
                raise TaskMdInfrastructureError(
                    "BenchFlow could not create its private evidence folder: "
                    + verifier_failure_reason(exc)
                ) from exc
            raise TaskMdEvidenceError(
                f"BenchFlow's copy of the saved output {path} has {here.relative_to(fs_root)} "
                "where the sandbox's archive needs a folder"
            ) from exc
    return here


def _set_mtime(dest: Path, mtime: float) -> None:
    # As tarfile: a time the host cannot set is left as it is.
    with contextlib.suppress(OSError, OverflowError, ValueError):
        os.utime(dest, (mtime, mtime), follow_symlinks=False)


class _StrictTarInfo(tarfile.TarInfo):
    """Do not let tarfile treat a malformed later header as end-of-archive."""

    @classmethod
    def fromtarfile(cls, tarfile: tarfile.TarFile) -> Self:
        try:
            return super().fromtarfile(tarfile)
        except (InvalidHeaderError, TruncatedHeaderError) as exc:
            # TarFile.next swallows these after the first member. ReadError instead
            # stops the whole pre-extraction walk, including malformed PAX records.
            # EOFHeaderError (a zero block) must retain its normal meaning.
            raise ReadError(str(exc)) from exc


def unpack_kept(
    archive: Path, path: str, fs_root: Path, refused: dict[str, str]
) -> str | None:
    """Extract the sandbox's tar of the kept folder ``path`` into ``fs_root``; returns why it is over the caps, or None.

    The tar comes from the sandbox, whose ``tar`` the submission may have
    replaced, so it is read as untrusted, and every check runs before
    anything is written. A compressed tar, a header over the caps
    (:func:`_scan_tar_headers`), a member outside ``path`` (an absolute name,
    ``..``) or a name that comes twice (after normalising) stops the judging
    by name. A link, a special file, or a hard link to anything but a regular
    file of this tar kept before it is left out and listed in ``refused``.
    BenchFlow writes each file itself (``O_NOFOLLOW``, every folder on the
    way a real folder under ``fs_root``) and makes each hard link to
    a retained, identity-checked file it wrote from this tar, so tarfile's extraction
    (whose hard-link fallback looks names up in the whole tar) never runs.
    Folders are made 0700 while their contents are written and get their own
    mode (with owner rwx) at the end, so a folder at 0555 or 0000 that holds
    files extracts as BenchFlow's own user, which ``docker cp`` cannot do
    (go-archive sets each folder's mode before writing into it). Files get
    owner read. The sandbox's modes for the stash are in the probe's record.
    """
    rel = path.strip("/")
    with destination(archive).open("rb") as raw:
        head = raw.read(6)
    for magic, kind in COMPRESSED.items():
        if head.startswith(magic):
            raise TaskMdEvidenceError(
                f"the sandbox's archive of the saved output {path} is compressed ({kind}); "
                "BenchFlow reads only an uncompressed tar"
            )
    _scan_tar_headers(archive, path)
    keep: list[tuple[str, tarfile.TarInfo]] = []
    from benchflow._utils.archive_paths import ArchivePaths, archive_path

    inventory = ArchivePaths()
    files: set[str] = set()
    total = 0
    with (
        destination(archive).open("rb") as raw,
        tarfile.open(fileobj=raw, mode="r:", tarinfo=_StrictTarInfo) as tar,
    ):
        for member in tar:
            name = posixpath.normpath(member.name)
            if member.name.startswith("/") or (
                name != rel and not name.startswith(rel + "/")
            ):
                raise TaskMdEvidenceError(
                    f"the sandbox's archive of the saved output {path} names {member.name!r}, outside it"
                )
            try:
                name = str(inventory.add(member.name, directory=member.isdir()))
                _kept_name(PurePosixPath(name).parts)
                if member.islnk():
                    archive_path(member.linkname)
                    _kept_name(PurePosixPath(member.linkname).parts)
            except ValueError as exc:
                raise TaskMdEvidenceError(
                    f"unsafe saved output archive: {exc}"
                ) from exc
            if member.islnk():
                if posixpath.normpath(member.linkname) not in files:
                    refused["/" + name] = (
                        "a hard link to anything but a regular file saved before it is never saved"
                    )
                    continue
            elif not (member.isfile() or member.isdir()):
                refused["/" + name] = "a symbolic link or special file is never saved"
                continue
            if member.isfile():
                files.add(name)
                total += member.size
            keep.append((name, member))
        if total > KEPT_CAPS["bytes"]:
            return f"{total} bytes, over the 100 MB cap"
        if len(files) > KEPT_CAPS["files"]:
            return f"{len(files)} files, over the 10,000 cap"

        written: dict[str, Path] = {}
        folders: list[tuple[Path, tarfile.TarInfo]] = []
        _real_folders(fs_root, (), path)
        for name, member in keep:
            parts = PurePosixPath(name).parts
            if member.isdir():
                folders.append((_real_folders(fs_root, parts, path), member))
                continue
            dest = _real_folders(fs_root, parts[:-1], path) / parts[-1]
            if member.islnk():
                source = written.get(posixpath.normpath(member.linkname))
                if source is None:
                    raise TaskMdEvidenceError(
                        f"the sandbox's archive of the saved output {path} links {name!r} "
                        "to a file BenchFlow did not write"
                    )
                target = destination(dest)
                with contextlib.suppress(FileNotFoundError):
                    if stat.S_ISREG(private_stat(target).st_mode):
                        target.unlink(private=True)
                _link_kept(source, dest)
                continue
            data = tar.extractfile(member)
            if data is None:
                raise TaskMdEvidenceError(
                    f"the sandbox's archive of the saved output {path} has no data for {name!r}"
                )
            # The extraction tree is private to the controller and the archive
            # can express only regular files/folders (or links to files we wrote).
            # Exclusive creation also supplies no-follow on Windows, which has
            # no O_NOFOLLOW/fchmod. Overlapping kept roots may replace only a
            # regular file from an earlier capture, never a reparse point.
            try:
                existing = private_stat(destination(dest))
            except FileNotFoundError:
                pass
            else:
                if (
                    not stat.S_ISREG(existing.st_mode)
                    or getattr(existing, "st_file_attributes", 0) & 0x400
                ):
                    raise TaskMdEvidenceError(
                        "the saved output would replace an unsafe file"
                    )
                destination(dest).unlink(private=True)
            with data, private_output(create_private_file(dest)) as out:
                shutil.copyfileobj(data, out)
                if hasattr(os, "fchmod"):
                    with private_storage_io("capture file mode"):
                        os.fchmod(out.fileno(), (member.mode & 0o777) | 0o400)
                    with contextlib.suppress(OSError, OverflowError, ValueError):
                        os.utime(out.fileno(), (member.mtime, member.mtime))
                else:
                    # The Windows file and ancestor handles deny replacement.
                    with private_storage_io("capture file mode"):
                        os.chmod(dest, (member.mode & 0o777) | 0o400)
                    _set_mtime(dest, member.mtime)
            written[name] = dest
        # After everything is written, so no later write changes a folder's time.
        for folder, member in reversed(folders):
            capability = destination(folder / ".anchor").parent
            assert isinstance(capability, ReachablePath)
            with capability._folders(directory=True, private=True) as held:
                if isinstance(held[-1], Path):
                    with private_storage_io("capture folder mode"):
                        os.chmod(held[-1], (member.mode & 0o777) | 0o700)
                    _set_mtime(held[-1], member.mtime)
                else:
                    with private_storage_io("capture folder mode"):
                        os.fchmod(held[-1], (member.mode & 0o777) | 0o700)
                    with contextlib.suppress(OSError, OverflowError, ValueError):
                        os.utime(held[-1], (member.mtime, member.mtime))
    return None


async def _tar_kept_dir(
    sandbox: Any,
    path: str,
    fs_root: Path,
    refused: dict[str, str],
    exclude: tuple[str, ...] = (),
) -> tuple[bool, str | None]:
    """Copy the kept folder ``path`` through root's tar in the sandbox (:func:`unpack_kept`).

    Returns whether it was tarred, and if so why it is over the caps (None
    when copied). Not tarred: the sandbox could not make the tar (no ``tar``
    in the image, a backend whose ``download_file`` cannot fetch it); the
    caller then stops judging rather than attempting an unbounded download.
    """
    try:
        made = await root_exec(
            sandbox, _as_root(_copy_script(path, exclude)), timeout_sec=300
        )
        found = re.search(r"^BF copy (\d+) (\d+)$", made.stdout or "", re.MULTILINE)
        if found is None:
            return False, None
        size, code = int(found.group(1)), int(found.group(2))
        if size >= COPY_TAR_LIMIT:
            return True, f"over the caps (a tar of more than {COPY_TAR_LIMIT} bytes)"
        if code != 0:
            return False, None
        with private_temporary_directory(prefix="benchflow-judge-copy-") as tmp:
            archive = Path(tmp) / "copy.tar"
            try:
                await sandbox.download_file(COPY_TAR, archive)
            except VerifierOutputParseError:
                raise
            except (
                UnsafeDownloadEntryError,
                RemoteFileAbsentError,
                ReachableWriteError,
            ) as exc:
                raise TaskMdEvidenceError(
                    "the saved output archive is missing or unsafe"
                ) from exc
            except Exception as exc:
                raise TaskMdInfrastructureError(
                    f"The bounded archive of {path} could not be downloaded: "
                    + verifier_failure_reason(exc)
                ) from exc
            try:
                if not archive.is_file() or archive.stat().st_size >= COPY_TAR_LIMIT:
                    return False, None
                return True, unpack_kept(archive, path, fs_root, refused)
            except tarfile.TarError as exc:
                raise TaskMdEvidenceError(
                    f"BenchFlow could not unpack the sandbox's archive of the saved output {path} ({exc})"
                ) from exc
            except OSError as exc:
                if not verifier_infrastructure_failure(exc):
                    raise TaskMdEvidenceError(
                        "the saved output archive could not be read or unpacked safely"
                    ) from exc
                raise TaskMdInfrastructureError(
                    f"BenchFlow could not write the private capture of {path}: "
                    + verifier_failure_reason(exc)
                ) from exc
    finally:
        await root_exec(sandbox, _as_root(f"rm -rf {COPY_DIR}"), timeout_sec=60)


async def copy_kept(
    sandbox: Any,
    paths: list[str],
    fs_root: Path,
    modes: dict[str, tuple[str, int]] | None = None,
    *,
    exclusions: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, str]:
    """Copy the saved outputs to ``fs_root`` at their absolute paths. Returns refused path -> why.

    ``modes`` (when given) gets each kept folder's and file's mode as the
    sandbox had it (:func:`parse_mode_record`), for the runner's stash: the
    host copy's modes are the download's, not the submission's.
    """
    refused: dict[str, str] = {}
    for path in paths:
        probe = await root_exec(sandbox, _as_root(_probe_script(path)), timeout_sec=60)
        lines = (getattr(probe, "stdout", "") or "").splitlines()
        kind = lines[0].strip() if lines else ""
        record = parse_mode_record(path, lines[1:])
        target = fs_root / path.lstrip("/")
        if kind == "none":
            continue
        if kind in ("link", "special"):
            refused[path] = (
                f"a {'symbolic link' if kind == 'link' else 'special file'} is never saved"
            )
            continue
        # Prepare the held parents, including the anchor when it is the
        # immediate parent, without treating the anchor as a destination.
        target_cap = destination(target)
        with _private_lookup_io("capture anchor create"):
            target_cap._anchor.ensure()
        with target_cap._folders(parents=True, private=True):
            pass
        # Files and directories use the same bounded staging archive. A direct
        # backend download has no transfer cap and cannot be a safe fallback.
        exclude = (exclusions or {}).get(path, ())
        tarred, over = await _tar_kept_dir(sandbox, path, fs_root, refused, exclude)
        if not tarred:
            raise TaskMdEvidenceError(
                f"the saved output {path} could not be captured with a bounded tar"
            )
        if over:
            refused[path] = over
            continue
        # Only regular files and folders are kept; a link or special file anywhere is refused.
        target_cap = destination(target)
        for found in (
            sorted(target_cap.rglob("*"), reverse=True) if target_cap.is_dir() else []
        ):
            if found.is_symlink() or not (found.is_file() or found.is_dir()):
                found.unlink(missing_ok=True, private=True)
                refused[
                    str(PurePosixPath(path) / found.relative_to(target_cap).as_posix())
                ] = "a symbolic link or special file is never saved"
        why = _within_caps(target)
        if why:
            _discard_copy(target_cap)
            refused[path] = why
        elif modes is not None:
            modes.update(record)
    return refused


# The runner's stash of the saved outputs, restored before every ``run`` call.
KEPT_TAR = "/run/taskmd-judge/kept.tar"


def custody_archive(
    custody: Path | None,
    kept: list[str],
    dest: Path,
    modes: dict[str, tuple[str, int]] | None = None,
) -> bool:
    """Write to ``dest`` a tar of the judges' host copy of the saved outputs; return whether it holds any.

    ``custody`` is the folder :func:`copy_kept` filled before ``test.sh`` ran
    (what ``read`` serves); each kept path found there goes in, at its path
    relative to ``/``, owned by root. Only regular files and folders are in
    it, as only those were kept, and the second name of a hard-linked file
    (``docker cp`` keeps hard links; cargo links ``target/debug/<bin>`` to
    ``deps/``), as a link to the first. ``modes`` is the sandbox's record
    (:func:`copy_kept`): each entry gets the mode the sandbox had, not the
    download's (on Daytona every file is 0644, so a program would lose its
    execute bit), and a folder the download skipped (one that held no file)
    is added back empty.
    """
    modes = modes or {}
    names: set[str] = set()

    def owned_by_root(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if info.islnk():
            # tarfile writes a hard link's second name as a link to the first one it added.
            if info.linkname not in names:
                return None
        elif not (info.isfile() or info.isdir()):
            return None
        recorded = modes.get("/" + info.name)
        if recorded is not None and recorded[0] == ("d" if info.isdir() else "f"):
            info.mode = recorded[1]
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        names.add(info.name)
        return info

    inodes: dict[tuple[int, int], str] = {}

    def add(tar, source: ReachablePath, name: str):
        status = source.stat()
        info = tarfile.TarInfo(name)
        info.mode = stat.S_IMODE(status.st_mode)
        info.mtime = status.st_mtime
        if stat.S_ISDIR(status.st_mode):
            info.type = tarfile.DIRTYPE
            tar.addfile(owned_by_root(info))
            for child in sorted(source.iterdir()):
                add(tar, child, f"{name}/{child.name}")
        elif stat.S_ISREG(status.st_mode):
            try:
                opened = source.open("rb")
            except OSError as exc:
                raise OSError(
                    exc.errno, exc.strerror, source.details_message()
                ) from exc
            with opened as stream:
                status = os.fstat(stream.fileno())
                identity = (status.st_dev, status.st_ino)
                info.mode = stat.S_IMODE(status.st_mode)
                info.mtime = status.st_mtime
                if status.st_nlink > 1 and identity in inodes:
                    info.type = tarfile.LNKTYPE
                    info.linkname = inodes[identity]
                    tar.addfile(owned_by_root(info))
                else:
                    info.size = status.st_size
                    tar.addfile(owned_by_root(info), stream)
                    inodes[identity] = name

    added = False
    with (
        private_output(create_private_file(dest)) as output,
        tarfile.open(fileobj=output, mode="w", format=tarfile.GNU_FORMAT) as tar,
    ):
        if custody is None:
            return False
        roots = []
        for path in kept:
            source = destination(custody / path.lstrip("/"))
            if source.is_symlink():
                continue
            roots.append(path.rstrip("/") or "/")
            if not source.exists():
                continue
            add(tar, source, path.lstrip("/"))
            added = True
        for name, (kind, mode) in sorted(modes.items()):
            arcname = name.lstrip("/")
            if kind != "d" or arcname in names:
                continue
            if not any(name == r or name.startswith(r + "/") for r in roots):
                continue
            # Only where the host copy has nothing (a link refused in the copy stays out).
            if destination(custody / arcname).exists():
                continue
            info = tarfile.TarInfo(arcname)
            info.type = tarfile.DIRTYPE
            info.mode = mode
            info.mtime = int(time.time())
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            tar.addfile(info)
            names.add(arcname)
            added = True
    return added


# The verifier's files and any oracle, removed from the judge's sandbox before its first call.
REMOVED = (
    "/verifier",
    "/tests",
    "/oracle",
    "/solution",
    "/oracle_backup",
    "/solution_oracle_backup",
)
# The verifier's and the solver's logs. BenchFlow bind-mounts them from the
# host, so they are hidden from the runner's uid, never deleted.
HIDDEN = ("/logs/verifier", "/logs/agent")


def _hide_script(
    root: str = "",
    mountinfo: str = "/proc/self/mountinfo",
    hidden: tuple[str, ...] = HIDDEN,
) -> str:
    """Shell that takes the verifier's files out of the judge's sandbox.

    Each path of :data:`REMOVED` is deleted, unless it is or holds a mount
    point: deleting through a bind mount deletes the host's files, so such a
    path is hidden instead (mode 700; the runner's uid cannot enter it), as is
    each path of ``hidden``. The hidden paths' modes are saved in
    ``/run/taskmd-judge/modes`` for :func:`restore_hidden`. ``root``
    prefixes every path, so tests can run the script against a folder.
    """
    state = shlex.quote(f"{root}/run/taskmd-judge")
    removed = " ".join(shlex.quote(root + p) for p in REMOVED)
    hide_paths = " ".join(shlex.quote(root + p) for p in hidden)
    return f"""
set -e
mkdir -p {state} && chmod 700 {state}
: > {state}/modes
mounts=$(awk '{{print $5}}' {shlex.quote(mountinfo)} 2>/dev/null || true)
held() {{ printf '%s\\n' "$mounts" | awk -v p="$1" '$0 == p || index($0, p "/") == 1 {{ f = 1 }} END {{ exit !f }}'; }}
hide() {{
  m=$(stat -c %a "$1" 2>/dev/null || stat -f %Lp "$1")
  echo "$m $1" >> {state}/modes
  chmod 700 "$1"
}}
for p in {removed}; do
  if [ -L "$p" ]; then rm -f "$p"
  elif [ -e "$p" ]; then
    if held "$p"; then hide "$p"; else rm -rf "$p"; fi
  fi
done
for p in {hide_paths}; do
  if [ -e "$p" ]; then hide "$p"; fi
done
"""


def _restore_script(root: str = "") -> str:
    """Shell that gives the hidden paths back their modes."""
    modes = shlex.quote(f"{root}/run/taskmd-judge/modes")
    return f"""
if [ -f {modes} ]; then
  while read -r m p; do chmod "$m" "$p" 2>/dev/null || true; done < {modes}
  : > {modes}
fi
"""


async def restore_hidden(sandbox: Any) -> None:
    """Give the paths hidden from the judge's runner their modes back (best effort; a no-op when none were)."""
    if not system_shell(sandbox):
        return  # the runner refused this sandbox before it hid anything
    try:
        await root_exec(sandbox, _as_root(_restore_script()), timeout_sec=60)
    except Exception as exc:  # the sandbox is discarded next; no score depends on this
        logger.warning("Could not restore the paths hidden from the judge: %s", exc)


# Where the runner looks for the tool that drops root: system folders only,
# never the image's PATH, which a kept output can sit on.
DROP_DIRS = ("/usr/bin", "/bin", "/usr/sbin", "/sbin")


# POSIX shell (dash, busybox ash) that prints how the runner drops root:
# ``BF drop setpriv <path>``, ``BF drop runuser <path> <name>``,
# ``BF drop su <path> <name>``, or ``BF drop none``. A tool counts only when
# it is a regular file owned by root that neither group nor others can write,
# in a folder of the same kind, or a link from such a folder to such a file
# under one of DROP_DIRS (busybox's ``su`` on Alpine is one); ``setpriv``
# counts only when it takes ``--reuid`` (busybox's does not). ``<name>`` is
# uid 65534's name in /etc/passwd, which ``runuser`` and ``su`` need.
def drop_probe(root: str = "", passwd: str = "/etc/passwd") -> str:
    """The probe; ``root`` prefixes the system folders and ``passwd`` names the user file, so tests can run it on a folder."""
    q = shlex.quote(root) if root else ""
    dirs = " ".join(shlex.quote(root + d) for d in DROP_DIRS)
    systems = "|".join(f"{q}{d}/*" for d in DROP_DIRS)
    return f"""
bf_safe_mode() {{ case "$1" in ''|*[!0-7]*) return 1;; esac; [ $(( 0$1 & 022 )) -eq 0 ]; }}
bf_root_only() {{
  set -- $(stat -L -c '%u %a' "$1" 2>/dev/null)
  [ "${{1:-x}}" = 0 ] && bf_safe_mode "${{2:-}}"
}}
bf_trusted() {{
  bf_p=$1
  bf_root_only "$(dirname "$bf_p")" || return 1
  if [ -L "$bf_p" ]; then
    bf_p=$(readlink -f "$bf_p" 2>/dev/null) || return 1
    case "$bf_p" in {systems}) ;; *) return 1;; esac
    bf_root_only "$(dirname "$bf_p")" || return 1
  fi
  [ -f "$bf_p" ] && [ ! -L "$bf_p" ] && bf_root_only "$bf_p"
}}
bf_name=$(awk -F: '$3 == {RUNNER_UID} {{ print $1; exit }}' {shlex.quote(passwd)} 2>/dev/null)
bf_found=""
for bf_d in {dirs}; do
  bf_t=$bf_d/setpriv
  if [ -e "$bf_t" ] && bf_trusted "$bf_t" && "$bf_t" --help 2>&1 | grep -q -e --reuid; then bf_found="setpriv $bf_t"; break; fi
done
if [ -z "$bf_found" ] && [ -n "$bf_name" ]; then
  for bf_tool in runuser su; do
    for bf_d in {dirs}; do
      bf_t=$bf_d/$bf_tool
      if [ -e "$bf_t" ] && bf_trusted "$bf_t"; then bf_found="$bf_tool $bf_t $bf_name"; break 2; fi
    done
  done
fi
echo "BF drop ${{bf_found:-none}}"
"""


DROP_PROBE = drop_probe()

_DROP_PATH = re.compile(r"/[A-Za-z0-9/_.+-]+")
_DROP_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")


def parse_drop(stdout: str) -> tuple[str, str, str | None] | None:
    """The tool the setup's ``BF drop`` line names: (tool, path, user name), or None when there is none."""
    line = next(
        (
            ln.split()[2:]
            for ln in reversed((stdout or "").splitlines())
            if ln.startswith("BF drop ")
        ),
        None,
    )
    if not line or line == ["none"]:
        return None
    tool, path = line[0], line[1] if len(line) > 1 else ""
    if not _DROP_PATH.fullmatch(path):
        return None
    if tool == "setpriv" and len(line) == 2:
        return tool, path, None
    if tool in ("runuser", "su") and len(line) == 3 and _DROP_NAME.fullmatch(line[2]):
        return tool, path, line[2]
    return None


def drop_prefix(drop: tuple[str, str, str | None], inner: str) -> str:
    """``inner`` (shell words, run without a shell of its own) as uid 65534, through the drop tool.

    setpriv and runuser exec ``inner``'s words directly; su hands them to
    ``/bin/sh -c`` as one quoted string, so ``inner`` holds nothing the judge
    wrote (the command itself is a file, :data:`COMMAND_FILE`).
    """
    tool, path, name = drop
    if tool == "setpriv":
        return f"{path} --reuid={RUNNER_UID} --regid={RUNNER_UID} --clear-groups --no-new-privs {inner}"
    if tool == "runuser":
        return f"{path} -u {name} -- {inner}"
    return f"{path} -s /bin/sh {name} -c {shlex.quote('exec ' + inner)}"


# The judge's command, as a file only root can change, in a folder made afresh before each call.
COMMAND_DIR = "/run/taskmd-judge-cmd"
COMMAND_FILE = f"{COMMAND_DIR}/command"
# Rounds of the leftover sweep before the runner gives up on processes that will not die.
SWEEP_ROUNDS = 200


def sweep_script(
    drop: tuple[str, str, str | None], proc: str = "/proc", rounds: int = SWEEP_ROUNDS
) -> str:
    """Shell that kills every process of uid 65534, until a round finds none.

    Each round sends ``kill -9 -1`` as uid 65534 (the one step a forking
    process cannot outrun), then walks ``/proc`` with shell builtins only, so
    a full process table cannot stop it, and kills each live process of that
    uid it finds (zombies are skipped: they are dead already). From the third
    round it pauses a tenth of a second. It prints ``BF survivors`` and
    returns 1 when processes are still there after ``rounds`` (:data:`SWEEP_ROUNDS`).
    PID 1 is never counted: a container's main process ignores SIGKILL sent
    from inside, and :meth:`SandboxRunner.setup` refuses an image whose
    processes already run as uid 65534 (:func:`busy_script`).
    ``proc`` is ``/proc`` but in tests.
    """
    kill_all = drop_prefix(drop, "/bin/sh -c 'kill -9 -1'")
    return f"""
bf_sweep() {{
  bf_r=0
  while [ $bf_r -lt {int(rounds)} ]; do
    bf_r=$((bf_r + 1))
    [ $bf_r -gt 2 ] && {{ sleep 0.1 2>/dev/null || sleep 1; }}
    {kill_all} >/dev/null 2>&1
    bf_n=0
    for bf_d in {shlex.quote(proc)}/[0-9]*; do
      [ "${{bf_d##*/}}" = 1 ] && continue
      bf_u=""; bf_s=""
      {{ while read -r bf_k bf_v bf_rest; do
          case "$bf_k" in Uid:) bf_u=$bf_v;; State:) bf_s=$bf_v;; esac
        done < "$bf_d/status"; }} 2>/dev/null
      if [ "$bf_u" = {RUNNER_UID} ] && [ "$bf_s" != Z ]; then
        kill -9 "${{bf_d##*/}}" 2>/dev/null
        bf_n=$((bf_n + 1))
      fi
    done
    [ $bf_n -eq 0 ] && return 0
  done
  echo "BF survivors"
  return 1
}}
"""


def busy_script(proc: str = "/proc") -> str:
    """Shell that prints ``BF busy <pid>:<name> ...`` when live processes of uid 65534 exist.

    Run at setup, before the runner has started anything: what it finds is
    the image's own (its USER is 65534, so the container's main process is
    one) or a leftover of ``test.sh``. The sweep would kill those, or spin on
    PID 1, which ignores SIGKILL from inside its container, so the runner
    refuses instead. Shell builtins only, as in :func:`sweep_script`.
    """
    return f"""
bf_busy=""
for bf_d in {shlex.quote(proc)}/[0-9]*; do
  bf_u=""; bf_s=""; bf_c=""
  {{ while read -r bf_k bf_v bf_rest; do
      case "$bf_k" in Name:) bf_c=$bf_v;; Uid:) bf_u=$bf_v;; State:) bf_s=$bf_v;; esac
    done < "$bf_d/status"; }} 2>/dev/null
  if [ "$bf_u" = {RUNNER_UID} ] && [ "$bf_s" != Z ]; then bf_busy="$bf_busy ${{bf_d##*/}}:$bf_c"; fi
done
[ -z "$bf_busy" ] || echo "BF busy$bf_busy"
"""


_BUSY_ENTRY = re.compile(r"([0-9]+):([A-Za-z0-9_.+-]{0,15})")


def parse_busy(stdout: str) -> list[str]:
    """The processes the setup's ``BF busy`` line names, as ``pid <pid> (<name>)`` (at most 5), or []."""
    line = next(
        (ln for ln in (stdout or "").splitlines() if ln.startswith("BF busy")), None
    )
    if line is None:
        return []
    found = [
        f"pid {m.group(1)}" + (f" ({m.group(2)})" if m.group(2) else "")
        for m in (_BUSY_ENTRY.fullmatch(w) for w in line.split()[2:])
        if m
    ]
    if not found:
        return ["processes it could not name"]
    if len(found) > 5:
        return [*found[:5], f"{len(found) - 5} more"]
    return found


@dataclass
class SandboxRunner:
    """judge-tools@1's ``run``: each call in a fresh copy of the submission's environment.

    The runner is the separate verifier sandbox: a fresh container of the
    task's image holding only the kept copy. Before the first call BenchFlow
    removes ``/verifier``, ``/tests``, and any oracle from it, hides
    ``/logs/verifier`` and ``/logs/agent`` (the verifier's and the solver's
    logs, mounted from the host) from the runner's uid unless they hold a
    kept path or a view, stashes the judges' host copy of the saved outputs
    (``custody``: made before ``test.sh`` ran, what ``read`` serves, so
    ``read`` and ``run`` see the same files whatever ``test.sh`` changed in
    the sandbox), and finds the system tool
    that drops root (``setpriv``, ``runuser``, or ``su``; an image with none
    is refused, since a judge's command never runs as root). A sandbox that
    already runs processes of uid 65534 (an image whose USER is 65534) is
    refused too, since the runner needs that uid to itself; so is a sandbox
    backend that cannot start root's shell by absolute path (only Docker and
    Daytona can). Before every call
    it kills whatever uid 65534 left running, restores the kept copy (a failed
    restore stops the judging), a fresh working folder and HOME, writes the
    command to a file only root can change, then runs it with ``bash
    --noprofile --norc`` as uid 65534 with no network, stdin at /dev/null,
    and stdout and stderr merged, under the call's timeout, and kills whatever
    the command left until nothing of that uid is running. Every shell it runs
    as root is ``/bin/sh`` by absolute path, started with the image's loader
    variables set empty, and starts with :data:`ROOT_PATH`, so root never runs
    a program from the image's PATH or loads a library its LD_LIBRARY_PATH or
    LD_PRELOAD names (either can lead into a kept path). What persists
    between calls outside the kept copy is the container's other writable
    paths: a fresh container per call is not provided. :func:`restore_hidden`
    gives the hidden paths their modes back.

    The command's environment is ``PATH`` (the image's, without the kept
    paths), ``HOME``, ``LANG=C.UTF-8`` and ``PYTHONNOUSERSITE=1``;
    ``PYTHONSAFEPATH`` only when the task's ``[verifier] env`` sets it
    (``safepath``), since it keeps ``python3 /app/main.py`` from importing
    its own sibling modules.
    """

    sandbox: Any
    kept: list[str]
    views: dict[str, bytes]
    path: str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    drop: tuple[str, str, str | None] | None = None
    safepath: str | None = None
    custody: Path | None = None
    modes: dict[str, tuple[str, int]] | None = None
    ready: bool = False

    async def setup(self) -> None:
        if not system_shell(self.sandbox):
            raise TaskMdInfrastructureError(
                f"the judge's runner cannot run in this sandbox ({type(self.sandbox).__name__}): it cannot "
                "start root's shell by absolute path, so the image's PATH, which can lead into a kept path, "
                "would choose the shell root runs. Use the Docker or Daytona sandbox"
            )
        # A log folder holding a kept path or a view stays visible: the judge reads those.
        hidden = tuple(
            h
            for h in HIDDEN
            if not any(
                k == h or k.startswith(h + "/") for k in [*self.kept, *self.views]
            )
        )
        script = f"""{_hide_script(hidden=hidden)}
cd /
rm -f {KEPT_TAR} && : > {KEPT_TAR} && chmod 600 {KEPT_TAR}
tr '\\0' '\\n' < /proc/1/environ 2>/dev/null | sed -n 's/^PATH=//p' | head -1 > /run/taskmd-judge/path || true
set +e
{busy_script()}
{DROP_PROBE}
"""
        result = await root_exec(self.sandbox, _as_root(script), timeout_sec=300)
        from benchflow.sandbox.lockdown import _exec_return_code

        if _exec_return_code(result) != 0:
            raise TaskMdVerifierError(
                f"the judge's runner could not be set up: {(diagnostic_result(result).stderr.render() or diagnostic_result(result).stdout.render() or '')[-300:]}"
            )
        busy = parse_busy(result.stdout or "")
        if busy:
            raise TaskMdVerifierError(
                f"the judge's runner needs uid {RUNNER_UID} to itself, and the verifier sandbox already "
                f"runs processes of that uid ({', '.join(busy)}): the task image's USER is probably "
                f"{RUNNER_UID} (nobody), which the container's main process runs as. Set the image's USER "
                f"to another user, or stop what test.sh left running as uid {RUNNER_UID}"
            )
        self.drop = parse_drop(result.stdout or "")
        if self.drop is None:
            raise TaskMdVerifierError(
                "the judge's runner cannot drop root in this image: it has no setpriv that takes --reuid, "
                f"and no runuser or su (with a user of uid {RUNNER_UID}), owned by root in "
                f"{', '.join(DROP_DIRS)}; BenchFlow never runs a judge's command as root. "
                "Add util-linux (it has setpriv and runuser) to the task's image"
            )
        await self._stash()
        path = await root_exec(
            self.sandbox, _as_root("cat /run/taskmd-judge/path"), timeout_sec=30
        )
        image_path = (path.stdout or "").strip()
        if image_path:
            keep = [
                entry
                for entry in image_path.split(":")
                if entry
                and not any(
                    entry == k or entry.startswith(k.rstrip("/") + "/")
                    for k in [*self.kept, "/tmp/taskmd-judge"]
                )
            ]
            self.path = ":".join(keep) or self.path
        for view_path, data in self.views.items():
            await root_exec(
                self.sandbox,
                _as_root(
                    f"mkdir -p {shlex.quote(str(PurePosixPath(view_path).parent))} && printf %s {shlex.quote(base64.b64encode(data).decode())} | base64 -d > {shlex.quote(view_path)} && chmod 644 {shlex.quote(view_path)}"
                ),
                timeout_sec=60,
            )
        self.ready = True

    async def _stash(self) -> None:
        """Put the custody copy's tar at :data:`KEPT_TAR` (root's, mode 600); without one it stays empty."""
        with private_temporary_directory(prefix="benchflow-judge-kept-") as tmp:
            archive = Path(tmp) / "kept.tar"
            try:
                if not custody_archive(self.custody, self.kept, archive, self.modes):
                    return
            except PrivateStorageError:
                raise
            except OSError as exc:
                raise TaskMdEvidenceError(
                    "the judge's runner could not archive the saved outputs "
                    f"({_os_error_text(exc)})"
                ) from exc
            with private_storage_io("custody archive stat"):
                size = archive.stat().st_size
            from benchflow.trust.state import IngressKind, admit

            admission = admit(self.sandbox, "main", IngressKind.AGENT_DERIVED_UPLOAD)
            try:
                await self.sandbox.upload_file(archive, KEPT_TAR, admission=admission)
            except VerifierOutputParseError:
                # A verified unsafe destination already carries evidence provenance.
                raise
            except Exception as exc:
                raise TaskMdInfrastructureError(
                    "the judge's runner could not upload the saved outputs: "
                    + verifier_failure_reason(exc)
                ) from exc
        stashed = await root_exec(
            self.sandbox,
            _as_root(
                f"[ -f {KEPT_TAR} ] && [ ! -L {KEPT_TAR} ] && chown 0:0 {KEPT_TAR} && chmod 600 {KEPT_TAR} "
                f"&& [ $(( $(wc -c < {KEPT_TAR}) )) -eq {size} ]"
            ),
            timeout_sec=60,
        )
        from benchflow.sandbox.lockdown import _exec_return_code

        if _exec_return_code(stashed) != 0:
            raise TaskMdVerifierError(
                "the judge's runner could not stash the saved outputs in the verifier sandbox "
                f"({KEPT_TAR} is not the {size} bytes BenchFlow sent)"
            )

    def command_script(self, command: str, timeout: int) -> str:
        """The shell one ``run`` call executes as root in the runner's sandbox."""
        if self.drop is None:
            raise TaskMdVerifierError(
                "the judge's runner has no way to drop root, so it runs nothing"
            )
        kept = " ".join(shlex.quote(p) for p in self.kept)
        encoded = base64.b64encode(command.encode("utf-8")).decode()
        env = (
            f"PATH={shlex.quote(self.path)} HOME=/tmp/taskmd-judge/home LANG=C.UTF-8 PYTHONNOUSERSITE=1"
            + (f" PYTHONSAFEPATH={shlex.quote(self.safepath)}" if self.safepath else "")
            # The verifier's invariants, as on every path.
            + "".join(
                f" {k}={shlex.quote(v)}" for k, v in pin_verifier_env(None).items()
            )
        )
        run = drop_prefix(
            self.drop,
            f"env -i {env} timeout -k 5 {int(timeout)} bash --noprofile --norc {COMMAND_FILE}",
        )
        u = f"{RUNNER_UID}:{RUNNER_UID}"
        return f"""{ROOT_PATH}
{sweep_script(self.drop)}
bf_sweep || exit 0
for p in {kept}; do rm -rf "$p"; done
if [ -s {KEPT_TAR} ] && ! tar -C / -xpf {KEPT_TAR} 2>/run/taskmd-judge/restore.err; then
  echo "BF restore-failed $(head -c 200 /run/taskmd-judge/restore.err | tr '\\n' ' ')"; exit 0
fi
for p in {kept}; do
  if [ -e "$p" ] && ! chown -R {u} "$p"; then echo "BF prepare-failed chown"; exit 0; fi
done
rm -rf /tmp/taskmd-judge {COMMAND_DIR} \\
  && mkdir -p /tmp/taskmd-judge/work /tmp/taskmd-judge/home {COMMAND_DIR} \\
  && chown -R {u} /tmp/taskmd-judge && chmod 755 {COMMAND_DIR} \\
  && printf %s {shlex.quote(encoded)} | base64 -d > {COMMAND_FILE} \\
  && chmod 444 {COMMAND_FILE} && [ $(( $(wc -c < {COMMAND_FILE}) )) -eq {len(command.encode("utf-8"))} ] \\
  || {{ echo "BF prepare-failed"; exit 0; }}
cd /tmp/taskmd-judge/work
{run} </dev/null >/run/taskmd-judge/out 2>&1
code=$?
cd /
bf_sweep || exit 0
size=$(wc -c < /run/taskmd-judge/out)
echo "$code $size"
if [ "$size" -le 16384 ]; then base64 -w0 /run/taskmd-judge/out; echo; else head -c 8192 /run/taskmd-judge/out | base64 -w0; echo; tail -c 8192 /run/taskmd-judge/out | base64 -w0; echo; fi
"""

    async def __call__(
        self, command: str, timeout: int
    ) -> tuple[int | None, bool, bytes | traj.CappedOutput]:
        if not self.ready:
            await self.setup()
        assert self.drop is not None
        script = self.command_script(command, timeout)
        # Judge commands can execute the submitted code too. Remember this
        # on the host, just as for test.sh, before handing off the command.
        self.sandbox._benchflow_test_command_issued = True
        command_task = asyncio.create_task(
            root_exec(self.sandbox, script, timeout_sec=int(timeout) + 60)
        )
        try:
            result = await asyncio.shield(command_task)
        except asyncio.CancelledError:
            # A cancelled exec client need not stop its remote command. Await
            # the bounded root wrapper before the final sweep, so a queued or
            # still-preparing wrapper cannot launch a process after that sweep.
            # A caller may cancel repeatedly while its timeout and teardown
            # race: both awaits stay shielded until their task ends, and a
            # sweep that fails is reported, never a clean cancellation.
            try:
                await _await_through_cancellation(command_task)
            finally:
                cleaned = await _await_through_cancellation(
                    asyncio.create_task(
                        root_exec(
                            self.sandbox,
                            _as_root(sweep_script(self.drop) + "\nbf_sweep"),
                            timeout_sec=60,
                        )
                    )
                )
                if cleaned.return_code != 0:
                    raise TaskMdVerifierError(
                        "the cancelled judge command could not be stopped"
                    ) from None
            raise
        except (TimeoutError, ExecTimeoutError):
            raise
        except Exception as exc:
            raise VerifierExecutionError(
                "Sandbox judge-command exec failed: " + verifier_failure_reason(exc)
            ) from exc
        lines = (result.stdout or "").splitlines()
        if lines and lines[0].startswith("BF "):
            why = {
                "BF survivors": f"processes of uid {RUNNER_UID} would not die",
                "BF restore-failed": "the saved outputs could not be restored",
                "BF prepare-failed": "the command's folders could not be prepared",
            }
            reason = next(
                (v for k, v in why.items() if lines[0].startswith(k)), lines[0]
            )
            raise TaskMdVerifierError(
                f"the judge's runner stopped before its result: {reason} ({lines[0][:300]})"
            )
        try:
            code_text, size_text = lines[0].split()
            code, size = int(code_text), int(size_text)
        except (IndexError, ValueError) as exc:
            raise TaskMdVerifierError(
                f"the judge's runner failed: {(diagnostic_result(result).stderr.render() or diagnostic_result(result).stdout.render() or '')[-300:]}"
            ) from exc
        output: bytes | traj.CappedOutput
        if size <= traj.SHELL_CAP:
            output = base64.b64decode(lines[1] if len(lines) > 1 else "")
        else:
            # Only the size of the middle is kept: rebuilding it on the host would
            # take as much memory as the command printed (gigabytes for a loop).
            output = traj.CappedOutput(
                head=base64.b64decode(lines[1]),
                tail=base64.b64decode(lines[2] if len(lines) > 2 else ""),
                size=size,
            )
        return code, code == 124, output


def _host_write(
    path: Path | ReachablePath, data: str | bytes, *, append: bool = False
) -> None:
    """Write (or append) ``data`` to ``path`` in the trial's verifier folder, never through a link.

    The verifier folder is bind-mounted into the verifier sandbox (mode 777),
    so ``test.sh``, what it runs, or a judge's command that can see it can
    leave a link there (``judgments.jsonl -> ~/.bashrc``) or a FIFO in place
    of a file BenchFlow writes. The content is written privately and placed
    by one rename through the held folder (reachable_write, R4-1), so no link
    or FIFO at the name is ever opened; an unreadable, linked or special entry means
    the trial scores 0 with a warning and nothing is written.
    """
    try:
        target = destination(path)
        if append:
            target.append(data)
        else:
            target.write(data, refuse_special=True)
    except OSError as exc:
        if isinstance(exc, ReachableWriteError) and not verifier_infrastructure_failure(
            exc
        ):
            raise TaskMdEvidenceError(
                f"BenchFlow did not write {path.name}: unsafe output entry ({exc})"
            ) from exc
        raise TaskMdInfrastructureError(
            f"BenchFlow could not write judge file {path.name}: "
            + verifier_failure_reason(exc)
        ) from exc


def _write_json(path: Path | ReachablePath, data: Any) -> None:
    _host_write(path, json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")


def _append_jsonl(path: Path | ReachablePath, data: Any) -> None:
    _host_write(
        path, json.dumps(data, allow_nan=False, default=str) + "\n", append=True
    )


def _real_dirs(*paths: Path) -> None:
    """Each path must be a folder, not a link to one: they sit in the verifier folder the sandbox can write."""
    for path in paths:
        if path.is_symlink() or not path.is_dir():
            raise TaskMdEvidenceError(
                f"{path.name} in the verifier folder was replaced by something in the verifier sandbox "
                "(it is no longer a folder BenchFlow made), so the judges do not run"
            )


def _safepath(config: dict[str, Any]) -> str | None:
    """``PYTHONSAFEPATH`` for the judge's commands: only what the task's ``[verifier] env`` sets, if anything."""
    value = table(table(config.get("verifier")).get("env")).get("PYTHONSAFEPATH")
    return value if isinstance(value, str) and value else None


async def verify_taskmd(verifier: Any, strategy: Any) -> Any:
    """Grade a materialized draft 2 package (``Verifier.verify()``'s ``taskmd`` strategy).

    The judges' copy of the saved outputs (what ``read``, file citations and
    the llm evidence read) is kept on the host in a private temporary folder,
    never in the verifier folder: the verifier sandbox mounts that folder
    read-write, so test.sh, which runs as root after the copy is made, could
    swap a copied file for a link to a host file there. When grading ends the
    copy and session records are published together to ``verifier/taskmd-judge/``
    by renaming the completed private directory, never by writing shared descendants.
    """
    # A sibling of the verifier mount is private to the controller and normally
    # on the same filesystem, so publication needs no recursive destination copy.
    target = verifier._rollout_paths.verifier_dir / JUDGE_DIR
    with staging_for(target, same_filesystem=True) as host_copy:
        try:
            return await _verify_taskmd(verifier, strategy, host_copy / "fs")
        finally:
            _publish_judge_copy(host_copy / "fs", target)


def _publish_judge_copy(fs_root: Path, judge_dir: ReachablePath) -> None:
    """Publish a completed private tree with directory-entry renames only; best effort.

    A soft-verify agent can still rename anything below the verifier mount.
    Never open a file there, even after checking its ancestors: keep every
    write private, then rename through held source/destination directory handles.
    Old records are moved into private custody too, for no-follow cleanup there.
    Unsupported filesystems refuse publication rather than falling back to copy.
    """
    try:
        if not destination(fs_root).is_dir():
            return
        if os.rename not in os.supports_dir_fd:
            raise OSError("safe judge publication requires directory-relative rename")
        private = fs_root.parent / JUDGE_DIR
        private_cap = destination(private)
        private_cap.mkdir(exist_ok=True)
        destination(fs_root).rename(private_cap / "fs")
        judge_dir.publish(private_cap)
    except OSError as exc:
        logger.warning(
            "The judges' copy of the saved outputs was not published: %s", exc
        )


async def _verify_taskmd(verifier: Any, strategy: Any, fs_root: Path) -> Any:
    from benchflow.task.verifier_errors import VerifierResult

    cfg = dict(strategy.config)
    task_dir = Path(verifier._task.paths.task_dir)
    meta = taskmd_metadata(task_dir)
    if meta is None:
        raise TaskMdVerifierError(
            f"{task_dir} is not a materialized task.md package (no metadata.taskmd)"
        )
    config = table(meta.get("config"))
    paths = verifier._rollout_paths
    # Session JSON has the same live-writer threat as evidence publication.
    # Keep it beside the private evidence until the final directory rename.
    judge_dir = fs_root.parent / JUDGE_DIR
    sandbox = verifier._sandbox
    offline_note = None
    if cfg.get("offline"):
        offline_note = await take_offline(sandbox)
    workdir = cfg.get("workdir") if isinstance(cfg.get("workdir"), str) else None
    grading_kind = cfg.get("grading")
    has_script = isinstance(cfg.get("command"), str)

    script_timeout = cfg.get("script_timeout")
    script_timeout = (
        float(script_timeout) if isinstance(script_timeout, (int, float)) else None
    )
    if grading_kind != "rubric":
        if not has_script:
            raise TaskMdVerifierError("the package has no test.sh and no rubric")
        return await verifier._verify_test_script(
            strategy=None,
            cwd=workdir,
            script_timeout_sec=script_timeout,
            parse_rewards=True,
        )

    pkg = judge_package_dir(task_dir)
    shared = shared_rubrics(task_dir, meta)
    document = ref.parse(pkg)
    rubric = document.rubric or {}
    criteria = jp.merged_criteria(rubric, shared)
    model_roles = {c.get("judge") for c in criteria} & {"llm", "agent"}
    if model_roles and not isinstance(table(config.get("sandbox")).get("workdir"), str):
        effective = getattr(verifier, "taskmd_workdir", None)
        if not isinstance(effective, str):
            # Direct Verifier callers have no rollout to carry the resolved cwd.
            probe = await root_exec(sandbox, _as_root("pwd -P"), timeout_sec=60)
            effective = (probe.stdout or "").strip()
        if not effective.startswith("/") or "\n" in effective:
            raise TaskMdVerifierError(
                "the task.md judge could not resolve the runtime working directory"
            )
        config = {
            **config,
            "sandbox": {
                **table(config.get("sandbox")),
                "workdir": jp.normalize_path(effective),
            },
        }
    _discard_copy(destination(judge_dir))
    (paths.verifier_dir / JUDGMENTS).unlink(missing_ok=True)
    refused: dict[str, str] = {}
    modes: dict[str, tuple[str, int]] = {}
    kept = kept_paths(config)
    if model_roles:
        destination(judge_dir).mkdir(parents=True, private=True)
        destination(fs_root).mkdir(private=True)
        exclusions = {
            path: tuple(output.get("exclude") or ())
            for output in listed(table(config.get("sandbox")).get("outputs"))
            if isinstance(output, dict)
            and (path := jp.restored_path(output))
            and output.get("exclude")
        }
        refused = await copy_kept(
            sandbox,
            kept,
            fs_root,
            modes,
            **({"exclusions": exclusions} if exclusions else {}),
        )

    return_code = None
    before_test = getattr(verifier, "before_test", None)
    if not has_script and model_roles and before_test is not None:
        # No test command runs, so the services the task keeps go on here,
        # after the copy above and before the judges' commands (K2-R1-A-2).
        from benchflow.task.paths import SandboxPaths

        await before_test(str(SandboxPaths().verifier_code_dir))
    if has_script:
        await verifier._verify_test_script(
            strategy=None,
            cwd=workdir,
            script_timeout_sec=script_timeout,
            parse_rewards=False,
        )
        return_code = getattr(verifier, "test_return_code", None)
    for name in ("reward.txt", "reward.json"):
        path = paths.verifier_dir / name
        if path.exists():
            path.rename_entry(f"script-{name}")
    ctrf = paths.verifier_dir / "ctrf.json"
    report = grading.read_ctrf(ctrf.read_bytes() if ctrf.is_file() else None)

    values: dict[str, Any] = {}
    records: dict[str, dict[str, Any]] = {}
    gate_failed = False
    for c in criteria:
        if c.get("judge") != "test":
            continue
        matched = grading.matching_tests(str(c.get("check", "")), report.tests)
        passed = bool(matched) and all(t.get("status") == "passed" for t in matched)
        values[c["id"]] = passed
        if c.get("gate") and not passed:
            gate_failed = True
        rationale = (
            grading.unmatched_check_note(str(c.get("check", "")), report.tests)
            if not matched
            else f"all {len(matched)} matching tests passed"
            if passed
            else "not passed: "
            + ", ".join(
                f"{t['name']} ({t['status']})"
                for t in matched
                if t.get("status") != "passed"
            )
        )
        records[c["id"]] = {
            "check": str(c.get("check")),
            "verdict": "pass" if passed else "fail",
            "judge": {"role": "test", "tool": report.tool}
            | (
                {"rubric_version": rubric["version"]}
                if isinstance(rubric.get("version"), str)
                else {}
            ),
            "rationale": rationale[:300],
        }

    lazy = True
    judges_table = table(table(config.get("verifier")).get("judges"))
    if judges_table.get("lazy") is False:
        lazy = False
    usage_total: dict[str, Any] = {}
    setup_hash = None
    if model_roles:
        if (paths.verifier_dir / "judge").is_dir():
            raise TaskMdVerifierError(
                "the verifier wrote files for the judges (/logs/verifier/judge/), which BenchFlow does not serve them yet"
            )
        skip = set()
        if lazy and gate_failed:
            skip = {
                c["id"]
                for c in criteria
                if c.get("judge") in ("llm", "agent")
                and not float(c.get("points", 0) or 0) < 0
            }
        try:
            judged, setup_hash, usage_total = await _run_judges(
                verifier=verifier,
                meta=meta,
                config=config,
                document=document,
                pkg=pkg,
                shared=shared,
                criteria=criteria,
                report=report,
                fs_root=fs_root,
                kept=kept,
                modes=modes,
                refused=refused,
                skip=skip,
                judge_dir=judge_dir,
                separate=cfg.get("isolation") == "separate",
                sandbox=sandbox,
            )
        except judging.JudgeError as exc:
            raise TaskMdJudgeInfrastructureError(
                "The judge could not run: " + verifier_failure_reason(exc)
            ) from exc
        finally:
            if "agent" in model_roles:
                await restore_hidden(sandbox)
        for ident, record in judged.items():
            records[ident] = record
            values[ident] = record.pop("_value")

    missing_ids = [c["id"] for c in criteria if c["id"] not in values]
    if missing_ids:
        raise TaskMdVerifierError(f"no verdict for {', '.join(missing_ids)}")
    result = grading.score(rubric, criteria, values)
    reward = grading.headline(rubric, result)
    verdicts = []
    summary: dict[str, int] = {}
    for c in criteria:
        record = records[c["id"]]
        entry = {"id": c["id"], **record}
        if c.get("gate"):
            entry["gate"] = True
        entry["score"] = grading.verdict_points(
            c, values[c["id"]] if record["verdict"] != "skip" else None
        )
        verdicts.append(entry)
        summary[entry["verdict"]] = summary.get(entry["verdict"], 0) + 1
    review: dict[str, Any] = {"$schema": grading.REVIEW_SCHEMA, "record": "verifier"}
    if isinstance(rubric.get("version"), str):
        review["rubric_version"] = rubric["version"]
    if model_roles:
        review["submission_tree"] = jp.submission_tree(fs_root)
    review["summary"] = summary
    review["verdicts"] = verdicts
    review["x-benchflow"] = {
        "format": "task.md draft 2",
        "reference_commit": (meta.get("reference") or {}).get("commit"),
        "script_exit": return_code,
        "tests": {"count": len(report.tests), "problem": report.problem},
        "raw": float(result.raw),
        "earned": float(result.earned),
        "maximum": float(result.maximum),
        "failed_gates": result.failed,
        "offline": offline_note,
        "refused_outputs": refused,
        "judge_setup_sha256": setup_hash,
        "judge_usage": usage_total,
        "not_checked": ["[runs]", "[integrity.controls]", "rubric validation"],
    }
    _write_json(paths.verifier_dir / "review.json", review)
    rewards = {
        "reward": reward,
        "strict": 1.0 if result.strict else 0.0,
        "partial": float(result.partial),
    }
    _write_json(paths.reward_json_path, rewards)
    _host_write(paths.reward_text_path, f"{reward}\n")
    return VerifierResult(rewards=rewards)


async def _run_judges(
    *,
    verifier: Any,
    meta: dict[str, Any],
    config: dict[str, Any],
    document: Any,
    pkg: Path,
    shared: dict[str, str],
    criteria: list[dict[str, Any]],
    report: grading.TestReport,
    fs_root: Path,
    kept: list[str],
    refused: dict[str, str],
    skip: set[str],
    modes: dict[str, tuple[str, int]] | None = None,
    judge_dir: Path,
    separate: bool,
    sandbox: Any,
) -> tuple[dict[str, dict[str, Any]], str | None, dict[str, Any]]:
    """Run every model-judged unit's samples and combine them per criterion."""
    credentials, why = judging.resolve_credentials()
    if credentials is None:
        raise TaskMdInfrastructureError(
            "the rubric has model-judged criteria, and "
            + (
                f"the judges cannot be served: {why}"
                if why
                else f"no judge credentials are set ({judging.CREDENTIALS_HELP})"
            )
        )
    paths = verifier._rollout_paths
    scrub = judging.Scrub.for_run(credentials)
    judgments = paths.verifier_dir / JUDGMENTS
    # /judge/ files: the instruction, the trajectory view, the tests.
    judge_root = fs_root / "judge"
    _real_dirs(judge_dir, fs_root)
    destination(judge_root).mkdir(exist_ok=True, private=True)
    _real_dirs(judge_root)
    instruction = verifier._task.instruction or ""
    for turn in meta.get("turns") or []:
        instruction = instruction.rstrip("\n") + "\n\n" + str(turn.get("prompt", ""))
    instruction_bytes = (instruction.rstrip("\n") + "\n").encode("utf-8")
    _host_write(judge_root / "instruction.md", instruction_bytes)
    events = _trial_events(paths)
    oracle_output = _oracle_output(paths)
    if oracle_output is not None:
        oracle_output = scrub.text(oracle_output.decode("utf-8", "replace")).encode(
            "utf-8"
        )
    # The ACP trajectory was redacted when it was written; the judge's own key and
    # a scripted seat's output are scrubbed here, before any view is built.
    record = scrub.obj(traj.solver_record(events, oracle_output))
    _host_write(judge_root / "trajectory.jsonl", jp.trajectory_view(record))
    _host_write(
        judge_root / "trajectory-reasoning.jsonl",
        jp.trajectory_view(record, reasoning=True),
    )
    _host_write(judge_root / "tests.json", scrub.text(grading.tests_view(report)))
    _write_json(judge_dir / "trajectory-1.json", record)
    scripted = traj.is_scripted(events)
    mounts = (str(meta.get("oracle_mount") or "/oracle"),) if scripted else ()

    workdir = (
        (config.get("sandbox") or {}).get("workdir")
        if isinstance(config.get("sandbox"), dict)
        else None
    )
    files = judging.JudgeFiles(
        root=fs_root, kept=kept, separate_verifier=separate, scripted_mounts=mounts
    )
    with private_storage_io("judge tests view read"):
        tests_text = (judge_root / "tests.json").read_text()
    tests_by_id = {t["id"]: t for t in json.loads(tests_text)["tests"]}
    by_id = {c["id"]: c for c in criteria}
    sessions = judging.compile_sessions(pkg, shared, fs_root)
    seat_visible = bool(mounts) and _seat_visible(files, mounts)
    # Criteria decided without a session (docs/runtime/judging.md, "Missing evidence").
    decided = {
        c["id"]: found
        for c in criteria
        if c.get("judge") in ("llm", "agent")
        and c["id"] not in skip
        and (found := _evidence_missing(c, fs_root, workdir, refused)) is not None
        and found[1] is not None
    }
    used_roles = sorted({c.get("judge") for c in criteria} & {"llm", "agent"})
    setup: dict[str, Any] | None = None
    samples: dict[str, list[dict[str, Any]]] = {}
    usage_total = {
        "sessions": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tool_calls": 0,
        "seconds": 0.0,
    }
    runner: SandboxRunner | None = None
    submission = jp.submission_tree(fs_root)
    judged_meta: dict[str, dict[str, Any]] = {}
    for session in sessions:
        role = session["role"]
        ids = [i for i in session["criteria"] if i not in skip and i not in decided]
        if not ids:
            continue
        settings, _ = ref.resolve_judge(document.config, role, None, pkg)
        model = judging.model_for(role, settings)
        if model is None:
            raise TaskMdInfrastructureError(
                f"[verifier.judges.{role}] names no model: set one, or {judging.MODEL_ENV}"
            )
        refusal = judging.model_refusal(model, credentials)
        if refusal is not None:
            raise TaskMdInfrastructureError(f"judge model {refusal}")
        if setup is None:
            models = {}
            for used in used_roles:
                chosen_model = judging.model_for(
                    used, ref.resolve_judge(document.config, used, None, pkg)[0]
                )
                if chosen_model is not None:
                    models[used] = chosen_model
            setup = jp.judge_setup(
                pkg, shared=shared, models=models, config=document.config
            )
        seconds = int(-(-(ref.duration_s(settings.get("timeout")) or 120) // 1))
        budget = table(settings.get("budget"))
        tokens = (
            ref.count_value(budget.get("tokens"))
            if budget.get("tokens") is not None
            else None
        )
        budget_text = ref.session_budget(document, pkg, session) or f"{seconds} seconds"
        calls = None
        if role == "agent":
            for part in budget_text.split(", "):
                if part.endswith(" tool calls"):
                    calls = int(part.split()[0])
        brief = (
            jp.normalize_brief((pkg / session["brief"]).read_bytes())
            if session.get("brief")
            else None
        )
        tools = (
            ("read", "run", "submit_review") if role == "agent" else ("submit_review",)
        )
        spec = judging.SessionSpec(
            role=role,
            unit=session["unit"],
            model=model,
            assignment=session["assignment"],
            prompt_masked=session["prompt"],
            brief=brief,
            budget_text=budget_text,
            seconds=seconds,
            tokens=tokens,
            tool_calls=calls,
            tools=tools,
        )
        if jp.prompt_text(brief, session["assignment"]) != session["prompt"]:
            raise TaskMdInfrastructureError(
                "judge-prompt@1 did not compile to the reference compiler's bytes"
            )
        if role == "agent" and runner is None:
            if not separate:
                raise TaskMdInfrastructureError(
                    'an agent judge needs [verifier] isolation = "separate"'
                )
            views = _views(settings, judge_root, record)
            files.views = views
            runner = SandboxRunner(
                sandbox=sandbox,
                kept=kept,
                views=views,
                safepath=_safepath(config),
                custody=fs_root,
                modes=modes,
            )
            await runner.setup()
        evidence = (
            judging.evidence_with_fence(session["assignment"], fs_root)
            if role == "llm"
            else None
        )
        n_samples = int(settings.get("samples", 1))
        client = judging.MessagesClient(credentials)
        unit_meta = {
            "role": role,
            "unit": session["unit"],
            "model": model,
            "prompt_sha256": session["prompt_sha256"],
            "evidence_sha256": session.get("evidence_sha256"),
            "settings": settings,
        }
        for sample in range(1, n_samples + 1):
            accepted = None
            failures: list[dict[str, str | None]] = []
            for attempt in (1, 2):
                seed = jp.judge_seed(submission, role, session["unit"], sample, attempt)
                judging.SESSIONS.acquire()
                result = await judging.run_session(
                    spec,
                    call=client,
                    credentials=credentials,
                    files=files,
                    evidence_text=evidence,
                    runner=runner,
                    scrub=scrub.evidence,
                )
                name = (
                    f"{role}-{session['unit'].replace(':', '-')}-s{sample}-a{attempt}"
                )
                _write_json(
                    judge_dir / f"{name}.json",
                    {
                        "role": role,
                        "unit": session["unit"],
                        "sample": sample,
                        "attempt": attempt,
                        "seed": seed,
                        "end": result.end,
                        "failure_kind": result.failure_kind,
                        "usage": result.usage,
                        "model": model,
                        "harness": f"{judging.HARNESS}@{judging.HARNESS_VERSION}",
                        "system_note": credentials.system_note,
                        "accepted": result.accepted,
                        "transcript": result.transcript,
                    },
                )
                _append_jsonl(
                    judgments,
                    {
                        "record": "session",
                        "name": name,
                        "role": role,
                        "unit": session["unit"],
                        "sample": sample,
                        "attempt": attempt,
                        "model": model,
                        "provider": credentials.provider,
                        "end": result.end,
                        "usage": result.usage,
                        "calls": judging.calls_of(result.transcript),
                    },
                )
                usage_total["sessions"] += 1
                for key in ("prompt_tokens", "completion_tokens", "tool_calls"):
                    usage_total[key] += int(result.usage.get(key, 0))
                usage_total["seconds"] += float(result.usage.get("seconds", 0))
                if result.accepted is not None:
                    accepted = (result, seed, name)
                    break
                failures.append({"end": result.end, "kind": result.failure_kind})
            for ident in ids:
                crit = by_id[ident]
                if accepted is None:
                    samples.setdefault(ident, []).append(
                        {
                            "verdict": "error",
                            "score": None,
                            "trajectory": name,
                            "end": result.end,
                            "failures": failures,
                        }
                    )
                    continue
                res, seed, name = accepted
                submitted = table(res.accepted)
                raw = next(
                    v for v in listed(submitted.get("verdicts")) if v["id"] == ident
                )
                ctx = judging.CitationContext(
                    files=files,
                    record=record,
                    tests=tests_by_id,
                    instruction=instruction_bytes,
                    workdir=workdir if isinstance(workdir, str) else None,
                    reasoning_served=any(
                        str(e).startswith(jp.RUNTIME_KINDS["trajectory:reasoning"])
                        for criterion in session["assignment"]["criteria"]
                        for e in criterion["evidence"]
                    ),
                    scrub=scrub.evidence,
                )
                cites = [
                    judging.check_citation(c, ctx, res.judge_steps)
                    for c in raw.get("citations", [])
                ]
                verdict = {**raw, "citations": cites}
                verdict, flags = judging.apply_citation_rules(crit, verdict)
                if seat_visible:
                    flags.append("seat-visible")
                entry: dict[str, Any] = {
                    "verdict": verdict["verdict"],
                    "score": grading.verdict_points(
                        crit, judging.verdict_value(crit, verdict)
                    ),
                    "citations": cites,
                    "rationale": str(verdict.get("rationale", ""))[
                        : judging.MAX_RATIONALE
                    ],
                    "trajectory": name,
                }
                if verdict["verdict"] == "level":
                    entry["level"] = str(verdict["level"])
                if verdict["verdict"] == "value":
                    entry["value"] = verdict["value"]
                if flags:
                    entry["flags"] = flags
                entry["_raw"] = verdict
                samples.setdefault(ident, []).append(entry)
        for ident in ids:
            judged_meta[ident] = unit_meta
    # Every session has finished. Classify all criteria still missing required
    # samples before deciding whether their failures are exclusively command
    # deadlines. A deadline in an earlier criterion cannot erase an observed
    # controller failure in a later one (7bc81e6ba).
    criterion_failures: list[str] = []
    controller_failure = False
    for c in criteria:
        ident = c["id"]
        if c.get("judge") not in ("llm", "agent") or ident in skip or ident in decided:
            continue
        entries = samples.get(ident, [])
        valid = [e for e in entries if e["verdict"] != "error"]
        settings = judged_meta.get(ident, {}).get("settings") or {}
        min_samples = int(settings.get("min_samples", settings.get("samples", 1)))
        if len(valid) >= min_samples:
            continue
        failed = [e for e in entries if e["verdict"] == "error"]
        deadlines_only = bool(failed) and all(
            e["failures"]
            and all(f["kind"] == "verification_timeout" for f in e["failures"])
            for e in failed
        )
        controller_failure |= not deadlines_only
        criterion_failures.append(
            f"criterion {ident}: {len(valid)} valid samples of {len(entries)}, "
            f"and min_samples is {min_samples}: "
            + (
                "verifier commands exceeded the judge deadline"
                if deadlines_only
                else "the judge failed"
            )
            + "; "
            + "; ".join(
                str(failure["end"]) for entry in failed for failure in entry["failures"]
            )
        )
    if criterion_failures:
        detail = "; ".join(criterion_failures)
        if controller_failure:
            raise TaskMdInfrastructureError(
                detail
                + "; the trial is not scored (BenchFlow does not judge a reference to attribute it)"
            )
        raise TaskMdEvidenceError(detail)
    setup_hash = jp.sha256_hex(jp.jcs(setup)) if setup is not None else None
    out: dict[str, dict[str, Any]] = {}
    for c in criteria:
        if c.get("judge") not in ("llm", "agent"):
            continue
        ident = c["id"]
        role = c["judge"]
        judge_record: dict[str, Any] = {"role": role}
        if ident in skip:
            out[ident] = {
                "verdict": "skip",
                "judge": judge_record,
                "flags": ["lazy"],
                "rationale": "a gate decided by a test failed, so no judge session ran for this criterion (lazy)",
                "_value": grading.skip_value(c),
            }
            continue
        if ident in decided:
            flag, label, value = decided[ident]
            out[ident] = {
                "verdict": label,
                "judge": judge_record,
                "flags": [flag],
                "rationale": (
                    "an output this criterion names was refused, so it fails without a judge session"
                    if flag == "output-refused"
                    else "the solver saved none of the files this criterion names, so it is decided without a judge session"
                ),
                "_value": value,
            }
            continue
        unit = judged_meta.get(ident, {})
        judge_record |= {
            "model": unit.get("model"),
            "harness": judging.HARNESS,
            "harness_version": judging.HARNESS_VERSION,
            "prompt_format": jp.FORMAT,
            "prompt_sha256": unit.get("prompt_sha256"),
            "provider": credentials.provider,
        }
        if setup_hash:
            judge_record["judge_setup_sha256"] = setup_hash
        if unit.get("evidence_sha256"):
            judge_record["evidence_sha256"] = unit["evidence_sha256"]
        if isinstance((document.rubric or {}).get("version"), str):
            judge_record["rubric_version"] = document.rubric["version"]
        entries = samples.get(ident, [])
        valid = [e for e in entries if e["verdict"] != "error"]
        settings = unit.get("settings") or {}
        public = [{k: v for k, v in e.items() if k != "_raw"} for e in entries]
        missing_file = _evidence_missing(c, fs_root, workdir, refused)
        numbers = [judging.sample_number(c, e["_raw"]) for e in valid]
        aggregate = str(settings.get("aggregate", "median"))
        combined = judging.combine(numbers, aggregate)
        chosen = next(e for e, n in zip(valid, numbers, strict=True) if n == combined)
        verdict = chosen["_raw"]
        value = judging.verdict_value(c, verdict)
        record_out: dict[str, Any] = {
            "verdict": verdict["verdict"],
            "judge": judge_record,
            "samples": public,
            "spread": float(max(numbers) - min(numbers)) if numbers else 0.0,
            "citations": chosen["citations"],
            "rationale": chosen["rationale"],
        }
        if verdict["verdict"] == "level":
            record_out["level"] = str(verdict["level"])
        if verdict["verdict"] == "value":
            record_out["value"] = verdict["value"]
        flags = list(chosen.get("flags", []))
        if missing_file is not None:
            flags.append(missing_file[0])  # judged on what remains (a bad outcome)
        if flags:
            record_out["flags"] = flags
        if aggregate == "mean" and numbers:
            record_out["x-mean"] = float(sum(numbers) / len(numbers))
        record_out["_value"] = value
        out[ident] = record_out
    for ident, judged_record in out.items():
        _append_jsonl(
            judgments,
            {"record": "criterion", "id": ident}
            | {k: v for k, v in judged_record.items() if k != "_value"},
        )
    return out, setup_hash, usage_total


def _evidence_missing(
    criterion: dict[str, Any], fs_root: Path, workdir: Any, refused: dict[str, str]
) -> tuple[str, str | None, Any] | None:
    """docs/runtime/judging.md, "Missing evidence": (flag, verdict, the score() input), or None.

    A criterion that names a refused output fails without a session, whatever
    its outcome. One whose every item is a file the solver never saved fails
    without a session when its outcome is good; a bad one is judged on what
    remains (None as its input), or skipped with ``missing = "no-penalty"``.
    """
    items = [str(i).split("#", 1)[0] for i in criterion.get("evidence") or []]
    files = [
        i for i in items if i not in jp.RUNTIME_KINDS and not i.startswith("diff:")
    ]
    paths = []
    for item in files:
        try:
            paths.append(
                jp.normalize_path(item, workdir if isinstance(workdir, str) else None)
            )
        except jp.JudgePromptError:
            return None
    if any(p == r or p.startswith(r.rstrip("/") + "/") for p in paths for r in refused):
        return ("output-refused", "fail", False)
    if not files or len(files) != len(items):
        return None
    if all(not (fs_root / p.lstrip("/")).exists() for p in paths):
        if criterion.get("outcome", "good") == "good":
            return ("evidence-missing", "fail", False)
        if criterion.get("missing") == "no-penalty":
            return ("evidence-missing", "skip", grading.skip_value(criterion))
        return ("evidence-missing", None, None)
    return None


def _seat_visible(files: judging.JudgeFiles, mounts: tuple[str, ...]) -> bool:
    """Whether anything a session could read still names the scripted seat's mount."""
    for path in (files.root / "judge").rglob("*"):
        if path.is_file():
            with contextlib.suppress(OSError):
                data = path.read_text(encoding="utf-8", errors="replace")
                if any(m in data for m in mounts) or "oracle/solve.sh" in data:
                    return True
    for kept in files.kept:
        base = files.host(kept)
        for path in (
            [base] if base.is_file() else base.rglob("*") if base.is_dir() else []
        ):
            if path.is_file() and path.stat().st_size < 4_000_000:
                with contextlib.suppress(OSError):
                    data = path.read_text(encoding="utf-8", errors="replace")
                    if any(m in data for m in mounts):
                        return True
    return False


def _views(
    settings: dict[str, Any], judge_root: Path, record: dict[str, Any]
) -> dict[str, bytes]:
    """The agent role's views: the trajectory view at each named path."""
    views: dict[str, bytes] = {}
    for view in settings.get("views") or []:
        if (
            isinstance(view, dict)
            and view.get("format") == "trajectory-1"
            and isinstance(view.get("path"), str)
        ):
            views[jp.normalize_path(view["path"])] = jp.trajectory_view(record).encode(
                "utf-8"
            )
    return views


def _trial_events(paths: Any) -> list[dict[str, Any]]:
    for folder in (
        paths.agent_dir,
        ReachablePath.rollout(paths.rollout_dir.parent, "agent"),
    ):
        path = folder / "acp_trajectory.jsonl"
        if path.is_file():
            return traj.load_events(path.read_bytes())
    return []


def _oracle_output(paths: Any) -> bytes | None:
    for folder in (
        paths.agent_dir,
        ReachablePath.rollout(paths.rollout_dir.parent, "agent"),
    ):
        path = folder / "oracle.txt"
        if path.is_file():
            return path.read_bytes()
    return None


__all__ = ["TaskMdVerifierError", "verify_taskmd"]
