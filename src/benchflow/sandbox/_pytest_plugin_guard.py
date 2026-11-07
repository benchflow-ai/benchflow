"""Pytest plugin provenance checks, also installed as a protected verifier guard.

Selective PR #1117 adaptation (tulerfeng).
Checks entry-point registrations and module paths without importing candidates.
The public pytest hooks run inside the final verifier/uvx Python environment.

That environment can be any Python 3 and any pytest a task image ships, and a
guard that fails to import or register aborts pytest, which test.sh records as
reward 0. So this module carries no annotations (``tuple[str, ...]`` raises at
import before Python 3.9) and hooks take only arguments pytest 7 passes.

Because test.sh may discard pytest's output, the armed guard also reports to
the verifier through marker files named ``<prefix>.<token>.<kind>``: a
``loading`` marker written when pytest's plugin loader imports it and removed
once pytest has registered the guard or the guard itself stopped pytest (one
left behind means pytest imported the guard but never ran it), and a
``registered`` marker once pytest has registered it. A marker turns a run the
solution may have failed into an unscored one, so the solution must not be
able to write one: markers go to a directory only the verifier's user can
write, not the world-writable ``/logs/verifier``, and each carries an
HMAC-SHA256, keyed by a secret hardening puts in a file only the verifier's
user can read, that the verifier checks. The secret is never in the
environment. Code the verifier itself runs as that user (a solution module a
test imports) can still read it, as it could rewrite the reward.

The guard stops pytest in two ways. Refusing a plugin raises ``Rejected``: the
hook leaves a ``refused`` marker holding the refusal, and the verifier reports
an error instead of a score: pytest stopped before any test ran,
so test.sh's 0 would score the guard, not the solution. Any other exception
from a guard hook is the guard failing (a bug, a pytest API it did not
expect); the hook leaves a ``crashed`` marker holding the traceback, still
stops pytest, and the verifier reports an error too. Inspecting plugin files
and metadata is a refusal, not a crash: an agent can plant those, and a
malformed ``entry_points.txt`` line alone makes CPython 3.12's
importlib.metadata raise TypeError, so any error while inspecting them is a
``Rejected``.

Where the armed guard lets plugin code come from: BenchFlow's
protected verifier runtime, a root-only tree built before the agent's turn
from wheels checked against PyPI (``_verifier_runtime``), whose plugins the
guard registers itself, compiling their sources so no byte-code cache is read
for them; the pytest already running; or the image. Nothing test.sh installed
after the agent stopped loads, wherever it put it and however root's it is:
three review rounds found ways around judging such files by their bytes
(link spellings, byte-code caches, system installs, ``.pth`` hooks), so the
verifier's plugins come from the runtime instead. The image's code is

* a file outside the agent's trees that is root's alone, parents included,
  and unchanged since the guard was armed (the arm is the change time of the
  key file hardening writes in the same exec);
* a file that still holds the image's bytes: hardening names the
  workspace and its pre-agent copy (``/testbed_verify``, root's alone), and a
  workspace file whose copy is a root-owned regular file with the same bytes
  (a directory: whose copy is a directory) is the image's own, such as the
  ``sphinx.testing.fixtures`` a Sphinx repository's conftest.py declares. An
  edited one differs from the copy and is refused.

For either, every byte-code cache Python or pytest would read in the
source's place, looked up by the spelling Python imported, must be the
image's or the source's own compilation (V2-REDESIGN-R3-1, R3-3). The record
of the agent's trees and the proof that the agent left them alone now only
choose a refusal's words (whose problem it is), never trust.

A refusal the verifier caused itself is a task problem: test.sh installed
the plugin after hardening into a place the guard still cannot trust (outside
any site directory, say, or not root's). When every refused plugin has at
least one untrusted file behind it and every one of those is test.sh's own
install by the record (new since it, in directories new since it, chosen by
project files that are new or the image's), the guard also leaves an
``installed`` marker listing them, each with why it is untrusted, so the
verifier's error says what the task should change. Without a record, or
when the agent's files had a hand in the install (a ``.venv`` or
``pyproject.toml`` it left), the refusal stays a refusal, which counts as a
failure: the agent cannot pick the task's error (V2-R1-A-3, V2-R1-B-1).
Every refusal message names each refused plugin and what about it the guard
could not trust.

Plugins BenchFlow finds in the image's Python or infers from test.sh (``ctrf``
for ``--ctrf``) are optional (#1175): ``PYTEST_ADDOPTS`` reaches every pytest
test.sh starts, including one in a fresh ``uv init`` project or a ``uvx``
environment that does not have them, so the guard loads each one only in a
pytest that has it installed, and refuses it there when it is not trusted.
Plugins the task declares (``[verifier] pytest_plugins``) stay ``-p`` flags,
required in every pytest.

A directory writable by its group is the agent's only when the agent's user
is in that group (#1176): Debian and Ubuntu ship ``/usr/local/lib/pythonX.Y``
as ``root:staff`` mode 2775, and the agent's user is not in ``staff``.
Hardening names the agent's groups; without them every group-writable path
stays untrusted.
"""

import sys


def _stdlib_dir():
    """The standard library folder of this Python, read without importing."""
    base = getattr(sys, "_stdlib_dir", None)
    if not base:
        module = sys.modules.get("os")
        base = getattr(module, "__file__", None)
        base = base.rsplit("/", 1)[0] if base else None
    if not base:
        base = (
            f"{sys.base_prefix}/lib/python{sys.version_info[0]}.{sys.version_info[1]}"
        )
    return base


_STDLIB = _stdlib_dir()
# compile()'s mode for a whole module (the guard runs no container exec).
_MODULE = "exec"
# What the guard imports, from the standard library alone (V2-REDESIGN-R3-2):
# pytest imports it with test.sh's sys.path, which can start with the
# workspace, where a hashlib.py the agent left would otherwise run first.
# Each name is imported with sys.path narrowed to the standard library's
# folders (not its site-packages), and every one, imported here or before,
# must come from there; anything else stops pytest before any plugin loads.
_SAVED_PATH = sys.path[:]
sys.path[:] = [
    f"{sys.base_prefix}/lib/python{sys.version_info[0]}{sys.version_info[1]}.zip",
    _STDLIB,
    _STDLIB + "/lib-dynload",
]
try:
    import contextlib
    import hashlib
    import hmac
    import importlib.machinery
    import importlib.util
    import json
    import marshal
    import os
    import pathlib
    import shlex
    import stat
    import struct
    import time
    import traceback

    # Python 3.7 has no importlib.metadata (pytest brings importlib_metadata);
    # a Python built without _ctypes has no statx (_statx then says so).
    with contextlib.suppress(ImportError):
        import importlib.metadata
    with contextlib.suppress(ImportError):
        import ctypes
finally:
    sys.path[:] = _SAVED_PATH


def _standard_origin(module):
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
    if origin in (None, "built-in", "frozen"):
        return True
    origin = str(origin)
    inside = origin.startswith(_STDLIB + "/") or origin.startswith(
        f"{sys.base_prefix}/lib/python{sys.version_info[0]}{sys.version_info[1]}.zip/"
    )
    return inside and not any(
        "/" + part + "/" in origin for part in ("site-packages", "dist-packages")
    )


for _name in (
    "contextlib",
    "hashlib",
    "hmac",
    "importlib",
    "importlib.machinery",
    "importlib.util",
    "importlib.metadata",
    "json",
    "marshal",
    "os",
    "re",
    "shlex",
    "stat",
    "struct",
    "time",
    "traceback",
    "ctypes",
    "zipfile",
    "email",
    "csv",
    "pathlib",
    "encodings",
):
    _module = sys.modules.get(_name)
    if _module is not None and not _standard_origin(_module):
        raise ImportError(
            "the pytest plugin guard refuses to run: "
            + _name
            + " was imported from "
            + str(getattr(_module, "__file__", "?"))
            + ", not this Python's standard library"
        )

suppress = contextlib.suppress
PathFinder = importlib.machinery.PathFinder

# Reassigned by lines appended to this source for the protected runtime plugin.
_BENCHFLOW_BLOCKED = ()  # type: tuple[str, ...]
_BENCHFLOW_REQUESTED = []  # type: list[str]
# In the agent's own sandbox code must also be root's alone (root-owned, not
# group- or world-writable, parents included). A separate verifier sandbox
# never ran the agent: only the blocked paths, which the transfer wrote, are
# the agent's, and everything else is the verifier image's own.
_BENCHFLOW_OWNERSHIP = True
# Discovery uses the image's pre-agent search folders as data. The armed
# runtime guard leaves this None and checks pytest's actual import path.
_BENCHFLOW_SEARCH_PATH = None
# Marker path prefix for this verification; empty writes no markers.
_BENCHFLOW_MARKERS = ""
# The file holding this verification's marker key (hex), readable by the
# verifier's user alone; markers are not written without it.
_BENCHFLOW_KEY_FILE = ""
# Where pre-verifier discovery looks for plugins and their metadata, or None
# for this interpreter's sys.path (the runtime guard). Discovery runs as root
# under ``python3 -I`` and takes the verifier's trusted PYTHONPATH as data,
# never as import paths: that PYTHONPATH may name the workspace (/app), and a
# json.py the agent left there must not run as root (PY8-MAIN-1).
_BENCHFLOW_SEARCH_PATH = None  # type: list[str] | None
# Plugins found in the image or inferred from test.sh: loaded by the guard in
# a pytest that has them installed, skipped in one that does not (#1175).
_BENCHFLOW_OPTIONAL = []  # type: list[str]
# The groups the agent's user is in (#1176); None when unknown, so every
# group-writable path stays untrusted.
_BENCHFLOW_AGENT_GIDS = None  # type: tuple[int, ...] | None
# The agent's workspace and its root-only pre-agent copy; empty
# when there is none (a separate verifier sandbox).
_BENCHFLOW_WORKSPACE = ""
_BENCHFLOW_SNAPSHOT = ""
# True in the guard hardening wrote, whose own change time marks when the
# agent had stopped; discovery before the verifier has no such time.
_BENCHFLOW_ARMED = False
# True when hardening saw test.sh run as root (V2-R1-A-2): with the agent's
# processes gone, no one left can use a write bit inside the agent's trees.
_BENCHFLOW_VERIFIER_ROOT = False
# The root-only record of the agent's trees taken before this guard was
# written: one ``<st_dev>:<st_ino>`` line per file or directory,
# with ``:<birth time>`` after it where the filesystem tells one (V2-R1-A-2),
# then ``end``. Empty: nothing in those trees counts as the verifier's install.
_BENCHFLOW_RECORD = ""
# The record, once read: a dict from ``<st_dev>:<st_ino>`` to the birth times
# recorded with it (None for a line without one), or False when it cannot be
# trusted.
_RECORD = None
# statx(2) through ctypes, once looked up: None before, False when this Python,
# its C library or the machine has none.
_STATX = None
# Dependency files the parser in ``lockdown`` finds named in the verifier's
# scripts: hints that order the proof (_hinted_input_changed), never evidence.
_BENCHFLOW_NAMED_INPUTS = None  # type: tuple[str, ...] | None
# The root-only file holding the proof, frozen before test.sh ran, that the
# agent left every file in its trees alone (_write_input_proof).
_BENCHFLOW_INPUT_PROOF = None  # type: str | None
# BenchFlow's protected verifier runtime: its site folder, root's
# alone, built before the agent's turn from wheels checked against PyPI; the
# pytest11 names of the plugins it holds (name: module), which the guard
# registers from there; and the top-level names it serves, from there alone.
_BENCHFLOW_RUNTIME_SITE = ""
_BENCHFLOW_RUNTIME_PLUGINS = {}  # type: dict[str, str]
_BENCHFLOW_RUNTIME_TOP = ()  # type: tuple[str, ...]
# The folder BenchFlow uploads the verifier's tests to (/tests, or /verifier):
# root's, written by BenchFlow, so newer than the arm without being test.sh's.
_BENCHFLOW_TESTS_DIR = ""
# Plugins test.sh's command line disables (``-p no:<name>``), as the guard
# read them before pytest loads the optional ones.
_DISABLED = set()
# How deep this process is in trust checks (_begin_check); a list so the
# hooks change it without a global statement.
_CHECKING = [0]
# The exception an optional plugin raised while pytest imported it, if any.
_OPTIONAL_FAILURE = None
# When the proof being written must give up (time.monotonic()), or None: it
# runs inside hardening's exec, whose timeout would fail the trial
# (V2-R1-A-2).
_PROOF_DEADLINE = None
# Keeps this process's markers apart from other pytest runs of one verifier.
_TOKEN = str(os.getpid()) + "-" + os.urandom(4).hex()


def _marker(kind):
    return ".".join((_BENCHFLOW_MARKERS, _TOKEN, kind))


def _mark(kind, detail=""):
    """Leave a signed ``kind`` marker for the verifier; never fail pytest doing so.

    One line: the HMAC-SHA256 of the marker's name and body, a tab, and the
    body, *detail* as a JSON string.
    """
    if not _BENCHFLOW_MARKERS or not _BENCHFLOW_KEY_FILE:
        return
    try:
        with open(_BENCHFLOW_KEY_FILE, "rb") as handle:
            key = handle.read().strip()
        if not key:
            return
        path = _marker(kind)
        # The verifier reads 64 KiB of a marker, and one bigger makes its read
        # incomplete; keep a traceback's end. JSON escapes a character in up
        # to 12 bytes, so 4,000 characters always fit.
        body = json.dumps(detail[-16000:])
        if len(body) > 60000:
            body = json.dumps(detail[-4000:])
        signed = (os.path.basename(path) + "\n" + body).encode("utf-8")
        mac = hmac.new(key, signed, hashlib.sha256).hexdigest()
        with open(path, "x") as handle:
            handle.write(mac + "\t" + body + "\n")
    except Exception:
        pass


def _imported_by_pytest():
    """Whether pytest's plugin loader (``-p``, ``PYTEST_PLUGINS``) is importing this.

    Any other import, such as code under test importing the guard by the
    name in ``PYTEST_ADDOPTS``, must not leave a ``loading`` marker behind.
    """
    try:
        frame = sys._getframe(1)
    except Exception:
        return False
    while frame is not None:
        if frame.f_code.co_name == "import_plugin" and str(
            frame.f_globals.get("__name__", "")
        ).startswith("_pytest."):
            return True
        frame = frame.f_back
    return False


def _unmark(kind):
    if _BENCHFLOW_MARKERS:
        with suppress(OSError):
            os.remove(_marker(kind))


class Rejected(RuntimeError):
    """The guard refuses a plugin; pytest stops and the run is not scored."""


def _stopping(exc):
    """Record why this guard, not a failure to load it, is stopping pytest.

    A refusal leaves a ``refused`` marker, any other failure a ``crashed``
    one. An optional plugin failing to import is that plugin's failure, as
    with ``-p``, so it leaves neither (see _load_optional). The record is
    written before the ``loading`` and ``checking`` markers go, so a process
    killed in between still leaves one of them.
    """
    if isinstance(exc, Rejected):
        _mark("refused", str(exc))
    elif exc is not _OPTIONAL_FAILURE:
        _mark(
            "crashed",
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
    _unmark("loading")
    _CHECKING[0] = 0
    _unmark("checking")


def _begin_check():
    """Enter a trust check: a ``checking`` marker stands until it completes.

    A pytest killed inside one (a FIFO the agent left as metadata, test.sh's
    own timeout) leaves the marker behind, and the verifier does not score
    the run. Checks nest: an optional plugin's own
    ``pytest_plugins`` are checked while it loads.
    """
    if _CHECKING[0] == 0:
        _mark("checking")
    _CHECKING[0] += 1


def _end_check():
    _CHECKING[0] -= 1
    if _CHECKING[0] <= 0:
        _CHECKING[0] = 0
        _unmark("checking")


def under(path, prefix):
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


# Why code installed after the agent stopped is refused.
_INPUTS_WHY = (
    "the agent's files could have chosen what was installed there, so the "
    "verifier loads no plugin installed after the agent stopped: pin its "
    "release in [verifier] pytest_plugin_releases, which BenchFlow serves "
    "from its protected verifier runtime, or install it into the "
    "image"
)


def _blocked_why():
    why = (
        "which the agent could write"
        if _BENCHFLOW_OWNERSHIP
        else "which holds files copied from the agent's sandbox"
    )
    if _BENCHFLOW_ARMED:
        return why + "; " + _INPUTS_WHY
    return why


def ownership_problem(path, st):
    """Why the inode at *path* (stat *st*) is not root's alone, or None.

    Group write counts only when the group is one of the agent's, or the
    agent's groups are unknown (#1176).
    """
    if st.st_uid != 0:
        return path + " is owned by uid " + str(st.st_uid) + ", not root"
    gids = _BENCHFLOW_AGENT_GIDS
    group_writable = st.st_mode & stat.S_IWGRP and (gids is None or st.st_gid in gids)
    if group_writable or st.st_mode & stat.S_IWOTH:
        mode = format(stat.S_IMODE(st.st_mode), "04o")
        return path + " is group- or world-writable (mode " + mode + ")"
    return None


def _without_write_bits(st):
    """*st* as ownership_problem should judge it when only its owner counts."""
    fields = list(st)
    fields[stat.ST_MODE] = st.st_mode & ~(stat.S_IWGRP | stat.S_IWOTH)
    return os.stat_result(fields)


def _root_alone(path):
    """Whether *path* and every directory above it are root's alone."""
    while True:
        try:
            if ownership_problem(path, os.stat(path)):
                return False
        except OSError:
            return False
        parent = os.path.dirname(path)
        if parent == path:
            return True
        path = parent


def _private_to_root(directory):
    """Whether only root can reach into *directory*: root's, mode 0700 or
    narrower, not a link, and every directory above it root's alone. What
    is inside then needs no owner or mode of its own."""
    try:
        st = os.lstat(directory)
    except OSError:
        return False
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != 0 or st.st_mode & 0o077:
        return False
    return _root_alone(os.path.dirname(directory))


def _image_bytes(path):
    """Whether *path* in the workspace still holds the image's bytes.

    The pre-agent copy is the workspace copied before the agent ran, which
    only root can reach (hardening locks it 0700; its files keep the
    workspace's modes, 0777 folders included on SWE-bench images). The
    counterpart of *path* there must be a regular file with the same bytes (a
    directory for a directory), reached without leaving the copy, so a link
    in it cannot point back into the live workspace.
    """
    workspace = _BENCHFLOW_WORKSPACE.rstrip("/")
    snapshot = _BENCHFLOW_SNAPSHOT.rstrip("/")
    if not workspace or not snapshot or not under(path, workspace):
        return False
    try:
        if os.path.realpath(snapshot) != snapshot or not _private_to_root(snapshot):
            return False
        copy = os.path.realpath(snapshot + path[len(workspace) :])
        if not under(copy, snapshot):
            return False
        mine, theirs = os.lstat(path), os.lstat(copy)
        if stat.S_ISDIR(mine.st_mode):
            return stat.S_ISDIR(theirs.st_mode)
        if not (stat.S_ISREG(mine.st_mode) and stat.S_ISREG(theirs.st_mode)):
            return False
        if mine.st_size != theirs.st_size:
            return False
        with open(path, "rb") as left, open(copy, "rb") as right:
            while True:
                # A multi-gigabyte file must not carry the proof past its
                # deadline; a raise here is "not the image's", so tainted.
                _check_proof_deadline()
                chunk = left.read(1 << 16)
                if chunk != right.read(1 << 16):
                    return False
                if not chunk:
                    return True
    except Exception:
        return False


def _identity(st):
    return str(st.st_dev) + ":" + str(st.st_ino)


def _statx():
    """statx(2) as a ctypes function, or None (V2-R1-A-2)."""
    global _STATX
    if _STATX is None:
        _STATX = False
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            try:
                _STATX = libc.statx
            except AttributeError:
                # A C library older than glibc 2.28 has no wrapper; Linux
                # 4.11 and later still has the system call.
                number = {"x86_64": 332, "aarch64": 291}.get(os.uname().machine)
                if number is not None:
                    syscall = libc.syscall

                    def _statx_syscall(*args):
                        return syscall(ctypes.c_long(number), *args)

                    _STATX = _statx_syscall
        except Exception:
            _STATX = False
    return _STATX or None


def _birth(path, st, follow):
    """When the inode *st* describes at *path* was made, or None (V2-R1-A-2).

    Unlike the inode number, which the filesystem hands to the next new file
    once the agent's file is deleted, an inode's birth time comes with the
    inode, and a chmod, a chown, a touch, a write or a move keeps it. None
    where this Python, the C library or the filesystem cannot say.
    """
    function = _statx()
    if function is None:
        return None
    try:
        buffer = ctypes.create_string_buffer(256)
        flags = 0 if follow else 0x100  # AT_SYMLINK_NOFOLLOW
        if (
            function(
                ctypes.c_int(-100),  # AT_FDCWD
                os.fsencode(path),
                ctypes.c_int(flags),
                ctypes.c_uint(0x800),  # STATX_BTIME
                buffer,
            )
            != 0
        ):
            return None
        raw = buffer.raw
        (mask,) = struct.unpack_from("=I", raw, 0)
        (ino,) = struct.unpack_from("=Q", raw, 32)
        seconds, nanoseconds = struct.unpack_from("=qI", raw, 80)
        if not mask & 0x800 or ino != st.st_ino or not (seconds or nanoseconds):
            return None
        return str(seconds) + "." + str(nanoseconds).zfill(9)
    except Exception:
        return None


def _in_record(path, record, follow=False):
    """Whether the entry at *path* (what it reaches, with *follow*) is in *record*.

    Its inode number must be recorded and, when both the record and this
    Python know a birth time for it, with the same birth time: a recorded
    number the filesystem reused for a file test.sh made after deleting the
    agent's is not the agent's file (V2-R1-A-2). Where either cannot tell,
    the number alone decides, and a reused one reads as the agent's.
    """
    st = os.stat(path) if follow else os.lstat(path)
    births = record.get(_identity(st))
    if births is None:
        return False
    if None in births:
        return True
    birth = _birth(path, st, follow)
    return birth is None or birth in births


def _record():
    """The pre-arm record of the agent's trees, or None.

    It counts only when it is a regular file root's alone, parents included,
    that hardening finished writing (its last line is ``end``).
    """
    global _RECORD
    if _RECORD is None:
        _RECORD = False
        try:
            st = os.lstat(_BENCHFLOW_RECORD)
            if (
                _BENCHFLOW_RECORD.startswith("/")
                and stat.S_ISREG(st.st_mode)
                and not ownership_problem(_BENCHFLOW_RECORD, st)
                and _root_alone(os.path.dirname(_BENCHFLOW_RECORD))
            ):
                with open(_BENCHFLOW_RECORD) as handle:
                    lines = handle.read().split("\n")
                while lines and not lines[-1]:
                    lines.pop()
                if lines and lines[-1] == "end":
                    record = {}
                    for line in lines[:-1]:
                        dev, _, rest = line.partition(":")
                        ino, _, birth = rest.partition(":")
                        record.setdefault(dev + ":" + ino, set()).add(birth or None)
                    _RECORD = record
        except Exception:
            _RECORD = False
    return None if _RECORD is False else _RECORD


def _new_since_record(path, record):
    """Whether neither *path* nor what it reaches was in the agent's trees when recorded."""
    return not _in_record(path, record) and not _in_record(path, record, True)


def _environment_root(path):
    """The directory holding *path*'s ``pyvenv.cfg``, else its site directory."""
    parts = path.split("/")
    site = max(
        i for i, part in enumerate(parts) if part in ("site-packages", "dist-packages")
    )
    directory = os.path.dirname(path)
    while True:
        cfg = os.path.join(directory, "pyvenv.cfg")
        with suppress(OSError):
            if stat.S_ISREG(os.lstat(cfg).st_mode):
                return directory
        parent = os.path.dirname(directory)
        if parent == directory:
            return "/".join(parts[: site + 1])
        directory = parent


class _ProofOutOfTime(Exception):
    """The proof walk passed its deadline (_write_input_proof)."""


def _check_proof_deadline():
    if _PROOF_DEADLINE is not None and time.monotonic() > _PROOF_DEADLINE:
        raise _ProofOutOfTime()


def _agent_files_unchanged(record):
    """Whether no file the agent left can have chosen what test.sh installs.

    Which files an installer reads is not read from test.sh (V2-R6 to
    V2-R13: options, requirement and configuration files, environment
    variables and formats no parser can enumerate). So every file and link
    in every agent-writable tree counts as an input, hidden and nested ones
    included, and each must be new since the record or hold the image's
    bytes. A directory by itself selects no bytes. Neither an unreadable
    directory nor a partial walk proves anything. In a real trial the
    agent's own logs are in ``/logs``, so this holds mostly where the agent
    left nothing, a nop; a plugin test.sh installs is otherwise trusted by
    its bytes (:func:`_published`).
    """

    def unreadable(error):
        raise error

    for prefix in _BENCHFLOW_BLOCKED:
        try:
            st = os.lstat(prefix)
        except FileNotFoundError:
            continue
        _check_proof_deadline()
        if not stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode):
            if not _new_since_record(prefix, record) and not _image_bytes(prefix):
                return False
            continue
        if (
            stat.S_ISLNK(st.st_mode)
            and not _new_since_record(prefix, record)
            and not _image_bytes(prefix)
        ):
            return False
        for directory, directories, files in os.walk(prefix, onerror=unreadable):
            for name in files + [
                name
                for name in directories
                if os.path.islink(os.path.join(directory, name))
            ]:
                path = os.path.join(directory, name)
                _check_proof_deadline()
                if not _new_since_record(path, record) and not _image_bytes(path):
                    return False
    return True


def _hinted_input_changed(record):
    """Whether a dependency file test.sh's parsed text names was the agent's.

    The parser in ``lockdown`` only orders the work: an absolute file it
    names is checked before the full walk, so a changed one ends the proof
    at once. It can only make the proof tainted sooner, never clean.
    """
    for hint in _BENCHFLOW_NAMED_INPUTS or ():
        if not hint.startswith("/") or not any(
            under(hint, prefix) for prefix in _BENCHFLOW_BLOCKED
        ):
            continue
        try:
            if not os.path.lexists(hint):
                continue
            if not _new_since_record(hint, record) and not _image_bytes(hint):
                return True
        except OSError:
            return True
    return False


def _write_input_proof(budget=None):
    """Freeze, before test.sh runs, whether the agent's files were left alone.

    The answer is written once, root-only, so test.sh consuming or deleting
    an input later cannot change it (:func:`_agent_files_unchanged`). The
    walk compares every recorded file with its pre-agent copy, which takes
    minutes on a workspace of many gigabytes; past *budget* seconds it stops
    and writes "tainted", inside hardening's exec timeout, so a nop there is
    scored, not errored (V2-R1-A-2). Tainted only refuses the plugins the
    proof alone would have trusted.
    """
    global _PROOF_DEADLINE
    path = _BENCHFLOW_INPUT_PROOF
    if not path:
        return
    clean = False
    try:
        if budget is not None:
            _PROOF_DEADLINE = time.monotonic() + float(budget)
            if float(budget) <= 0:
                raise _ProofOutOfTime()
        record = _record()
        clean = (
            record is not None
            and not _hinted_input_changed(record)
            and _agent_files_unchanged(record)
        )
    except Exception:
        pass
    with open(path, "x") as handle:
        handle.write("clean\n" if clean else "tainted\n")


def _input_proof():
    try:
        path = _BENCHFLOW_INPUT_PROOF
        if not path:
            return False
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or ownership_problem(path, st):
            return False
        if not _root_alone(os.path.dirname(path)):
            return False
        with open(path) as handle:
            return handle.read(16) == "clean\n"
    except Exception:
        return False


def _project_inputs_new(root, record):
    """Whether no file the agent left can have chosen what went into *root*.

    The frozen proof when hardening wrote one, else the same walk now.
    """
    if _BENCHFLOW_INPUT_PROOF:
        return _input_proof()
    return _agent_files_unchanged(record)


def _arm_time():
    """When hardening armed this guard: its key file's change time, or None.

    Hardening writes that file in the exec that arms the guard, after the
    agent's processes are gone and before test.sh starts.
    """
    try:
        return os.stat(_BENCHFLOW_KEY_FILE).st_ctime if _BENCHFLOW_KEY_FILE else None
    except OSError:
        return None


def _after_arm(path):
    """Whether *path*, or what it reaches, changed after the guard was armed.

    A file that test.sh installed, wherever it put it and however root's it
    is, changed then (V2-REDESIGN-R3-2); the image's own files did not.
    Anything that cannot be told counts as changed. One change is not: a
    package folder of the image into which Python wrote byte-code caches of
    its own sources (:func:`_only_caches_since_arm`).
    """
    arm = _arm_time()
    if arm is None:
        return True
    try:
        if max(os.lstat(path).st_ctime, os.stat(path).st_ctime) < arm:
            return False
        return not _only_caches_since_arm(path, arm)
    except OSError:
        return True


def _only_caches_since_arm(directory, arm):
    """Whether the folder *directory* changed after the arm only by Python
    writing ``__pycache__`` folders of compilations of its own sources.

    An image built with no byte code, imported by test.sh's ``python3 -I``
    (which ignores PYTHONDONTWRITEBYTECODE), gets them at its first import:
    Terminal-Bench 4.1's verifier image and its ``anyio`` plugin. Every other
    entry, in every folder below, must predate the arm (an install, rename or
    replacement changes the entry's own change time), each ``__pycache__``
    must be a real folder made after the arm, and each cache in it must be
    this Python's compilation of the source beside it (:func:`_pyc_matches`),
    however old: Python writes its caches into the folder after making it, so
    an older one was moved there, or came with a folder renamed into place. A
    pytest rewrite cache cannot be told from another rewrite, so one counts
    as changed.
    """
    if not stat.S_ISDIR(os.lstat(directory).st_mode):
        return False
    caches = None
    for name in os.listdir(directory):
        entry = os.path.join(directory, name)
        info = os.lstat(entry)
        if name == "__pycache__":
            if not stat.S_ISDIR(info.st_mode) or info.st_ctime < arm:
                return False
            caches = entry
        elif info.st_ctime >= arm and not (
            stat.S_ISDIR(info.st_mode) and _only_caches_since_arm(entry, arm)
        ):
            return False
    if caches is None:
        return False
    tag = "." + str(sys.implementation.cache_tag) + "."
    for name in os.listdir(caches):
        pyc = os.path.join(caches, name)
        info = os.lstat(pyc)
        if not stat.S_ISREG(info.st_mode):
            return False
        stem, sep, rest = name.partition(tag)
        if not sep or not stem:
            return False
        if rest == "pyc":
            optimize = 0
        elif rest in ("opt-1.pyc", "opt-2.pyc"):
            optimize = int(rest[4])
        else:
            return False
        if not _pyc_matches(pyc, os.path.join(directory, stem + ".py"), optimize):
            return False
    return True


def _image_owned(path, blocked):
    """Whether the file *path* reaches is the image's: outside the agent's
    trees, unchanged since the arm, and root's alone with every folder above."""
    real = os.path.realpath(path)
    if any(under(real, prefix) for prefix in blocked):
        return False
    return not _after_arm(real) and _root_alone(real)


def _same_code(cached, compiled):
    """Whether the code objects *cached* and *compiled* run the same instructions.

    ``==`` compares instructions with their specializations undone, so on
    Python 3.11 and later a cache can carry specialized instructions and
    inline-cache data that compare equal and run other code (a global load
    that reads another name). The raw instructions, the stack size and the
    local names must match too, in every nested code object.
    """
    if cached != compiled:
        return False
    for attr in (
        "_co_code_adaptive",
        "co_stacksize",
        "co_nlocals",
        "co_varnames",
        "co_cellvars",
        "co_freevars",
    ):
        if getattr(cached, attr, None) != getattr(compiled, attr, None):
            return False
    code = type(compiled)
    inner = [c for c in cached.co_consts if isinstance(c, code)]
    expected = [c for c in compiled.co_consts if isinstance(c, code)]
    return len(inner) == len(expected) and all(
        _same_code(a, b) for a, b in zip(inner, expected, strict=True)
    )


def _pyc_matches(pyc, source, optimize):
    """Whether this Python, reading *pyc* for *source*, runs *source*'s code.

    True when *pyc* holds the compilation of *source* at *optimize*
    (:func:`_same_code`), or carries another magic number (this Python
    ignores it). Any other byte code would run in the source's place: Python
    reads a cache whose header matches, and an unchecked hash-based one always.
    """
    try:
        if not stat.S_ISREG(os.stat(pyc).st_mode):
            return False
        with open(pyc, "rb") as handle:
            data = handle.read()
        if data[:4] != importlib.util.MAGIC_NUMBER:
            return True
        with open(source, "rb") as handle:
            text = handle.read()
        header = 16 if sys.version_info >= (3, 7) else 12
        compiled = compile(text, source, _MODULE, dont_inherit=True, optimize=optimize)
        return _same_code(marshal.loads(data[header:]), compiled)
    except Exception:
        return False


def _caches_harmless(spelling, source, blocked):
    """Whether every byte-code cache this process would read for the module
    imported as *spelling* (its file: *source*) runs that file's code.

    Python looks the cache up by the spelling it imported, under
    ``sys.pycache_prefix`` when set (``PYTHONPYCACHEPREFIX``, ``-X
    pycache_prefix``), which can be any folder (V2-REDESIGN-R1-1, R3-1,
    R3-3); pytest's assertion rewriter reads its own
    (``<name>.<tag>-pytest-<version>.pyc``) for plugin modules. A cache the
    image shipped (:func:`_image_owned`) is the image's; any other must be
    the compilation of *source*'s bytes, and a rewrite cache, whose bytes
    the guard cannot tell from another rewrite, must not exist.
    """
    try:
        for optimize in ("", 1, 2):
            pyc = importlib.util.cache_from_source(spelling, optimization=optimize)
            if not os.path.lexists(pyc) or _image_owned(pyc, blocked):
                continue
            if not _pyc_matches(pyc, source, optimize or 0):
                return False
        rewrite = sys.modules.get("_pytest.assertion.rewrite")
        tail = getattr(rewrite, "PYC_TAIL", None)
        if tail is None:
            module = sys.modules.get("pytest") or sys.modules.get("_pytest")
            tail = "." + str(sys.implementation.cache_tag) + "-pytest-"
            tail += str(getattr(module, "__version__", "")) + ".pyc"
        cache_dir = getattr(rewrite, "get_cache_dir", None)
        if cache_dir is not None:
            directory = str(cache_dir(pathlib.Path(spelling)))
        elif getattr(sys, "pycache_prefix", None):
            directory = sys.pycache_prefix + os.path.dirname(spelling)
        else:
            directory = os.path.join(os.path.dirname(spelling), "__pycache__")
        cached = os.path.join(directory, os.path.basename(spelling)[:-3] + tail)
        return not os.path.lexists(cached) or _image_owned(cached, blocked)
    except Exception:
        return False


def _in_runtime(path):
    """Whether *path* is in BenchFlow's protected verifier runtime:
    below its site folder, root's alone with every folder above it."""
    site = _BENCHFLOW_RUNTIME_SITE
    if not site or not under(path, site):
        return False
    try:
        return ownership_problem(path, os.lstat(path)) is None and _root_alone(site)
    except OSError:
        return False


def _running_pytest(path):
    """Whether *path* is in the pytest package this process is running."""
    module = sys.modules.get("_pytest")
    origin = getattr(module, "__file__", None)
    return bool(origin) and under(path, os.path.dirname(os.path.realpath(origin)))


class _RuntimeLoader(importlib.machinery.SourceFileLoader):
    """Compiles a protected-runtime module from its source: no byte-code
    cache, wherever ``sys.pycache_prefix`` points, is ever read for it."""

    def get_code(self, fullname):
        source = self.get_data(self.path)
        return compile(source, self.path, _MODULE, dont_inherit=True)


class _RuntimeFinder:
    """Imports the runtime plugins' top-level names from the protected tree
    alone, ahead of every path test.sh's Python searches."""

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname.partition(".")[0] not in _BENCHFLOW_RUNTIME_TOP:
            return None
        search = [_BENCHFLOW_RUNTIME_SITE] if "." not in fullname else path
        spec = PathFinder.find_spec(fullname, search)
        if spec is None:
            raise ImportError(
                fullname + " is missing from BenchFlow's protected verifier runtime"
            )
        origin = spec.origin if str(spec.origin or "").startswith("/") else None
        locations = ([origin] if origin else []) + [
            str(p) for p in (spec.submodule_search_locations or [])
        ]
        if not locations or not all(
            _in_runtime(os.path.realpath(p)) for p in locations
        ):
            raise ImportError(
                fullname + " is not in BenchFlow's protected verifier runtime"
            )
        if origin and origin.endswith(".py"):
            spec.loader = _RuntimeLoader(fullname, origin)
        return spec

    @classmethod
    def find_distributions(cls, context=None):
        """A runtime distribution's metadata comes from the tree first: a
        plugin that reads its own version (pytest-asyncio does) must not get a
        dist-info the agent left on test.sh's path, nor fail without one."""
        name = getattr(context, "name", None)
        if not _BENCHFLOW_RUNTIME_SITE or not name:
            return iter(())
        import importlib.metadata as metadata

        return iter(
            list(
                metadata.MetadataPathFinder.find_distributions(
                    metadata.DistributionFinder.Context(
                        name=name, path=[_BENCHFLOW_RUNTIME_SITE]
                    )
                )
            )
        )

    @classmethod
    def invalidate_caches(cls):
        pass


def _install_runtime_finder():
    if _BENCHFLOW_RUNTIME_SITE and _RuntimeFinder not in sys.meta_path:
        sys.meta_path.insert(0, _RuntimeFinder)
        # Pytest installs its rewriter ahead of meta_path finders. It must
        # return our authenticated spec for protected names, never search
        # the ambient path again (V2-R5-C3).
        from _pytest.assertion.rewrite import AssertionRewritingHook

        original = AssertionRewritingHook.find_spec

        def protected_spec(self, fullname, path=None, target=None):
            if fullname.partition(".")[0] in _BENCHFLOW_RUNTIME_TOP:
                return _RuntimeFinder.find_spec(fullname, path, target)
            return original(self, fullname, path, target)

        AssertionRewritingHook.find_spec = protected_spec


# --- BenchFlow pinned-plugin entry points: begin ---
# The protected runner (_verifier_runtime's run.py) carries this block's exact
# text, so a pinned release's plugin is loaded and registered the same way by
# the runner and by this guard (V2PT-2, V2PT-3). Standard library only.


def _bf_entry_parts(reference):
    """``module:attribute.chain [extras]`` as (module name, attribute names)."""
    module, _, attribute = reference.split("[", 1)[0].partition(":")
    return module.strip(), [part for part in attribute.strip().split(".") if part]


def _bf_entry_object(module, attributes):
    """The object a pytest11 entry point names: the module or an attribute
    chain in it (``grade = grading:checks``), as pluggy's loader resolves it."""
    plugin = module
    for attribute in attributes:
        plugin = getattr(plugin, attribute)
    return plugin


def _bf_record_distribution(manager, plugin, name, folders):
    """Record the distribution in *folders* that owns pytest11 entry point
    *name*, as pluggy's entry-point loading does, so pytest's
    ``required_plugins`` finds a pinned release BenchFlow registered itself.
    Only *folders* (the protected runtime's) are read; nothing else is."""
    try:
        import importlib.metadata as metadata

        from pluggy._manager import DistFacade
    except ImportError:
        return
    if any(known is plugin for known, _ in manager._plugin_distinfo):
        return
    for distribution in metadata.MetadataPathFinder.find_distributions(
        metadata.DistributionFinder.Context(path=list(folders))
    ):
        try:
            found = any(
                entry.group == "pytest11" and entry.name == name
                for entry in distribution.entry_points
            )
        except Exception:
            continue  # unrelated malformed metadata cannot stop pytest
        if found:
            manager._plugin_distinfo.append((plugin, DistFacade(distribution)))
            return


# --- BenchFlow pinned-plugin entry points: end ---


def _from_runtime(module):
    origin = getattr(module, "__file__", None)
    return bool(origin) and _in_runtime(os.path.realpath(origin))


def _load_runtime(manager):
    """Register each protected-runtime plugin under its pytest11 name.

    Before any later ``-p`` and before entry points are looked up, so the
    copy in test.sh's environment, whoever installed it, is never imported:
    pytest skips a name already registered.
    """
    for name, reference in sorted(_BENCHFLOW_RUNTIME_PLUGINS.items()):
        if name in _DISABLED or manager.is_blocked(name):
            continue
        module_name, attributes = _bf_entry_parts(reference)
        existing = manager.get_plugin(name)
        if existing is not None:
            if not _from_runtime(existing):
                raise Rejected(
                    "Verifier plugin trust rejected: "
                    + name
                    + " (loaded before the guard from outside BenchFlow's "
                    "protected verifier runtime)"
                )
            continue
        module = importlib.import_module(module_name)
        if not _from_runtime(module):
            raise Rejected(
                "Verifier plugin trust rejected: "
                + name
                + " ("
                + module_name
                + " was imported from outside BenchFlow's protected verifier "
                "runtime)"
            )
        # The entry point's own object (V2-R6-C2 on this path, V2PT-2), with
        # its owning distribution recorded for required_plugins (V2PT-3).
        plugin = _bf_entry_object(module, attributes)
        manager.register(plugin, name)
        _bf_record_distribution(manager, plugin, name, [_BENCHFLOW_RUNTIME_SITE])
        consider = getattr(manager, "consider_module", None)
        if consider is not None:
            consider(module)
    _register_runtime_once(manager)


def _register_runtime_once(manager):
    """A runtime plugin asked for again under another name is already there.

    pytest registers a plugin under the name it was asked for: the pytest11
    name above, the module name for a conftest's ``pytest_plugins`` or
    ``-p <module>``. pluggy refuses one module under two names, so a
    verifier conftest that lists a pinned plugin's module would stop pytest.
    The same module object, from the runtime, needs no second registration.
    """
    if not _BENCHFLOW_RUNTIME_PLUGINS:
        return
    original = getattr(manager, "register", None)
    if original is None or getattr(original, "_benchflow_runtime_once", False):
        return

    def register(plugin, name=None):
        if manager.is_registered(plugin) and _from_runtime(plugin):
            return manager.get_name(plugin)
        return original(plugin, name)

    setattr(register, "_benchflow_runtime_once", True)  # noqa: B010
    manager.register = register


def _install_root(path, record):
    """The directory test.sh installed *path* into, judged against *record*.

    Below a site directory: the environment's root, which must be new since
    the record, with every directory down to *path*. Elsewhere (``pip
    install --target``): the topmost directory above *path* that is new
    since the record, else *path* itself. None when the environment is not
    new.
    """
    parts = path.split("/")
    if "site-packages" in parts or "dist-packages" in parts:
        root = _environment_root(path)
        directory = os.path.dirname(path)
        while under(directory, root):
            if _in_record(directory, record):
                return None
            if directory == root:
                break
            directory = os.path.dirname(directory)
        return root
    root = path
    directory = os.path.dirname(path)
    while directory != os.path.dirname(directory) and _new_since_record(
        directory, record
    ):
        root = directory
        directory = os.path.dirname(directory)
    return root


def _installed_by_verifier(path):
    """Whether test.sh installed *path* itself, into an environment it made (#1170).

    *path* must be below a site directory, and neither it (nor what it
    reaches) nor any directory up to its environment's root may be in the
    pre-arm record: a file the agent left is never the verifier's, whatever
    a later chmod, chown or link did to it. The project files
    that chose the install must be the verifier's or the image's too.
    """
    parts = path.split("/")
    if "site-packages" not in parts and "dist-packages" not in parts:
        return False
    try:
        record = _record()
        if record is None or not _new_since_record(path, record):
            return False
        root = _install_root(path, record)
        return root is not None and _project_inputs_new(root, record)
    except Exception:
        return False


def untrusted_reason(path, blocked):
    """Why pytest must not load code from *path*, or None when it may.

    *path* is judged as Python imports it and as what it reaches: the
    spelling picks the byte-code caches Python reads, the resolved file is
    what it opens (V2-REDESIGN-R3-1). In the armed guard code loads only
    from BenchFlow's protected verifier runtime, from the pytest
    already running, or from the image: a file outside the agent's trees
    that is root's alone and unchanged since the guard was armed (one test.sh
    installed, even as root into the system's site-packages, changed then:
    R3-2), or a workspace file that still holds the image's bytes (the
    bytes, compared with the root-only copy, stand for owners and modes,
    which say nothing in a workspace the image ships world-writable). Either
    way every cache Python or pytest would read in the source's place must
    be the image's or the source's own compilation (R3-3).
    """
    if not path or not path.startswith("/"):
        return repr(path) + " is not an absolute path"
    real = os.path.realpath(path)
    if _BENCHFLOW_ARMED and (_in_runtime(real) or _running_pytest(real)):
        return None
    for candidate in dict.fromkeys((path, real)):
        image = _BENCHFLOW_ARMED and _image_bytes(candidate)
        for prefix in blocked:
            if under(candidate, prefix) and not image:
                return candidate + " is under " + prefix + ", " + _blocked_why()
        try:
            os.stat(candidate)
        except OSError as exc:
            return candidate + " cannot be read (" + type(exc).__name__ + ")"
        tests = bool(_BENCHFLOW_TESTS_DIR) and under(candidate, _BENCHFLOW_TESTS_DIR)
        if _BENCHFLOW_ARMED and not (image or tests) and _after_arm(candidate):
            return (
                candidate
                + " was installed or changed after the agent stopped; "
                + _INPUTS_WHY
            )
        if not _BENCHFLOW_OWNERSHIP or image:
            continue
        # A protected file is replaceable when any containing directory is
        # writable. Check both the spelling and the resolved path, including
        # symlink parents, rather than trusting the final inode alone.
        while True:
            try:
                st = os.stat(candidate)
            except OSError as exc:
                return candidate + " cannot be read (" + type(exc).__name__ + ")"
            problem = ownership_problem(candidate, st)
            if problem:
                return problem
            parent = os.path.dirname(candidate)
            if parent == candidate:
                break
            candidate = parent
    if _BENCHFLOW_ARMED and path.endswith(".py"):
        for spelling in dict.fromkeys((path, real)):
            if not _caches_harmless(spelling, real, blocked):
                return (
                    spelling + " has byte code Python or pytest would run in its "
                    "place that is not its source's compilation"
                )
    return None


def trusted(path, blocked):
    return untrusted_reason(path, blocked) is None


def trusted_module(module, blocked):
    parts = module.split(".")
    if not parts or not all(part.isidentifier() for part in parts):
        return False
    search = _BENCHFLOW_SEARCH_PATH
    for i in range(len(parts)):
        try:
            spec = PathFinder.find_spec(".".join(parts[: i + 1]), path=search)
        except Exception:
            return False
        if spec is None:
            return False
        # A namespace package has no origin, only search locations; a
        # builtin has a non-path origin. Requiring at least one real path
        # keeps both from passing by vacuous truth.
        locations = [spec.origin] if (spec.origin or "").startswith("/") else []
        locations += [str(p) for p in (spec.submodule_search_locations or [])]
        if not locations or not all(trusted(p, blocked) for p in locations):
            return False
        search = [str(p) for p in (spec.submodule_search_locations or [])]
        if i < len(parts) - 1 and not search:
            return False
    return True


def registrations():
    """Return ``(entry point, dist-info path or None)`` for every pytest11 entry."""
    if _BENCHFLOW_SEARCH_PATH is not None:
        from importlib.metadata import distributions

        return [
            (ep, getattr(dist, "_path", None))
            for dist in distributions(path=_BENCHFLOW_SEARCH_PATH)
            for ep in dist.entry_points
            if ep.group == "pytest11"
        ]
    try:
        from importlib.metadata import entry_points
    except ImportError:  # Python 3.7: pytest itself requires the backport there
        from importlib_metadata import entry_points

    try:
        # Discovery's search path is data: only distributions() takes one.
        eps = (
            None
            if _BENCHFLOW_SEARCH_PATH is not None
            else list(entry_points(group="pytest11"))
        )
    except TypeError:
        eps = None
    # CPython's filesystem-backed distributions expose the metadata path.
    # Unknown/custom distribution providers cannot establish provenance.
    if eps is not None and all(hasattr(ep, "dist") for ep in eps):
        return [(ep, getattr(ep.dist, "_path", None)) for ep in eps]
    # Before Python 3.10 an entry point does not name its distribution, so take
    # each one from the distribution that declares it, as pluggy loads them.
    try:
        from importlib.metadata import distributions
    except ImportError:
        from importlib_metadata import distributions
    search = {} if _BENCHFLOW_SEARCH_PATH is None else {"path": _BENCHFLOW_SEARCH_PATH}
    return [
        (ep, getattr(dist, "_path", None))
        for dist in distributions(**search)
        for ep in dist.entry_points
        if ep.group == "pytest11"
    ]


def discover(blocked, requested, builtin_names=(), ignore_blocked_registrations=False):
    found = registrations()
    if ignore_blocked_registrations:
        # Pre-verifier discovery only proposes ``-p`` names. A registration in
        # an agent-writable tree can never be one, and must not veto or drop a
        # protected plugin of the same name either; the runtime guard still
        # counts it wherever the verifier's pytest can actually see it.
        found = [
            (ep, registration)
            for ep, registration in found
            if registration is None
            or not any(
                under(candidate, prefix)
                for candidate in (
                    os.path.abspath(registration),
                    os.path.realpath(registration),
                )
                for prefix in blocked
            )
        ]
    eps = [ep for ep, _ in found]
    counts = {}
    for ep in eps:
        counts[ep.name] = counts.get(ep.name, 0) + 1
    names = []
    for ep, registration in found:
        # pytest loads entry points by name. A second registration with that
        # name can select different code from the one just inspected.
        if counts[ep.name] != 1:
            continue
        if registration is None or not trusted(os.path.abspath(registration), blocked):
            continue
        registration_file = os.path.join(str(registration), "entry_points.txt")
        if not trusted(os.path.abspath(registration_file), blocked):
            continue
        if trusted_module(ep.value.split(":")[0].strip(), blocked):
            names.append(ep.name)
    for name in requested:
        module = "_pytest." + name if name in builtin_names else name
        if name not in counts and trusted_module(module, blocked):
            names.append(name)
    trusted_names = sorted(set(names))
    return {
        "plugins": trusted_names,
        "rejected": sorted(
            (set(ep.name for ep in eps) | set(requested)) - set(trusted_names)
        ),
    }


def _module_paths(module):
    """The files and package directories *module* resolves to, without importing it."""
    found = []
    search = _BENCHFLOW_SEARCH_PATH
    parts = module.split(".")
    for i in range(len(parts)):
        spec = PathFinder.find_spec(".".join(parts[: i + 1]), path=search)
        if spec is None:
            break
        if (spec.origin or "").startswith("/"):
            found.append(spec.origin)
        search = [str(p) for p in (spec.submodule_search_locations or [])]
        found.extend(search)
        if not search:
            break
    return found


def _plugin_paths(name, entry_points, builtin_names):
    """The registrations and code that decide what pytest loads for *name*."""
    if not entry_points:
        return _module_paths(name)
    paths = []
    modules = []
    for ep, registration in registrations():
        if ep.name != name:
            continue
        if registration is None:
            return []
        paths.append(str(registration))
        paths.append(os.path.join(str(registration), "entry_points.txt"))
        modules.append(ep.value.split(":")[0].strip())
    if not modules:
        modules.append("_pytest." + name if name in builtin_names else name)
    for module in modules:
        paths.extend(_module_paths(module))
    return paths


def _installed_during_verification(
    names, entry_points, blocked, builtin_names, proven=True
):
    """Return the untrusted files showing the verifier itself installed the refused *names*.

    Every refused name needs an untrusted file behind it, and every untrusted
    file behind any of them must be test.sh's own install by the rules of
    _installed_by_verifier, owners aside: hardening recorded the agent's
    trees, the file is new since that record, so are the directories it was
    installed into, and the files that chose the install (_project_inputs_new)
    are new or the image's. An agent-made ``.venv`` or ``pyproject.toml``
    that test.sh's install reused is the agent's doing, not the task's
    (V2-R1-A-3, V2-R1-B-1); without a record nothing is evidence either way
    (_unrecorded, V2-R2-A-1). A refusal that no
    untrusted file explains (a name two trusted distributions register, a
    plugin installed nowhere) is not the verifier installing where the guard
    cannot trust. Returns an empty list when that does not hold or anything
    cannot be read: the refusal then stays a refusal, which counts as a
    failure. Either way the run is not scored; this only says whose problem
    it is. *proven* False drops the last condition: the files are test.sh's
    install, but the agent changed files that could have chosen it.
    """
    try:
        record = _record()
        if record is None:
            return []
        evidence = []
        for name in names:
            untrusted = [
                path
                for path in _plugin_paths(name, entry_points, builtin_names)
                if not trusted(path, blocked)
            ]
            if not untrusted:
                return []
            for path in untrusted:
                if not _new_since_record(path, record):
                    return []
                root = _install_root(path, record)
                if root is None or (proven and not _project_inputs_new(root, record)):
                    return []
            evidence.extend(untrusted)
        return evidence
    except Exception:
        return []


# A ``*.dist-info`` directory's name: ``{name}-{version}.dist-info``, the name
# with each ``-`` written ``_``. Nothing else reaches a marker or a message.
def _unrecorded(names, entry_points, blocked, builtin_names):
    """Why no record can say who put the refused *names* there, and their files.

    Returns ``(why, untrusted files)`` when there is no usable record and
    every refused name has an untrusted file behind it, each in the agent's
    trees: the case the record decides (_installed_during_verification).
    Without it no one can tell test.sh's own install (USACO's ``uv add``
    into /app/.venv) from a file the agent left, and the record is missing
    for reasons the agent does not control: a verifier that is not root, an
    image without python3 whose find or stat failed, a probe of uv's state
    that failed (V2-R2-A-1). The refusal still counts as a failure, so the
    agent gains nothing by it, but it is not blamed on the agent. None when
    there is a record or anything cannot be read.
    """
    try:
        if _record() is not None:
            return None
        why = "the record did not finish" if _BENCHFLOW_RECORD else "none was taken"
        evidence = []
        for name in names:
            untrusted = [
                path
                for path in _plugin_paths(name, entry_points, builtin_names)
                if not trusted(path, blocked)
            ]
            if not untrusted:
                return None
            for path in untrusted:
                forms = (os.path.abspath(path), os.path.realpath(path))
                if not any(under(f, prefix) for f in forms for prefix in blocked):
                    return None
            evidence.extend(untrusted)
        return why, evidence
    except Exception:
        return None


def _why_refused(name, entry_points, blocked, builtin_names):
    """Say what made the guard refuse *name*; never raise."""
    try:
        if entry_points:
            found = [reg for ep, reg in registrations() if ep.name == name]
            if len(found) > 1:
                return (
                    "registered " + str(len(found)) + " times, so the name does "
                    "not say which code pytest loads"
                )
            if found and found[0] is None:
                return "registered by a distribution whose files cannot be checked"
        paths = _plugin_paths(name, entry_points, builtin_names)
        if not paths:
            return "not installed where this pytest looks"
        for path in paths:
            if not trusted(path, blocked):
                return untrusted_reason(path, blocked) or path + " is not trusted"
        return "not trusted"
    except Exception as exc:
        return "cannot inspect: " + type(exc).__name__


def _validate(names, *, entry_points=True):
    from _pytest.config import builtin_plugins

    # The protected runtime's plugins are registered from there before any
    # lookup by name (_load_runtime); a copy elsewhere is never imported.
    names = set(names) - {__name__} - set(_BENCHFLOW_RUNTIME_PLUGINS)
    if not entry_points:
        # A module name the runtime holds is imported from there alone
        # (_RuntimeFinder), never from test.sh's sys.path.
        names = {n for n in names if n.partition(".")[0] not in _BENCHFLOW_RUNTIME_TOP}
    if not names:
        return
    # pytest maps its builtin short names to _pytest.<name> before importing.
    modules = ["_pytest." + name if name in builtin_plugins else name for name in names]
    blocked = tuple(
        candidate
        for path in _BENCHFLOW_BLOCKED
        for candidate in (os.path.abspath(path), os.path.realpath(path))
    )
    try:
        if entry_points:
            # CLI -p checks entry-point aliases BEFORE falling back to builtin
            # module names, including optional builtins such as pytester.
            receipt = discover(blocked, names, builtin_plugins)
            refused = names - set(receipt["plugins"])
        else:
            # pytest_plugins / PYTEST_PLUGINS import module names directly; a
            # same-named trusted entry point does not authorize that module.
            refused = {name for name in modules if not trusted_module(name, blocked)}
    except Exception as exc:
        # Files and metadata on sys.path may be planted; what cannot be
        # inspected cannot be trusted.
        raise Rejected(
            "Verifier plugin trust rejected: "
            + ", ".join(sorted(names))
            + " (cannot inspect: "
            + type(exc).__name__
            + ": "
            + str(exc)
            + ")"
        ) from exc
    if refused:
        refused = sorted(refused)
        reasons = [
            name + ": " + _why_refused(name, entry_points, blocked, builtin_plugins)
            for name in refused
        ]
        message = (
            "Verifier plugin trust rejected: "
            + ", ".join(refused)
            + " ("
            + "; ".join(reasons)
            + ")"
        )
        installed = _installed_during_verification(
            refused, entry_points, blocked, builtin_plugins
        )
        if installed:
            _mark(
                "installed",
                "".join(
                    path
                    + "\t"
                    + (untrusted_reason(path, blocked) or "untrusted")
                    + "\n"
                    for path in installed
                ),
            )
            message += (
                "; the verifier installed it after the agent stopped, "
                "so the run is not scored. A task problem: pin its release in "
                "[verifier] pytest_plugin_releases or install it into the "
                "image; `bench check` flags this task"
            )
        else:
            unrecorded = None
            fresh = _installed_during_verification(
                refused, entry_points, blocked, builtin_plugins, proven=False
            )
            if fresh:
                # test.sh installed every refused file after the agent stopped,
                # but the agent changed files that could have chosen it, so
                # the guard cannot vouch for it. Not the agent's plugin: the
                # task's unpinned install (V2-R1-B-2). Still a failure.
                _mark(
                    "unproven",
                    "".join(
                        path
                        + "\t"
                        + (untrusted_reason(path, blocked) or "untrusted")
                        + "\n"
                        for path in fresh
                    ),
                )
                message += (
                    "; the verifier installed it after the agent stopped, from "
                    "inputs the agent could have changed"
                )
            else:
                unrecorded = _unrecorded(
                    refused, entry_points, blocked, builtin_plugins
                )
            if unrecorded:
                why, paths = unrecorded
                _mark(
                    "unrecorded",
                    why
                    + "\n"
                    + "".join(
                        path
                        + "\t"
                        + (untrusted_reason(path, blocked) or "untrusted")
                        + "\n"
                        for path in paths
                    ),
                )
                message += (
                    "; with no record of the agent's files (" + why + "), the "
                    "guard cannot tell whether the verifier installed it after "
                    "the agent stopped or the agent left it"
                )
        raise Rejected(message)


def _no_cache_folder():
    """A folder that is never made, in the guard's own folder: root's, made
    at hardening under a random name, so the agent can neither make it nor
    write into the folder above it."""
    return os.path.join(os.path.dirname(_BENCHFLOW_KEY_FILE), ".no-pycache")


def _keep_byte_code_out():
    """Write no byte code in this pytest, and read none: every module
    imported from here on compiles from its source.

    A cache this process wrote is one the guard cannot tell from a planted
    one: pytest's assertion rewriter wrote one for each plugin it imported,
    and the guard then refused that plugin (Terminal-Bench 4.1's anyio).
    Caches are read only under :func:`_no_cache_folder`, as BenchFlow's
    protected runner does (``-X pycache_prefix``), never under a folder the
    environment names: ``python3 -I`` leaves PYTHONPYCACHEPREFIX unapplied,
    and the folder it names (the task's verifier environment, or test.sh)
    may be one the agent can write.
    """
    sys.dont_write_bytecode = True
    if not hasattr(sys, "pycache_prefix"):  # Python 3.7: no prefix to set
        return
    folder = _no_cache_folder()
    if os.path.lexists(folder):
        raise Rejected(
            "Verifier plugin trust: " + folder + " exists; the guard reads byte "
            "code only from a folder that is never made"
        )
    sys.pycache_prefix = folder


def pytest_addhooks(pluginmanager):
    """Validate actual invocation -p imports before pytest visits later aliases."""
    try:
        if _BENCHFLOW_ARMED:
            _keep_byte_code_out()
        # pytest reads this after the -p plugins, the guard first among them:
        # a .pth or a startup hook that cleared it cannot turn autoload back on
        # for registrations nothing has checked (V2-REDESIGN-R3-4).
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        _install_runtime_finder()
        config = pluginmanager.get_plugin("pytestconfig")
        # pytest < 5.1 (4.6.x aside) has no invocation_params; its parse() keeps
        # the command line in _origargs before importing any -p plugin.
        params = getattr(config, "invocation_params", None)
        args = [
            *shlex.split(os.environ.get("PYTEST_ADDOPTS", "")),
            *(params.args if params is not None else config._origargs),
        ]
        names = list(_BENCHFLOW_REQUESTED)
        index = 0
        while index < len(args):
            arg = args[index]
            index += 1
            if arg == "-p":
                if index == len(args):
                    raise Rejected("Verifier plugin trust: missing -p argument")
                name = args[index]
                index += 1
            elif arg.startswith("-p"):
                name = arg[2:]
            else:
                continue
            name = name.strip()
            if name.startswith("no:"):
                _DISABLED.add(name[3:].strip())
            else:
                names.append(name)
        _validate(names)
        _validate(
            [
                name.strip()
                for name in os.environ.get("PYTEST_PLUGINS", "").split(",")
                if name.strip()
            ],
            entry_points=False,
        )
    except Exception as exc:
        _stopping(exc)
        raise


def _present(name):
    """Whether this pytest has optional plugin *name* installed: a pytest11 registration.

    Every optional name is an entry point's name (discovery's, or ``ctrf``
    inferred from ``--ctrf``), so a module that only shares the name (the
    workspace's ``benchmark.py`` on test.sh's PYTHONPATH, the standard
    library's ``asyncio``) is not that plugin and is never loaded as one.
    Only *where* it is: whether it may be loaded is _validate's question.
    Registrations that cannot be inspected count as present, so _validate
    judges (and refuses) them rather than the guard skipping them.
    """
    try:
        return any(ep.name == name for ep, _ in registrations())
    except Exception:
        return True


def _load_optional(pluginmanager, disabled=()):
    """Load each optional plugin this pytest has installed, refusing untrusted ones (#1175).

    As ``-p`` would, by its entry point name first, once pytest has
    registered the guard with all its hooks (so a loaded plugin's own
    ``pytest_plugins`` are checked too) and before any plugin of the
    command line. One the command line blocks (``-p no:<name>``, in
    *disabled* or already applied), one already loaded, and one absent from
    this pytest are skipped.
    """
    import_plugin = pluginmanager.import_plugin
    code = getattr(import_plugin, "__code__", None)
    by_entry_point = code is not None and "consider_entry_points" in code.co_varnames
    for name in _BENCHFLOW_OPTIONAL:
        if name in _BENCHFLOW_RUNTIME_PLUGINS:
            continue  # registered from the protected runtime (_load_runtime)
        if name in disabled or pluginmanager.is_blocked(name):
            continue
        if pluginmanager.get_plugin(name) is not None:
            continue
        if not _present(name):
            continue
        _validate([name])
        try:
            if by_entry_point:
                import_plugin(name, consider_entry_points=True)
            else:
                # pytest before 4.4 has no such argument.
                import_plugin(name)
        except Exception as exc:
            global _OPTIONAL_FAILURE
            _OPTIONAL_FAILURE = exc
            raise


def _check_registered(plugin, manager):
    """Refuse a plugin module registered from anywhere the guard does not trust.

    Every plugin pytest registers, not only the names the guard was asked
    about (an entry point autoloaded, a module a plugin registers itself:
    V2-REDESIGN-R3-4). Conftest modules (registered under their path) are
    V1's, objects that are not modules are pytest's own.
    """
    if not _BENCHFLOW_ARMED or not isinstance(plugin, type(sys)):
        return
    name = manager.get_name(plugin) or ""
    if "/" in name or plugin is sys.modules.get(__name__):
        return
    origin = getattr(plugin, "__file__", None)
    if not origin:
        return  # a builtin or namespace module holds no file to load
    blocked = tuple(
        candidate
        for path in _BENCHFLOW_BLOCKED
        for candidate in (os.path.abspath(path), os.path.realpath(path))
    )
    reason = untrusted_reason(str(origin), blocked)
    if reason:
        raise Rejected("Verifier plugin trust rejected: " + name + " (" + reason + ")")


def pytest_plugin_registered(plugin, manager):
    """Check declared dependencies before pytest's consider_module imports them."""
    try:
        _begin_check()
        if plugin is sys.modules.get(__name__):
            # pytest calls this for the guard itself only after pluggy
            # accepted every one of its hooks. The ``checking`` marker is
            # already there, so the run is never left with ``registered``
            # alone while plugins are still being inspected.
            _mark("registered")
            _unmark("loading")
            _load_runtime(manager)
            _load_optional(manager, _DISABLED)
        else:
            _check_registered(plugin, manager)
        dependencies = getattr(plugin, "pytest_plugins", ())
        if isinstance(dependencies, str):
            dependencies = dependencies.split(",") if dependencies else ()
        if isinstance(dependencies, (tuple, list)):
            _validate(dependencies, entry_points=False)
        _end_check()
    except Exception as exc:
        _stopping(exc)
        raise


if __name__ == "__main__":
    prefixes = json.loads(sys.argv[1])
    blocked = tuple(
        candidate
        for path in prefixes
        for candidate in (os.path.abspath(path), os.path.realpath(path))
    )
    try:
        requested = json.loads(sys.argv[2]) if len(sys.argv) > 2 else []
        # argv[3]: the policy the armed guard gets from hardening.
        policy = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        _BENCHFLOW_OWNERSHIP = bool(policy.get("ownership", True))
        if "_bf_paths" in globals():
            pythonpath = policy.get("pythonpath")
            if pythonpath is None:
                pythonpath = sys.argv[4] if len(sys.argv) > 4 else ""
            # integ's policy carries the trusted PYTHONPATH as a list of
            # entries, PT's as one string; both are data, never sys.path.
            entries = (
                [p for p in pythonpath if isinstance(p, str)]
                if isinstance(pythonpath, list)
                else [p for p in pythonpath.split(":") if p]
            )
            # The pre-agent search path of the Python test.sh will run (the
            # first on the verifier's PATH with a record), else root's
            # python3's (PT-R13-A-2).
            recorded = globals()["_bf_paths"]
            if "verifier_path" in policy and "_bf_paths_for" in globals():
                recorded = globals()["_bf_paths_for"](policy["verifier_path"])
            _BENCHFLOW_SEARCH_PATH = list(dict.fromkeys(entries + recorded["path"]))
        elif "pythonpath" in policy:
            # Without PT's pre-agent record (no protected kit), integ's rule:
            # the verifier's PYTHONPATH entries first, as in its interpreter;
            # this interpreter's own sys.path stays as -I left it.
            _BENCHFLOW_SEARCH_PATH = [
                entry for entry in policy["pythonpath"] if entry.startswith("/")
            ] + sys.path
        if policy.get("agent_gids") is not None:
            _BENCHFLOW_AGENT_GIDS = tuple(int(g) for g in policy["agent_gids"])
        print(
            json.dumps(discover(blocked, requested, ignore_blocked_registrations=True))
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)
