"""BenchFlow's protected runtime tree in a sandbox (C8).

Every program BenchFlow runs as root in a sandbox after the agent's turn has
started comes from places the sandbox user cannot write. This module names
those places and holds the shell text that prepares them; the sandbox's exec
route (:meth:`benchflow.sandbox._base.BaseSandbox.exec_route`) uses them.

The tree
--------

:data:`RUNTIME_ROOT` (``/opt/runtime``) is a reserved path: no workspace
ownership grant ever covers it (``setup_sandbox_user`` refuses a workspace of
``/`` or inside the tree, and keeps a workspace above it, ``/opt``, root's and
sticky so the agent user can add files there but never rename the tree). It
already holds BenchFlow's runtimes that root installs and runs: the pinned uv
(``uv/``), uv's managed CPython (``python/``), Node and the JS agents
(``node/``, ``js-agents/``, ``bin/``), the cached proxy venv. Root checks each
of those with :func:`benchflow._root_only.root_only_tree_check` (or
:func:`~benchflow._root_only.root_only_path_check` for one program) in the
same command, before it runs anything from it, version probes included.

The kit, :data:`KIT` (``/opt/runtime/rt``), is what root's helpers run on
once the agent has had a turn. Prepared by root before the agent's first
turn, from root-only files of the image (copies, never links out):

- ``bin/sh`` (and ``bin/bash`` when the image has one): the shells root's
  plumbing starts by absolute path;
- ``py/python3`` (:data:`KIT_PYTHON`): the image's Python with its standard
  library (no site-packages) under ``lib/python3.X`` (the modules root's
  helpers import compiled at the build, :data:`KIT_PRECOMPILED_MODULES`), and
  any shared library it loads from outside the system library folders (a
  Python image's ``libpython`` in /usr/local/lib) under ``lib/``, started
  through the system loader with that folder; a wrapper that always adds
  ``-I -S``. Not on root's PATH: BenchFlow's helpers name it by path, and a
  bare ``python3`` stays the task's (the plugin guard inspects that one).
  Absent when the image has no Python 3 root may copy;
- ``bin/<tool>`` for :data:`KIT_TOOLS` (the coreutils, ``find``, ``tar``,
  ``setpriv`` and the rest root's helpers call), each a link to its copy
  under ``libexec/``;
- ``RECORD``: what the kit holds, and the image folders of
  :data:`SYSTEM_FOLDERS` (``/usr/local/sbin`` to ``/bin``) the preparation
  proved root's alone (each folder, every folder above it, every file in it
  and every hop of every link in it); ``MANIFEST``: the sha256 of every file in the kit.

Root's PATH once the kit exists is ``/opt/runtime/rt/bin`` followed by the
proven folders, never one the proof did not pass and never the image's PATH.

:data:`RUN_DIR` (``/opt/runtime/run``, mode 700) is where root stages a
launch's environment file (one fresh name per launch, created exclusively).

V2's verifier environment (contract)
------------------------------------

:data:`VERIFIER_DIR` (``/opt/runtime/verifier``) is made empty, root's,
mode 755, by the same preparation, before the agent's turn. The pytest plugin
guard (V2) unpacks its pinned pytest wheels into ``VERIFIER_DIR/site`` there,
at setup (before the agent's turn), and then:

- runs the verifier's pytest with :data:`KIT_PYTHON` ``-I -S``, from ``/`` or
  a root-only folder, adding ``VERIFIER_DIR/site`` to ``sys.path`` in its own
  ``-c`` text (``-S``: no ``.pth`` file runs); the task's tests are imported
  from where the task puts them, as before;
- checks ``VERIFIER_DIR`` with :func:`root_only_tree_check` in the same
  command as each run; the kit itself is checked against ``MANIFEST`` when
  a sandbox generation first finds it (:func:`kit_load_command`);
- treats :func:`runtime_kit_python` returning None (an image with no Python
  root could copy, or no sandbox user, so no kit) as "no protected Python":
  it refuses with a message, never falls back to the image's ``python3``.

The Python API: :data:`KIT`, :data:`KIT_PYTHON`, :data:`VERIFIER_DIR`,
:func:`runtime_kit_python` (``[python, "-I", "-S", "-B"]`` or None) and
``sandbox.runtime_kit`` (a :class:`RuntimeKit` or None).
"""

from __future__ import annotations

import posixpath
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from benchflow._image_python_paths import CAPTURE as CAPTURE_IMAGE_PYTHON_PATHS
from benchflow._image_python_paths import CAPTURE_ONE as CAPTURE_ONE_IMAGE_PYTHON_PATHS
from benchflow._image_python_paths import (
    GUARD_DIRS_SH,
    GUARD_ID_SH,
    GUARD_SEARCH,
    SHEBANG_PY_SH,
)
from benchflow._image_python_paths import SEARCH_ENV as IMAGE_PYTHON_SEARCH_ENV
from benchflow._kit_seal import MANIFEST_WALK, SEAL_WALK, TOOLS_WALK
from benchflow._root_only import ROOT_CHAIN_FUNCS, root_only_tree_check

__all__ = [
    "KIT",
    "KIT_BASH",
    "KIT_BIN",
    "KIT_CHANNEL_PYTHON",
    "KIT_PYTHON",
    "KIT_SH",
    "KIT_TOOLS",
    "RUNTIME_ROOT",
    "RUN_DIR",
    "SYSTEM_FOLDERS",
    "VERIFIER_DIR",
    "RuntimeKit",
    "kit_load_command",
    "kit_prepare_command",
    "parse_kit_record",
    "runtime_custody_check",
    "runtime_kit_python",
    "workspace_grant_problem",
]

RUNTIME_ROOT = "/opt/runtime"
KIT = f"{RUNTIME_ROOT}/rt"
KIT_BIN = f"{KIT}/bin"
KIT_SH = f"{KIT_BIN}/sh"
KIT_BASH = f"{KIT_BIN}/bash"
# Not on root's PATH: a bare ``python3`` there stays the task's own (the
# plugin guard's discovery inspects it on purpose); BenchFlow's own helpers
# name this one by path.
KIT_PYTHON = f"{KIT}/py/python3"
# The stdlib-only command channel may use a separate authenticated CPython.
KIT_CHANNEL_PYTHON = f"{KIT}/py/channel-python3"
RUN_DIR = f"{RUNTIME_ROOT}/run"
#: root's HOME on the route: empty and root's alone (never the image's HOME).
ROOT_HOME = f"{RUN_DIR}/home"
VERIFIER_DIR = f"{RUNTIME_ROOT}/verifier"
KIT_RECORD = f"{KIT}/RECORD"
#: The image's values root alone could have written before the agent's turn.
KIT_IMAGE_ENV_OK = f"{KIT}/IMAGE_ENV_OK"

#: The image folders root's PATH may keep after the agent's turn (after the
#: kit's bin, in this order), each only when the preparation proved it
#: root's alone; /usr/local is no exception and gets no pass.
SYSTEM_FOLDERS = (
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
)

#: Folders a workspace may not be, or hold: the agent user would own the
#: programs and libraries root's helpers load.
_SYSTEM_TREES = (
    "/bin",
    "/sbin",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/usr/bin",
    "/usr/sbin",
    "/usr/lib",
    "/usr/lib32",
    "/usr/lib64",
    "/usr/libexec",
    "/etc",
)

#: Folders the loader searches that a workspace may hold but never be or be
#: inside (ld.so.conf.d's libc.conf and multiarch file list them on most
#: images). A workspace above them (``/usr/local``) is granted without them
#: (:func:`workspace_grant_command`, PT-C8-R9-4): they and the files directly
#: in them stay root's, so no library the agent writes is ever where the
#: loader, or a later ldconfig, looks; their other subfolders (Python's
#: site-packages, node_modules) are granted.
_LOADER_TREES = ("/usr/local/lib", "/usr/local/lib64")

#: The programs root's helpers call after the agent's turn (the kill and
#: release scripts, verifier hardening, evidence capture, the launch
#: bootstrap, uploads and env files). A program the image lacks is left out.
KIT_TOOLS = (
    "sh",
    "bash",
    "env",
    "cat",
    "cp",
    "mv",
    "rm",
    "rmdir",
    "ln",
    "mkdir",
    "mktemp",
    "chmod",
    "chown",
    "chgrp",
    "stat",
    "readlink",
    "realpath",
    "dirname",
    "basename",
    "ls",
    "find",
    "xargs",
    "grep",
    "sed",
    "awk",
    "tr",
    "cut",
    "sort",
    "uniq",
    "head",
    "tail",
    "wc",
    "tee",
    "touch",
    "sleep",
    "timeout",
    "kill",
    "ps",
    "id",
    "setpriv",
    "su",
    "tar",
    "gzip",
    "base64",
    "sha256sum",
    "date",
    "uname",
    "du",
    "df",
    "sync",
    "nohup",
    "setsid",
    "true",
    "false",
    "printf",
    "test",
    "stty",
    "mkfifo",
    "od",
    "cksum",
    "getent",
)

# Top-level entries of a standard library the kit never copies: installed
# packages (``-S`` never imports them), the test suite, GUI and packaging
# tools, and the static library and build files in ``config-*``.
_STDLIB_SKIP = (
    "site-packages",
    "dist-packages",
    "test",
    "tests",
    "idlelib",
    "tkinter",
    "turtledemo",
    "ensurepip",
    "lib2to3",
    "__phello__",
    "config-*",
)

# What the kit's Python must import from ``-I -S`` to be kept: the modules
# root's helpers use (capture, unpack, hardening probes).
_PYTHON_SMOKE = (
    "import sys, os, json, fnmatch, hashlib, stat, tarfile, zipfile, shutil, "
    "tempfile, base64, re; print(sys.prefix)"
)

# The standard-library modules root's helpers import on the kit's Python
# (the smoke's, the exec channel's helper's, and those of every script
# _root_python runs, recorded on Terminal-Bench 2 trials). The build
# compiles them, and the modules they import in turn, into the kit (PERF-G).
# The kit is built in every trial's sandbox (no per-image cache), so this
# is paid per trial: +0.44 s at sandbox start on one CPU, against -0.62 s
# between the agent's end and the verifier (stub TB2 trials, PERF-G). The
# kit's Python runs with -B and an image's own .pyc files
# are often stale for its sources, so every helper start compiled them
# again (json and pathlib ~70 ms, importlib.metadata ~160 ms on one CPU,
# about ten starts between the agent's end and the verifier). A module
# left out is compiled in memory at each import, as before.
KIT_PRECOMPILED_MODULES = (
    "ast",
    "base64",
    "binascii",
    "contextlib",
    "errno",
    "fnmatch",
    "glob",
    "hashlib",
    "hmac",
    "importlib.machinery",
    "importlib.metadata",
    "importlib.util",
    "json",
    "os",
    "pathlib",
    "re",
    "selectors",
    "shlex",
    "shutil",
    "signal",
    "stat",
    "struct",
    "subprocess",
    "sysconfig",
    "tarfile",
    "tempfile",
    "time",
    "traceback",
    "zipfile",
)

# Run on the kit's Python once the smoke kept it, before the seal (so the
# MANIFEST covers what it writes): each module above is imported, and every
# loaded module whose source is in the kit's standard library is compiled
# next to it. The files are checked-hash .pyc: Python uses one only while
# its source's hash matches, so the code that runs is always the source's.
# A failure leaves the module to be compiled at import, as before.
_PYTHON_PRECOMPILE = (
    "import importlib, importlib.util, py_compile, sys\n"
    f"for name in {list(KIT_PRECOMPILED_MODULES)!r}:\n"
    "    try:\n"
    "        importlib.import_module(name)\n"
    "    except Exception:\n"
    "        pass\n"
    "lib = sys.prefix.rstrip('/') + '/lib/'\n"
    "for module in list(sys.modules.values()):\n"
    "    source = getattr(module, '__file__', None) or ''\n"
    "    if source.startswith(lib) and source.endswith('.py'):\n"
    "        py_compile.compile(\n"
    "            source,\n"
    "            cfile=importlib.util.cache_from_source(source),\n"
    "            invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,\n"
    "            doraise=False,\n"
    "            quiet=2,\n"
    "        )\n"
)


@dataclass(frozen=True)
class RuntimeKit:
    """The kit a sandbox holds, as its ``RECORD`` says.

    *trusted*: the system folders proven root's alone, in PATH order;
    *python*: the kit has a Python; *bash*: the kit has bash.
    """

    trusted: tuple[str, ...]
    python: bool
    bash: bool

    @property
    def path(self) -> str:
        """Root's PATH on the kit: its own bin first, then the proven
        system folders; never empty."""
        return ":".join((KIT_BIN, *self.trusted))

    @property
    def shell(self) -> str:
        return KIT_SH

    @property
    def bash_shell(self) -> str | None:
        return KIT_BASH if self.bash else None


def parse_kit_record(text: str) -> RuntimeKit | None:
    """The :class:`RuntimeKit` a ``RECORD`` (between its markers) describes,
    or None when *text* holds no complete record."""
    lines = (text or "").splitlines()
    try:
        start = lines.index("BF kit-record")
        end = lines.index("BF kit-end", start)
    except ValueError:
        return None
    fields: dict[str, str] = {}
    for line in lines[start + 1 : end]:
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    if fields.get("kit") != "1":
        return None
    trusted = tuple(
        d
        for d in fields.get("trusted", "").split(":")
        if d in SYSTEM_FOLDERS or (d.startswith(f"{KIT}/view/") and ".." not in d)
    )
    return RuntimeKit(
        trusted=trusted,
        python=fields.get("python") == "1",
        bash=fields.get("bash") == "1",
    )


def runtime_kit_python(sandbox: Any) -> list[str] | None:
    """``[KIT_PYTHON, "-I", "-S", "-B"]`` when *sandbox*'s kit has a Python, else None."""
    kit = getattr(sandbox, "runtime_kit", None)
    if isinstance(kit, RuntimeKit) and kit.python:
        return [KIT_PYTHON, "-I", "-S", "-B"]
    return None


def _is_under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def workspace_grant_problem(workspace: str) -> str | None:
    """Why the sandbox user may not be granted *workspace*, or None.

    A workspace of ``/``, inside :data:`RUNTIME_ROOT`, or equal to or above a
    system folder would hand the agent user programs root runs.
    """
    # Normalized as the kernel resolves it: ``..`` and ``.`` parts, and any
    # number of leading slashes (normpath keeps exactly two), must not move a
    # workspace out of the reservation (FF-install-C1).
    ws = "/" + posixpath.normpath("/" + workspace).lstrip("/")
    if ws == "/":
        return (
            "the workspace / would give the sandbox user every folder, "
            f"BenchFlow's runtime tree ({RUNTIME_ROOT}) and the system "
            "folders included; set the task's workdir to a folder of its own "
            "(for example /app)"
        )
    if _is_under(ws, RUNTIME_ROOT):
        return (
            f"the workspace {ws} is inside BenchFlow's runtime tree "
            f"({RUNTIME_ROOT}), which only root may write; set the task's "
            "workdir to a folder of its own (for example /app)"
        )
    for system in _SYSTEM_TREES:
        if _is_under(system, ws):
            return (
                f"the workspace {ws} holds the system folder {system}, whose "
                "programs BenchFlow runs as root; set the task's workdir to a "
                "folder of its own (for example /app)"
            )
    for loader in _LOADER_TREES:
        below = ws[len(loader) + 1 :].split("/", 1)[0] if _is_under(ws, loader) else ""
        if ws == loader or "-linux-" in below:
            return (
                f"the workspace {ws} is a folder the system loader searches "
                f"({loader}), so a library the agent wrote there would load "
                "into root's programs; set the task's workdir to a folder of "
                "its own (for example /app)"
            )
    return None


def _reserved_in(ws: str) -> list[str]:
    """The folders under *ws* its grant leaves root's: the runtime tree's
    top folder, and the loader's folders."""
    reserved = []
    if _is_under(RUNTIME_ROOT, ws):
        reserved.append(
            ws.rstrip("/") + "/" + RUNTIME_ROOT[len(ws) :].strip("/").split("/")[0]
        )
    reserved += [d for d in _LOADER_TREES if _is_under(d, ws) and d != ws]
    return reserved


def workspace_grant_command(workspace: str, owner: str) -> str:
    """sh: give *owner* the workspace. A workspace that holds a reserved
    folder (``/opt`` holds the runtime tree, ``/usr/local`` the loader's
    folders, see :func:`_reserved_in`) stays root's and sticky, and every
    entry but the reserved ones is granted: the user may add and change its
    own files there, never rename a reserved folder. A loader folder stays
    root's with the files directly in it; its subfolders other than the
    loader's own (``*-linux-*``) are granted (PT-C8-R9-4).

    A provider daemon's folder directly in the workspace
    (``SANDBOX_RUNTIME_STATE_PATHS``: Daytona's daemon keeps each exec's
    script, output and exit code in ``.daytona`` in root's home) is reserved
    too when the sandbox has it as a folder: it stays root's, without group
    or other write. The daemon runs as root and still writes there; the
    agent can put nothing there for test.sh to read while the review leaves
    it out (#1232)."""
    from benchflow.sandbox.protocol import SANDBOX_RUNTIME_STATE_PATHS

    ws = "/" + workspace.strip("/")
    reserved = _reserved_in(ws)
    if any("/" in r[len(ws.rstrip("/")) + 1 :] for r in reserved):
        raise ValueError(f"the workspace {ws} holds a reserved folder below its top")
    held = [ws.rstrip("/") + "/" + name for name in SANDBOX_RUNTIME_STATE_PATHS]
    present = " || ".join(
        f"{{ [ -d {q} ] && [ ! -L {q} ]; }}" for q in map(shlex.quote, held)
    )
    return (
        f"if {present}; then {_grant_command(ws, owner, reserved, held)}; "
        f"else {_grant_command(ws, owner, reserved, [])}; fi"
    )


def _grant_command(ws: str, owner: str, reserved: list[str], held: list[str]) -> str:
    """:func:`workspace_grant_command`'s sh for *reserved* folders and the
    daemon folders *held* (each kept root's when it is a folder)."""
    q_ws = shlex.quote(ws)
    q_owner = shlex.quote(f"{owner}:{owner}")
    if not reserved and not held:
        return f"chown -R {q_owner} {q_ws}"
    skip = " ".join(f"! -path {shlex.quote(r)}" for r in (*reserved, *held))
    parts = [
        f"chown 0:0 {q_ws} && chmod 1777 {q_ws} && "
        f"find {q_ws} -mindepth 1 -maxdepth 1 {skip} -exec chown -R {q_owner} {{}} +"
    ]
    for loader in (r for r in reserved if r in _LOADER_TREES):
        q = shlex.quote(loader)
        parts.append(
            # Made first when missing: in the now sticky workspace the user
            # could otherwise create it.
            f"{{ [ -e {q} ] || [ -L {q} ] || mkdir -m 755 {q}; }} && "
            f"if [ -d {q} ] && [ ! -L {q} ]; then chown 0:0 {q} && chmod go-w {q} && "
            f"find {q} -mindepth 1 -maxdepth 1 -type d ! -name '*-linux-*' "
            f"-exec chown -R {q_owner} {{}} +; fi"
        )
    for folder in held:
        q = shlex.quote(folder)
        parts.append(
            f"if [ -d {q} ] && [ ! -L {q} ]; then chown 0:0 {q} && chmod go-w {q}; fi"
        )
    return " && ".join(parts)


def _sys_lib(var: str) -> str:
    """sh test: the folder in ``$var`` is a system library folder."""
    return (
        f'case "${var}" in /lib|/lib64|/lib32|/libx32|/usr/lib|/usr/lib64|'
        "/usr/lib32|/usr/libx32|/lib/*-linux-*|/usr/lib/*-linux-*) true;; "
        "*) false;; esac"
    )


# PERF-G's compile (_PYTHON_PRECOMPILE), each module compiled once (PERF-H):
# the import itself writes the checked-hash .pyc of every module it compiles
# from the kit's standard library, the bytes py_compile writes for it (the
# same code object, source hash and file mode), instead of compiling it for
# the import and again for py_compile. Modules loaded before the hook (the
# interpreter's start-up) and any the hook could not write go through
# PERF-G's py_compile pass, as before; without importlib's helpers (another
# Python's internals) the hook is not installed and that pass does all.
_PYTHON_PRECOMPILE_ONCE = (
    "import importlib, importlib.util, importlib.machinery as m, os, sys\n"
    # It runs before the smoke: only a Python whose prefix is the kit (argv
    # 1) compiles, so nothing is written outside the kit (PERF-H r2).
    "if sys.prefix.rstrip('/') != sys.argv[1]:\n"
    "    raise SystemExit(0)\n"
    "lib = sys.prefix.rstrip('/') + '/lib/'\n"
    "done = set()\n"
    "try:\n"
    "    from importlib import _bootstrap_external as be\n"
    "    be._code_to_hash_pyc, be._write_atomic, be._calc_mode\n"
    "\n"
    "    class Once(m.SourceFileLoader):\n"
    "        def source_to_code(self, data, path, *, _optimize=-1):\n"
    "            code = super().source_to_code(data, path, _optimize=_optimize)\n"
    "            if (isinstance(path, str) and path.startswith(lib)\n"
    "                    and path.endswith('.py') and _optimize == -1):\n"
    "                try:\n"
    "                    cfile = importlib.util.cache_from_source(path)\n"
    "                    pyc = be._code_to_hash_pyc(\n"
    "                        code, importlib.util.source_hash(data), True)\n"
    "                    if os.path.dirname(cfile):\n"
    "                        os.makedirs(os.path.dirname(cfile), exist_ok=True)\n"
    "                    be._write_atomic(cfile, pyc, be._calc_mode(path))\n"
    "                    done.add(path)\n"
    "                except Exception:\n"
    "                    pass\n"
    "            return code\n"
    "\n"
    "    sys.path_hooks.insert(0, m.FileFinder.path_hook(\n"
    "        (m.ExtensionFileLoader, m.EXTENSION_SUFFIXES),\n"
    "        (Once, m.SOURCE_SUFFIXES),\n"
    "        (m.SourcelessFileLoader, m.BYTECODE_SUFFIXES)))\n"
    "    sys.path_importer_cache.clear()\n"
    "except Exception:\n"
    "    pass\n"
    + _PYTHON_PRECOMPILE.replace(
        "    if source.startswith(lib) and source.endswith('.py'):\n",
        "    if source.startswith(lib) and source.endswith('.py') and source not in done:\n",
    )
)

#: How many image interpreters, or ``ldd`` runs, go at once in the build.
_BUILD_BATCH = 8


#: The system loader's configuration: every file it reads and every folder
#: it searches root's alone, so no library a kit program resolves can be
#: replaced or shadowed later (PY8-PT-R8-4, PT-C8-R8-5); otherwise refuse.
#: The seal's sh, and :func:`loader_check_command` before the agent's
#: install (DFX-SEAL). Needs ROOT_CHAIN_FUNCS and bf_die.
_LOADER_PROOF = [
    # Owner and mode of a path the proof refuses, for the message.
    "bf_who() { stat -L -c 'owner %u:%g, mode %A' -- \"$1\" 2>/dev/null || "
    "stat -c 'owner %u:%g, mode %A' -- \"$1\" 2>/dev/null; }",
    # One folder per line: a folder's name may hold spaces.
    "bf_nl='\n'; bf_ldirs=$(printf '%s\\n' /lib /lib64 /lib32 /usr/lib /usr/lib64 /usr/lib32)",
    "for d in /lib/*-linux-* /usr/lib/*-linux-*; do "
    '[ -d "$d" ] && bf_ldirs="$bf_ldirs$bf_nl$d"; done',
    # Each configuration file in its own grammar (PT-C8-R9-2,
    # PT-C8-R10-2). glibc's ld.so.conf files: a comment runs from ``#``
    # to the end of its line; ``include`` adds the files its
    # whitespace-separated patterns name (relative to the including
    # file's folder); ``hwcap`` lines name no folder; any other line,
    # trimmed and without glibc's =TYPE suffix, is one folder. Spaces
    # and commas inside the directory name stay literal. musl's
    # ld-musl-*.path: folders separated by ``:`` or newlines.
    # A queue, not recursion, with a bound on the files read.
    "bf_lq=$(printf '%s ' /etc/ld.so.conf /etc/ld.so.conf.d/* /etc/ld-musl-*.path)",
    "bf_lfiles=' '; bf_ln=",
    "while :; do",
    '  set -f; set -- $bf_lq; set +f; [ $# -gt 0 ] || break; f=$1; shift; bf_lq="$*"',
    '  case "$bf_lfiles" in *" $f "*) continue;; esac',
    '  bf_lfiles="$bf_lfiles$f "; bf_ln="${bf_ln}x"',
    '  [ ${#bf_ln} -le 256 ] || bf_die "the loader configuration includes too many files"',
    '  [ -f "$f" ] || continue',
    "  bf_lbase=${f%/*}",
    "  case $f in */ld-musl-*.path)",
    "    bf_m=$(tr ':' '\\n' < \"$f\"); set -f; IFS=$bf_nl; for d in $bf_m; do "
    'case $d in /*) bf_ldirs="$bf_ldirs$bf_nl$d";; esac; done; unset IFS; set +f; '
    "continue;; esac",
    '  while IFS= read -r l || [ -n "$l" ]; do',
    "    l=${l%%#*}",
    # Trimmed of the whitespace around it; inside, kept as written.
    '    l=${l#"${l%%[![:space:]]*}"}; l=${l%"${l##*[![:space:]]}"}',
    '    [ -n "$l" ] || continue',
    "    case $l in",
    "      include[[:space:]]*) set -f; set -- ${l#include}; set +f; "
    'for p in "$@"; do case $p in /*) ;; *) p=$bf_lbase/$p;; esac; '
    'for g in $p; do bf_lq="$bf_lq $g"; done; done;;',
    "      hwcap[[:space:]]*) ;;",
    # glibc 2.36 and earlier cut the line at its first '=' (a library
    # type: dir=libc6); 2.37 and later read it as written (PT-R13-B2).
    # Both readings are searched by some image, so both are checked.
    '      /*) bf_ldirs="$bf_ldirs$bf_nl$l"; case $l in *=*) l=${l%%=*}; '
    'l=${l%"${l##*[![:space:]]}"}; bf_ldirs="$bf_ldirs$bf_nl$l";; esac;;',
    "    esac",
    '  done < "$f"',
    "done",
    # A file it would read that is missing passes only when nobody but
    # root could create it.
    "for f in /etc/ld.so.cache /etc/ld.so.preload /etc/ld.so.conf.d $bf_lfiles; do",
    '  bfc_chain "$f" 0; s=$?; [ "$s" = 0 ] || [ "$s" = 2 ] || { w=$(bf_who "$f"); '
    'bf_die "the system loader reads $f${w:+ ($w)}, which is not '
    "root's alone; BenchFlow will not run root steps whose libraries could "
    'change"; }',
    "done",
    "set -f; IFS=$bf_nl; for d in $bf_ldirs; do unset IFS; set +f",
    # A folder it would search that is missing passes only when nobody
    # but root could create it.
    '  bfc_chain "$d" 0; s=$?; [ "$s" = 0 ] || [ "$s" = 2 ] || { w=$(bf_who "$d"); '
    'bf_die "the system loader searches $d${w:+ ($w)}, which is not '
    "root's alone (a workspace or an image setting lets another user write "
    "it); BenchFlow will not run root steps whose libraries could change: "
    "set the task's workdir to a folder of its own\"; }",
    # What it finds there: every file directly in the folder, and in the
    # hardware-capability subfolders the loader also searches
    # (glibc-hwcaps/*, and the legacy tls, x86_64, haswell, ... ones),
    # root's alone (PT-C8-R9-4 sol). The standard dependencies of every
    # program on root's PATH come from these files.
    '  [ "$s" = 0 ] || continue',
    "  for h in . glibc-hwcaps tls x86_64 haswell xeon_phi avx512_1 i686 sse2 aarch64; do",
    '    for p in "$d/$h" "$d/$h"/*; do',
    '      [ -d "$p" ] || continue',
    "      case $p in */.) p=${p%/.};; esac",
    '      bfc_chain "$p" 0 || { w=$(bf_who "$p"); bf_die "the system loader searches '
    "$p${w:+ ($w)}, which is not root's alone\"; }",
    # The following walks use newline-delimited names. Refuse unsupported
    # entries before splitting them; a fixed marker cannot disappear.
    # Follow the directory argument, not child links, which may form a loop.
    '      bf_ln=$(find -H "$p" -maxdepth 1 -mindepth 1 -name "*$bf_nl*" '
    '-exec printf x \\; 2>/dev/null) || bf_die "cannot inspect loader names in $p"',
    '      [ -z "$bf_ln" ] || bf_die "the system loader searches $p '
    "(contains a name with a newline), which is not root's alone; refusing it\"",
    '      bf_lb=$(find -L "$p" -maxdepth 1 -mindepth 1 ! -type d ! -type l '
    "\\( ! -user 0 -o -perm -0020 -o -perm -0002 \\) -print 2>/dev/null | head -n 1)",
    # A link to a missing file (find -L's only -type l) is judged by its
    # hops, as a missing file is: it passes only when the link is root's and
    # nobody but root could create what it names (DFX-SEAL); a link's own
    # mode bits mean nothing.
    '      [ -n "$bf_lb" ] || bf_lb=$(find -L "$p" -maxdepth 1 -mindepth 1 -type l '
    '-print 2>/dev/null | while IFS= read -r l; do bfc_chain "$l" 0; s=$?; '
    '[ "$s" = 0 ] || [ "$s" = 2 ] || { printf \'%s\\n\' "$l"; break; }; done)',
    '      [ -z "$bf_lb" ] || { w=$(bf_who "$bf_lb"); bf_die "the system loader loads '
    "$bf_lb${w:+ ($w)}, which is not root's alone; BenchFlow will not run root "
    'steps whose libraries could change"; }',
    '      [ "$h" = . ] && break',
    "    done",
    "  done",
    "done; unset IFS; set +f",
]


def loader_check_command() -> str:
    """sh, run as root on the image before the agent's install: the seal's
    loader proof alone (:data:`_LOADER_PROOF`), so an image whose loader
    files another user owns is refused before anything is installed. Exits
    1 after the same ``benchflow:`` line the seal prints; the seal still
    runs after the workspace grant."""
    return "\n".join(
        [
            "set -u",
            ROOT_CHAIN_FUNCS,
            'bf_die() { echo "benchflow: $*" >&2; exit 1; }',
            *_LOADER_PROOF,
        ]
    )


_BUILD = "# --- build ---"
_SEAL = "# --- seal ---"


def _transfer_roots_func(roots: Sequence[str]) -> str:
    """sh defining ``bf_names_transfer VALUE``: exit 0 when a path *VALUE*
    names lies at or under one of *roots*: the deepest folder of it that
    exists now resolved (links, '..'), the rest as written."""
    resolve = "; ".join(
        f"bf_r{i}=$(cd -P -- {shlex.quote(root)} 2>/dev/null && pwd -P) "
        f"|| bf_r{i}={shlex.quote(root)}"
        for i, root in enumerate(roots)
    )
    under = "|".join(f'"$bf_r{i}"/*' for i in range(len(roots)))
    return (
        f"{resolve}; "
        "bf_names_transfer() ( set -f; IFS=': =,'; for v in $1; do "
        "case $v in /*) ;; *) continue;; esac; p=$v; t=; "
        'until d=$(cd -P -- "$p" 2>/dev/null && pwd -P); do '
        '[ "$p" != / ] || continue 2; t=${p##*/}${t:+/$t}; p=${p%/*}; '
        '[ -n "$p" ] || p=/; done; '
        f"case ${{d%/}}/$t in {under}) exit 0;; esac; "
        "done; exit 1 )"
    )


def kit_prepare_command(
    phase: str = "all",
    *,
    walk: bool = True,
    stage: str | None = None,
    transfer_roots: Sequence[str] = (),
) -> str:
    """sh, run as root on the plumbing route before the agent's first turn:
    make (or keep) the protected runtime tree and print its record.

    Before the agent's turn the image is the task's own, so the system and
    /usr/local folders are what the image put there; only files root alone
    owns and may write are copied. A kit already made in this sandbox is
    kept when it is root's alone and matches its MANIFEST, never rebuilt
    from what the image holds now. Prints the record between ``BF
    kit-record`` and ``BF kit-end``; any other failure exits non-zero with
    a ``benchflow:`` line saying what to do.

    *phase*: ``"build"`` copies the programs and the Python in, and runs
    before the workspace is granted (a workspace of ``/usr/local`` would
    otherwise hand the agent user the Python first, PT-C8-R3-10);
    ``"seal"`` proves the image folders, writes the record and the
    MANIFEST, and runs after the grant and the PATH lock; ``"all"`` does
    both. ``"restore"`` stands for ``"build"`` with a kit this host built for
    the same image and settings (:mod:`benchflow.sandbox._kit_cache`): the
    same preparation of the runtime tree, then the kit placed from *stage*
    (``<stage>/rt``, uploaded root's alone just before) instead of copied.

    *walk*: the build's tool copies on the image's python3 and the seal's
    proofs and MANIFEST on the kit's Python (:mod:`benchflow._kit_seal`)
    when there is one; False (tests only) runs the sh they mirror. The seal prints ``BF kit-seal walk`` or
    ``BF kit-seal sh``.

    *transfer_roots*: where a separate verifier will unpack the agent's
    bytes as root after this seal. No image value naming a path there is
    admitted, as in the agent's sandbox, whose seal comes after the grant.
    """
    from benchflow.sandbox._base import IMAGE_RESTORED_VARS, env_value_check_func
    from benchflow.sandbox._channel_python import prepare_command as channel_python

    image_names = "|".join(IMAGE_RESTORED_VARS)
    rt = shlex.quote(RUNTIME_ROOT)
    kit = shlex.quote(KIT)
    skip = "|".join(_STDLIB_SKIP)
    folders = " ".join(SYSTEM_FOLDERS)
    kit_py = '"$py"' if walk else "0"
    kit_walk = 1 if walk else 0

    # The build's sh, its own function (the kit's Python launches stay
    # in kit_prepare_command, test_c8_r10_root_python's allowed owner).
    def _build_lines(skip: str) -> list[str]:
        """The build's sh (PERF-H): the same files as the steps one
        after the other, with less work in a one-CPU container and the rest
        side by side. The image Python capture runs as a job (each
        interpreter once per path, :data:`_BUILD_BATCH` at a time, each into
        its own file, joined in the order one after the other gives); the
        tools are copied by one walk on the image's root-only python3
        (:data:`benchflow._kit_seal.TOOLS_WALK`), else by the loop; the
        Python copy runs as a job (one ``cp -a``, one ``ldd`` on glibc, the
        compile before the smoke). Every job is waited for before the build
        decides anything: a failure in any of them fails the build with its
        own ``benchflow:`` line, as before."""
        tools = " ".join(KIT_TOOLS)
        batch = "x" * _BUILD_BATCH
        return [
            'rm -rf "$KIT" && mkdir -p "$KIT/bin" "$KIT/py" "$KIT/libexec" "$KIT/lib" '
            "|| bf_die 'cannot make the protected runtime kit'",
            # bf_wait <pid>...: each job waited for; 1 when any failed.
            'bf_wait() { bf_ws=0; for bf_p in "$@"; do wait "$bf_p" || bf_ws=1; done; '
            "return $bf_ws; }",
            # Capture the image interpreter's actual .pth-expanded search path
            # while every startup hook is still the trusted image's. The record
            # is sealed by MANIFEST and survives reconnect/snapshot recovery.
            'printf \'{"path": [], "user_site": []}\\n\' > "$KIT/IMAGE_PYTHON_PATHS.json"',
            ': > "$KIT/IMAGE_PYTHON_PATHS.jsonl" || bf_die "cannot write the image Python record"',
            ': > "$KIT/IMAGE_PYTHON_GUARD" || bf_die "cannot write the image Python record"',
            "bf_image_py=$(command -v python3) || bf_image_py=",
            "bf_cwd=$PWD",
            "(",
            '  mkdir "$KIT/.cap" || exit 1',
            # test.sh's search-path inputs go along as text (PT-R16-B-1): the
            # route's data copies in the environment (never argv)
            # and the working folder; the capture reads PID
            # 1's values itself.
            '  bf_c0=; if [ -n "$bf_image_py" ]; then ( cd / && '
            + IMAGE_PYTHON_SEARCH_ENV
            + '"$bf_image_py" -I -B -c '
            + shlex.quote(CAPTURE_IMAGE_PYTHON_PATHS)
            + ' "$KIT/IMAGE_PYTHON_PATHS.json" "$bf_cwd" ) & bf_c0=$!; fi',
            # And every interpreter on the image's own PATH (PID 1's), keyed by
            # the path it is started by: the verifier's trusted PATH is drawn
            # from these folders, so test.sh's interpreter, a venv first on
            # PATH included, is among them (PT-R13-A-2). One that fails here is
            # left out (it cannot run the tests either); at most 64.
            "  bf_ipath=$(tr '\\000' '\\n' < /proc/1/environ 2>/dev/null | sed -n 's/^PATH=//p' | head -n 1)",
            # bf_capture_py <python>: its record line, once, at most 64, into a
            # file of its own; appended below in this order.
            # The same path a second time (a folder twice on PATH) appends its
            # first capture's line again: the bytes one after the other gives,
            # the interpreter started once.
            "  bf_pyn=; bf_cq=; bf_cn=; bf_cf=; bf_cran=",
            '  bf_capture_py() { [ -f "$1" ] && [ -x "$1" ] || return 0; '
            'bf_pyn="${bf_pyn}x"; [ ${#bf_pyn} -le 64 ] || return 1; '
            'bf_same=; case "$1" in *"$bf_nl"*) ;; *) IFS=$bf_nl; set -f; '
            'for bf_e in $bf_cran; do [ "${bf_e#* }" = "$1" ] && { bf_same=${bf_e%% *}; break; }; '
            "done; unset IFS; set +f;; esac; "
            'if [ -n "$bf_same" ]; then bf_cf="$bf_cf $KIT/.cap/$bf_same"; return 0; fi; '
            'bf_cran="$bf_cran$bf_nl${#bf_pyn} $1"; '
            'bf_cf="$bf_cf $KIT/.cap/${#bf_pyn}"; ( cd / && "$1" -I -B -c '
            + shlex.quote(CAPTURE_ONE_IMAGE_PYTHON_PATHS)
            + ' "$KIT/.cap/${#bf_pyn}" "$1" ) </dev/null >/dev/null 2>&1 & '
            f'bf_cq="$bf_cq $!"; bf_cn="${{bf_cn}}x"; [ "$bf_cn" != {batch} ] || '
            "{ bf_wait $bf_cq || :; bf_cq=; bf_cn=; }; return 0; }",
            # A conda base on PATH (its prefix has conda-meta): its envs too,
            # though off PATH; `conda activate` in test.sh puts one first
            # (PT-R14-A-1, SWE-bench's testbed env).
            "  bf_nl='\n'; bf_conda=; IFS=:; set -f; for d in $bf_ipath; do unset IFS; set +f; "
            "case $d in /opt/runtime|/opt/runtime/*) continue;; /*) ;; *) continue;; esac; "
            'for py in "$d"/python "$d"/python3 "$d"/python3.[0-9] "$d"/python3.[0-9][0-9]; do '
            'bf_capture_py "$py" || break 2; done; '
            '[ -d "${d%/*}/conda-meta" ] && bf_conda="$bf_conda${bf_conda:+$bf_nl}${d%/*}"; '
            "done; unset IFS; set +f",
            "  IFS=$bf_nl; set -f; for b in $bf_conda; do unset IFS; set +f; "
            'for py in "$b"/envs/*/bin/python3; do bf_capture_py "$py" || break 2; done; '
            "done; unset IFS; set +f",
            # The pytest guard's search path for every interpreter its
            # installer may meet after the agent's turn (PT-C8-CONF-2b): each
            # python* in the folders of PID 1's PATH and of the fallback
            # verifier PATH, and each pytest script's shebang interpreter
            # there, keyed by the path it is started by and held to the file
            # it was (bf_guard_id). The same -I -S -B start the installer made
            # as root after the agent's turn, now before it; an interpreter not
            # recorded here is not started then, and its guard comes from
            # PYTHONPATH (CONF-2b round 2). A native
            # binary named twice in one folder (python3 -> python3.12) starts
            # once: its answer depends only on that folder, the file and the
            # pyvenv.cfg files, which bf_guard_id holds.
            "  " + GUARD_ID_SH,
            "  " + SHEBANG_PY_SH,
            '  mkdir "$KIT/.gcap" || exit 1',
            "  bf_gn=; bf_gb=; bf_gq=; bf_gf=; bf_gkeys=; bf_gseen=$bf_nl; bf_elf=$(printf '\\177ELF')",
            '  bf_guard_cap() { case "$1" in *"$bf_nl"*|"") return 0;; esac; '
            '[ -f "$1" ] && [ -x "$1" ] || return 0; '
            'case "$bf_gseen" in *"$bf_nl$1$bf_nl"*) return 0;; esac; '
            'bf_gseen="$bf_gseen$1$bf_nl"; bf_gn="${bf_gn}x"; [ ${#bf_gn} -le 64 ] || return 1; '
            'bf_gi=$(bf_guard_id "$1"); '
            'printf "exe %s\\nid %s\\n" "$1" "$bf_gi" > "$KIT/.gcap/${#bf_gn}.h" || return 1; '
            'bf_gk=; [ "$(head -c 4 "$1" 2>/dev/null)" != "$bf_elf" ] || bf_gk="${1%/*} $bf_gi"; '
            'bf_gm=; if [ -n "$bf_gk" ]; then IFS=$bf_nl; set -f; for bf_e in $bf_gkeys; do '
            '[ "${bf_e#* }" = "$bf_gk" ] && { bf_gm=${bf_e%% *}; break; }; done; unset IFS; set +f; fi; '
            'if [ -n "$bf_gm" ]; then bf_gf="$bf_gf ${#bf_gn}:$bf_gm"; return 0; fi; '
            '[ -z "$bf_gk" ] || bf_gkeys="$bf_gkeys$bf_nl${#bf_gn} $bf_gk"; '
            'bf_gf="$bf_gf ${#bf_gn}:${#bf_gn}"; ( cd / && "$1" -I -S -B -c '
            + shlex.quote(GUARD_SEARCH)
            + ' ) </dev/null >"$KIT/.gcap/${#bf_gn}.o" 2>/dev/null & '
            f'bf_gq="$bf_gq $!"; bf_gb="${{bf_gb}}x"; [ "$bf_gb" != {batch} ] || '
            "{ bf_wait $bf_gq || :; bf_gq=; bf_gb=; }; return 0; }",
            # The folders: one list, split once (bf_guard_dirs).
            "  " + GUARD_DIRS_SH,
            '  bf_guard_dirs "$bf_ipath"; IFS=:; set -f; for d in $bf_gdirs; do unset IFS; set +f; '
            'for py in "$d"/python "$d"/python3 "$d"/python3.[0-9] "$d"/python3.[0-9][0-9]; do '
            'bf_guard_cap "$py" || break 2; done; '
            'for s in "$d"/pytest "$d"/py.test; do [ -f "$s" ] || continue; '
            'bf_guard_cap "$(bf_shebang_py "$s")" || break 2; done; '
            "done; unset IFS; set +f",
            "  bf_wait $bf_cq $bf_gq || :",
            '  for f in $bf_cf; do [ ! -f "$f" ] || cat "$f" >> "$KIT/IMAGE_PYTHON_PATHS.jsonl" '
            '|| :; done; rm -rf "$KIT/.cap"',
            '  for f in $bf_gf; do cat "$KIT/.gcap/${f%:*}.h" "$KIT/.gcap/${f#*:}.o" '
            '>> "$KIT/IMAGE_PYTHON_GUARD" 2>/dev/null || :; done; rm -rf "$KIT/.gcap"',
            '  [ -z "$bf_c0" ] || wait "$bf_c0" || '
            '{ echo "benchflow: cannot capture image Python search paths" >&2; exit 1; }',
            ") & bf_capj=$!",
            # Bash 3.2 can discard a completed child's saved status during
            # the many command substitutions in the tool-copy loop when
            # this script runs in a pipeline subshell (the exec receipt).
            # Join before that loop; retain and check the same failure below.
            'bf_jfail=; wait "$bf_capj" || bf_jfail=1',
            # bf_copy <file>: its root-only copy under libexec, once per file.
            'bf_copy() { bf_n=$(printf %s "$1" | tr / _); '
            '[ -e "$KIT/libexec/$bf_n" ] || cp "$1" "$KIT/libexec/$bf_n" || return 1; '
            "printf '%s\\n' \"$bf_n\"; }",
            # The image's python3 the kit copies: root's alone, with its
            # version and standard library (root's alone too).
            # Match the verifier's image Python selection (V2-R5-C6). Python
            # images commonly keep the task version in /usr/local alongside an
            # older distribution Python; custody must not change that version.
            "bf_py=; for c in /usr/local/bin/python3 /usr/bin/python3 /bin/python3; do",
            '  if [ -x "$c" ]; then bf_py=$(readlink -f "$c") && '
            'bfc_chain "$bf_py" 0 && break; bf_py=; fi; done',
            # bf_pyprobe: its version and standard library; bf_pyok=1 when
            # the kit can copy it (run again before a second copy, ACC-RUN2).
            "bf_pyprobe() {",
            '  bf_v=; bf_std=; bf_pyok=; [ -n "$bf_py" ] || return 0',
            '  bf_info=$(cd / && "$bf_py" -I -S -c '
            + shlex.quote(
                'import sys, sysconfig; print("%d.%d" % sys.version_info[:2]); '
                'print(sysconfig.get_paths()["stdlib"]); '
                'print(getattr(sys, "platlibdir", "lib"))'
            )
            + " 2>/dev/null) || bf_info=",
            '  bf_v=$(printf %s "$bf_info" | head -n 1); '
            'bf_std=$(printf %s "$bf_info" | sed -n 2p); '
            'bf_platlib=$(printf %s "$bf_info" | sed -n 3p)',
            '  [ -n "$bf_v" ] && [ -d "$bf_std" ] && bfc_chain "$bf_std" 0 && bf_pyok=1; return 0',
            "}",
            "bf_pyprobe",
            # The tools copied by one walk on that python3 (_kit_seal.TOOLS_WALK,
            # the loop's rules), else by the loop below, one after the other.
            f'bf_tw=; if [ -n "$bf_pyok" ] && [ {kit_walk} = 1 ]; then',
            '  ( cd / && "$bf_py" -I -S -B -c '
            + shlex.quote(TOOLS_WALK)
            + f' "$KIT" {" ".join(KIT_TOOLS)} ); bf_s=$?',
            "  case $bf_s in 0) bf_tw=1;; 3) wait; exit 1;; *) "
            '[ ! -e "$KIT/.tools-applying" ] || { wait; bf_die "the protected runtime '
            "kit's tools stopped part way; start a fresh sandbox\"; };; esac",
            "fi",
            'if [ -z "$bf_tw" ]; then',
            # Tools: the system folders first, /usr/local last.
            f"  for t in {tools}; do",
            "    p=; for d in /usr/sbin /usr/bin /sbin /bin /usr/local/sbin /usr/local/bin; do",
            '      if [ -f "$d/$t" ] && [ -x "$d/$t" ]; then p=$d/$t; break; fi; done',
            '    [ -n "$p" ] || continue',
            '    r=$(readlink -f "$p") || continue',
            '    bfc_chain "$r" 0 || continue',
            '    n=$(bf_copy "$r") || continue',
            '    ln -s "../libexec/$n" "$KIT/bin/$t" || { wait; bf_die "cannot link $t"; }',
            "  done",
            "fi",
            # Python: a root-only python3, its standard library, and the
            # libraries it loads from outside the system library folders; the
            # job's status is 0 when the kit keeps it. A function, so a copy
            # that failed can run again (ACC-RUN2); each failure says which
            # step in $KIT/.py-why.
            'bf_pywhy() { printf \'%s\\n\' "$1" > "$KIT/.py-why"; }',
            "bf_pyjob() (",
            '  [ -n "$bf_pyok" ] || exit 1',
            '  mkdir -p "$KIT/lib/python$bf_v"',
            # Fedora's interpreter searches lib64 relative to its executable.
            # Keep both spellings inside the sealed tree; never fall back to
            # the live image's standard library when the isolated copy fails.
            '  if [ "$bf_platlib" = lib64 ] && [ ! -e "$KIT/lib64" ]; then '
            'ln -s lib "$KIT/lib64" || exit 1; fi',
            # The same entries in one cp -a; any that fails leaves the kit
            # without its Python, as before.
            '  set --; for e in "$bf_std"/* "$bf_std"/.[!.]*; do [ -e "$e" ] || continue; '
            f'case "${{e##*/}}" in {skip}) continue;; esac; set -- "$@" "$e"; done',
            '  [ $# = 0 ] || cp -a "$@" "$KIT/lib/python$bf_v/" || bf_std=',
            '  cp "$bf_py" "$KIT/libexec/python3" || bf_std=',
            "  bf_ld=; for l in /lib64/ld-linux-*.so.* /lib/ld-linux-*.so.* "
            '/lib/ld-musl-*.so.1; do [ -e "$l" ] && { bf_ld=$l; break; }; done',
            '  bf_deps() { ldd "$1" 2>/dev/null || { [ -n "$bf_ld" ] && '
            '"$bf_ld" --list "$1" 2>/dev/null; }; }',
            "  bf_extra=0",
            # On glibc one ldd names the Python and every extension module:
            # each file's list under its own "<file>:" line, the lines ldd
            # prints for it alone. Used only when it succeeds with one such
            # line per file (musl's ldd reads one file); else one ldd each.
            '  bf_dok=; set -- "$bf_py"; for o in "$KIT/lib/python$bf_v"/lib-dynload/*.so; do '
            '[ -e "$o" ] && set -- "$@" "$o"; done',
            '  case "$bf_ld" in */ld-linux-*) if [ $# -gt 1 ] && ldd "$@" > "$KIT/lib/.ldd" 2>/dev/null '
            '&& [ "$(grep -c \':$\' "$KIT/lib/.ldd")" = "$#" ]; then '
            'sed -n \'s/.*=> \\(\\/[^ ]*\\) (.*/\\1/p\' "$KIT/lib/.ldd" | sort -u > "$KIT/lib/.deps" '
            '&& bf_dok=1; fi; rm -f "$KIT/lib/.ldd";; esac',
            '  if [ -z "$bf_dok" ]; then',
            # One file per program, sorted together as one stream would be.
            '    mkdir "$KIT/lib/.deps.d" || exit 1; bf_di=; bf_dq=; bf_dn=',
            '    for o in "$bf_py" "$KIT/lib/python$bf_v"/lib-dynload/*.so; do '
            '[ -e "$o" ] || continue; bf_di="${bf_di}x"; '
            "( bf_deps \"$o\" | sed -n 's/.*=> \\(\\/[^ ]*\\) (.*/\\1/p' ) "
            '> "$KIT/lib/.deps.d/${#bf_di}" & bf_dq="$bf_dq $!"; bf_dn="${bf_dn}x"; '
            f'[ "$bf_dn" != {batch} ] || {{ bf_wait $bf_dq || :; bf_dq=; bf_dn=; }}; done',
            "    bf_wait $bf_dq || :",
            '    cat "$KIT/lib/.deps.d"/* 2>/dev/null | sort -u > "$KIT/lib/.deps"; '
            'rm -rf "$KIT/lib/.deps.d"',
            "  fi",
            "  while IFS= read -r lib; do",
            "    ld=${lib%/*}; if " + _sys_lib("ld") + "; then continue; fi",
            '    lr=$(readlink -f "$lib") && bfc_chain "$lr" 0 && '
            'cp "$lr" "$KIT/lib/${lib##*/}" || { bf_std=; break; }',
            "    bf_extra=1",
            '  done < "$KIT/lib/.deps"; rm -f "$KIT/lib/.deps"',
            # bin/python3 always starts it isolated (-I -S): no script or
            # current folder, user site, site-packages or PYTHON* value
            # reaches a root helper's imports, whatever flags it passes.
            '  if [ -n "$bf_std" ] && [ "$bf_extra" = 1 ] && [ -n "$bf_ld" ]; then',
            "    printf '#!%s\\nexec %s --library-path %s %s -I -S -B \"$@\"\\n' "
            '"$KIT/bin/sh" "$bf_ld" "$KIT/lib" "$KIT/libexec/python3" '
            '> "$KIT/py/python3" && chmod 755 "$KIT/py/python3" || bf_std=',
            '  elif [ -n "$bf_std" ] && [ "$bf_extra" = 0 ]; then',
            "    printf '#!%s\\nexec %s -I -S -B \"$@\"\\n' "
            '"$KIT/bin/sh" "$KIT/libexec/python3" '
            '> "$KIT/py/python3" && chmod 755 "$KIT/py/python3" || bf_std=',
            "  else bf_std=; fi",
            '  [ -n "$bf_std" ] || { bf_pywhy "copying $bf_py, its standard library '
            'or the libraries it loads failed"; exit 1; }',
            # The helpers' modules compiled (KIT_PRECOMPILED_MODULES) first, so
            # the smoke loads them compiled: a kit whose smoke fails loses its
            # whole lib below, the compiled files with it.
            '  ( cd / && "$KIT/py/python3" -c '
            + shlex.quote(_PYTHON_PRECOMPILE_ONCE)
            + ' "$KIT" ) >/dev/null 2>&1 || :',
            '  bf_pre=$(cd / && "$KIT/py/python3" -I -S -c '
            + shlex.quote(_PYTHON_SMOKE)
            + " 2>/dev/null) || bf_pre=",
            '  [ "$bf_pre" = "$KIT" ] || { bf_pywhy "the copied Python did not start"; exit 1; }',
            ")",
            "bf_pyjob & bf_pyj=$!",
            # Every job ends before anything is decided.
            'py=0; if wait "$bf_pyj"; then py=1; fi',
            # An image with a root-only python3 whose copy failed: once more,
            # alone, from a clean start. A step that failed once on a busy
            # sandbox is no answer about the image, and a kit recorded without
            # Python here makes the trial unscoreable after the agent's turn
            # (usaco/987, ACC-RUN2). Still failing: no Python, as before, and
            # the build says why.
            'if [ "$py" != 1 ] && [ -n "$bf_py" ]; then',
            '  rm -rf "$KIT/py" "$KIT/libexec/python3" "$KIT/lib" && mkdir -p "$KIT/lib" "$KIT/py" '
            "|| bf_die 'cannot make the protected runtime kit'",
            "  bf_pyprobe",
            '  [ -n "$bf_pyok" ] || bf_pywhy "$bf_py did not report its version and standard library, '
            'or someone other than root may change them"',
            "  if bf_pyjob; then py=1; else",
            "    printf 'BF kit-python-failed %s\\n' "
            '"$(cat "$KIT/.py-why" 2>/dev/null || echo "$bf_py could not be copied")"',
            "  fi",
            "fi",
            'rm -f "$KIT/.py-why"',
            '[ -z "$bf_jfail" ] || exit 1',
            '[ -e "$KIT/bin/sh" ] || bf_die "the image has no root-owned /bin/sh '
            'BenchFlow could copy"',
            'if [ "$py" != 1 ]; then rm -rf "$KIT/py" "$KIT/libexec/python3" '
            '"$KIT/lib"; mkdir -p "$KIT/lib" "$KIT/py"; fi',
            channel_python(),
        ]

    lines = [
        "set -u",
        "umask 022",
        ROOT_CHAIN_FUNCS,
        'bf_die() { echo "benchflow: $*" >&2; exit 1; }',
        f"RT={rt}; KIT={kit}",
        # /opt: a link, or a folder others may write and not sticky,
        # would let the agent user replace the tree.
        '[ ! -L /opt ] || bf_die "/opt is a link; BenchFlow keeps its '
        'runtime tree in /opt/runtime and needs /opt to be a folder"',
        "mkdir -p /opt || bf_die 'cannot make /opt'",
        "if ! bfc_dir /opt s; then chown 0:0 /opt && chmod 1777 /opt "
        "|| bf_die 'cannot make /opt root-owned'; fi",
        # The runtime root and its kit are fixed paths with no alias: a link
        # at either could be swapped for another tree (PY8-PT-R8-1).
        '[ ! -L "$RT" ] && [ ! -L "$KIT" ] || bf_die "$RT and $KIT must be '
        'folders, not links; remove the link from the image"',
        'mkdir -p "$RT" && chown 0:0 "$RT" && chmod go-w "$RT" '
        "|| bf_die 'cannot make /opt/runtime root-only'",
        'bfc_chain "$RT" 0 || bf_die "BenchFlow\'s runtime tree $RT is not '
        "root's alone (a folder or link on the way to it can be written by "
        'another user); remove that from the image"',
        # The folders under it are fixed paths too: an image link at one
        # (the staging folder into a folder the agent owns) would hand root
        # a payload, a HOME or the verifier's files the agent can swap or
        # read (DX-1 review A M1). Each is checked before it is made, the
        # one inside run only once run is proved a folder.
        '[ ! -L "$RT/run" ] && [ ! -L "$RT/verifier" ] || bf_die "$RT/run and '
        '$RT/verifier must be folders, not links; remove the link from the image"',
        'mkdir -p "$RT/run" && chown 0:0 "$RT/run" && chmod 700 "$RT/run" '
        "|| bf_die 'cannot make the private staging folder'",
        '[ ! -L "$RT/run/home" ] || bf_die "$RT/run/home must be a folder, not '
        'a link; remove the link from the image"',
        # root's HOME on the route: empty, root's alone, never the image's.
        'mkdir -p "$RT/run/home" && chown 0:0 "$RT/run/home" && '
        "chmod 700 \"$RT/run/home\" || bf_die 'cannot make the protected HOME for root'",
        'mkdir -p "$RT/verifier" && chown 0:0 "$RT/verifier" && '
        "chmod 755 \"$RT/verifier\" || bf_die 'cannot make the verifier folder'",
        'bfc_chain "$RT/run/home" 0 && bfc_chain "$RT/verifier" 0 || bf_die '
        "\"BenchFlow's staging, HOME or verifier folder under $RT is not "
        "root's alone\"",
        # A kit made earlier in this sandbox: keep it only as it was made.
        'if [ -f "$KIT/RECORD" ]; then',
        f"  {root_only_tree_check(KIT, 'the protected runtime kit')} || "
        "bf_die \"the protected runtime kit $KIT is not root's alone; "
        'BenchFlow will not run root steps from it"',
        '  ( cd "$KIT" && "$KIT/bin/sha256sum" -c MANIFEST >/dev/null 2>&1 ) || '
        '( cd "$KIT" && sha256sum -c MANIFEST >/dev/null 2>&1 ) || '
        'bf_die "the protected runtime kit $KIT changed since BenchFlow made '
        'it; start a fresh sandbox"',
        '  echo "BF kit-record"; cat "$KIT/RECORD"; echo "BF kit-end"; exit 0',
        "fi",
        _BUILD,
        *_build_lines(skip),
        _SEAL,
        # What the build left (the seal runs as its own command).
        'py=0; [ -x "$KIT/py/python3" ] && py=1',
        # The proofs below (the loader, each kit program's libraries, the
        # system folders) in one walk on the kit's Python, applying the same
        # rules (benchflow._kit_seal, PERF-H). The sh after it is their
        # specification and runs instead when the kit has no Python or the
        # walk leaves the decision to it (exit 4, before any change).
        "bf_walked=; bf_trusted=",
        f"if [ {kit_py} = 1 ]; then",
        '  chown -R 0:0 "$KIT" && chmod -R go-w,a+rX "$KIT" || '
        "bf_die 'cannot make the protected runtime kit root-only'",
        '  rm -f "$KIT/.walk-applying"',
        '  bf_trusted=$(cd / && "$KIT/py/python3" -c '
        + shlex.quote(SEAL_WALK)
        + f' "$KIT" {folders})',
        "  case $? in",
        '    0) bf_walked=1; rm -f "$KIT/.walk-applying";;',
        "    3) exit 1;;",
        '    *) [ ! -e "$KIT/.walk-applying" ] || bf_die "the protected runtime '
        "kit's seal stopped part way; start a fresh sandbox\"; bf_trusted=;;",
        "  esac",
        "fi",
        'if [ -z "$bf_walked" ]; then',
        # The system folders root's PATH may keep: each one, every folder
        # above it, every file in it and every hop of every link in it
        # root's alone. Proven here, positively, or left out.
        # A folder that is a link (usrmerge's /bin -> usr/bin) is judged
        # by its route and then as the folder it resolves to, once.
        # The system loader's configuration (_LOADER_PROOF).
        *_LOADER_PROOF,
        # Every library each kit program resolves, the loader included:
        # root's alone along its whole route, or the program is dropped.
        "bf_ld=; for l in /lib64/ld-linux-*.so.* /lib/ld-linux-*.so.* "
        '/lib/ld-musl-*.so.1; do [ -e "$l" ] && { bf_ld=$l; break; }; done',
        'bf_deps() { ldd "$1" 2>/dev/null || { [ -n "$bf_ld" ] && '
        '"$bf_ld" --list "$1" 2>/dev/null; }; }',
        "bf_libok=' '; bf_dirok=' '; bf_glibc=",
        'case "$bf_ld" in */ld-linux-*) bf_glibc=1;; esac',
        'for x in "$KIT"/libexec/*; do',
        '  [ -f "$x" ] || continue',
        '  case "${x##*/}" in python3) continue;; esac',
        "  bf_xok=1",
        "  for lib in $(bf_deps \"$x\" | sed -n -e 's/.*=> \\(\\/[^ ]*\\) (.*/\\1/p' "
        "-e 's/^[[:space:]]*\\(\\/[^ ]*\\) (.*/\\1/p'); do",
        # Each library is walked once: most tools share libc and its loader.
        '    case "$bf_libok" in *" $lib "*) continue;; esac',
        '    bfc_chain "$lib" 0 || { bf_xok=; break; }; bf_libok="$bf_libok$lib "',
        "  done",
        # Every folder the loader tries before it finds each library: the
        # program's RUNPATH/RPATH folders and their hardware-capability
        # subfolders included (PT-C8-R9-4 sol). glibc names each file it
        # tries; the folder of each, root's alone (missing: only root's to
        # make), or the program is dropped. musl tries no capability
        # folders and prints no search; there the folder each library
        # resolved from is proven.
        '  if [ -n "$bf_xok" ]; then',
        '    if [ -n "$bf_glibc" ]; then bf_tried=$(LD_DEBUG=libs "$bf_ld" --list "$x" '
        "2>&1 >/dev/null | sed -n 's/.*trying file=\\(\\/[^ ]*\\).*/\\1/p'); "
        "else bf_tried=$(bf_deps \"$x\" | sed -n -e 's/.*=> \\(\\/[^ ]*\\) (.*/\\1/p'); fi",
        '    for p in $bf_tried; do p=${p%/*}; [ -n "$p" ] || p=/',
        '      case "$bf_dirok" in *" $p "*) continue;; esac',
        '      bfc_chain "$p" 0; s=$?; [ "$s" = 0 ] || [ "$s" = 2 ] || { bf_xok=; break; }',
        '      bf_dirok="$bf_dirok$p "',
        "    done",
        "  fi",
        '  [ -n "$bf_xok" ] && continue',
        # The dropped program's name stays in the kit's bin, first on root's
        # PATH, as a refusal: PATH never falls through to the image's own
        # copy, which loads the same library (PT-C8-R9-6 astra).
        '  for l in "$KIT"/bin/*; do t=$(readlink "$l") || continue; '
        '[ "${t##*/}" = "${x##*/}" ] || continue; rm -f "$l" && '
        "printf '#!%s\\necho \"benchflow: %s loads a library another user can "
        'replace; root does not run it" >&2\\nexit 126\\n\' "$KIT/bin/sh" '
        '"${l##*/}" > "$l" && chmod 755 "$l"; done',
        '  rm -f "$x"',
        "done",
        '[ -L "$KIT/bin/sh" ] || bf_die "the image\'s /bin/sh loads a library '
        'another user can replace; BenchFlow will not run root steps with it"',
        # A folder that fails only on some of its programs stays usable
        # through a view: links to the programs that pass, in the kit
        # (PT-C8-R8-6). A folder others may write is left out whole.
        'bf_view() { v="$KIT/view/$(printf %s "$1" | tr / _)"; '
        'mkdir -p "$v" || return 1; for e in "$1"/* "$1"/.[!.]*; do '
        '[ -e "$e" ] || [ -L "$e" ] || continue; n=${e##*/}; '
        'case "$n" in *[!A-Za-z0-9._+-]*) continue;; esac; '
        '[ -d "$e" ] && continue; '
        'if [ -L "$e" ]; then t=$(readlink "$e") || continue; '
        'case "$t" in /*) ;; *) t="$1/$t";; esac; bfc_link "$e" && '
        'bfc_chain "$t" 1 || continue; else bfc_file "$e" || continue; fi; '
        'ln -s "$e" "$v/$n" || return 1; done; printf \'%s\\n\' "$v"; }',
        "bf_trusted=; bf_good=' '; bf_bad=' '",
        f"for d in {folders}; do",
        '  [ -d "$d" ] || continue',
        '  bfc_chain "$d" 0 || continue',
        '  dr=$(cd -P -- "$d" 2>/dev/null && pwd -P) || continue',
        '  case "$bf_bad" in *" $dr "*) continue;; esac',
        '  case "$bf_good" in *" $dr "*) '
        'bf_trusted="${bf_trusted:+$bf_trusted:}$d"; continue;; esac',
        # Every file it holds, links followed (a dangling link is judged
        # by its hops below).
        '  [ -z "$(find -L "$dr" -maxdepth 1 -mindepth 1 ! -type d ! -type l '
        '\\( ! -user 0 -o -perm -0020 -o -perm -0002 \\) -print 2>/dev/null | head -n 1)" ] '
        '|| { bf_bad="$bf_bad$dr "; v=$(bf_view "$dr") && '
        'bf_trusted="${bf_trusted:+$bf_trusted:}$v"; continue; }',
        "  bf_ok=1; bf_seen=' '",
        '  for l in $(find "$dr" -maxdepth 1 -type l 2>/dev/null); do',
        '    t=$(readlink "$l") || { bf_ok=; break; }',
        '    case "$t" in /*) ;; *) t="$dr/$t";; esac',
        '    case "$bf_seen" in *" $t "*) continue;; esac; bf_seen="$bf_seen$t "',
        '    bfc_chain "$t" 1; s=$?; [ "$s" = 0 ] || [ "$s" = 2 ] || '
        "{ bf_ok=; break; }",
        "  done",
        '  if [ -n "$bf_ok" ]; then bf_good="$bf_good$dr "; '
        'bf_trusted="${bf_trusted:+$bf_trusted:}$d"; else bf_bad="$bf_bad$dr "; '
        'v=$(bf_view "$dr") && bf_trusted="${bf_trusted:+$bf_trusted:}$v"; fi',
        "done",
        "fi",
        'if [ -n "$bf_walked" ]; then echo "BF kit-seal walk"; else echo "BF kit-seal sh"; fi',
        'bash=0; [ -L "$KIT/bin/bash" ] && bash=1',
        # The image's values that name files root alone owns now, after the
        # grant and before the agent's turn (PT-C8-R9-5 astra / R9-3 sol):
        # the task route restores a loader or start-up value as root only
        # when it is listed here and still passes, so a file the agent wrote
        # that verification's freeze later made root's never qualifies.
        ': > "$KIT/IMAGE_ENV_OK" || bf_die "cannot write the image env record"',
        # Every name restore_image_env restores, checked as it checks them
        # (bf_value_ok): path lists root's alone, options naming no file the
        # agent could write (PT-C8-CONF-6).
        env_value_check_func(),
        *([_transfer_roots_func(transfer_roots)] if transfer_roots else []),
        "if [ -r /proc/1/environ ]; then",
        "  tr '\\000' '\\n' < /proc/1/environ | while IFS= read -r l; do",
        "    case $l in *=) continue;; esac",
        # AgentCore and native Daytona preserve these names before their
        # provider-owned shell starts. Record the same canonical name that
        # restore_image_env checks, for both representations of the image.
        "    n=${l%%=*}; n=${n#_BF_IMAGE_}",
        f"    case $n in {image_names}) ;; *) continue;; esac",
        "    l=$n=${l#*=}",
        '    bf_value_ok "$n" "${l#*=}" && '
        + ('! bf_names_transfer "${l#*=}" && ' if transfer_roots else "")
        + 'printf \'%s\\n\' "$l" >> "$KIT/IMAGE_ENV_OK"',
        "  done",
        "fi",
        # RECORD first, MANIFEST (which hashes it) last: a kit without a
        # MANIFEST that matches is never used.
        "{ echo kit=1; echo python=$py; echo bash=$bash; "
        'echo "trusted=$bf_trusted"; } > "$KIT/RECORD" || bf_die "cannot write the kit record"',
        'chown -R 0:0 "$KIT" && chmod -R go-w,a+rX "$KIT" || '
        "bf_die 'cannot make the protected runtime kit root-only'",
        # MANIFEST's lines from the kit's Python when it has one (the same
        # bytes, _kit_seal.MANIFEST_WALK), else from sha256sum.
        f'( cd "$KIT" || exit 1; if [ {kit_py} = 1 ] && "$KIT/py/python3" -c '
        + shlex.quote(MANIFEST_WALK)
        + " > MANIFEST.new; then :; else find . -type f ! -name 'MANIFEST*' "
        '-exec sha256sum {} + > MANIFEST.new; fi ) && chmod 644 "$KIT/MANIFEST.new" && '
        'mv "$KIT/MANIFEST.new" "$KIT/MANIFEST" || '
        "bf_die 'cannot hash the protected runtime kit'",
        f"{root_only_tree_check(KIT, 'the protected runtime kit')} || "
        "bf_die 'the protected runtime kit is not root-only after it was made'",
        'echo "BF kit-record"; cat "$KIT/RECORD"; echo "BF kit-end"',
    ]
    b, s = lines.index(_BUILD), lines.index(_SEAL)
    head, build, seal = lines[:b], lines[b + 1 : s], lines[s + 1 :]
    if phase == "build":
        body = [*head, *build, ': > "$KIT/.built"', 'echo "BF kit-built"']
    elif phase == "restore":
        if not stage or not re.fullmatch(r"/\.runtime-kit-[0-9a-f]{32}", stage):
            raise ValueError(f"not a kit stage: {stage!r}")
        q = shlex.quote(stage)
        restore = [
            f"BF_STAGE={q}",
            'bf_unstage() { rm -rf "$BF_STAGE"; }',
            '{ [ ! -L "$BF_STAGE" ] && [ -d "$BF_STAGE/rt" ] && [ ! -L "$BF_STAGE/rt" ]; } '
            "|| { bf_unstage; bf_die 'the kept protected runtime kit did not arrive'; }",
            f"{root_only_tree_check(stage, 'the kept protected runtime kit')} || "
            '{ bf_unstage; bf_die "the kept protected runtime kit is not root\'s alone"; }',
            'rm -rf "$KIT" && mv "$BF_STAGE/rt" "$KIT" && rmdir "$BF_STAGE" || '
            "{ bf_unstage; bf_die 'cannot place the kept protected runtime kit'; }",
        ]
        # A kit this sandbox already holds is kept by the head, as for a
        # build; the stage goes first then.
        early = f'[ ! -f "{KIT}/RECORD" ] || rm -rf {q}'
        body = [early, *head, *restore, ': > "$KIT/.built"', 'echo "BF kit-built"']
    elif phase == "seal":
        built = (
            '[ -f "$KIT/.built" ] || bf_die "the protected runtime kit was not '
            'built before the workspace was granted"'
        )
        body = [*head, built, *seal]
    else:
        body = [*head, *build, *seal]
    return "\n".join(body)


def kit_load_command() -> str:
    """sh for the kit's own shell (``KIT_SH -c``): print the record of a kit
    that is root's alone and matches its MANIFEST; exit 3 when there is no
    kit. Every program it names is the kit's."""
    kit = shlex.quote(KIT)
    return (
        f"PATH={shlex.quote(KIT_BIN)}; export PATH; "
        f"[ -f {kit}/RECORD ] || exit 3; "
        f"{root_only_tree_check(KIT, 'the protected runtime kit')} || exit 1; "
        f"( cd {kit} && sha256sum -c MANIFEST >/dev/null 2>&1 ) || "
        "{ echo 'benchflow: the protected runtime kit changed since BenchFlow "
        "made it' >&2; exit 1; }; "
        f'echo "BF kit-record"; cat {kit}/RECORD; echo "BF kit-end"'
    )


#: BenchFlow's Node, its JS agents and their launchers: what root runs from
#: the tree when it probes or configures a JS agent after installing it.
NODE_RUNTIME_PATHS = (
    f"{RUNTIME_ROOT}/node",
    f"{RUNTIME_ROOT}/js-agents",
    f"{RUNTIME_ROOT}/bin",
)


def runtime_custody_check(paths: tuple[str, ...], what: str) -> str:
    """sh: status 0 only when each of *paths* that exists is root's alone
    (:func:`~benchflow._root_only.root_only_tree_check`). Put it in the same
    command, before root runs anything from them, version probes included
    (C8, PT-C8-R3-6)."""
    return " && ".join(
        f"{{ {{ [ ! -e {shlex.quote(p)} ] && [ ! -L {shlex.quote(p)} ]; }} || "
        f"{root_only_tree_check(p, what)}; }}"
        for p in paths
    )
