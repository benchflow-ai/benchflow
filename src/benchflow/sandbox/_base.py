"""BaseSandbox — abstract base for sandbox backends.

Named in RL-first terminology:
- environment -> sandbox
- rollout_paths -> rollout_paths
- EnvironmentConfig -> SandboxConfig
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import logging
import re
import shlex
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, PrivateAttr

from benchflow._root_only import ROOT_CHAIN_FUNCS
from benchflow._runtime_tree import (
    KIT_BIN,
    KIT_IMAGE_ENV_OK,
    KIT_SH,
    ROOT_HOME,
    RuntimeKit,
    kit_load_command,
    parse_kit_record,
)
from benchflow.sandbox._exec_merge import KeptIdentity, env_key
from benchflow.sandbox._exec_output import DiagnosticResult, diagnostic_result, exec_raw
from benchflow.sandbox._task_route import as_task_command, in_task_command
from benchflow.sandbox.protocol import SandboxImage, SandboxSnapshotNotSupported
from benchflow.task.config import SandboxConfig
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import RolloutPaths

if TYPE_CHECKING:
    from collections.abc import Iterator

    from benchflow._utils.reachable_write import ReachablePath
    from benchflow.sandbox.process import LiveProcess

logger = logging.getLogger("benchflow")

# Docker Compose service names are restricted to this grammar. Used to filter
# `docker compose config --services` output, whose stdout may be polluted with
# warning lines because the compose command merges stderr into stdout.
_COMPOSE_SERVICE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


def _filter_compose_service_names(output: str) -> list[str]:
    """Extract valid compose service names from `config --services` output.

    Drops blank lines and anything that does not match the Docker Compose
    service-name grammar, so a warning line merged into stdout cannot be
    mistaken for a service (#248).
    """
    return [
        line
        for raw in output.splitlines()
        if (line := raw.strip()) and _COMPOSE_SERVICE_NAME_RE.match(line)
    ]


# A POSIX shell identifier: a name the shell can `export`. Keys outside this
# grammar (e.g. containing `.` or `-`) are valid process env keys but cannot be
# assigned via `export NAME=...`; sourcing such a line aborts the whole
# `. {env_path}` step, so the user command would never run (PR #323).
_SHELL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UPLOAD_MODE_RE = re.compile(r"^[0-7]{3,4}$")

# The sandboxes (by id) whose kept services (x-keep-services) are running
# beside root's commands in the current task: from the verifier's start to
# the kill that ends them (BaseSandbox.services_kept).
_SERVICES_KEPT: contextvars.ContextVar[frozenset[int]] = contextvars.ContextVar(
    "benchflow_services_kept", default=frozenset()
)
# The sandboxes (by id) reading their protected runtime kit's record: their
# exec_route answers KIT_PROBE_ROUTE without probing again.
_KIT_PROBING: contextvars.ContextVar[frozenset[int]] = contextvars.ContextVar(
    "benchflow_kit_probing", default=frozenset()
)


def validate_upload_mode(mode: str | None) -> str | None:
    """Return a safe octal upload mode, rejecting shell fragments."""

    if mode is not None and not _UPLOAD_MODE_RE.fullmatch(mode):
        raise ValueError(
            f"upload mode must be a 3- or 4-digit octal string, got {mode!r}"
        )
    return mode


ROOT_EXEC_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _root_exec_command(
    command: str, explicit_root: bool, path: str = ROOT_EXEC_PATH
) -> str:
    """Pin infrastructure tools, not the task role's deliberate environment.

    Capture the image PATH for the lock and verifier-PATH probes before
    replacing it. An env-file wrapper inside this prefix may then apply the
    verifier's explicit, validated PATH. Default-user task/oracle commands
    keep their image PATH; non-root execution keeps it too.
    *explicit_root*: the exec names a user whose uid is 0
    (:attr:`ExecRoute.explicit_root`), whatever its spelling. *path* is the
    sandbox's :attr:`BaseSandbox.root_exec_path`.
    """
    if not explicit_root:
        return command
    return (
        "_BENCHFLOW_IMAGE_PATH=$PATH; export _BENCHFLOW_IMAGE_PATH; "
        f"export PATH={shlex.quote(path)}; unset BASH_ENV ENV;\n{command}"
    )


# What the bash-backed sandboxes (Daytona, Modal, AgentCore) put before each
# ``bash -c`` and ``su -s /bin/bash -c`` they wrap a command in. Every
# non-interactive bash first runs the file BASH_ENV names, so an image whose
# BASH_ENV named a file the agent could write (``/app/startup.sh``) ran that
# file as root in the wrapping shells of every exec after the agent's turn,
# verifier hardening and test.sh included, before the command itself: the
# file could write a reward and exit, or change anything as root. The
# wrapping shells now start without BASH_ENV and carry the image's value
# under another name; RESTORE_BASH_ENV, the first thing the command's own
# shell runs, exports it again. That is what Docker's ``sh -c`` does: it never
# reads the file but passes the variable on, so the command and every bash it
# starts (agent install, the oracle's ``bash solve.sh``, an agent's tool
# shells) still read the image's file, a conda or nvm activation, on every
# sandbox kind. A BASH_ENV the caller passes in *env* (the verifier's, see
# ``lockdown._build_verifier_env``) is exported after the restore and wins.
# HIDE_BASH_ENV needs a POSIX shell to expand it; nested ones are harmless
# (an inner one finds BASH_ENV unset and passes the carried value on). An
# empty BASH_ENV names no file and is not carried, so a carried value already
# in the environment passes through: Daytona starts its own shells with
# ``BASH_ENV=""`` and the image's value under IMAGE_BASH_ENV (see
# ``DaytonaSandbox._hide_bash_env_from_toolbox``), because the shell the
# toolbox opens for each session is itself a bash that would read the file
# before any of this runs.
IMAGE_BASH_ENV = "_BF_IMAGE_BASH_ENV"
HIDE_BASH_ENV = f'/usr/bin/env -u BASH_ENV ${{BASH_ENV:+"{IMAGE_BASH_ENV}=$BASH_ENV"}}'
# Absolute already (PT, C8): K1's callers' name for the same prefix.
ABSOLUTE_HIDE_BASH_ENV = HIDE_BASH_ENV
RESTORE_BASH_ENV = (
    f'if [ -n "${{{IMAGE_BASH_ENV}+set}}" ]; then '
    f'export BASH_ENV="${IMAGE_BASH_ENV}"; unset {IMAGE_BASH_ENV}; fi; '
)


def restore_bash_env(route: Any) -> str:
    """:data:`RESTORE_BASH_ENV` for *route*: nothing for root on the
    protected kit, where :func:`restore_image_env` puts BASH_ENV back only
    after proving the file it names (PY8-PT-R8-7)."""
    if getattr(route, "kit", False) and getattr(route, "root", False):
        return ""
    return RESTORE_BASH_ENV


def bash_argv_without_bash_env(command: str, bash: str = "bash") -> list[str]:
    """argv that runs *command* under ``bash -c`` without the image's BASH_ENV file.

    For transports that take an argv, not a shell line (Modal, AgentCore).
    The outer bash starts in POSIX mode, which reads no startup file, only
    to expand HIDE_BASH_ENV; *command* must start with RESTORE_BASH_ENV (put
    it inside any ``su`` the command wraps itself in).
    """
    return [
        bash,
        "--posix",
        "-c",
        f'exec {HIDE_BASH_ENV} {bash} -c "$1"',
        bash,
        command,
    ]


def env_file_body(env: dict[str, str]) -> str:
    """The ``export KEY=value`` lines a shell sources to get *env*.

    Keys that are not POSIX shell identifiers are skipped with a warning (see
    :func:`wrap_command_with_env_file`).
    """
    exportable: dict[str, str] = {}
    skipped: list[str] = []
    for k, v in env.items():
        if _SHELL_IDENTIFIER_RE.match(k):
            exportable[k] = v
        else:
            skipped.append(k)
    if skipped:
        logger.warning(
            "Skipping env var(s) with non-identifier names (cannot be "
            "exported by the shell): %s",
            ", ".join(sorted(skipped)),
        )
    return "".join(f"export {k}={shlex.quote(v)}\n" for k, v in exportable.items())


# The positional parameters are the env wrappers' only scratch state: no task
# variable can be named ``1``, so applying an env never sets or unsets a name
# the task might use (b156 PY8-MAIN-32/33: the round-9 wrapper's trailing
# ``unset _bf_env`` removed a task's own ``_bf_env``). A wrapper sets one
# parameter before it applies the env; the env's own last line clears it. An
# env that was not applied to its end (an empty stdin, a substitution that
# failed) leaves it set, and the command never runs.
_ENV_PENDING = "set -- benchflow-env-pending; "
_ENV_APPLIED_LINE = "set --\n"
_ENV_CHECK = "case $# in 0) ;; *) exit 1;; esac\n"

# The shell reads the env from its own standard input: a path with a slash,
# so nothing is looked up on PATH, and ``.`` is a special builtin in every
# POSIX shell. A pipe from the backend's own CLI carries the env, so no value
# is on any command line and no file is written; another user cannot open
# root's pipe through /proc (its fd folder is the owner's alone). Then stdin
# is /dev/null (the builtin ``exec`` with a redirect only): ``. /dev/stdin``
# opens a description of its own, so a staged file left as stdin would still
# be at its start and the command could read the env back (b162
# PY8-MAIN-38). A redirect the command makes itself still applies.
# (Daytona's staged file is gone since b173; the reset stays for any stdin.)
ENV_FROM_STDIN = f"{_ENV_PENDING}. /dev/stdin || exit 1; {_ENV_CHECK}exec </dev/null\n"


def env_stdin_body(env: dict[str, str]) -> bytes:
    """What :data:`ENV_FROM_STDIN` reads: the ``export`` lines, then the check's line."""
    return (env_file_body(env) + _ENV_APPLIED_LINE).encode()


def wrap_command_with_env_stdin(command: str) -> str:
    """*command* prefixed to set the env :func:`env_stdin_body` gives on stdin.

    For transports that can feed the exec's standard input (``docker exec
    -i``, Apple ``container exec -i``). There is no size limit but the shell's: each value is a
    shell variable (an exported one over 128 KiB cannot reach a program the
    command starts, the kernel's limit on one environment string). The
    command's stdin is /dev/null, as with no stdin at all.
    """
    return f"{ENV_FROM_STDIN}{command}"


def wrap_command_with_env_file(
    env: dict[str, str],
    command: str,
    *,
    env_path_prefix: str = "",
    path: str | None = None,
    private: bool = False,
) -> str:
    """Return *command* prefixed to set *env* without any program or file.

    For a transport with no stdin of its own (Daytona's session commands).
    An env of any size is written this way; when the command would pass the
    transport bound, the whole text travels as data
    (:mod:`benchflow.sandbox._transport`). Docker and Apple container feed
    the env on stdin (:func:`wrap_command_with_env_stdin`).

    The ``export KEY=value`` lines are written into the command as octal
    escapes of every byte (``\\101\\102...``), so no value is visible as
    ``KEY=VALUE`` in ``ps aux`` or a provider's command audit log. The shell's
    own ``printf`` turns them back into text and ``eval`` runs them. Before
    (b127, b143) they were base64 decoded into a file under ``/tmp`` and
    sourced: the decoder came from a PATH (the image's let an agent-written
    ``/app/bin/base64`` run as root; the system folders alone missed a
    trusted image whose only decoder is in ``/usr/local/bin``, PY8-MAIN-31),
    the file sat where a process that can write ``/tmp`` could swap it before
    root sourced it, and ``rm`` came from a PATH too. Now nothing is looked
    up on any PATH and nothing is written: ``printf`` and ``eval`` are shell
    builtins in every POSIX shell a sandbox runs. No variable is used either
    (see ``_ENV_PENDING``).

    The command follows on its own line, after the check, rather than as
    the last link of an ``&&`` chain. A trailing ``&`` in the command (a
    detached service start, ``nohup CMD </dev/null >log 2>&1 &``) then
    backgrounds only the command. Chained with ``&&`` it backgrounded the whole
    list, including a subshell that kept the caller's stdout open, so a Daytona
    session command never finished.

    Only keys that are valid POSIX shell identifiers are emitted as ``export``
    lines. Keys that are not (e.g. containing ``.`` or ``-``) cannot be assigned
    by the shell, so they are skipped with a warning rather than silently
    breaking exec (regression guarded, PR #323).

    *env_path_prefix*, *path* and *private* are accepted for the callers that
    still pass them: no file is written and no program runs (only the shell's
    ``printf`` and ``eval``), so there is no file to make private and no tool
    to take from the route.
    """
    del env_path_prefix, path, private
    text = env_file_body(env) + _ENV_APPLIED_LINE
    escaped = "".join(f"\\{byte:03o}" for byte in text.encode())
    # A failed substitution stops here, as a failed decode did: eval of an
    # empty string succeeds, but leaves the pending parameter set.
    applied = f"eval \"$(printf '{escaped}')\" || exit 1; "
    return f"{_ENV_PENDING}{applied}{_ENV_CHECK}{command}"


class ExecResult(BaseModel):
    stdout: str | None = None
    stderr: str | None = None
    return_code: int
    # Never serialize the diagnostic copy in place of a parser's raw streams.
    _diagnostic_result: DiagnosticResult | None = PrivateAttr(default=None)


# ``exec(..., system_shell=True)``: the shell is started by this absolute path,
# and PATH is set to the system folders before anything else in the command runs
# (the env-file wrapper included). A sandbox's image PATH can lead into a folder
# the agent wrote (``ENV PATH=/app/.venv/bin:$PATH`` with /app kept), so a plain
# ``sh -c`` as root would run whatever ``sh`` the agent left there.
SYSTEM_SHELL = "/bin/sh"
SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"

# Loader and shell start-up variables that name a file or folder. ``/bin/sh``
# by absolute path is still a dynamically linked program, and ``docker compose
# exec -u root`` is no setuid start, so the loader would honor the image's
# ``ENV LD_LIBRARY_PATH=/app/lib`` (a kept path) before the first line of the
# command runs. glibc, musl and the shells treat an empty value as none, so a
# system shell is started with each of these set empty (``docker exec -e
# NAME=``): nothing secret, and the image's value never reaches the loader.
SYSTEM_SHELL_EMPTY_VARS = (
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "LD_DEBUG_OUTPUT",
    "LD_PROFILE",
    "LD_PROFILE_OUTPUT",
    "GCONV_PATH",
    "LOCPATH",
    "NLSPATH",
    "HOSTALIASES",
    "MALLOC_TRACE",
    "BASH_ENV",
    "ENV",
)
# Every variable that changes how a program root starts loads or what code it
# runs: the ones above, the loader's flags (glibc acts on their mere presence,
# so they are never passed, only unset), and the tools' own (tar's options,
# node's, Perl's and Python's). The system shell unsets them before anything
# else runs, so no root tool it starts sees the image's values.
SYSTEM_SHELL_UNSET_VARS = (
    *SYSTEM_SHELL_EMPTY_VARS,
    "LD_DEBUG",
    "LD_ORIGIN_PATH",
    "LD_DYNAMIC_WEAK",
    "LD_BIND_NOT",
    "LD_SHOW_AUXV",
    "LD_USE_LOAD_BIAS",
    "LD_HWCAP_MASK",
    "LD_ASSUME_KERNEL",
    # The rest of glibc's ld.so switches (rtld.c process_envvars, the old
    # releases' included); musl reads only LD_PRELOAD and LD_LIBRARY_PATH.
    # LD_TRACE_LOADED_OBJECTS lists a program's libraries instead of running
    # it (SCRIPT-AGENT-DOCKER-F1 round 2, review B M3).
    "LD_TRACE_LOADED_OBJECTS",
    "LD_TRACE_PRELINKING",
    "LD_BIND_NOW",
    "LD_WARN",
    "LD_VERBOSE",
    "LD_POINTER_GUARD",
    "LD_PREFER_MAP_32BIT_EXEC",
    "GLIBC_TUNABLES",
    "RES_OPTIONS",
    "TAR_OPTIONS",
    "NODE_OPTIONS",
    "NODE_PATH",
    "APT_CONFIG",
    "PERL5OPT",
    "PERL5LIB",
    "PERLLIB",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
)


# The image's start-up values root's task commands get back (restore_image_env),
# that the seal records before the agent's turn (KIT_IMAGE_ENV_OK) and that
# PID 1's copy is read for (capture_pid1_image_env): every name the route and
# S1's bootstrap clear, so no honest image value is lost to the clearing
# (PT-C8-CONF-6: PYTHONPATH, TAR_OPTIONS, ... were cleared and never came back).
IMAGE_RESTORED_VARS = SYSTEM_SHELL_UNSET_VARS
# Of those, the loader flags whose one effect is output at every program's
# start: glibc prints the aux vector to stdout for LD_SHOW_AUXV, and lists a
# program's libraries instead of running it for LD_TRACE_LOADED_OBJECTS (and
# the old LD_TRACE_PRELINKING). No command can depend on them, and BenchFlow
# parses its task commands' stdout (receipts, rewards, markers), so the value
# is kept aside and never given back (SCRIPT-AGENT-DOCKER-F1: an image's
# LD_SHOW_AUXV=1 broke every receipt read).
IMAGE_UNRESTORED_VARS = frozenset(
    {"LD_SHOW_AUXV", "LD_TRACE_LOADED_OBJECTS", "LD_TRACE_PRELINKING"}
)
# The loader's switches: glibc acts on their presence or on any value, and no
# file or search folder is named. A script agent gets the image's other
# cleared values back as ACP agents do, never these.
LOADER_FLAG_VARS = frozenset(
    {
        "LD_DEBUG",
        "LD_DYNAMIC_WEAK",
        "LD_BIND_NOT",
        "LD_BIND_NOW",
        "LD_SHOW_AUXV",
        "LD_TRACE_LOADED_OBJECTS",
        "LD_TRACE_PRELINKING",
        "LD_WARN",
        "LD_VERBOSE",
        "LD_PREFER_MAP_32BIT_EXEC",
    }
)
# Of those, the ones whose value lists files or search folders: every entry
# must be an absolute path root alone owns. A bare word there is a search
# (LD_PRELOAD=libx.so), never accepted for root.
_PATH_VALUED_VARS = (
    *SYSTEM_SHELL_EMPTY_VARS,
    "LD_ORIGIN_PATH",
    "NODE_PATH",
    "APT_CONFIG",
    "PERL5LIB",
    "PERLLIB",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
)
# Options that can load code: only plain flags pass (no module, loader or
# include flag, nothing naming a file), since a bare module name resolves
# against the command's folder.
_CODE_OPTION_VARS = ("NODE_OPTIONS", "PERL5OPT")
# Path lists where an empty entry (the command's own folder) is dropped and
# the rest kept, rather than the whole value refused: ``ENV
# PYTHONPATH=${PYTHONPATH}:/opt/vendor`` leaves a leading ':' (PT-R13-B3).
# The loader's names keep refusing it.
_DROP_EMPTY_VARS = ("PYTHONPATH", "NODE_PATH", "PERL5LIB", "PERLLIB")


def env_value_check_func() -> str:
    """Shell text defining ``bf_value_ok NAME VALUE`` (needs ROOT_CHAIN_FUNCS):
    exit 0 when *VALUE* is one root's task command may get back for *NAME*.

    Path-valued names: every ':'/' '-separated entry absolute and root's
    alone along its route. Code-loading options: plain ``-`` flags only.
    Other options (TAR_OPTIONS, GLIBC_TUNABLES, LD_DEBUG, ...): a token that
    names a file (it holds a '/') must be an absolute path root alone owns.
    Builtins and ``bfc_chain`` only; runs in a subshell, so IFS and
    globbing stay as they were.
    """
    paths = "|".join(_PATH_VALUED_VARS)
    code = "|".join(_CODE_OPTION_VARS)
    drop = "|".join(_DROP_EMPTY_VARS)
    chain = 'bfc_chain "$v" 0; s=$?; [ "$s" = 0 ] || [ "$s" = 2 ] || exit 1'
    return (
        "bf_value_ok() ( set -f; case $1 in "
        f"{drop}) IFS=':'; bf_any=; for v in $2; do case $v in '') continue;; "
        f'/*) ;; *) exit 1;; esac; {chain}; bf_any=1; done; [ -n "$bf_any" ] || exit 1;; '
        # The loader's lists: an empty entry anywhere (':' first, last or
        # doubled) is the command's own folder; it is dropped (bf_value_clean)
        # and the rest kept, as for the language lists (PT-R13-A-1, then
        # PT-R14-B1: refusing the whole value lost `ENV LD_LIBRARY_PATH=
        # /opt/vendor/lib:$LD_LIBRARY_PATH`). ';', which glibc also splits
        # on, is still refused, and so is a value with nothing left.
        f"{paths}) case $2 in *\\;*) exit 1;; esac; "
        "IFS=': '; bf_any=; for v in $2; do case $v in '') continue;; "
        f'/*) ;; *) exit 1;; esac; {chain}; bf_any=1; done; [ -n "$bf_any" ] || exit 1;; '
        f"{code}) IFS=' '; for v in $2; do case $v in "
        "-r|-r*|--require|--require=*|--import|--import=*|--loader|--loader=*|"
        "--experimental-loader|--experimental-loader=*|-M*|-m*|-I*|-d*|*/*) exit 1;; "
        "-*) ;; *) exit 1;; esac; done;; "
        "*) IFS=' '; for t in $2; do case $t in */*) v=${t#*=}; "
        f"case $v in /*) {chain};; *) exit 1;; esac;; esac; done;; "
        "esac; exit 0 ); "
    )


def env_value_clean_func() -> str:
    """Shell text defining ``bf_value_clean NAME VALUE``: *VALUE* as root's
    task command gets it, the empty entries of a path list left out: the
    language lists split on ':' (PT-R13-B3), the loader's on ':' and ' ' as
    glibc does, joined with ':' (PT-R14-B1). A subshell: no variable of its
    own outlives it in the command's shell."""
    drop = "|".join(_DROP_EMPTY_VARS)
    paths = "|".join(_PATH_VALUED_VARS)
    join = 'for v in $2; do [ -n "$v" ] && bf_o=${bf_o:+$bf_o:}$v; done; printf %s "$bf_o";; '
    return (
        f"bf_value_clean() ( set -f; bf_o=; case $1 in {drop}) IFS=':'; {join}"
        f"{paths}) IFS=': '; {join}"
        '*) printf %s "$2";; esac ); '
    )


def system_shell_env_flags() -> list[str]:
    """``-e NAME=`` for each of :data:`SYSTEM_SHELL_EMPTY_VARS`, for ``docker (compose) exec``."""
    return [arg for name in SYSTEM_SHELL_EMPTY_VARS for arg in ("-e", f"{name}=")]


# What the dynamic loader and libc read while a program starts, before its
# first line runs: every LD_ and MALLOC_ name (glibc acts on the mere presence
# of LD_SHOW_AUXV, LD_DYNAMIC_WEAK, LD_TRACE_LOADED_OBJECTS, ..., so an empty
# value is not "none" for them) and libc's start-up files.
_LOADER_PREFIXES = ("LD_", "MALLOC_")
_LOADER_NAMES = frozenset(
    {"GLIBC_TUNABLES", "GCONV_PATH", "LOCPATH", "NLSPATH", "HOSTALIASES", "RES_OPTIONS"}
)


def loader_variable(name: str) -> bool:
    """Whether a program's loader or libc reads *name* before ``main``.

    Such a name must be absent, never empty, where Docker starts a root
    process (SCRIPT-AGENT-DOCKER-F1): ``docker exec -e NAME`` (no ``=``)
    removes it, provided the Docker client's own environment lacks it.
    Only a shell identifier can be one: glibc and musl look up fixed
    identifier names, so ``LD_ODD-NAME`` is read by nothing, and a shell
    could not unset it from the client's environment either (round 2,
    review B M1)."""
    return (
        name.startswith(_LOADER_PREFIXES) or name in _LOADER_NAMES
    ) and _SHELL_NAME.fullmatch(name) is not None


_SHELL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def loader_switches(entries: Any) -> list[str]:
    """The loader names *entries* (Config.Env rows) set with a value, an
    empty one included, of those a root system shell unsets, outside
    :data:`SYSTEM_SHELL_EMPTY_VARS`: glibc acts on LD_SHOW_AUXV's or
    LD_TRACE_LOADED_OBJECTS' mere presence, before the shell's own unset, and
    empty is not none for them. Where Docker starts root's shell they are
    removed (a bare ``-e NAME``), never emptied; the file-naming names stay
    empty (SCRIPT-AGENT-DOCKER-F1 round 2)."""
    names = set()
    for row in entries if isinstance(entries, list) else ():
        if isinstance(row, str):
            name, separator, _ = row.partition("=")
            if (
                separator
                and name in SYSTEM_SHELL_UNSET_VARS
                and name not in SYSTEM_SHELL_EMPTY_VARS
                and loader_variable(name)
            ):
                names.add(name)
    return sorted(names)


#: The task's Python search-path inputs, kept as data under these names
#: before the route unsets them, as the image PATH is: root's site-hook
#: cleanup reads them as arguments, so it clears the folders test.sh's
#: Python searches (U5a-CONF-5) while no root Python ever reads them
#: (the same names as py8/v2-on-pt).
PYTHON_SEARCH_DATA = {
    "PYTHONPATH": "_BENCHFLOW_IMAGE_PYTHONPATH",
    "PYTHONHOME": "_BENCHFLOW_IMAGE_PYTHONHOME",
    "PYTHONUSERBASE": "_BENCHFLOW_IMAGE_PYTHONUSERBASE",
}


def _keep_python_search_data() -> str:
    return "".join(
        f'{data}="${{{data}-${{{name}-}}}}"; export {data}; '
        for name, data in PYTHON_SEARCH_DATA.items()
    )


def system_shell_prefix(path: str = SYSTEM_PATH) -> str:
    """Shell text: keep the image PATH as data, then unset and pin.

    The image PATH is kept in ``_BENCHFLOW_IMAGE_PATH`` (unless an outer
    layer kept it already), the variable the PATH lock and the verifier's
    PATH probe read; only builtins run before PATH is *path*
    (:data:`SYSTEM_PATH`, or :data:`ROOT_EXEC_PATH` for root plumbing).
    The Python search-path inputs are kept the same way
    (:data:`PYTHON_SEARCH_DATA`). Then every other ``PYTHON*`` variable goes
    too (an image's ``PYTHONUSERBASE`` or ``PYTHONWARNINGS`` can make root's
    Python load code), read with the system ``env`` and ``sed`` once nothing
    can load into them; BenchFlow's own values come after, from its env file.
    """
    return (
        '_BENCHFLOW_IMAGE_PATH="${_BENCHFLOW_IMAGE_PATH-$PATH}"; '
        "export _BENCHFLOW_IMAGE_PATH; "
        + _keep_python_search_data()
        + f"unset {' '.join(SYSTEM_SHELL_UNSET_VARS)}; PATH={path}; export PATH; "
        "for bf_v in $(env | sed -n 's/^\\(PYTHON[A-Za-z0-9_]*\\)=.*/\\1/p'); "
        'do unset "$bf_v"; done; unset bf_v'
    )


@dataclass(frozen=True)
class ExecRoute:
    """How one exec runs (C8), as :meth:`BaseSandbox.exec_route` picks it.

    *pin* is the PATH a plumbing exec pins (None for the task route);
    *task_root*, a task command that runs as uid 0; *root*, the exec runs as
    uid 0 (or as the image's user, which is not looked up); *explicit_root*,
    the exec names a user whose uid is 0 (``"root"``, ``0``, ``"00"``, a
    second name for uid 0); *own_vm*, a command on BenchFlow's own DinD VM,
    outside the task's containers (root, the VM's PATH, no task values);
    *root_path*, the PATH a uid-0 command pins (the sandbox's
    :attr:`BaseSandbox.root_exec_path`); *shell*, the absolute shell the
    backend starts for a plumbing or system-shell exec, and *bash* where a
    backend needs bash syntax: once the sandbox holds its protected runtime
    kit (:mod:`benchflow._runtime_tree`), the kit's, never the image's;
    *tools*: the PATH of BenchFlow's own steps around a command (the env
    file's ``base64`` and ``rm``, PID 1's ``tr``): the kit's on the kit,
    else :data:`SYSTEM_PATH`; *kit*: the route runs on the kit.

    A backend's lowest exec (the one that hands a command to the provider)
    takes a route and has no default, so no probe or clean-up reaches the
    provider without one.
    """

    pin: str | None
    task_root: bool
    system_shell: bool = False
    root: bool = True
    explicit_root: bool = False
    own_vm: bool = False
    root_path: str = ROOT_EXEC_PATH
    shell: str = SYSTEM_SHELL
    bash: str = "/bin/bash"
    tools: str = SYSTEM_PATH
    kit: bool = False


# The route of ``exec(..., user="root", system_shell=True)``, for a backend's
# own root steps that run below its ``exec`` (Daytona's upload prep and
# credential scrub).
SYSTEM_SHELL_ROUTE = ExecRoute(
    pin=SYSTEM_PATH, task_root=False, system_shell=True, explicit_root=True
)
# The route that reads the protected runtime kit's record: the kit's own
# shell by absolute path, the kit's bin alone on PATH. An image without the
# kit has no such shell, so the exec fails without running anything.
KIT_PROBE_ROUTE = ExecRoute(
    pin=KIT_BIN,
    task_root=False,
    system_shell=True,
    explicit_root=True,
    shell=KIT_SH,
    bash=KIT_SH,
    tools=KIT_BIN,
    kit=True,
)


async def system_shell_route(sandbox: Any, service: str = "main") -> ExecRoute:
    """The route of a backend's own root step below its ``exec`` (upload
    prep, the credential scrub, a start probe): ``exec_route``'s system-shell
    route, the protected kit's once the sandbox has one (C8). A stand-in
    without a real ``exec_route`` gets :data:`SYSTEM_SHELL_ROUTE`."""
    exec_route = getattr(sandbox, "exec_route", None)
    if not inspect.iscoroutinefunction(exec_route):
        return SYSTEM_SHELL_ROUTE
    if service == "main":
        return await exec_route("root", system_shell=True)
    return await exec_route("root", system_shell=True, service=service)


class RuntimeKitBrokenError(RuntimeError):
    """The sandbox's protected runtime kit is no longer as BenchFlow made it."""


class SystemShellUnavailableError(RuntimeError):
    """A sandbox cannot start root's shell by absolute path (C8)."""


def require_system_shell(sandbox: Any, what: str) -> None:
    """Refuse ``what`` on a sandbox without ``exec(..., system_shell=True)``.

    Root plumbing never falls back to a shell the image's PATH picks: that
    PATH can lead into a folder the agent wrote.
    """
    if getattr(sandbox, "supports_system_shell", False) is not True:
        raise SystemShellUnavailableError(
            f"{what} runs as root and needs a sandbox that starts root's shell "
            f"by absolute path with the system PATH; {type(sandbox).__name__} "
            "does not (no supports_system_shell). Use the Docker, remote "
            "Docker, Daytona, Modal, Apple container or AgentCore sandbox, or "
            "give this backend exec(..., system_shell=True)."
        )


async def system_shell_exec(
    sandbox: Any, command: str, *, what: str, **kwargs: Any
) -> ExecResult:
    """``command`` as root on the sandbox's system shell, or refuse."""
    require_system_shell(sandbox, what)
    return await exec_raw(
        sandbox.exec, command, user="root", system_shell=True, **kwargs
    )


# The image's values a root step on the protected kit keeps: plain data the
# container was created with (proxies, locale, terminal), none naming a file
# or folder a program loads. Everything else starts empty (C8).
ROUTE_DATA_VARS = (
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "LANG",
    "LC_ALL",
    "TERM",
)


#: Shell text defining ``bf_carry NAME VALUE``, builtins only: prints
#: ``NAME='VALUE'; export NAME`` (VALUE single-quoted, its quotes escaped),
#: so a value reaches the next shell on a pipe, never in an argv.
CARRY_FUNC = (
    'bf_carry() { bf_q="\'"; bf_s=$2; bf_r=; while :; do case $bf_s in '
    '*"$bf_q"*) bf_r=$bf_r${bf_s%%"$bf_q"*}$bf_q\\\\$bf_q$bf_q; '
    'bf_s=${bf_s#*"$bf_q"};; *) break;; esac; done; '
    'printf \'%s=%s%s%s; export %s\\n\' "$1" "$bf_q" "$bf_r$bf_s" "$bf_q" '
    '"$1"; }; '
)


def with_system_path(command: str, path: str = SYSTEM_PATH, keep: Any = ()) -> str:
    """``command`` with :data:`SYSTEM_SHELL_UNSET_VARS` unset and PATH set to *path* first.

    On the protected kit (*path* starts with its bin) root's process starts
    from nothing (C8, round 9): the kit's ``env -i`` re-executes the kit's
    shell with only :data:`ROUTE_DATA_VARS`, the image PATH as data, PATH
    *path* and HOME the kit's empty one, so no value of the image's (a
    library, a startup file, a Python path, a HOME the agent can write)
    reaches it. *keep*: more names to carry, the values BenchFlow hands
    over as the process environment (Modal); never a route-owned one.
    Every carried value (the route data, *keep*'s, the image PATH and the
    Python search data) crosses ``env -i`` on a pipe, by name, never as an
    ``N=value`` argument: ``env``'s argv holds
    only the route's PATH and HOME.
    Nothing may follow this text: the shell ``exec``s.
    """
    inner = f"{system_shell_prefix(path)}\n{command}"
    if path.split(":", 1)[0] != KIT_BIN:
        return inner
    fixed = {"_BENCHFLOW_IMAGE_PATH", *PYTHON_SEARCH_DATA.values()}
    names = [*ROUTE_DATA_VARS, *(n for n in keep if not route_owned_name(n))]
    # The carried values cross env -i on a pipe (fd 3), by name, never as
    # ``N=value`` words in its argv: the outer
    # shell's builtins quote each one into the heredoc, the new shell reads
    # it with the kit's cat and evals it. The route's own values come last
    # and win, as their words did after the kept ones.
    carried = "".join(
        f'[ -z "${{{n}+x}}" ] || bf_carry {n} "${n}"; '
        for n in dict.fromkeys(names)
        if n.isidentifier() and n not in fixed
    )
    carried += 'bf_carry _BENCHFLOW_IMAGE_PATH "${_BENCHFLOW_IMAGE_PATH-$PATH}"; '
    carried += "".join(
        f'bf_carry {data} "${{{data}-${{{name}-}}}}"; '
        for name, data in PYTHON_SEARCH_DATA.items()
    )
    load = (
        f'eval "$({KIT_BIN}/cat <&3)"; exec 3<&-; '
        f"PATH={shlex.quote(path)}; HOME={ROOT_HOME}; export PATH HOME\n"
    )
    return (
        CARRY_FUNC + f"exec {KIT_BIN}/env -i PATH={shlex.quote(path)} HOME={ROOT_HOME} "
        f"{KIT_SH} -c {shlex.quote(load + inner)} 3<<_BF_CARRY_\n"
        f"$({carried})\n"
        "_BF_CARRY_"
    )


# Two routes for a command BenchFlow sends (C8):
#
# - Plumbing: ``exec(..., system_shell=True)``, and every exec as root or as
#   the image's own user (``user=None``) outside :meth:`BaseSandbox.task_command`.
#   The shell is started by absolute path with the loader and start-up
#   variables empty where BenchFlow starts the process; its first text,
#   builtins only, unsets them and pins PATH (:data:`SYSTEM_PATH` for the
#   system shell, :data:`ROOT_EXEC_PATH` otherwise: BenchFlow's root steps
#   run ``python3``, which a Python image keeps in /usr/local/bin); and the
#   task's persistent env reaches it without those variables
#   (:func:`plumbing_env`). The agent may have replaced a library the
#   image's ``LD_PRELOAD`` names, or a file its ``BASH_ENV`` names, in an
#   earlier turn; neither ever loads into these processes.
# - Task: inside :meth:`BaseSandbox.task_command`, the task's own command
#   (test.sh and the verifier's runner, the oracle's solve.sh, task-declared
#   services and setup, an SDK caller's ``Runtime.exec``) keeps the image's
#   PATH and loader and start-up variables, as before. Its shell still
#   starts with them empty; they come back (:func:`restore_image_env`) after
#   BenchFlow's env file is sourced, just before the command, so only the
#   task's command and what it starts see them. A named non-root user
#   (``user="agent"``) keeps its environment on both routes.
#
# Where the provider starts the first shell itself (Daytona's toolbox,
# AgentCore's command runner), the image's values are moved under
# IMAGE_ENV_PREFIX names at the provider (sandbox env, image ENV), so that
# shell starts with them empty too.
IMAGE_ENV_PREFIX = "_BF_IMAGE_"


def uid_spelling(user: str | int | None) -> int | None:
    """The uid *user* spells without a lookup: a number in any spelling
    (``0``, ``"00"``, ``"0:0"``) or ``"root"``; ``None`` for another name."""
    if user is None or isinstance(user, bool):
        return None
    if isinstance(user, int):
        return user
    name = str(user).split(":", 1)[0]
    if name == "root":
        return 0
    if name.isascii() and name.isdigit():
        return int(name)
    return None


def is_root_user(user: str | int | None) -> bool:
    """Whether *user* (an exec's ``user``) spells uid 0 without a lookup.

    A second name for uid 0 is not seen here: routes come from
    :meth:`BaseSandbox.exec_route`, which looks names up.
    """
    return uid_spelling(user) == 0


# The names bash itself imports at start, before the first command it reads:
# an exported function (``BASH_FUNC_<any name>%%``) and its option sets. Root's
# shells may be bash (Daytona's toolbox, the route's su), and a function
# overrides every builtin, ``unset`` and ``export`` included, so these are
# start-up code like the loader's (K1-R16-2).
BASH_STARTUP_PREFIX = "BASH_FUNC_"
BASH_STARTUP_VARS = ("SHELLOPTS", "BASHOPTS")


def bash_startup_name(name: str) -> bool:
    """Whether bash imports *name* at start (:data:`BASH_STARTUP_PREFIX`,
    :data:`BASH_STARTUP_VARS`)."""
    return name.startswith(BASH_STARTUP_PREFIX) or name in BASH_STARTUP_VARS


def startup_name(name: str) -> bool:
    """The C8 rule's one list of start-up names: what a shell or the loader
    reads from its environment before the first command BenchFlow sends.

    Every backend takes its names from here, with the same
    prefix match:

    - :data:`SYSTEM_SHELL_EMPTY_VARS` are set empty where the provider or
      Docker starts root's first shell, and the image's values are carried
      aside under :data:`IMAGE_ENV_PREFIX` names for the task's own command
      (:func:`restore_image_env`).
    - bash's (:func:`bash_startup_name`) are refused when an image or a
      sandbox env sets one to a value (:func:`refused_startup_names`): no
      shell can give a ``%%`` name back to the task's command, and bash keeps
      SHELLOPTS and BASHOPTS read-only, so carrying them aside would change
      what the task's own command runs. An empty value defines nothing.
    """
    return name in SYSTEM_SHELL_EMPTY_VARS or bash_startup_name(name)


def startup_name_pattern() -> str:
    """A shell ``case`` pattern that matches exactly :func:`startup_name`."""
    return "|".join(
        (*SYSTEM_SHELL_EMPTY_VARS, *BASH_STARTUP_VARS, f"{BASH_STARTUP_PREFIX}*")
    )


def environ_rows(data: bytes) -> list[tuple[str, str]]:
    """``(name, value)`` for each ``NAME=VALUE`` entry of a process's
    environment block (``/proc/<pid>/environ``: NUL-terminated entries)."""
    rows = []
    for entry in data.split(b"\0"):
        name, separator, value = entry.partition(b"=")
        if separator:
            rows.append(
                (
                    name.decode("utf-8", "surrogateescape"),
                    value.decode("utf-8", "surrogateescape"),
                )
            )
    return rows


def refused_startup_names(rows: Any) -> list[str]:
    """The bash start-up names (:func:`bash_startup_name`) *rows* set to a
    value: Config.Env strings (``NAME=VALUE``) or ``(name, value)`` pairs."""
    names = set()
    for row in rows or ():
        if isinstance(row, str):
            name, separator, value = row.partition("=")
            if not separator:
                continue
        else:
            name, value = row
        if bash_startup_name(name) and value:
            names.add(name)
    return sorted(names)


def startup_refusal(names: Any, where: str) -> str:
    """The ``benchflow:`` line for an image whose environment sets bash
    start-up names (:func:`refused_startup_names`). Names only, never a
    value."""
    shown = ", ".join(sorted(names))
    if len(shown) > 300:
        shown = shown[:300] + " ..."
    return (
        f"benchflow: {where} sets bash start-up names ({shown}); bash would run "
        "them as root before any command BenchFlow sends, and they cannot be "
        "given back to the task's own command; remove them from the image or "
        "the sandbox's environment"
    )


#: argv a backend whose provider starts processes with the image's ENV (Modal,
#: Apple) runs once at start, no shell, to read that environment exactly:
#: ``cat``'s own start environment is what every shell there starts with.
STARTUP_ENV_ARGV = ("/bin/cat", "/proc/self/environ")


def startup_env_refusal(output: Any, code: int, where: str) -> str | None:
    """The ``benchflow:`` line for :data:`STARTUP_ENV_ARGV`'s result
    (*output*: bytes or text, exit *code*) when the image sets bash
    start-up names, or when the environment could not be read; None when
    it sets none."""
    if code != 0:
        return f"benchflow: could not read {where}'s start environment"
    data = (
        output
        if isinstance(output, bytes)
        else str(output or "").encode("utf-8", "surrogateescape")
    )
    refused = refused_startup_names(environ_rows(data))
    return startup_refusal(refused, where) if refused else None


class StartupNamesRefused(RuntimeError):
    """An image or sandbox env sets bash start-up names
    (:func:`startup_refusal`): the sandbox does not start."""


def route_owned_name(name: str) -> bool:
    """Whether *name* is one only the route sets for root's plumbing (C8):
    PATH, the loader's and the shells' start-up variables (bash's imported
    functions and option sets included), every ``LD_*`` and every
    ``PYTHON*``. The route builds these once; nothing a caller passes, and
    nothing an env file sources afterwards, may set them again.
    """
    return (
        name == "PATH"
        or name in SYSTEM_SHELL_UNSET_VARS
        or startup_name(name)
        or name.startswith(("LD_", "PYTHON"))
    )


def plumbing_caller_env(env: dict[str, str] | None) -> dict[str, str] | None:
    """A caller's *env* for a plumbing exec, without :func:`route_owned_name`
    names: root's PATH, loader and Python settings are the route's alone."""
    if not env:
        return env
    return {k: v for k, v in env.items() if not route_owned_name(k)}


def plumbing_env(env: dict[str, str]) -> dict[str, str]:
    """*env* (the task's persistent env) without loader or start-up variables.

    Its PATH is kept as ``_BENCHFLOW_IMAGE_PATH`` (data the PATH lock reads),
    never as the command's PATH, and its Python search-path inputs as
    :data:`PYTHON_SEARCH_DATA`: inert names no interpreter reads, which only
    the site-hook clean-up reads as data (PT-R16).
    """
    out = {
        k: v
        for k, v in env.items()
        if k not in SYSTEM_SHELL_UNSET_VARS
        and not startup_name(k)
        and not k.startswith(("LD_", "PYTHON"))
    }
    if "PATH" in out:
        path = out.pop("PATH")
        out.setdefault("_BENCHFLOW_IMAGE_PATH", path)
    for name, data in PYTHON_SEARCH_DATA.items():
        if name in env:
            out.setdefault(data, env[name])
    return out


def restore_image_env(skip: Any = (), tools: str | None = None) -> str:
    """Builtins: the image's loader and start-up values back, from their
    IMAGE_ENV_PREFIX copies, for the names *skip* (the env BenchFlow
    sourced, which wins) does not hold. The copies are removed.

    With *tools* (the protected kit's PATH: the task's own command as root
    on the kit, test.sh included) a value comes back only when every file
    or folder it names is root's alone along its whole route
    (:data:`~benchflow._root_only.ROOT_CHAIN_FUNCS`, run with *tools*);
    otherwise it stays unset and a ``benchflow:`` line names it
    (PY8-PT-R8-7): an image's ``LD_AUDIT`` or ``LD_PRELOAD`` naming a
    library the agent can replace never loads into root's command.
    """
    parts = []
    if tools is not None:
        # bf_safe_value NAME VALUE: the seal recorded NAME=VALUE before the
        # agent's turn (KIT_IMAGE_ENV_OK, PT-C8-R9-5 astra / R9-3 sol), and
        # every file it names is still root's alone.
        parts.append(
            f"bf_safe_value() ( PATH={shlex.quote(tools)}; {ROOT_CHAIN_FUNCS}"
            f"{env_value_check_func()}"
            "bf_seen=; while IFS= read -r l; do "
            '[ "$l" = "$1=$2" ] && bf_seen=1; '
            f"done < {KIT_IMAGE_ENV_OK} 2>/dev/null; "
            '[ -n "$bf_seen" ] || exit 1; '
            'bf_value_ok "$1" "$2" )'
        )
        parts.append(env_value_clean_func().rstrip("; "))
    for name in IMAGE_RESTORED_VARS:
        kept = f"{IMAGE_ENV_PREFIX}{name}"
        if name not in skip and name not in IMAGE_UNRESTORED_VARS:
            if tools is None:
                parts.append(
                    f'if [ -n "${{{kept}-}}" ]; then {name}=${kept}; export {name}; fi'
                )
            else:
                # The value the exec started with goes first: an image value
                # Docker passed straight into the exec (a container S1 did
                # not bootstrap) is the same one PID 1 holds, and is judged
                # here like the rest (PT-C8-CONF-6).
                parts.append(
                    f'if [ -n "${{{kept}-}}" ]; then unset {name}; '
                    f'if bf_safe_value {name} "${kept}"; '
                    f'then {name}=$(bf_value_clean {name} "${kept}"); export {name}; else echo '
                    f"'benchflow: not restoring {name} for root: a file it names "
                    "was not root-only before the agent ran, or is not now' >&2; fi; fi"
                )
        parts.append(f"unset {kept}")
    if tools is not None:
        parts.append("unset -f bf_safe_value bf_value_clean")
    return "; ".join(parts) + "; "


def capture_pid1_image_env(tools: str = SYSTEM_PATH) -> str:
    """Shell text: copy PID 1's loader and start-up values (the image's and
    the compose file's, which an exec that set them empty does not see) to
    their IMAGE_ENV_PREFIX names, for :func:`restore_image_env`.

    Runs first in a shell started with those variables empty, so the
    ``tr`` it names by path loads nothing they name: the system one, or
    the protected kit's when *tools* (the route's
    :attr:`ExecRoute.tools`) is the kit's. A shell that cannot read PID 1's
    environment (another user's) copies nothing.
    """
    arms = "".join(
        f'{n}=*) [ -n "${{{IMAGE_ENV_PREFIX}{n}+x}}" ] || '
        f"{{ {IMAGE_ENV_PREFIX}{n}=${{_bf_l#{n}=}}; export {IMAGE_ENV_PREFIX}{n}; }};; "
        for n in IMAGE_RESTORED_VARS
    )
    tr = f"{KIT_BIN}/tr" if tools.split(":", 1)[0] == KIT_BIN else "/usr/bin/tr"
    return (
        "if [ -r /proc/1/environ ]; then\n"
        f"while IFS= read -r _bf_l; do case $_bf_l in {arms}esac; done <<_BF_PID1_ENV\n"
        f"$({tr} '\\000' '\\n' < /proc/1/environ 2>/dev/null)\n"
        "_BF_PID1_ENV\n"
        "fi; unset _bf_l\n"
    )


class BaseSandbox(ABC):
    """Abstract base for sandbox environments (Docker, Daytona, Modal).

    Provides the containerized execution environment for agent rollouts.

    Every transfer primitive a subclass defines (``upload_file``,
    ``upload_dir``, ``live_process``, ``script_runner_upload``) takes a required ``admission`` keyword:
    the :class:`~benchflow.trust.state.Admission` that ``trust.admit()``
    returns for this sandbox (TRUSTED-RUNTIME §5). The subclass's own method
    never sees it.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        from benchflow.trust.state import TRANSFER_PRIMITIVES, guard_transfer

        for name in TRANSFER_PRIMITIVES:
            impl = cls.__dict__.get(name)
            if (
                impl is None
                or getattr(impl, "__isabstractmethod__", False)
                or getattr(impl, "__benchflow_guarded__", False)
            ):
                continue
            setattr(cls, name, guard_transfer(name, impl))

    def _exec_env_upload(self) -> Callable[..., Awaitable[None]]:
        """``upload_file`` for an exec's own environment file, which
        ``uploaded_agent_env`` stages: BenchFlow's bytes for an exec that is
        admitted where it is called."""
        from benchflow.trust.state import TRUSTED_TRANSFER, admit

        return functools.partial(
            self.upload_file,
            admission=admit(
                self,
                "main",
                TRUSTED_TRANSFER,
                reason="an exec's environment, staged by BenchFlow",
            ),
        )

    environment_dir: Path
    # Generic exec's command shell; Docker establishes the barrier at creation.
    agent_exec_shell = "/bin/sh"
    agent_container_no_new_privileges = False
    environment_name: str
    session_id: str
    rollout_paths: RolloutPaths | None
    task_env_config: SandboxConfig
    logger: logging.Logger
    default_user: str | int | None
    # PY8 U12: class defaults, so sandboxes built with __new__ (tests) have
    # them. Each start, stop and restore moves to a new generation.
    _exec_generation: int = 0
    _kept_identity: KeptIdentity | None = None
    # (generation, uid by (service, user name)) for exec_route's lookups.
    _named_uids: tuple[int, dict[tuple[str, str], int]] | None = None
    # (generation, the kit, or "broken", or None for none) for exec_route.
    _runtime_kit_state: tuple[int, RuntimeKit | str | None] | None = None
    # Set by _runtime_kit_unknown (a restore, a start from a snapshot): the
    # next root step reads the kit. Otherwise none is assumed until
    # setup_sandbox_user makes or keeps one.
    _runtime_kit_probe: bool = False
    # (generation, kit, "broken" or None) per compose service other than
    # ``main``: each service that gets root's plumbing has its own kit (C8,
    # PY8-PT-R8-6, PT-C8-R8-3).
    _service_kits: dict[str, tuple[int, RuntimeKit | str | None]] | None = None
    # /usr/local folders root's plumbing no longer trusts (root_exec_path).
    _distrusted_root_path: frozenset[str] = frozenset()
    # Set by the sandbox factory for a task that keeps services
    # (x-keep-services): a Compose provider then stacks
    # docker-compose-kept-freezer.yaml on main (compose_needs_kept_freezer),
    # so root can hold the kept processes in a frozen cgroup (K1-CONFIRM-1).
    kept_service_freezer: bool = False
    # What runs agent code in this sandbox of a task that keeps services,
    # set by the rollout before start(): "oracle" or "agent"; None for nop,
    # a verifier-only sandbox and everything else, which never ask for the
    # freezer (_compose.decide_kept_service_freezer).
    kept_service_runner: str | None = None
    # The Engine's reason it cannot give the freezer this start asked for
    # (None when it gave it or nobody asked): the rollout refuses a non-root
    # agent that keeps services with it, once identities are measured
    # .
    kept_service_freezer_refusal: str | None = None
    # The accounts the task lets its agent switch to (x-agent-identity-switch,
    # K1-R11-1), set by the rollout only where the agent's container has
    # no_new_privs (K1-R15-2): the kills stop these accounts' processes too.
    agent_identity_switch: tuple[str, ...] = ()
    # How bfx, BenchFlow's static helper, reaches /.bfx in this sandbox
    # (TRUSTED-RUNTIME.md §6): "bind" or "volume" (read-only, set when the
    # container is created: holds against any agent), "upload" (root's
    # 0555 folder before the agent: holds against a non-root agent), or None
    # (not delivered; the backend keeps the kit path). See bfx_guarantee.
    bfx_delivery: str | None = None

    @property
    def bfx_guarantee(self) -> str | None:
        """Who bfx's delivery holds against: "any-agent", "non-root-agent",
        or None when bfx is not delivered."""
        from benchflow._bfx.delivery import GUARANTEE

        mode = getattr(self, "bfx_delivery", None)
        return GUARANTEE.get(mode) if mode else None

    async def identity_container_config(
        self, service: str = "main"
    ) -> dict[str, Any] | None:
        """Default exec identity, when the backend exposes Docker Config."""
        return None

    async def identity_exec(
        self, command: str, *, service: str = "main", **kwargs: Any
    ) -> ExecResult:
        """Read trusted accounts before the agent starts, using POSIX sh."""
        return await exec_raw(
            self.exec, command, user="root", service=service, **kwargs
        )

    async def script_container_config(self) -> dict[str, Any] | None:
        """Docker Config for operator-script env parity, or the legacy capture."""
        return None

    async def script_runner_exec(self, command: str, **kwargs: Any) -> ExecResult:
        """Supervise scripts only with a backend's trusted root transport.

        An env -i inside ordinary exec is too late if that transport has
        already started an image-PATH shell or loaded agent-owned hooks.
        """
        raise RuntimeError(
            "script: this sandbox lacks a trusted root runner transport; use Docker or remote Docker"
        )

    async def script_runner_upload(self, source: str, target: str) -> None:
        """Upload into an already prepared, protected runner directory."""
        raise RuntimeError("script: this sandbox lacks a trusted runner upload")

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        rollout_paths: RolloutPaths | None,
        task_env_config: SandboxConfig,
        _logger: logging.Logger | None = None,
        override_cpus: int | None = None,
        override_memory_mb: int | None = None,
        override_storage_mb: int | None = None,
        override_gpus: int | None = None,
        suppress_override_warnings: bool = False,
        persistent_env: dict[str, str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.environment_dir = environment_dir
        self.environment_name = environment_name
        self.session_id = session_id
        self.rollout_paths = rollout_paths
        self.default_user = None
        self.task_env_config = task_env_config

        self._override_cpus = override_cpus
        self._override_memory_mb = override_memory_mb
        self._override_storage_mb = override_storage_mb
        self._override_gpus = override_gpus
        self._suppress_override_warnings = suppress_override_warnings
        self._persistent_env: dict[str, str] = persistent_env or {}

        self.logger = (_logger or logger).getChild(type(self).__name__)

        self._maybe_override_task_env_config()
        self._maybe_resolve_task_env()
        self._validate_definition()

    @property
    def _uses_compose(self) -> bool:
        return False

    @property
    def is_mounted(self) -> bool:
        """Whether the rollout dir is host-bind-mounted into the sandbox.

        When ``True`` the agent container's verifier output is already visible
        on the host, so the verifier can skip a ``download_dir`` round-trip.
        Backends that run remotely (Daytona, Modal) have no bind mount and
        override this to ``False``. The default is ``False`` so non-mounted
        backends are the safe assumption.
        """
        return False

    def _maybe_resolve_task_env(self) -> None:
        if self.task_env_config.env and not self._uses_compose:
            resolved = resolve_env_vars(self.task_env_config.env)
            self._persistent_env = {**resolved, **self._persistent_env}

    def _maybe_override_task_env_config(self) -> None:
        if self._override_cpus is not None:
            self.task_env_config.cpus = self._override_cpus
            if not self._suppress_override_warnings:
                self.logger.warning(
                    "Overriding CPU count to %d alters the task from its "
                    "intended configuration.",
                    self._override_cpus,
                )
        if self._override_memory_mb is not None:
            self.task_env_config.memory_mb = self._override_memory_mb
            if not self._suppress_override_warnings:
                self.logger.warning(
                    "Overriding memory to %d MB alters the task from its "
                    "intended configuration.",
                    self._override_memory_mb,
                )
        if self._override_storage_mb is not None:
            self.task_env_config.storage_mb = self._override_storage_mb
            if not self._suppress_override_warnings:
                self.logger.warning(
                    "Overriding storage to %d MB alters the task from its "
                    "intended configuration.",
                    self._override_storage_mb,
                )
        if self._override_gpus is not None:
            self.task_env_config.gpus = self._override_gpus
            if not self._suppress_override_warnings:
                self.logger.warning(
                    "Overriding GPU count to %d alters the task from its "
                    "intended configuration.",
                    self._override_gpus,
                )

    def _forget_kept_identity(self) -> None:
        """Drop what the start probe kept. Called when the sandbox starts,
        stops or is restored, so no answer outlives the container it came
        from (a failed start leaves nothing kept).

        What the sandbox knows about its protected runtime kit carries over
        (``stop`` keeps the container); ``start`` and ``restore`` then say
        what the new container holds (:meth:`_runtime_kit_fresh`,
        :meth:`_runtime_kit_unknown`)."""
        state = self._runtime_kit_state
        services = self._service_kits or {}
        self._exec_generation += 1
        self._kept_identity = None
        if state is not None and state[0] == self._exec_generation - 1:
            self._runtime_kit_state = (self._exec_generation, state[1])
        self._service_kits = {
            name: (self._exec_generation, kit)
            for name, (generation, kit) in services.items()
            if generation == self._exec_generation - 1
        }

    def _runtime_kit_fresh(self) -> None:
        """A container just made from the task's image: no kit yet, so no
        exec reads one (setup_sandbox_user makes it, or keeps one an image
        already holds after checking it)."""
        self._runtime_kit_state = (self._exec_generation, None)
        self._runtime_kit_probe = False
        self._service_kits = {}

    def _runtime_kit_unknown(self) -> None:
        """A container from a snapshot: the first root step reads its kit."""
        self._runtime_kit_state = None
        self._runtime_kit_probe = True
        self._service_kits = {}

    def _keep_identity(
        self,
        *,
        user: str | int | None,
        env: dict[str, str] | None,
        pwd: str | None,
        uid: str | None,
    ) -> None:
        """Keep the start probe's answers; *user* and *env* are the ones its
        exec ran with."""
        if pwd is None and uid is None:
            return
        self._kept_identity = KeptIdentity(
            generation=self._exec_generation,
            user=user,
            env_key=env_key(env),
            pwd=pwd,
            uid=uid,
        )

    def kept_identity(self) -> KeptIdentity | None:
        """The kept identity, if an exec as the default user with no extra
        environment would run exactly as the probe's did (see
        :class:`KeptIdentity`)."""
        kept = self._kept_identity
        if kept is None or kept.generation != self._exec_generation:
            return None
        if kept.user != self._resolve_user(None):
            return None
        if kept.env_key != env_key(self._merge_env(None)):
            return None
        return kept

    def _resolve_user(self, user: str | int | None) -> str | int | None:
        return user if user is not None else self.default_user

    @contextlib.contextmanager
    def services_kept(self) -> Iterator[None]:
        """Mark, for the current task, that agent code (the services a task
         keeps) runs beside root's commands in this sandbox.

         While it does, exec adds no persistent env: every backend's exec
         would wrap it into a file the command creates under /tmp and sources
         as root, and a kept service that can write /tmp can replace that file
         first; the verifier hands the persistent env to its own command
         through a file only root can reach instead. And a
         backend leaves the owner of what it copies out of the sandbox alone
        .
        """
        token = _SERVICES_KEPT.set(_SERVICES_KEPT.get() | {id(self)})
        try:
            yield
        finally:
            _SERVICES_KEPT.reset(token)

    # Whether exec's env reaches the command's process with no value on any
    # command line (Docker's and Apple's stdin), so a verifier user other
    # than root gets it privately beside live agent code. A backend that
    # writes small envs into the command (Daytona) does not; nor does one
    # nobody checked (b162 PY8-MAIN-36).
    exec_env_is_private: bool = False

    @property
    def _services_are_kept(self) -> bool:
        return id(self) in _SERVICES_KEPT.get()

    @contextlib.contextmanager
    def task_command(self) -> Iterator[None]:
        """Mark the execs inside as the task's own command (C8's task route).

        test.sh and the verifier's runner, the oracle, task-declared
        services and setup, and an SDK caller's ``Runtime.exec`` run here:
        they keep the image's PATH and loader and start-up variables (see
        ``IMAGE_ENV_PREFIX``). Everything else BenchFlow runs as root or as
        the image's user is plumbing and gets neither.
        """
        with as_task_command(self):
            yield

    @property
    def _in_task_command(self) -> bool:
        return in_task_command()

    async def exec_route(
        self,
        user: str | int | None,
        *,
        system_shell: bool = False,
        plumbing: bool = False,
        service: str = "main",
    ) -> ExecRoute:
        """The route of one exec as *user* in *service* (C8). Every
        backend's ``exec`` picks its route here and nowhere else.

        By effective uid, not by name: a name that is not ``root`` and no
        number is looked up once per sandbox generation in the container
        that runs the command, through its own exec (:meth:`_named_user_uid`),
        so a second name for uid 0 (``toor``) is root, as is ``"00"``, and a
        name another service's account database gives another uid is not.
        ``user=None`` runs as the image's user, root in most images, and is
        taken as root without a lookup.

        - System shell: plumbing, :data:`SYSTEM_PATH`.
        - Inside :meth:`task_command`: the task route; a uid-0 exec starts
          with the loader values empty and gets the image's back before the
          command (:func:`restore_image_env`).
        - Otherwise uid 0: plumbing, :data:`ROOT_EXEC_PATH`. Any other uid:
          that user's own environment.

        Once the sandbox holds its protected runtime kit (``main`` only,
        where the sandbox user lives; :mod:`benchflow._runtime_tree`), the
        plumbing and system-shell routes start the kit's shell and pin the
        kit's PATH instead (:meth:`runtime_kit_for_route`); a kit that no
        longer matches what BenchFlow made refuses every root step.

        *plumbing*: a backend's own step below its ``exec`` (AgentCore's
        sealed channel), BenchFlow's even when a task command started it.
        """
        if id(self) in _KIT_PROBING.get():
            return KIT_PROBE_ROUTE
        resolved = self._resolve_user(user)
        if resolved is None:
            root = True
        else:
            uid = uid_spelling(resolved)
            if uid is None:
                uid = await self._named_user_uid(
                    str(resolved).split(":", 1)[0], service=service
                )
            root = uid == 0
        if system_shell:
            pin: str | None = SYSTEM_PATH
        elif self._in_task_command and not plumbing:
            pin = None
        else:
            pin = self.root_exec_path if root else None
        task_command = self._in_task_command and not plumbing
        shell, bash, tools, on_kit = SYSTEM_SHELL, "/bin/bash", SYSTEM_PATH, False
        # On the kit of the container that runs the command, BenchFlow's own
        # steps around any command use the kit's tools (a provider that
        # starts every command as root and switches user takes the kit's
        # ``su``), and every uid-0 exec starts the kit's shell, the task's
        # own command included (its programs still come from the image's
        # PATH, put back before it runs). Each service has its own kit.
        # A service without a kit of its own refuses root's steps only: a
        # named non-root user's command there runs as before.
        try:
            kit = await self.runtime_kit_for_route(service)
        except RuntimeKitBrokenError:
            if root or system_shell or service == "main":
                raise
            kit = None
        if kit is not None:
            tools, on_kit = kit.path, True
            if root or system_shell:
                shell = kit.shell
                bash = kit.bash_shell or kit.shell
            if pin is not None:
                pin = kit.path
        return ExecRoute(
            pin=pin,
            task_root=pin is None and root and task_command,
            system_shell=system_shell,
            root=root,
            explicit_root=root and user is not None,
            root_path=self.root_exec_path,
            shell=shell,
            bash=bash,
            tools=tools,
            kit=on_kit,
        )

    @property
    def runtime_kit(self) -> RuntimeKit | None:
        """The protected runtime kit this sandbox generation holds, as
        BenchFlow made or found it; None when none is known (yet)."""
        state = self._runtime_kit_state
        if state is None or state[0] != self._exec_generation:
            return None
        return state[1] if isinstance(state[1], RuntimeKit) else None

    def set_runtime_kit(self, kit: RuntimeKit | None, service: str = "main") -> None:
        """Record the kit ``prepare_runtime_tree`` made or kept (C8); for a
        service other than ``main``, None records that it has none."""
        if service != "main":
            self._service_kits = {
                **(self._service_kits or {}),
                service: (self._exec_generation, kit),
            }
            return
        self._runtime_kit_state = (self._exec_generation, kit)
        self._runtime_kit_probe = False

    async def runtime_kit_for_route(self, service: str = "main") -> RuntimeKit | None:
        if service != "main":
            return await self._service_kit_for_route(service)
        return await self._main_kit_for_route()

    async def _service_kit_for_route(self, service: str) -> RuntimeKit | None:
        """*service*'s own kit. Before the main kit exists (before the
        agent's first turn, or no sandbox user) there is none to need; after
        it, root's plumbing in a service without its own kit is refused
        rather than run on that container's image tools."""
        main = await self._main_kit_for_route()
        kept = (self._service_kits or {}).get(service)
        if kept is None or kept[0] != self._exec_generation:
            if main is None:
                return None
            found = await self._load_runtime_kit(service)
            if found == "unknown":
                found = None
            kept = (self._exec_generation, found)
            self.set_runtime_kit(found if found != "broken" else None, service)
            if found == "broken":
                kept = (self._exec_generation, "broken")
        if isinstance(kept[1], RuntimeKit):
            return kept[1]
        if main is None:
            return None
        raise RuntimeKitBrokenError(
            f"the compose service {service!r} has no protected runtime of its "
            "own (BenchFlow could not prepare one before the agent's turn), so "
            "BenchFlow runs no root step there; give that service's image a "
            "root-owned /bin/sh and sha256sum, or keep root steps in main"
        )

    async def _main_kit_for_route(self) -> RuntimeKit | None:
        """The kit root's plumbing runs on in this generation.

        Known from ``prepare_runtime_tree``, or read once per generation
        (a restored snapshot, a branch child, a regrade's sandbox) with the
        kit's own shell, which an image without the kit does not have
        (:func:`benchflow._runtime_tree.kit_load_command`). Raises
        :class:`RuntimeKitBrokenError` when the kit is there but is not
        root's alone or does not match its MANIFEST.
        """
        state = self._runtime_kit_state
        if state is None or state[0] != self._exec_generation:
            if not self._runtime_kit_probe:
                return None
            found = await self._load_runtime_kit()
            if found == "unknown":
                # The sandbox did not answer (not running yet, a timeout):
                # nothing kept, the next exec reads it again.
                return None
            state = (self._exec_generation, found)
            self._runtime_kit_state = state
            self._runtime_kit_probe = False
        if state[1] == "broken":
            raise RuntimeKitBrokenError(
                "BenchFlow's protected runtime (/opt/benchflow/rt) in this "
                "sandbox is not as BenchFlow made it, so no root step runs; "
                "start the trial in a fresh sandbox"
            )
        return state[1] if isinstance(state[1], RuntimeKit) else None

    async def _load_runtime_kit(self, service: str = "main") -> RuntimeKit | str | None:
        """The kit, ``"broken"``, None (no kit: the kit's shell does not
        exist, or it holds no record) or ``"unknown"`` (no answer)."""
        if getattr(self, "supports_runtime_kit", False) is not True:
            return None
        token = _KIT_PROBING.set(_KIT_PROBING.get() | {id(self)})
        try:
            if service == "main":
                result = await exec_raw(
                    self.exec, kit_load_command(), user="root", timeout_sec=60
                )
            else:
                result = await exec_raw(
                    self.exec,
                    kit_load_command(),
                    user="root",
                    timeout_sec=60,
                    service=service,
                )
        except Exception as exc:
            self.logger.debug("No protected runtime kit read: %s", exc)
            return "unknown"
        finally:
            _KIT_PROBING.reset(token)
        code = getattr(result, "return_code", 1)
        kit = parse_kit_record(getattr(result, "stdout", "") or "")
        if code == 0 and kit is not None:
            return kit
        if code in (3, 126, 127):
            return None
        if code == 1 and "benchflow:" in (getattr(result, "stderr", "") or ""):
            return "broken"
        return "unknown"

    @property
    def supports_runtime_kit(self) -> bool:
        """Whether this backend's exec starts :attr:`ExecRoute.shell` for
        plumbing, so the kit can be read and used (every real backend)."""
        return False

    @property
    def root_exec_path(self) -> str:
        """The PATH root's plumbing pins: :data:`ROOT_EXEC_PATH` without a
        /usr/local folder the PATH lock could not show to be root's alone
        (:meth:`distrust_root_path_folders`). The system folders always
        stay, so it is never empty."""
        dropped = self._distrusted_root_path
        if not dropped:
            return ROOT_EXEC_PATH
        system = SYSTEM_PATH.split(":")
        return ":".join(
            d for d in ROOT_EXEC_PATH.split(":") if d in system or d not in dropped
        )

    def distrust_root_path_folders(self, folders: Any) -> None:
        """Leave *folders* out of :attr:`root_exec_path` from now on (C8):
        the PATH lock left them writable by the agent user, or they are in
        its workspace."""
        self._distrusted_root_path = frozenset(self._distrusted_root_path) | set(
            folders
        )

    def _pre_agent_uid(self, name: str, service: str) -> int | None:
        """*name*'s uid in *service* from the account table read before the
        agent started (``_identities.snapshot``), or None where that table
        is absent or does not name the account (TRUSTED-RUNTIME v2 §4: root
        route and account discovery from facts taken while pristine)."""
        from benchflow.sandbox._identities import Identities

        own = vars(self)
        table = (
            own.get("_benchflow_identities")
            if service == "main"
            else (own.get("_benchflow_service_identity_tables") or {}).get(service)
        )
        if not isinstance(table, Identities):
            return None
        try:
            return table.resolve(name).uid
        except (ValueError, TypeError, OverflowError):
            return None

    async def _named_user_uid(self, name: str, *, service: str = "main") -> int:
        """*name*'s uid in *service*, read once per sandbox generation by
        root's system shell in that service's own container (``id -u``
        through its own exec: no image PATH or loader values, and that
        container's account database, never another service's). A name
        that does not resolve to one number counts as uid 0 and is not kept:
        plumbing is the safe route for an exec as an unknown user, which
        fails anyway (PT-C8-R3-4).
        """
        kept = self._named_uids
        if kept is None or kept[0] != self._exec_generation:
            kept = (self._exec_generation, {})
            self._named_uids = kept
        key = (service, name)
        if key in kept[1]:
            return kept[1][key]
        if not name or name.startswith("-"):
            return 0
        known = self._pre_agent_uid(name, service)
        if known is not None:
            kept[1][key] = known
            return known
        if getattr(self, "supports_system_shell", False) is not True:
            return 0
        try:
            if service == "main":
                result = await exec_raw(
                    self.exec,
                    f"id -u {shlex.quote(name)}",
                    user="root",
                    system_shell=True,
                    timeout_sec=30,
                )
            else:
                result = await exec_raw(
                    self.exec,
                    f"id -u {shlex.quote(name)}",
                    user="root",
                    system_shell=True,
                    timeout_sec=30,
                    service=service,
                )
        except Exception as exc:
            self.logger.debug("Cannot look up user %s in %s: %s", name, service, exc)
            return 0
        out = (getattr(result, "stdout", None) or "").strip()
        if getattr(result, "return_code", 1) != 0 or not (
            out.isascii() and out.isdigit()
        ):
            return 0
        kept[1][key] = int(out)
        return int(out)

    def _merge_env(
        self,
        env: dict[str, str] | None,
        user: str | int | None = None,
        *,
        plumbing: bool = False,
        explicit_root: bool | None = None,
    ) -> dict[str, str] | None:
        """The persistent env with *env* over it, for one ``exec`` as *user*.

        For an explicit root command the task's persistent PATH is data, not
        the command's PATH: it is sourced after ``_root_exec_command`` pins
        the system PATH, so a folder on it the agent can write (a task's
        ``/app/.venv/bin``) would choose root's ``chown`` (PT-CONF-1). It is
        passed as ``_BENCHFLOW_IMAGE_PATH`` instead, the effective task PATH
        the PATH lock and the verifier's PATH probe read (PT-CONF-3). Only
        the caller's own PATH (the verifier's validated one) replaces root's.

        For a *plumbing* exec (C8) the persistent env loses its loader and
        start-up variables before the caller's *env* goes over it
        (:func:`plumbing_env`): it is sourced as root after the shell pinned
        PATH and unset them, and would put the image's `LD_PRELOAD` back for
        every tool the command runs (PT-C8-2). The caller's own values are
        BenchFlow's and stay, except the names only the route sets
        (:func:`route_owned_name`): root's PATH, loader and Python settings
        are built once by the route and never added back (PT-C8-R3-3,
        PT-C8-R3-7).
        """
        if plumbing:
            env = plumbing_caller_env(env)
        if self._services_are_kept:
            return dict(env) if env else None
        if not self._persistent_env and not env:
            return None
        persistent = (
            plumbing_env(self._persistent_env) if plumbing else self._persistent_env
        )
        merged = {**persistent}
        if env:
            merged.update(env)
        if explicit_root is None:
            explicit_root = is_root_user(user)
        if explicit_root and "PATH" in persistent and not (env and "PATH" in env):
            merged["_BENCHFLOW_IMAGE_PATH"] = merged.pop("PATH")
        return merged or None

    @property
    def sandbox_id(self) -> str | None:
        """Provider-side identifier for this sandbox instance.

        Backends that allocate a remote resource override this to return the
        provider's ID once the sandbox has been created. Today only Daytona
        does (Modal does not expose one); Docker and process backends have no
        provider-side id and keep this ``None``. Used for post-mortem cleanup
        and audit.
        """
        return None

    @abstractmethod
    def _validate_definition(self) -> None: ...

    @classmethod
    @abstractmethod
    def preflight(cls) -> None:
        """Check that required credentials/config are available."""
        ...

    @abstractmethod
    async def start(self, force_build: bool) -> None: ...

    @abstractmethod
    async def stop(self, delete: bool) -> None: ...

    @abstractmethod
    async def upload_file(
        self, source_path: Path | str, target_path: str, *, mode: str | None = None
    ) -> None: ...

    async def _apply_upload_mode(self, target_path: str, mode: str | None) -> None:
        """chmod an uploaded file when the caller requested a private mode.

        Canonical post-upload step for backends whose transfer primitive
        cannot set the mode atomically. Fails loudly: a secret-bearing file
        silently left world-readable is worse than a failed upload.
        """
        mode = validate_upload_mode(mode)
        if mode is None:
            return
        result = await exec_raw(
            self.exec,
            f"chmod {mode} {shlex.quote(str(target_path))}",
            user="root",
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"chmod {mode} {target_path} failed: "
                f"{(diagnostic_result(result).stderr.render() or diagnostic_result(result).stdout.render() or '')[:300]}"
            )

    @abstractmethod
    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        """Upload a directory into a compose service container.

        ``service`` selects which container receives the files. The default
        ``"main"`` is the agent container. Multi-container (vulhub-style)
        tasks pass a target service so the test-script verifier's ``/tests``
        dir lands in the container being inspected (#248). Single-container
        backends reject non-``main`` values.
        """
        ...

    def pin_download_root(self, root: Path | str) -> ReachablePath:
        """Choose and hold an output root before starting any agent execution.

        Required for operator aliases or separate output filesystems outside
        the controller's initial working/temp roots. Pass the returned capability
        (joined with a relative name) to downloads, or retain plain paths below
        the registered root. The sandbox owns the root for its own lifetime.
        """
        from benchflow._utils.reachable_write import pin_destination

        capability = pin_destination(Path(root))
        if not hasattr(self, "_download_roots"):
            self._download_roots: list[ReachablePath] = []
        self._download_roots.append(capability)
        return capability

    @abstractmethod
    async def download_file(
        self, source_path: str, target_path: Path | str
    ) -> None: ...

    @abstractmethod
    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        """Download a directory from a compose service container.

        ``service`` selects which container the files come from. The default
        ``"main"`` is the agent container. Multi-container (vulhub-style)
        tasks pass a target service so target-side verifier output — e.g. a
        ``reward.txt`` written by a ``test.sh`` running in the target — can be
        retrieved (#248). Single-container backends reject non-``main`` values.
        """
        ...

    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        """Run a command inside the sandbox.

        Harness readers call this through ``sandbox._exec_output.exec_raw``.
        Backends must preserve both streams for that task-local request;
        transport redaction belongs only in the separate diagnostic copy.

        ``service`` selects which compose service (container) the command
        runs in. The default ``"main"`` is the agent container. Multi-
        container (vulhub-style) tasks define additional services in the
        task's ``docker-compose.yaml`` and target them via this argument
        — for flag injection into a vulnerable target before the agent
        runs, or target-side verification afterwards (#248). Sandbox
        backends without compose support reject non-``main`` values.
        """
        ...

    async def exec_in_service(
        self,
        service: str,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Run a command in a named compose service — ergonomic wrapper around ``exec``.

        Sugar for ``exec(command, ..., service=service)``. Useful for
        multi-container tasks where verifier code needs to inspect a
        target container's state, e.g.::

            await sandbox.exec_in_service("target", "test -f /tmp/pwned")

        See #248.
        """
        return await exec_raw(
            self.exec,
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            service=service,
        )

    async def services(self) -> list[str]:
        """List the compose service (container) names available in this sandbox.

        Multi-container (vulhub-style) tasks declare extra services in their
        ``docker-compose.yaml`` alongside the agent's ``main`` container; this
        enumerates them so verifier code can target each via
        ``exec(..., service=...)`` (#248).

        Single-container backends (Modal, direct Daytona) have no compose
        topology and raise a clear error — overriding backends must implement
        this themselves.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is a single-container backend; "
            "services() requires a multi-container docker-compose task. "
            "Use the Docker sandbox or the Daytona DinD (compose) sandbox "
            "for vulhub-style tasks (#248)."
        )

    async def is_dir(
        self, path: str, user: str | int | None = None, service: str = "main"
    ) -> bool:
        result = await exec_raw(
            self.exec,
            f"test -d {shlex.quote(path)}",
            timeout_sec=10,
            user=user,
            service=service,
        )
        return result.return_code == 0

    async def is_file(
        self, path: str, user: str | int | None = None, service: str = "main"
    ) -> bool:
        result = await exec_raw(
            self.exec,
            f"test -f {shlex.quote(path)}",
            timeout_sec=10,
            user=user,
            service=service,
        )
        return result.return_code == 0

    async def attach(self) -> None:
        raise NotImplementedError("This environment does not support attaching.")

    async def live_process(self, *, agent: str | None = None) -> LiveProcess:
        """Open the bidirectional line pipe an ACP agent speaks over.

        ``agent`` names the agent being launched; backends that pick a
        transport per agent (Daytona: SSH for gemini, PTY otherwise) use it.

        Each backend knows how to reach into its own sandbox, so the transport
        choice belongs here rather than in a provider-name ``if/elif`` at the
        ACP layer — that chain had to re-derive the backend from a string while
        already holding this object, and sniff private attributes
        (``env._strategy``) to tell Daytona's variants apart.

        Backends with no live-process transport leave this default in place
        and fail with an actionable message instead of being handed another
        backend's transport.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a live agent transport, so "
            "it cannot host an ACP agent. Use a backend that implements "
            "live_process() (docker, daytona, modal, apple-container, agentcore)."
        )

    def configure_agent_timeout(self, timeout_sec: int) -> None:
        """Receive the rollout's effective agent wall-clock budget.

        Most backends do not need this hint. Providers whose remote session has
        its own lifecycle cap can override it so the infrastructure does not
        expire before BenchFlow's agent budget does.
        """
        return None

    # Container-level snapshot/restore (Branch substrate)
    #
    # Part of the Sandbox contract (``docs/architecture.md``): the Branch
    # lifecycle composes container ⊃ environment-state ⊃ agent-session in
    # that order. Backends that cannot snapshot the container (Modal, the
    # Daytona DinD/compose strategy today) leave the defaults in place and
    # raise :class:`SandboxSnapshotNotSupported`. ``Rollout.branch()`` gates
    # on :attr:`supports_snapshot` and fails closed with a clear diagnostic
    # rather than producing a half-consistent checkpoint (#384).

    @property
    def supports_system_shell(self) -> bool:
        """Whether ``exec`` takes ``system_shell=True`` (see :data:`SYSTEM_SHELL`).

        Default ``False``; a backend opts in by overriding this and ``exec``.
        """
        return False

    @property
    def supports_snapshot(self) -> bool:
        """Whether this backend implements container-level snapshot/restore.

        Default ``False`` so new backends are safe-by-default — they must
        opt-in by overriding both this property and ``snapshot``/``restore``.
        """
        return False

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        """Capture the current container state as a re-usable image.

        Default implementation raises :class:`SandboxSnapshotNotSupported`;
        Docker and Daytona-direct override this with provider-native commit
        and snapshot APIs respectively.
        """
        raise SandboxSnapshotNotSupported(
            f"{type(self).__name__} does not support container-level snapshots. "
            "Branch requires a provider whose Sandbox can checkpoint the "
            "container layer; see docs/architecture.md, 'The hard part'."
        )

    async def restore(self, image: SandboxImage) -> None:
        """Restore the container to a previously captured snapshot.

        Default implementation raises :class:`SandboxSnapshotNotSupported`.
        """
        raise SandboxSnapshotNotSupported(
            f"{type(self).__name__} does not support container-level restore. "
            "Branch requires a provider whose Sandbox can checkpoint the "
            "container layer; see docs/architecture.md, 'The hard part'."
        )

    async def delete_snapshot(self, image: SandboxImage) -> bool:
        """Delete a snapshot this sandbox created.

        Returns True when the snapshot is gone (including already gone) and
        False when the provider still uses it; the sandbox then deletes it when
        it stops. Default implementation raises
        :class:`SandboxSnapshotNotSupported`.
        """
        raise SandboxSnapshotNotSupported(
            f"{type(self).__name__} does not support container-level snapshots."
        )


def _guard_base_primitives() -> None:
    # BaseSandbox's own primitives (the refusing defaults) take the same
    # required admission as every subclass's.
    from benchflow.trust.state import TRANSFER_PRIMITIVES, guard_transfer

    for name in TRANSFER_PRIMITIVES:
        impl = BaseSandbox.__dict__.get(name)
        if impl is not None and not getattr(impl, "__isabstractmethod__", False):
            setattr(BaseSandbox, name, guard_transfer(name, impl))


_guard_base_primitives()
