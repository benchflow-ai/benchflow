"""Native ModalSandbox, in RL-first terminology.

Uses Modal sandboxes for cloud-based container execution.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import math
import os
import re
import shlex
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from benchflow._bfx import delivery as bfx_delivery
from benchflow._bfx.locate import SANDBOX_DIR as BFX_DIR
from benchflow._bfx.locate import SANDBOX_PATH as BFX_PATH
from benchflow._bfx.locate import BfxUnavailable
from benchflow._paths import iter_safe_tree
from benchflow._runtime_tree import KIT_BIN
from benchflow._utils.modal_gpu import modal_gpu_request
from benchflow.sandbox import _transport
from benchflow.sandbox._base import (
    IMAGE_ENV_PREFIX,
    STARTUP_ENV_ARGV,
    SYSTEM_PATH,
    SYSTEM_SHELL_EMPTY_VARS,
    SYSTEM_SHELL_UNSET_VARS,
    BaseSandbox,
    ExecResult,
    StartupNamesRefused,
    _root_exec_command,
    bash_argv_without_bash_env,
    bash_startup_name,
    capture_pid1_image_env,
    restore_bash_env,
    restore_image_env,
    startup_env_refusal,
    startup_name,
    with_system_path,
)
from benchflow.sandbox._dir_archive import (
    NO_TAR,
    NOT_ROOTS,
    extract_command,
    source_mode,
    write_archive,
)
from benchflow.sandbox._dockerfile_env import final_stage_env
from benchflow.sandbox._download import private_download
from benchflow.sandbox._exec_output import exec_raw
from benchflow.sandbox._exec_receipt import receipt_exec
from benchflow.sandbox._transport import (
    fit,
    prepare_stage_root,
    reader_cat,
    stage_folder,
    stage_proof,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths, SandboxPaths

if TYPE_CHECKING:
    from benchflow.sandbox.process import LiveProcess

logger = logging.getLogger(__name__)

#: Inert Secret names for the values a Modal command re-exports by name
#:: no loader or shell reads them.
MODAL_CARRY_PREFIX = "_BF_MODAL_ENV_"


def _reexport(name: str) -> str:
    """Builtins: *name* back from its MODAL_CARRY_PREFIX copy, by name."""
    kept = MODAL_CARRY_PREFIX + name
    return f'{name}="${kept}"; export {name}; unset {kept}; '


# Modal refuses a sandbox name over 64 characters or outside [A-Za-z0-9._-].
_MODAL_NAME_MAX = 64
_MODAL_NAME_INVALID = re.compile(r"[^A-Za-z0-9._-]")


def modal_sandbox_name(session_id: str) -> str:
    """``session_id`` as a name Modal accepts for a sandbox.

    Rollout names are ``<task>__<id>`` and a separate verifier's adds
    ``-verifier``, so a long task name passes Modal's 64-character limit and
    the sandbox is refused at create. A shortened or cleaned name ends in a
    hash of the whole one, so two rollouts never share a name.
    """
    name = _MODAL_NAME_INVALID.sub("-", session_id)
    if name == session_id and 0 < len(name) <= _MODAL_NAME_MAX:
        return name
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:10]
    return f"{name[: _MODAL_NAME_MAX - len(digest) - 1]}-{digest}"


#: The most bytes Modal's ``Sandbox.filesystem`` moves in one file (its file
#: tool refuses more: "file is N bytes, which exceeds the 5368709120 byte
#: limit"). A workspace capture or transfer over it fails with that reason.
MODAL_FILE_TRANSFER_LIMIT = 5 * 1024**3


class ModalFileTooLargeError(RuntimeError):
    """A file is over :data:`MODAL_FILE_TRANSFER_LIMIT`: nothing of it was
    copied, and the caller's step (a capture, a transfer) fails with this."""


def modal_exec_timeout(timeout_sec: float | None) -> int | None:
    """``Sandbox.exec``'s ``timeout``: whole seconds, rounded up.

    Modal sends it as an integer field: a float such as a task's
    ``timeout_sec: 600.0`` fails with "'float' object cannot be interpreted
    as an integer".
    """
    if timeout_sec is None:
        return None
    return max(1, math.ceil(timeout_sec))


#: ``0`` (or off, false, no): give Modal the task's CPUs and memory as
#: requests alone, as BenchFlow did before #1245.
HARD_LIMITS_ENV = "BENCHFLOW_MODAL_HARD_LIMITS"
_OFF_VALUES = frozenset({"0", "off", "false", "no"})


def modal_resource(
    value: float | None, environ: Mapping[str, str] | None = None
) -> float | tuple[float, float] | None:
    """``Sandbox.create``'s ``cpu`` or ``memory`` for the task's *value*.

    ``(value, value)``: the request and a hard limit at it, as Docker's
    ``cpus`` and memory limit. Modal reads a bare number as a request alone:
    its CPU limit is then 16 cores above it and memory has none (measured
    with modal 1.6.2: 28 CPUs visible at ``cpu=12``, 1.28 TB of memory).
    The bare *value* with ``BENCHFLOW_MODAL_HARD_LIMITS=0``, or when the task
    gives none.
    """
    env = os.environ if environ is None else environ
    if not value or (env.get(HARD_LIMITS_ENV) or "").strip().lower() in _OFF_VALUES:
        return value
    return (value, value)


#: The thread counts Modal sets in every sandbox, to the CPU request, over
#: the image's own values (measured, modal 1.6.2); ``Sandbox.create``'s
#: ``env`` puts the image's back. Modal also sets ``PYTHONHASHSEED=0``, and
#: ``PYTHONPATH=/pkg/:/root/``, which it appends to a value given at create.
MODAL_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "ORT_INTER_OP_NUM_THREADS",
    "ORT_INTRA_OP_NUM_THREADS",
)


@dataclass(frozen=True)
class ModalImageEnv:
    """What a Modal sandbox is given so its processes see the image's values
    where Modal sets its own (:func:`modal_image_env`)."""

    #: ``Sandbox.create``'s ``env``.
    create: dict[str, str] = field(default_factory=dict)
    #: The ``PYTHONPATH`` every exec starts with; None leaves Modal's.
    pythonpath: str | None = None
    #: The names left as Modal sets them: the image's value is not known.
    kept: tuple[str, ...] = ()


def modal_image_env(
    dockerfile: str | None, task_env: Mapping[str, str] | None = None
) -> ModalImageEnv:
    """The image's values of the names Modal sets, from its Dockerfile.

    *dockerfile* is the task's Dockerfile text (None for a prebuilt image,
    whose values cannot be read without the image); the task's own env
    (*task_env*, which Docker hands the container too) goes over it. A name
    whose value the reader cannot tell, or every name of a Dockerfile it
    cannot follow, keeps Modal's value (``kept``).

    - A thread count the image sets is given at create; one it leaves unset
      stays as Modal sets it (its client cannot unset a variable, and libgomp
      refuses an empty ``OMP_NUM_THREADS``).
    - ``PYTHONHASHSEED`` is given at create, empty when the image leaves it
      unset: Python reads an empty one as unset, so hashing is randomized as
      on Docker.
    - ``PYTHONPATH`` (the image's, or empty) goes with every exec, since
      Modal appends ``/pkg/:/root/`` to a value given at create; and at
      create under its ``IMAGE_ENV_PREFIX`` name, the copy root's task
      commands get back (``restore_image_env``), in place of PID 1's.
    """
    # None: no Dockerfile, or one the reader cannot follow as Docker would.
    image = final_stage_env(dockerfile) if dockerfile is not None else None
    known = image is not None
    found: dict[str, str | None] = dict(image or {})
    found.update(task_env or {})
    create: dict[str, str] = {}
    kept: list[str] = []
    for name in MODAL_THREAD_VARS:
        value = found.get(name)
        if value:
            create[name] = value
        else:
            kept.append(name)
    seed = found.get("PYTHONHASHSEED", "" if known else None)
    if seed is None:
        kept.append("PYTHONHASHSEED")
    else:
        create["PYTHONHASHSEED"] = seed
    pythonpath = found.get("PYTHONPATH", "" if known else None)
    if pythonpath is None:
        kept.append("PYTHONPATH")
    else:
        create[f"{IMAGE_ENV_PREFIX}PYTHONPATH"] = pythonpath
    return ModalImageEnv(create, pythonpath, tuple(kept))


#: Opt-in: the open-file and stack limits every exec runs with. ``1`` (or
#: on, true, yes, docker): Docker's defaults (:data:`DOCKER_RLIMITS`); or
#: ``nofile=SOFT:HARD,stack=SOFT:HARD`` (the stack in bytes; ``unlimited``
#: allowed). Unset or ``0``: Modal's own, as before.
RLIMITS_ENV = "BENCHFLOW_MODAL_RLIMITS"
_ON_VALUES = frozenset({"1", "on", "true", "yes", "docker"})
#: Limits by name, ``(soft, hard)``, None for unlimited.
Rlimits = dict[str, tuple[int | None, int | None]]
#: Docker's defaults when no ``--ulimit`` is given, ``(soft, hard)`` with None
#: for unlimited. A Modal exec starts with nofile 1048576/1048576 and an
#: unlimited stack, so glibc's default thread stack is 2 MiB there and 8 MiB
#: on Docker.
DOCKER_RLIMITS: Rlimits = {
    "nofile": (1024, 524288),
    "stack": (8388608, None),
}
#: The ``pam_limits`` file the limits go in for ``su`` and ``runuser``, which
#: take a limit their config does not name from PID 1 (Debian's and Ubuntu's
#: ``pam_limits``); under gVisor every new exec resets PID 1's to Modal's.
RLIMITS_CONF = "/etc/security/limits.d/benchflow-rlimits.conf"
#: util-linux's ``prlimit`` and ``setpriv`` and coreutils' ``timeout``, by
#: absolute path as root's shell is named.
PRLIMIT = "/usr/bin/prlimit"
TIMEOUT = "/usr/bin/timeout"
SETPRIV = "/usr/bin/setpriv"
#: ``pam_limits``'s unit for each limit, in that limit's bytes (KiB stacks).
_PAM_UNIT = {"nofile": 1, "stack": 1024}


def _rlimit_value(text: str) -> int | None:
    if text == "unlimited":
        return None
    if text.isascii() and text.isdigit():
        return int(text)
    raise ValueError(text)


def modal_rlimits(environ: Mapping[str, str] | None = None) -> Rlimits | None:
    """The limits ``BENCHFLOW_MODAL_RLIMITS`` asks every exec to run with, or
    None (unset or off: Modal's). ValueError for a value it cannot read."""
    env = os.environ if environ is None else environ
    raw = (env.get(RLIMITS_ENV) or "").strip()
    if raw.lower() in _OFF_VALUES or not raw:
        return None
    if raw.lower() in _ON_VALUES:
        return dict(DOCKER_RLIMITS)
    limits: Rlimits = {}
    try:
        for item in raw.split(","):
            name, _, pair = item.strip().partition("=")
            soft_text, _, hard_text = pair.partition(":")
            if name not in DOCKER_RLIMITS or name in limits:
                raise ValueError(name)
            soft, hard = _rlimit_value(soft_text), _rlimit_value(hard_text)
            if soft is None and hard is not None:
                raise ValueError(pair)
            if soft is not None and hard is not None and soft > hard:
                raise ValueError(pair)
            limits[name] = (soft, hard)
    except ValueError:
        raise ValueError(
            f"{RLIMITS_ENV}={raw!r}: give 1 for Docker's limits (nofile "
            "1024:524288, stack 8388608:unlimited), or "
            "nofile=SOFT:HARD,stack=SOFT:HARD (soft at most hard; a number or "
            "unlimited; the stack in bytes)"
        ) from None
    return limits


def _limit(value: int | None) -> str:
    return "unlimited" if value is None else str(value)


def held_argv(argv: Sequence[str], limits: Rlimits) -> list[str]:
    """*argv* run with *limits*, held across Modal's later execs.

    Under gVisor, PID 1 and every top-level exec process share one set of
    limits, which each new exec resets to Modal's; a forked child keeps its
    own. So the first ``prlimit`` sets the limits on this exec, ``timeout``
    forks the payload (and passes on its exit status and the signals it
    gets), ``setpriv --pdeathsig KILL`` ends the payload if Modal's exec
    timeout kills its waiter, and the second ``prlimit`` sets the limits on
    the payload's own copy whatever another exec did in between.
    """
    words = [
        f"--{name}={_limit(soft)}:{_limit(hard)}"
        for name, (soft, hard) in limits.items()
    ]
    return [
        PRLIMIT,
        *words,
        "--",
        TIMEOUT,
        "--foreground",
        "0",
        SETPRIV,
        "--pdeathsig",
        "KILL",
        PRLIMIT,
        *words,
        "--",
        *argv,
    ]


def rlimits_conf(limits: Rlimits) -> str:
    """:data:`RLIMITS_CONF`'s text: *limits* for every user and for root by
    name (Debian's ``pam_limits`` applies no ``*`` line to root)."""
    lines = [f"# BenchFlow ({RLIMITS_ENV}): the limits su and runuser give"]
    for domain in ("*", "root"):
        for name, pair in limits.items():
            for kind, value in zip(("soft", "hard"), pair, strict=True):
                shown = "unlimited" if value is None else str(value // _PAM_UNIT[name])
                lines.append(f"{domain}\t{kind}\t{name}\t{shown}")
    return "\n".join(lines) + "\n"


def _rlimits_conf_command(limits: Rlimits) -> str:
    """Root's POSIX sh: write :data:`RLIMITS_CONF`, exit 3 (printing each
    line) when the image's own ``pam_limits`` config names one of *limits*:
    which line wins for su would then be the image's to say."""
    names = "|".join(limits)
    conf = shlex.quote(RLIMITS_CONF)
    return (
        f"f={conf}; found=; "
        "for c in /etc/security/limits.conf /etc/security/limits.d/*.conf; do "
        '[ -f "$c" ] && [ "$c" != "$f" ] || continue; '
        "while read -r d t i v rest; do "
        "case $d in ''|'#'*) continue;; esac; "
        f'case $i in {names}) echo "$c: $d $t $i $v"; found=1;; esac; '
        'done < "$c"; done; '
        '[ -z "$found" ] || exit 3; '
        "umask 022; mkdir -p /etc/security/limits.d && "
        f'printf %s {shlex.quote(rlimits_conf(limits))} > "$f.tmp" && '
        'mv -f "$f.tmp" "$f"'
    )


#: Root's POSIX sh: the limits ``su`` and ``runuser`` give ``nobody`` and
#: root, each route's block headed ``== <route> rc=<status>``.
_RLIMITS_SU_CHECK = (
    "show='cat /proc/self/limits'; "
    "if id nobody >/dev/null 2>&1; then users='nobody root'; else users=root; fi; "
    "if command -v su >/dev/null 2>&1; then for u in $users; do "
    'out=$(su -s /bin/sh "$u" -c "$show" 2>&1); echo "== su $u rc=$?"; '
    'printf "%s\\n" "$out"; done; fi; '
    "if command -v runuser >/dev/null 2>&1; then for u in $users; do "
    'out=$(runuser -u "$u" -- /bin/sh -c "$show" 2>&1); echo "== runuser $u rc=$?"; '
    'printf "%s\\n" "$out"; done; fi'
)
_LIMIT_ROWS = {"Max open files": "nofile", "Max stack size": "stack"}


def rlimit_check_problems(out: str, limits: Rlimits) -> list[str]:
    """Where :data:`_RLIMITS_SU_CHECK`'s output shows a route that failed or
    gave other limits than *limits*."""
    failed: list[str] = []
    seen: dict[str, Rlimits] = {}
    route = None
    for line in out.splitlines():
        if line.startswith("== "):
            route, _, status = line[3:].rpartition(" rc=")
            seen[route] = {}
            if status != "0":
                failed.append(f"{route} failed (exit status {status})")
            continue
        for head, name in _LIMIT_ROWS.items():
            if route is not None and line.startswith(head):
                words = line[len(head) :].split()[:2]
                try:
                    seen[route][name] = (
                        _rlimit_value(words[0]),
                        _rlimit_value(words[1]),
                    )
                except (IndexError, ValueError):
                    seen[route][name] = (-1, -1)
    differ = [
        f"{route} gives the {name} limit {read.get(name)}, not {pair}"
        for route, read in seen.items()
        if not any(f.startswith(f"{route} failed") for f in failed)
        for name, pair in limits.items()
        if read.get(name) != pair
    ]
    return failed + differ


class ModalRlimitsRefused(RuntimeError):
    """The sandbox cannot hold ``BENCHFLOW_MODAL_RLIMITS``'s limits; it was
    ended."""


#: How long a check for Modal's own end of a sandbox waits for ``poll``.
PLATFORM_END_POLL_SEC = 15.0

#: How long a failure that looks like Modal ending the sandbox waits for
#: ``poll`` to report the end, asking every :data:`PLATFORM_END_SETTLE_STEP_SEC`.
#: While Modal shuts a sandbox down, an exec fails with "Modal Sandbox is
#: shutting down." and ``poll`` still answers None (measured, modal 1.6.2).
PLATFORM_END_SETTLE_SEC = 20.0
PLATFORM_END_SETTLE_STEP_SEC = 1.0
#: The modal errors an exec in an ending or ended sandbox raises.
_ENDING_ERRORS = frozenset(
    {"ConflictError", "NotFoundError", "SandboxTerminatedError", "SandboxTimeoutError"}
)


def modal_ending_error(exc: BaseException) -> bool:
    """Whether *exc* is how Modal fails an exec in a sandbox it is ending or
    ended ("is shutting down", "not found ... has already shut down")."""
    text = str(exc).lower()
    return (
        type(exc).__name__ in _ENDING_ERRORS
        or "shutting down" in text
        or "shut down" in text
    )


#: The start of every error raised for a sandbox Modal ended itself. A trial
#: error that holds "sandbox not found" is an infrastructure error
#: (``benchflow._utils.scoring``).
MODAL_ENDED = "Sandbox not found: Modal ended the sandbox"


def platform_end_reason(returncode: int) -> str:
    """What the ``returncode`` Modal reports for an ended sandbox says ended it.

    Measured with modal 1.6.2: 124 when its lifetime (``Sandbox.create``'s
    ``timeout``) ran out; 137 when it was killed, which is also what a
    process over a hard memory limit does (Modal ends the whole sandbox).
    """
    if returncode == 124:
        return "its lifetime ran out (BENCHFLOW_MODAL_SANDBOX_TIMEOUT)"
    if returncode == 137:
        return (
            "it was killed: a process went over its memory limit, or Modal "
            "evicted or preempted it, or it was terminated from outside BenchFlow"
        )
    return f"its main process exited with {returncode}"


def modal_ended_message(end: Mapping[str, Any], what: str) -> str:
    """The error for *what* failing in a sandbox Modal ended (*end*:
    :attr:`ModalSandbox.platform_end`)."""
    return (
        f"{MODAL_ENDED} before BenchFlow stopped it (returncode "
        f"{end['returncode']}: {end['reason']}); {what}"
    )


class ModalSandboxEndedError(RuntimeError):
    """An exec failed because Modal ended the sandbox before BenchFlow stopped
    it. The message starts with :data:`MODAL_ENDED`."""


#: ``Sandbox.filesystem``'s errors and the builtin ones the legacy file API
#: raised, which BenchFlow's callers catch (FileNotFoundError, ...).
_FILESYSTEM_ERRORS: dict[str, type[OSError]] = {
    "SandboxFilesystemNotFoundError": FileNotFoundError,
    "SandboxFilesystemIsADirectoryError": IsADirectoryError,
    "SandboxFilesystemNotADirectoryError": NotADirectoryError,
    "SandboxFilesystemPermissionError": PermissionError,
}


@contextlib.contextmanager
def _builtin_filesystem_errors(path: str, direction: str) -> Iterator[None]:
    """Raise ``Sandbox.filesystem``'s path errors as the builtin ones, and its
    size refusal as :class:`ModalFileTooLargeError`, naming the limit."""
    try:
        yield
    except Exception as exc:
        if type(exc).__name__ == "SandboxFilesystemFileTooLargeError":
            raise ModalFileTooLargeError(
                f"{path} cannot be copied {direction}: it is over Modal's "
                f"per-file transfer limit of {MODAL_FILE_TRANSFER_LIMIT} bytes "
                f"(5 GiB), and no part of it was copied ({exc})"
            ) from exc
        builtin = _FILESYSTEM_ERRORS.get(type(exc).__name__)
        if builtin is None:
            raise
        raise builtin(str(exc)) from exc


class ModalSandbox(BaseSandbox):
    # a root agent here runs under the provider's own metadata
    # policy (the block's record says provider-managed).
    cloud_metadata_provider_managed = True
    # a Modal sandbox has no route to any cloud metadata endpoint.
    # Measured on 2026-10-26 from a public-network sandbox (gVisor), as root
    # and as uid 1000: every IPv4 endpoint the block covers timed out on
    # ports 80 and 443, every IPv6 one had no route, the metadata host names
    # did not resolve, and 1.1.1.1:443 connected. So no rule is needed (none
    # can go in: root has no NET_ADMIN); each trial proves it again with the
    # same connects and records not_reachable, or stops on any answer.
    cloud_metadata_unreachable = True

    agent_exec_shell = "/bin/bash"

    # exec passes the env as a Modal Secret, never in the command (exec below),
    # so another verifier user gets it privately (b173 PY8-MAIN-40).
    exec_env_is_private = True

    #: How Modal ended this sandbox before BenchFlow stopped it
    #: (``returncode``, ``reason``, ``seen_at``), or None
    #: (:meth:`check_platform_end`).
    platform_end: dict[str, Any] | None = None
    # True once stop() has looked for Modal's own end: an end seen after that
    # is BenchFlow's terminate.
    _stopping = False
    # The image's values of the names Modal sets (_read_image_env), read
    # when the sandbox is created.
    _image_env: ModalImageEnv = ModalImageEnv()
    # The limits every exec runs behind (BENCHFLOW_MODAL_RLIMITS, held by
    # _hold_rlimits); None: Modal's own.
    _rlimits: Rlimits | None = None

    @classmethod
    def preflight(cls) -> None:
        modal_config = Path.home() / ".modal.toml"
        has_env_token = os.environ.get("MODAL_TOKEN_ID") and os.environ.get(
            "MODAL_TOKEN_SECRET"
        )
        if not modal_config.exists() and not has_env_token:
            raise SystemExit(
                "Modal requires authentication. Run 'modal token new' to set up "
                "credentials, or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET "
                "environment variables."
            )

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def _environment_definition_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    def _validate_definition(self) -> None:
        if self.task_env_config.docker_image:
            return
        if not self._environment_definition_path.exists():
            raise FileNotFoundError(
                f"{self._environment_definition_path} not found. Please ensure the "
                "file exists."
            )

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        rollout_paths: RolloutPaths | None,
        task_env_config: SandboxConfig,
        secrets: list[str] | None = None,
        registry_secret: str | None = None,
        volumes: dict[str, str] | None = None,
        sandbox_timeout_secs: int = 60 * 60 * 24,
        sandbox_idle_timeout_secs: int | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            rollout_paths=rollout_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        # Lazy imports — Modal is optional
        self._image = None
        self._app = None
        self._sandbox = None
        self._secrets = secrets or []
        self._registry_secret = registry_secret
        self._volumes = volumes or {}
        self._sandbox_timeout = sandbox_timeout_secs
        self._sandbox_idle_timeout = sandbox_idle_timeout_secs

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _create_sandbox(
        self,
        gpu_config: str | None,
        secrets_config: Sequence[object],
        volumes_config: Mapping[str, object],
    ) -> object:
        from modal import Sandbox

        self._image_env = self._read_image_env()
        sandbox = await Sandbox.create.aio(
            app=self._app,
            image=self._image,
            timeout=self._sandbox_timeout,
            idle_timeout=self._sandbox_idle_timeout,
            name=modal_sandbox_name(self.session_id),
            cpu=modal_resource(self.task_env_config.cpus),
            memory=modal_resource(self.task_env_config.memory_mb),
            gpu=gpu_config,
            block_network=not self.task_env_config.allow_internet,
            secrets=secrets_config,
            volumes=volumes_config,
            env=self._image_env.create or None,
        )
        self.platform_end = None
        self._stopping = False
        return sandbox

    def _read_image_env(self) -> ModalImageEnv:
        """The image's values of the names Modal sets (:func:`modal_image_env`),
        from the task's Dockerfile and its env; logs the names left as Modal
        sets them."""
        dockerfile = None
        if not self.task_env_config.docker_image:
            try:
                dockerfile = self._environment_definition_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError as exc:
                logger.warning("Could not read the task's Dockerfile: %s", exc)
        image = modal_image_env(dockerfile, self._persistent_env)
        if image.kept:
            if dockerfile is None:
                source = (
                    "the image's own values are read from a Dockerfile, and "
                    "this task names a prebuilt image"
                )
            elif final_stage_env(dockerfile) is None:
                source = "BenchFlow cannot read the task's Dockerfile as Docker would"
            else:
                source = (
                    "the task's Dockerfile gives no value of its own BenchFlow "
                    "can read for them"
                )
            logger.warning(
                "Modal sets %s in every sandbox, and %s, so these keep "
                "Modal's values on Modal only",
                ", ".join(image.kept),
                source,
            )
        return image

    def hold_argv(self, argv: Sequence[str]) -> list[str]:
        """*argv* behind :func:`held_argv` once the sandbox holds
        ``BENCHFLOW_MODAL_RLIMITS``'s limits, else unchanged."""
        limits = self._rlimits
        return list(argv) if limits is None else held_argv(argv, limits)

    async def _hold_rlimits(self) -> None:
        """With ``BENCHFLOW_MODAL_RLIMITS``, run every later exec with its
        limits (:func:`held_argv`), and write them for ``su`` and ``runuser``
        (:data:`RLIMITS_CONF`). Refuses, and ends, a sandbox whose image
        cannot hold them: a ``pam_limits`` config of its own naming one of
        the limits, no ``prlimit``, ``setpriv`` or ``timeout``, or a ``su`` or
        ``runuser`` that gives other limits. Nothing without the setting.
        Every step is a root command on the system-shell route."""
        limits = modal_rlimits()
        self._rlimits = None
        if limits is None:
            return
        written = await exec_raw(
            self.exec,
            _rlimits_conf_command(limits),
            user="root",
            system_shell=True,
            timeout_sec=60,
        )
        if written.return_code != 0:
            why = (written.stdout or written.stderr or "").strip()[-500:]
            await self._refuse_rlimits(
                f"the image's own pam_limits config sets one of these limits ({why})"
                if written.return_code == 3
                else f"{RLIMITS_CONF} could not be written ({why})"
            )
        # From here every exec runs behind the held prefix; the first proves
        # the prefix's programs run in the image.
        self._rlimits = limits
        try:
            held = await exec_raw(
                self.exec, "true", user="root", system_shell=True, timeout_sec=60
            )
            code, err = held.return_code, held.stderr or ""
        except Exception as exc:
            code, err = None, repr(exc)
        if code != 0:
            self._rlimits = None
            await self._refuse_rlimits(
                f"{PRLIMIT}, {TIMEOUT} or {SETPRIV} could not run (exit status "
                f"{code}: {err.strip()[-300:]!r}); the image needs util-linux "
                "and coreutils"
            )
        checked = await exec_raw(
            self.exec, _RLIMITS_SU_CHECK, user="root", system_shell=True, timeout_sec=60
        )
        problems = rlimit_check_problems(checked.stdout or "", limits)
        if problems:
            await self._refuse_rlimits("; ".join(problems))

    async def _refuse_rlimits(self, why: str) -> None:
        refusal = (
            f"sandbox startup refused: {RLIMITS_ENV} is set, and this sandbox "
            f"cannot hold its limits: {why}"
        )
        logger.error(refusal)
        try:
            await self._terminate_sandbox()
        except Exception:
            logger.warning("Could not end the refused Modal sandbox")
        else:
            self._sandbox = None
        raise ModalRlimitsRefused(refusal)

    def exec_process_env(self) -> dict[str, str]:
        """What every exec's process env adds: the image's ``PYTHONPATH``
        (or empty), over the ``/pkg/:/root/`` Modal sets in every sandbox."""
        pythonpath = self._image_env.pythonpath
        return {} if pythonpath is None else {"PYTHONPATH": pythonpath}

    async def start(self, force_build: bool) -> None:
        gpu_config = modal_gpu_request(
            self.task_env_config.gpus, self.task_env_config.gpu_types
        )
        from modal import App, Image, Secret, Volume

        from benchflow.trust import state as trust_state

        trust_state.begin_fresh(self)

        docker_image = self.task_env_config.docker_image

        if docker_image:
            registry_secret = (
                Secret.from_name(self._registry_secret)
                if self._registry_secret
                else None
            )
            if ".dkr.ecr." in docker_image:
                self._image = Image.from_aws_ecr(
                    docker_image,
                    secret=registry_secret,
                )
            else:
                self._image = Image.from_registry(
                    docker_image,
                    secret=registry_secret,
                )
        else:
            self._image = Image.from_dockerfile(
                self._environment_definition_path,
                context_dir=self.environment_dir,
            )

        self._app = await App.lookup.aio(
            name="__benchflow__",
            create_if_missing=True,
        )

        secrets_config = [Secret.from_name(secret) for secret in self._secrets]
        volumes_config = {
            mount_path: Volume.from_name(volume_name)
            for mount_path, volume_name in self._volumes.items()
        }

        # bfx at /.bfx: a read-only Volume, set when the sandbox is created
        # (TRUSTED-RUNTIME.md §6). Until it passes a live run it is opt-in
        # (bfx_delivery.provider_enabled); without it the kit path stays.
        self.bfx_delivery = None
        bfx_volume = None
        if BFX_DIR not in volumes_config and bfx_delivery.provider_enabled("modal"):
            import modal

            try:
                bfx_volume = await asyncio.to_thread(bfx_delivery.modal_volume, modal)
            except BfxUnavailable as exc:
                self.logger.debug("bfx: not delivered (%s)", exc)
        if bfx_volume is not None:
            volumes_config[BFX_DIR] = bfx_volume

        self._sandbox = await self._create_sandbox(
            gpu_config=gpu_config,
            secrets_config=secrets_config,
            volumes_config=volumes_config,
        )
        await self._refuse_startup_names()

        await self._make_log_dirs()

        await exec_raw(
            self.exec, f"chmod 777 {SandboxPaths.agent_dir} {SandboxPaths.verifier_dir}"
        )
        await self._hold_rlimits()
        if bfx_volume is not None:
            await self._check_bfx()

    async def _make_log_dirs(self) -> None:
        """The agent's and verifier's log folders, with their parents.

        ``Sandbox.filesystem`` (modal 1.6.2): Modal's servers refuse the
        legacy ``Sandbox.mkdir`` / ``open`` / ``ls`` with ``UnimplementedError``
        ("The legacy Sandbox filesystem API is no longer supported"), so every
        Modal rollout failed right after its sandbox started.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        for path in (SandboxPaths.agent_dir, SandboxPaths.verifier_dir):
            await self._sandbox.filesystem.make_directory.aio(str(path))

    async def _refuse_startup_names(self) -> None:
        """Refuse an image whose ENV sets bash's start-up names
        (``startup_name``): Modal starts every process with the image's
        environment, and the routes' ``/bin/bash`` would import them as root
        before its first command. Read by ``cat`` (argv, no shell) before
        any shell runs; the sandbox is ended on a refusal."""
        _transport.sent("Modal", list(STARTUP_ENV_ARGV))
        process = await self._sandbox.exec.aio(*STARTUP_ENV_ARGV, timeout=60)
        out = await process.stdout.read.aio()
        refusal = startup_env_refusal(out, await process.wait.aio(), "the Modal image")
        if refusal is not None:
            self.logger.error(refusal)
            try:
                await self._terminate_sandbox()
            except Exception:
                self.logger.warning("Could not end the refused Modal sandbox")
            else:
                self._sandbox = None
            raise StartupNamesRefused(refusal)

    async def _check_bfx(self) -> None:
        """The mounted bfx reports the pinned hash (argv exec, no shell,
        no environment); then it is this sandbox's helper."""
        try:
            _transport.sent("Modal", [BFX_PATH, "version"])
            process = await self._sandbox.exec.aio(BFX_PATH, "version", timeout=60)
            out = await process.stdout.read.aio()
            code = await process.wait.aio()
            if code != 0:
                raise BfxUnavailable(f"bfx version exited {code}")
            bfx_delivery.check_version(bfx_delivery.version_answer(out), "x86_64")
        except BfxUnavailable as exc:
            self.logger.warning("bfx: the mounted volume is not usable (%s)", exc)
            return
        self.bfx_delivery = bfx_delivery.VOLUME

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _terminate_sandbox(self) -> None:
        if self._sandbox:
            await self._sandbox.terminate.aio()

    async def check_platform_end(
        self, seen_at: str, *, settle_sec: float = 0.0
    ) -> dict[str, Any] | None:
        """Whether Modal ended the sandbox itself, from its public ``poll``.

        Modal ends a sandbox on its own when its lifetime runs out, when a
        process goes over a hard memory limit, and when it evicts or
        preempts it; the next exec then fails with Modal's "shutting down" or
        "not found". *settle_sec*: how long to keep asking while ``poll``
        still answers None (a sandbox Modal is shutting down). The first end
        seen is kept in :attr:`platform_end` and logged. None while the
        sandbox runs, once :meth:`stop` has begun (BenchFlow's own
        terminate), or when ``poll`` does not answer within
        :data:`PLATFORM_END_POLL_SEC`. Never raises.
        """
        if self.platform_end is not None:
            return self.platform_end
        sandbox = self._sandbox
        if sandbox is None or self._stopping:
            return None
        deadline = time.monotonic() + settle_sec
        while True:
            try:
                code = await asyncio.wait_for(sandbox.poll.aio(), PLATFORM_END_POLL_SEC)
            except Exception as exc:
                logger.debug("Modal did not answer poll: %r", exc)
                return None
            if self._stopping:
                return None
            if isinstance(code, int) and not isinstance(code, bool):
                break
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(PLATFORM_END_SETTLE_STEP_SEC)
        self.platform_end = {
            "returncode": code,
            "reason": platform_end_reason(code),
            "seen_at": seen_at,
        }
        logger.warning(
            "Modal ended sandbox %s before BenchFlow stopped it (returncode %d: %s)",
            getattr(sandbox, "object_id", "?"),
            code,
            self.platform_end["reason"],
        )
        return self.platform_end

    async def stop(self, delete: bool) -> None:
        if not self._sandbox:
            return

        # Before the terminate, after which an ended sandbox is BenchFlow's
        # doing: whether Modal ended it first (platform_end). A sandbox Modal
        # ended needs no terminate.
        ended = await self.check_platform_end("stop")
        self._stopping = True
        try:
            if ended is None:
                await self._terminate_sandbox()
                await self._sandbox.wait.aio(raise_on_termination=False)
        except Exception as e:
            self.logger.warning(f"Error terminating Modal sandbox: {e}")
        finally:
            self._sandbox = None
            self._app = None
            self._image = None

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        # A file over Modal's transfer limit is refused again on a retry.
        retry=retry_if_not_exception_type(ModalFileTooLargeError),
        reraise=True,
    )
    async def upload_file(
        self, source_path: Path | str, target_path: str, *, mode: str | None = None
    ) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        # Streams the file in, as root, making missing parent folders
        # (Sandbox.filesystem: the legacy Sandbox.open is refused, see
        # _make_log_dirs).
        with _builtin_filesystem_errors(target_path, "into the Modal sandbox"):
            await self._sandbox.filesystem.copy_from_local.aio(
                str(source_path), target_path
            )
        await self._apply_upload_mode(target_path, mode)

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        if service != "main":
            raise ValueError(
                f"Modal sandbox is single-container and cannot target "
                f"service {service!r}. Multi-container (vulhub-style) tasks "
                "require the Docker sandbox (#248)."
            )
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory {source_dir} does not exist")

        await exec_raw(
            self.exec, f"mkdir -p {shlex.quote(str(target_dir))}", user="root"
        )
        # One archive, whatever the file count (DAYTONA-GAPS rule 1): one
        # upload into a fresh folder of root's staging folder and one root
        # exec that proves its route root's alone and unpacks it, with the
        # source's permission bits, owned by root. Walk with
        # followlinks=False and skip symlinks (#411): a task or workspace
        # symlink under source_path must not exfiltrate host files into the
        # remote Modal sandbox.
        archive, count = await asyncio.to_thread(
            write_archive,
            source_path,
            mode=source_mode,
            context=f"modal upload_dir {source_path}",
        )
        try:
            if not count:
                return
            staged = f"{stage_folder()}/dir.tar"
            folder = shlex.quote(staged.rsplit("/", 1)[0])
            root = await self.exec_route("root", system_shell=True)
            preparation = prepare_stage_root(
                shell=root.shell, path=root.pin or SYSTEM_PATH
            )
            made = await exec_raw(
                self.exec,
                f"umask 077; {preparation} && mkdir {folder}",
                user="root",
                system_shell=True,
                timeout_sec=60,
            )
            if made.return_code != 0:
                raise RuntimeError("Could not make Modal's directory upload folder")
            try:
                await self.upload_file(archive, staged)
                result = await exec_raw(
                    self.exec,
                    extract_command(
                        staged,
                        str(target_dir),
                        stage_proof(
                            [staged], shell=root.shell, path=root.pin or SYSTEM_PATH
                        ),
                    ),
                    user="root",
                    system_shell=True,
                    timeout_sec=600,
                )
            except Exception:
                # The unpack removes its folder; an upload or exec that
                # raised did not get there (DAYTONA-GAPS review A L4).
                with contextlib.suppress(Exception):
                    await exec_raw(
                        self.exec,
                        f"rm -rf -- {folder}",
                        user="root",
                        system_shell=True,
                        timeout_sec=60,
                    )
                raise
        finally:
            archive.unlink(missing_ok=True)
        if result.return_code == NO_TAR:
            # A sandbox without tar: file by file, inside this primitive (the
            # guard admitted this call).
            for file_path in iter_safe_tree(
                source_path, context=f"modal upload_dir {source_path}"
            ):
                relative_path = file_path.relative_to(source_path).as_posix()
                target_file_path = str(PurePosixPath(target_dir) / relative_path)

                target_file_parent = str(PurePosixPath(target_file_path).parent)
                if target_file_parent != target_dir:
                    await exec_raw(
                        self.exec,
                        f"mkdir -p {shlex.quote(str(target_file_parent))}",
                        user="root",
                    )

                await self.upload_file(
                    file_path,
                    target_file_path,
                    mode=f"{file_path.stat().st_mode & 0o777:o}",
                )
        elif result.return_code != 0:
            raise RuntimeError(
                "Could not unpack the Modal directory upload"
                + (
                    ": its staging folder is not root's alone"
                    if result.return_code == NOT_ROOTS
                    else ""
                )
            )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        # A file over Modal's transfer limit is refused again on a retry.
        retry=retry_if_not_exception_type(ModalFileTooLargeError),
        reraise=True,
    )
    @private_download(directory=False)
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Streams into a temporary file beside the target (private_download's
        # staging folder), then renames it (Sandbox.filesystem: the legacy
        # Sandbox.open is refused, see _make_log_dirs).
        with _builtin_filesystem_errors(source_path, "out of the Modal sandbox"):
            await self._sandbox.filesystem.copy_to_local.aio(source_path, target_path)

    @private_download(directory=True)
    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        if service != "main":
            raise ValueError(
                f"Modal sandbox is single-container and cannot target "
                f"service {service!r}. Multi-container (vulhub-style) tasks "
                "require the Docker sandbox (#248)."
            )
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # NUL-framed names, carried as a receipt: a name may hold a newline.
        result = await receipt_exec(
            self, f"find {shlex.quote(source_dir)} -type f -print0"
        )
        if result.return_code != 0:
            raise RuntimeError(f"Failed to list files in {source_dir}: {result.stderr}")

        file_paths = [p for p in (result.stdout or "").split("\0") if p]
        if not file_paths:
            return

        from benchflow._utils.archive_paths import ArchivePaths

        inventory = ArchivePaths()
        relative_paths = {
            remote: inventory.add(
                str(PurePosixPath(remote).relative_to(PurePosixPath(source_dir))),
                directory=False,
            )
            for remote in file_paths
        }

        sem = asyncio.Semaphore(5)

        async def _download_one(remote_path: str) -> None:
            async with sem:
                rel = relative_paths[remote_path]
                local_path = target_dir / rel
                local_path.parent.mkdir(parents=True, exist_ok=True)
                await self.download_file(remote_path, local_path)

        async with asyncio.TaskGroup() as tg:
            for p in file_paths:
                tg.create_task(_download_one(p))

    # is_dir / is_file: BaseSandbox's ``test -d`` / ``test -f`` through exec,
    # which honours ``user`` and refuses a service other than main; the
    # legacy Sandbox.ls they used is refused (see _make_log_dirs).

    @property
    def supports_system_shell(self) -> bool:
        return True

    @property
    def supports_runtime_kit(self) -> bool:
        return True

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
        system_shell: bool = False,
    ) -> ExecResult:
        """Run ``command`` in the sandbox.

        ``system_shell``: the loader variables the caller does not set are
        passed empty with the exec (Modal hands env over as the process's
        environment), and PATH is the system folders' before anything else
        runs (see :data:`benchflow.sandbox._base.SYSTEM_SHELL`).

        Every command, K1's agent bootstrap and its root staging included,
        is built on PT's route (C8) in :meth:`_dispatch_exec`; an agent
        command then also gets K1's barrier.
        """
        from benchflow.sandbox.agent_exec import AgentExecCommand

        agent_bootstrap = isinstance(command, AgentExecCommand)
        if service != "main":
            raise ValueError(
                f"Modal sandbox is single-container and cannot target "
                f"service {service!r}. Multi-container (vulhub-style) tasks "
                "require the Docker sandbox (#248)."
            )

        requested_user = user
        user = self._resolve_user(user)
        route = await self.exec_route(requested_user, system_shell=system_shell)
        env = self._merge_env(
            env,
            requested_user,
            plumbing=route.pin is not None,
            explicit_root=route.explicit_root,
        )

        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        if agent_bootstrap and env:
            from benchflow.sandbox.native_agent_env import uploaded_agent_env

            root_route = await self.exec_route("root", system_shell=True)

            async def root_exec(body: str) -> ExecResult:
                # Do not merge caller/persistent env into staging or cleanup:
                # root's plumbing route, no env (K1).
                return await self._dispatch_exec(
                    body,
                    route=root_route,
                    user=self._resolve_user("root"),
                    timeout_sec=30,
                    agent_bootstrap=True,
                )

            async with uploaded_agent_env(
                command, env, root_exec=root_exec, upload=self._exec_env_upload()
            ) as (_restored, opening):
                return await self._dispatch_exec(
                    command,
                    route=route,
                    cwd=cwd,
                    user=user,
                    timeout_sec=timeout_sec,
                    agent_bootstrap=True,
                    agent_env_open=opening,
                    agent_restore=True,
                )
        return await self._dispatch_exec(
            command,
            route=route,
            cwd=cwd,
            env=env,
            user=user,
            timeout_sec=timeout_sec,
            agent_bootstrap=agent_bootstrap,
        )

    async def _dispatch_exec(
        self,
        command: str,
        *,
        route: Any,
        agent_bootstrap: bool,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        agent_env_open: str = "",
        agent_restore: bool = False,
    ) -> ExecResult:
        """Start *command* on *route* and wait for it. *agent_restore*:
        *command* is a K1 agent command whose uploaded env the native
        decoder restores (``restore_command``) before it runs."""
        from modal import Secret

        from benchflow.sandbox.agent_exec import native_agent_argv

        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        # The programs this wrapping names by path: the protected kit's for
        # root once the sandbox has one (C8), else the image's system ones.
        su = f"{KIT_BIN}/su" if route.kit else "/bin/su"
        bash = route.bash if route.kit and route.bash.endswith("/bash") else "/bin/bash"
        # C8: Modal hands env over as the process's environment, so the
        # loader and start-up variables (the merged env's and the image's)
        # never go there: the process starts with the loader ones empty, and
        # the values the command should have are exported in its own shell
        # instead. A plumbing exec (route.pin) gets only the caller's values,
        # after its shell pinned PATH. Every other command gets them back
        # just before it runs: the image's BASH_ENV carried past the wrapping
        # bash (bash_argv_without_bash_env), its loader values copied from
        # PID 1 by the root shell that starts it, the merged env's over both.
        # bash's start-up names (startup_name) never reach the process env
        # either: the wrapping /bin/bash would import them as root before
        # its first command, and no shell can export a function's name back.
        startup = {
            k: v
            for k, v in (env or {}).items()
            if k in SYSTEM_SHELL_UNSET_VARS or startup_name(k) or k.startswith("LD_")
        }
        # No value is ever written into the command
        # text, which Modal runs as argv. The values the command's own shells
        # export (the start-up names, PATH, the image PATH) travel in the
        # Secret under inert MODAL_CARRY_PREFIX names, and the text re-exports
        # them by name.
        carried = {
            k: v
            for k, v in (env or {}).items()
            if (k in startup or k in ("PATH", "_BENCHFLOW_IMAGE_PATH"))
            and k.isidentifier()
            and not bash_startup_name(k)
        }
        exports = "".join(_reexport(k) for k in startup if k in carried)
        process_env = {k: v for k, v in (env or {}).items() if k not in startup}
        process_env.update({MODAL_CARRY_PREFIX + k: v for k, v in carried.items()})
        # The names a pinned shell carries across its env -i: the process
        # env before its empty start-up names are added below.
        keep = dict(process_env)
        # The image's PYTHONPATH (or empty) over Modal's /pkg/:/root/, as a
        # docker exec starts with the image's (#1245).
        process_env.update(self.exec_process_env())

        def argv_for(command: str) -> list[str]:
            if agent_restore:
                from benchflow.sandbox.native_agent_env import restore_command

                command = restore_command(command)
            if route.pin is None:
                command = (
                    restore_bash_env(route)
                    + restore_image_env(
                        env or (),
                        tools=route.tools if route.kit and route.root else None,
                    )
                    + exports
                    + command
                )
            else:
                command = exports + command
            if user is not None:
                from benchflow.sandbox._su import su_command

                # K1's agent bootstrap starts POSIX sh with absolute tools (a
                # minimal image may have no bash), the route's su either way.
                command = su_command(
                    command,
                    user,
                    shell="/bin/sh" if agent_bootstrap else bash,
                    su=su,
                    trusted_tools=agent_bootstrap,
                )

            # Modal supplies env directly rather than through the env-file
            # wrapper. Keep the verifier's explicit validated PATH after
            # trusted bootstrap.
            if "PATH" in carried:
                command = _reexport("PATH") + command
            # Likewise the task's PATH a root command carries as data
            # (_merge_env), which the bootstrap's capture would otherwise
            # replace.
            if "_BENCHFLOW_IMAGE_PATH" in carried:
                command = _reexport("_BENCHFLOW_IMAGE_PATH") + command
            if route.pin is not None:
                command = with_system_path(command, route.pin, keep=keep)
                argv = [route.shell, "-c", command]
            else:
                command = capture_pid1_image_env(route.tools) + _root_exec_command(
                    command, route.explicit_root, route.root_path
                )
                argv = bash_argv_without_bash_env(command, bash)
            if agent_bootstrap:
                # K1's barrier around the routed command: no_new_privs before
                # the first shell; the private payload is opened and unlinked
                # inside it, before su; only the selected account's fixed
                # decoder reads its descriptor.
                argv = native_agent_argv(agent_env_open + command)
            return argv

        if route.pin is not None:
            process_env.update({name: "" for name in SYSTEM_SHELL_EMPTY_VARS})
        else:
            # The image's BASH_ENV stays in the process env for the wrapping
            # bash to carry (it reads no file in POSIX mode).
            process_env.update(
                {name: "" for name in SYSTEM_SHELL_EMPTY_VARS if name != "BASH_ENV"}
            )
        # The transport contract (benchflow.sandbox._transport): a command
        # over the bound is the process's stdin and the reader runs it, in
        # the same shell, as the same user. A K1 agent command's payload is
        # the agent's own command, its barrier rebuilt around the reader.
        agent = command if hasattr(command, "agent_target") else None

        def held(command: str) -> list[str]:
            # Behind BENCHFLOW_MODAL_RLIMITS's limits once they are held.
            return self.hold_argv(argv_for(command))

        fitted = fit(held, command, cat=reader_cat(route), agent=agent)
        _transport.sent("Modal", fitted.sent)
        env = process_env
        try:
            process = await self._sandbox.exec.aio(
                *fitted.sent,
                workdir=cwd,
                secrets=[Secret.from_dict(env)] if env else [],
                timeout=modal_exec_timeout(timeout_sec),
            )
            if fitted.payload is not None:
                process.stdin.write(fitted.payload)
                process.stdin.write_eof()
                await process.stdin.drain.aio()

            stdout = await process.stdout.read.aio()
            stderr = await process.stderr.read.aio()

            return_code = await process.wait.aio()
        except Exception as exc:
            # Modal ended the sandbox (its lifetime, a memory kill, an
            # eviction): its error names no cause, so poll says what ended.
            end = await self.check_platform_end(
                "command",
                settle_sec=PLATFORM_END_SETTLE_SEC if modal_ending_error(exc) else 0.0,
            )
            if end is None:
                raise
            raise ModalSandboxEndedError(
                modal_ended_message(end, "an exec failed with it")
            ) from exc

        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
        )

    async def live_process(self, *, agent: str | None = None) -> LiveProcess:
        """The agent's stdio over ``Sandbox.exec`` streams (:class:`ModalProcess`):
        root's launch bootstrap as on Docker, the agent's environment as the
        first stdin line."""
        from benchflow.sandbox.process import ModalProcess

        return ModalProcess.from_sandbox_env(self)

    async def attach(self) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        os.execvp(
            "modal",
            [
                "modal",
                "shell",
                self._sandbox.object_id,
            ],
        )
