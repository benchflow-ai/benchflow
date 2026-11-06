"""Native DaytonaSandbox, in RL-first terminology.

Supports two strategies:
- Direct: single-container sandbox (Dockerfile only)
- DinD: Docker-in-Docker with compose (docker-compose.yaml present)
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import contextvars
import importlib
import json
import logging
import re
import shlex
import time
import types
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

from benchflow.sandbox._download import private_download

try:
    from tenacity import (
        retry,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential,
    )
except ImportError:  # base install without ``sandbox-daytona`` extras (#358)
    # ``tenacity`` ships under the ``sandbox-daytona`` extra. The fallbacks below
    # let this module import cleanly in a base install — ``DaytonaSandbox()`` is
    # what requires the real SDK (and tenacity along with it).

    def retry(*_args: Any, **_kwargs: Any) -> Any:
        def _decorator(fn: Any) -> Any:
            return fn

        return _decorator

    def retry_if_exception(*_args: Any, **_kwargs: Any) -> None:
        return None

    def stop_after_attempt(*_args: Any, **_kwargs: Any) -> Any:
        return None

    def wait_exponential(*_args: Any, **_kwargs: Any) -> Any:
        return None


from benchflow._paths import iter_safe_tree
from benchflow._runtime_tree import KIT_BIN
from benchflow.sandbox import _transport, daytona_start
from benchflow.sandbox._base import (
    ABSOLUTE_HIDE_BASH_ENV,
    BASH_STARTUP_PREFIX,
    BASH_STARTUP_VARS,
    HIDE_BASH_ENV,
    IMAGE_BASH_ENV,
    IMAGE_ENV_PREFIX,
    SYSTEM_PATH,
    SYSTEM_SHELL,
    SYSTEM_SHELL_EMPTY_VARS,
    BaseSandbox,
    ExecResult,
    ExecRoute,
    StartupNamesRefused,
    _root_exec_command,
    restore_bash_env,
    restore_image_env,
    startup_name,
    startup_name_pattern,
    startup_refusal,
    system_shell_prefix,
    system_shell_route,
    with_system_path,
)
from benchflow.sandbox._compose import compose_definition_path
from benchflow.sandbox._dir_archive import (
    NO_TAR,
    NOT_ROOTS,
    daytona_mode,
    extract_command,
    write_archive,
)
from benchflow.sandbox._transport import (
    fit,
    open_staged,
    reader_cat,
    stage_folder,
    stage_proof,
)
from benchflow.sandbox.daytona_delete import delete_sandbox
from benchflow.sandbox.daytona_dind import _DaytonaDinD
from benchflow.sandbox.daytona_layers import (
    InstallLayer,
    InstallLayerPlan,
    plan_install_layers,
)
from benchflow.sandbox.daytona_ledger import LEDGER

# Re-export the extracted command-wrapping helpers so existing imports of
# ``benchflow.sandbox.daytona._wrap_daytona_command_with_env_file`` (and the
# sibling underscore helpers) keep resolving from this module path unchanged.
# The unused names below are intentional façade re-exports, not dead imports.
from benchflow.sandbox.daytona_pty import (
    _DAYTONA_EMPTY_EXIT_CODE_MARKERS,  # noqa: F401
    _DAYTONA_ENV_FILE_PREFIX,  # noqa: F401
    _DAYTONA_TRANSIENT_RETRY_CLASS_NAMES,  # noqa: F401
    _daytona_preflight,
    _exec_failure_output,  # noqa: F401
    _is_daytona_transient_retry_error,
    _reject_non_main_service,  # noqa: F401
    _wrap_daytona_command_with_env_file,
    stamp_transient_transport,
)

# Re-export the extracted ownership labels + auto-reaper so existing imports of
# ``benchflow.sandbox.daytona.reap_stale_sandboxes`` (and the private ownership
# helpers/constants) keep resolving from this module path unchanged.
from benchflow.sandbox.daytona_reaper import (
    _BENCHFLOW_LABEL_NAMESPACE,  # noqa: F401
    _BENCHFLOW_LEASE_LABEL,
    _BENCHFLOW_MANAGED_LABEL,
    _BENCHFLOW_MANAGED_VALUE,  # noqa: F401
    _HEARTBEAT_SEC,
    _LEASE_RENEW_SEC,
    _REAP_DEFAULT_MAX_AGE_MIN,  # noqa: F401
    _REAP_FAILED_MAX_AGE_MIN,  # noqa: F401
    _REAP_FAILED_STATE_MARKERS,  # noqa: F401
    AUTO_STOP_MIN,
    _benchflow_managed_value,
    _benchflow_owned_labels,  # noqa: F401
    _is_benchflow_label_orphan,  # noqa: F401
    _is_benchflow_owned,  # noqa: F401
    lease_labels,
    reap_abandoned_sandboxes,  # noqa: F401
    reap_stale_sandboxes,  # noqa: F401
    reap_stale_snapshots,  # noqa: F401
)
from benchflow.sandbox.daytona_strategies import (
    _DaytonaDirect,
    _DaytonaStrategy,
)
from benchflow.sandbox.metadata import persist_sandbox_info
from benchflow.sandbox.protocol import (
    ExecTimeoutError,
    SandboxImage,
    SandboxStartupError,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths

if TYPE_CHECKING:
    from collections.abc import Collection

    from benchflow.sandbox.process import LiveProcess

# ``SandboxStartupError`` used to live in this module. It now lives in
# ``benchflow.sandbox.protocol`` so a base install without the
# ``sandbox-daytona`` extra can still import ``benchflow.rollout`` (issue #358).
# Re-export here for backward compatibility — existing imports of
# ``benchflow.sandbox.daytona.SandboxStartupError`` keep working.
#
# The reaper (``reap_stale_sandboxes`` + ownership labels), the command-wrapping
# helpers (``_wrap_daytona_command_with_env_file`` …) and the strategy classes
# (``_DaytonaDirect`` / ``_DaytonaDinD``) now live in sibling ``daytona_*``
# modules; they are re-exported above so every name previously importable from
# ``benchflow.sandbox.daytona`` keeps resolving from this path unchanged.
__all__ = ["DaytonaSandbox", "SandboxStartupError"]


def _ensure_daytona_anyio_compat() -> None:
    """Patch the anyio symbol that Daytona 0.176 imports on newer anyio."""
    try:
        import anyio
    except ImportError:
        return

    if hasattr(anyio, "AsyncContextManagerMixin"):
        return

    class _AsyncContextManagerMixin:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            aclose = getattr(self, "aclose", None)
            if aclose is not None:
                await aclose()

    anyio.AsyncContextManagerMixin = _AsyncContextManagerMixin  # type: ignore[attr-defined]


# Module-level handles for the Daytona SDK. The SDK is shipped under the
# ``sandbox-daytona`` extra and is pulled in lazily — see ``_load_daytona_sdk``
# — so importing this module in a base install does not require it (#358).
_DAYTONA_SDK_LOADED = False
AsyncDaytona: Any = None
AsyncSandbox: Any = None
CreateSandboxFromImageParams: Any = None
CreateSandboxFromSnapshotParams: Any = None
DaytonaNotFoundError: Any = None
DaytonaTimeoutError: Any = TimeoutError
FileDownloadRequest: Any = None
FileUpload: Any = None
Image: Any = None
Resources: Any = None
SessionExecuteRequest: Any = None
SnapshotState: Any = None


def _load_daytona_sdk() -> None:
    """Import the optional Daytona SDK on first use.

    The Daytona Python SDK is shipped under the ``sandbox-daytona`` extra; a
    base install of ``benchflow`` must not require it. This helper materializes
    the module-level handles the strategy classes consume, and is idempotent so
    it is cheap to call at the top of each entry-point method.
    """
    global _DAYTONA_SDK_LOADED
    global AsyncDaytona, AsyncSandbox
    global CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams
    global DaytonaNotFoundError, DaytonaTimeoutError, FileDownloadRequest, FileUpload
    global Image, Resources, SessionExecuteRequest, SnapshotState

    if _DAYTONA_SDK_LOADED:
        return

    _ensure_daytona_anyio_compat()
    try:
        _daytona = importlib.import_module("daytona")
        _snapshot = importlib.import_module("daytona._async.snapshot")
    except ImportError as e:
        from benchflow.sandbox.providers import extra_install_hint

        raise ImportError(
            "The Daytona sandbox requires the 'sandbox-daytona' extra. "
            f"Install it with: {extra_install_hint('sandbox-daytona')}"
        ) from e

    AsyncDaytona = _daytona.AsyncDaytona
    AsyncSandbox = _daytona.AsyncSandbox
    CreateSandboxFromImageParams = _daytona.CreateSandboxFromImageParams
    CreateSandboxFromSnapshotParams = _daytona.CreateSandboxFromSnapshotParams
    DaytonaNotFoundError = _daytona.DaytonaNotFoundError
    DaytonaTimeoutError = getattr(_daytona, "DaytonaTimeoutError", TimeoutError)
    FileDownloadRequest = _daytona.FileDownloadRequest
    FileUpload = _daytona.FileUpload
    Image = _daytona.Image
    Resources = _daytona.Resources
    SessionExecuteRequest = _daytona.SessionExecuteRequest
    SnapshotState = _snapshot.SnapshotState
    # The SDK's PTY transport rides socket.io; engineio's write loop logs
    # "packet queue is empty, aborting" at ERROR on a benign disconnect race
    # every time a PTY closes normally (fresh-user dogfood 2026-08-09: two such
    # lines mid-run on a passing rollout). It is not actionable and reads as a
    # failure — silence the logger for benchflow processes.
    logging.getLogger("engineio.client").setLevel(logging.CRITICAL)
    _DAYTONA_SDK_LOADED = True


def build_sync_client(api_key: str | None = None) -> Any:
    """Return a synchronous ``daytona.Daytona`` client with anyio compat applied.

    Canonical entry point for the *sync* SDK client (the async strategy classes
    use :func:`_load_daytona_sdk`). Applies :func:`_ensure_daytona_anyio_compat`
    first — the SDK's sync client imports ``anyio.AsyncContextManagerMixin`` at
    import time, which the pinned anyio may not expose — then builds the client
    with an explicit key (no ``os.environ`` mutation) when one is given,
    otherwise letting the SDK read ``DAYTONA_API_KEY`` itself.
    """
    _ensure_daytona_anyio_compat()
    from daytona import Daytona

    if not api_key:
        return Daytona()
    from daytona import DaytonaConfig

    return Daytona(DaytonaConfig(api_key=api_key))


logger = logging.getLogger("benchflow")

# ``_SandboxParams`` was previously a top-level union of two SDK types. The
# SDK types are now loaded lazily (issue #358), so concrete sandbox params are
# typed as ``Any`` here — callers build them inside methods that have already
# called ``_load_daytona_sdk()``.
_SandboxParams = Any
# Daytona tells whether a session command has finished only when asked, so
# ``_poll_response`` asks until it has. The first wait is 0.05 s and each later
# one doubles, up to 1.0 s. A trial makes dozens of execs one after another
# (the user, credentials, skills, lockdown, hardening) and most finish in well
# under a second, so a fixed 1.0 s wait added up to a second to each. A
# command still running at the first status request now makes up to four more
# of them (a 0.76 to 1.0 s command makes 6 against the old 2); after 1.55 s the
# polls settle at one a second, the old rate.
_DAYTONA_COMMAND_POLL_FIRST_SEC = 0.05
_DAYTONA_COMMAND_POLL_INTERVAL_SEC = 1.0  # the longest wait between two polls
# An exec whose budget is at most this many seconds (and more than 0) is sent
# with ``run_async=False``: Daytona answers the execute request when the
# command has finished, with its exit code, stdout and stderr (SDK 0.184.0,
# ``AsyncProcess.execute_session_command``), so it needs no status or logs
# request. Longer or unbounded execs keep the asynchronous execute and the
# poll above: one held request would hold an HTTP connection for their whole
# run. 0 turns the synchronous path off (every exec as before U12; RM keeps
# it the same in both U5b arms). Disabled: a held HTTP
# request cannot distinguish command runtime from delivery latency, nor keep
# the poll's deadline and transport-error contracts. Keep the asynchronous
# path until those contracts can be preserved without replaying a command.
_DAYTONA_SYNC_EXEC_MAX_SEC = 0
# One request of a command poll (status, then logs). A request on a connection
# that died silently (an Azure SNAT drop, a Daytona proxy restart) otherwise
# waits forever: the SDK sets no read timeout, and the retry below only
# retries a call that raises, so a verifier whose command had finished waited
# out its whole budget (#1136). A timed-out request raises TimeoutError, which
# _SDK_RETRY retries on a new connection; both requests are idempotent reads.
_DAYTONA_POLL_STATUS_TIMEOUT_SEC = 30.0
_DAYTONA_POLL_LOGS_TIMEOUT_SEC = 120.0
_STARTUP_HARD_TIMEOUT_BUFFER_SEC = 120
# The SDK's error once a new sandbox reaches an error state (``BUILD_FAILED``,
# ``ERROR``). It neither returns nor deletes that sandbox, so this id is the
# only handle left for cleaning it up.
_FAILED_START_SANDBOX_ID = re.compile(
    r"\bSandbox (\S+) (?:failed to start with|entered error) state\b"
)


_CREATE_ATTEMPT_LABEL = "benchflow.create-attempt"
# After a create timeout, how long the create keeps looking for the attempt's
# sandbox before it gives up with a terminal error, and how often: a found
# one is deleted and the create retried (DAYTONA-START round 3). Whatever is
# not found goes to the process ledger (daytona_ledger), which keeps it until
# its removal is confirmed or the job exits.
_CREATE_SETTLE_SEC = 90.0
_CREATE_WATCH_INTERVAL_SEC = 10.0

# The descriptor a session command's own staged text is read on when its
# stdin is already a caller's payload (the DinD VM's ``docker compose exec``).
_SECOND_PAYLOAD_FD = 9


def _on_stdin(payload: bytes | None) -> dict[int, bytes] | None:
    return None if payload is None else {0: payload}


# Files per SDK ``upload_files`` call where a sandbox has no tar to unpack
# one archive: the SDK opens each batch's files at once.
_UPLOAD_BATCH = 100

# Daytona's toolbox runs every exec (session commands, ``process.exec``, the
# PTY) through a shell it starts with the sandbox's own environment, the
# image's ENV included, and that shell is bash whenever the image lists bash
# (and not zsh) in /etc/shells. A non-interactive bash runs the file BASH_ENV
# names before the first command it reads, as the toolbox's user (root), so
# an image whose BASH_ENV names a file the agent can write ran the agent's
# file as root before every hardening exec and test.sh. Nothing
# sent in the command can stop that; only the sandbox's create-time env can.
# The probe reads the toolbox's BASH_ENV with POSIXLY_CORRECT set, which puts
# bash in POSIX mode, where it reads no BASH_ENV file.
# Probe only a trusted image, before any agent runs. A checkpoint instead
# carries the controller's pre-agent receipt and starts with these names
# empty at the provider. Unsetting them inside an exec is already too late
# for the toolbox's outer shell (FF-sw-daytona-C7).
# The names are the C8 rule's one list (``startup_name``): the probe prints
# every entry of the toolbox shell's own start environment that the list
# matches (bash's exported functions and option sets included, by prefix),
# read from /proc/$$/environ with ``read -d ''``, which bash, busybox ash and
# zsh have; where ``read`` takes no ``-d`` (dash), the named ones from the
# shell's variables (dash passes no ``BASH_FUNC_*%%`` entry on, so no shell
# it starts imports one). Then the functions bash imported, by ``compgen``.
# Builtins only (no value is proven empty yet, so no program may start):
# each record as a tag, its byte length, a colon and its bytes, under the C
# locale so the shell counts bytes, and ``Z`` last. A NUL separator never
# survives a session command's output, and no framing may depend on the
# endpoint keeping it.
_TOOLBOX_STARTUP_PROBE = (
    "LC_ALL=C; (read -r -d '' bf_e </dev/null) 2>/dev/null; case $? in "
    "1) while IFS= read -r -d '' bf_e; do case ${bf_e%%=*} in "
    f"{startup_name_pattern()}) "
    'printf \'E%s:%s\' "${#bf_e}" "$bf_e";; esac; done </proc/$$/environ || exit 1;; '
    "*) "
    + "".join(
        f'case ${{{name}+x}} in x) bf_e="{name}=${{{name}}}"; '
        'printf \'E%s:%s\' "${#bf_e}" "$bf_e";; esac; '
        for name in (*SYSTEM_SHELL_EMPTY_VARS, *BASH_STARTUP_VARS)
    )
    + ";; esac; "
    "if command -v compgen >/dev/null 2>&1; then "
    "for bf_f in $(compgen -A function); do "
    'printf \'F%s:%s\' "${#bf_f}" "$bf_f"; done; fi; printf Z'
)


# What bash prints, before any command, for an imported function it refuses
# in POSIX mode: one named like a special builtin (bash then exits) and one
# whose name is no identifier.
_BASH_IMPORT_ERROR = re.compile(
    r"`([^'`\n]{1,200})': is a special builtin"
    r"|error importing function definition for `([^'`\n]{1,200})'"
)


def _parse_startup_probe(output: str) -> tuple[dict[str, str], list[str]] | None:
    """What :data:`_TOOLBOX_STARTUP_PROBE` printed: the value of each of
    :data:`SYSTEM_SHELL_EMPTY_VARS` ("" when unset) and the bash start-up
    names set to a value (an imported function as its ``BASH_FUNC_<name>%%``
    entry); None when the output is not exactly that (a record cut, added
    to or changed on the way, a name twice)."""
    data = output.encode("utf-8", "surrogateescape")
    entries: dict[str, str] = {}
    refused: set[str] = set()
    at = 0
    while data[at : at + 1] in (b"E", b"F"):
        tag = data[at : at + 1]
        colon = data.find(b":", at + 1)
        digits = data[at + 1 : colon] if colon >= 0 else b""
        if not digits.isdigit() or len(digits) > 9:
            return None
        start, at = colon + 1, colon + 1 + int(digits)
        if at > len(data):
            return None
        try:
            record = data[start:at].decode("utf-8")
        except UnicodeDecodeError:
            return None
        if tag == b"F":
            refused.add(f"{BASH_STARTUP_PREFIX}{record}%%")
            continue
        name, separator, value = record.partition("=")
        if not separator or not startup_name(name) or name in entries:
            return None
        entries[name] = value
        if value and name not in SYSTEM_SHELL_EMPTY_VARS:
            refused.add(name)
    # The provider may end the output with one newline of its own.
    if data[at:] not in (b"Z", b"Z\n"):
        return None
    values = {name: entries.get(name, "") for name in SYSTEM_SHELL_EMPTY_VARS}
    return values, sorted(refused)


_TOOLBOX_STARTUP_META = "benchflow_toolbox_startup_v1"
_TOOLBOX_PROBE_ATTEMPTS = 3


def _checkpoint_startup_env(image: SandboxImage) -> dict[str, str]:
    try:
        values = json.loads(image.meta[_TOOLBOX_STARTUP_META])
        if (
            not isinstance(values, dict)
            or set(values) != set(SYSTEM_SHELL_EMPTY_VARS)
            or any(
                not isinstance(value, str) or "\0" in value for value in values.values()
            )
        ):
            raise ValueError("invalid startup receipt")
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "Daytona startup state is missing or unsupported for this checkpoint; "
            "create a new checkpoint with this BenchFlow version before restoring"
        ) from exc
    return values


def _tag_create_attempt(
    params: Any, lease_extra_sec: float = 0.0
) -> dict[str, str] | None:
    """Give one create call a unique label so its sandbox can be found when
    the create fails without returning it, and this run's id and lease (the
    lease covers the create's own timeout too); ``None`` when params take
    none."""
    labels = getattr(params, "labels", None)
    if not isinstance(labels, dict):
        return None
    tag = {_CREATE_ATTEMPT_LABEL: uuid.uuid4().hex}
    labels.update(tag)
    labels.update(lease_labels(lease_extra_sec))
    return tag


def _create_answered(exc: BaseException) -> bool:
    """Whether the API answered the create with a 4xx: it refused the request,
    so an empty lookup by the attempt's label proves nothing landed. Any
    other failure (a timeout, a 5xx, a dropped connection) leaves it open."""
    status = getattr(exc, "status_code", None)
    return (
        isinstance(status, int) and not isinstance(status, bool) and 400 <= status < 500
    )


def _labels_query(labels: dict[str, str]) -> Any:
    try:
        from daytona import ListSandboxesQuery
    except ImportError:  # a faked client (tests without the Daytona extra)
        return types.SimpleNamespace(labels=labels)
    return ListSandboxesQuery(labels=labels)


def _failed_start_sandbox_id(exc: BaseException) -> str | None:
    match = _FAILED_START_SANDBOX_ID.search(str(exc))
    return match.group(1) if match else None


def _sync_exec_eligible(timeout_sec: object) -> bool:
    """Whether an exec with this budget goes synchronous (0 < budget <=
    :data:`_DAYTONA_SYNC_EXEC_MAX_SEC`, which 0 turns off)."""
    limit = _DAYTONA_SYNC_EXEC_MAX_SEC
    return (
        limit > 0
        and isinstance(timeout_sec, int | float)
        and not isinstance(timeout_sec, bool)
        and 0 < timeout_sec <= limit
    )


def _sync_response_complete(response: object) -> bool:
    """Whether a synchronous execute's answer holds the whole result: an exit
    code, and stdout and stderr as text. A toolbox that fills only the
    combined ``output`` is not complete (its streams are read from the
    command's logs instead)."""
    exit_code = getattr(response, "exit_code", None)
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        return False
    stdout = getattr(response, "stdout", None)
    stderr = getattr(response, "stderr", None)
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        return False
    output = getattr(response, "output", None)
    return bool(stdout or stderr) or not (isinstance(output, str) and output)


def _command_poll_waits() -> Iterator[float]:
    """The waits between one command's status requests: 0.05 s, doubling,
    then 1.0 s each (``_DAYTONA_COMMAND_POLL_FIRST_SEC`` up to
    ``_DAYTONA_COMMAND_POLL_INTERVAL_SEC``, read when the poll starts)."""
    wait = _DAYTONA_COMMAND_POLL_FIRST_SEC
    cap = _DAYTONA_COMMAND_POLL_INTERVAL_SEC
    while True:
        yield min(wait, cap)
        wait *= 2


# Safety-net ceiling for ``_poll_response`` when the caller passes no
# ``timeout_sec``. A Daytona *session* command only reports its ``exit_code``
# once it completes, and Daytona treats the command as still-running while any
# child holds the session's stdout/stderr stream open. So a backgrounded daemon
# launched without redirecting its std fds (``mysvc &`` with no
# ``</dev/null >log 2>&1``) keeps that stream open, ``exit_code`` never arrives,
# and an unbounded poll loop would wedge ``exec`` forever (BF-6). This cap is
# deliberately sized *well above* any legitimately long-running command (an
# agent rollout can run for many minutes) so it never trips for real work — it
# exists purely so a never-completing session command cannot spin the poll loop
# indefinitely. It applies ONLY when no positive ``timeout_sec`` is supplied
# (``None``, or a non-positive value — both previously meant "no deadline");
# an explicit positive ``timeout_sec`` is honored byte-for-byte as before.
_DAYTONA_EXEC_HARD_CAP_SEC = 3600
_DAYTONA_TRANSIENT_RETRY: Any = retry_if_exception(_is_daytona_transient_retry_error)

# Retry-attempt budgets for transient Daytona failures, named here so the
# repeated ``3`` cannot silently drift between the idempotent-SDK policy and
# sandbox creation (#532). ``_stop_sandbox`` deliberately uses a smaller budget.
_DAYTONA_RETRY_ATTEMPTS = 3
_DAYTONA_STOP_RETRY_ATTEMPTS = 2
# A delete that Daytona's API gateway failed (HTTP 502, 503 or 504) gets more
# attempts. The gateway answers at once, so four attempts cost about 7 s of
# backoff, while a timed-out delete keeps the short budget above (each attempt
# can take the SDK's 60 s delete timeout). A sandbox whose only delete got
# "502 Bad Gateway" was otherwise leaked.
_DAYTONA_STOP_GATEWAY_RETRY_ATTEMPTS = 4
_DAYTONA_GATEWAY_STATUS_CODES = frozenset({502, 503, 504})
# A status or logs request of a running command that Daytona failed with a
# 5xx gets this many attempts, about 90 s of backoff (1, 2, 4, 8, 16, 30, 30
# s): one 502 behind a proxy restart crashed verifiers deep into test.sh
# ("verifier crashed: Failed to get session command"). Both are
# idempotent reads; a transport error keeps _DAYTONA_RETRY_ATTEMPTS.
_DAYTONA_POLL_SERVER_RETRY_ATTEMPTS = 8
_GATEWAY_ERROR_TEXT = re.compile(
    r"\b50[234] (?:bad gateway|service (?:temporarily )?unavailable"
    r"|gateway time-?out)\b",
    re.IGNORECASE,
)
_HTML_TITLE = re.compile(r"<title>\s*(.*?)\s*</title>", re.IGNORECASE | re.DOTALL)


def _daytona_status(exc: BaseException) -> int | None:
    """The HTTP status of a Daytona SDK error, when it carries one."""
    if not type(exc).__module__.startswith("daytona."):
        return None
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _is_daytona_gateway_error(exc: BaseException) -> bool:
    """Daytona's API gateway failed the request (HTTP 502, 503 or 504)."""
    if not type(exc).__module__.startswith("daytona."):
        return False
    status = _daytona_status(exc)
    if status is not None:
        return status in _DAYTONA_GATEWAY_STATUS_CODES
    return bool(_GATEWAY_ERROR_TEXT.search(str(exc)))


def _is_daytona_server_error(exc: BaseException) -> bool:
    """Daytona answered 5xx: its API, proxy or toolbox failed the request.

    SDK 0.196 raises a plain ``DaytonaError`` with ``status_code``; 0.207 a
    class per status (``DaytonaInternalServerError``, ``DaytonaBadGatewayError``,
    ``DaytonaServiceUnavailableError``, and ``DaytonaTimeoutError`` for 504),
    which keeps ``status_code`` too. Without a status, the gateway page's text.
    """
    status = _daytona_status(exc)
    if status is not None:
        return 500 <= status <= 599
    return _is_daytona_gateway_error(exc)


def _is_daytona_poll_retry_error(exc: BaseException) -> bool:
    """Whether to ask a running command's status or logs again: a transport
    blip, a timeout, a rate limit or a 5xx. A 4xx fails at once."""
    return _is_daytona_transient_retry_error(exc) or _is_daytona_server_error(exc)


def _poll_retries_exhausted(retry_state: Any) -> bool:
    """Tenacity stop rule for the command polls: a 5xx gets more tries."""
    outcome = retry_state.outcome
    error = outcome.exception() if outcome is not None else None
    budget = (
        _DAYTONA_POLL_SERVER_RETRY_ATTEMPTS
        if error is not None and _is_daytona_server_error(error)
        else _DAYTONA_RETRY_ATTEMPTS
    )
    return retry_state.attempt_number >= budget


def _is_daytona_delete_retry_error(exc: BaseException) -> bool:
    """Whether to repeat a sandbox delete: transient and gateway errors.

    Only a delete is repeated on a gateway error. Repeating it is safe (a
    sandbox that is already gone counts as deleted), unlike a create, which
    could leave a second sandbox behind.
    """
    return _is_daytona_transient_retry_error(exc) or _is_daytona_gateway_error(exc)


def _stop_retries_exhausted(retry_state: Any) -> bool:
    """Tenacity stop rule for ``_stop_sandbox``: a gateway error gets more tries."""
    outcome = retry_state.outcome
    error = outcome.exception() if outcome is not None else None
    budget = (
        _DAYTONA_STOP_GATEWAY_RETRY_ATTEMPTS
        if error is not None and _is_daytona_gateway_error(error)
        else _DAYTONA_STOP_RETRY_ATTEMPTS
    )
    return retry_state.attempt_number >= budget


def _one_line_error(exc: BaseException) -> str:
    """An error for a log line: an HTML error page shrinks to its title."""
    text = str(exc)
    start = text.lower().find("<html")
    if start >= 0:
        title = _HTML_TITLE.search(text, start)
        text = text[:start] + (title.group(1) if title else "(HTML error page)")
    text = " ".join(text.split())
    return text if len(text) <= 300 else text[:297] + "..."


# Shared tenacity policy for the idempotent Daytona SDK calls — session-command
# polling and filesystem up/download. Three attempts with exponential backoff,
# re-raising the final failure. ``_create_sandbox`` and ``_stop_sandbox`` keep
# their own policies (different backoff bounds, and ``_stop_sandbox`` a smaller
# attempt budget), so they are intentionally not folded in here. Reusing one
# ``retry(...)`` decorator across methods is safe: tenacity builds a fresh
# controller per decorated function.
_SDK_RETRY = retry(
    stop=stop_after_attempt(_DAYTONA_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=_DAYTONA_TRANSIENT_RETRY,
    reraise=True,
)
# The command polls (status, then logs): _SDK_RETRY's rule plus a 5xx, which
# gets _DAYTONA_POLL_SERVER_RETRY_ATTEMPTS (_poll_retries_exhausted).
_SDK_POLL_RETRY = retry(
    stop=_poll_retries_exhausted,
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception(_is_daytona_poll_retry_error),
    reraise=True,
)


def _leave_sigint_to_benchflow() -> None:
    """Keep Ctrl-C with BenchFlow's handler.

    From SDK 0.197 every client opens a Socket.IO event connection, and
    python-engineio's asyncio client then replaces the running loop's SIGINT
    handler with one that cancels every task: a Ctrl-C printed no
    "Interrupted" line, did not exit 130, and cancelled the cleanups BenchFlow
    shields. engineio installs its handler only while this flag is unset.
    """
    try:
        from engineio import async_client
    except Exception:  # an SDK without the event connection
        return
    async_client.async_signal_handler_set = True


class DaytonaClientManager:
    """Singleton manager for the AsyncDaytona client."""

    _instance: DaytonaClientManager | None = None
    _lock = asyncio.Lock()

    def __init__(self) -> None:
        self._client: AsyncDaytona | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._client_lock = asyncio.Lock()
        self._logger = logger.getChild("DaytonaClientManager")
        self._cleanup_registered = False

    @classmethod
    async def get_instance(cls) -> DaytonaClientManager:
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        assert cls._instance is not None
        return cls._instance

    async def get_client(self) -> AsyncDaytona:
        # The SDK loads lazily; a helper script can ask for a client before any
        # DaytonaSandbox exists.
        _load_daytona_sdk()
        loop = asyncio.get_running_loop()
        if self._client is not None and self._client_loop is not loop:
            # The client (and a lock it may have bound) belongs to another,
            # usually closed, event loop, e.g. an earlier asyncio.run(); it can
            # be neither used nor closed from this loop. Start over.
            self._logger.debug("Event loop changed; creating a new AsyncDaytona client")
            self._client = None
            self._client_lock = asyncio.Lock()
        async with self._client_lock:
            if self._client is None:
                self._logger.debug("Creating new AsyncDaytona client")
                _leave_sigint_to_benchflow()
                self._client = AsyncDaytona()
                self._client_loop = loop
                if not self._cleanup_registered:
                    atexit.register(self._cleanup_sync)
                    self._cleanup_registered = True
            return self._client

    def _cleanup_sync(self) -> None:
        # Runs as an atexit callback after a (typically successful) run.
        # ``CancelledError`` is a BaseException, so a bare ``except Exception``
        # let it escape into a 40-line "Exception ignored in atexit callback"
        # traceback printed AFTER the final score line (fresh-user dogfood
        # 2026-08-09) — closing the SDK client tears down its aiohttp/websocket
        # readers, which surface cancellation on loop shutdown. Nothing here is
        # actionable at exit: log at debug, never print.
        try:
            asyncio.run(self._cleanup())
        except (Exception, asyncio.CancelledError) as e:
            self._logger.debug(f"Daytona client cleanup swallowed at exit: {e!r}")

    async def _cleanup(self) -> None:
        async with self._client_lock:
            if self._client is not None:
                try:
                    self._logger.debug("Closing AsyncDaytona client at program exit")
                    await self._client.close()
                except (Exception, asyncio.CancelledError) as e:
                    self._logger.debug(f"Error closing AsyncDaytona client: {e!r}")
                finally:
                    self._client = None


# Written in the sandbox's folder (the trial folder; a separate verifier
# sandbox writes its own under verifier-sandbox/) when the sandbox stops: what
# its execs cost in Daytona requests and time (see
# DaytonaSandbox._write_exec_counts).
EXEC_COUNTS_FILE = "daytona-execs.json"

# True while a transient exec (one whose session is deleted afterward, see
# DaytonaSandbox.exec_transient) is polled, so its status requests are counted
# apart. A context variable, not an attribute: the live token counter's reads
# run as their own task alongside the trial's other execs.
_IN_TRANSIENT_EXEC: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "benchflow_daytona_in_transient_exec", default=False
)


@dataclass
class _ExecCounts:
    """Execs split in two. The first three fields count the ordinary execs
    (setup, the agent's, the verifier's). The transient_ ones count execs
    whose session is deleted afterward: the live token counter's log reads,
    which repeat about once a second for the whole agent run, and short probes
    (health checks, the credential stager's root shell). How many of those
    run depends on how long the agent runs and on how fast each poll returns,
    so they are kept out of the first three."""

    exec_calls: int = 0
    # Status requests of the polls, retries included.
    status_requests: int = 0
    # Wall time of each exec, from creating the session to the result, summed;
    # execs that run at the same time are each counted in full.
    exec_seconds: float = 0.0
    transient_exec_calls: int = 0
    transient_status_requests: int = 0
    transient_exec_seconds: float = 0.0
    # Execs (ordinary and transient) sent with run_async=False (PY8 U12); they
    # are also in the counts above. One that came back without its result
    # polls like any other, and its status requests are counted above.
    sync_exec_calls: int = 0


class DaytonaSandbox(BaseSandbox):
    # a root agent here runs under the provider's own metadata
    # policy (the block's record says provider-managed).
    cloud_metadata_provider_managed = True
    # A missing saved provider handle cannot prove that stop released the
    # allocation. The rollout checks this capability without importing us.
    requires_release_handle = True

    # True once a create request did not finish (it timed out, or was cancelled
    # before it returned): Daytona may still create a sandbox this process has
    # no handle for. A class default so stop() can read it on any instance.
    _create_outcome_unknown = False
    # harden_before_verify runs its preparation steps as one script in one
    # exec (lockdown._run_prep_script): every exec here is a polled round trip.
    verifier_prep_script: ClassVar[bool] = True

    @classmethod
    def preflight(cls) -> None:
        _daytona_preflight()

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        rollout_paths: RolloutPaths | None,
        task_env_config: SandboxConfig,
        snapshot_template_name: str | None = None,
        network_block_all: bool | None = None,
        auto_stop_interval_mins: int = AUTO_STOP_MIN,
        auto_delete_interval_mins: int = 0,
        **kwargs: object,
    ) -> None:
        # Materialize the optional Daytona SDK on first DaytonaSandbox
        # instantiation. Importing this module is now free of the SDK
        # dependency (issue #358); the SDK is required only at construction
        # time of a real Daytona sandbox.
        _load_daytona_sdk()
        # Detect compose mode before super().__init__ calls _validate_definition
        self._compose_mode = compose_definition_path(environment_dir) is not None
        self._kwargs = kwargs

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            rollout_paths=rollout_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        self._auto_stop_interval = auto_stop_interval_mins
        self._auto_delete_interval = auto_delete_interval_mins
        self._snapshot_template_name = snapshot_template_name
        # Set by start_from_snapshot(): start() creates the sandbox from this
        # branch snapshot, with no fallback to the task image.
        self._start_snapshot_ref: str | None = None
        self._start_snapshot_env: dict[str, str] | None = None
        self._toolbox_startup_receipt: dict[str, str] | None = None
        if network_block_all is not None:
            self._network_block_all = network_block_all
            expected = not task_env_config.allow_internet
            if network_block_all != expected:
                self.logger.warning(
                    f"network_block_all={network_block_all} overrides task config "
                    f"allow_internet={task_env_config.allow_internet}"
                )
        else:
            self._network_block_all = not task_env_config.allow_internet

        self._sandbox: AsyncSandbox | None = None  # pyright: ignore[reportInvalidTypeForm]
        self._client_manager: DaytonaClientManager | None = None
        # Calls of _create_sandbox in the current start(), retries included.
        self._create_attempts = 0
        # The proxy and agent installs the rollout plans (daytona_layers),
        # the keys of the layers the sandbox's image carries, and the record
        # sandbox.json keeps of them.
        self._install_layers: InstallLayerPlan | None = None
        self._install_layer_keys: dict[str, str] = {}
        self._install_layers_record: dict[str, Any] | None = None
        self._layer_build_allowance_sec = 0
        # Set while a recipe with cached install layers is created: its
        # BUILD_FAILED goes to the plain-image fallback, not a fresh create.
        self._layer_build_falls_back = False

        self._strategy: _DaytonaStrategy = (
            _DaytonaDinD(self) if self._compose_mode else _DaytonaDirect(self)
        )
        self.logger.debug(f"Selected strategy: {self._strategy.__class__.__name__}")

    @property
    def _uses_compose(self) -> bool:
        return self._compose_mode

    @property
    def sandbox_id(self) -> str | None:
        return self._sandbox.id if self._sandbox else None

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def _environment_docker_compose_path(self) -> Path:
        return compose_definition_path(self.environment_dir) or (
            self.environment_dir / "docker-compose.yaml"
        )

    def _validate_definition(self) -> None:
        if self._compose_mode:
            path = self._environment_docker_compose_path
        else:
            if self.task_env_config.docker_image:
                # Prebuilt-image task: start() uses Image.base(docker_image)
                # and never reads a Dockerfile (matches modal/agentcore/apple).
                return
            path = self._dockerfile_path
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Please ensure the file exists.")

    # Shared helpers used by both strategies

    def _require_sandbox(self) -> AsyncSandbox:  # pyright: ignore[reportInvalidTypeForm]
        """Return the started sandbox, raising if ``start()`` has not run.

        Centralizes the ``if not self._sandbox`` precondition repeated across the
        SDK helpers (and the direct strategy's ``is_dir``/``is_file``) so the type
        checker narrows ``AsyncSandbox | None`` to ``AsyncSandbox`` at each call
        site. Uses the same falsiness check and message as the inlined guards.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")
        return self._sandbox

    def _on_sandbox_created(self) -> None:
        """Persist sandbox.json the instant the Daytona sandbox id is known.

        Called from ``_create_sandbox`` the moment ``self._sandbox`` is assigned
        — before ``start()`` does its remaining (and, for DinD, long) work:
        launching dockerd, polling ``_wait_for_docker_daemon`` for tens of
        seconds, mkdir/chmod, compose build/up. If the run is interrupted
        (CancelledError/SIGINT/timeout) anywhere in that stretch, the Daytona
        sandbox already exists server-side; persisting here means there is still
        a ``sandbox.json`` to audit and clean up (#554/#563).

        Best-effort and self-contained: a missing ``rollout_paths`` (e.g. a
        snapshot/branch sandbox built outside a rollout dir) is a no-op, and the
        underlying write swallows-and-logs its own failures. The rollout-layer
        ``on_started`` callback still runs after ``start()`` returns as an
        idempotent fallback.
        """
        if self.rollout_paths is None:
            return
        persist_sandbox_info(self, self.rollout_paths.rollout_dir)

    # The start path's rule (daytona_start): a rejected key fails at once; a
    # 429 or quota refusal is waited out for up to 10 minutes, a 5xx or
    # transport error gets 6 attempts, a create that timed out 3; every wait
    # is jittered and a 429's Retry-After is honoured. A 5xx or 429 used to
    # lose the trial after at most 3 un-jittered attempts (DAYTONA-START).
    @retry(
        stop=daytona_start.create_retries_exhausted,
        wait=daytona_start.create_wait,
        retry=daytona_start.create_should_retry,
        reraise=True,
    )
    async def _create_sandbox(
        self,
        params: _SandboxParams,
    ) -> None:
        if self._create_outcome_unknown:
            raise self._startup_failure(TimeoutError("previous create is unreconciled"))
        if not self._client_manager:
            raise RuntimeError(
                "Client manager not initialized. This should never happen."
            )
        self._create_attempts += 1

        # Clean up any previous failed sandbox before retry
        if self._sandbox is not None:
            try:
                self.logger.warning("Cleaning up previous sandbox before retry")
                await delete_sandbox(self._sandbox)
            except BaseException as cleanup_err:
                self._log_sandbox_left_behind(cleanup_err, when="before a retry")
                if not isinstance(cleanup_err, Exception):
                    raise
                raise self._startup_failure(cleanup_err) from cleanup_err
            self._sandbox = None

        daytona = await self._client_manager.get_client()
        # A recipe with cached install layers also builds them (daytona_layers).
        build_timeout = round(self.task_env_config.build_timeout_sec) + getattr(
            self, "_layer_build_allowance_sec", 0
        )
        hard_timeout = build_timeout + _STARTUP_HARD_TIMEOUT_BUFFER_SEC
        attempt_labels = _tag_create_attempt(params, hard_timeout)

        create_task = asyncio.ensure_future(
            daytona.create(
                params=params,
                timeout=build_timeout,
            )
        )
        try:
            self._sandbox = await asyncio.wait_for(
                asyncio.shield(create_task), timeout=hard_timeout
            )
            # Persist the id the moment the sandbox exists — before start()'s
            # remaining (DinD: long) work — so an interrupt mid-start() still
            # leaves a sandbox.json to audit/clean up (#554/#563).
            self._on_sandbox_created()
            self._lease_create_labels = dict(getattr(params, "labels", None) or {})
            self._ensure_lease_renewal()
        except (TimeoutError, DaytonaTimeoutError) as exc:
            if _daytona_status(exc) is not None:
                # SDK 0.207 raises an HTTP 504 or 408 as DaytonaTimeoutError:
                # Daytona answered, so it is handled as every HTTP failure (a
                # 504 retried as a gateway error), not as a timed-out create.
                await self._delete_sandbox_that_failed_to_start(
                    daytona, exc, attempt_labels=attempt_labels
                )
                raise
            self.logger.error(
                f"Sandbox creation timed out after {hard_timeout}s "
                f"(build_timeout={build_timeout}s + buffer={_STARTUP_HARD_TIMEOUT_BUFFER_SEC}s)"
            )
            create_task.cancel()
            self._create_outcome_unknown = True
            ownership = None
            if attempt_labels:
                ownership = self._watch_create_attempt(
                    daytona, attempt_labels, reason="create timed out", unknown=True
                )
                ownership.busy = True
            # Every create can leave a remote resource after a timeout,
            # including snapshots and recipes without install layers. Reap
            # the SDK task and require positive deletion before any retry.
            try:
                done, _ = await asyncio.wait({create_task}, timeout=30)
                if done:

                    async def reconcile() -> bool:
                        try:
                            late_sandbox = create_task.result()
                        except (asyncio.CancelledError, Exception):
                            pass
                        else:
                            await delete_sandbox(late_sandbox)
                            return True
                        return bool(
                            attempt_labels
                        ) and await self._delete_sandboxes_labelled(
                            daytona, attempt_labels, require_match=True
                        )

                    try:
                        reconciled = await asyncio.wait_for(reconcile(), timeout=30)
                    except Exception:
                        reconciled = False
                    if reconciled:
                        if attempt_labels:
                            LEDGER.close_attempt(attempt_labels)
                        else:
                            self._create_outcome_unknown = False
                if self._create_outcome_unknown and attempt_labels:
                    # Keep looking in place: a sandbox that lands and is deleted
                    # within the settle window lets the create retry.
                    give_up = time.monotonic() + _CREATE_SETTLE_SEC
                    while time.monotonic() < give_up:
                        await asyncio.sleep(
                            max(
                                0.0,
                                min(
                                    _CREATE_WATCH_INTERVAL_SEC,
                                    give_up - time.monotonic(),
                                ),
                            )
                        )
                        if await self._delete_sandboxes_labelled(
                            daytona, attempt_labels, require_match=True
                        ):
                            LEDGER.close_attempt(attempt_labels)
                            break
                if self._create_outcome_unknown:
                    # Not found: the ledger keeps looking for it by its label
                    # until it is found and deleted, and past the job's end.
                    # A terminal error also prevents the SDK retry decorator and
                    # snapshot-to-image fallback from issuing another create.
                    raise self._startup_failure(exc) from exc
                raise
            finally:
                if ownership is not None:
                    ownership.busy = False
        except asyncio.CancelledError:
            try:
                self._sandbox = await asyncio.wait_for(create_task, timeout=30)
                # Sandbox came back even though we were cancelled — persist its
                # id before re-raising so it is not orphaned (#554/#563).
                self._on_sandbox_created()
            except (TimeoutError, asyncio.CancelledError, Exception):
                create_task.cancel()
                self._create_outcome_unknown = True
                if attempt_labels:
                    self._watch_create_attempt(
                        daytona,
                        attempt_labels,
                        reason="create cancelled",
                        unknown=True,
                    )
            raise
        except Exception as exc:
            await self._delete_sandbox_that_failed_to_start(
                daytona, exc, attempt_labels=attempt_labels
            )
            status = daytona_start.key_rejected_status(exc)
            if status is not None:
                # A configuration error: every retry, here or the trial's
                # (scoring._UNRECOVERABLE_STARTUP_RES), would fail the same way.
                raise SandboxStartupError(
                    f"Daytona rejected DAYTONA_API_KEY (HTTP {status}): "
                    f"{_one_line_error(exc)}. Check the key at "
                    "https://app.daytona.io/dashboard/keys",
                    sandbox_state="error",
                    attempts=self._create_attempts,
                    build_timeout_sec=self.task_env_config.build_timeout_sec,
                ) from exc
            raise

    async def _toolbox_startup_env(self) -> dict[str, str]:
        _transport.sent("Daytona", _TOOLBOX_STARTUP_PROBE)
        # The probe only reads the environment, so a transient failure (a 5xx,
        # a dropped connection) is asked again instead of losing the trial.
        attempt = 1
        while True:
            try:
                response = await self._require_sandbox().process.exec(
                    _TOOLBOX_STARTUP_PROBE, env={"POSIXLY_CORRECT": "1"}, timeout=60
                )
                break
            except Exception as exc:
                wait = daytona_start.wait_sec(exc, attempt)
                if (
                    attempt >= _TOOLBOX_PROBE_ATTEMPTS
                    or daytona_start.error_kind(exc) not in daytona_start.RETRIED_KINDS
                    or wait > daytona_start.READ_MAX_WAIT_SEC
                ):
                    raise
                await asyncio.sleep(wait)
                attempt += 1
        output = str(getattr(response, "result", "") or "")
        parsed = _parse_startup_probe(output)
        if getattr(response, "exit_code", 1) != 0 or parsed is None:
            # bash's POSIX mode stops at an imported function named like a
            # special builtin, and reports one whose name is no identifier,
            # before the probe runs: name those, never their values.
            refused = sorted(
                f"{BASH_STARTUP_PREFIX}{a or b}%%"
                for a, b in _BASH_IMPORT_ERROR.findall(output)
            )
            if refused:
                raise StartupNamesRefused(
                    startup_refusal(refused, "the Daytona image or sandbox env")
                )
            # Never echo activation values or provider diagnostics here.
            raise RuntimeError("Could not verify Daytona toolbox startup environment")
        values, refused = parsed
        if refused:
            raise StartupNamesRefused(
                startup_refusal(refused, "the Daytona image or sandbox env")
            )
        return values

    async def _create_sandbox_without_toolbox_bash_env(
        self, params: _SandboxParams, *, startup_env: dict[str, str] | None = None
    ) -> None:
        """Admit trusted images before agent execution; restore only from a receipt.

        Every loader and start-up name is set empty at the provider, and
        each value the image sets is carried aside as data under its
        IMAGE_ENV_PREFIX name (DAYTONA-GAPS rule 2): the task route restores
        exactly those values (``restore_image_env``; BASH_ENV also through
        ``RESTORE_BASH_ENV``), root's routes never do. So no root or system
        shell the toolbox starts reads the image's start-up files or loads
        its libraries (C8, S3-CONF-ROOT-1): the toolbox's own shell starts
        with every name empty, and a name under another name is read by no
        shell or loader. An image or sandbox env that sets one of bash's
        start-up names (an exported function, SHELLOPTS, BASHOPTS:
        ``startup_name``) is refused with a ``benchflow:`` line and its
        sandbox deleted; every probe, the second and a
        checkpoint's included, refuses one.
        A checkpoint may contain agent bytes: override its provider startup
        environment before the first shell, never probe its original values.
        """

        def hidden(values: dict[str, str]) -> _SandboxParams:
            return params.model_copy(
                update={
                    "env_vars": {
                        **(params.env_vars or {}),
                        **dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""),
                        # BASH_ENV's copy always (RESTORE_BASH_ENV reads it),
                        # every other name's when the image sets it.
                        IMAGE_BASH_ENV: values["BASH_ENV"],
                        **{
                            f"{IMAGE_ENV_PREFIX}{name}": value
                            for name, value in values.items()
                            if value and name != "BASH_ENV"
                        },
                    }
                }
            )

        self._toolbox_startup_receipt = None
        try:
            if startup_env is None:
                await self._create_sandbox(params=params)
                values = await self._toolbox_startup_env()
                if any(values.values()):
                    await delete_sandbox(self._require_sandbox())
                    self._sandbox = None
                    await self._create_sandbox(params=hidden(values))
                    if any((await self._toolbox_startup_env()).values()):
                        raise RuntimeError(
                            "Daytona did not clear toolbox startup inputs"
                        )
            else:
                values = startup_env
                await self._create_sandbox(params=hidden(values))
                if any((await self._toolbox_startup_env()).values()):
                    raise RuntimeError("Daytona did not clear toolbox startup inputs")
            self._toolbox_startup_receipt = dict(values)
        except BaseException as exc:
            # A refusal must not leave a usable handle or an untracked probe
            # sandbox. Keep the handle if deletion fails so stop() can retry.
            sandbox = self._sandbox
            if sandbox is not None:
                try:
                    await delete_sandbox(sandbox)
                except Exception:
                    self.logger.warning(
                        "Could not delete refused Daytona startup sandbox"
                    )
                    self._create_outcome_unknown = True
                    # Stop content-snapshot and layer-build fallbacks from
                    # replacing the only handle to this refused sandbox.
                    raise self._startup_failure(
                        RuntimeError("Could not delete refused Daytona startup sandbox")
                    ) from None
                else:
                    self._sandbox = None
            if isinstance(exc, StartupNamesRefused):
                # The image's own setting: no content-snapshot or layer-build
                # fallback would start it either.
                self.logger.error(str(exc))
                raise self._startup_failure(exc) from exc
            raise

    async def _delete_sandbox_that_failed_to_start(
        self,
        daytona: Any,
        exc: Exception,
        *,
        attempt_labels: dict[str, str] | None = None,
    ) -> None:
        """Confirm removal before any retry or image fallback can allocate.

        Record ownership before awaiting the provider: a create can fail without
        returning its sandbox, and cancellation during deletion must not orphan it.
        A failed confirmation preserves the create error as the startup cause.
        """
        sandbox_id = _failed_start_sandbox_id(exc)
        if sandbox_id is not None:
            self._create_outcome_unknown = True
            entry = LEDGER.open_sandbox(
                daytona,
                sandbox_id,
                "failed to start; awaiting confirmed deletion",
                managed=_benchflow_managed_value(),
                holder=self,
                on_resolved=self._create_cleanup_resolved,
            )
            entry.busy = True
            try:
                entry.handle = await daytona.get(sandbox_id)
                await delete_sandbox(entry.handle)
            except BaseException as cleanup_err:
                if _daytona_status(cleanup_err) == 404 or (
                    type(cleanup_err).__name__ == "DaytonaNotFoundError"
                ):
                    LEDGER.close_sandbox(sandbox_id)
                    return
                self.logger.warning(
                    f"Could not delete sandbox {sandbox_id} that failed to start: "
                    f"{cleanup_err}"
                )
                LEDGER.ensure_sweeper()
                if not isinstance(cleanup_err, Exception):
                    raise
                raise self._startup_failure(exc) from exc
            else:
                LEDGER.close_sandbox(sandbox_id)
                self.logger.info(f"Deleted sandbox {sandbox_id} that failed to start")
            finally:
                entry.busy = False
            return
        if not attempt_labels:
            return
        # A refused request (4xx) cannot still land. All other failures need a
        # positive lookup/deletion; an empty list is not confirmation.
        self._create_outcome_unknown = True
        self._watch_create_attempt(
            daytona,
            attempt_labels,
            reason=f"create failed: {_one_line_error(exc)}",
            unknown=True,
        )
        entry = LEDGER.entries[f"attempt:{attempt_labels['benchflow.create-attempt']}"]
        entry.busy = True
        try:
            cleaned = await self._delete_sandboxes_labelled(
                daytona, attempt_labels, require_match=not _create_answered(exc)
            )
            if cleaned:
                LEDGER.close_attempt(attempt_labels)
            else:
                raise self._startup_failure(exc) from exc
        finally:
            entry.busy = False

    def _create_cleanup_resolved(self) -> None:
        if not any(e.on_resolved is not None for e in LEDGER.held_by(self)):
            self._create_outcome_unknown = False

    def _watch_create_attempt(
        self,
        daytona: Any,
        labels: dict[str, str],
        *,
        reason: str = "create failed",
        unknown: bool = False,
    ) -> Any:
        """Put an attempt whose sandbox may exist in the process ledger
        (daytona_ledger): it is looked for by its label until it is found and
        deleted, while the job lives and at its exit, never dropped by time.
        *unknown*: this attempt set ``_create_outcome_unknown``; its confirmed
        deletion clears it."""

        def resolved() -> None:
            if unknown:
                self._create_cleanup_resolved()

        entry = LEDGER.open_attempt(
            daytona,
            labels,
            reason,
            managed=_benchflow_managed_value(),
            holder=self,
            on_resolved=resolved if unknown else None,
        )
        LEDGER.ensure_sweeper()
        return entry

    async def _sweep_create_attempts(self) -> None:
        """One look at each of this env's open ledger entries."""
        await LEDGER.sweep(LEDGER.held_by(self))

    async def _drain_create_attempts(self) -> None:
        """stop(): sweep this env's entries until each is resolved or past its
        settle time; any still open stays in the ledger for the job's sweeper
        and its exit sweep."""
        await LEDGER.drain(self)

    def _ensure_lease_renewal(self) -> None:
        task = getattr(self, "_lease_renewal", None)
        if task is None or task.done():
            self._lease_renewal = asyncio.ensure_future(self._renew_lease())

    def _stop_lease_renewal(self) -> None:
        task = getattr(self, "_lease_renewal", None)
        if task is not None and not task.done():
            task.cancel()
        self._lease_renewal = None

    async def _renew_lease(self) -> None:
        """While this env holds a sandbox: send Daytona an activity heartbeat
        every ``_HEARTBEAT_SEC`` and renew the run lease every
        ``_LEASE_RENEW_SEC``. The heartbeat is what keeps a living
        controller's sandbox from Daytona's inactivity stop
        (``AUTO_STOP_MIN``); a paused, cut-off or dead controller stops
        sending it, and Daytona stops the sandbox on its own clock
        (DAYTONA-START round 3, rule B). ``set_labels`` is not activity."""
        last_renewal = time.monotonic()
        while True:
            await asyncio.sleep(_HEARTBEAT_SEC)
            sandbox = self._sandbox
            if sandbox is None:
                continue
            try:
                await asyncio.wait_for(sandbox.refresh_activity(), timeout=30)
            except Exception as exc:
                self.logger.debug(
                    f"Daytona activity heartbeat of {getattr(sandbox, 'id', '?')} "
                    f"failed: {_one_line_error(exc)}"
                )
            if time.monotonic() - last_renewal < _LEASE_RENEW_SEC:
                continue
            last_renewal = time.monotonic()
            # set_labels replaces every label: renew from the labels the
            # create sent and those Daytona holds, never dropping ownership.
            held = getattr(sandbox, "labels", None)
            labels = {
                **getattr(self, "_lease_create_labels", {}),
                **(held if isinstance(held, dict) else {}),
            }
            if _BENCHFLOW_MANAGED_LABEL not in labels:
                continue
            renewed = {**labels, **lease_labels()}
            try:
                await asyncio.wait_for(sandbox.set_labels(renewed), timeout=60)
            except Exception as exc:
                self.logger.warning(
                    f"Could not renew the run lease of Daytona sandbox "
                    f"{getattr(sandbox, 'id', '?')}: {_one_line_error(exc)}; "
                    f"it lapses at {labels.get(_BENCHFLOW_LEASE_LABEL)} unless "
                    "a later renewal goes through"
                )

    async def _delete_sandboxes_labelled(
        self, daytona: Any, labels: dict[str, str], *, require_match: bool = False
    ) -> bool:
        try:
            leftovers = [s async for s in daytona.list(_labels_query(labels))]
        except Exception as list_err:
            self.logger.warning(
                f"Could not look up a sandbox the failed create left: {list_err}"
            )
            return False
        cleaned = True
        for leftover in leftovers:
            try:
                await delete_sandbox(leftover)
            except Exception as cleanup_err:
                cleaned = False
                self.logger.warning(
                    f"Could not delete sandbox {leftover.id} that failed to "
                    f"start: {cleanup_err}"
                )
            else:
                self.logger.info(f"Deleted sandbox {leftover.id} that failed to start")
        # After a timeout, an empty list does not prove the server finished
        # creating the sandbox. Only a positive handle/deletion reconciles it.
        return cleaned and (bool(leftovers) or not require_match)

    def _startup_failure(self, exc: BaseException) -> SandboxStartupError:
        """The startup error for a failed create, with the attempts it made.

        Only transient SDK errors are retried, so a build failure is one
        attempt; the id comes from the SDK error when create returned nothing.
        """
        sandbox_id = getattr(self._sandbox, "id", None) if self._sandbox else None
        attempts = self._create_attempts
        return SandboxStartupError(
            f"Sandbox creation failed after {attempts} "
            f"attempt{'' if attempts == 1 else 's'}: {exc}",
            sandbox_id=sandbox_id or _failed_start_sandbox_id(exc),
            sandbox_state="error",
            attempts=attempts,
            build_timeout_sec=self.task_env_config.build_timeout_sec,
        )

    def _log_no_sandbox_to_stop(self) -> None:
        """stop() found no sandbox handle; warn only if one may still exist.

        Normally there is nothing to delete: the create failed before a
        sandbox existed (a failed build's leftover is deleted at once), or an
        earlier stop() already deleted it. Only a create that never finished
        can leave a sandbox on Daytona that this process cannot delete.
        """
        if self._create_outcome_unknown:
            self.logger.warning(
                "No Daytona sandbox handle to delete, but a create request did "
                "not finish, so a sandbox may still exist on Daytona. It is "
                "looked for by its create-attempt label until it is found and "
                "deleted, up to the job's exit; one still not found then is "
                "recorded for `bench sandbox cleanup`."
            )
        else:
            self.logger.debug(
                "No Daytona sandbox to delete: none was created, or it is "
                "already deleted."
            )

    def _log_sandbox_left_behind(
        self, exc: BaseException, *, when: str = "at stop"
    ) -> None:
        """The delete failed or was cancelled: the ledger retries it while
        the job lives and at its exit, and records it for `bench sandbox
        cleanup`. A cancelled delete may still go through; its 404 then
        closes the entry."""
        error = _one_line_error(exc) or type(exc).__name__
        self.logger.error(
            f"Could not delete Daytona sandbox {self.sandbox_id} {when}: "
            f"{error}. It is retried until the job exits; one still left then "
            "is recorded for `bench sandbox cleanup`."
        )
        sandbox_id = self.sandbox_id
        if sandbox_id:
            LEDGER.open_sandbox(
                None,
                sandbox_id,
                f"delete {when} failed: {error}",
                managed=_benchflow_managed_value(),
                handle=self._sandbox,
                holder=self,
            )
            LEDGER.ensure_sweeper()

    @retry(
        stop=_stop_retries_exhausted,
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_daytona_delete_retry_error),
        reraise=True,
    )
    async def _stop_sandbox(self) -> None:
        if not self._sandbox:
            return
        try:
            await delete_sandbox(self._sandbox)
        except Exception as exc:
            if _daytona_status(exc) != 404 and type(exc).__name__ != (
                "DaytonaNotFoundError"
            ):
                raise
            # Already gone: an earlier attempt's delete went through behind a
            # failed gateway response, or Daytona deleted it on its own.
            self.logger.debug(
                f"Daytona sandbox {self._sandbox.id} is already deleted: "
                f"{_one_line_error(exc)}"
            )
        # An earlier failed delete of it may be open in the ledger.
        LEDGER.close_sandbox(self._sandbox.id)

    @stamp_transient_transport
    @_SDK_POLL_RETRY
    async def _get_session_command_with_retry(
        self, session_id: str, command_id: str
    ) -> object:
        sandbox = self._require_sandbox()
        counts = self._exec_counts()
        if _IN_TRANSIENT_EXEC.get():
            counts.transient_status_requests += 1
        else:
            counts.status_requests += 1
        return await asyncio.wait_for(
            sandbox.process.get_session_command(session_id, command_id),
            timeout=_DAYTONA_POLL_STATUS_TIMEOUT_SEC,
        )

    @stamp_transient_transport
    @_SDK_POLL_RETRY
    async def _get_session_command_logs_with_retry(
        self, session_id: str, command_id: str
    ) -> object:
        sandbox = self._require_sandbox()
        return await asyncio.wait_for(
            sandbox.process.get_session_command_logs(session_id, command_id),
            timeout=_DAYTONA_POLL_LOGS_TIMEOUT_SEC,
        )

    @staticmethod
    def _poll_timeout_error(
        timeout_sec: int | float | None, capped: bool
    ) -> RuntimeError:
        """Build the ``RuntimeError`` raised when a poll deadline is exceeded.

        The bounded path (explicit positive ``timeout_sec``) keeps the exact
        legacy message so its semantics are byte-for-byte unchanged. The
        safety-net path (no positive ``timeout_sec``, falling back to
        :data:`_DAYTONA_EXEC_HARD_CAP_SEC`) raises the *same* ``RuntimeError``
        type for consistent error handling, but with a message that points at
        the most likely cause — a backgrounded command still holding the Daytona
        session's stdout/stderr stream open.
        """
        if capped:
            return RuntimeError(
                f"Command timed out after {_DAYTONA_EXEC_HARD_CAP_SEC} seconds "
                "(no positive timeout_sec given; hit the Daytona exec "
                "safety-net cap). "
                "A backgrounded command is likely holding the session "
                "stdout/stderr stream open — redirect its std fds, e.g. "
                "`nohup CMD </dev/null >log 2>&1 &`."
            )
        return ExecTimeoutError(timeout_sec)

    async def _poll_response(
        self,
        session_id: str,
        command_id: str,
        timeout_sec: int | float | None = None,
    ) -> ExecResult:
        self._require_sandbox()

        loop = asyncio.get_running_loop()
        # When the caller passes an explicit, positive ``timeout_sec`` we honor
        # it exactly as before. When it is ``None`` (or non-positive) we fall
        # back to a generous safety-net ceiling so ``deadline`` is *never*
        # ``None`` — a Daytona session command whose ``exit_code`` never resolves
        # (e.g. a backgrounded child still holding the session stdout/stderr
        # stream open) can therefore never spin this loop forever (BF-6). The
        # cap is sized well above any real command, so normal behavior on the
        # unbounded path is unchanged for everything except the wedge case.
        if timeout_sec is not None and timeout_sec > 0:
            deadline = loop.time() + float(timeout_sec)
            capped = False
        else:
            deadline = loop.time() + float(_DAYTONA_EXEC_HARD_CAP_SEC)
            capped = True

        response = await self._get_session_command_with_retry(session_id, command_id)

        waits = _command_poll_waits()
        while response.exit_code is None:  # type: ignore[union-attr]
            remaining = deadline - loop.time()
            if remaining > 0:
                await asyncio.sleep(min(next(waits), remaining))
            # The timeout is raised only on a "still running" answer to a
            # request made at or after the deadline. The pending answer in
            # hand was asked for before the last wait (and a request's
            # latency before that), so a command that finished within its
            # budget, exit 124 included, was reported as timed out without
            # anyone asking again.
            asked_at = loop.time()
            response = await self._get_session_command_with_retry(
                session_id,
                response.id,  # type: ignore[union-attr]
            )
            if response.exit_code is None and asked_at >= deadline:  # type: ignore[union-attr]
                raise self._poll_timeout_error(timeout_sec, capped)

        logs = await self._get_session_command_logs_with_retry(session_id, command_id)

        return ExecResult(
            stdout=logs.stdout,  # type: ignore[union-attr]
            stderr=logs.stderr,  # type: ignore[union-attr]
            return_code=int(response.exit_code),  # type: ignore[union-attr]
        )

    def _default_root_path(
        self,
        command: str,
        user: str | int | None,
        env: dict[str, str] | None,
        route: ExecRoute,
    ) -> str:
        """A task command as a root default user gets the task's PATH back."""
        if (
            route.pin is None
            and user is None
            and route.root
            and getattr(self, "default_user", None) is not None
            and not (env or {}).get("PATH")
        ):
            return 'export PATH="${_BENCHFLOW_IMAGE_PATH-$PATH}"; ' + command
        return command

    @stamp_transient_transport
    async def _sandbox_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        shell: str = "/bin/bash -c",
        user: str | int | None = None,
        cleanup_session: bool = False,
        *,
        route: ExecRoute,
        stdin_payload: bytes | None = None,
    ) -> ExecResult:
        """Run ``command`` as a Daytona session command and poll to completion.

        ``timeout_sec=None`` falls back to the
        :data:`_DAYTONA_EXEC_HARD_CAP_SEC` safety-net deadline in
        :meth:`_poll_response`, so a backgrounded command that never releases the
        session stdout/stderr stream cannot wedge ``exec`` forever (BF-6).
        Backgrounded daemons must redirect all std fds
        (``</dev/null >log 2>&1``) — see :meth:`exec`.

        *route* (C8, no default: every caller names one) is
        :meth:`exec_route`'s for a command in the sandbox, or
        :data:`~benchflow.sandbox.daytona_dind.VM_ROUTE` for BenchFlow's own
        DinD VM. A plumbing route's inner shell is ``/bin/sh`` by absolute
        path, and the session command first unsets the loader and start-up
        variables (:data:`~benchflow.sandbox._base.SYSTEM_SHELL_UNSET_VARS`) and
        sets PATH to the system folders, so ``su``, ``timeout`` and the shell
        are never found on the image's PATH and load nothing its loader
        variables name (see :meth:`exec`). The session's own shell starts
        before that line, with the sandbox's environment.

        The session command keeps the transport contract
        (:mod:`benchflow.sandbox._transport`): one over its bound sends the
        text the inner shell runs (env included) as data, staged by root
        (:meth:`_sandbox_session_exec`), and a constant reader in its place.
        *stdin_payload*: the caller's own data for the command's stdin (the
        DinD VM's ``docker compose exec`` carrying a container's staged
        command), sent the same way. A K1 agent command
        (``AgentExecCommand``) gets its env through ``uploaded_agent_env``
        (staged by root on root's plumbing route) and runs inside K1's
        barrier, on the same route.
        """
        self._require_sandbox()
        from benchflow.sandbox.agent_exec import AgentExecCommand

        cat = reader_cat(route)

        def session(inner: str, **kw: Any) -> str:
            return self._session_command(
                inner,
                timeout_sec=timeout_sec,
                shell=shell,
                cwd=cwd,
                user=user,
                route=route,
                **kw,
            )

        if isinstance(command, AgentExecCommand):
            # A root body (the env upload's own mkdir and cleanup) has no
            # agent command to rebuild: its whole text is the payload.
            agent = command if hasattr(command, "agent_target") else None
            if env:
                from benchflow.sandbox.native_agent_env import (
                    restore_command,
                    uploaded_agent_env,
                )

                root_route = await system_shell_route(self)

                async def root_exec(body: str) -> ExecResult:
                    return await self._sandbox_exec(
                        AgentExecCommand(body),
                        user="root",
                        timeout_sec=30,
                        cleanup_session=True,
                        route=root_route,
                    )

                async with uploaded_agent_env(
                    command, env, root_exec=root_exec, upload=self._exec_env_upload()
                ) as (_restored, opening):
                    fitted = fit(
                        lambda c: session(
                            restore_command(c),
                            agent_bootstrap=True,
                            agent_env_open=opening,
                        ),
                        command,
                        cat=cat,
                        agent=agent,
                    )
                    return await self._sandbox_session_exec(
                        fitted.sent,
                        payloads=_on_stdin(fitted.payload),
                        route=route,
                        timeout_sec=timeout_sec,
                        cleanup_session=cleanup_session,
                    )
            fitted = fit(
                lambda c: session(c, agent_bootstrap=True),
                command,
                cat=cat,
                agent=agent,
            )
            return await self._sandbox_session_exec(
                fitted.sent,
                payloads=_on_stdin(fitted.payload),
                route=route,
                timeout_sec=timeout_sec,
                cleanup_session=cleanup_session,
            )
        # Env vars are set inside the command rather than passed as ``env
        # KEY=value ...`` argv. The argv form would leak verifier API keys /
        # agent secrets into the remote process list and any provider-side
        # command audit log (#412). The env is written into the command (octal,
        # set by the shell's own printf and eval); an env (or command) too
        # large for one session command travels with it as staged data.
        # The wrapping must happen before the ``timeout``/``su`` prefixes so
        # they too see the exported vars; the wrapper itself runs under
        # ``sh``-compatible POSIX constructs so the surrounding shells handle
        # it fine. Ordinary commands recover the image's value only inside the
        # command shell, after its bash startup. System-shell plumbing never
        # restores that activation state.
        #
        # C8: a plumbing command (route.pin: root or the system shell) pins PATH
        # and never gets the loader variables back; every other command gets
        # the image's values (the sandbox keeps them aside, see
        # _create_sandbox_without_toolbox_bash_env) after its env.
        inner = self._inner_text(command, env, route)
        if stdin_payload is not None:
            # The caller's data is the stdin (the DinD VM's ``docker compose
            # exec`` carrying a container's staged command). This command
            # keeps the bound too (DX-1 review A M2): over it, it is staged
            # beside that stdin and read from its own descriptor.
            fitted = fit(session, inner, cat=cat, fd=_SECOND_PAYLOAD_FD)
            payloads = {0: stdin_payload}
            if fitted.payload is not None:
                payloads[_SECOND_PAYLOAD_FD] = fitted.payload
            return await self._sandbox_session_exec(
                fitted.sent,
                payloads=payloads,
                route=route,
                timeout_sec=timeout_sec,
                cleanup_session=cleanup_session,
            )
        fitted = fit(session, inner, cat=cat)
        return await self._sandbox_session_exec(
            fitted.sent,
            payloads=_on_stdin(fitted.payload),
            route=route,
            timeout_sec=timeout_sec,
            cleanup_session=cleanup_session,
        )

    @staticmethod
    def _inner_text(command: str, env: dict[str, str] | None, route: ExecRoute) -> str:
        """What the innermost shell of a (non-agent) exec runs: *command*
        after the image's values and *env*."""
        if not route.pin and not route.own_vm:
            command = (
                restore_image_env(
                    env or (), tools=route.tools if route.kit and route.root else None
                )
                + command
            )
        return (
            _wrap_daytona_command_with_env_file(
                env, command, path=route.tools, private=route.kit and route.root
            )
            if env
            else command
        )

    def _session_command(
        self,
        inner: str,
        *,
        timeout_sec: int | None,
        shell: str,
        cwd: str | None,
        user: str | int | None,
        route: ExecRoute,
        agent_bootstrap: bool = False,
        agent_env_open: str = "",
    ) -> str:
        """The session command that runs *inner* (already wrapped with its
        env) on *route*: text only, nothing sent.

        ``agent_bootstrap``: *inner* is a generic agent exec (an
        ``AgentExecCommand``), run through the native agent launcher with
        absolute tools; ``agent_env_open`` opens its uploaded environment.
        """
        hide_bash_env = ABSOLUTE_HIDE_BASH_ENV if agent_bootstrap else HIDE_BASH_ENV
        if agent_bootstrap:
            shell = "/bin/sh -c"
        pin = route.pin
        # The programs this wrapping names by path: the protected kit's once
        # the sandbox has one (C8), else the system's.
        kit = route.kit

        def tool(name: str, system: str) -> str:
            return f"{KIT_BIN}/{name}" if kit else system

        if pin:
            shell = f"{route.shell} -c"
            inner = with_system_path(inner, pin)
            command = f"{shell} {shlex.quote(inner)}"
        else:
            command = (
                f"{hide_bash_env} {shell} "
                f"{shlex.quote(restore_bash_env(route) + inner)}"
            )

        if timeout_sec:
            command = f"{tool('timeout', '/usr/bin/timeout')} {timeout_sec} {command}"

        if cwd:
            command = f"cd -- {shlex.quote(cwd)} && {command}"

        if user is not None:
            from benchflow.sandbox._su import su_command

            if pin or agent_bootstrap:
                command = su_command(
                    command,
                    user,
                    shell=route.shell if pin else SYSTEM_SHELL,
                    su=tool("su", "/bin/su"),
                    trusted_tools=agent_bootstrap,
                )
            else:
                command = (
                    f"{HIDE_BASH_ENV} /bin/bash -c "
                    f"{shlex.quote(su_command(command, user, su=tool('su', '/bin/su')))}"
                )

        # The provider's command parser sees only an absolute system shell.
        # A system shell has pinned PATH itself (inner) and does so again
        # before the session shell starts anything (outer). Under K1's
        # barrier the parser already sees absolute programs first
        # (/usr/bin/setpriv, then /bin/sh -c), so no extra quoting layer is
        # added: K1's restored env text keeps its size bound (R17, R18).
        if not pin:
            command = _root_exec_command(
                command,
                route.own_vm or (route.root and user is not None),
                route.root_path,
            )
        if not agent_bootstrap:
            command = (
                f"{route.shell if route.root else SYSTEM_SHELL} -c "
                f"{shlex.quote(command)}"
            )

        if pin:
            command = f"{system_shell_prefix(pin)}; {command}"

        if agent_bootstrap:
            # K1's barrier around the whole routed command.
            from benchflow.sandbox.agent_exec import native_agent_argv

            command = shlex.join(native_agent_argv(agent_env_open + command))
        return command

    async def _stage_payloads(self, payloads: dict[int, bytes]) -> dict[int, str]:
        """Upload each payload (by the descriptor it is read on) to one fresh
        folder in root's staging folder; their paths (see
        :meth:`_sandbox_session_exec`). A failed upload removes the folder."""
        folder = stage_folder()
        paths: dict[int, str] = {}
        try:
            for fd, data in sorted(payloads.items()):
                paths[fd] = f"{folder}/payload" if fd == 0 else f"{folder}/fd{fd}"
                await self._sdk_upload_bytes(data, paths[fd])
        except BaseException:
            await asyncio.shield(self._unstage_payload(folder))
            raise
        return paths

    async def _unstage_payload(self, folder: str) -> None:
        try:
            await self._sdk_delete_tree(folder)
        except Exception as exc:
            # Root's own folder, unreachable to anyone else: a leftover is
            # removed with the sandbox.
            logger.debug("could not remove staged command %s: %s", folder, exc)

    async def _stage_proof(self, route: ExecRoute, files: list[str]) -> str:
        """:func:`~benchflow.sandbox._transport.stage_proof` for *files* on
        the box *route* runs on: the DinD VM's system tools for BenchFlow's
        own VM, else root's system-shell route here (the protected kit's
        tools once the sandbox has one, C8)."""
        if route.own_vm:
            return stage_proof(files, shell=SYSTEM_SHELL, path=SYSTEM_PATH)
        root = await system_shell_route(self)
        return stage_proof(files, shell=root.shell, path=root.pin or SYSTEM_PATH)

    async def _sandbox_session_exec(
        self,
        command: str,
        *,
        payloads: dict[int, bytes] | None = None,
        route: ExecRoute,
        timeout_sec: int | None,
        cleanup_session: bool,
    ) -> ExecResult:
        """Send the session command *command* (built by
        :meth:`_session_command`) and poll it to completion.

        *payloads*: data the command reads, by descriptor (0, its standard
        input; another, a staged command that leaves stdin to its own
        caller). Each is uploaded by the toolbox (root) to one fresh folder
        in :data:`~benchflow.sandbox._transport.STAGE_ROOT` and opened by the
        root session shell before anything in *command* runs, and only after
        that shell has proved, in the same command, that the staging folder
        is no link, is root's with mode 700, and that the folder and every
        file are root's alone along their whole route (DX-1 review A M1:
        an image link at the staging folder that leads into a folder the
        agent owns). Nobody but root can change a route so proven, so what
        root opens is what root staged. A failed proof runs nothing and the
        exec's status is 126. The folder is removed after the command ends,
        on every path. *route*: the exec's own, which names the box.
        """
        sandbox = self._require_sandbox()
        staged = await self._stage_payloads(payloads) if payloads else None
        try:
            if staged is not None:
                proof = await self._stage_proof(route, list(staged.values()))
                opened = open_staged(command, staged, proof)
                if _transport.size(opened) > _transport.COMMAND_BOUND:
                    # A nested DinD exec can fit before its caller's stdin
                    # proof is added. Stage this outer command as well,
                    # leaving every existing payload on its own descriptor.
                    fd = next(fd for fd in range(3, 10) if fd not in staged)
                    folder = next(iter(staged.values())).rsplit("/", 1)[0]
                    staged[fd] = f"{folder}/command"
                    end = _transport.new_end()
                    await self._sdk_upload_bytes(
                        _transport.framed(command.encode(), end), staged[fd]
                    )
                    proof = await self._stage_proof(route, list(staged.values()))
                    opened = open_staged(
                        _transport.reader(reader_cat(route), end, fd=fd),
                        staged,
                        proof,
                    )
                command = opened
            _transport.sent("Daytona", command)
            # Deliberately NOT wrapped in ``_SDK_RETRY``, unlike every other SDK
            # call on this corridor: neither of the two calls below is safe to
            # replay, so a transient here is stamped (see the decorator) and
            # retried one level up, as a fresh rollout against a fresh sandbox.
            #
            # ``create_session``: the id is fixed before the call, so a create
            # that lands but whose response times out comes back
            # ``DaytonaConflictError`` on replay — a permanent error, and not in
            # the transient set, so the retry would convert a blip into a hard
            # failure.
            #
            # ``execute_session_command``: replay is silently destructive rather
            # than loud. The LiteLLM launcher binds an ephemeral port
            # (``("127.0.0.1", 0)``), so a second exec does not collide on
            # EADDRINUSE — it binds a *different* port, and its ``rm -f`` +
            # rewrite of the state/pid files erases the first launcher's
            # bookkeeping. The first proxy then survives as an orphan holding the
            # provider key and LITELLM_MASTER_KEY, invisible to
            # ``_terminate_sandbox_litellm`` (which only knows the overwritten
            # pid) for the remaining life of the sandbox.
            counts = self._exec_counts()
            if cleanup_session:
                counts.transient_exec_calls += 1
            else:
                counts.exec_calls += 1
            started = time.monotonic()
            session_id = str(uuid4())
            await sandbox.process.create_session(session_id)
            transient_token = _IN_TRANSIENT_EXEC.set(cleanup_session)
            try:
                if _sync_exec_eligible(timeout_sec):
                    assert timeout_sec is not None
                    return await self._sync_exec(session_id, command, timeout_sec)
                response = await sandbox.process.execute_session_command(
                    session_id,
                    SessionExecuteRequest(
                        command=command,
                        run_async=True,
                    ),
                    timeout=timeout_sec,
                )

                if response.cmd_id is None:
                    raise RuntimeError("Cannot find command ID.")

                # Normal sandbox execs intentionally retain their Daytona session:
                # deleting one can kill detached child processes that the caller
                # started deliberately. Short-lived polling commands opt into
                # cleanup_session=True and must not leave children behind.
                return await self._poll_response(
                    session_id,
                    response.cmd_id,
                    timeout_sec=timeout_sec,
                )
            finally:
                _IN_TRANSIENT_EXEC.reset(transient_token)
                elapsed = time.monotonic() - started
                if cleanup_session:
                    counts.transient_exec_seconds += elapsed
                    with contextlib.suppress(Exception):
                        await sandbox.process.delete_session(session_id)
                else:
                    counts.exec_seconds += elapsed
        finally:
            if staged is not None:
                folder = next(iter(staged.values())).rsplit("/", 1)[0]
                await asyncio.shield(self._unstage_payload(folder))

    async def _sync_exec(
        self, session_id: str, command: str, timeout_sec: int
    ) -> ExecResult:
        """Experimental ``run_async=False`` handling, disabled by default
        (see :data:`_DAYTONA_SYNC_EXEC_MAX_SEC`).

        - A complete answer (an exit code, stdout and stderr) is the result,
          including exit 124. HTTP latency says nothing about how long the
          command ran, as with a completed status in the asynchronous poll.
        - An answer without the result polls the command it names, like an
          asynchronous execute; the command is never sent again.
        - A failed request is the request's own error (a transport failure
          stays transient), unless the budget had passed by then: the
          session is asked for its command; one still running raises the
          poll's timeout, one that finished returns its result, and with no
          command the request's error stands. These recovery/deadline rules
          are not the asynchronous path's contract; do not enable by default.
        - Cancellation is never caught.
        """
        sandbox = self._require_sandbox()
        self._exec_counts().sync_exec_calls += 1
        loop = asyncio.get_running_loop()
        sent = loop.time()
        try:
            response = await sandbox.process.execute_session_command(
                session_id,
                SessionExecuteRequest(command=command, run_async=False),
                timeout=timeout_sec,
            )
        except Exception as exc:
            if loop.time() - sent < timeout_sec:
                raise
            return await self._sync_exec_after_budget(session_id, timeout_sec, exc)
        if _sync_response_complete(response):
            exit_code = int(response.exit_code)  # type: ignore[union-attr]
            return ExecResult(
                stdout=response.stdout,  # type: ignore[union-attr]
                stderr=response.stderr,  # type: ignore[union-attr]
                return_code=exit_code,
            )
        cmd_id = getattr(response, "cmd_id", None)
        if cmd_id is None:
            raise RuntimeError("Cannot find command ID.")
        return await self._poll_response(session_id, cmd_id, timeout_sec=timeout_sec)

    async def _sync_exec_after_budget(
        self, session_id: str, timeout_sec: int, error: Exception
    ) -> ExecResult:
        """A synchronous execute failed after its budget: tell a command that
        overran from a request that failed (see :meth:`_sync_exec`)."""
        sandbox = self._require_sandbox()
        counts = self._exec_counts()
        if _IN_TRANSIENT_EXEC.get():
            counts.transient_status_requests += 1
        else:
            counts.status_requests += 1
        try:
            session = await asyncio.wait_for(
                sandbox.process.get_session(session_id),
                timeout=_DAYTONA_POLL_STATUS_TIMEOUT_SEC,
            )
        except Exception:
            raise error from None
        commands = list(getattr(session, "commands", None) or [])
        if not commands:
            raise error
        command = commands[-1]
        exit_code = getattr(command, "exit_code", None)
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            raise self._poll_timeout_error(timeout_sec, False)
        logs = await self._get_session_command_logs_with_retry(
            session_id,
            command.id,
        )
        return ExecResult(
            stdout=logs.stdout,  # type: ignore[union-attr]
            stderr=logs.stderr,  # type: ignore[union-attr]
            return_code=exit_code,
        )

    @stamp_transient_transport
    @_SDK_RETRY
    async def _sdk_upload_file(self, source_path: Path | str, target_path: str) -> None:
        sandbox = self._require_sandbox()
        await sandbox.fs.upload_file(str(source_path), target_path)

    @stamp_transient_transport
    @_SDK_RETRY
    async def _sdk_upload_bytes(self, data: bytes, target_path: str) -> None:
        sandbox = self._require_sandbox()
        await sandbox.fs.upload_file(data, target_path)

    @stamp_transient_transport
    @_SDK_RETRY
    async def _sdk_delete_tree(self, path: str) -> None:
        sandbox = self._require_sandbox()
        await sandbox.fs.delete_file(path, recursive=True)

    @stamp_transient_transport
    @_SDK_RETRY
    async def _sdk_upload_dir(
        self,
        source_dir: Path | str,
        target_dir: str,
        exclude: Collection[str] = (),
        *,
        route: ExecRoute | None = None,
    ) -> None:
        """Upload the regular files under ``source_dir`` to ``target_dir``.

        ``exclude`` names top-level entries of ``source_dir`` to leave out
        (DinD's task-folder upload keeps the oracle and verifier behind).
        *route*: the box's root route that unpacks it,
        :data:`~benchflow.sandbox.daytona_dind.VM_ROUTE` for the DinD VM's
        own folders; root's system-shell route here by default.

        One archive, whatever the file count (``_dir_archive``, DAYTONA-GAPS
        rule 1): one SDK upload of one open file to a fresh folder in root's
        staging folder, and one root exec on that route that proves the
        archive's route root's alone (as a staged command's), unpacks it,
        owned by root, executable for anyone when the source had any execute
        bit and never set-id, and removes it. The SDK's ``upload_files``
        opened every source file at once (EMFILE at 2,000 files under
        ``ulimit -n`` 1024).

        *target_dir* must be a folder only root writes, as every caller's is
        (``/tests``, ``/verifier``, the runtime tree): ``tar`` replaces a link
        at a member's own name, but follows one at a member's folder, as the
        per-file upload did (DAYTONA-GAPS review B L2).
        """
        source_dir = Path(source_dir)
        if route is None:
            route = await system_shell_route(self)
        # Walk with followlinks=False and skip symlinks (#411): a task or
        # workspace symlink under source_dir must not exfiltrate host files
        # into the remote Daytona sandbox.
        archive, count = await asyncio.to_thread(
            write_archive,
            source_dir,
            exclude=exclude,
            context=f"daytona upload_dir {source_dir}",
        )
        try:
            if not count:
                return
            staged = f"{stage_folder()}/dir.tar"
            try:
                await self._require_sandbox().fs.upload_file(str(archive), staged)
                result = await self._sandbox_exec(
                    extract_command(
                        staged, target_dir, await self._stage_proof(route, [staged])
                    ),
                    shell="sh -c",
                    timeout_sec=600,
                    route=route,
                )
            except Exception:
                # The unpack removes its folder; an upload or exec that
                # raised (a timeout, a lost transport) did not get there
                # (DAYTONA-GAPS review A L4). A refused proof (126) keeps
                # its folder on purpose: nothing of it is removed.
                await self._remove_stage_folder(staged, route)
                raise
        finally:
            archive.unlink(missing_ok=True)
        if result.return_code == NO_TAR:
            # An image without tar: the files one SDK batch at a time.
            await self._upload_dir_by_file(source_dir, target_dir, exclude, route)
        elif result.return_code != 0:
            raise RuntimeError(
                "Could not unpack the Daytona directory upload"
                + (
                    ": its staging folder is not root's alone"
                    if result.return_code == NOT_ROOTS
                    else ""
                )
            )

    async def _remove_stage_folder(self, staged: str, route: ExecRoute) -> None:
        """Best effort: remove *staged*'s fresh stage folder as root on
        *route*, after an upload or unpack that raised."""
        folder = shlex.quote(staged.rsplit("/", 1)[0])
        try:
            await self._sandbox_exec(
                f"rm -rf -- {folder}", shell="sh -c", timeout_sec=60, route=route
            )
        except Exception:
            self.logger.debug("Could not remove the stage folder of %s", staged)

    async def _upload_dir_by_file(
        self,
        source_dir: Path,
        target_dir: str,
        exclude: Collection[str],
        route: ExecRoute,
    ) -> None:
        """:meth:`_sdk_upload_dir` where the sandbox has no ``tar``: at most
        :data:`_UPLOAD_BATCH` files open per SDK call, then root's
        ``chown``/``chmod`` per batch."""
        sandbox = self._require_sandbox()
        entries: list[tuple[str, str, int]] = []
        for file_path in iter_safe_tree(
            source_dir, context=f"daytona upload_dir {source_dir}"
        ):
            relative = file_path.relative_to(source_dir)
            if relative.parts[0] in exclude:
                continue
            entries.append(
                (
                    str(file_path),
                    f"{target_dir}/{relative.as_posix()}",
                    daytona_mode(file_path.stat().st_mode),
                )
            )
        for offset in range(0, len(entries), _UPLOAD_BATCH):
            batch = entries[offset : offset + _UPLOAD_BATCH]
            await sandbox.fs.upload_files(
                files=[FileUpload(source=src, destination=dst) for src, dst, _ in batch]
            )
            # The SDK transfers bytes only. Restore safe executable intent
            # for every caller (bootstrap, oracle and verifier), before DinD
            # copies the staged files into a service. Never carry set-id bits.
            for mode in (0o644, 0o755):
                paths = [dst for _, dst, m in batch if m == mode]
                if not paths:
                    continue
                result = await self._sandbox_exec(
                    shlex.join(["chown", "0:0", "--", *paths])
                    + " && "
                    + shlex.join(["chmod", f"{mode:o}", "--", *paths]),
                    shell="sh -c",
                    timeout_sec=60,
                    route=route,
                )
                if result.return_code != 0:
                    raise RuntimeError("Could not restore Daytona upload permissions")

    @stamp_transient_transport
    @_SDK_RETRY
    @private_download(directory=False)
    async def _sdk_download_file(
        self, source_path: str, target_path: Path | str
    ) -> None:
        sandbox = self._require_sandbox()
        try:
            info = await sandbox.fs.get_file_info(source_path)
        except DaytonaNotFoundError as exc:
            source = PurePosixPath(source_path)
            entries = await sandbox.fs.list_files(str(source.parent))
            if any(entry.name == source.name for entry in entries):
                raise RuntimeError(
                    "Daytona download_file could not inspect a listed file"
                ) from exc
            # A missing sandbox can produce the same API exception. Only
            # a successful parent inventory establishes optional absence.
            raise FileNotFoundError(source_path) from exc
        if info.is_dir:
            raise IsADirectoryError(source_path)
        try:
            await sandbox.fs.download_file(source_path, str(target_path))
        except DaytonaNotFoundError as exc:
            # It existed at the probe: losing it during capture is not an
            # absent optional companion that reward selection may ignore.
            raise RuntimeError("Daytona download_file lost an existing file") from exc

    @stamp_transient_transport
    @_SDK_RETRY
    @private_download(directory=True)
    async def _sdk_download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        sandbox = self._require_sandbox()

        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        search_result = await sandbox.fs.search_files(source_dir, "*")

        from benchflow._utils.archive_paths import ArchivePaths

        inventory = ArchivePaths()
        file_downloads = []
        file_paths = list(search_result.files)
        # Search can omit empty directories. Preserve immediate directories
        # too, particularly a reward.json/ beside a scalar reward.txt. The
        # scorer must see that unsafe companion instead of accepting a pass.
        directories = set()
        for entry in await sandbox.fs.list_files(source_dir):
            if entry.is_dir:
                path = str(PurePosixPath(source_dir) / entry.name)
                directories.add(path)
                if path not in file_paths:
                    file_paths.append(path)
        for file_path in file_paths:
            try:
                file_info = await sandbox.fs.get_file_info(file_path)
            except DaytonaNotFoundError as exc:
                raise RuntimeError(
                    "Daytona download_dir lost a listed file before transfer"
                ) from exc
            except Exception as exc:
                self.logger.warning(
                    "Could not read Daytona file info for %s; treating search "
                    "hit as a file: %s",
                    file_path,
                    exc,
                )
                file_info = None

            is_directory = file_path in directories or (
                file_info is not None and file_info.is_dir
            )
            relative_path = inventory.add(
                str(PurePosixPath(file_path).relative_to(PurePosixPath(source_dir))),
                directory=is_directory,
            )
            if is_directory:
                (target_dir / relative_path).mkdir(parents=True, exist_ok=True)
            else:
                local_file_path = target_dir / relative_path
                file_downloads.append(
                    FileDownloadRequest(
                        source=file_path,
                        destination=str(local_file_path),
                    )
                )

        if file_downloads:
            for request in file_downloads:
                Path(request.destination).parent.mkdir(parents=True, exist_ok=True)
            try:
                responses = await sandbox.fs.download_files(files=file_downloads)
                # The SDK returns per-file failures without raising. A
                # complete request needs exactly one successful response
                # and an actual destination file for each source.
                by_source: dict[str, list[Any]] = {}
                for response in responses or []:
                    by_source.setdefault(response.source, []).append(response)
                failed = []
                for request in file_downloads:
                    matches = by_source.get(request.source, [])
                    if (
                        len(matches) != 1
                        or matches[0].error is not None
                        or matches[0].result != request.destination
                        or not Path(request.destination).is_file()
                    ):
                        failed.append(request)
            except Exception as batch_error:
                self.logger.warning(
                    "Daytona batch download_dir failed for %s (%d files); "
                    "retrying files individually: %s",
                    source_dir,
                    len(file_downloads),
                    batch_error,
                )
                failed = file_downloads
            if failed:
                await self._download_files_individually(failed)

    async def _download_files_individually(self, file_downloads: list[Any]) -> None:
        sandbox = self._require_sandbox()

        failures: list[str] = []
        last_error: Exception | None = None
        downloaded = 0
        for request in file_downloads:
            source = request.source
            destination = request.destination
            try:
                # A failed batch may have left truncated bytes. A retry
                # must produce a new file, not bless the old destination.
                Path(destination).unlink(missing_ok=True)
                await sandbox.fs.download_file(source, destination)
                if not Path(destination).is_file():
                    raise RuntimeError("download returned without a destination file")
                downloaded += 1
            except Exception as exc:
                last_error = exc
                Path(destination).unlink(missing_ok=True)
                failures.append(f"{source}: {exc}")

        if failures:
            preview = "; ".join(failures[:3])
            if len(failures) > 3:
                preview += f"; ... {len(failures) - 3} more"
            raise RuntimeError(
                f"Daytona individual download_dir failed ({downloaded} recovered; "
                f"{'recovered no files; ' if not downloaded else ''}{preview})"
            ) from last_error
        if downloaded == 0:
            raise RuntimeError("Daytona individual download_dir recovered no files")

    # Public interface — delegates to strategy

    @property
    def sandbox_daemon_guard(self) -> str:
        """Where the agent user is kept from Daytona's root toolbox daemon:
        inside the sandbox ("container", see ``lockdown.guard_sandbox_daemon``)
        or, for a compose task, on the DinD VM when it starts ("vm")."""
        return "vm" if self._compose_mode else "container"

    def configure_install_layers(
        self,
        *,
        agent: str | None,
        model: str | None,
        agent_env: Mapping[str, str] | None = None,
        fresh_install: bool = False,
        install_agent: bool = True,
        sandbox_setup_timeout: int = 120,
        only_if_reused: bool = False,
    ) -> None:
        """Plan the trial's proxy and agent installs; call before :meth:`start`.

        The direct strategy builds them into the image recipe as cached layers
        unless ``fresh_install`` (with ``only_if_reused``, only as
        ``daytona_layers.claim_layer_build`` decides), and asks for disk to
        hold them either way (#1169). Compose tasks keep installing in the
        sandbox.
        """
        if getattr(self, "_compose_mode", False):
            return
        self._install_layers = plan_install_layers(
            agent=agent,
            model=model,
            agent_env=agent_env,
            # Live installers inherit task/manifest env via exec(). A build
            # does not. Keep those overrides live, never embed their values
            # (which may be secrets) in the recipe or its cache identity.
            fresh=fresh_install or bool(getattr(self, "_persistent_env", {})),
            install_agent=install_agent,
            sandbox_setup_timeout=sandbox_setup_timeout,
            only_if_reused=only_if_reused,
        )

    def cached_install_layer(
        self, kind: str, name: str
    ) -> tuple[InstallLayer, str] | None:
        """The ``kind``/``name`` layer this sandbox's image was built with, and its key."""
        plan = getattr(self, "_install_layers", None)
        key = getattr(self, "_install_layer_keys", {}).get(f"{kind}-{name}")
        if plan is None or key is None:
            return None
        layer = plan.proxy if kind == "proxy" else plan.agent
        if layer is None or layer.name != name:
            return None
        return layer, key

    def note_install_layer(self, kind: str, name: str, outcome: str) -> None:
        """Record in sandbox.json whether the trial used a cached layer."""
        record = getattr(self, "_install_layers_record", None)
        if record is None:
            return
        entry = record.setdefault("layers", {}).setdefault(f"{kind}-{name}", {})
        entry["used"] = outcome
        self._on_sandbox_created()

    async def start(self, force_build: bool) -> None:
        # A new sandbox: nothing a previous start's probe kept holds (PY8 U12).
        self._forget_kept_identity()
        from benchflow.trust import state as trust_state

        if getattr(self, "_start_snapshot_ref", None):
            self._runtime_kit_unknown()
            # Inherit the snapshot's trust state before the sandbox exists.
            trust_state.restore_from(self, getattr(self, "_start_snapshot_image", None))
        else:
            self._runtime_kit_fresh()
            trust_state.begin_fresh(self)
        return await self._strategy.start(force_build)

    async def metadata_main_restarts(self) -> str:
        """Why the cloud metadata block cannot hold ``main`` because it could
        restart ("" when it cannot). Only a Docker-in-Docker task has a
        Compose ``main`` with a restart policy."""
        if isinstance(self._strategy, _DaytonaDinD):
            return await self._strategy.main_restart_refusal()
        return ""

    async def script_container_config(self) -> dict[str, Any] | None:
        if isinstance(self._strategy, _DaytonaDinD):
            return await self._strategy.script_container_config()
        return None

    async def identity_container_config(
        self, service: str = "main"
    ) -> dict[str, Any] | None:
        if isinstance(self._strategy, _DaytonaDinD):
            return await self._strategy.identity_container_config(service)
        return None

    async def identity_exec(
        self, command: str, *, service: str = "main", **kwargs: Any
    ) -> ExecResult:
        if isinstance(self._strategy, _DaytonaDinD):
            return await self._strategy.identity_exec(
                command, service=service, **kwargs
            )
        return await super().identity_exec(command, service=service, **kwargs)

    async def stop(self, delete: bool) -> None:
        self._forget_kept_identity()
        # The sandbox is going away: its lease stops being renewed.
        self._stop_lease_renewal()
        # Written after the strategy stops, so the execs of stopping (a
        # compose task's ``compose down``) are counted too.
        try:
            return await self._strategy.stop(delete)
        finally:
            try:
                # Every create attempt is accounted for before stop returns.
                await self._drain_create_attempts()
            finally:
                self._write_exec_counts()

    def _exec_counts(self) -> _ExecCounts:
        # Made on first use: tests build sandboxes with __new__, without
        # __init__.
        counts = getattr(self, "_exec_counts_record", None)
        if counts is None:
            counts = self._exec_counts_record = _ExecCounts()
        return counts

    def _write_exec_counts(self) -> None:
        """Write ``daytona-execs.json`` in the sandbox's folder: the execs
        this sandbox ran, the status requests their polls made and the
        seconds they took, ordinary and transient execs apart (see
        ``_ExecCounts``), so a run on Daytona can be compared by its round
        trips. Best effort, like ``sandbox.json``: a failed write only logs."""
        counts = getattr(self, "_exec_counts_record", None)
        paths = getattr(self, "rollout_paths", None)
        if counts is None or paths is None:
            return
        record = asdict(counts)
        for field in ("exec_seconds", "transient_exec_seconds"):
            record[field] = round(record[field], 3)
        try:
            (paths.rollout_dir / EXEC_COUNTS_FILE).write_text(
                json.dumps(record, indent=2)
            )
        except Exception as e:
            logger.warning(f"Could not write {EXEC_COUNTS_FILE}: {e}")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
        *,
        system_shell: bool = False,
    ) -> ExecResult:
        """Run ``command`` to completion in the sandbox and return its result.

        Daytona runs every command as a *session* command and reports its
        ``exit_code`` only once the command completes; it treats the command as
        still-running while any child keeps the session's stdout/stderr stream
        open. A backgrounded process that inherits those fds (``mysvc &`` with no
        redirection) therefore holds ``exec`` open until ``timeout_sec`` elapses
        — or, when ``timeout_sec`` is ``None``, until the safety-net cap
        (:data:`_DAYTONA_EXEC_HARD_CAP_SEC`, currently 3600s) is reached, at
        which point a "Command timed out" :class:`RuntimeError` is raised. To run
        a daemon in the background, fully detach its std fds so it does not hold
        the session stream open, e.g.
        ``nohup CMD </dev/null >log 2>&1 &`` (redirection alone severs the
        inherited session stream; ``disown`` is bash/zsh-only and absent from a
        plain ``sh`` hook shell).

        This is a Daytona-specific asymmetry: ``DockerSandbox.exec`` returns as
        soon as the foreground ``sh -c`` exits regardless of orphaned background
        children, so a ``mysvc &`` setup hook can pass under Docker yet wedge on
        Daytona without the redirection above.

        ``system_shell`` starts ``/bin/sh`` by absolute path with PATH set to
        the system folders before anything else runs (see
        :data:`benchflow.sandbox._base.SYSTEM_SHELL`).
        """
        route = await self.exec_route(user, system_shell=system_shell, service=service)
        env = self._merge_env(
            env,
            user,
            plumbing=route.pin is not None,
            explicit_root=route.explicit_root,
        )
        command = self._default_root_path(command, user, env, route)
        user = self._resolve_user(user)
        return await self._strategy.exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            service=service,
            route=route,
        )

    async def exec_transient(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        """Run a child-free command and delete its Daytona session afterward.

        Daytona session commands may leave their wrapper shell alive after the
        command has completed. High-frequency polling must use this method so a
        long rollout does not accumulate thousands of orphaned shells. Callers
        must not use it for commands intended to leave background services
        running, because deleting the session may terminate those children.
        """
        route = await self.exec_route(user, service=service)
        env = self._merge_env(
            env,
            user,
            plumbing=route.pin is not None,
            explicit_root=route.explicit_root,
        )
        command = self._default_root_path(command, user, env, route)
        user = self._resolve_user(user)
        return await self._strategy.exec_transient(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            service=service,
            route=route,
        )

    async def services(self) -> list[str]:
        return await self._strategy.services()

    async def upload_file(
        self, source_path: Path | str, target_path: str, *, mode: str | None = None
    ) -> None:
        await self._strategy.upload_file(source_path, target_path)
        await self._apply_upload_mode(target_path, mode)

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        return await self._strategy.upload_dir(source_dir, target_dir, service=service)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        return await self._strategy.download_file(source_path, target_path)

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        return await self._strategy.download_dir(
            source_dir, target_dir, service=service
        )

    async def is_dir(
        self, path: str, user: str | int | None = None, service: str = "main"
    ) -> bool:
        # The strategy fast-path only knows the agent's `main` container; a
        # non-main service must route through the generic `test -d` exec path
        # so the compose `service` selector is honored.
        if service != "main":
            return await super().is_dir(path, user=user, service=service)
        return await self._strategy.is_dir(
            path, user=self._resolve_user(user), route=await self.exec_route(user)
        )

    async def is_file(
        self, path: str, user: str | int | None = None, service: str = "main"
    ) -> bool:
        # The strategy fast-path only knows the agent's `main` container; a
        # non-main service must route through the generic `test -f` exec path
        # so the compose `service` selector is honored.
        if service != "main":
            return await super().is_file(path, user=user, service=service)
        return await self._strategy.is_file(
            path, user=self._resolve_user(user), route=await self.exec_route(user)
        )

    async def live_process(self, *, agent: str | None = None) -> LiveProcess:
        """Pick the Daytona ACP transport (PTY by default, SSH for gemini).

        Selection lives here rather than at the ACP layer so the choice and the
        provenance string reported by ``selected_acp_transport`` stay derived
        from the same helper.
        """
        from benchflow.acp.selection import selected_acp_transport
        from benchflow.sandbox.process import DaytonaProcess, DaytonaPtyProcess

        transport = selected_acp_transport(agent=agent or "", environment="daytona")
        if transport == "ssh":
            logger.info("Using SSH transport for %s on Daytona", agent)
            return await DaytonaProcess.from_sandbox_env(self)
        logger.info("Using PTY transport for Daytona sandbox")
        return await DaytonaPtyProcess.from_sandbox_env(self)

    async def attach(self) -> None:
        return await self._strategy.attach()

    # Container snapshot/restore — delegates to active strategy (#384)

    @property
    def supports_system_shell(self) -> bool:
        return True

    @property
    def supports_runtime_kit(self) -> bool:
        return True

    @property
    def agent_container_no_new_privileges(self) -> bool:
        return self._uses_compose

    @property
    def agent_exec_shell(self) -> str:
        return "/bin/sh" if self._uses_compose else "/bin/bash"

    @property
    def supports_snapshot(self) -> bool:
        return self._strategy.supports_snapshot

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        from benchflow.sandbox import _path_inspector_state
        from benchflow.trust import state as trust_state

        return trust_state.stamp_snapshot(
            self,
            _path_inspector_state.checkpoint(self, await self._strategy.snapshot(name)),
        )

    async def restore(self, image: SandboxImage) -> None:
        from benchflow.sandbox import _path_inspector_state

        path_inspector = _path_inspector_state.read(image)
        # What the start probe kept describes the state being replaced.
        self._forget_kept_identity()
        self._runtime_kit_unknown()
        from benchflow.trust import state as trust_state

        trust_state.restore_from(self, image)
        restored = await self._strategy.restore(image)
        _path_inspector_state.restored(self, path_inspector)
        return restored

    async def adopt_snapshot(self, image: SandboxImage) -> None:
        """Associate the live credential files with ``image`` for restores."""
        from benchflow.trust import state as trust_state

        trust_state.admit(self, "main", trust_state.IngressKind.SNAPSHOT_RESTORE)
        adopt = getattr(self._strategy, "adopt_snapshot", None)
        if adopt is not None:
            await adopt(image)

    def start_from_snapshot(self, image: SandboxImage) -> bool:
        """Make the next ``start()`` create the sandbox from ``image``.

        Used for isolated branch children. Returns False when this sandbox
        cannot (another provider's snapshot, or a strategy without snapshot
        support); the caller then starts normally and restores ``image``.
        """
        if image.provider != "daytona" or not self._strategy.supports_snapshot:
            return False
        from benchflow.sandbox import _path_inspector_state

        path_inspector = _path_inspector_state.read(image)
        self._start_snapshot_env = _checkpoint_startup_env(image)
        self._start_snapshot_ref = image.ref
        self._start_snapshot_image = image
        _path_inspector_state.restored(self, path_inspector)
        return True

    async def delete_snapshot(self, image: SandboxImage) -> bool:
        return await self._strategy.delete_snapshot(image)
