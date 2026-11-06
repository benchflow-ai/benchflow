"""Command-wrapping and exec helpers for the Daytona backend.

Extracted from ``benchflow.sandbox.daytona`` as a cohesion seam: the small,
SDK-free helpers that shape what gets run on a Daytona sandbox — the secret-safe
env-file command wrapper, exec-failure output formatting, the single-container
service guard, and the API-key preflight. The names here are re-exported from
``benchflow.sandbox.daytona`` so existing imports such as
``from benchflow.sandbox.daytona import _wrap_daytona_command_with_env_file``
keep working unchanged.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Coroutine
from typing import Any, cast

from benchflow._utils.text import describe_exception
from benchflow.diagnostics import TransientSandboxTransportError
from benchflow.sandbox._base import (
    SYSTEM_PATH,
    ExecResult,
    wrap_command_with_env_file,
)

# Kept for the imports that still name it; no env file is written.
_DAYTONA_ENV_FILE_PREFIX = "/tmp/.benchflow_daytona_env_"


def _wrap_daytona_command_with_env_file(
    env: dict[str, str], command: str, *, path: str = SYSTEM_PATH, private: bool = False
) -> str:
    """Return *command* prefixed to set *env* with the shell's own builtins.

    Thin wrapper over the canonical
    :func:`benchflow.sandbox._base.wrap_command_with_env_file` so the
    secret-redaction logic lives in exactly one place. See that function for
    the full contract: no value is in the remote command as ``KEY=VALUE``
    (visible via ``ps``, Daytona audit logs, or any provider-side command
    logging); they are octal escapes the shell's ``printf`` and ``eval``
    apply, with no program, file or variable of its own. A command this
    makes too large for one session command is sent as data, staged by root
    (``DaytonaSandbox._sandbox_session_exec``, the transport contract).

    Issue #412: previously this used ``env K=V ...`` argv, which placed raw
    secret values into the remote command line.
    """
    return wrap_command_with_env_file(
        env,
        command,
        env_path_prefix=_DAYTONA_ENV_FILE_PREFIX,
        path=path,
        private=private,
    )


def _exec_failure_output(result: ExecResult) -> str:
    output = " ".join(
        text.strip()
        for text in (result.stdout or "", result.stderr or "")
        if text and text.strip()
    )
    return output[:4000]


def _reject_non_main_service(service: str) -> None:
    """Raise ``ValueError`` for a non-``main`` service on the direct strategy.

    The direct (single-container) Daytona sandbox cannot target additional
    compose services; multi-container (vulhub-style) tasks require a
    ``docker-compose.yaml`` (#248). Centralizes the identical guard that
    ``_DaytonaDirect.exec``/``upload_dir``/``download_dir`` each raised inline.
    """
    if service != "main":
        raise ValueError(
            f"Direct (non-compose) Daytona sandbox is single-container "
            f"and cannot target service {service!r}. Multi-container "
            "(vulhub-style) tasks require a docker-compose.yaml (#248)."
        )


def _daytona_preflight() -> None:
    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit(
            "Daytona requires DAYTONA_API_KEY to be set. "
            "Please set this environment variable and try again."
        )


# Exact names: a subclass such as SDK 0.207's DaytonaProcessExecutionTimeoutError
# (a command outran its own timeout) is no transport failure. SDK 0.201+ raises
# DaytonaConnectionTimeoutError (connection and timeout) for a transport
# timeout, often with an empty message.
_DAYTONA_TRANSIENT_RETRY_CLASS_NAMES = frozenset(
    {
        "DaytonaConnectionError",
        "DaytonaConnectionTimeoutError",
        "DaytonaRateLimitError",
        "DaytonaTimeoutError",
    }
)
_DAYTONA_EMPTY_EXIT_CODE_MARKERS = (
    "failed to convert exit code to int",
    'strconv.Atoi: parsing "": invalid syntax',
)


def _is_daytona_transient_retry_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    exc_type = type(exc)
    if exc_type.__module__.startswith("daytona.") and all(
        marker in str(exc) for marker in _DAYTONA_EMPTY_EXIT_CODE_MARKERS
    ):
        return True
    return (
        exc_type.__module__.startswith("daytona.")
        and exc_type.__name__ in _DAYTONA_TRANSIENT_RETRY_CLASS_NAMES
    )


class DaytonaSandboxStoppedError(RuntimeError):
    """Daytona answered that the trial's sandbox is not running."""


def _sandbox_stopped_error(exc: BaseException) -> DaytonaSandboxStoppedError | None:
    """Name the likely cause when Daytona says the sandbox is not running.

    The SDK's own text ("Is the Sandbox started? ... SANDBOX_NOT_RUNNING")
    does not say why. A sandbox BenchFlow holds is kept running by its
    controller's calls; Daytona stops it after its inactivity interval
    (``daytona_reaper.AUTO_STOP_MIN``) without one (DAYTONA-START round 3,
    rule B). It stays an infrastructure error (``SANDBOX_NOT_RUNNING``
    marks it for ``scoring``).
    """
    if not type(exc).__module__.startswith("daytona.") or (
        "SANDBOX_NOT_RUNNING" not in str(exc)
    ):
        return None
    from benchflow.sandbox import daytona_reaper

    return DaytonaSandboxStoppedError(
        "Daytona stopped the sandbox (SANDBOX_NOT_RUNNING). It stops a "
        f"sandbox after {daytona_reaper.AUTO_STOP_MIN} min without activity "
        "from its controller: was this process paused, asleep or offline "
        f"that long? Daytona said: {describe_exception(exc)}"
    )


def stamp_transient_transport[M: Callable[..., Coroutine[Any, Any, Any]]](fn: M) -> M:
    """Re-raise a transient Daytona SDK failure as a benchflow-owned error.

    Wrap every method that touches the Daytona SDK — exec, upload, download,
    stat. The vendor exception type is only alive *here*: by the time an
    error reaches classification it has been flattened to a string (and may
    have crossed a worker boundary), so ``DaytonaTimeoutError`` is
    indistinguishable from any other sentence. Deciding at the raise site
    lets :func:`_is_daytona_transient_retry_error` inspect the real class,
    and stamps one stable marker that
    :func:`benchflow._utils.scoring._looks_like_infra_error` can match
    forever — instead of the classifier tracking one message prefix per
    vendor method.

    Only transient failures are stamped. A permanent 401/400/409 shares the
    same vendor prefix but must stay out of ``infra_failure``, or a dead
    credential would be retried as though it were a blip.

    Apply this *outside* ``_SDK_RETRY`` so the retry budget is spent first
    and only the final, still-failing error is stamped.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except TransientSandboxTransportError:
            raise
        except Exception as exc:
            stopped = _sandbox_stopped_error(exc)
            if stopped is not None:
                raise stopped from exc
            if _is_daytona_transient_retry_error(exc):
                raise TransientSandboxTransportError(describe_exception(exc)) from exc
            raise

    return cast(M, wrapper)
