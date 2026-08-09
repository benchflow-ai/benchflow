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
from benchflow.sandbox._base import ExecResult, wrap_command_with_env_file

# Prefix for the decoded env file inside the Daytona sandbox. A unique 16-hex
# suffix is appended by the shared wrapper so concurrent exec() calls can't
# clobber each other's env file.
_DAYTONA_ENV_FILE_PREFIX = "/tmp/.benchflow_daytona_env_"


def _wrap_daytona_command_with_env_file(env: dict[str, str], command: str) -> str:
    """Return *command* prefixed to materialize *env* from a file.

    Thin wrapper over the canonical
    :func:`benchflow.sandbox._base.wrap_command_with_env_file` so the
    secret-redaction logic lives in exactly one place (shared with the Docker
    backend). See that function for the full contract: secrets never reach the
    remote process argv (visible via ``ps``, Daytona audit logs, or any
    provider-side command logging) — they are base64-encoded into the command
    string, decoded to a mode-0600 file inside the sandbox, sourced, and
    unconditionally removed via ``trap ... EXIT``.

    Issue #412: previously this used ``env K=V ...`` argv, which placed raw
    secret values into the remote command line.
    """
    return wrap_command_with_env_file(
        env, command, env_path_prefix=_DAYTONA_ENV_FILE_PREFIX
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


_DAYTONA_TRANSIENT_RETRY_CLASS_NAMES = frozenset(
    {
        "DaytonaConnectionError",
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
            if _is_daytona_transient_retry_error(exc):
                raise TransientSandboxTransportError(describe_exception(exc)) from exc
            raise

    return cast(M, wrapper)
