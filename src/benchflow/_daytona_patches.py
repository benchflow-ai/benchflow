"""Idempotent patches to the Daytona SDK to work around upstream bugs.

Imported (and applied) lazily from ``_env_setup._create_environment`` so the
SDK is only touched when a Daytona environment is actually being built.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED = False


def apply() -> None:
    """Install workarounds for known Daytona SDK bugs.

    Currently:
      * ``AsyncProcess.get_session_command_logs`` occasionally raises
        ``pydantic.ValidationError`` because the server returns an empty
        string instead of a JSON object for ``SessionCommandLogsResponse``.
        Reproduces in SDK 0.168.x and 0.169.x. Wrap with a small bounded
        retry that returns an empty-but-valid response if every attempt
        fails — callers can still observe the command's exit_code via
        ``get_session_command``, so a missing logs payload is recoverable.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        from daytona._async.process import AsyncProcess
        from daytona.common.errors import DaytonaError
        from daytona.common.process import SessionCommandLogsResponse
    except Exception:  # pragma: no cover — SDK not installed / layout changed
        logger.debug("daytona SDK not importable; skipping patches", exc_info=True)
        return

    try:
        from pydantic import ValidationError
    except Exception:  # pragma: no cover
        return

    # AsyncProcess.get_session_command_logs is decorated by intercept_errors
    # at class definition, which converts every inner exception (including
    # the pydantic ValidationError we care about) into a DaytonaError. The
    # decorated bound method is what we capture here, so we have to match
    # on the wrapped DaytonaError shape too — not just ValidationError.
    original = AsyncProcess.get_session_command_logs

    _MALFORMED_MARKER = "SessionCommandLogsResponse"

    def _is_malformed_logs_error(exc: BaseException) -> bool:
        if isinstance(exc, ValidationError):
            return True
        if isinstance(exc, DaytonaError) and _MALFORMED_MARKER in str(exc):
            return True
        return False

    async def _patched_get_session_command_logs(
        self: Any, session_id: str, command_id: str
    ) -> SessionCommandLogsResponse:
        # Harbor already wraps this call in tenacity (3 attempts), so
        # additional retries here are usually wasted on a deterministic
        # malformed payload. Try once more with a small delay in case it
        # IS transient, then return an empty-but-valid response so the
        # caller can still observe the command's exit_code via
        # get_session_command. Original error is logged for triage.
        attempts = 2
        delay = 0.5
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await original(self, session_id, command_id)
            except (ValidationError, DaytonaError) as exc:
                if not _is_malformed_logs_error(exc):
                    raise
                last_exc = exc
                logger.warning(
                    "daytona get_session_command_logs malformed payload "
                    "(attempt %d/%d) for session=%s command=%s: %s",
                    attempt,
                    attempts,
                    session_id,
                    command_id,
                    exc,
                )
                if attempt < attempts:
                    await asyncio.sleep(delay)

        logger.error(
            "daytona get_session_command_logs malformed %d times for "
            "session=%s command=%s; falling back to empty logs (%s)",
            attempts,
            session_id,
            command_id,
            last_exc,
        )
        return SessionCommandLogsResponse(output="", stdout="", stderr="")

    AsyncProcess.get_session_command_logs = _patched_get_session_command_logs  # type: ignore[method-assign]
    _PATCHED = True
    logger.debug("daytona SDK patches applied")
