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
        from daytona.common.process import SessionCommandLogsResponse
    except Exception:  # pragma: no cover — SDK not installed / layout changed
        logger.debug("daytona SDK not importable; skipping patches", exc_info=True)
        return

    try:
        from pydantic import ValidationError
    except Exception:  # pragma: no cover
        return

    original = AsyncProcess.get_session_command_logs

    async def _patched_get_session_command_logs(
        self: Any, session_id: str, command_id: str
    ) -> SessionCommandLogsResponse:
        attempts = 4
        delay = 0.5
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await original(self, session_id, command_id)
            except ValidationError as exc:
                last_exc = exc
                logger.warning(
                    "daytona get_session_command_logs ValidationError (attempt %d/%d) "
                    "for session=%s command=%s: %s",
                    attempt,
                    attempts,
                    session_id,
                    command_id,
                    exc,
                )
                if attempt < attempts:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 4.0)

        logger.error(
            "daytona get_session_command_logs returned malformed payloads "
            "%d times for session=%s command=%s; falling back to empty logs",
            attempts,
            session_id,
            command_id,
        )
        # Return a valid empty response so callers can still inspect the
        # command's exit_code via get_session_command. The original error
        # is logged above for debuggability.
        _ = last_exc  # retained for log context
        return SessionCommandLogsResponse(output="", stdout="", stderr="")

    AsyncProcess.get_session_command_logs = _patched_get_session_command_logs  # type: ignore[method-assign]
    _PATCHED = True
    logger.debug("daytona SDK patches applied")
