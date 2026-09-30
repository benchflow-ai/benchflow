"""Idempotent patches to the Daytona SDK to work around upstream bugs.

Imported (and applied) lazily from ``_env_setup._create_environment`` so the
SDK is only touched when a Daytona environment is actually being built.
"""

import asyncio
import inspect
import logging
import os
from typing import Any, cast

logger = logging.getLogger(__name__)

_PATCHED = False
_HEARTBEAT_PATCHED = "_benchflow_ws_heartbeat"

# Seconds between websocket pings on the SDK's PTY and log websockets; 0 or
# "off" disables. aiohttp closes a websocket whose pong is missing after half
# the interval.
DAYTONA_WS_HEARTBEAT_ENV = "BENCHFLOW_DAYTONA_WS_HEARTBEAT_SEC"
_DAYTONA_WS_HEARTBEAT_DEFAULT_SEC = 120.0


def daytona_ws_heartbeat_sec() -> float | None:
    """The websocket ping interval, or None when pings are off."""
    raw = os.environ.get(DAYTONA_WS_HEARTBEAT_ENV, "").strip().lower()
    if not raw:
        return _DAYTONA_WS_HEARTBEAT_DEFAULT_SEC
    if raw in {"0", "off", "none", "false"}:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not value > 0:  # also rejects NaN
        logger.warning(
            "Invalid %s=%r; pinging every %.0fs",
            DAYTONA_WS_HEARTBEAT_ENV,
            raw,
            _DAYTONA_WS_HEARTBEAT_DEFAULT_SEC,
        )
        return _DAYTONA_WS_HEARTBEAT_DEFAULT_SEC
    return value


def _patch_websocket_heartbeat(sdk_process: Any = None) -> None:
    """Open the SDK's websockets with aiohttp's ``heartbeat``.

    The agent's ACP stream rides a Daytona PTY websocket that the SDK opens
    with no pings, and the Daytona side sends none. A quiet channel (a long
    model turn, a long tool call) then carries no frames: a NAT on the path
    (Azure SNAT drops a flow after about four idle minutes) can drop it
    without a close frame, and the next ACP message vanishes while both ends
    wait. With a heartbeat, frames keep flowing, and a missed pong closes the
    websocket, which DaytonaPtyProcess reports at once as a lost transport
    (#1143, #1144). Only ``AsyncProcess._open_ws(self, url, headers)`` of the
    SDK layout this was written against (0.184) is wrapped; another layout is
    left alone.
    """
    if sdk_process is None:
        try:
            from daytona._async import process as sdk_process
        except Exception:  # SDK not installed, or its layout changed
            logger.debug("daytona SDK process module not importable", exc_info=True)
            return
    cls = cast(Any, getattr(sdk_process, "AsyncProcess", None))
    original = getattr(cls, "_open_ws", None)
    if original is None or not hasattr(sdk_process, "http_session_of"):
        logger.debug("daytona SDK has no _open_ws/http_session_of; no ws heartbeat")
        return
    if getattr(original, _HEARTBEAT_PATCHED, False):
        return
    try:
        params = list(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        return
    if params != ["self", "url", "headers"]:
        logger.debug("daytona SDK _open_ws%s changed; no ws heartbeat", params)
        return

    async def _open_ws(self: Any, url: str, headers: dict[str, str]) -> Any:
        heartbeat = daytona_ws_heartbeat_sec()
        if heartbeat is None:
            return await original(self, url, headers)
        # The SDK's own body, plus the heartbeat.
        session = sdk_process.http_session_of(self._api_client.api_client)
        return await session.ws_connect(url, headers=headers, heartbeat=heartbeat)

    setattr(_open_ws, _HEARTBEAT_PATCHED, True)
    _open_ws.__wrapped__ = original  # ty: ignore[unresolved-attribute]
    cls._open_ws = _open_ws


def apply() -> None:
    """Install workarounds for known Daytona SDK bugs.

    Currently:
      * ``AsyncProcess._open_ws`` opens websockets without pings
        (:func:`_patch_websocket_heartbeat`).
      * ``AsyncProcess.get_session_command_logs`` occasionally raises
        ``pydantic.ValidationError`` because the server returns an empty
        string instead of a JSON object for ``SessionCommandLogsResponse``.
        Reproduces in SDK 0.168.x and 0.169.x. Wrap with a small bounded
        retry that returns an empty-but-valid response if every attempt
        fails — callers can still observe the command's exit_code via
        ``get_session_command``, so a missing logs payload is recoverable.
    """
    global _PATCHED
    _patch_websocket_heartbeat()
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
        return isinstance(exc, DaytonaError) and _MALFORMED_MARKER in str(exc)

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

    async_process_cls = cast(Any, AsyncProcess)
    async_process_cls.get_session_command_logs = _patched_get_session_command_logs
    _PATCHED = True
    logger.debug("daytona SDK patches applied")
