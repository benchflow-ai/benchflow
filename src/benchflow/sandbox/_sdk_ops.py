"""Idempotent patches to the Daytona SDK to work around upstream bugs.

Imported (and applied) lazily from ``_env_setup._create_environment`` so the
SDK is only touched when a Daytona environment is actually being built.
"""

import asyncio
import hashlib
import importlib.metadata
import inspect
import logging
import os
from collections.abc import Callable
from typing import Any, cast

from benchflow.sandbox.daytona_build_keys import context_hash

logger = logging.getLogger(__name__)

_PATCHED = False
_HEARTBEAT_PATCHED = "_benchflow_ws_heartbeat"
_CONTEXT_HASH_PATCHED = "_benchflow_sorted_context_hash"
# sha256 of ``inspect.getsource`` of the SDK's two context hashes, the source
# daytona_build_keys.context_hash reproduces: daytona 0.207.0's, the same
# text from 0.184.0 to 0.225.0. Another source keeps the SDK's own walk.
_AUDITED_CONTEXT_HASH_SHA256 = {
    "ObjectStorage": "27e66cddea2e8783d08304524fdb2a864e03bbedccb1ca3d5b295bce3c883544",
    "AsyncObjectStorage": "b75fb096334c799833ca4920958feb77db44b6e9b5586b6c5d20b9005369bd51",
}
# Set once the stand-down is reported, so each trial's apply() stays quiet.
_context_hash_refusal_logged = False

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
    (#1143, #1144). Only the SDK layouts this was written against are wrapped:
    ``AsyncProcess._open_ws(self, url, headers)`` (0.184 to 0.196) and
    ``_open_ws(self, url, headers, subprotocols=None)`` (0.197 to 0.207, which
    carries the PTY's exit-control and env subprotocols); another layout is
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

    async def _open_ws(self: Any, url: str, headers: dict[str, str]) -> Any:
        heartbeat = daytona_ws_heartbeat_sec()
        if heartbeat is None:
            return await original(self, url, headers)
        # The SDK's own body (0.184 to 0.196), plus the heartbeat.
        session = sdk_process.http_session_of(self._api_client.api_client)
        return await session.ws_connect(url, headers=headers, heartbeat=heartbeat)

    async def _open_ws_with_subprotocols(
        self: Any,
        url: str,
        headers: dict[str, str],
        subprotocols: list[str] | None = None,
    ) -> Any:
        heartbeat = daytona_ws_heartbeat_sec()
        if heartbeat is None:
            return await original(self, url, headers, subprotocols)
        # The SDK's own body (0.197 to 0.207), plus the heartbeat.
        session = sdk_process.http_session_of(self._api_client.api_client)
        return await session.ws_connect(
            url, headers=headers, protocols=subprotocols or (), heartbeat=heartbeat
        )

    layouts: dict[tuple[str, ...], Callable[..., Any]] = {
        ("self", "url", "headers"): _open_ws,
        ("self", "url", "headers", "subprotocols"): _open_ws_with_subprotocols,
    }
    wrapper = layouts.get(tuple(params))
    if wrapper is None:
        logger.debug("daytona SDK _open_ws%s changed; no ws heartbeat", params)
        return
    setattr(wrapper, _HEARTBEAT_PATCHED, True)
    wrapper.__wrapped__ = original  # ty: ignore[unresolved-attribute]
    cls._open_ws = wrapper


def _source_sha256(function: Any) -> str | None:
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode()).hexdigest()


def _patch_context_hash(sync_storage: Any = None, async_storage: Any = None) -> None:
    """Hash a build context's folders in sorted order, so a task keeps one
    Daytona build key on every machine (``daytona_build_keys``).

    The SDK's ``_compute_hash_for_path_md5`` (``ObjectStorage`` and
    ``AsyncObjectStorage``) walks a COPY folder in the filesystem's listing
    order, and Daytona keys the build by that hash. Both become
    ``daytona_build_keys.context_hash`` only while both are the source it
    reproduces (:data:`_AUDITED_CONTEXT_HASH_SHA256`); otherwise both keep
    the SDK's walk, and a warning says so once. The async one hashes in a
    worker thread, as the SDK walks its folders.
    """
    global _context_hash_refusal_logged
    if sync_storage is None or async_storage is None:
        try:
            from daytona._async.object_storage import AsyncObjectStorage
            from daytona._sync.object_storage import ObjectStorage
        except Exception:  # SDK not installed, or its layout changed
            logger.debug("daytona SDK object storage not importable", exc_info=True)
            return
        sync_storage, async_storage = ObjectStorage, AsyncObjectStorage
    owners = {"ObjectStorage": sync_storage, "AsyncObjectStorage": async_storage}
    originals = {
        name: getattr(owner, "_compute_hash_for_path_md5", None)
        for name, owner in owners.items()
    }
    if all(getattr(f, _CONTEXT_HASH_PATCHED, False) for f in originals.values()):
        return
    changed = [
        name
        for name, function in originals.items()
        if _source_sha256(function) != _AUDITED_CONTEXT_HASH_SHA256[name]
    ]
    if changed:
        if not _context_hash_refusal_logged:
            _context_hash_refusal_logged = True
            try:
                release = importlib.metadata.version("daytona")
            except importlib.metadata.PackageNotFoundError:
                release = "?"
            logger.warning(
                "Daytona build keys keep the SDK's folder order: %s in daytona "
                "%s is not the source BenchFlow's sorted context hash was "
                "written against, so a task with a COPY folder can get another "
                "build key on another machine",
                " and ".join(f"{name}._compute_hash_for_path_md5" for name in changed),
                release,
            )
        return

    def _compute_hash_for_path_md5(
        self: Any, path_str: str, archive_base_path: str | None = None
    ) -> str:
        return context_hash(path_str, archive_base_path, self.compute_archive_base_path)

    async def _compute_hash_for_path_md5_async(
        self: Any, path_str: str, archive_base_path: str | None = None
    ) -> str:
        return await asyncio.to_thread(
            context_hash, path_str, archive_base_path, self.compute_archive_base_path
        )

    wrappers: dict[str, Any] = {
        "ObjectStorage": _compute_hash_for_path_md5,
        "AsyncObjectStorage": _compute_hash_for_path_md5_async,
    }
    for name, wrapper in wrappers.items():
        setattr(wrapper, _CONTEXT_HASH_PATCHED, True)
        wrapper.__wrapped__ = originals[name]
        owners[name]._compute_hash_for_path_md5 = wrapper


def apply() -> None:
    """Install workarounds for known Daytona SDK bugs.

    Currently:
      * ``AsyncProcess._open_ws`` opens websockets without pings
        (:func:`_patch_websocket_heartbeat`).
      * ``ObjectStorage._compute_hash_for_path_md5`` (sync and async) hashes
        a build context's folders in the filesystem's listing order, so a
        task's build key depends on the machine (:func:`_patch_context_hash`).
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
    _patch_context_hash()
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
        # The caller already wraps this call in tenacity (3 attempts), so
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
