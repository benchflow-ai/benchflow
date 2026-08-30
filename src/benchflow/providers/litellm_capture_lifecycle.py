"""Fail-closed capture accounting and graceful LiteLLM shutdown helpers."""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import subprocess
from typing import Any

LITELLM_CAPTURE_STATE_ENV = "BENCHFLOW_LITELLM_CAPTURE_STATE_PATH"
PROVIDER_DRAIN_TIMEOUT_SEC = 610


def capture_journal_error(
    payload: dict[str, Any] | None,
    *,
    exchange_count: int,
) -> str | None:
    """Validate durable accepted/terminal counts against imported JSONL rows."""

    if payload is None:
        return "LiteLLM capture attempt journal is missing"
    attempt_count = payload.get("attempt_count")
    terminal_count = payload.get("terminal_count")
    if (
        not isinstance(attempt_count, int)
        or isinstance(attempt_count, bool)
        or attempt_count < 0
        or not isinstance(terminal_count, int)
        or isinstance(terminal_count, bool)
        or terminal_count < 0
    ):
        return "LiteLLM capture attempt journal is malformed"
    if attempt_count != terminal_count or terminal_count != exchange_count:
        return (
            "LiteLLM capture did not drain every accepted provider request: "
            f"attempts={attempt_count}, terminal={terminal_count}, "
            f"exchanges={exchange_count}"
        )
    return None


async def drain_host_process(process: subprocess.Popen[bytes]) -> str | None:
    """Stop host acceptance and wait for the proxy to drain active requests."""

    if process.poll() is not None:
        return None
    process.terminate()
    try:
        await asyncio.to_thread(process.wait, PROVIDER_DRAIN_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        process.kill()
        await asyncio.to_thread(process.wait, 10)
        return "LiteLLM graceful shutdown timed out before provider requests drained"
    return None


async def drain_sandbox_process(
    sandbox: Any,
    *,
    pid_path: str,
) -> str | None:
    """Stop sandbox acceptance and wait for the proxy to drain active requests."""

    drain = await sandbox.exec(
        (
            f"if [ ! -s {shlex.quote(pid_path)} ]; then exit 0; fi\n"
            f"read -r pid < {shlex.quote(pid_path)}\n"
            'kill -TERM "$pid" 2>/dev/null || exit 0\n'
            f"for attempt in $(seq 1 {PROVIDER_DRAIN_TIMEOUT_SEC * 10}); do\n"
            '  if ! kill -0 "$pid" 2>/dev/null; then exit 0; fi\n'
            '  if [ -r "/proc/$pid/stat" ] && '
            "[ \"$(awk '{print $3}' /proc/$pid/stat)\" = Z ]; then exit 0; fi\n"
            "  sleep 0.1\n"
            "done\n"
            "exit 124"
        ),
        timeout_sec=PROVIDER_DRAIN_TIMEOUT_SEC + 10,
    )
    if drain.return_code == 0:
        return None
    with contextlib.suppress(Exception):
        await sandbox.exec(
            f"if [ -s {shlex.quote(pid_path)} ]; then "
            f"kill -KILL $(cat {shlex.quote(pid_path)}) 2>/dev/null || true; fi",
            timeout_sec=10,
        )
    return "LiteLLM graceful shutdown timed out before provider requests drained"
