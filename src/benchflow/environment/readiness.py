"""Readiness probing — the framework's gate before the agent runs.

Readiness is a framework guarantee, never the benchmark's burden
(architecture.md, design principle 8).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import httpx

from benchflow.environment.protocol import ReadinessProbe


async def _default_http_check(url: str) -> bool:
    """True if the URL returns a 2xx status."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
            return 200 <= resp.status_code < 300
    except Exception:
        return False


async def _default_tcp_check(port: int) -> bool:
    """True if a TCP connection to localhost:port succeeds."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection("localhost", port), timeout=2.0
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def wait_for_readiness(
    *,
    http: list[str],
    tcp: list[int],
    timeout_sec: int,
    poll_interval: float = 1.0,
    _http_check: Callable[[str], Awaitable[bool]] | None = None,
    _tcp_check: Callable[[int], Awaitable[bool]] | None = None,
) -> ReadinessProbe:
    """Poll HTTP and TCP probes until all pass or the timeout elapses.

    The ``_http_check`` / ``_tcp_check`` parameters are injection seams for
    tests; production callers omit them.
    """
    http_check = _http_check or _default_http_check
    tcp_check = _tcp_check or _default_tcp_check
    checked = [*http, *(f"tcp://localhost:{p}" for p in tcp)]

    if not http and not tcp:
        return ReadinessProbe(ready=True, checked=[], error=None)

    deadline = time.monotonic() + timeout_sec
    while True:
        http_ok = all([await http_check(u) for u in http])
        tcp_ok = all([await tcp_check(p) for p in tcp])
        if http_ok and tcp_ok:
            return ReadinessProbe(ready=True, checked=checked, error=None)
        if time.monotonic() >= deadline:
            return ReadinessProbe(
                ready=False,
                checked=checked,
                error=f"readiness timed out after {timeout_sec}s",
            )
        await asyncio.sleep(poll_interval)
