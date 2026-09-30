"""Live stdio to an agent inside a Hugging Face Sandbox, over an in-sandbox HTTP bridge.

HF Sandboxes have no interactive exec (no PTY, no SSH): ``Sandbox.run`` takes
stdin up front and returns when the command ends. ACP agents need a long-lived
bidirectional pipe, so BenchFlow uploads a small standard-library bridge
(``hf_bridge_server.py``) that runs the agent as its child and serves its
stdin/stdout on ``127.0.0.1:<port>``. The host reaches it through
``Sandbox.proxy_url_for(port)`` with ``Sandbox.proxy_headers``:

- one long ``GET /out?from=N`` stream carries stdout lines (resumed from the
  last line seen if the proxy drops it; heartbeats keep it open while idle);
- each ``POST /in?seq=K`` writes one stdin line (``seq`` makes retries safe).

Streaming HTTP was chosen over WebSocket: the Sept 30 probe found a WebSocket
through the sandbox proxy needs ``compression=None`` and a hand-rolled
standard-library WebSocket server timed out, while chunked HTTP worked with
30-40 ms round trips and 200 KB lines. It also needs no client dependency beyond
``httpx``, which ``huggingface_hub`` already requires.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import shlex
import uuid
from pathlib import Path
from typing import Any

from benchflow.sandbox.process import LiveProcess

logger = logging.getLogger(__name__)

BRIDGE_DIR = "/tmp/.benchflow-bridge"
_READLINE_TIMEOUT_ENV = "BENCHFLOW_HF_BRIDGE_READLINE_TIMEOUT"
_READLINE_TIMEOUT_DEFAULT_SEC = 900.0
_READY_TIMEOUT_SEC = 60.0
_MAX_STREAM_RECONNECTS = 8
_HEARTBEAT_SEC = 5.0
_STREAM_READ_TIMEOUT_SEC = 60.0
_STDERR_TAIL_BYTES = 4000

# Python for the bridge: the image's own, else the uv-managed interpreter that the
# sandbox LiteLLM install leaves behind, else install one with uv.
_FIND_PYTHON = r"""
export PATH="$HOME/.local/bin:$PATH"
for c in python3 python; do
  p="$(command -v "$c" 2>/dev/null || true)"
  if [ -n "$p" ] && "$p" -c 'import http.server, json, subprocess' >/dev/null 2>&1; then echo "$p"; exit 0; fi
done
for p in /tmp/benchflow-litellm/*/venv/bin/python; do
  if [ -x "$p" ]; then echo "$p"; exit 0; fi
done
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
fi
if command -v uv >/dev/null 2>&1; then
  uv python install 3.12 >/dev/null 2>&1 || true
  p="$(uv python find 3.12 2>/dev/null || true)"
  if [ -n "$p" ]; then echo "$p"; exit 0; fi
fi
exit 1
"""


def _readline_timeout_sec() -> float:
    raw = os.environ.get(_READLINE_TIMEOUT_ENV)
    if not raw:
        return _READLINE_TIMEOUT_DEFAULT_SEC
    try:
        value = float(raw)
    except ValueError:
        return _READLINE_TIMEOUT_DEFAULT_SEC
    return value if value > 0 else _READLINE_TIMEOUT_DEFAULT_SEC


def bridge_source() -> str:
    return (Path(__file__).with_name("hf_bridge_server.py")).read_text()


def _transport_closed(msg: str, diagnosis: str, **extra: Any) -> Exception:
    from benchflow.diagnostics import TransportClosedDiagnostic, TransportClosedError

    return TransportClosedError(
        msg,
        TransportClosedDiagnostic(
            raw_message=msg[:500], transport_diagnosis=diagnosis, **extra
        ),
    )


class _Closed:
    """Queue sentinel: the stream ended; ``error`` is the exception to raise."""

    def __init__(self, error: Exception) -> None:
        self.error = error


class HFBridgeProcess(LiveProcess):
    """LiveProcess for an agent inside a Hugging Face Sandbox (see module docstring)."""

    _process = None  # not a local subprocess; readline/writeline/close are overridden
    bridge_dir = BRIDGE_DIR

    def __init__(self, sandbox: Any, *, http_client: Any = None) -> None:
        self._sandbox = sandbox
        self._client = http_client
        self._owns_client = http_client is None
        self._base: str | None = None
        self._bg: Any = None
        self._reader: asyncio.Task | None = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._next_line = 0
        self._seq = 0
        self._write_lock = asyncio.Lock()
        self._closed = False
        self._id = uuid.uuid4().hex[:12]
        self._stderr_path = f"{self.bridge_dir}/{self._id}.stderr"
        self._config_path = f"{self.bridge_dir}/{self._id}.json"

    @classmethod
    async def from_sandbox_env(cls, env: Any) -> HFBridgeProcess:
        sandbox = getattr(env, "_sandbox", None)
        if sandbox is None:
            raise RuntimeError("HF sandbox not started")
        return cls(sandbox)

    # ------------------------------------------------------------------ helpers

    def _headers(self) -> dict[str, str]:
        return dict(self._sandbox.proxy_headers or {})

    async def _run(self, argv: list[str], **kwargs: Any) -> Any:
        return await asyncio.to_thread(self._sandbox.run, argv, shell=False, **kwargs)

    async def _find_python(self) -> str:
        result = await self._run(["/bin/sh", "-c", _FIND_PYTHON], check=False, timeout=300)
        path = (getattr(result, "stdout", "") or "").strip().splitlines()
        if getattr(result, "exit_code", 1) != 0 or not path:
            raise RuntimeError(
                "HF sandbox bridge needs a Python 3 interpreter and none was found or "
                f"installable: {(getattr(result, 'stderr', '') or '')[-500:]}"
            )
        return path[-1]

    async def _health(self) -> dict[str, Any] | None:
        assert self._client is not None and self._base is not None
        try:
            r = await self._client.get(self._base + "health", headers=self._headers(), timeout=15)
        except Exception:
            return None
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except ValueError:
            return None

    async def _stderr_tail(self) -> str:
        with contextlib.suppress(Exception):
            result = await self._run(
                ["/bin/sh", "-c", f"tail -c {_STDERR_TAIL_BYTES} {shlex.quote(self._stderr_path)} 2>/dev/null || true"],
                check=False,
                timeout=15,
            )
            return (getattr(result, "stdout", "") or "").strip()
        return ""

    # ------------------------------------------------------------------ lifecycle

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        import httpx

        port = random.randint(20000, 39999)
        config = {
            "port": port,
            "argv": ["bash", "-lc", command],
            "stderr": self._stderr_path,
            "heartbeat_sec": _HEARTBEAT_SEC,
            "linger_sec": 120,
        }
        try:
            await asyncio.to_thread(self._sandbox.files.write, f"{self.bridge_dir}/bridge.py", bridge_source())
            await asyncio.to_thread(self._sandbox.files.write, self._config_path, json.dumps(config))
            python = await self._find_python()
            # env goes in the API payload (HTTPS), never on a command line.
            self._bg = await self._run(
                [python, f"{self.bridge_dir}/bridge.py", self._config_path],
                env=env or None,
                cwd=cwd,
                background=True,
            )
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
            self._base = self._sandbox.proxy_url_for(port, "/")
            loop = asyncio.get_running_loop()
            deadline = loop.time() + _READY_TIMEOUT_SEC
            while True:
                health = await self._health()
                if health is not None:
                    break
                if loop.time() > deadline:
                    raise _transport_closed(
                        f"HF sandbox bridge did not answer on port {port} within "
                        f"{_READY_TIMEOUT_SEC:.0f}s; stderr: {await self._stderr_tail()}",
                        "bridge_startup_timeout",
                    )
                await asyncio.sleep(0.5)
            logger.info("HFBridgeProcess: bridge up (port %s, child pid %s)", port, health.get("pid"))
            self._reader = asyncio.create_task(self._read_stream(), name=f"hf-bridge-{self._id}")
        except BaseException:
            await self.close()
            raise

    async def _read_stream(self) -> None:
        import httpx

        failures = 0
        timeout = httpx.Timeout(connect=30.0, read=_STREAM_READ_TIMEOUT_SEC, write=30.0, pool=30.0)
        while not self._closed:
            try:
                async with self._client.stream(
                    "GET",
                    self._base + "out",
                    params={"from": self._next_line},
                    headers=self._headers(),
                    timeout=timeout,
                ) as response:
                    if response.status_code != 200:
                        raise httpx.HTTPStatusError(
                            f"bridge stream HTTP {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    failures = 0
                    async for line in response.aiter_lines():
                        if not line:
                            continue  # heartbeat
                        self._next_line += 1
                        await self._queue.put(line.encode() + b"\n")
                # The bridge ends the stream only after the child exited (or on close).
                health = await self._health()
                if health is not None and health.get("exit") is None and not self._closed:
                    continue  # the proxy ended it early; resume
                if health is not None and self._next_line < int(health.get("lines", 0)):
                    continue  # lines left to drain
                rc = None if health is None else health.get("exit")
                tail = await self._stderr_tail()
                msg = f"HF sandbox agent process exited (rc={rc})"
                if tail:
                    msg += f"\nstderr: {tail[-2000:]}"
                await self._queue.put(
                    _Closed(
                        _transport_closed(
                            msg,
                            "process_exited",
                            process_exit_code=rc if isinstance(rc, int) else None,
                            stderr_snippet=tail[-2000:] or None,
                        )
                    )
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # network drop, proxy 5xx, read timeout
                if self._closed:
                    return
                failures += 1
                if failures > _MAX_STREAM_RECONNECTS:
                    await self._queue.put(
                        _Closed(
                            _transport_closed(
                                f"HF sandbox bridge stream failed {failures} times: {exc!r}",
                                "remote_session_killed",
                            )
                        )
                    )
                    return
                logger.warning(
                    "HFBridgeProcess: stream dropped (%r); reconnect %d/%d from line %d",
                    exc,
                    failures,
                    _MAX_STREAM_RECONNECTS,
                    self._next_line,
                )
                await asyncio.sleep(min(2.0**failures, 15.0))

    async def readline(self) -> bytes:
        if self._closed and self._queue.empty():
            raise _transport_closed("HF bridge closed", "pty_error")
        timeout = _readline_timeout_sec()
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except TimeoutError as exc:
            raise _transport_closed(f"HF bridge readline timeout ({timeout:g}s)", "pty_error") from exc
        if isinstance(item, _Closed):
            self._closed = True
            raise item.error
        return item

    async def writeline(self, data: str) -> None:
        import httpx

        if self._closed or self._client is None or self._base is None:
            raise RuntimeError("HF bridge not started")
        async with self._write_lock:
            self._seq += 1
            seq = self._seq
            body = (data + "\n").encode()
            last: Exception | None = None
            for attempt in range(4):
                try:
                    r = await self._client.post(
                        self._base + "in",
                        params={"seq": seq},
                        content=body,
                        headers=self._headers(),
                        timeout=60,
                    )
                except httpx.TransportError as exc:
                    last = exc
                else:
                    if r.status_code == 200:
                        return
                    if r.status_code == 410:
                        raise _transport_closed(
                            f"HF sandbox agent stdin closed: {r.text[:300]}", "process_exited"
                        )
                    last = RuntimeError(f"bridge POST /in HTTP {r.status_code}: {r.text[:300]}")
                await asyncio.sleep(0.5 * (attempt + 1))
            raise _transport_closed(f"HF bridge write failed: {last!r}", "pty_error")

    async def close(self) -> None:
        self._closed = True
        if self._client is not None and self._base is not None:
            with contextlib.suppress(Exception):
                await self._client.post(self._base + "close", headers=self._headers(), timeout=10)
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(BaseException):
                await self._reader
            self._reader = None
        if self._bg is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._bg.kill)
            self._bg = None
        with contextlib.suppress(Exception):
            await self._run(
                ["/bin/sh", "-c", f"rm -f {shlex.quote(self._config_path)}"], check=False, timeout=15
            )
        if self._owns_client and self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None
        logger.info("HFBridgeProcess terminated")

    @property
    def is_running(self) -> bool:
        return self._reader is not None and not self._closed
