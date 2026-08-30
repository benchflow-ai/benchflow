"""Transport-agnostic ``LiveProcess`` contract and its subprocess implementation.

``LiveProcess`` is the bidirectional line pipe an ACP agent speaks over. It
deliberately declares *only* the four operations the ACP transport needs
(``start``/``readline``/``writeline``/``close``) plus ``is_running``, with no
assumption about what carries the bytes.

:class:`SubprocessLiveProcess` supplies the local-``asyncio``-subprocess
implementation shared by the Docker, Apple Container, and Daytona-SSH
backends: for those three the pipe *is* a child process's stdio.

The split exists because two transports are not subprocess-backed at all —
``DaytonaPtyProcess`` (Daytona PTY WebSocket) and ``AgentCoreProcess``
(Bedrock AgentCore shell WebSocket). Before the split they inherited
subprocess semantics they could not honour and had to neutralize the base
class with ``_process = None  # Not used`` plus a full override of
``readline``/``writeline``/``close``. One such escape hatch is a wart; two
means the base class was modelling the wrong thing.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_BUFFER_LIMIT = 10 * 1024 * 1024  # 10MB readline buffer
_DIAG_TRUNCATE = 2000  # max chars for diagnostic stderr in error messages
_STDERR_TAIL_LIMIT = 64 * 1024  # bounded stderr retained for rollout diagnostics
_STDERR_DRAIN_TIMEOUT_SEC = 2
_BOOTSTRAP_DONE = "__BENCHFLOW_BOOTSTRAP_DONE__"
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Terminal control sequences a PTY-backed transport can interleave with
# protocol output: ECMA-48 CSI (parameter bytes 0x30-0x3F, intermediate
# bytes 0x20-0x2F, one final byte 0x40-0x7E) and OSC (BEL- or ST-terminated,
# e.g. the ``\x1b]0;<title>\x07`` window-title update shell prompts emit).
# One canonical pair — the AgentCore stdout chunker and the ACP container
# transport must not drift apart on what counts as terminal noise.
_ANSI_CSI_PATTERN = r"\x1b\[[0-?]*[ -/]*[@-~]"
_ANSI_OSC_PATTERN = r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
_ANSI_CSI_RE = re.compile(_ANSI_CSI_PATTERN)
_ANSI_OSC_RE = re.compile(_ANSI_OSC_PATTERN)
_ANSI_CSI_RE_BYTES = re.compile(_ANSI_CSI_PATTERN.encode())
_ANSI_OSC_RE_BYTES = re.compile(_ANSI_OSC_PATTERN.encode())


def _timeout_sec_from_env(env_var: str, default: float) -> float:
    """Read a positive seconds value from *env_var*, else *default*.

    The one parser for operator-tunable timeout knobs on this plane (PTY
    readline timeouts, the ACP handshake window). Read at use time so
    long-lived processes and tests see env changes. Unset or empty means
    the default; non-numeric, non-positive, or NaN values warn and fall
    back rather than disabling or breaking the guarded wait.
    """
    raw = os.environ.get(env_var)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not value > 0:  # also rejects NaN
        logger.warning("Invalid %s=%r; using default %.0fs", env_var, raw, default)
        return default
    return value


async def drain_oversized_line(reader: asyncio.StreamReader) -> int:
    """Drain an oversized line from *reader* after a buffer overflow.

    Clears the internal buffer and attempts to skip ahead to the next
    newline.  Returns the number of bytes discarded.
    """
    # Reach into asyncio.StreamReader internals to clear the buffer after
    # a LimitOverrunError. There's no public API for this; the private
    # attributes are stable across Python 3.10+.
    skipped = len(reader._buffer)  # ty: ignore[unresolved-attribute]
    reader._buffer.clear()  # ty: ignore[unresolved-attribute]
    reader._maybe_resume_transport()  # ty: ignore[unresolved-attribute]
    try:
        await asyncio.wait_for(reader.readuntil(b"\n"), timeout=5)
    except Exception:
        logger.debug("Could not find next newline after buffer overflow")
    return skipped


class LiveProcess(ABC):
    """Abstract live stdin/stdout connection to a process inside a sandbox.

    Implementations carry the bytes however their backend allows — a local
    child process's stdio, an SSH pipe, or a WebSocket terminal. Nothing in
    this contract presumes a subprocess; backends that *are* subprocess-backed
    should extend :class:`SubprocessLiveProcess` instead of implementing the
    read/write/close trio by hand.
    """

    @abstractmethod
    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """Start the process with live stdin/stdout."""

    @abstractmethod
    async def readline(self) -> bytes:
        """Read one line from the process's stdout."""

    @abstractmethod
    async def writeline(self, data: str) -> None:
        """Write one line to the process's stdin."""

    @abstractmethod
    async def close(self) -> None:
        """Terminate the process (idempotent — safe to call after death)."""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Whether the transport is still usable."""


class SubprocessLiveProcess(LiveProcess):
    """A :class:`LiveProcess` whose pipe is a local ``asyncio`` subprocess.

    Subclasses only implement ``start`` — spawning whatever CLI reaches into
    their sandbox (``docker compose exec -i``, ``container exec -i``, ``ssh``)
    and assigning the result to ``self._process``. Everything else is shared.
    """

    _process: asyncio.subprocess.Process | None = None

    def _set_process(self, process: asyncio.subprocess.Process) -> None:
        """Store a subprocess and drain stderr without blocking its stdout pipe."""
        self._process = process
        self._stderr_tail = bytearray()
        self._stderr_task = (
            asyncio.create_task(self._drain_stderr(process.stderr))
            if isinstance(process.stderr, asyncio.StreamReader)
            else None
        )

    async def _drain_stderr(self, stderr: asyncio.StreamReader | None) -> None:
        if stderr is None:
            return
        while chunk := await stderr.read(8192):
            self._stderr_tail.extend(chunk)
            if len(self._stderr_tail) > _STDERR_TAIL_LIMIT:
                del self._stderr_tail[:-_STDERR_TAIL_LIMIT]

    async def _finish_stderr_drain(self, *, cancel_on_timeout: bool) -> None:
        """Bound the stderr drain without leaking its transport failures."""
        stderr_task = getattr(self, "_stderr_task", None)
        if not stderr_task or stderr_task.cancelled():
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(stderr_task), timeout=_STDERR_DRAIN_TIMEOUT_SEC
            )
        except asyncio.CancelledError:
            if not stderr_task.cancelled():
                raise
        except TimeoutError:
            if cancel_on_timeout:
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stderr_task
        except Exception:
            logger.debug("Could not finish draining subprocess stderr", exc_info=True)

    @property
    def stderr_tail(self) -> str:
        """Bounded stderr captured while the subprocess was alive."""
        return bytes(getattr(self, "_stderr_tail", b"")).decode(errors="replace")

    async def readline(self) -> bytes:
        """Read one line from stdout."""
        if not self._process or not self._process.stdout:
            raise RuntimeError("Process not started")
        try:
            line = await self._process.stdout.readline()
        except (ValueError, asyncio.LimitOverrunError) as e:
            # Buffer overflow — line exceeds _BUFFER_LIMIT.
            skipped = await drain_oversized_line(self._process.stdout)
            logger.warning(f"Skipped oversized line ({skipped} bytes): {e}")
            # Return empty line — caller will retry readline
            return b""
        if not line:
            stderr_task = getattr(self, "_stderr_task", None)
            if stderr_task:
                await self._finish_stderr_drain(cancel_on_timeout=False)
                stderr_text = self.stderr_tail.strip()
            else:
                stderr_text = ""
            if not stderr_task and self._process and self._process.stderr:
                try:
                    stderr_bytes = await asyncio.wait_for(
                        self._process.stderr.read(8192), timeout=2
                    )
                    stderr_text = stderr_bytes.decode(errors="replace").strip()
                except Exception:
                    logger.debug("Could not read stderr from closed process")
            rc = self._process.returncode if self._process else None
            # Diagnose: rc=None with closed stdout usually means the *transport*
            # died (SSH/Daytona idle sleep, container killed) while the local
            # subprocess wrapper is still alive. rc set means the local process
            # actually exited. Surfacing the distinction makes the failure
            # actionable instead of cryptic.
            pid = self._process.pid if self._process else None
            if rc is None:
                hint = (
                    f"Local subprocess (pid={pid}) is still alive but its "
                    "stdout/transport closed. This usually means the remote "
                    "container or SSH session was killed (e.g. Daytona idle "
                    "sleep, agent hung with no output)."
                )
                diagnosis = "remote_session_killed"
            else:
                hint = f"Local subprocess exited with rc={rc} before stdout closed."
                diagnosis = "process_exited"
            msg = f"Process closed stdout (rc={rc}): {hint}"
            stderr_snippet: str | None = None
            if stderr_text:
                from benchflow.trajectories.redaction import redact_trajectory_text

                stderr_snippet = redact_trajectory_text(stderr_text)[:_DIAG_TRUNCATE]
                msg += f"\nstderr: {stderr_snippet}"
            # Raise a structured TransportClosedError at the source so
            # downstream code (rollout._build_rollout_result) doesn't have
            # to regex-parse the human-readable message back into fields
            # (issue #504).
            from benchflow.diagnostics import (
                TransportClosedDiagnostic,
                TransportClosedError,
            )

            raise TransportClosedError(
                msg,
                TransportClosedDiagnostic(
                    raw_message=msg[:500],
                    process_exit_code=rc,
                    process_pid=pid,
                    transport_diagnosis=diagnosis,
                    stderr_snippet=stderr_snippet,
                ),
            )
        return line

    async def writeline(self, data: str) -> None:
        """Write one line to stdin."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("Process not started")
        self._process.stdin.write((data + "\n").encode())
        await self._process.stdin.drain()

    async def close(self) -> None:
        """Terminate the process (idempotent — safe to call after process death)."""
        if self._process:
            if self._process.stdin:
                with contextlib.suppress(OSError):  # already closed
                    self._process.stdin.close()
            if self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5)
                except TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            await self._finish_stderr_drain(cancel_on_timeout=True)
            logger.info("Process terminated")

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None
