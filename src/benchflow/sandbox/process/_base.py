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
import shlex
import signal
import uuid
from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_BUFFER_LIMIT = 10 * 1024 * 1024  # 10MB readline buffer
_DIAG_TRUNCATE = 2000  # max chars for diagnostic stderr in error messages
_STDERR_TAIL_LIMIT = 64 * 1024  # bounded stderr retained for rollout diagnostics
_STDERR_DRAIN_TIMEOUT_SEC = 2
_PROCESS_TREE_TERM_TIMEOUT_SEC = 5.0
_PROCESS_TREE_KILL_TIMEOUT_SEC = 5.0
_DEFAULT_PROCESS_CLOSE_TIMEOUT_SEC = 35.0
_PROCESS_TREE_POLL_INTERVAL_SEC = 0.05
_BOOTSTRAP_DONE = "__BENCHFLOW_BOOTSTRAP_DONE__"
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _consume_task_result(task: asyncio.Task[None]) -> None:
    """Retrieve a detached cleanup task result without surfacing it later."""
    with contextlib.suppress(BaseException):
        task.result()


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


@dataclass(frozen=True, slots=True)
class _ProcessTreeTermination:
    graceful_termination: bool
    force_kill_required: bool
    process_tree_stopped: bool


_UNKNOWN_PROCESS_TREE_TERMINATION = _ProcessTreeTermination(
    graceful_termination=False,
    force_kill_required=False,
    process_tree_stopped=False,
)


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

    def set_output_redaction_values(self, values: Collection[str]) -> None:
        """Privately retain exact values that must never cross output boundaries."""
        self._output_redaction_values = tuple(value for value in values if value)

    def _redact_output(self, text: str) -> str:
        from benchflow.trajectories.types import (
            redact_trajectory_text_with_exact_values,
        )

        return redact_trajectory_text_with_exact_values(
            text, getattr(self, "_output_redaction_values", ())
        )

    async def close_stdin(self) -> bool:
        """Close the ACP input stream when the backend exposes it independently."""
        return False

    async def wait_for_session_closed(self, timeout: float) -> bool:
        """Wait for an independently requested stream close to end the adapter."""
        return False

    async def terminate_process_tree(self) -> _ProcessTreeTermination:
        """Bound unsupported close paths without claiming descendant liveness."""
        close_task = asyncio.create_task(self.close())
        try:
            done, _ = await asyncio.wait(
                {close_task}, timeout=_DEFAULT_PROCESS_CLOSE_TIMEOUT_SEC
            )
        except asyncio.CancelledError:
            close_task.cancel()
            close_task.add_done_callback(_consume_task_result)
            raise
        if close_task in done:
            await close_task
        else:
            close_task.cancel()
            close_task.add_done_callback(_consume_task_result)
            await asyncio.sleep(0)
            logger.warning(
                "%s close exceeded %.0fs; descendant liveness remains unknown",
                type(self).__name__,
                _DEFAULT_PROCESS_CLOSE_TIMEOUT_SEC,
            )
        return _UNKNOWN_PROCESS_TREE_TERMINATION

    async def process_tree_stopped(self) -> bool:
        """Whether all descendants are proven stopped.

        Non-subprocess transports cannot infer this from a closed pipe or
        WebSocket. They remain unsafe unless a backend overrides the method
        with a real liveness check.
        """
        return False

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

    def _set_process(
        self,
        process: asyncio.subprocess.Process,
        *,
        owns_process_group: bool = False,
    ) -> None:
        """Store a subprocess and the process-group identity created at launch."""
        self._process = process
        self._process_group_id = process.pid if owns_process_group else None
        self._stdin_closed = False
        self._termination_status: _ProcessTreeTermination | None = None
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
                stderr_snippet = self._redact_output(stderr_text)[:_DIAG_TRUNCATE]
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

    async def close_stdin(self) -> bool:
        """Close the ACP input stream without assuming test doubles are writers."""
        process = self._process
        if process is None:
            return False
        stdin = process.stdin
        if stdin is None:
            return process.returncode is not None
        is_closing = getattr(type(stdin), "is_closing", None)
        if getattr(self, "_stdin_closed", False) or (
            callable(is_closing) and is_closing(stdin) is True
        ):
            self._stdin_closed = True
            return True
        try:
            stdin.close()
            wait_closed = getattr(type(stdin), "wait_closed", None)
            if callable(wait_closed):
                await wait_closed(stdin)
        except OSError:
            return process.returncode is not None
        self._stdin_closed = True
        return True

    async def wait_for_session_closed(self, timeout: float) -> bool:
        """Wait briefly for an ACP adapter to honor EOF before signaling it."""
        process = self._process
        if process is None:
            return False
        if process.returncode is not None:
            return True
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout=timeout)
        except TimeoutError:
            return False
        return True

    @property
    def termination_status(self) -> _ProcessTreeTermination:
        return (
            getattr(self, "_termination_status", None)
            or _UNKNOWN_PROCESS_TREE_TERMINATION
        )

    def _process_group_is_alive(self) -> bool:
        process_group_id = getattr(self, "_process_group_id", None)
        if process_group_id is None:
            return False
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    async def process_tree_stopped(self) -> bool:
        process_group_id = getattr(self, "_process_group_id", None)
        if process_group_id is None:
            return self.termination_status.process_tree_stopped
        stopped = not self._process_group_is_alive()
        if stopped:
            self._process_group_id = None
        return stopped

    async def _wait_for_process_tree(self, timeout: float) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self._process_group_is_alive():
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(_PROCESS_TREE_POLL_INTERVAL_SEC, remaining))
        self._process_group_id = None
        return True

    def _signal_process_group(self, sig: signal.Signals) -> bool:
        process_group_id = getattr(self, "_process_group_id", None)
        if process_group_id is None:
            return False
        if process_group_id == os.getpgrp():
            logger.error("Refusing to signal BenchFlow's own process group")
            return False
        try:
            os.killpg(process_group_id, sig)
        except ProcessLookupError:
            return False
        except (OSError, PermissionError):
            logger.warning(
                "Could not signal owned process group %s with %s",
                process_group_id,
                sig.name,
                exc_info=True,
            )
            return False
        return True

    async def _close_unowned_process(self) -> None:
        """Preserve legacy leader cleanup without claiming descendant safety."""
        if not self._process or self._process.returncode is not None:
            return
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5)
        except TimeoutError:
            self._process.kill()
            await self._process.wait()

    async def terminate_process_tree(self) -> _ProcessTreeTermination:
        """Stop the exact process group retained when this transport launched."""
        existing = getattr(self, "_termination_status", None)
        if existing is not None:
            return existing
        await self.close_stdin()

        if getattr(self, "_process_group_id", None) is None:
            await self._close_unowned_process()
            await self._finish_stderr_drain(cancel_on_timeout=True)
            self._termination_status = _UNKNOWN_PROCESS_TREE_TERMINATION
            return self._termination_status

        graceful = False
        force_required = False
        stopped = await self.process_tree_stopped()
        if not stopped:
            term_sent = self._signal_process_group(signal.SIGTERM)
            stopped = await self._wait_for_process_tree(_PROCESS_TREE_TERM_TIMEOUT_SEC)
            graceful = term_sent and stopped
        if not stopped:
            force_required = True
            self._signal_process_group(signal.SIGKILL)
            stopped = await self._wait_for_process_tree(_PROCESS_TREE_KILL_TIMEOUT_SEC)
        if self._process and self._process.returncode is None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._process.wait(), timeout=0.5)
        await self._finish_stderr_drain(cancel_on_timeout=True)
        self._termination_status = _ProcessTreeTermination(
            graceful_termination=graceful,
            force_kill_required=force_required,
            process_tree_stopped=stopped,
        )
        logger.info(
            "Process tree termination: graceful=%s forced=%s stopped=%s",
            graceful,
            force_required,
            stopped,
        )
        return self._termination_status

    async def close(self) -> None:
        """Terminate the owned process tree (idempotent after the first result)."""
        await self.terminate_process_tree()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None


class _RemoteProcessGroupLiveProcess(SubprocessLiveProcess):
    """Subprocess adapter that owns the exact process group inside its sandbox."""

    _remote_process_group_path: str | None = None

    def _build_remote_process_group_wrapper(self, command: str) -> tuple[str, str]:
        """Build an isolated-group command without publishing identity yet."""
        path = f"/tmp/.benchflow_agent_pgid_{uuid.uuid4().hex}"
        inner = (
            f"umask 077; printf '%s\\n' \"$$\" > {shlex.quote(path)}; "
            f"exec bash -c {shlex.quote(command)}"
        )
        return f"exec setsid bash -c {shlex.quote(inner)}", path

    def _wrap_remote_process_group(self, command: str) -> str:
        """Create an isolated remote group and retain its launch identity."""
        wrapped, path = self._build_remote_process_group_wrapper(command)
        self._remote_process_group_path = path
        self._termination_status = None
        return wrapped

    async def _exec_remote_process_group_command(self, command: str) -> int:
        raise NotImplementedError

    def _remote_process_group_command(self, operation: str) -> str | None:
        path = self._remote_process_group_path
        if path is None:
            return None
        prelude = (
            f"pgid=$(cat {shlex.quote(path)} 2>/dev/null) || exit 2; "
            "case \"$pgid\" in ''|*[!0-9]*) exit 2;; esac; "
        )
        if operation == "check":
            return prelude + 'kill -0 -- "-$pgid" 2>/dev/null'
        if operation in {"TERM", "KILL"}:
            return prelude + f'kill -{operation} -- "-$pgid" 2>/dev/null'
        raise ValueError(f"Unsupported process-group operation: {operation}")

    async def _remote_process_group_alive(self) -> bool | None:
        command = self._remote_process_group_command("check")
        if command is None:
            return None
        try:
            return_code = await self._exec_remote_process_group_command(command)
        except Exception:
            logger.warning(
                "Remote process-group liveness command failed", exc_info=True
            )
            return None
        if return_code == 0:
            return True
        if return_code == 1:
            return False
        return None

    async def _signal_remote_process_group(self, operation: str) -> bool:
        command = self._remote_process_group_command(operation)
        if command is None:
            return False
        try:
            return await self._exec_remote_process_group_command(command) == 0
        except Exception:
            logger.warning(
                "Remote process-group %s command failed", operation, exc_info=True
            )
            return False

    async def _wait_for_remote_process_tree(self, timeout: float) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            try:
                alive = await asyncio.wait_for(
                    self._remote_process_group_alive(),
                    timeout=remaining,
                )
            except TimeoutError:
                return False
            if alive is False:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(_PROCESS_TREE_POLL_INTERVAL_SEC, remaining))

    async def process_tree_stopped(self) -> bool:
        if self._remote_process_group_path is None:
            return self.termination_status.process_tree_stopped
        alive = await self._remote_process_group_alive()
        return alive is False

    async def _cleanup_remote_process_group_identity(self) -> None:
        path = self._remote_process_group_path
        if path is None:
            return
        try:
            await self._exec_remote_process_group_command(f"rm -f {shlex.quote(path)}")
        except Exception:
            logger.warning(
                "Could not remove remote process-group identity %s",
                path,
                exc_info=True,
            )
        self._remote_process_group_path = None

    async def terminate_process_tree(self) -> _ProcessTreeTermination:
        existing = getattr(self, "_termination_status", None)
        if existing is not None:
            return existing

        await self.close_stdin()

        graceful = False
        force_required = False
        identity_known = self._remote_process_group_path is not None
        alive = await self._remote_process_group_alive()
        stopped = alive is False
        if identity_known and not stopped:
            term_sent = await self._signal_remote_process_group("TERM")
            stopped = await self._wait_for_remote_process_tree(
                _PROCESS_TREE_TERM_TIMEOUT_SEC
            )
            graceful = term_sent and stopped
        if identity_known and not stopped:
            force_required = True
            await self._signal_remote_process_group("KILL")
            stopped = await self._wait_for_remote_process_tree(
                _PROCESS_TREE_KILL_TIMEOUT_SEC
            )

        await self._close_unowned_process()
        await self._finish_stderr_drain(cancel_on_timeout=True)
        if stopped:
            await self._cleanup_remote_process_group_identity()
        self._termination_status = _ProcessTreeTermination(
            graceful_termination=graceful,
            force_kill_required=force_required,
            process_tree_stopped=stopped,
        )
        logger.info(
            "Remote process tree termination: graceful=%s forced=%s stopped=%s",
            graceful,
            force_required,
            stopped,
        )
        return self._termination_status
