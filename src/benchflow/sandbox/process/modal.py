"""Live stdio to a process in a Modal sandbox, over ``Sandbox.exec``.

Modal's SDK gives an exec'd process a writable stdin stream and stdout and
stderr streams (through the sandbox's command router). That is the whole of
what an ACP agent needs, so a Modal sandbox can host one.

The agent starts the way the other live transports start it (C8,
:mod:`benchflow.sandbox._root_launch`). Modal starts root's first shell
itself, as ``docker exec`` does, so the launch is Docker's:

- the argv is root's bash by absolute path (the route's: the protected kit's
  once the sandbox has one) with no profile, rc or ``BASH_ENV`` file (``-p``);
- the process environment Modal is handed holds only the loader and start-up
  variables, set empty (Modal starts a process with the image's ``ENV``): root's
  shell and the system tools it names load nothing they name, and the image's
  values come back from PID 1 for the agent after the privilege drop;
- the agent's environment is the first line of the process's stdin (base64 of
  ``export`` lines), applied with builtins after root's bootstrap pinned the
  route's PATH, as on Docker: no value is on the command line, in a file, or
  in a Modal Secret;
- lines over the ACP limit are rewritten in the sandbox, after the drop (#1138).

stdout is framed into lines here by one reader task, so a read the caller
cancels loses nothing; stderr is drained into a bounded tail for diagnostics.

Modal still cannot enforce ``network_mode: denylist``; that refusal is
separate from this transport and unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import shlex
from typing import Any

from benchflow.sandbox import _transport
from benchflow.sandbox._base import SYSTEM_SHELL_EMPTY_VARS
from benchflow.sandbox._root_launch import (
    inert_task_env,
    launch_bash,
    launch_path,
    launch_route,
    root_launch_argv,
    root_launch_script,
)
from benchflow.sandbox.process._base import (
    _DIAG_TRUNCATE,
    _ENV_KEY_RE,
    _OVERSIZED_LINE_CAP,
    _STDERR_DRAIN_TIMEOUT_SEC,
    _STDERR_TAIL_LIMIT,
    LiveProcess,
)

logger = logging.getLogger(__name__)

#: Bytes per stdin write. Modal buffers at most 2 MiB between drains (16 MiB
#: through the command router), and one ACP message can be larger.
_STDIN_CHUNK = 1024 * 1024
#: How long a closed stdout waits for Modal to report the exit code.
_EXIT_CODE_WAIT_SEC = 5.0
#: Bound on the env line's write at start and on close's EOF.
_STDIN_TIMEOUT_SEC = 30.0
#: Queue sentinel: the stdout reader has ended (EOF or a failed stream).
_STDOUT_ENDED = object()


class ModalProcess(LiveProcess):
    """Live stdin/stdout via ``Sandbox.exec`` (bytes streams, no PTY)."""

    def __init__(self, sandbox: Any, env: Any = None) -> None:
        # The Modal Sandbox handle, and BenchFlow's sandbox whose exec route
        # gives root's launch its shell and PATH (C8); None for a bare
        # process (tests), which keeps the system's.
        self._sandbox = sandbox
        self._bf_env = env
        self._proc: Any = None
        self._lines: asyncio.Queue[Any] = asyncio.Queue()
        self._partial = bytearray()
        self._scanned = 0
        self._skipping = False
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = bytearray()
        self._write_lock = asyncio.Lock()
        self._stdout_failure: BaseException | None = None
        self._stdout_done = False
        # (message, exit code, diagnosis, stderr snippet) once stdout closed.
        self._closed_why: tuple[str, int | None, str, str | None] | None = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @classmethod
    def from_sandbox_env(cls, env: Any) -> ModalProcess:
        """Create from a started ModalSandbox."""
        sandbox = getattr(env, "_sandbox", None)
        if sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        return cls(sandbox, env=env)

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        from modal import Secret

        if self._proc is not None:
            raise RuntimeError("Modal process already started")
        # The launch env's loader and start-up values travel under inert
        # names and become real only after the drop (C8, PT-C8-3).
        env = inert_task_env(env)
        route = await launch_route(self._bf_env)

        # The first stdin line belongs to the transport, before ACP starts:
        # no credential touches a sandbox pathname, an argv or a Modal
        # Secret. Bash's read consumes exactly one line, leaving the ACP
        # bytes after it intact.
        payload = None
        if env:
            if any(not _ENV_KEY_RE.fullmatch(key) for key in env):
                raise ValueError("Invalid Modal environment variable name")
            exports = "".join(
                f"export {key}={shlex.quote(value)}\n" for key, value in env.items()
            )
            payload = base64.b64encode(exports.encode()) + b"\n"
        # Root's shell has the image's loader variables empty and the route's
        # PATH until the privilege drop (C8); the env line is applied there
        # with PATH unset (root_launch_script's env_stdin).
        script = root_launch_script(
            command,
            env_stdin=payload is not None,
            filter_lines=True,
            loader_cleared=True,
            path=launch_path(route),
        )
        # Root's shell is the route's bash by absolute path with -p, as on
        # Docker: neither the image's PATH nor its BASH_ENV or imported
        # functions may select what root runs (C8).
        argv = root_launch_argv(script, launch_bash(route))
        argv = [*argv[:-2], "-p", *argv[-2:]]
        # Modal starts the process with the image's ENV: the loader and
        # start-up names go in empty (``docker exec -e NAME=``), so nothing
        # an image value names loads into root's shell before its first line.
        process_env: dict[str, str | None] = dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, "")
        # The script can hold the agent's command: never log it.
        logger.debug("ModalProcess: starting the agent's exec")
        _transport.sent("Modal", argv)
        try:
            self._proc = await self._sandbox.exec.aio(
                *argv,
                workdir=cwd,
                secrets=[Secret.from_dict(process_env)],
                text=False,
            )
            self._stdout_task = asyncio.create_task(self._read_stdout())
            self._stderr_task = asyncio.create_task(self._drain_stderr())
            if payload is not None:
                await asyncio.wait_for(self._write(payload), timeout=_STDIN_TIMEOUT_SEC)
        except BaseException:
            # Includes cancellation and a partial handoff: there is no file to
            # recover; close the owned transport before returning failure.
            await asyncio.shield(self.close())
            raise
        logger.info("Modal process started")

    # -- stdout -----------------------------------------------------------

    async def _read_stdout(self) -> None:
        """Frame stdout into lines on the queue; the sentinel when it ends."""
        try:
            async for chunk in self._proc.stdout:
                self._frame(chunk.encode() if isinstance(chunk, str) else chunk)
            if self._partial and not self._skipping:
                # The last line, without a newline: as read_whole_line.
                self._lines.put_nowait(bytes(self._partial))
            self._partial.clear()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._stdout_failure = exc
            logger.warning("Modal stdout stream stopped: %r", exc)
        finally:
            # Wake a waiting readline now; a later one finds it again.
            self._stdout_done = True
            self._lines.put_nowait(_STDOUT_ENDED)

    def _frame(self, data: bytes) -> None:
        if not data:
            return
        if self._skipping:
            newline = data.find(b"\n")
            if newline < 0:
                return
            self._skipping = False
            data = data[newline + 1 :]
        self._partial.extend(data)
        while True:
            newline = self._partial.find(b"\n", self._scanned)
            if newline < 0:
                self._scanned = len(self._partial)
                break
            self._lines.put_nowait(bytes(self._partial[: newline + 1]))
            del self._partial[: newline + 1]
            self._scanned = 0
        if len(self._partial) > _OVERSIZED_LINE_CAP:
            # As SubprocessLiveProcess: a line this long is skipped whole.
            logger.warning("Skipped an output line over %d bytes", _OVERSIZED_LINE_CAP)
            self._partial.clear()
            self._scanned = 0
            self._skipping = True

    async def readline(self) -> bytes:
        """Read one line from stdout; a closed stream raises TransportClosedError."""
        if self._proc is None:
            raise RuntimeError("Process not started")
        line = await self._lines.get()
        if line is _STDOUT_ENDED:
            self._lines.put_nowait(_STDOUT_ENDED)
            raise await self._closed_error()
        return line

    async def _exit_code(self) -> int | None:
        """The exit code Modal reports once stdout has closed, else None."""
        proc = self._proc
        if proc is None:
            return None
        try:
            return await asyncio.wait_for(proc.wait.aio(), timeout=_EXIT_CODE_WAIT_SEC)
        except Exception:
            logger.debug("Modal did not report the agent's exit code", exc_info=True)
        with contextlib.suppress(Exception):
            return await asyncio.wait_for(proc.poll.aio(), timeout=_EXIT_CODE_WAIT_SEC)
        return None

    async def _closed_error(self) -> Exception:
        """A structured TransportClosedError for the closed stdout.

        The exit code and stderr are read once; every later read or write
        raises a new error with the same diagnosis.
        """
        from benchflow.diagnostics import (
            TransportClosedDiagnostic,
            TransportClosedError,
        )

        if self._closed_why is None:
            from benchflow.trajectories.types import redact_trajectory_text

            rc = await self._exit_code()
            await self._finish_stderr_drain(cancel_on_timeout=False)
            stderr_text = self.stderr_tail.strip()
            stderr_snippet = (
                redact_trajectory_text(stderr_text)[:_DIAG_TRUNCATE]
                if stderr_text
                else None
            )
            if self._stdout_failure is not None:
                hint = (
                    f"its stdout stream failed ({self._stdout_failure!r}): the "
                    "Modal sandbox or the connection to it was probably lost"
                )
                diagnosis = "remote_session_killed"
            elif rc is None:
                hint = (
                    "Modal reports no exit code: the sandbox was probably ended "
                    "(its timeout, a terminate, or a lost worker)"
                )
                diagnosis = "remote_session_killed"
            else:
                hint = f"the process exited with rc={rc}"
                diagnosis = "process_exited"
            msg = f"Modal process closed stdout (rc={rc}): {hint}."
            if stderr_snippet:
                msg += f"\nstderr: {stderr_snippet}"
            self._closed_why = (msg, rc, diagnosis, stderr_snippet)
        msg, rc, diagnosis, stderr_snippet = self._closed_why
        return TransportClosedError(
            msg,
            TransportClosedDiagnostic(
                raw_message=msg[:500],
                process_exit_code=rc,
                process_pid=None,
                transport_diagnosis=diagnosis,
                stderr_snippet=stderr_snippet,
            ),
        )

    # -- stderr -----------------------------------------------------------

    async def _drain_stderr(self) -> None:
        try:
            async for chunk in self._proc.stderr:
                data = (
                    chunk.encode(errors="replace") if isinstance(chunk, str) else chunk
                )
                self._stderr_tail.extend(data)
                if len(self._stderr_tail) > _STDERR_TAIL_LIMIT:
                    del self._stderr_tail[:-_STDERR_TAIL_LIMIT]
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Could not finish draining Modal stderr", exc_info=True)

    async def _finish_stderr_drain(self, *, cancel_on_timeout: bool) -> None:
        """Bound the stderr drain without leaking its transport failures."""
        task = self._stderr_task
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(task), timeout=_STDERR_DRAIN_TIMEOUT_SEC
            )
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
        except TimeoutError:
            if cancel_on_timeout:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        except Exception:
            logger.debug("Could not finish draining Modal stderr", exc_info=True)

    @property
    def stderr_tail(self) -> str:
        """Bounded stderr captured while the process was alive."""
        return bytes(self._stderr_tail).decode(errors="replace")

    # -- stdin ------------------------------------------------------------

    async def _write(self, data: bytes) -> None:
        # One writer at a time: a drain sends the buffer at the writer's
        # offset, so two interleaved drains would send bytes twice.
        async with self._write_lock:
            for start in range(0, len(data), _STDIN_CHUNK):
                self._proc.stdin.write(data[start : start + _STDIN_CHUNK])
                await self._proc.stdin.drain.aio()

    async def writeline(self, data: str) -> None:
        """Write one line to stdin."""
        if self._proc is None:
            raise RuntimeError("Process not started")
        if self._stdout_done:
            # The agent's stdout is gone: a request could never be answered.
            raise await self._closed_error()
        if self._closed:
            raise RuntimeError("Modal process closed")
        try:
            await self._write((data + "\n").encode())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._stdout_done:
                raise await self._closed_error() from exc
            from benchflow.diagnostics import (
                TransportClosedDiagnostic,
                TransportClosedError,
            )

            msg = f"Modal process stdin write failed: {exc!r}"
            raise TransportClosedError(
                msg,
                TransportClosedDiagnostic(
                    raw_message=msg[:500],
                    transport_diagnosis="remote_session_killed",
                ),
            ) from exc

    # -- lifecycle --------------------------------------------------------

    async def close(self) -> None:
        """Close the agent's stdin and stop reading (idempotent).

        As with ``docker exec`` without a terminal, the agent sees EOF on its
        stdin; whatever still runs ends with the kill before grading or the
        sandbox's stop.
        """
        # Closing rejects new writes immediately, but cleanup has its own
        # lifetime: cancellation of one caller must not orphan the readers
        # or make a later close skip the unfinished EOF/drain.
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_streams())
        await asyncio.shield(self._close_task)

    async def _close_streams(self) -> None:
        proc = self._proc
        try:
            if proc is not None:
                try:
                    await asyncio.wait_for(
                        self._send_eof(proc), timeout=_STDIN_TIMEOUT_SEC
                    )
                except Exception:
                    logger.debug(
                        "Could not close the Modal process's stdin", exc_info=True
                    )
            await self._finish_stderr_drain(cancel_on_timeout=True)
        finally:
            readers = [
                task
                for task in (self._stdout_task, self._stderr_task)
                if task is not None
            ]
            for task in readers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
        if proc is not None:
            logger.info("Modal process closed")

    async def _send_eof(self, proc: Any) -> None:
        async with self._write_lock:
            proc.stdin.write_eof()
            await proc.stdin.drain.aio()

    @property
    def is_running(self) -> bool:
        return self._proc is not None and not self._closed and not self._stdout_done
