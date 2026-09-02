"""Live stdio through the Bedrock AgentCore interactive shell WebSocket.

``InvokeAgentRuntimeCommand`` (used by ``AgentCoreSandbox.exec``) is one-shot:
each call spawns a fresh bash, runs it to completion, and returns. It cannot
hold the long-lived bidirectional pipe an ACP agent speaks JSON-RPC over. That
is what ``open_shell`` provides — a persistent WebSocket terminal attached to
the *same* runtime session, so the agent started here shares a filesystem with
every ``exec()`` the kernel and verifier make.

The channel is a **PTY**, not a raw pipe: it echoes input, emits bracketed-paste
control sequences, and terminates lines with CRLF. That is the same shape as
``DaytonaPtyProcess``, and this class handles it the same proven way — put the
line discipline into raw/no-echo mode, synchronize on a nonce marker before
handing the terminal to the agent, and strip CR while framing lines.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shlex
import uuid
from typing import Any

from benchflow.sandbox.process._base import (
    _ANSI_CSI_RE_BYTES,
    _ANSI_OSC_RE_BYTES,
    _ENV_KEY_RE,
    LiveProcess,
    _timeout_sec_from_env,
)

logger = logging.getLogger(__name__)

#: Channels the AgentCore shell multiplexes alongside agent stdout. Forwarding
#: these into the ACP stream would let a diagnostic be parsed as JSON-RPC.
_SIDE_CHANNELS = ("STDERR", "STATUS", "CLOSE")

#: Queue sentinel meaning "the frame reader has ended"; see _drain_frames.
_READER_ENDED = object()

_START_MARKER_TIMEOUT_SEC = 180
_READLINE_TIMEOUT_ENV = "BENCHFLOW_AGENTCORE_READLINE_TIMEOUT"
_READLINE_TIMEOUT_DEFAULT_SEC = 900.0
_AGENTCORE_READER_CLOSE_TIMEOUT_SEC = 5.0
_AGENTCORE_SHELL_CLOSE_TIMEOUT_SEC = 15.0


def _consume_cleanup_result(future: asyncio.Future[Any]) -> None:
    with contextlib.suppress(BaseException):
        future.result()


async def _await_cleanup_bounded(awaitable: Any, timeout: float) -> bool:
    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except asyncio.CancelledError:
        task.cancel()
        task.add_done_callback(_consume_cleanup_result)
        raise
    if task in done:
        _consume_cleanup_result(task)
        return True
    task.cancel()
    task.add_done_callback(_consume_cleanup_result)
    await asyncio.sleep(0)
    return False


def _readline_timeout_sec() -> float:
    return _timeout_sec_from_env(_READLINE_TIMEOUT_ENV, _READLINE_TIMEOUT_DEFAULT_SEC)


class AgentCoreProcess(LiveProcess):
    """Live stdin/stdout over an AgentCore ``open_shell`` WebSocket terminal."""

    def __init__(
        self,
        sandbox: Any,
        runtime_arn: str,
        session_id: str,
        region: str,
    ) -> None:
        self._sandbox = sandbox
        self._runtime_arn = runtime_arn
        self._session_id = session_id
        self._region = region
        self._shell: Any = None
        self._reader_task: asyncio.Task[None] | None = None
        self._line_buffer: asyncio.Queue[Any] = asyncio.Queue()
        self._partial = b""
        self._closed = False
        self._failure: BaseException | None = None
        self._reader_done = False

    @classmethod
    def from_sandbox_env(cls, env: Any) -> AgentCoreProcess:
        """Create from a started :class:`AgentCoreSandbox`."""
        runtime_arn = getattr(env, "runtime_arn", None)
        session_id = getattr(env, "runtime_session_id", None)
        if not runtime_arn or not session_id:
            raise RuntimeError("AgentCore sandbox not started")
        return cls(
            sandbox=env,
            runtime_arn=runtime_arn,
            session_id=session_id,
            region=env.region,
        )

    async def _drain_frames(self) -> None:
        """Frame STDOUT payloads into newline-terminated lines.

        Only the STDOUT channel becomes ACP input. The shell multiplexes
        STDERR and lifecycle channels over the same socket, and forwarding
        those verbatim would let a diagnostic line enter the JSON-RPC stream
        and corrupt — or impersonate — protocol traffic.
        """
        try:
            async for frame in self._shell:
                if not self._is_stdout(frame):
                    self._log_non_stdout(frame)
                    continue
                payload = frame.payload
                if isinstance(payload, str):
                    payload = payload.encode()
                if not payload:
                    continue
                self._partial += payload
                while b"\n" in self._partial:
                    line, self._partial = self._partial.split(b"\n", 1)
                    # Bash/readline emits bracketed-paste CSI sequences when a
                    # command takes over the PTY, including a standalone
                    # ``ESC[?2004l`` *after* the startup marker, and shell
                    # prompts emit OSC window-title updates. Those bytes are
                    # terminal state, never ACP JSON-RPC.
                    line = line.replace(b"\r", b"")
                    line = _ANSI_OSC_RE_BYTES.sub(b"", line)
                    line = _ANSI_CSI_RE_BYTES.sub(b"", line)
                    if line:
                        await self._line_buffer.put(line + b"\n")
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._failure = exc
            logger.warning(
                "AgentCore shell reader stopped: %s", self._redact_output(str(exc))
            )
        finally:
            # Wake any waiting readline() right now. Without this sentinel a
            # dead transport surfaces only when the read timeout expires, so an
            # infrastructure disconnect becomes a 15-minute silent hang.
            self._reader_done = True
            with contextlib.suppress(Exception):
                self._line_buffer.put_nowait(_READER_ENDED)

    @staticmethod
    def _is_stdout(frame: Any) -> bool:
        """Whether *frame* carries agent stdout rather than a side channel.

        A frame that carries **no** channel information is treated as agent
        output, so an SDK emitting one undifferentiated stream is not muted.
        A frame that *is* typed must be ``STDOUT`` — an unrecognized typed
        channel is a side channel this code has not learned about yet, and
        letting it through would put non-protocol bytes into JSON-RPC input.
        """
        channel = getattr(frame, "channel", None)
        if channel is None:
            return True
        name = getattr(channel, "name", None)
        if name is None:
            name = str(channel)
        name = str(name).strip().upper()
        if not name:
            return True
        # Enum reprs arrive as "SHELLCHANNEL.STDOUT"; compare the final
        # component *exactly*. Substring membership admitted NOT_STDOUT and
        # STDOUT_METADATA, which contradicts the fail-closed rule even though
        # today's SDK happens not to define such names.
        return name.rsplit(".", 1)[-1] == "STDOUT"

    def _log_non_stdout(self, frame: Any) -> None:
        payload = frame.payload
        if isinstance(payload, bytes):
            payload = payload.decode(errors="replace")
        if payload and payload.strip():
            logger.debug(
                "AgentCore shell %s: %s",
                getattr(frame.channel, "name", frame.channel),
                self._redact_output(payload.strip())[:500],
            )

    async def _write_env_file(self, env: dict[str, str]) -> str:
        """Materialize *env* as a mode-0600 file inside the session container.

        Routed through the sandbox's own ``exec`` (and therefore through the
        canonical base64 env-file wrapper) so secrets never appear as literal
        text typed into the terminal, where the PTY would echo them straight
        back into the agent log.
        """
        invalid = [key for key in env if not _ENV_KEY_RE.match(key)]
        if invalid:
            raise ValueError(
                "Invalid environment variable name(s): " + ", ".join(sorted(invalid))
            )
        remote_path = f"/tmp/.benchflow_agent_env_{uuid.uuid4().hex[:16]}"
        body = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())
        result = await self._sandbox.write_text_file(remote_path, body, mode="600")
        if result is False:
            with contextlib.suppress(Exception):
                await self._sandbox.exec(
                    f"rm -f {shlex.quote(remote_path)}", timeout_sec=30
                )
            raise RuntimeError("Failed to stage AgentCore agent env file")
        return remote_path

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        from bedrock_agentcore.runtime import AgentCoreRuntimeClient

        remote_env_path: str | None = None
        client = AgentCoreRuntimeClient(region=self._region)
        shell = client.open_shell(
            runtime_arn=self._runtime_arn,
            session_id=self._session_id,
        )
        try:
            self._shell = await shell.__aenter__()
            logger.info(
                "AgentCore shell connected (shell_id=%s, session=%s)",
                getattr(self._shell, "shell_id", "?"),
                self._session_id,
            )
            self._reader_task = asyncio.create_task(self._drain_frames())

            # Take the terminal out of cooked mode before any ACP traffic.
            # ACP JSON-RPC frames routinely exceed the 4096-byte canonical-mode
            # line limit, and echo would feed the agent's own output back to it.
            marker = f"__BENCHFLOW_ACP_{uuid.uuid4().hex[:12]}__"
            await self._send(
                "stty raw -echo 2>/dev/null || "
                "stty -echo -icanon min 1 time 0 2>/dev/null || true; "
                f"echo '{marker}'\n"
            )
            await self._await_marker(marker)
            self._clear_buffered_output()
            if env:
                remote_env_path = await self._write_env_file(env)
            launch = self._launch_command(command, remote_env_path, cwd)
            await self._send(launch + "\n")
        except BaseException:
            if remote_env_path:
                with contextlib.suppress(Exception):
                    await self._sandbox.exec(
                        f"rm -f {shlex.quote(remote_env_path)}", timeout_sec=30
                    )
            await self.close()
            raise
        logger.info("AgentCore shell marker seen, agent starting")

    @staticmethod
    def _launch_command(
        command: str, remote_env_path: str | None, cwd: str | None
    ) -> str:
        """Build a subshell launch that cannot strand a staged env file."""
        parts: list[str] = []
        if remote_env_path:
            quoted = shlex.quote(remote_env_path)
            # The subshell is essential: if ``cd`` or sourcing fails, it exits
            # and runs the trap. The long-lived PTY shell itself stays alive.
            parts.append(f"trap 'rm -f {quoted}' EXIT")
        if cwd:
            parts.append(f"cd {shlex.quote(cwd)}")
        if remote_env_path:
            quoted = shlex.quote(remote_env_path)
            parts.append(f". {quoted}")
            parts.append(f"rm -f {quoted}")
            parts.append("trap - EXIT")
        parts.append(f"exec bash -lc {shlex.quote(command)}")
        return "( " + " && ".join(parts) + " )"

    async def _await_marker(self, marker: str) -> None:
        from benchflow.diagnostics import (
            TransportClosedDiagnostic,
            TransportClosedError,
        )

        while True:
            try:
                line = await asyncio.wait_for(
                    self._line_buffer.get(), timeout=_START_MARKER_TIMEOUT_SEC
                )
            except TimeoutError as e:
                msg = (
                    "AgentCore shell: timed out waiting for the start marker "
                    f"(session={self._session_id})"
                )
                raise TransportClosedError(
                    msg,
                    TransportClosedDiagnostic(
                        raw_message=msg,
                        transport_diagnosis="pty_startup_timeout",
                    ),
                ) from e
            if line is _READER_ENDED:
                # The shell died during startup. Without this the sentinel is
                # consumed here (or cleared by the drain below) and the next
                # readline waits out the full read timeout instead.
                failure = (
                    self._redact_output(str(self._failure))
                    if self._failure is not None
                    else "EOF"
                )
                msg = (
                    "AgentCore shell closed before the start marker "
                    f"(session={self._session_id}): {failure}"
                )
                raise TransportClosedError(
                    msg,
                    TransportClosedDiagnostic(
                        raw_message=msg[:500],
                        transport_diagnosis="remote_session_killed",
                    ),
                ) from self._failure
            # The PTY echoes the entire `stty ...; echo '<marker>'` command
            # before echo suppression takes effect. Substring matching accepts
            # that echoed command and releases startup too early, putting the
            # prompt/command noise into ACP's JSON-RPC stream. The real marker
            # is its own line, possibly wrapped in bracketed-paste ANSI codes
            # or an OSC window-title update from the shell prompt.
            cleaned = _ANSI_CSI_RE_BYTES.sub(b"", _ANSI_OSC_RE_BYTES.sub(b"", line))
            text = cleaned.decode(errors="replace").strip()
            if text == marker:
                return

    def _clear_buffered_output(self) -> None:
        """Drop pre-agent terminal noise, but never the end sentinel.

        Discarding it would strand a later readline for the whole read timeout
        even though the transport is already known to be gone.
        """
        self._partial = b""
        saw_end = False
        while not self._line_buffer.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                if self._line_buffer.get_nowait() is _READER_ENDED:
                    saw_end = True
        if saw_end or self._reader_done:
            with contextlib.suppress(Exception):
                self._line_buffer.put_nowait(_READER_ENDED)

    async def readline(self) -> bytes:
        from benchflow.diagnostics import (
            TransportClosedDiagnostic,
            TransportClosedError,
        )

        def _closed(msg: str, diagnosis: str) -> TransportClosedError:
            return TransportClosedError(
                msg,
                TransportClosedDiagnostic(
                    raw_message=msg[:500], transport_diagnosis=diagnosis
                ),
            )

        if self._closed:
            raise _closed("AgentCore shell closed", "pty_error")
        timeout = _readline_timeout_sec()
        try:
            line = await asyncio.wait_for(self._line_buffer.get(), timeout=timeout)
        except TimeoutError as e:
            raise _closed(
                f"AgentCore shell readline timeout ({timeout:g}s)", "pty_error"
            ) from e
        if line is _READER_ENDED:
            # The reader finished — either an error or a clean EOF. Both mean
            # the transport is gone, and both must surface now rather than at
            # the read timeout.
            if self._failure is not None:
                failure = self._redact_output(str(self._failure))
                raise _closed(
                    f"AgentCore shell transport failed: {failure}",
                    "remote_session_killed",
                ) from self._failure
            raise _closed(
                "AgentCore shell closed by the remote session",
                "remote_session_killed",
            )
        return line

    def _closed_error(self, message: str):
        from benchflow.diagnostics import (
            TransportClosedDiagnostic,
            TransportClosedError,
        )

        return TransportClosedError(
            message,
            TransportClosedDiagnostic(
                raw_message=message[:500],
                transport_diagnosis="remote_session_killed",
            ),
        )

    async def _send(self, data: str) -> None:
        if not self.is_running:
            # Half-close guard: ContainerTransport.send() does not pre-check
            # liveness, so without this an ACP request is handed to a socket
            # whose read side is already gone and the reply can never arrive.
            raise self._closed_error(
                "AgentCore shell is not running (reader ended); refusing to send "
                "to a half-closed transport"
            )
        await self._shell.send(data)

    async def writeline(self, data: str) -> None:
        await self._send(data + "\n")

    async def close(self) -> None:
        self._closed = True
        if self._reader_task is not None:
            reader_task = self._reader_task
            self._reader_task = None
            reader_task.cancel()
            if not await _await_cleanup_bounded(
                reader_task, _AGENTCORE_READER_CLOSE_TIMEOUT_SEC
            ):
                logger.warning("AgentCore shell reader did not stop before deadline")
        if self._shell is not None:
            shell = self._shell
            self._shell = None
            closed = await _await_cleanup_bounded(
                shell.close(), _AGENTCORE_SHELL_CLOSE_TIMEOUT_SEC
            )
            if closed:
                logger.info("AgentCore shell terminated")
            else:
                logger.warning("AgentCore shell close did not finish before deadline")

    @property
    def is_running(self) -> bool:
        """Liveness includes the reader: a dead reader is a dead transport."""
        return (
            self._shell is not None
            and not self._closed
            and not self._reader_done
            and self._failure is None
        )
