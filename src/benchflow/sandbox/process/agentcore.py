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

from benchflow.sandbox.process._base import LiveProcess

logger = logging.getLogger(__name__)

_START_MARKER_TIMEOUT_SEC = 180
_READLINE_TIMEOUT_ENV = "BENCHFLOW_AGENTCORE_READLINE_TIMEOUT"
_READLINE_TIMEOUT_DEFAULT_SEC = 900.0


def _readline_timeout_sec() -> float:
    import os

    value = os.environ.get(_READLINE_TIMEOUT_ENV)
    if value is None:
        return _READLINE_TIMEOUT_DEFAULT_SEC
    try:
        timeout = float(value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default %.0fs",
            _READLINE_TIMEOUT_ENV,
            value,
            _READLINE_TIMEOUT_DEFAULT_SEC,
        )
        return _READLINE_TIMEOUT_DEFAULT_SEC
    return timeout if timeout > 0 else _READLINE_TIMEOUT_DEFAULT_SEC


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
        self._line_buffer: asyncio.Queue[bytes] = asyncio.Queue()
        self._partial = b""
        self._closed = False
        self._failure: BaseException | None = None

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
        """Frame WebSocket payloads into newline-terminated lines."""
        try:
            async for frame in self._shell:
                payload = frame.payload
                if isinstance(payload, str):
                    payload = payload.encode()
                if not payload:
                    continue
                self._partial += payload
                while b"\n" in self._partial:
                    line, self._partial = self._partial.split(b"\n", 1)
                    await self._line_buffer.put(line.replace(b"\r", b"") + b"\n")
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            # Record rather than raise: this runs in a background task whose
            # exception would otherwise be swallowed, leaving readline() to
            # hang until its timeout instead of reporting the real cause.
            self._failure = exc
            logger.warning("AgentCore shell reader stopped: %s", exc)

    async def _write_env_file(self, env: dict[str, str]) -> str:
        """Materialize *env* as a mode-0600 file inside the session container.

        Routed through the sandbox's own ``exec`` (and therefore through the
        canonical base64 env-file wrapper) so secrets never appear as literal
        text typed into the terminal, where the PTY would echo them straight
        back into the agent log.
        """
        remote_path = f"/tmp/.benchflow_agent_env_{uuid.uuid4().hex[:16]}"
        body = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())
        result = await self._sandbox.write_text_file(remote_path, body, mode="600")
        if result is False:
            raise RuntimeError("Failed to stage AgentCore agent env file")
        return remote_path

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        from bedrock_agentcore.runtime import AgentCoreRuntimeClient

        parts: list[str] = []
        if cwd:
            parts.append(f"cd {shlex.quote(cwd)}")
        if env:
            remote_env_path = await self._write_env_file(env)
            quoted = shlex.quote(remote_env_path)
            parts.append(f". {quoted}")
            parts.append(f"rm -f {quoted}")
        parts.append(f"exec bash -lc {shlex.quote(command)}")
        launch = " && ".join(parts)

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
            await self._shell.send(
                "stty raw -echo 2>/dev/null || "
                "stty -echo -icanon min 1 time 0 2>/dev/null || true; "
                f"echo '{marker}'\n"
            )
            await self._await_marker(marker)
            self._clear_buffered_output()
            await self._shell.send(launch + "\n")
        except BaseException:
            await self.close()
            raise
        logger.info("AgentCore shell marker seen, agent starting")

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
            if marker in line.decode(errors="replace"):
                return

    def _clear_buffered_output(self) -> None:
        self._partial = b""
        while not self._line_buffer.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._line_buffer.get_nowait()

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
            return await asyncio.wait_for(self._line_buffer.get(), timeout=timeout)
        except TimeoutError as e:
            # A dead reader task means the WebSocket dropped; report that rather
            # than the timeout, which would misattribute an infra failure to a
            # silent agent.
            if self._failure is not None:
                raise _closed(
                    f"AgentCore shell transport failed: {self._failure}",
                    "remote_session_killed",
                ) from self._failure
            raise _closed(
                f"AgentCore shell readline timeout ({timeout:g}s)", "pty_error"
            ) from e

    async def writeline(self, data: str) -> None:
        if not self._shell or self._closed:
            raise RuntimeError("AgentCore shell not started")
        await self._shell.send(data + "\n")

    async def close(self) -> None:
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task
            self._reader_task = None
        if self._shell is not None:
            with contextlib.suppress(Exception):
                await self._shell.close()
            self._shell = None
            logger.info("AgentCore shell terminated")

    @property
    def is_running(self) -> bool:
        return self._shell is not None and not self._closed
