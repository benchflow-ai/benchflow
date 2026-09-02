"""ACP transport over a live stdio pipe to a sandbox process."""

import json
import logging
from collections.abc import Collection
from pathlib import Path
from typing import Any, TextIO

from benchflow.sandbox.process import LiveProcess
from benchflow.sandbox.process._base import (
    _ANSI_CSI_RE,
    _ANSI_OSC_RE,
    _UNKNOWN_PROCESS_TREE_TERMINATION,
    _ProcessTreeTermination,
)
from benchflow.trajectories.types import (
    redact_trajectory_obj_with_exact_values,
    redact_trajectory_text_with_exact_values,
)

from .transport import Transport, decode_json_rpc_message

logger = logging.getLogger(__name__)

_ACP_SESSION_CLOSE_GRACE_SEC = 2.0


def _decode_pty_json_rpc_message(text: str) -> dict[str, Any] | None:
    """Decode one JSON-RPC message from a line that may carry PTY noise.

    Daytona's PTY transport gives the agent a real terminal, so the shell
    prompt (with its CSI/OSC escape sequences) can land on the SAME line as
    the agent's first protocol message — observed on BugSwarm-style images
    as ``\\x1b[?2004h\\x1b]0;root@…\\x07root@…# \\x1b[?2004l{"jsonrpc":…}``.
    Whole-line ``json.loads`` fails on that prefix, the response is treated
    as non-protocol noise, and the handshake "times out" with the completed
    initialize JSON sitting in the captured agent log — at ANY timeout.

    Recovery is bounded to where the bug lives: :meth:`ContainerTransport.
    receive` calls this lenient path only until the first successfully
    decoded protocol message per connection (prompt glue is by construction
    a pre-first-message PTY phenomenon), and stdio pipes keep the strict
    whole-line contract in :func:`decode_json_rpc_message`.

    1. fast path — the plain whole-line decode, unchanged;
    2. retry from the first ``{`` when it is not at position 0 (drops a
       glued prompt prefix);
    3. if the line carries ``\\x1b``, strip CSI/OSC sequences (drops escapes
       trailing or embedded around the JSON) and retry from the first ``{``.

    Every retry still goes through :func:`decode_json_rpc_message`, so the
    JSON-RPC 2.0 envelope check keeps rejecting agent log lines that merely
    contain JSON (e.g. ``INFO {"jsonrpc": …}`` with a bad envelope).
    """
    message = decode_json_rpc_message(text)
    if message is not None:
        return message
    if "{" not in text:
        return None
    brace = text.index("{")
    if brace:  # brace == 0 would just repeat the whole-line decode
        message = decode_json_rpc_message(text[brace:])
        if message is not None:
            return message
    if "\x1b" not in text:
        return None
    cleaned = _ANSI_CSI_RE.sub("", _ANSI_OSC_RE.sub("", text))
    if "{" not in cleaned:
        return None
    return decode_json_rpc_message(cleaned[cleaned.index("{") :])


class ContainerTransport(Transport):
    """ACP transport that speaks to an agent running inside a sandbox.

    Uses a LiveProcess (DockerProcess or DaytonaProcess) to maintain a live
    stdin/stdout connection. Non-JSON lines from the agent (debug output,
    errors, warnings) are captured to a log file if agent_log_path is set.
    """

    def __init__(
        self,
        container_process: LiveProcess,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        agent_log_path: Path | None = None,
        runtime_credential_values: Collection[str] = (),
    ):
        self._cp = container_process
        self._command = command
        self._env = env or {}
        self._cwd = cwd
        self._agent_log_path = agent_log_path
        self._agent_log_file: TextIO | None = None
        self._runtime_credential_values = tuple(
            value for value in runtime_credential_values if value
        )
        configure_output_redaction = getattr(
            type(container_process), "set_output_redaction_values", None
        )
        if callable(configure_output_redaction):
            configure_output_redaction(
                container_process, self._runtime_credential_values
            )
        # First-message latch: the lenient PTY-noise decode applies only
        # until the first successfully decoded protocol message (prompt glue
        # is a startup phenomenon). Afterwards the strict whole-line decode
        # rules, so an agent echoing valid protocol envelopes into its logs
        # mid-session can never impersonate protocol traffic.
        self._saw_protocol = False
        self._termination_status = _UNKNOWN_PROCESS_TREE_TERMINATION
        self._session_closed = False
        self._stderr_appended = False

    async def start(self) -> None:
        """Start the agent process inside the sandbox."""
        if self._agent_log_path:
            self._agent_log_path.parent.mkdir(parents=True, exist_ok=True)
            # Clear any stale log from a previous connect attempt. _connect_acp_session
            # reuses the same agent/<agent>.txt path across retries (runtime.py), so a
            # failed attempt that logged a non-protocol warning before raising would
            # otherwise leave stale text behind when a later JSON-RPC-only retry succeeds
            # (which never re-opens the file). Unlink rather than truncating so we keep
            # the lazy-open contract: no empty placeholder for protocol-only runs.
            self._agent_log_path.unlink(missing_ok=True)
        await self._cp.start(
            command=self._command,
            env=self._env,
            cwd=self._cwd,
        )
        logger.info(f"ContainerTransport: agent started ({self._command})")

    async def send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message to the agent."""
        data = json.dumps(message)
        await self._cp.writeline(data)

    async def receive(self) -> dict[str, Any]:
        """Receive a JSON-RPC message from the agent."""
        while True:
            line = await self._cp.readline()
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            if self._saw_protocol:
                message = decode_json_rpc_message(text)
            else:
                message = _decode_pty_json_rpc_message(text)
            if message is not None:
                self._saw_protocol = True
                return message
            safe_text = redact_trajectory_text_with_exact_values(
                text, self._runtime_credential_values
            )
            # Capture non-protocol output (agent debug logs, errors, warnings).
            if self._agent_log_path:
                if self._agent_log_file is None:
                    self._agent_log_file = self._agent_log_path.open("w")
                self._agent_log_file.write(safe_text + "\n")
                self._agent_log_file.flush()
            logger.debug("Non-JSON-RPC from container agent: %s", safe_text[:200])

    def redact_protocol_value(self, value: Any) -> Any:
        """Exact-redact a protocol-derived value only at persistence boundaries."""
        return redact_trajectory_obj_with_exact_values(
            value, self._runtime_credential_values
        )

    @property
    def termination_status(self) -> _ProcessTreeTermination:
        """Last observed process-tree termination outcome."""
        return self._termination_status

    @property
    def session_closed(self) -> bool:
        """Whether the ACP input stream has been closed."""
        return self._session_closed

    async def process_tree_stopped(self) -> bool:
        """Re-check descendant liveness through the owning live process."""
        return await self._cp.process_tree_stopped()

    async def close_session(self) -> bool:
        """Close the ACP stream without signaling the owned process tree."""
        if self._agent_log_file:
            self._agent_log_file.close()
            self._agent_log_file = None
        if self._session_closed:
            return True
        self._session_closed = await self._cp.close_stdin()
        if self._session_closed:
            await self._cp.wait_for_session_closed(_ACP_SESSION_CLOSE_GRACE_SEC)
        return self._session_closed

    async def terminate_process_tree(self) -> _ProcessTreeTermination:
        """Terminate descendants after the ACP stream-close stage."""
        try:
            status = await self._cp.terminate_process_tree()
            if isinstance(status, _ProcessTreeTermination):
                self._termination_status = status
            self._session_closed = True
            return self._termination_status
        finally:
            stderr = getattr(self._cp, "stderr_tail", "")
            if (
                not self._stderr_appended
                and isinstance(stderr, str)
                and stderr
                and self._agent_log_path
            ):
                with self._agent_log_path.open("a") as agent_log:
                    agent_log.write(
                        redact_trajectory_text_with_exact_values(
                            stderr, self._runtime_credential_values
                        )
                    )
                self._stderr_appended = True

    async def close(self) -> None:
        """Close the ACP stream, then terminate the owned process tree."""
        await self.close_session()
        await self.terminate_process_tree()
