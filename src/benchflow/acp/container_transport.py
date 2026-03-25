"""ACP transport over a live stdio pipe to a sandbox process."""

import json
import logging
from pathlib import Path
from typing import Any

from benchflow.process import LiveProcess

from .transport import Transport

logger = logging.getLogger(__name__)


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
    ):
        self._cp = container_process
        self._command = command
        self._env = env or {}
        self._cwd = cwd
        self._agent_log_path = agent_log_path
        self._agent_log_file = None

    async def start(self) -> None:
        """Start the agent process inside the sandbox."""
        if self._agent_log_path:
            self._agent_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._agent_log_file = open(self._agent_log_path, "w")
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
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Capture non-JSON output (agent debug logs, errors, warnings)
                if self._agent_log_file:
                    self._agent_log_file.write(text + "\n")
                    self._agent_log_file.flush()
                logger.debug(f"Non-JSON from container agent: {text[:200]}")
                continue

    async def close(self) -> None:
        """Terminate the agent process."""
        if self._agent_log_file:
            self._agent_log_file.close()
            self._agent_log_file = None
        await self._cp.close()
