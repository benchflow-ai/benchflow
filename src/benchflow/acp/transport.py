"""ACP transports — stdio and SSE."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from benchflow.sandbox.process import drain_oversized_line

logger = logging.getLogger(__name__)


def decode_json_rpc_message(text: str) -> dict[str, Any] | None:
    """Decode one JSON-RPC message line.

    ACP transports are line-delimited JSON, but the protocol message itself
    must be a JSON-RPC 2.0 object. Some agents write JSON-encoded logs to
    stdout; treat those like non-protocol output instead of returning them to
    the client.
    """
    try:
        message = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return None

    has_result = "result" in message
    has_error = "error" in message
    if "method" in message:
        if isinstance(message["method"], str) and not has_result and not has_error:
            return message
        return None
    if "id" in message and has_result != has_error:
        return message
    return None


class Transport(ABC):
    """Base class for ACP transports."""

    @abstractmethod
    async def start(self) -> None:
        """Start the transport connection."""

    @abstractmethod
    async def send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message."""

    @abstractmethod
    async def receive(self) -> dict[str, Any]:
        """Receive a JSON-RPC message."""

    @abstractmethod
    async def close(self) -> None:
        """Close the transport."""


class StdioTransport(Transport):
    """Communicate with an agent process via stdin/stdout."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ):
        self._command = command
        self._args = args or []
        self._env = env
        self._cwd = cwd
        self._process: asyncio.subprocess.Process | None = None
        self._read_buffer = ""

    async def start(self) -> None:
        import os

        proc_env = os.environ.copy()
        if self._env:
            proc_env.update(self._env)

        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
            cwd=self._cwd,
            limit=1024
            * 1024,  # 1MB line buffer (default 64KB too small for large tool results)
        )
        logger.info(f"Started agent process: {self._command} (pid={self._process.pid})")

    async def send(self, message: dict[str, Any]) -> None:
        if not self._process or not self._process.stdin:
            raise RuntimeError("Transport not started")
        data = json.dumps(message) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()

    async def receive(self) -> dict[str, Any]:
        if not self._process or not self._process.stdout:
            raise RuntimeError("Transport not started")
        while True:
            try:
                line = await self._process.stdout.readline()
            except (ValueError, asyncio.LimitOverrunError) as e:
                await drain_oversized_line(self._process.stdout)
                logger.warning(f"Skipped oversized line: {e}")
                continue
            if not line:
                raise ConnectionError("Agent process closed stdout")
            text = line.decode().strip()
            if not text:
                continue
            message = decode_json_rpc_message(text)
            if message is not None:
                return message
            logger.debug(f"Non-JSON-RPC line from agent: {text[:200]}")

    async def close(self) -> None:
        if self._process:
            if self._process.stdin:
                self._process.stdin.close()
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()
            logger.info("Agent process terminated")
