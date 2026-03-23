"""ACP transports — stdio and SSE."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


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
            line = await self._process.stdout.readline()
            if not line:
                raise ConnectionError("Agent process closed stdout")
            text = line.decode().strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.debug(f"Non-JSON line from agent: {text}")
                continue

    async def close(self) -> None:
        if self._process:
            if self._process.stdin:
                self._process.stdin.close()
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            logger.info("Agent process terminated")


class SSETransport(Transport):
    """Communicate with an agent over HTTP/SSE."""

    def __init__(self, url: str):
        self._url = url
        self._client: Any = None
        self._response_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._listener_task: asyncio.Task | None = None

    async def start(self) -> None:
        import httpx

        self._client = httpx.AsyncClient(timeout=None)
        logger.info(f"SSE transport connected to {self._url}")

    async def send(self, message: dict[str, Any]) -> None:
        if not self._client:
            raise RuntimeError("Transport not started")
        response = await self._client.post(
            self._url,
            json=message,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            await self._response_queue.put(data)

    async def receive(self) -> dict[str, Any]:
        return await self._response_queue.get()

    async def close(self) -> None:
        if self._listener_task:
            self._listener_task.cancel()
        if self._client:
            await self._client.aclose()
