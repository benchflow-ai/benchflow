"""A small Docker Engine API client over the daemon's Unix socket (PERF-E).

Only the direct Docker path (:mod:`benchflow.sandbox._direct_docker`) uses
it. Each call is one HTTP/1.1 request on its own connection to the socket the
``docker`` CLI would use: no ``docker`` or ``docker compose``
process, so a call costs a few milliseconds instead of the CLI's start-up.
The request bodies are the ones Compose sends for the same container
(the differential in ``tests/test_direct_docker_parity.py`` records both).

Nothing here interprets a response beyond its status, its JSON and Docker's
``message`` field; the caller decides what a status means.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

#: The newest Engine API version this client speaks: the one Compose 2.40's
#: Docker client (v28.5) negotiates down from. A daemon that offers less is
#: spoken to at its own version, as the Docker client does.
MAX_API_VERSION = "1.51"
#: Docker's own floor for the calls used here (multiple EndpointsConfig on
#: create needs 1.44): an older daemon goes to Compose.
MIN_API_VERSION = "1.44"

_USER_AGENT = "benchflow-direct-docker"


#: A call's timeout that means none: container create and start wait as long
#: as Compose's ``up`` would (PERF-E-B-L4).
UNBOUNDED = -1.0


class EngineError(RuntimeError):
    """The daemon answered a call with an error status."""

    def __init__(self, method: str, path: str, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"Docker API {method} {path}: {status} {message}")


class EngineUnavailable(RuntimeError):
    """The socket could not be reached or the reply was not HTTP."""


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def negotiated(server: str) -> str:
    """The version to speak: the lower of the daemon's and ours."""
    if _version_tuple(server) < _version_tuple(MAX_API_VERSION):
        return server
    return MAX_API_VERSION


@dataclass
class Reply:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        if not self.body:
            return None
        return json.loads(self.body)


async def _read_reply(reader: asyncio.StreamReader, method: str) -> Reply:
    line = await reader.readline()
    parts = line.decode("latin-1").split(" ", 2)
    if len(parts) < 2 or not parts[0].startswith("HTTP/1."):
        raise EngineUnavailable(f"not an HTTP reply: {line[:80]!r}")
    status = int(parts[1])
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        key, _, value = line.decode("latin-1").partition(":")
        headers[key.strip().lower()] = value.strip()
    if method == "HEAD" or status in (204, 304) or 100 <= status < 200:
        return Reply(status, headers, b"")
    if headers.get("transfer-encoding", "").lower() == "chunked":
        chunks = []
        while True:
            size_line = await reader.readline()
            size = int(size_line.split(b";", 1)[0].strip() or b"0", 16)
            if size == 0:
                # Trailers, then the blank line.
                while (await reader.readline()) not in (b"\r\n", b"\n", b""):
                    pass
                break
            chunks.append(await reader.readexactly(size))
            await reader.readexactly(2)
        return Reply(status, headers, b"".join(chunks))
    if "content-length" in headers:
        return Reply(
            status, headers, await reader.readexactly(int(headers["content-length"]))
        )
    return Reply(status, headers, await reader.read())


class EngineClient:
    """Requests to one daemon socket at one negotiated API version."""

    def __init__(self, socket_path: str, *, timeout: float = 60.0) -> None:
        self.socket_path = socket_path
        self.timeout = timeout
        self.version: str | None = None
        # The daemon's own API version (its /_ping), before negotiation, and
        # its Engine release (the ping's Server header).
        self.server_version: str | None = None
        self.engine_version: str | None = None

    async def _raw(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout: float | None = None,
    ) -> Reply:
        async def call() -> Reply:
            try:
                reader, writer = await asyncio.open_unix_connection(self.socket_path)
            except OSError as exc:
                raise EngineUnavailable(
                    f"cannot reach the Docker socket {self.socket_path}: {exc}"
                ) from exc
            try:
                head = [
                    f"{method} {path} HTTP/1.1",
                    "Host: api.moby.localhost",
                    f"User-Agent: {_USER_AGENT}",
                    "Connection: close",
                ]
                if body is not None:
                    head.append(f"Content-Type: {content_type or 'application/json'}")
                    head.append(f"Content-Length: {len(body)}")
                writer.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1"))
                if body:
                    writer.write(body)
                await writer.drain()
                return await _read_reply(reader, method)
            except (OSError, asyncio.IncompleteReadError, ValueError) as exc:
                raise EngineUnavailable(
                    f"Docker API {method} {path}: the connection failed ({exc})"
                ) from exc
            finally:
                writer.close()
                with contextlib.suppress(OSError):
                    await writer.wait_closed()

        if timeout == UNBOUNDED:
            return await call()
        return await asyncio.wait_for(call(), timeout or self.timeout)

    async def negotiate(self) -> str:
        """Ask the daemon its API version (``GET /_ping``) and pick ours."""
        if self.version is None:
            reply = await self._raw("GET", "/_ping")
            server = reply.headers.get("api-version")
            if reply.status != 200 or not server:
                raise EngineUnavailable("the Docker daemon did not answer /_ping")
            self.server_version = server
            # "Docker/29.1.3 (linux)": the Engine release, for the record.
            engine = re.match(r"Docker/(\S+)", reply.headers.get("server", ""))
            self.engine_version = engine.group(1) if engine else None
            try:
                version = negotiated(server)
                too_old = _version_tuple(version) < _version_tuple(MIN_API_VERSION)
            except ValueError:
                # "1.52-beta": not a version this client can compare (PERF-E-r3-A-L1).
                raise EngineUnavailable(
                    f"the Docker daemon reports API version {server!r}, which does not parse"
                ) from None
            if too_old:
                raise EngineUnavailable(
                    f"the Docker daemon speaks API {server}, below {MIN_API_VERSION}"
                )
            self.version = version
        return self.version

    async def call(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: Any = None,
        timeout: float | None = None,
        raw: bytes | None = None,
        content_type: str | None = None,
    ) -> Reply:
        """One versioned call. *body* is sent as JSON, *raw* as it is with
        *content_type* (an archive upload); an error status raises
        :class:`EngineError` with Docker's message."""
        version = await self.negotiate()
        url = f"/v{version}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(
                {
                    key: (json.dumps(value) if isinstance(value, dict) else value)
                    for key, value in query.items()
                }
            )
        data = raw
        if body is not None:
            data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        reply = await self._raw(
            method,
            url,
            body=data,
            content_type=content_type if raw is not None else None,
            timeout=timeout,
        )
        if reply.status >= 400:
            try:
                message = str((reply.json() or {}).get("message", ""))
            except (ValueError, AttributeError):
                message = reply.body[:300].decode(errors="replace")
            raise EngineError(method, path, reply.status, message)
        return reply

    @staticmethod
    def quote(name: str) -> str:
        return urllib.parse.quote(name, safe="")
