"""A recording proxy in front of the Docker daemon's socket (PERF-E).

Point a client at it (``DOCKER_HOST=unix://<listen>``) and every Engine API
request on the way through is written to a JSON-lines file: method, path,
JSON body, reply status and, for small JSON replies, the reply body. Bytes are
passed on unchanged, so Compose, the ``docker`` CLI and BenchFlow's own
Engine API client behave exactly as they do against the daemon.

When a container start succeeds, the proxy inspects that container and each
network it joined over its own connection before it relays the reply, and
writes those as ``snapshot`` records: the parity test compares them under
:func:`benchflow.sandbox._direct_docker.normalise_inspect`.

Run it on its own: ``python -I tests/_docker_api_recorder.py LISTEN UPSTREAM LOG``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

_MAX_RECORDED_BODY = 4 << 20
_START = re.compile(r"^/v[0-9.]+/containers/([^/?]+)/start(?:\?|$)")


async def _read_head(reader: asyncio.StreamReader) -> tuple[bytes, list[bytes]] | None:
    first = await reader.readline()
    if not first:
        return None
    lines = []
    while True:
        line = await reader.readline()
        if not line:
            return None
        lines.append(line)
        if line in (b"\r\n", b"\n"):
            break
    return first, lines


def _headers(lines: list[bytes]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in lines:
        key, sep, value = line.decode("latin-1").partition(":")
        if sep:
            out[key.strip().lower()] = value.strip()
    return out


async def _relay_body(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter | None,
    headers: dict[str, str],
    *,
    keep: bool,
    until_eof: bool,
) -> bytes:
    """Pass one message body on; return it when *keep* (and small)."""
    kept = bytearray()

    def take(data: bytes) -> None:
        if writer is not None:
            writer.write(data)
        if keep and len(kept) <= _MAX_RECORDED_BODY:
            kept.extend(data)

    if headers.get("transfer-encoding", "").lower() == "chunked":
        while True:
            size_line = await reader.readline()
            if writer is not None:
                writer.write(size_line)
            size = int(size_line.split(b";", 1)[0].strip() or b"0", 16)
            if size == 0:
                while True:
                    trailer = await reader.readline()
                    if writer is not None:
                        writer.write(trailer)
                    if trailer in (b"\r\n", b"\n", b""):
                        break
                break
            chunk = await reader.readexactly(size)
            if writer is not None:
                writer.write(chunk)
            if keep and len(kept) <= _MAX_RECORDED_BODY:
                kept.extend(chunk)
            if writer is not None:
                writer.write(await reader.readexactly(2))
            else:
                await reader.readexactly(2)
            if writer is not None:
                await writer.drain()
    elif "content-length" in headers:
        left = int(headers["content-length"])
        while left:
            data = await reader.read(min(left, 1 << 16))
            if not data:
                raise asyncio.IncompleteReadError(b"", left)
            left -= len(data)
            take(data)
            if writer is not None:
                await writer.drain()
    elif until_eof:
        while data := await reader.read(1 << 16):
            take(data)
            if writer is not None:
                await writer.drain()
    return bytes(kept)


def _decode(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return {"_bytes": len(body)}


async def _splice(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    with contextlib.suppress(OSError, ConnectionError):
        while data := await reader.read(1 << 16):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()


class Recorder:
    def __init__(self, listen: str, upstream: str, log: Path) -> None:
        self.listen = listen
        self.upstream = upstream
        self.log = log
        self.seq = 0
        self._out = log.open("a", buffering=1)
        self._lock = asyncio.Lock()
        self.server: asyncio.AbstractServer | None = None

    async def write(self, record: dict) -> None:
        async with self._lock:
            self.seq += 1
            record = {"seq": self.seq, "t": time.time(), **record}
            self._out.write(json.dumps(record, sort_keys=True) + "\n")

    async def _get(self, path: str) -> tuple[int, Any]:
        reader, writer = await asyncio.open_unix_connection(self.upstream)
        try:
            writer.write(
                f"GET {path} HTTP/1.1\r\nHost: recorder\r\nConnection: close\r\n\r\n".encode()
            )
            await writer.drain()
            head = await _read_head(reader)
            assert head is not None
            status = int(head[0].split()[1])
            body = await _relay_body(
                reader, None, _headers(head[1]), keep=True, until_eof=True
            )
            return status, _decode(body)
        finally:
            writer.close()

    async def snapshot(self, version_prefix: str, container: str) -> None:
        status, inspected = await self._get(
            f"{version_prefix}/containers/{container}/json"
        )
        await self.write(
            {
                "kind": "snapshot",
                "what": "container",
                "status": status,
                "body": inspected,
            }
        )
        networks = ((inspected or {}).get("NetworkSettings") or {}).get(
            "Networks"
        ) or {}
        for name in sorted(networks):
            if name in ("none", "host", "bridge"):
                continue
            status, net = await self._get(f"{version_prefix}/networks/{name}")
            await self.write(
                {"kind": "snapshot", "what": "network", "status": status, "body": net}
            )

    async def handle(
        self, client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter
    ) -> None:
        try:
            up_r, up_w = await asyncio.open_unix_connection(self.upstream)
        except OSError:
            client_w.close()
            return
        try:
            while True:
                head = await _read_head(client_r)
                if head is None:
                    break
                first, lines = head
                method, path, _ = first.decode("latin-1").split(" ", 2)
                headers = _headers(lines)
                up_w.write(first + b"".join(lines))
                body = await _relay_body(
                    client_r, up_w, headers, keep=True, until_eof=False
                )
                await up_w.drain()
                record: dict[str, Any] = {
                    "kind": "request",
                    "method": method,
                    "path": path,
                    "agent": headers.get("user-agent", ""),
                    "body": _decode(body),
                }
                reply = await _read_head(up_r)
                if reply is None:
                    record["status"] = None
                    await self.write(record)
                    break
                status_line, reply_lines = reply
                status = int(status_line.split()[1])
                reply_headers = _headers(reply_lines)
                record["status"] = status
                upgraded = (
                    status == 101 or "upgrade" in headers.get("connection", "").lower()
                )
                streamed = reply_headers.get("content-type", "") in (
                    "application/vnd.docker.raw-stream",
                    "application/vnd.docker.multiplexed-stream",
                )
                start = _START.match(path) if method == "POST" else None
                if start and 200 <= status < 300:
                    prefix = path.split("/containers/", 1)[0]
                    await self.snapshot(prefix, start.group(1))
                client_w.write(status_line + b"".join(reply_lines))
                await client_w.drain()
                if upgraded or streamed:
                    await self.write(record)
                    await asyncio.gather(
                        _splice(client_r, up_w), _splice(up_r, client_w)
                    )
                    return
                is_json = "json" in reply_headers.get("content-type", "")
                bodiless = method == "HEAD" or status in (204, 304)
                reply_body = (
                    b""
                    if bodiless
                    else await _relay_body(
                        up_r,
                        client_w,
                        reply_headers,
                        keep=is_json,
                        until_eof="content-length" not in reply_headers
                        and "chunked" not in reply_headers.get("transfer-encoding", ""),
                    )
                )
                await client_w.drain()
                if is_json:
                    record["reply"] = _decode(reply_body)
                await self.write(record)
        except (OSError, ConnectionError, asyncio.IncompleteReadError, ValueError):
            pass
        finally:
            for writer in (client_w, up_w):
                with contextlib.suppress(Exception):
                    writer.close()

    async def start(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.listen)
        self.server = await asyncio.start_unix_server(self.handle, path=self.listen)
        os.chmod(self.listen, 0o600)

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            # 3.12's wait_closed also waits for open connections, and a
            # hijacked exec or attach may stay open: do not hang on them.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.server.wait_closed(), 2)
        self._out.close()


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def _main(listen: str, upstream: str, log: str) -> None:
    recorder = Recorder(listen, upstream, Path(log))
    await recorder.start()
    print("ready", flush=True)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    import signal

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    await recorder.close()


if __name__ == "__main__":
    asyncio.run(_main(*sys.argv[1:4]))
