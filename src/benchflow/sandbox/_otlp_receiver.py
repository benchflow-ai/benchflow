"""Receive an agent's OpenTelemetry on loopback and keep each export as one JSON line.

BenchFlow starts this as root in the agent's sandbox (:mod:`benchflow.sandbox.agent_otel`)
on root's protected Python, with the standard library only. An agent that exports
OTLP over HTTP posts to ``http://127.0.0.1:<port>/v1/logs``, ``/v1/metrics`` or
``/v1/traces``; each request becomes one line: when it arrived, its path, its content
type, and its body (parsed when JSON, else base64). The file is root's (0600) in a
folder only root can enter, so the agent can send telemetry but not change what was
received. A cap on the bytes kept stops a runaway exporter from filling the disk;
requests past it are counted, not kept. Every request is answered 200 with an empty
partial-success body, so an exporter never retries against a full file.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PATHS = ("/v1/logs", "/v1/metrics", "/v1/traces")
MAX_REQUEST_BYTES = 8 << 20


class Store:
    def __init__(self, path: str, max_bytes: int) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(path, flags, 0o600)
        self.max_bytes = max_bytes
        self.kept = os.fstat(self.fd).st_size
        self.dropped = 0
        self.lock = threading.Lock()

    def add(self, record: dict) -> None:
        line = (json.dumps(record, separators=(",", ":")) + "\n").encode()
        with self.lock:
            if self.kept + len(line) > self.max_bytes:
                self.dropped += 1
                return
            os.write(self.fd, line)
            self.kept += len(line)


def _decode(body: bytes, encoding: str) -> bytes:
    """*body* as sent, decompressed in bounded steps: a small gzip bomb never
    grows past :data:`MAX_REQUEST_BYTES` in memory."""
    if encoding not in ("gzip", "deflate"):
        return body
    inflater = zlib.decompressobj(
        16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS
    )
    out = inflater.decompress(body, MAX_REQUEST_BYTES + 1)
    if len(out) > MAX_REQUEST_BYTES or inflater.unconsumed_tail:
        raise ValueError("body too large once decompressed")
    return out


def handler(store: Store) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: bytes = b"{}") -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._reply(200, b'{"ok":true}')
            else:
                self._reply(404)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if path not in PATHS:
                self._reply(404)
                return
            if not 0 <= length <= MAX_REQUEST_BYTES:
                self._reply(413)
                return
            raw = self.rfile.read(length)
            ctype = (
                (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            )
            record: dict = {
                "received_at": time.time(),
                "path": path,
                "content_type": ctype,
            }
            try:
                body = _decode(
                    raw, (self.headers.get("Content-Encoding") or "").strip().lower()
                )
                if len(body) > MAX_REQUEST_BYTES:
                    raise ValueError("body too large once decompressed")
                if ctype == "application/json":
                    record["body"] = json.loads(body)
                else:
                    record["body_b64"] = base64.b64encode(body).decode()
            except Exception as exc:  # kept, unparsed, with the reason
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["body_b64"] = base64.b64encode(raw[:65536]).decode()
            store.add(record)
            self._reply(200)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="A loopback OTLP/HTTP receiver.")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-bytes", type=int, default=64 << 20)
    args = parser.parse_args()
    store = Store(args.out, args.max_bytes)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(store))
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
