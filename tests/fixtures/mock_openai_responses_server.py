#!/usr/bin/env python3
"""Minimal local /v1/responses stub for custom-provider routing checks.

This server is intentionally tiny. It does not try to emulate the full
Responses API or support WebSockets. Its job is to:

1. Accept POSTs to /v1/responses over plain HTTP
2. Record the request path, headers, and body to a JSONL log
3. Return a deterministic OpenAI-style 401 error

That is enough to prove whether codex-acp actually routed to the custom base
URL. The expected BenchFlow invocation still fails; the test passes if the
request hits this server instead of api.openai.com.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _Handler(BaseHTTPRequestHandler):
    server_version = "MockResponses/0.1"
    log_path: Path

    def log_message(self, format: str, *args) -> None:
        return

    def _write_log(self, body: bytes) -> None:
        entry = {
            "method": self.command,
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
            "body": body.decode("utf-8", errors="replace"),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def do_GET(self) -> None:
        if self.path == "/health":
            payload = {"ok": True}
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404, "not found")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self._write_log(body)

        if self.path != "/v1/responses":
            self.send_error(404, "not found")
            return

        payload = {
            "error": {
                "message": "mock-auth-failure",
                "type": "authentication_error",
            }
        }
        response = json.dumps(payload).encode("utf-8")
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args()

    _Handler.log_path = Path(args.log_file)
    _Handler.log_path.parent.mkdir(parents=True, exist_ok=True)
    _Handler.log_path.write_text("", encoding="utf-8")

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
