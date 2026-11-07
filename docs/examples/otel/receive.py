"""Receive OpenTelemetry from an agent and keep every export as one JSON line.

BenchFlow 0.9.1 doesn't receive OpenTelemetry itself. This is enough for Claude Code's
logs and metrics over OTLP/HTTP with JSON bodies (OTEL_EXPORTER_OTLP_PROTOCOL=http/json).

    python3 receive.py <address the sandbox reaches> 4318 otlp.jsonl
"""
import gzip
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST, PORT, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if (self.headers.get("Content-Encoding") or "").lower() == "gzip":
            body = gzip.decompress(body)
        line = {"path": self.path, "body": json.loads(body or b"{}")}
        with lock, open(OUT, "a") as f:
            f.write(json.dumps(line) + "\n")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
