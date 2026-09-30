"""In-sandbox stdio bridge for Hugging Face Sandboxes (standard library only).

BenchFlow uploads this file into the sandbox and starts it as a background
process. It runs one child process (the ACP agent) and exposes its stdin/stdout
over plain HTTP on ``127.0.0.1:<port>``, which the host reaches through
``Sandbox.proxy_url_for(port)``:

- ``GET  /health``         -> ``{"pid", "exit", "lines"}`` (``exit`` is null while the child runs)
- ``GET  /out?from=N``     -> chunked NDJSON stream of stdout lines N, N+1, ...; a bare
                              newline every few seconds keeps idle proxies from closing
                              it; the stream ends after the last line once the child exited
- ``POST /in?seq=K``       -> write the body as one stdin line; a repeated ``seq`` is ignored,
                              so the host may retry a POST whose reply it lost
- ``POST /close``          -> kill the child's process group and stop the bridge

Lines are kept in memory so a dropped stream resumes with ``from=`` and loses
nothing. Empty stdout lines are dropped and carriage returns removed, so the host
and the bridge count lines the same way. The child's stderr goes to a file.

Usage: python3 hf_bridge_server.py CONFIG.json
CONFIG: {"port": int, "argv": [...], "stderr": path, "heartbeat_sec": float, "linger_sec": float}
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    port = int(cfg["port"])
    heartbeat = float(cfg.get("heartbeat_sec", 5.0))
    linger = float(cfg.get("linger_sec", 120.0))
    stderr_file = open(cfg.get("stderr") or os.devnull, "ab", buffering=0)
    child = subprocess.Popen(
        cfg["argv"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        bufsize=0,
        start_new_session=True,
    )
    lines = []
    state = {"exit": None, "last_seq": 0, "closing": False}
    cond = threading.Condition()
    stdin_lock = threading.Lock()

    def pump():
        for raw in iter(child.stdout.readline, b""):
            line = raw.replace(b"\r", b"").rstrip(b"\n")
            if not line:
                continue
            with cond:
                lines.append(line + b"\n")
                cond.notify_all()
        rc = child.wait()
        with cond:
            state["exit"] = rc
            cond.notify_all()

    def kill_child():
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except OSError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except OSError:
                    pass

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _chunk(self, data):
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

        def do_GET(self):
            url = urlparse(self.path)
            if url.path.endswith("/health"):
                with cond:
                    snap = {"pid": child.pid, "exit": state["exit"], "lines": len(lines)}
                return self._json(200, snap)
            if not url.path.endswith("/out"):
                return self._json(404, {"error": "not found"})
            pos = int(parse_qs(url.query).get("from", ["0"])[0])
            self.send_response(200)
            self.send_header("content-type", "application/x-ndjson")
            self.send_header("cache-control", "no-cache")
            self.send_header("x-accel-buffering", "no")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    with cond:
                        if pos >= len(lines) and state["exit"] is None and not state["closing"]:
                            cond.wait(timeout=heartbeat)
                        batch = lines[pos:]
                        done = state["exit"] is not None or state["closing"]
                    if batch:
                        self._chunk(b"".join(batch))
                        pos += len(batch)
                    elif done:
                        break
                    else:
                        self._chunk(b"\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            url = urlparse(self.path)
            n = int(self.headers.get("content-length") or 0)
            data = self.rfile.read(n) if n else b""
            if url.path.endswith("/close"):
                self._json(200, {"ok": True})
                with cond:
                    state["closing"] = True
                    cond.notify_all()
                threading.Thread(target=shutdown, daemon=True).start()
                return
            if not url.path.endswith("/in"):
                return self._json(404, {"error": "not found"})
            seq = int(parse_qs(url.query).get("seq", ["0"])[0])
            with stdin_lock:
                if seq and seq <= state["last_seq"]:
                    return self._json(200, {"duplicate": True})
                if child.poll() is not None:
                    return self._json(410, {"error": "child exited", "exit": child.returncode})
                try:
                    child.stdin.write(data if data.endswith(b"\n") else data + b"\n")
                    child.stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    return self._json(410, {"error": "stdin closed: %s" % exc})
                if seq:
                    state["last_seq"] = seq
            self._json(200, {"ok": True})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True

    def shutdown():
        time.sleep(0.5)
        kill_child()
        server.shutdown()

    def reaper():
        # After the child exits, keep serving for a while so the host can drain the
        # last lines and read the exit code, then stop so the sandbox can go idle.
        while child.poll() is None:
            time.sleep(1)
        time.sleep(linger)
        server.shutdown()

    threading.Thread(target=pump, daemon=True).start()
    threading.Thread(target=reaper, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=shutdown, daemon=True).start())
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        kill_child()


if __name__ == "__main__":
    main()
