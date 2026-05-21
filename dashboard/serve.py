#!/usr/bin/env python3
"""Serve the BenchFlow v0.5 dashboard on http://localhost:8777.

Refreshes ``data.json`` (test results + rollout artifacts + the authored
sections) once on startup, then serves the static ``dashboard/`` directory.

Usage::

    python dashboard/serve.py                 # serve (reuse the last junit.xml)
    python dashboard/serve.py --run-tests     # re-run the test suite first
    python dashboard/serve.py --port 9000     # pick a port

Re-run ``python dashboard/generate.py`` any time to refresh the data, then
reload the page — no server restart needed.
"""

from __future__ import annotations

import contextlib
import http.server
import socketserver
import subprocess
import sys
import webbrowser
from functools import partial
from pathlib import Path

DASH = Path(__file__).resolve().parent


def main() -> int:
    argv = sys.argv[1:]
    port = 8777
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])

    gen = [sys.executable, str(DASH / "generate.py")]
    if "--run-tests" in argv:
        gen.append("--run-tests")
    print("refreshing data.json ...", flush=True)
    subprocess.run(gen, check=False)

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(DASH))
    # quieter logs — one line per request is enough
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://localhost:{port}/"
        print(f"\n  BenchFlow dashboard → {url}")
        print("  Ctrl-C to stop.\n")
        with contextlib.suppress(Exception):
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
