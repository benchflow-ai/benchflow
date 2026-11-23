"""A server on the host that keeps the query of each GET it is sent (the
rows' reports), and that always stops.

Its listener waits at most ``LISTEN_SEC`` in ``accept()``: the selector can
call it readable for a client that is gone by the time it is accepted, and a
blocking listener would then wait there for the next client, holding
``shutdown()`` with it. A handler waits at most ``HANDLER_SEC`` on its
client, and its thread never holds the close. :meth:`ReportServer.close`
ends within about ``2 * CLOSE_SEC`` or says why not.
"""

from __future__ import annotations

import contextlib
import http.server
import socket
import socketserver
import threading
import time
import urllib.parse

LISTEN_SEC = 1.0
HANDLER_SEC = 10.0
CLOSE_SEC = 5.0


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        """A client that went away mid-request is not the row's concern."""


class ReportServer:
    def __init__(self) -> None:
        self.hits: list[dict] = []
        hits = self.hits

        class Handler(http.server.BaseHTTPRequestHandler):
            timeout = HANDLER_SEC

            def do_GET(self):
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                hits.append({k: v[0] for k, v in query.items()} | {"at": time.time()})
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.server = _Server(("0.0.0.0", 0), Handler)
        self.server.socket.settimeout(LISTEN_SEC)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        """Stop serving and close the listener. ``shutdown()`` waits for the
        serving loop's next turn; past ``CLOSE_SEC`` the listener is shut
        down under it (which ends an ``accept()`` on Linux) and closed."""
        stopping = threading.Thread(target=self.server.shutdown, daemon=True)
        stopping.start()
        stopping.join(CLOSE_SEC)
        with contextlib.suppress(OSError):
            self.server.socket.shutdown(socket.SHUT_RDWR)
        self.server.server_close()
        self.thread.join(CLOSE_SEC)
        if self.thread.is_alive():
            raise RuntimeError(
                f"the report server on port {self.port} still serves "
                f"{2 * CLOSE_SEC:.0f}s after it was told to stop"
            )
