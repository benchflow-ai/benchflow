"""The rows' report server (tests/_report_server.py) stops when told to, even
when its loop was told a client was waiting that is gone by the accept, and
with a client still connected that never sends a request."""

from __future__ import annotations

import selectors
import socket
import socketserver
import struct
import threading
import urllib.request

from tests import _report_server
from tests._report_server import ReportServer


class _AlwaysReadable(selectors.SelectSelector):
    """A selector that calls the listener readable on every turn, whether or
    not a client waits: the serving loop's ``accept()`` then has nothing to
    take, as when the client it was told of has gone."""

    def select(self, timeout=None):
        return [(key, selectors.EVENT_READ) for key in self.get_map().values()]


def _vanish(port: int) -> None:
    """Connect, then go with a reset instead of a close."""
    client = socket.create_connection(("127.0.0.1", port))
    client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    client.close()


def test_the_report_server_stops_after_a_client_vanished(monkeypatch):
    """The close begins only once the loop has taken all three clients and
    gone back into ``accept()`` with none left, where a blocking listener
    would wait for good."""
    monkeypatch.setattr(socketserver, "_ServerSelector", _AlwaysReadable)
    accepted: list[object] = []
    waiting = threading.Event()

    def get_request(self):
        if len(accepted) >= 3:
            waiting.set()  # the loop checked for its stop just before this
        request = socketserver.TCPServer.get_request(self)
        accepted.append(request)
        return request

    monkeypatch.setattr(_report_server._Server, "get_request", get_request)
    server = ReportServer()
    with urllib.request.urlopen(
        f"http://127.0.0.1:{server.port}/r?who=row", timeout=10
    ) as answer:
        assert answer.read() == b"ok"
    assert [h["who"] for h in server.hits] == ["row"]
    _vanish(server.port)
    silent = socket.create_connection(("127.0.0.1", server.port))
    assert waiting.wait(10), "the loop did not take the three clients"
    errors: list[BaseException] = []

    def close() -> None:
        try:
            server.close()
        except BaseException as exc:  # the check reports it below
            errors.append(exc)

    closing = threading.Thread(target=close, daemon=True)
    closing.start()
    closing.join(3 * _report_server.CLOSE_SEC)
    silent.close()
    assert not closing.is_alive(), "the report server's close did not return"
    assert not errors, errors
    assert not server.thread.is_alive()
