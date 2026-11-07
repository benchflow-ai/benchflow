"""The Hub client's API key never follows a redirect to another origin.

An independent audit (2026-11-04) showed a Hub redirect to another host received
``Authorization: Bearer <key>``: the client set the header on the request and
urllib copies it to every redirect. The first test is the audit's probe, with a
synthetic key. Now the key goes only to the origin it was sent to; a redirect
elsewhere is followed without it, and HTTPS never redirects to HTTP.
"""

from __future__ import annotations

import http.client
import io
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from benchflow.hub.client import (
    HubClient,
    HubError,
    _DowngradeRefused,
    _SameOriginCredentials,
)

TOKEN = "audit-synthetic-token"


@contextmanager
def _server(handler: type[BaseHTTPRequestHandler]) -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _handlers(received: list, location: str | None = None):
    class Sink(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append((self.path, self.headers.get("Authorization")))
            if location is not None and self.path == "/start":
                self.send_response(302)
                self.send_header(
                    "Location", location.format(port=self.server.server_port)
                )
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            return

    return Sink


def test_a_redirect_to_another_origin_does_not_carry_the_key():
    """The audit's probe: 127.0.0.1:<a> redirects to localhost:<b>."""
    sink_seen: list = []
    with _server(_handlers(sink_seen)) as sink:
        origin_seen: list = []
        target = f"http://localhost:{sink.server_port}/capture"
        with _server(_handlers(origin_seen, location=target)) as origin:
            client = HubClient(f"http://127.0.0.1:{origin.server_port}", token=TOKEN)
            assert client.get_json("/start") == {}
    assert origin_seen == [("/start", f"Bearer {TOKEN}")]
    assert sink_seen == [("/capture", None)]


def test_a_redirect_on_the_same_origin_keeps_the_key():
    seen: list = []
    with _server(_handlers(seen, location="http://127.0.0.1:{port}/moved")) as hub:
        client = HubClient(f"http://127.0.0.1:{hub.server_port}", token=TOKEN)
        assert client.get_json("/start") == {}
    assert seen == [("/start", f"Bearer {TOKEN}"), ("/moved", f"Bearer {TOKEN}")]


def test_https_never_redirects_to_http():
    request = Request(
        "https://hub.example/api/v1/registry", headers={"Authorization": "Bearer k"}
    )
    with pytest.raises(HTTPError, match="refused a redirect from HTTPS"):
        _SameOriginCredentials().redirect_request(
            request,
            io.BytesIO(),
            302,
            "Found",
            http.client.HTTPMessage(),
            "http://hub.example/x",
        )


def test_a_refused_downgrade_is_a_hub_error(monkeypatch):
    seen: list = []
    with _server(_handlers(seen)) as hub:
        client = HubClient(f"http://127.0.0.1:{hub.server_port}", token=TOKEN)

        def refuse(request, timeout):
            raise _DowngradeRefused(
                request.full_url,
                302,
                "refused a redirect from HTTPS to http://x/",
                None,
                None,
            )

        monkeypatch.setattr("benchflow.hub.client._OPENER.open", refuse)
        with pytest.raises(HubError, match="refused a redirect from HTTPS"):
            client.get_json("/start")
