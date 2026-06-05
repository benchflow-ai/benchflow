"""Tests for the dashboard Daytona key persistence endpoint."""

from __future__ import annotations

import http.client
import json
import socketserver
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from dashboard import serve


@contextmanager
def _dashboard_server(directory: Path) -> Iterator[int]:
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    socketserver.ThreadingTCPServer.daemon_threads = True
    handler = partial(serve.SyncingDashboardHandler, directory=str(directory))
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield int(httpd.server_address[1])
        finally:
            httpd.shutdown()
            thread.join(timeout=5)


def _request(
    port: int,
    method: str,
    path: str,
    *,
    body: str | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, str]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    payload = resp.read().decode(errors="replace")
    conn.close()
    return resp.status, payload


def test_dotfile_guard_blocks_encoded_daytona_key(tmp_path):
    """Guards PR #625 against encoded dotfile requests leaking .daytona_key."""
    (tmp_path / ".daytona_key").write_text("SECRET_DTN_KEY")
    (tmp_path / "index.html").write_text("ok")

    with _dashboard_server(tmp_path) as port:
        assert _request(port, "GET", "/index.html") == (200, "ok")
        for path in ("/.daytona_key", "/%2edaytona_key", "/%2Edaytona_key"):
            status, body = _request(port, "GET", path)
            assert status == 404
            assert "SECRET_DTN_KEY" not in body


def test_daytona_key_post_persists_and_feeds_snapshot(tmp_path, monkeypatch):
    """Guards PR #625 so Save persists a mode-600 key used by /daytona.json."""
    key_file = tmp_path / ".daytona_key"
    monkeypatch.setattr(serve, "_DAYTONA_KEY_FILE", key_file)
    seen: list[str | None] = []

    def fake_snapshot(api_key):
        seen.append(api_key)
        return {"count": 0, "by_state": {}, "rows": [], "as_of": "now"}

    monkeypatch.setattr(serve, "daytona_snapshot", fake_snapshot)

    with _dashboard_server(tmp_path) as port:
        origin = f"http://127.0.0.1:{port}"
        status, body = _request(
            port,
            "POST",
            "/daytona/key",
            body=json.dumps({"key": "dtn_saved"}),
            headers={"Content-Type": "application/json", "Origin": origin},
        )
        assert status == 200
        assert json.loads(body) == {"ok": True, "persisted": True}
        assert key_file.read_text() == "dtn_saved"
        assert stat.S_IMODE(key_file.stat().st_mode) == 0o600

        status, _body = _request(port, "GET", "/daytona.json")
        assert status == 200
        assert seen[-1] == "dtn_saved"

        status, body = _request(
            port,
            "POST",
            "/daytona/key",
            body=json.dumps({"key": ""}),
            headers={"Content-Type": "application/json", "Origin": origin},
        )
        assert status == 200
        assert json.loads(body) == {"ok": True, "persisted": False}
        assert not key_file.exists()


def test_daytona_key_post_rejects_cross_origin_and_large_payload(tmp_path, monkeypatch):
    """Guards PR #625 against cross-origin overwrite/clear and oversized bodies."""
    key_file = tmp_path / ".daytona_key"
    key_file.write_text("dtn_original")
    monkeypatch.setattr(serve, "_DAYTONA_KEY_FILE", key_file)

    with _dashboard_server(tmp_path) as port:
        status, _body = _request(
            port,
            "POST",
            "/daytona/key",
            body=json.dumps({"key": "dtn_attacker"}),
            headers={"Content-Type": "application/json", "Origin": "https://evil.test"},
        )
        assert status == 403
        assert key_file.read_text() == "dtn_original"

        status, _body = _request(
            port,
            "POST",
            "/daytona/key",
            body="x" * (serve._MAX_DAYTONA_KEY_BYTES + 1),
            headers={"Sec-Fetch-Site": "same-origin"},
        )
        assert status == 413
        assert key_file.read_text() == "dtn_original"
