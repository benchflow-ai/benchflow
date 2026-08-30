"""Tests for the trajectory viewer's --confirm flow (in-browser approve/reject).

Guards the feature shipped in PR #1021: --confirm must add the decision bar +
/decision endpoint and the DECISION stdout/exit-code contract, while plain
mode remains compatible with the pre-#1021 viewer.

The eval-prize contributor loop is agent-driven: the agent serves the viewer,
the human clicks **Approve & submit** or **Not this one**, and the process
reports the decision via a machine-readable ``DECISION:`` stdout line plus the
exit code (0 approve, 3 reject) so the agent can wait on it instead of a chat
reply.
"""

from __future__ import annotations

import http.client
import http.server
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.trajectories import viewer
from benchflow.trajectories.viewer import render_jsonl_file, render_rollout, serve
from benchflow.trajectories.viewer.server import _serve_browse

runner = CliRunner()

_BAR_NEEDLES = (
    "Submit this trajectory to the BenchFlow eval prize?",
    "Approve &amp; submit",
    "Not this one",
    "/decision",
)

_CONFIRM_TOKEN_RE = re.compile(
    r'"X-BenchFlow-Confirm-Token":\s*"(?P<token>[A-Za-z0-9_-]+)"'
)


def _write_session(tmp_path: Path) -> Path:
    session = tmp_path / "session.jsonl"
    session.write_text(
        json.dumps({"type": "user", "message": {"content": "review me"}}) + "\n"
    )
    return session


def _write_rollout(tmp_path: Path) -> Path:
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    (rollout / "turn1.txt").write_text(
        '{"type":"system","session_id":"s","model":"m"}\n'
    )
    return rollout


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _serve_in_thread(
    path: Path, *, confirm: bool, redaction_summary: str | None = None
) -> tuple[int, threading.Thread, dict]:
    """Run serve() in a daemon thread; poll GET / until it answers."""
    port = _free_port()
    result: dict = {}

    def run() -> None:
        result["decision"] = serve(
            str(path), port, confirm=confirm, redaction_summary=redaction_summary
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while True:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/", timeout=1) as r:
                result["page"] = r.read().decode()
                break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
    return port, thread, result


def _confirm_token(port: int) -> str:
    with urllib.request.urlopen(f"http://localhost:{port}/", timeout=5) as response:
        page = response.read().decode()
    match = _CONFIRM_TOKEN_RE.search(page)
    assert match is not None
    return match.group("token")


def _request_decision(
    port: int,
    body: str,
    *,
    token: str | None,
    origin: str | None,
    host: str | None = None,
) -> str:
    headers = {}
    if token is not None:
        headers["X-BenchFlow-Confirm-Token"] = token
    if origin is not None:
        headers["Origin"] = origin
    if host is not None:
        headers["Host"] = host
    req = urllib.request.Request(
        f"http://localhost:{port}/decision",
        data=body.encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.read().decode()


def _post_decision(port: int, body: str) -> str:
    return _request_decision(
        port,
        body,
        token=_confirm_token(port),
        origin=f"http://localhost:{port}",
    )


def test_confirm_page_has_bar_and_plain_page_does_not(tmp_path):
    """--confirm injects the sticky decision bar; plain rendering carries none of it."""
    session = _write_session(tmp_path)

    plain_html = render_jsonl_file(session)
    for needle in _BAR_NEEDLES:
        assert needle not in plain_html

    port, thread, result = _serve_in_thread(session, confirm=True)
    for needle in _BAR_NEEDLES:
        assert needle in result["page"]

    # Shut the server down so the thread exits.
    _post_decision(port, "reject")
    thread.join(timeout=10)
    assert not thread.is_alive()


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_confirm_page_cannot_be_embedded_by_foreign_sites(tmp_path, method):
    """Guards PR #1034: confirmation pages deny framing on GET and HEAD."""
    session = _write_session(tmp_path)
    port, thread, _result = _serve_in_thread(session, confirm=True)

    request = urllib.request.Request(
        f"http://localhost:{port}/",
        method=method,
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Content-Security-Policy"] == "frame-ancestors 'none'"

    _post_decision(port, "reject")
    thread.join(timeout=10)
    assert not thread.is_alive()


@pytest.mark.parametrize(
    "body, expected", [("approve", "approved"), ("reject", "rejected")]
)
def test_post_decision_returns_decision_and_prints_line(
    tmp_path, capsys, body, expected
):
    """POST /decision shuts the server down; serve() returns the decision and
    prints exactly one machine-readable DECISION line to stdout."""
    session = _write_session(tmp_path)
    port, thread, result = _serve_in_thread(session, confirm=True)

    response = _post_decision(port, body)
    thread.join(timeout=10)
    assert not thread.is_alive()

    assert response == expected
    assert result["decision"] == expected
    out = capsys.readouterr().out
    assert out.count(f"DECISION: {expected}") == 1


def test_post_decision_rejects_garbage_body(tmp_path):
    """A body other than approve/reject is a 400 and keeps the server alive."""
    session = _write_session(tmp_path)
    port, thread, result = _serve_in_thread(session, confirm=True)

    with pytest.raises(urllib.error.HTTPError) as exc:
        _post_decision(port, "maybe")
    assert exc.value.code == 400
    assert thread.is_alive()  # still waiting for a real decision

    _post_decision(port, "reject")
    thread.join(timeout=10)
    assert result["decision"] == "rejected"


@pytest.mark.parametrize(
    ("token_mode", "origin_mode", "host_mode"),
    [
        ("missing", "local", "local"),
        ("wrong", "local", "local"),
        ("nonascii", "local", "local"),
        ("valid", "missing", "local"),
        ("valid", "foreign", "local"),
        ("valid", "local", "foreign"),
    ],
)
def test_confirm_rejects_cross_site_or_unauthenticated_posts(
    tmp_path, token_mode, origin_mode, host_mode
):
    """Guards PR #1034: localhost confirmation requires its nonce, Host, and Origin."""
    session = _write_session(tmp_path)
    port, thread, result = _serve_in_thread(session, confirm=True)
    valid_token = _confirm_token(port)
    token = {
        "missing": None,
        "wrong": "not-the-server-token",
        "nonascii": "é",
        "valid": valid_token,
    }[token_mode]
    origin = {
        "missing": None,
        "foreign": "https://attacker.example",
        "local": f"http://localhost:{port}",
    }[origin_mode]
    host = {
        "foreign": "attacker.example",
        "local": None,
    }[host_mode]

    with pytest.raises(urllib.error.HTTPError) as exc:
        _request_decision(
            port,
            "approve",
            token=token,
            origin=origin,
            host=host,
        )
    assert exc.value.code == 403
    assert thread.is_alive()

    _post_decision(port, "reject")
    thread.join(timeout=10)
    assert result["decision"] == "rejected"


def test_confirm_nonce_is_unique_per_server(tmp_path):
    """Guards PR #1034: every confirmation server gets a cryptographic nonce."""
    session = _write_session(tmp_path)
    port_a, thread_a, _result_a = _serve_in_thread(session, confirm=True)
    port_b, thread_b, _result_b = _serve_in_thread(session, confirm=True)

    token_a = _confirm_token(port_a)
    token_b = _confirm_token(port_b)
    assert token_a != token_b
    assert len(token_a) >= 32
    assert len(token_b) >= 32

    _post_decision(port_a, "reject")
    _post_decision(port_b, "reject")
    thread_a.join(timeout=10)
    thread_b.join(timeout=10)


def test_confirm_rejects_oversized_body_before_recording_decision(tmp_path):
    """Guards PR #1034: confirmation bodies are bounded before the server reads them."""
    session = _write_session(tmp_path)
    port, thread, result = _serve_in_thread(session, confirm=True)

    with pytest.raises(urllib.error.HTTPError) as exc:
        _request_decision(
            port,
            "x" * 33,
            token=_confirm_token(port),
            origin=f"http://localhost:{port}",
        )
    assert exc.value.code == 413
    assert thread.is_alive()

    _post_decision(port, "reject")
    thread.join(timeout=10)
    assert result["decision"] == "rejected"


def test_confirm_rejects_missing_content_length_without_reading(tmp_path):
    """Guards PR #1034: a missing Content-Length gets a bounded 411 response."""
    session = _write_session(tmp_path)
    port, thread, result = _serve_in_thread(session, confirm=True)
    connection = http.client.HTTPConnection("localhost", port, timeout=5)
    connection.putrequest("POST", "/decision")
    connection.putheader("Origin", f"http://localhost:{port}")
    connection.putheader("X-BenchFlow-Confirm-Token", _confirm_token(port))
    connection.endheaders()
    response = connection.getresponse()
    assert response.status == 411
    response.read()
    connection.close()
    assert thread.is_alive()

    _post_decision(port, "reject")
    thread.join(timeout=10)
    assert result["decision"] == "rejected"


def test_confirm_server_whitelists_get_and_head_paths(tmp_path):
    """Guards PR #1034: inherited file-serving GET/HEAD paths stay unreachable."""
    session = _write_session(tmp_path)
    port, thread, _result = _serve_in_thread(session, confirm=True)

    with pytest.raises(urllib.error.HTTPError) as get_exc:
        urllib.request.urlopen(f"http://localhost:{port}/secret.txt", timeout=5)
    assert get_exc.value.code == 404

    head = urllib.request.Request(f"http://localhost:{port}/secret.txt", method="HEAD")
    with pytest.raises(urllib.error.HTTPError) as head_exc:
        urllib.request.urlopen(head, timeout=5)
    assert head_exc.value.code == 405
    assert head_exc.value.headers["Allow"] == "GET"

    _post_decision(port, "reject")
    thread.join(timeout=10)


def test_confirm_js_keeps_controls_on_explicit_http_rejection(tmp_path):
    """Guards PR #1034: an HTTP 4xx must not be displayed as a saved decision."""
    session = _write_session(tmp_path)
    port, thread, result = _serve_in_thread(session, confirm=True)

    assert "if (!response.ok)" in result["page"]
    assert "Could not record that decision. Please try again." in result["page"]
    assert "button.disabled = false" in result["page"]

    _post_decision(port, "reject")
    thread.join(timeout=10)


def test_browse_server_rejects_dns_rebinding_host(tmp_path, monkeypatch):
    """Guards PR #1034: browse rejects foreign Hosts and its retired list route."""
    captured = {}

    class CapturingServer(http.server.HTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["server"] = self

    monkeypatch.setattr(http.server, "HTTPServer", CapturingServer)
    port = _free_port()
    thread = threading.Thread(
        target=_serve_browse,
        args=(tmp_path, port, 0),
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 10
    while True:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/", timeout=1
            ) as response:
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                assert response.headers["X-Content-Type-Options"] == "nosniff"
                assert response.headers["X-Frame-Options"] == "DENY"
                assert (
                    response.headers["Content-Security-Policy"]
                    == "frame-ancestors 'none'"
                )
                break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)

    request = urllib.request.Request(
        f"http://localhost:{port}/",
        headers={"Host": "attacker.example"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request, timeout=5)
    assert exc.value.code == 403
    assert thread.is_alive()

    with pytest.raises(urllib.error.HTTPError) as route_exc:
        urllib.request.urlopen(f"http://localhost:{port}/api/rollouts", timeout=5)
    assert route_exc.value.code == 404

    captured["server"].shutdown()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_plain_mode_has_no_decision_endpoint(tmp_path):
    """Without --confirm, POST /decision does not exist (501 from the stdlib
    handler) and GET / serves the page exactly as before."""
    session = _write_session(tmp_path)
    port, _thread, result = _serve_in_thread(session, confirm=False)

    assert "review me" in result["page"]
    with pytest.raises(urllib.error.HTTPError) as exc:
        _request_decision(port, "approve", token=None, origin=None)
    assert exc.value.code == 501
    # The plain server only stops on Ctrl+C; the daemon thread is reaped at
    # process exit, matching the documented always-on behavior.


def test_confirm_sidecar_stays_plain(tmp_path):
    """The trajectory.html sidecar written for directories must not embed the
    one-shot confirm bar."""
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    (rollout / "turn1.txt").write_text(
        '{"type":"system","session_id":"s","model":"m"}\n'
    )
    port, thread, _result = _serve_in_thread(rollout, confirm=True)
    sidecar = (rollout / "trajectory.html").read_text()
    for needle in _BAR_NEEDLES:
        assert needle not in sidecar

    _post_decision(port, "reject")
    thread.join(timeout=10)


def test_local_rollout_sidecar_is_written_explicitly_as_utf8(tmp_path, monkeypatch):
    """Guards PR #1034: Unicode viewer sidecars never use the Windows locale codec."""
    rollout = _write_rollout(tmp_path)
    encodings: list[str | None] = []
    original_write_text = Path.write_text

    def checked_write_text(path, data, *args, **kwargs):
        if path.name == "trajectory.html":
            encodings.append(kwargs.get("encoding"))
        return original_write_text(path, data, *args, **kwargs)

    class StopBeforeListen:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("stop before listen")

    monkeypatch.setattr(Path, "write_text", checked_write_text)
    monkeypatch.setattr("http.server.HTTPServer", StopBeforeListen)
    with pytest.raises(RuntimeError, match="stop before listen"):
        serve(str(rollout), port=0)

    assert encodings == ["utf-8"]
    assert "trajectory viewer" in (rollout / "trajectory.html").read_text(
        encoding="utf-8"
    )


def test_legacy_result_fields_and_rollout_title_escape_hostile_json(tmp_path):
    """Guards PR #1034: legacy result fields and titles escape hostile JSON."""
    rollout = tmp_path / "unsafe&name"
    rollout.mkdir()
    hostile_turns = "</span><script>turnsPwned()</script><span>"
    hostile_result = {"summary": "</div><script>resultPwned()</script>"}
    (rollout / "turn1.txt").write_text(
        json.dumps(
            {
                "type": "result",
                "num_turns": hostile_turns,
                "total_cost_usd": {"not": "a number"},
                "result": hostile_result,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    page = render_rollout(rollout)

    assert "<title>benchflow — unsafe&amp;name</title>" in page
    assert hostile_turns not in page
    assert "&lt;script&gt;turnsPwned()&lt;/script&gt;" in page
    assert "<script>resultPwned()</script>" not in page
    assert "&lt;script&gt;resultPwned()&lt;/script&gt;" in page
    assert "cost=$0.0000" in page


def test_single_file_response_normalizes_all_text_to_valid_utf8(tmp_path):
    """Guards PR #1034: file pages and confirm summaries always encode as UTF-8."""
    session = tmp_path / "session.jsonl"
    session.write_text(
        '{"type":"user_message","text":"lead \\ud800 tail"}\n',
        encoding="ascii",
    )

    port, thread, result = _serve_in_thread(
        session,
        confirm=True,
        redaction_summary=f"mask {chr(0xD800)}",
    )
    _post_decision(port, "reject")
    thread.join(timeout=10)

    page = result["page"]
    assert not thread.is_alive()
    assert "lead ? tail" in page
    assert "mask ?." in page
    assert "\ud800" not in page
    page.encode("utf-8")


def test_legacy_rollout_sidecar_normalizes_text_to_valid_utf8(tmp_path):
    """Guards PR #1034: legacy trajectory sidecars always encode as UTF-8."""
    rollout = _write_rollout(tmp_path)
    (rollout / "turn1.txt").write_text(
        '{"type":"assistant","message":{"content":['
        '{"type":"text","text":"lead \\ud800 tail"}]}}\n',
        encoding="ascii",
    )

    port, thread, result = _serve_in_thread(rollout, confirm=True)
    _post_decision(port, "reject")
    thread.join(timeout=10)

    sidecar = (rollout / "trajectory.html").read_text(encoding="utf-8")
    assert not thread.is_alive()
    assert "lead ? tail" in sidecar
    assert "\ud800" not in sidecar
    sidecar.encode("utf-8")
    result["page"].encode("utf-8")


def test_hf_rollout_view_never_writes_into_shared_snapshot(tmp_path, monkeypatch):
    """Guards PR #1034: viewing an HF rollout cannot mutate its shared cache snapshot."""
    rollout = _write_rollout(tmp_path)

    class StopBeforeListen:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("stop before listen")

    monkeypatch.setattr(
        "benchflow.trajectories.viewer.server.resolve_hf_dataset",
        lambda _source: rollout,
    )
    monkeypatch.setattr("http.server.HTTPServer", StopBeforeListen)
    with pytest.raises(RuntimeError, match="stop before listen"):
        serve("hf://org/dataset", port=0)

    assert not (rollout / "trajectory.html").exists()


@pytest.mark.parametrize(
    "decision, expected_exit", [("approved", 0), ("rejected", 3), (None, 0)]
)
def test_eval_view_maps_decision_to_exit_code(
    tmp_path, monkeypatch, decision, expected_exit
):
    """The Typer command owns the exit-code mapping: approve → 0, reject → 3
    (non-1/2 so it cannot collide with error or usage exits), Ctrl+C → 0."""
    calls: dict = {}

    def fake_serve(
        path, port=8888, prompts=None, confirm=False, redaction_summary=None
    ):
        calls["confirm"] = confirm
        return decision

    monkeypatch.setattr(viewer, "serve", fake_serve)
    res = runner.invoke(app, ["eval", "view", str(tmp_path), "--confirm"])
    assert res.exit_code == expected_exit
    assert calls["confirm"] is True


def test_eval_view_defaults_to_no_confirm(tmp_path, monkeypatch):
    """Without the flag, serve() must be called with confirm=False so today's
    behavior (no bar, no endpoint) is preserved."""
    calls: dict = {}

    def fake_serve(
        path, port=8888, prompts=None, confirm=False, redaction_summary=None
    ):
        calls["confirm"] = confirm
        return None

    monkeypatch.setattr(viewer, "serve", fake_serve)
    res = runner.invoke(app, ["eval", "view", str(tmp_path)])
    assert res.exit_code == 0
    assert calls["confirm"] is False


def test_confirm_bar_shows_redaction_summary_when_given(tmp_path):
    """Guards the redaction-transparency feature from PR #1022: --redaction-summary renders the masked-secret
    breakdown in the confirm bar (HTML-escaped), above the buttons."""
    session = _write_session(tmp_path)
    port, thread, result = _serve_in_thread(
        session,
        confirm=True,
        redaction_summary="2 API keys, 1 bearer token <&>",
    )

    page = result["page"]
    assert "Before upload, BenchFlow masks: 2 API keys, 1 bearer token" in page
    assert "Originals never leave this machine." in page
    assert "&lt;&amp;&gt;" in page  # summary is escaped, never raw HTML
    assert "<&>" not in page
    # The note precedes the question/buttons inside the bar markup.
    assert page.index('<div class="confirm-note">') < page.index(
        '<span class="confirm-question">'
    )

    _post_decision(port, "reject")
    thread.join(timeout=10)


def test_redaction_summary_cannot_replace_confirm_token_marker(tmp_path):
    """Guards PR #1034: user text cannot consume the server nonce placeholder."""
    session = _write_session(tmp_path)
    marker = "__BENCHFLOW_CONFIRM_TOKEN__"
    port, thread, result = _serve_in_thread(
        session,
        confirm=True,
        redaction_summary=marker,
    )

    assert f"Before upload, BenchFlow masks: {marker}." in result["page"]
    assert _CONFIRM_TOKEN_RE.search(result["page"]) is not None
    _post_decision(port, "reject")
    thread.join(timeout=10)


def test_confirm_bar_without_redaction_summary_is_unchanged(tmp_path):
    """Guards the redaction-transparency feature from PR #1022: the flag is optional — without it the confirm bar
    carries no note markup, and plain (non-confirm) pages never carry any."""
    session = _write_session(tmp_path)

    plain_html = render_jsonl_file(session)
    assert "confirm-note" not in plain_html
    assert "Before upload, BenchFlow masks" not in plain_html

    port, thread, result = _serve_in_thread(session, confirm=True)
    assert "Before upload, BenchFlow masks" not in result["page"]
    assert '<div class="confirm-note">' not in result["page"]

    _post_decision(port, "reject")
    thread.join(timeout=10)


def test_eval_view_passes_redaction_summary_through(tmp_path, monkeypatch):
    """Guards the redaction-transparency feature from PR #1022: the Typer command forwards --redaction-summary to
    serve() untouched."""
    calls: dict = {}

    def fake_serve(
        path, port=8888, prompts=None, confirm=False, redaction_summary=None
    ):
        calls["summary"] = redaction_summary
        return None

    monkeypatch.setattr(viewer, "serve", fake_serve)
    res = runner.invoke(
        app,
        [
            "eval",
            "view",
            str(tmp_path),
            "--confirm",
            "--redaction-summary",
            "2 API keys, 1 bearer token",
        ],
    )
    assert res.exit_code == 0
    assert calls["summary"] == "2 API keys, 1 bearer token"
