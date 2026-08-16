"""Tests for the trajectory viewer's --confirm flow (in-browser approve/reject).

Guards the feature shipped in PR #1021: --confirm must add the decision bar +
/decision endpoint and the DECISION stdout/exit-code contract, while plain
mode stays byte-for-byte the pre-#1021 viewer.

The eval-prize contributor loop is agent-driven: the agent serves the viewer,
the human clicks **Approve & submit** or **Not this one**, and the process
reports the decision via a machine-readable ``DECISION:`` stdout line plus the
exit code (0 approve, 3 reject) so the agent can wait on it instead of a chat
reply.
"""

from __future__ import annotations

import json
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
from benchflow.trajectories.viewer import render_jsonl_file, serve

runner = CliRunner()

_BAR_NEEDLES = (
    "Submit this trajectory to the BenchFlow eval prize?",
    "Approve &amp; submit",
    "Not this one",
    "/decision",
)


def _write_session(tmp_path: Path) -> Path:
    session = tmp_path / "session.jsonl"
    session.write_text(
        json.dumps({"type": "user", "message": {"content": "review me"}}) + "\n"
    )
    return session


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


def _post_decision(port: int, body: str) -> str:
    req = urllib.request.Request(
        f"http://localhost:{port}/decision", data=body.encode(), method="POST"
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.read().decode()


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


def test_plain_mode_has_no_decision_endpoint(tmp_path):
    """Without --confirm, POST /decision does not exist (501 from the stdlib
    handler) and GET / serves the page exactly as before."""
    session = _write_session(tmp_path)
    port, _thread, result = _serve_in_thread(session, confirm=False)

    assert "review me" in result["page"]
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post_decision(port, "approve")
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
    """Redaction transparency: --redaction-summary renders the masked-secret
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


def test_confirm_bar_without_redaction_summary_is_unchanged(tmp_path):
    """Redaction transparency: the flag is optional — without it the confirm bar
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
    """Redaction transparency: the Typer command forwards --redaction-summary to
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
