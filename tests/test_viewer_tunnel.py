"""`bench view` on a remote machine, through an ssh tunnel (DF-p1-first-run-16).

The usual newcomer runs Docker on a VM and opens the viewer from a laptop. The
viewer answered only ``Host: localhost:<its own port>``, so a tunnel on any
other local port (``ssh -L 9000:localhost:8888``, because 8888 is Jupyter's)
got 403, and nothing printed the tunnel recipe. The host name still has to be
``localhost`` (that is what stops DNS rebinding); the port may now differ, the
confirm page's Origin must match the Host the browser used, the server prints
the recipe when it runs over ssh, and the refusal page lost its "..".
"""

from __future__ import annotations

import http.client
import json
import subprocess
import sys
import threading
import urllib.error
from pathlib import Path

import pytest

from tests.test_viewer_confirm import (
    _confirm_token,
    _free_port,
    _request_decision,
    _serve_in_thread,
    _write_session,
)


def _browse_target(tmp_path: Path) -> Path:
    rollout = tmp_path / "jobs" / "task__abcd1234"
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "trajectory" / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "agent_message", "text": "hi"}) + "\n"
    )
    (rollout / "result.json").write_text(json.dumps({"task_name": "task"}))
    return tmp_path / "jobs"


def _get(port: int, host: str) -> tuple[int, str]:
    conn = http.client.HTTPConnection("localhost", port, timeout=5)
    conn.request("GET", "/", headers={"Host": host})
    response = conn.getresponse()
    return response.status, response.read().decode()


@pytest.mark.parametrize("mode", ["single", "browse"])
def test_a_tunnel_on_another_local_port_reaches_the_viewer(tmp_path, mode):
    target = _write_session(tmp_path) if mode == "single" else _browse_target(tmp_path)
    port, _thread, _ = _serve_in_thread(target, confirm=False)

    status, _body = _get(port, "localhost:9000")  # ssh -L 9000:localhost:<port>

    assert status == 200


@pytest.mark.parametrize("mode", ["single", "browse"])
@pytest.mark.parametrize("host", ["attacker.example:{port}", "127.0.0.1:{port}"])
def test_other_host_names_are_still_refused_with_one_period(tmp_path, mode, host):
    target = _write_session(tmp_path) if mode == "single" else _browse_target(tmp_path)
    port, _thread, _ = _serve_in_thread(target, confirm=False)

    status, body = _get(port, host.format(port=port))

    assert status == 403
    assert f"http://localhost:{port}/" in body
    assert "through an ssh tunnel" in body
    assert ").." not in body


def test_confirm_through_a_tunnel_needs_a_matching_origin(tmp_path):
    session = _write_session(tmp_path)
    port, thread, result = _serve_in_thread(session, confirm=True)
    token = _confirm_token(port)

    with pytest.raises(urllib.error.HTTPError) as exc:
        _request_decision(
            port,
            "approve",
            token=token,
            origin="http://localhost:3000",  # another local site
            host="localhost:9000",
        )
    assert exc.value.code == 403

    _request_decision(
        port,
        "reject",
        token=token,
        origin="http://localhost:9000",
        host="localhost:9000",
    )
    thread.join(timeout=10)
    assert result["decision"] == "rejected"


@pytest.mark.parametrize("over_ssh", [True, False])
def test_the_viewer_prints_the_tunnel_recipe_over_ssh(tmp_path, over_ssh):
    """`bench eval view` in a session reached by ssh says how to reach it."""
    import os
    import time

    session = _write_session(tmp_path)
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if k != "SSH_CONNECTION"}
    if over_ssh:
        env["SSH_CONNECTION"] = "203.0.113.5 50000 10.0.0.2 22"
    code = (
        "import sys; from benchflow.trajectories.viewer import serve; "
        f"serve({str(session)!r}, {port})"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    lines: list[str] = []

    def read() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        deadline = time.monotonic() + 20
        while not any("Press Ctrl+C" in line for line in lines):
            assert time.monotonic() < deadline, "".join(lines)
            assert proc.poll() is None, "".join(lines)
            time.sleep(0.05)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    out = "".join(lines)
    recipe = f"ssh -N -L {port}:localhost:{port} <this machine>"
    assert (recipe in out) is over_ssh, out
    if over_ssh:
        assert f"-L 9000:localhost:{port}, then http://localhost:9000/" in out
