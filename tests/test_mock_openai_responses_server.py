from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_health(port: int) -> None:
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=0.2
            ) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.05)
    raise AssertionError("stub server did not become healthy")


def test_mock_openai_responses_server_logs_and_401s(tmp_path: Path) -> None:
    port = _free_port()
    log_file = tmp_path / "stub.jsonl"
    server = subprocess.Popen(
        [
            sys.executable,
            "tests/fixtures/mock_openai_responses_server.py",
            "--port",
            str(port),
            "--log-file",
            str(log_file),
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(port)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/responses",
            data=json.dumps({"model": "mock", "input": "hi"}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer dummy",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=1):  # pragma: no cover
            raise AssertionError("expected 401 from stub")
    except urllib.error.HTTPError as e:
        assert e.code == 401
        payload = json.loads(e.read().decode("utf-8"))
        assert payload["error"]["message"] == "mock-auth-failure"
    finally:
        server.terminate()
        server.wait(timeout=5)

    entries = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["path"] == "/v1/responses"
    assert entries[0]["headers"]["Authorization"] == "Bearer dummy"
