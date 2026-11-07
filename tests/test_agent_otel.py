"""BenchFlow ingests the OpenTelemetry an agent sends.

A run that turns the agent's exporter on (Claude Code's
CLAUDE_CODE_ENABLE_TELEMETRY=1) and names no collector gets BenchFlow's
loopback OTLP/HTTP receiver in the agent's sandbox; what arrives is kept one
export per line, root's alone, capped, and copied to the trial at teardown.
"""

from __future__ import annotations

import gzip
import json
import os
import stat
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox import _otlp_receiver as receiver
from benchflow.sandbox import agent_otel


def test_the_run_asks_for_it_by_turning_the_agents_exporter_on():
    assert agent_otel.wants_agent_otel({"CLAUDE_CODE_ENABLE_TELEMETRY": "1"})
    assert agent_otel.wants_agent_otel({"OTEL_LOGS_EXPORTER": "otlp"})
    assert not agent_otel.wants_agent_otel({})
    assert not agent_otel.wants_agent_otel({"CLAUDE_CODE_ENABLE_TELEMETRY": "0"})
    # A collector of the run's own: BenchFlow leaves the exporter alone.
    assert not agent_otel.wants_agent_otel(
        {
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://c:4318",
        }
    )


def test_the_agents_exporter_points_at_the_receiver_in_json():
    env = agent_otel.agent_otel_env(
        {"CLAUDE_CODE_ENABLE_TELEMETRY": "1", "OTEL_LOGS_EXPORTER": "console,otlp"}
    )
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://127.0.0.1:4318"
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/json"
    assert env["OTEL_LOGS_EXPORTER"] == "console,otlp"  # the run's choice kept
    assert env["OTEL_METRICS_EXPORTER"] == "otlp"


@pytest.fixture
def served(tmp_path):
    out = tmp_path / "received.jsonl"
    store = receiver.Store(str(out), max_bytes=4000)
    server = ThreadingHTTPServer(("127.0.0.1", 0), receiver.handler(store))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", out, store
    server.shutdown()


def _post(url, body: bytes, ctype="application/json", encoding=None):
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", ctype)
    if encoding:
        request.add_header("Content-Encoding", encoding)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_each_export_is_kept_as_one_line(served):
    base, out, _ = served
    logs = {
        "resourceLogs": [
            {
                "scopeLogs": [
                    {
                        "logRecords": [
                            {"body": {"stringValue": "claude_code.api_request"}}
                        ]
                    }
                ]
            }
        ]
    }
    assert _post(base + "/v1/logs", json.dumps(logs).encode()) == 200
    assert (
        _post(
            base + "/v1/metrics",
            gzip.compress(b'{"resourceMetrics": []}'),
            encoding="gzip",
        )
        == 200
    )
    assert (
        _post(base + "/v1/traces", b"\x0a\x00", ctype="application/x-protobuf") == 200
    )
    assert _post(base + "/v1/other", b"{}") == 404
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [line["path"] for line in lines] == ["/v1/logs", "/v1/metrics", "/v1/traces"]
    assert lines[0]["body"] == logs and lines[1]["body"] == {"resourceMetrics": []}
    assert lines[2]["body_b64"] == "CgA="  # protobuf kept, not parsed
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o600
    with urllib.request.urlopen(base + "/healthz", timeout=5) as response:
        assert response.status == 200


def test_a_runaway_exporter_cannot_fill_the_disk(served):
    base, out, store = served
    big = json.dumps({"resourceLogs": [{"x": "y" * 1500}]}).encode()
    statuses = [_post(base + "/v1/logs", big) for _ in range(5)]
    assert statuses == [200] * 5  # never a retry against a full file
    assert out.stat().st_size <= 4000 and store.dropped >= 3


def test_a_gzip_bomb_is_refused_without_inflating_it(served, monkeypatch):
    base, out, _ = served
    monkeypatch.setattr(receiver, "MAX_REQUEST_BYTES", 1 << 14)
    bomb = gzip.compress(b"0" * (1 << 20))  # 1 MiB of zeros in about 1 KiB
    assert _post(base + "/v1/logs", bomb, encoding="gzip") == 200
    kept = json.loads(out.read_text().splitlines()[-1])
    assert "too large" in kept["error"] and "body" not in kept


@pytest.mark.asyncio
async def test_a_task_image_without_python_runs_the_trial_without_it():
    env = MagicMock()
    env.exec = AsyncMock(
        side_effect=[
            SimpleNamespace(return_code=0, stdout="", stderr=""),
            SimpleNamespace(
                return_code=87, stdout="", stderr="the task image has no Python 3"
            ),
        ]
    )
    env.upload_dir = AsyncMock()
    assert await agent_otel.start_agent_otel(env) is False
    assert all(call.kwargs.get("user") == "root" for call in env.exec.call_args_list)
