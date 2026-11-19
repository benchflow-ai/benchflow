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


# a downstream harness patch: the stop waits for the agent's last export.
_LAST = json.dumps(
    {"resourceLogs": [{"x": "claude_code.api_request", "n": "last"}]}
).encode()


class _HostShell:
    """A sandbox whose root exec runs on the host only for commands naming the
    test's receiver file (the wait, the teardown's copy); any other command
    (the agent kill, the receiver's stop) is answered without running."""

    def __init__(self, out):
        self.out = str(out)
        self.calls = []

    async def exec(self, command, user=None, timeout_sec=None, **_):
        self.calls.append((command, user, timeout_sec))
        if self.out not in command:
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        import asyncio

        proc = await asyncio.create_subprocess_exec(
            "sh", "-c", command, stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout_sec)
        return SimpleNamespace(
            return_code=proc.returncode, stdout=stdout.decode(), stderr=""
        )

    def settles(self):
        return [call for call in self.calls if "while" in call[0]]


def _connected(tmp_path, monkeypatch, served, after_sec):
    """A rollout whose agent sends its last batch *after_sec* after its turn,
    unless it is stopped first."""
    from pathlib import Path

    from benchflow.rollout import Rollout, RolloutConfig

    base, out, _ = served
    monkeypatch.setattr(agent_otel, "RECEIVED", str(out))
    _post(base + "/v1/logs", b'{"resourceLogs": [{"n": "first"}]}')
    agent = threading.Timer(after_sec, _post, (base + "/v1/logs", _LAST))
    seen_at_close = []

    async def close():
        agent.cancel()  # the agent's process ends with its connection
        seen_at_close.append(out.read_bytes())

    rollout = Rollout(RolloutConfig(task_path=Path("task")))
    rollout._env = _HostShell(out)
    rollout._acp_client = SimpleNamespace(close=close)
    rollout._agent_otel = True
    rollout._agent_otel_export_sec = 2.0
    rollout._rollout_dir = tmp_path / "rollout"
    rollout._phase = "connected"
    agent.start()
    return rollout, seen_at_close


@pytest.mark.asyncio
async def test_the_stop_waits_for_the_agents_last_export(tmp_path, monkeypatch, served):
    """The last batch came 1.5 s after the turn and the agent was closed at once,
    so the session's last model call was in no event (a downstream report)."""
    rollout, seen_at_close = _connected(tmp_path, monkeypatch, served, 1.5)
    await rollout.disconnect()
    assert b'"last"' in seen_at_close[0]
    await rollout._teardown_agent_otel()
    kept = (tmp_path / "rollout" / "trajectory" / "agent-otel.jsonl").read_text()
    assert '"first"' in kept and '"last"' in kept
    (settle,) = rollout._env.settles()
    assert settle[1] == "root" and settle[2] == 10  # the 5 s cap, plus 5


@pytest.mark.asyncio
async def test_an_agent_the_host_deadline_stopped_is_stopped_at_once(
    tmp_path, monkeypatch, served
):
    import time

    rollout, seen_at_close = _connected(tmp_path, monkeypatch, served, 1.5)
    rollout._agent_deadline_reason = "Agent timed out: still running at the host"
    started = time.monotonic()
    await rollout.disconnect()
    assert time.monotonic() - started < 1.5  # before the batch was due
    assert rollout._env.settles() == [] and b'"last"' not in seen_at_close[0]


@pytest.mark.asyncio
async def test_a_cancelled_agent_phase_is_stopped_at_once(
    tmp_path, monkeypatch, served
):
    import asyncio

    rollout, seen_at_close = _connected(tmp_path, monkeypatch, served, 1.5)

    async def phase():
        try:
            await asyncio.sleep(30)
        finally:
            await rollout.disconnect()

    task = asyncio.ensure_future(phase())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert rollout._env.settles() == [] and seen_at_close


def test_the_wait_has_bounds():
    assert agent_otel.export_interval_sec({}) == 2.0
    assert agent_otel.export_interval_sec({"OTEL_LOGS_EXPORT_INTERVAL": "500"}) == 0.5
    for bad in ("soon", "inf", "-1", "0"):
        assert agent_otel.export_interval_sec({"OTEL_LOGS_EXPORT_INTERVAL": bad}) == 2.0
    assert agent_otel.settle_bounds(2.0) == (2.5, 5.0)
    assert agent_otel.settle_bounds(60.0) == (10.0, 10.0)  # SETTLE_CAP_SEC


@pytest.mark.asyncio
async def test_an_exporter_that_never_stops_is_waited_for_at_most_the_cap(
    tmp_path, monkeypatch, served
):
    import time

    base, out, _ = served
    monkeypatch.setattr(agent_otel, "RECEIVED", str(out))
    stop = threading.Event()

    def chatty():
        while not stop.wait(0.2):
            _post(base + "/v1/logs", b'{"resourceLogs": []}')

    threading.Thread(target=chatty, daemon=True).start()
    env = _HostShell(out)
    started = time.monotonic()
    try:
        assert await agent_otel.settle_agent_otel(env, 0.5) is False
    finally:
        stop.set()
    # Never quiet: the whole cap (2 x 0.5 + 1 s), inside the exec bound (+5 s).
    assert 2.0 <= time.monotonic() - started < 7.0
    assert env.calls[0][2] == 7
