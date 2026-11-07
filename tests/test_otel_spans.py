"""A trial as OpenTelemetry spans, written per trial and sent where the environment says.

The model calls are the content-free proxy record of a Claude Code trial on
MiMo v2.6 flash (two calls). An OpenTelemetry Collector (otelcol-contrib
0.164.0, otlp and otlpjsonfile receivers) accepted these spans unchanged on
a build VM; here the shape is checked, and a local HTTP server stands
in for the collector.
"""

from __future__ import annotations

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from benchflow.trajectories import otel_spans

FIXTURES = Path(__file__).parent / "fixtures" / "model_calls"


def _trial(tmp_path: Path, **result) -> Path:
    trial = tmp_path / "hello-py__97e8abc3"
    (trial / "trajectory").mkdir(parents=True)
    shutil.copy(
        FIXTURES / "otel-proxy.jsonl", trial / "trajectory" / "llm_trajectory.jsonl"
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "hello-py",
                "rollout_name": trial.name,
                "agent_name": "claude-agent-acp",
                "model": "xiaomi/mimo-v2.6-flash",
                "rewards": {"reward": 1.0},
                "started_at": "2026-11-04T21:45:00+00:00",
                "finished_at": "2026-11-04T21:45:44+00:00",
                **result,
            }
        )
    )
    (trial / "timing.json").write_text(
        json.dumps({"agent_execution": 14.2, "verifier": 1.1})
    )
    return trial


def _attrs(span):
    return {a["key"]: next(iter(a["value"].values())) for a in span["attributes"]}


def test_a_trial_span_and_one_span_per_model_call(tmp_path):
    payload = otel_spans.trial_spans(_trial(tmp_path))
    resource = payload["resourceSpans"][0]
    assert _attrs(resource["resource"])["service.name"] == "benchflow"
    spans = resource["scopeSpans"][0]["spans"]
    trial, *calls = spans
    assert trial["name"] == "trial hello-py" and trial["status"] == {"code": 1}
    assert _attrs(trial)["benchflow.reward"] == 1.0
    assert _attrs(trial)["benchflow.timing.agent_execution_s"] == 14.2
    assert len(calls) == 2
    assert {c["traceId"] for c in spans} == {trial["traceId"]}
    assert all(c["parentSpanId"] == trial["spanId"] and c["kind"] == 3 for c in calls)
    first = _attrs(calls[0])
    assert first["gen_ai.operation.name"] == "chat"
    assert (
        first["gen_ai.usage.input_tokens"] == "14758"
    )  # int64 as a string, per OTLP/JSON
    assert first["gen_ai.usage.output_tokens"] == "55"
    assert first["benchflow.model_call.source"] == "proxy"
    assert calls[0]["name"].startswith("chat ")
    assert int(calls[0]["endTimeUnixNano"]) > int(calls[0]["startTimeUnixNano"])


def test_writing_a_trial_again_gives_the_same_spans(tmp_path):
    trial = _trial(tmp_path)
    assert otel_spans.trial_spans(trial) == otel_spans.trial_spans(trial)


def test_an_errored_trial_says_so(tmp_path):
    payload = otel_spans.trial_spans(
        _trial(tmp_path, error="AgentTimeoutError: ran out", rewards=None)
    )
    trial = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert (
        trial["status"]["code"] == 2
        and "AgentTimeoutError" in trial["status"]["message"]
    )


@pytest.fixture
def collector():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append(
                (
                    self.path,
                    {k.lower(): v for k, v in self.headers.items()},
                    json.loads(body),
                )
            )
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", received
    server.shutdown()


def test_spans_are_written_and_sent_where_the_environment_says(
    tmp_path, monkeypatch, collector
):
    url, received = collector
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", url)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-team=bench%20flow")
    trial = _trial(tmp_path)
    payload = otel_spans.emit_trial_spans(trial)
    assert json.loads((trial / otel_spans.SPANS_FILE).read_text()) == payload
    path, headers, body = received[0]
    assert path == "/v1/traces" and body == payload
    assert headers["x-team"] == "bench flow"


def test_another_protocol_is_not_sent(tmp_path, monkeypatch, collector):
    url, received = collector
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", url)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    trial = _trial(tmp_path)
    assert otel_spans.emit_trial_spans(trial) is not None
    assert received == [] and (trial / otel_spans.SPANS_FILE).is_file()


def test_a_collector_that_is_down_never_fails_the_trial(tmp_path, monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "500")
    assert otel_spans.emit_trial_spans(_trial(tmp_path)) is not None
