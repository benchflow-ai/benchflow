"""Tests for benchflow.otel_capture (Claude Code OTLP usage receiver)."""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

import pytest

from benchflow.cost import recompute_rollout_cost
from benchflow.otel_capture import OtelUsageReceiver, agent_env, parse_otlp_logs

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "otel"
LOGS_FIXTURE = FIXTURE_DIR / "claude_code_logs.json"
METRICS_FIXTURE = FIXTURE_DIR / "claude_code_metrics.json"


def _post(url: str, body: bytes, *, gzip_body: bool = False) -> int:
    headers = {"Content-Type": "application/json"}
    if gzip_body:
        body = gzip.compress(body)
        headers["Content-Encoding"] = "gzip"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status


def test_parse_otlp_logs_real_claude_code_payload():
    """Fixture captured from Claude Code 2.1.263 (2026-09-07) with OTEL_EXPORTER_OTLP_PROTOCOL=http/json."""
    payload = json.loads(LOGS_FIXTURE.read_text())
    records = parse_otlp_logs(payload)
    # 5 log records in the payload; only the two api_request events are usage
    assert [r["event"] for r in records] == ["api_request", "api_request"]
    haiku, fable = records
    assert haiku["model"] == "claude-haiku-4-5-20251001"
    assert (
        haiku["input_tokens"],
        haiku["output_tokens"],
        haiku["cache_read_tokens"],
        haiku["cache_creation_tokens"],
    ) == (899, 8, 0, 0)
    assert haiku["query_source"] == "generate_session_title"
    assert fable["model"] == "claude-fable-5-1"
    assert (
        fable["input_tokens"],
        fable["output_tokens"],
        fable["cache_read_tokens"],
        fable["cache_creation_tokens"],
    ) == (2, 4, 16602, 0)
    assert fable["cost_usd"] == pytest.approx(0.0043705)
    assert fable["duration_ms"] == 3667
    assert fable["request_id"].startswith("req_")
    assert fable["effort"] == "high"
    assert fable["timestamp"].startswith("2026-09-07T17:57:22")
    assert fable["session_id"] == haiku["session_id"]


def test_parse_otlp_logs_ignores_other_events_and_empty():
    assert parse_otlp_logs({}) == []
    payload = {
        "resourceLogs": [
            {
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "body": {"stringValue": "claude_code.tool_result"},
                                "attributes": [
                                    {
                                        "key": "tool_name",
                                        "value": {"stringValue": "Bash"},
                                    }
                                ],
                            },
                            {
                                "attributes": [
                                    {
                                        "key": "event.name",
                                        "value": {"stringValue": "api_error"},
                                    },
                                    {
                                        "key": "model",
                                        "value": {"stringValue": "claude-opus-5"},
                                    },
                                    {
                                        "key": "status_code",
                                        "value": {"intValue": "529"},
                                    },
                                ]
                            },
                        ]
                    }
                ]
            }
        ]
    }
    records = parse_otlp_logs(payload)
    assert len(records) == 1
    assert records[0]["event"] == "api_error" and records[0]["status_code"] == 529


def test_agent_env_points_at_receiver():
    env = agent_env("http://host.docker.internal:4318/")
    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert env["OTEL_LOGS_EXPORTER"] == "otlp"
    assert env["OTEL_METRICS_EXPORTER"] == "none"
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/json"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://host.docker.internal:4318"
    assert agent_env("http://x", metrics=True)["OTEL_METRICS_EXPORTER"] == "otlp"


def test_receiver_writes_usage_jsonl_and_cost_recompute_reads_it(tmp_path):
    rollout = tmp_path / "rollout"
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "model": "claude-fable-5-1",
                "agent_result": {"usage_source": "agent_native_acp"},
                "final_metrics": {},
            }
        )
    )
    out = rollout / "trajectory" / "otel_usage.jsonl"
    logs = LOGS_FIXTURE.read_bytes()
    metrics = METRICS_FIXTURE.read_bytes()

    with OtelUsageReceiver(out) as rx:
        env = agent_env(rx.endpoint)
        base = env["OTEL_EXPORTER_OTLP_ENDPOINT"]
        assert _post(f"{base}/v1/logs", logs) == 200
        assert _post(f"{base}/v1/metrics", metrics) == 200  # acknowledged, not recorded
        assert _post(f"{base}/v1/logs", logs, gzip_body=True) == 200  # gzip accepted
        assert rx.n_requests == 3
    assert rx.n_records == 4

    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 4
    assert all(line["event"] == "api_request" for line in lines)

    report = recompute_rollout_cost(rollout)
    assert report.source == "otel_usage"
    # the second (gzip) post repeated the same request_ids -> deduped to 2 records
    assert report.n_records == 2
    by_model = {m.model: m.cost_usd for m in report.per_model}
    assert by_model["claude-fable-5-1"] == pytest.approx(0.0043705, abs=1e-6)
    assert by_model["claude-haiku-4-5-20251001"] == pytest.approx(0.000939, abs=1e-6)
    assert report.total_cost_usd == pytest.approx(0.0043705 + 0.000939, abs=2e-6)
