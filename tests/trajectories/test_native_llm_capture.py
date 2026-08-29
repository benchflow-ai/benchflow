"""Regression coverage for uniform LLM trajectory capture across auth modes."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.trajectories.llm_capture import (
    LLMTrajectoryCapture,
    _download_optional_dir,
)
from benchflow.trajectories.llm_capture_manifest import (
    CaptureFidelity,
    CaptureSource,
    CaptureStatus,
)
from benchflow.trajectories.native_capture_parsers import (
    parse_claude_raw_capture,
    parse_claude_sessions,
    parse_codex_sessions,
    project_acp_trajectory,
)
from benchflow.trajectories.results import build_rollout_results_record

STARTED_AT = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _otel_value(value: str) -> dict:
    return {"stringValue": value}


def _otel_record(name: str, timestamp_ns: int, **attributes: str) -> dict:
    return {
        "timeUnixNano": str(timestamp_ns),
        "body": _otel_value(name),
        "attributes": [
            {"key": key, "value": _otel_value(value)}
            for key, value in attributes.items()
        ],
    }


def test_claude_otel_raw_bodies_become_provider_wire_exchanges(tmp_path: Path) -> None:
    """Guards this PR's exact Claude OAuth raw-body capture contract."""

    capture = tmp_path / "capture"
    raw = capture / "raw"
    raw.mkdir(parents=True)
    request_path = raw / "random.request.json"
    response_path = raw / "req_123.response.json"
    request_path.write_text(
        json.dumps(
            {
                "model": "claude-opus-4-1",
                "messages": [{"role": "user", "content": "Solve it"}],
            }
        )
    )
    response_path.write_text(
        json.dumps(
            {
                "id": "msg_123",
                "role": "assistant",
                "content": [{"type": "text", "text": "Done"}],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        )
    )
    otel = {
        "resourceLogs": [
            {
                "scopeLogs": [
                    {
                        "logRecords": [
                            _otel_record(
                                "claude_code.api_request",
                                1_777_000_000_000_000_000,
                                session_id="session-1",
                                body_ref=str(request_path),
                            ),
                            _otel_record(
                                "claude_code.api_response",
                                1_777_000_000_500_000_000,
                                session_id="session-1",
                                body_ref=str(response_path),
                                request_id="req_123",
                            ),
                        ]
                    }
                ]
            }
        ]
    }
    otel_dir = capture / "otel"
    otel_dir.mkdir()
    (otel_dir / "000.json").write_text(json.dumps(otel))

    result = parse_claude_raw_capture(
        capture,
        agent="claude-agent-acp",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )

    assert result is not None
    assert result.source is CaptureSource.CLAUDE_OTEL_RAW_BODY
    assert result.fidelity is CaptureFidelity.PROVIDER_WIRE
    assert result.request_complete is True
    assert result.response_complete is True
    assert len(result.trajectory.exchanges) == 1
    exchange = result.trajectory.exchanges[0]
    assert exchange.request.body["messages"][0]["content"] == "Solve it"
    assert exchange.response.body["id"] == "msg_123"
    assert exchange.duration_ms == 500
    assert exchange.metadata["provider_request_id"] == "req_123"


def test_claude_concurrent_raw_pairing_fails_closed_for_training(
    tmp_path: Path,
) -> None:
    """Guards this PR against claiming FIFO pairing for concurrent Claude calls."""

    capture = tmp_path / "capture"
    raw = capture / "raw"
    raw.mkdir(parents=True)
    paths = {
        name: raw / name
        for name in (
            "one.request.json",
            "two.request.json",
            "req_two.response.json",
            "req_one.response.json",
        )
    }
    for name, path in paths.items():
        path.write_text(json.dumps({"marker": name}))
    records = [
        _otel_record(
            "claude_code.api_request",
            1_777_000_000_000_000_000,
            session_id="shared",
            body_ref=str(paths["one.request.json"]),
        ),
        _otel_record(
            "claude_code.api_request",
            1_777_000_000_100_000_000,
            session_id="shared",
            body_ref=str(paths["two.request.json"]),
        ),
        _otel_record(
            "claude_code.api_response",
            1_777_000_000_200_000_000,
            session_id="shared",
            body_ref=str(paths["req_two.response.json"]),
            request_id="req_two",
        ),
        _otel_record(
            "claude_code.api_response",
            1_777_000_000_300_000_000,
            session_id="shared",
            body_ref=str(paths["req_one.response.json"]),
            request_id="req_one",
        ),
    ]
    otel_dir = capture / "otel"
    otel_dir.mkdir()
    (otel_dir / "events.json").write_text(
        json.dumps({"resourceLogs": [{"scopeLogs": [{"logRecords": records}]}]})
    )

    result = parse_claude_raw_capture(
        capture,
        agent="claude-agent-acp",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )

    assert result is not None
    assert result.request_complete is False
    assert any("pairing is ambiguous" in error for error in result.errors)
    assert all(
        exchange.metadata["correlation_complete"] is False
        and exchange.metadata["request_complete"] is False
        for exchange in result.trajectory.exchanges
    )


def test_claude_first_raw_request_survives_missing_otel_body_event(
    tmp_path: Path,
) -> None:
    """Guards this PR against Claude omitting its first request-body OTLP event."""

    capture = tmp_path / "capture"
    raw = capture / "raw"
    raw.mkdir(parents=True)
    request_one = raw / "one.request.json"
    request_two = raw / "two.request.json"
    response_one = raw / "req_one.response.json"
    response_two = raw / "req_two.response.json"
    for path in (request_one, request_two, response_one, response_two):
        path.write_text(json.dumps({"marker": path.name}))
    request_one_timestamp = 1_776_999_999_000_000_000
    os.utime(
        request_one,
        ns=(request_one_timestamp, request_one_timestamp),
    )
    records = [
        _otel_record(
            "claude_code.api_response_body",
            1_777_000_000_500_000_000,
            session_id="shared",
            body_ref=str(response_one),
            request_id="req_one",
        ),
        _otel_record(
            "claude_code.api_request_body",
            1_777_000_000_600_000_000,
            session_id="shared",
            body_ref=str(request_two),
        ),
        _otel_record(
            "claude_code.api_response_body",
            1_777_000_001_000_000_000,
            session_id="shared",
            body_ref=str(response_two),
            request_id="req_two",
        ),
    ]
    otel_dir = capture / "otel"
    otel_dir.mkdir()
    (otel_dir / "events.json").write_text(
        json.dumps({"resourceLogs": [{"scopeLogs": [{"logRecords": records}]}]})
    )

    result = parse_claude_raw_capture(
        capture,
        agent="claude-agent-acp",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )

    assert result is not None
    assert result.request_complete is True
    assert result.errors == []
    assert len(result.trajectory.exchanges) == 2
    first, second = result.trajectory.exchanges
    assert first.request.body["marker"] == "one.request.json"
    assert first.metadata["pairing"] == "raw_file_mtime_fifo"
    assert first.metadata["correlation_complete"] is True
    assert second.request.body["marker"] == "two.request.json"
    assert second.metadata["pairing"] == "otel_session_fifo"


def test_claude_session_fallback_is_truthfully_lower_fidelity(tmp_path: Path) -> None:
    """Guards this PR's Claude OAuth fallback when raw OTel is unavailable."""

    session = tmp_path / "claude" / "session.jsonl"
    _write_jsonl(
        session,
        [
            {
                "type": "user",
                "timestamp": "2026-08-28T12:00:00Z",
                "message": {"role": "user", "content": "Inspect the repo"},
            },
            {
                "type": "assistant",
                "requestId": "req-1",
                "timestamp": "2026-08-28T12:00:01Z",
                "message": {
                    "role": "assistant",
                    "model": "claude-opus-4-1",
                    "content": [{"type": "text", "text": "I will inspect it"}],
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                },
            },
            {
                "type": "assistant",
                "requestId": "req-1",
                "timestamp": "2026-08-28T12:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tool-1", "name": "Read"}
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 4},
                },
            },
        ],
    )

    result = parse_claude_sessions(
        session.parent,
        agent="claude-agent-acp",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )

    assert result is not None
    assert result.fidelity is CaptureFidelity.AGENT_SESSION
    assert result.request_complete is False
    assert result.response_complete is False
    assert len(result.trajectory.exchanges) == 1
    assert len(result.trajectory.exchanges[0].response.body["content"]) == 2
    assert "tool_definitions" in result.missing_fields


def test_codex_oauth_session_splits_calls_and_preserves_usage(tmp_path: Path) -> None:
    """Guards this PR's Codex OAuth native-session trajectory reconstruction."""

    session = tmp_path / "codex" / "session.jsonl"
    _write_jsonl(
        session,
        [
            {
                "type": "session_meta",
                "timestamp": "2026-08-28T12:00:00Z",
                "payload": {"model": "gpt-5.6"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-08-28T12:00:01Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Run tests"}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-08-28T12:00:02Z",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "call-1",
                    "arguments": '{"cmd":"pytest"}',
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-08-28T12:00:03Z",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 20,
                            "output_tokens": 10,
                            "reasoning_output_tokens": 4,
                            "total_tokens": 110,
                        }
                    },
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-08-28T12:00:04Z",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "passed",
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-08-28T12:00:05Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "All green"}],
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-08-28T12:00:06Z",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 50,
                            "output_tokens": 5,
                            "total_tokens": 55,
                        }
                    },
                },
            },
        ],
    )

    result = parse_codex_sessions(
        session.parent,
        agent="codex-acp",
        session_id="rollout-1",
        started_at=STARTED_AT,
        configured_model="gpt-5.6",
    )

    assert result is not None
    assert result.source is CaptureSource.CODEX_NATIVE_SESSION
    assert result.fidelity is CaptureFidelity.AGENT_SESSION
    assert len(result.trajectory.exchanges) == 2
    first, second = result.trajectory.exchanges
    assert first.response.body["usage"]["input_tokens"] == 100
    assert first.response.body["output"][0]["name"] == "exec_command"
    assert second.request.body["input"][-1]["type"] == "function_call_output"
    assert second.response.body["output"][0]["role"] == "assistant"


@pytest.mark.asyncio
async def test_capture_always_emits_empty_jsonl_and_terminal_manifest(
    tmp_path: Path,
) -> None:
    """Guards this PR's always-present artifact invariant for zero-call runs."""

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="codex-acp",
        model="gpt-5.6",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    capture.configure({"OPENAI_API_KEY": "test-key"})

    await capture.finalize(None, acp_events=[], model_call_seen=False)

    trajectory = tmp_path / "trajectory" / "llm_trajectory.jsonl"
    manifest_path = tmp_path / "trajectory" / "llm_trajectory.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert trajectory.exists()
    assert trajectory.read_text() == ""
    assert manifest["status"] == CaptureStatus.NO_MODEL_CALL
    assert manifest["exchange_count"] == 0
    assert manifest["auth_mode"] == "api_key"


@pytest.mark.parametrize(
    ("auth_json", "expected"),
    [
        ('{"auth_mode":"apikey","OPENAI_API_KEY":"test-key"}', "api_key"),
        (
            '{"auth_mode":"chatgpt","tokens":{"refresh_token":"test-token"}}',
            "oauth_subscription",
        ),
    ],
)
def test_codex_native_auth_file_mode_is_not_assumed_to_be_oauth(
    tmp_path: Path,
    auth_json: str,
    expected: str,
) -> None:
    """Guards this PR's Codex auth provenance for API-key auth.json files."""

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="codex-acp",
        model="gpt-5.6",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    capture.configure({"CODEX_AUTH_JSON": auth_json})

    manifest = json.loads(
        (tmp_path / "trajectory" / "llm_trajectory.manifest.json").read_text()
    )
    assert manifest["auth_mode"] == expected


@pytest.mark.asyncio
async def test_provider_jsonl_gets_complete_fidelity_metadata(tmp_path: Path) -> None:
    """Guards this PR's uniform metadata for existing API-key proxy capture."""

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="codex-acp",
        model="azure-foundry-openai/gpt-5.6",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    capture.configure({"OPENAI_API_KEY": "test-key"})
    capture.trajectory_path.write_text(
        json.dumps(
            {
                "request": {"body": {"input": "hello"}},
                "response": {"status_code": 200, "body": {"output": []}},
            }
        )
        + "\n"
    )

    await capture.finalize(None, acp_events=[], model_call_seen=True)

    exchange = json.loads(capture.trajectory_path.read_text())
    manifest = json.loads(
        (tmp_path / "trajectory" / "llm_trajectory.manifest.json").read_text()
    )
    assert exchange["metadata"]["capture_fidelity"] == "provider_wire"
    assert exchange["metadata"]["auth_mode"] == "api_key"
    assert manifest["status"] == "complete"
    assert manifest["capture_source"] == "litellm_proxy"


@pytest.mark.asyncio
async def test_provider_capture_early_return_cleans_native_raw_bodies(
    tmp_path: Path,
) -> None:
    """Guards this PR against raw-body leakage from mixed-role rollouts."""

    commands: list[str] = []

    class CleanupEnv:
        async def exec(self, command, **_kwargs):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    capture._native_agents["claude-agent-acp"] = "claude-sonnet-4-6"
    capture._capture_root_prepared = True
    capture.trajectory_path.write_text(
        json.dumps(
            {
                "request": {"body": {"input": "hello"}},
                "response": {"status_code": 200, "body": {"output": []}},
            }
        )
        + "\n"
    )

    await capture.finalize(CleanupEnv(), acp_events=[], model_call_seen=True)

    assert any("find /tmp/benchflow-llm-capture-" in command for command in commands)
    assert any("-depth -delete" in command for command in commands)


@pytest.mark.asyncio
async def test_claude_capture_setup_failure_degrades_without_aborting(
    tmp_path: Path,
) -> None:
    """Guards this PR against observability setup breaking Claude OAuth runs."""

    commands: list[str] = []

    class FailingCollectorEnv:
        async def exec(self, command, **_kwargs):
            commands.append(command)
            if "nohup" in command:
                return SimpleNamespace(
                    return_code=1,
                    stdout="",
                    stderr="node runtime not found",
                )
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def upload_file(self, *_args, **_kwargs):
            return None

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    prepared = await capture.prepare_agent(
        FailingCollectorEnv(),
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        agent_env={"CLAUDE_CODE_OAUTH_TOKEN": "oauth-test-token"},
        credential_home="/home/agent",
        sandbox_user="agent",
    )

    assert prepared["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert prepared["OTEL_LOG_RAW_API_BODIES"].startswith("file:")
    assert "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT" not in prepared
    assert any("/opt/benchflow/node/bin/node" in command for command in commands)


@pytest.mark.asyncio
async def test_optional_session_download_creates_docker_copy_parent(
    tmp_path: Path,
) -> None:
    """Guards this PR's Docker native-session download path."""

    class DockerLikeEnv:
        async def exec(self, *_args, **_kwargs):
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def download_dir(self, _remote, local):
            assert Path(local).parent.is_dir()
            Path(local).mkdir()

    destination = tmp_path / "missing-parent" / "sessions"
    downloaded = await _download_optional_dir(
        DockerLikeEnv(), "/home/agent/.claude/projects", destination
    )

    assert downloaded is True
    assert destination.is_dir()


def test_capture_failure_repairs_invalid_jsonl_and_redacts_manifest_error(
    tmp_path: Path,
) -> None:
    """Guards this PR's valid-JSONL and secret-redaction failure invariant."""

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="codex-acp",
        model="gpt-5.6",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    capture.trajectory_path.write_text("{not-json\n")

    capture.record_failure(
        "authorization: Bearer sk-test-secret-value", model_call_seen=True
    )

    manifest = json.loads(
        (tmp_path / "trajectory" / "llm_trajectory.manifest.json").read_text()
    )
    assert capture.trajectory_path.read_text() == ""
    assert manifest["status"] == "capture_failed"
    assert "sk-test-secret-value" not in json.dumps(manifest)


def test_acp_projection_retains_the_actual_auth_mode() -> None:
    """Guards this PR against labeling API-key fallback rows as OAuth."""

    result = project_acp_trajectory(
        [{"type": "agent_message", "text": "finished"}],
        agent="codex-acp",
        session_id="rollout-1",
        started_at=STARTED_AT,
        auth_mode="api_key",
    )

    assert result is not None
    assert result.trajectory.exchanges[0].metadata["auth_mode"] == "api_key"


def test_agent_session_capture_is_audit_only_not_training_ready(tmp_path: Path) -> None:
    """Guards this PR against silently training on reconstructed OAuth payloads."""

    trajectory_dir = tmp_path / "trajectory"
    trajectory_dir.mkdir()
    (trajectory_dir / "llm_trajectory.jsonl").write_text(
        json.dumps(
            {
                "request": {
                    "body": {"messages": [{"role": "user", "content": "hello"}]}
                },
                "response": {
                    "status_code": 200,
                    "body": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "hi"}],
                    },
                },
                "metadata": {
                    "capture_fidelity": "agent_session",
                    "request_complete": False,
                    "response_complete": True,
                },
            }
        )
        + "\n"
    )
    (trajectory_dir / "llm_trajectory.manifest.json").write_text(
        json.dumps(
            {
                "status": "partial",
                "capture_fidelity": "agent_session",
                "auth_mode": "oauth_subscription",
                "exchange_count": 1,
                "request_complete": False,
                "response_complete": True,
            }
        )
    )

    row = build_rollout_results_record(
        tmp_path,
        task_name="task",
        rollout_name="rollout",
        agent="claude-agent-acp",
        agent_name="Claude Code",
        model="claude-opus-4-1",
        n_tool_calls=0,
        prompts=["hello"],
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        error=None,
        verifier_error=None,
        agent_result={"usage_source": "agent_native_acp", "total_tokens": 2},
    )

    assert row["info"]["training_ready"] is False
    assert row["info"]["training_ready_reason"] == "insufficient_capture_fidelity"
    assert row["is_completed"] is True
    assert row["error"] is None


def test_corrupt_capture_manifest_fails_closed_for_training(tmp_path: Path) -> None:
    """Guards this PR against treating a corrupt new sidecar as a legacy artifact."""

    trajectory_dir = tmp_path / "trajectory"
    trajectory_dir.mkdir()
    (trajectory_dir / "llm_trajectory.jsonl").write_text(
        json.dumps(
            {
                "request": {
                    "body": {"messages": [{"role": "user", "content": "hello"}]}
                },
                "response": {
                    "status_code": 200,
                    "body": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "hi"}],
                    },
                },
                "metadata": {
                    "capture_fidelity": "provider_wire",
                    "request_complete": True,
                    "response_complete": True,
                },
            }
        )
        + "\n"
    )
    (trajectory_dir / "llm_trajectory.manifest.json").write_text("{broken")

    row = build_rollout_results_record(
        tmp_path,
        task_name="task",
        rollout_name="rollout",
        agent="claude-agent-acp",
        agent_name="Claude Code",
        model="claude-opus-4-1",
        n_tool_calls=0,
        prompts=["hello"],
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        error=None,
        verifier_error=None,
        agent_result={"total_tokens": 2},
    )

    assert row["info"]["training_ready"] is False
    assert row["info"]["training_ready_reason"] == (
        "missing_healthy_structured_llm_trajectory"
    )
    assert row["is_completed"] is False
