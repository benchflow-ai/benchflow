"""Failure-boundary regressions for native LLM trajectory capture."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.trajectories import llm_capture as llm_capture_module
from benchflow.trajectories.llm_capture import LLMTrajectoryCapture, _CaptureTarget
from benchflow.trajectories.llm_capture_manifest import (
    AuthMode,
    CaptureFidelity,
    CaptureStatus,
)
from benchflow.trajectories.llm_capture_records import (
    NativeCaptureBundle,
    assemble_capture,
    load_provider_wire_records,
)
from benchflow.trajectories.native_capture_collection import NativeCollection
from benchflow.trajectories.native_capture_parsers import (
    parse_codex_sessions,
    project_acp_trajectory,
)

STARTED_AT = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


def test_native_parser_normalizes_fallback_and_record_timestamps(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against mixing naive fallback and aware record times."""

    session = tmp_path / "sessions" / "rollout-session-one.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    },
                },
                {
                    "timestamp": "2026-08-29T12:00:01Z",
                    "type": "event_msg",
                    "payload": {"type": "token_count", "info": {}},
                },
            )
        )
        + "\n"
    )

    result = parse_codex_sessions(
        session.parent,
        agent="codex-acp",
        session_id="rollout-1",
        started_at=datetime(2020, 1, 1),
        configured_model="gpt-5.6",
    )

    assert result is not None
    exchange = result.trajectory.exchanges[0]
    assert exchange.request.timestamp.tzinfo is UTC
    assert exchange.response.timestamp.tzinfo is UTC
    assert exchange.duration_ms > 0


def test_provider_role_attribution_uses_proxy_model_aliases(tmp_path: Path) -> None:
    """Guards PR #1057 against losing roles after LiteLLM model translation."""

    targets = [
        _CaptureTarget(
            agent="claude-agent-acp",
            model="aws-bedrock/us.anthropic.claude-opus-4-8",
            credential_home="/home/solver",
            auth_mode=AuthMode.API_KEY,
            native=False,
            role="solver",
        ),
        _CaptureTarget(
            agent="codex-acp",
            model="azure-foundry-openai/gpt-5.5",
            credential_home="/home/reviewer",
            auth_mode=AuthMode.API_KEY,
            native=False,
            role="reviewer",
        ),
        _CaptureTarget(
            agent="opencode",
            model="openai/gpt-5.5",
            credential_home="/home/critic",
            auth_mode=AuthMode.API_KEY,
            native=False,
            role="critic",
        ),
    ]
    trajectory = tmp_path / "llm_trajectory.jsonl"
    trajectory.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                {
                    "request": {
                        "body": {"model": "bedrock/us.anthropic.claude-opus-4-8"}
                    },
                    "response": {"status_code": 200, "body": {}},
                    "metadata": {
                        "model_group": (
                            "benchflow-aws-bedrock-us.anthropic.claude-opus-4-8"
                        )
                    },
                },
                {
                    "request": {"body": {"model": "gpt-5.5"}},
                    "response": {"status_code": 200, "body": {}},
                    "metadata": {
                        "benchflow_requested_model": ("azure-foundry-openai/gpt-5.5"),
                        "benchflow_model_alias": (
                            "benchflow-azure-foundry-openai-gpt-5.5"
                        ),
                        "request_model": "gpt-5.5",
                    },
                },
                {
                    "request": {"body": {"model": "gpt-5.5"}},
                    "response": {"status_code": 200, "body": {}},
                    "metadata": {
                        "benchflow_requested_model": "openai/gpt-5.5",
                        "benchflow_model_alias": "benchflow-openai-gpt-5.5",
                        "request_model": "gpt-5.5",
                    },
                },
            )
        )
    )

    records = load_provider_wire_records(
        trajectory,
        targets=targets,
        fallback_agent="mixed",
        fallback_model=None,
        fallback_auth=AuthMode.API_KEY,
    )

    assert [record["metadata"]["role"] for record in records] == [
        "solver",
        "reviewer",
        "critic",
    ]
    assert all(
        record["metadata"]["role_attribution_complete"] is True for record in records
    )


def test_local_provider_failure_downgrades_mixed_capture(tmp_path: Path) -> None:
    """Guards PR #1057 against treating local LiteLLM failures as responses."""

    trajectory_path = tmp_path / "llm_trajectory.jsonl"
    rows = [
        {
            "request": {"body": {"model": "gpt-5.6", "input": "first"}},
            "response": {
                "status_code": 200,
                "body": {
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            },
            "metadata": {
                "benchflow_requested_model": "openai/gpt-5.6",
                "request_complete": True,
                "response_complete": True,
                "request_capture_source": ("litellm_pre_api_call_complete_input_dict"),
            },
        },
        {
            "request": {"body": {"model": "gpt-5.6", "input": "second"}},
            "response": {
                "status_code": 500,
                "body": {"error": {"message": "connection failed locally"}},
            },
            "metadata": {
                "benchflow_requested_model": "openai/gpt-5.6",
                "request_complete": True,
                "response_complete": False,
                "request_capture_source": ("litellm_pre_api_call_complete_input_dict"),
            },
        },
    ]
    trajectory_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    target = _CaptureTarget(
        agent="codex-acp",
        model="openai/gpt-5.6",
        credential_home="/home/agent",
        auth_mode=AuthMode.API_KEY,
        native=False,
    )

    provider_records = load_provider_wire_records(
        trajectory_path,
        targets=[target],
        fallback_agent="codex-acp",
        fallback_model="openai/gpt-5.6",
        fallback_auth=AuthMode.API_KEY,
    )
    assembly = assemble_capture(
        provider_records=provider_records,
        native_bundles=[],
        targets=[target],
        collection_errors=[],
        model_call_seen=True,
        fallback_auth=AuthMode.API_KEY,
    )

    assert assembly.status is CaptureStatus.PARTIAL
    assert assembly.fidelity is CaptureFidelity.MIXED
    assert assembly.response_complete is False
    assert "provider_response" in assembly.missing_fields


def test_provider_role_attribution_uses_runtime_identity_for_same_model(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 for API-key scene roles sharing one model route."""
    model = "openai/gpt-5.5"
    targets = [
        _CaptureTarget(
            agent="opencode",
            model=model,
            credential_home=f"/home/{role}",
            auth_mode=AuthMode.API_KEY,
            native=False,
            role=role,
        )
        for role in ("solver", "reviewer")
    ]
    trajectory = tmp_path / "llm_trajectory.jsonl"
    trajectory.write_text(
        "".join(
            json.dumps(
                {
                    "request": {"body": {"model": "gpt-5.5"}},
                    "response": {"status_code": 200, "body": {}},
                    "metadata": {
                        "benchflow_agent": "opencode",
                        "benchflow_role": role,
                        "benchflow_requested_model": model,
                    },
                }
            )
            + "\n"
            for role in ("solver", "reviewer")
        )
    )

    records = load_provider_wire_records(
        trajectory,
        targets=targets,
        fallback_agent="opencode",
        fallback_model=model,
        fallback_auth=AuthMode.API_KEY,
    )

    assert [record["metadata"]["role"] for record in records] == [
        "solver",
        "reviewer",
    ]
    assert all(
        record["metadata"]["role_attribution_complete"] is True for record in records
    )


def test_provider_role_attribution_intersects_repeated_role_with_model(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 when scenes reuse a role name with different models."""
    targets = [
        _CaptureTarget(
            agent="opencode",
            model=model,
            credential_home=f"/home/{index}",
            auth_mode=AuthMode.API_KEY,
            native=False,
            role="solver",
        )
        for index, model in enumerate(("openai/gpt-5.5", "openai/gpt-5.6"))
    ]
    trajectory = tmp_path / "llm_trajectory.jsonl"
    trajectory.write_text(
        "".join(
            json.dumps(
                {
                    "request": {"body": {"model": model.rsplit("/", 1)[-1]}},
                    "response": {"status_code": 200, "body": {}},
                    "metadata": {
                        "benchflow_agent": "opencode",
                        "benchflow_role": "solver",
                        "benchflow_requested_model": model,
                    },
                }
            )
            + "\n"
            for model in ("openai/gpt-5.5", "openai/gpt-5.6")
        )
    )

    records = load_provider_wire_records(
        trajectory,
        targets=targets,
        fallback_agent="opencode",
        fallback_model=None,
        fallback_auth=AuthMode.API_KEY,
    )

    assert [record["metadata"]["model"] for record in records] == [
        "openai/gpt-5.5",
        "openai/gpt-5.6",
    ]
    assert all(
        record["metadata"]["role_attribution_complete"] is True for record in records
    )


@pytest.mark.asyncio
async def test_root_sandbox_provider_capture_is_retained_but_audit_only(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against training on root-agent-writable proxy capture."""

    model = "openai/gpt-5.5"
    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="codex-acp",
        model=model,
        session_id="rollout-root",
        started_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
    )
    await capture.prepare_agent(
        None,
        agent="codex-acp",
        model=model,
        agent_env={"OPENAI_API_KEY": "test-key"},
        credential_home="/root",
        sandbox_user=None,
    )
    capture.bind_provider_capture_trust(
        agent="codex-acp",
        model=model,
        credential_home="/root",
        trusted=False,
    )
    capture.trajectory_path.write_text(
        json.dumps(
            {
                "request": {
                    "body": {
                        "model": "gpt-5.5",
                        "messages": [{"role": "user", "content": "solve"}],
                    }
                },
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {"message": {"role": "assistant", "content": "done"}}
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    },
                },
                "metadata": {
                    "benchflow_agent": "codex-acp",
                    "benchflow_role": "primary",
                    "benchflow_requested_model": model,
                },
            }
        )
        + "\n"
    )

    await capture.finalize(None, acp_events=[], model_call_seen=True)

    record = json.loads(capture.trajectory_path.read_text())
    assert record["metadata"]["capture_fidelity"] == "agent_session"
    assert record["metadata"]["capture_custody"] == "agent_writable_sandbox"
    assert capture.manifest.status is CaptureStatus.PARTIAL
    assert capture.manifest.capture_fidelity is CaptureFidelity.AGENT_SESSION
    assert any("shared root custody" in error for error in capture.manifest.errors)


@pytest.mark.asyncio
async def test_malformed_provider_capture_preserves_native_evidence_and_cleanup(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against malformed provider rows suppressing native evidence."""

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="provider-agent",
        model="provider-model",
        session_id="rollout-1",
        started_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
    )
    provider = _CaptureTarget(
        agent="provider-agent",
        model="provider-model",
        credential_home="/home/agent",
        auth_mode=AuthMode.API_KEY,
        native=False,
        role="solver",
    )
    native = _CaptureTarget(
        agent="codex-acp",
        model="gpt-5.6",
        credential_home="/home/agent",
        auth_mode=AuthMode.OAUTH_SUBSCRIPTION,
        native=True,
        role="reviewer",
        native_session_ids=("session-one",),
    )
    for target in (provider, native):
        capture._targets[
            (target.role, target.agent, target.model, target.credential_home)
        ] = target
    capture.trajectory_path.write_text("{malformed provider row\n")
    capture._otel_collector.owned = True
    capture._otel_collector.root_prepared = True
    commands: list[str] = []

    session = tmp_path / "native-session" / "rollout-session-one.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-29T12:00:01Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "native answer"}],
                },
            }
        )
        + "\n"
    )
    native_result = parse_codex_sessions(
        session.parent,
        agent=native.agent,
        session_id="rollout-1",
        started_at=datetime(2026, 8, 29, 11, 59, tzinfo=UTC),
        configured_model=native.model,
    )
    assert native_result is not None

    class RecordingEnv:
        async def exec(self, command, **_kwargs):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def collect_native(env, *, targets):
        assert targets == [native]
        await capture._otel_collector.stop(env)
        return NativeCollection(
            bundles=(NativeCaptureBundle(targets=(native,), result=native_result),)
        )

    capture._native_collector.collect = collect_native
    await capture.finalize(RecordingEnv(), acp_events=[], model_call_seen=True)

    stop_index = next(i for i, command in enumerate(commands) if "old_pid" in command)
    cleanup_index = next(
        i for i, command in enumerate(commands) if "for attempt in 1 2 3" in command
    )
    assert stop_index < cleanup_index
    row = json.loads(capture.trajectory_path.read_text())
    assert row["metadata"]["agent"] == "codex-acp"
    assert capture.manifest.status is CaptureStatus.PARTIAL
    assert any("provider capture parse failed" in e for e in capture.manifest.errors)
    assert capture._otel_collector.owned is False
    assert capture._otel_collector.root_prepared is False


@pytest.mark.asyncio
async def test_manifest_write_failure_preserves_already_assembled_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #1057 against erasing rows after a sidecar write failure."""

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="codex-acp",
        model="openai/gpt-5.6",
        session_id="rollout-1",
        started_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
    )
    target = _CaptureTarget(
        agent="codex-acp",
        model="openai/gpt-5.6",
        credential_home="/home/agent",
        auth_mode=AuthMode.API_KEY,
        native=False,
        role="agent",
    )
    capture._targets[("agent", target.agent, target.model, target.credential_home)] = (
        target
    )
    capture.trajectory_path.write_text(
        json.dumps(
            {
                "request": {"body": {"model": "gpt-5.6", "input": "hello"}},
                "response": {
                    "status_code": 200,
                    "body": {
                        "output": [],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                },
                "metadata": {
                    "benchflow_requested_model": "openai/gpt-5.6",
                    "request_complete": True,
                    "request_capture_source": (
                        "litellm_pre_api_call_complete_input_dict"
                    ),
                },
            }
        )
        + "\n"
    )
    write_manifest = llm_capture_module.write_llm_trajectory_manifest
    calls = 0

    def fail_once(rollout_dir, manifest):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("manifest disk write failed")
        write_manifest(rollout_dir, manifest)

    monkeypatch.setattr(llm_capture_module, "write_llm_trajectory_manifest", fail_once)
    with pytest.raises(OSError, match="manifest disk write failed"):
        await capture.finalize(None, acp_events=[], model_call_seen=True)

    assembled = capture.trajectory_path.read_text()
    capture.record_failure("manifest disk write failed", model_call_seen=True)

    assert capture.trajectory_path.read_text() == assembled
    assert len(assembled.splitlines()) == 1
    manifest = json.loads(
        (tmp_path / "trajectory" / "llm_trajectory.manifest.json").read_text()
    )
    assert manifest["status"] == "capture_failed"
    assert manifest["capture_source"] == "litellm_proxy"
    assert manifest["capture_fidelity"] == "provider_wire"
    assert manifest["exchange_count"] == 1


@pytest.mark.asyncio
async def test_claude_capture_setup_failure_degrades_without_aborting(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against observability setup breaking Claude OAuth runs."""

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

    assert "CLAUDE_CODE_ENABLE_TELEMETRY" not in prepared
    assert "OTEL_LOG_RAW_API_BODIES" not in prepared
    assert "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT" not in prepared
    assert any("/opt/benchflow/node/bin/node" in command for command in commands)


def test_capture_failure_repairs_invalid_jsonl_and_redacts_manifest_error(
    tmp_path: Path,
) -> None:
    """Guards PR #1057's valid-JSONL and secret-redaction failure invariant."""

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
    """Guards PR #1057 against labeling API-key fallback rows as OAuth."""

    result = project_acp_trajectory(
        [{"type": "agent_message", "text": "finished"}],
        agent="codex-acp",
        session_id="rollout-1",
        started_at=STARTED_AT,
        auth_mode="api_key",
    )

    assert result is not None
    assert result.trajectory.exchanges[0].metadata["auth_mode"] == "api_key"
