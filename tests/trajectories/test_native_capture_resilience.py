"""Failure-boundary regressions for native LLM trajectory capture."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.trajectories.llm_capture import LLMTrajectoryCapture, _CaptureTarget
from benchflow.trajectories.llm_capture_manifest import AuthMode
from benchflow.trajectories.llm_capture_records import load_provider_wire_records
from benchflow.trajectories.native_capture_parsers import parse_codex_sessions


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


@pytest.mark.asyncio
async def test_malformed_provider_capture_stops_owned_collector_before_cleanup(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against leaking OTel on malformed mixed capture input."""

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
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
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
    capture._collector_owned = True
    capture._capture_root_prepared = True
    commands: list[str] = []

    class RecordingEnv:
        async def exec(self, command, **_kwargs):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    with pytest.raises(ValueError, match="invalid LLM trajectory JSONL"):
        await capture.finalize(
            RecordingEnv(),
            acp_events=[],
            model_call_seen=True,
        )

    stop_index = next(i for i, command in enumerate(commands) if "old_pid" in command)
    cleanup_index = next(
        i for i, command in enumerate(commands) if "for attempt in 1 2 3" in command
    )
    assert stop_index < cleanup_index
    assert capture._collector_owned is False
    assert capture._capture_root_prepared is False
