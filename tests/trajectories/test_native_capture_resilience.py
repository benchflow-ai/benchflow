"""Failure-boundary regressions for native LLM trajectory capture."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.trajectories.llm_capture import LLMTrajectoryCapture, _CaptureTarget
from benchflow.trajectories.llm_capture_manifest import AuthMode
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
