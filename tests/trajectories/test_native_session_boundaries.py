"""Session-boundary regressions for native OAuth trajectory capture."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.trajectories.llm_capture import (
    LLMTrajectoryCapture,
    _download_recent_session_files,
)
from benchflow.trajectories.native_capture_parsers import parse_codex_sessions

STARTED_AT = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _codex_session(prompt: str, answer: str, second: int) -> list[dict]:
    return [
        {
            "type": "response_item",
            "timestamp": f"2026-08-28T12:00:{second:02d}Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        },
        {
            "type": "response_item",
            "timestamp": f"2026-08-28T12:00:{second + 1:02d}Z",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": answer}],
            },
        },
        {
            "type": "event_msg",
            "timestamp": f"2026-08-28T12:00:{second + 2:02d}Z",
            "payload": {
                "type": "token_count",
                "info": {"input_tokens": 10, "output_tokens": 2},
            },
        },
    ]


@pytest.mark.asyncio
async def test_phase_api_registers_provisional_primary_oauth_target(
    tmp_path: Path,
) -> None:
    """Guards PR #1057's native capture when no scene role name is supplied."""

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="codex-acp",
        model="gpt-5.6",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    await capture.prepare_agent(
        None,
        agent="codex-acp",
        model="gpt-5.6",
        agent_env={
            "CODEX_AUTH_JSON": '{"auth_mode":"chatgpt","tokens":{"refresh_token":"test"}}'
        },
        credential_home="/home/agent",
        sandbox_user="agent",
    )

    targets = capture._native_targets()
    assert len(targets) == 1
    assert targets[0].role == "primary"


@pytest.mark.asyncio
async def test_native_download_selects_recent_files_before_copying(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against copying an entire reused native-session tree."""

    commands: list[str] = []
    downloads: list[tuple[str, Path]] = []

    class DockerLikeEnv:
        async def exec(self, command, **_kwargs):
            commands.append(command)
            return SimpleNamespace(
                return_code=0,
                stdout="2026/08/28/session.jsonl\n",
                stderr="",
            )

        async def download_file(self, remote, local):
            destination = Path(local)
            downloads.append((remote, destination))
            assert destination.parent.is_dir()
            destination.write_text("{}\n")

    destination = tmp_path / "missing-parent" / "sessions"
    downloaded = await _download_recent_session_files(
        DockerLikeEnv(),
        "/home/agent/.codex/sessions",
        destination,
        started_at=STARTED_AT,
    )

    assert downloaded is True
    assert len(downloads) == 1
    assert downloads[0][0].endswith("/2026/08/28/session.jsonl")
    assert downloads[0][1].is_file()
    assert "-newermt" in commands[0]


def test_codex_session_files_do_not_share_request_history(tmp_path: Path) -> None:
    """Guards PR #1057 against merging unrelated reconnect histories."""

    sessions = tmp_path / "codex"
    _write_jsonl(sessions / "one.jsonl", _codex_session("first prompt", "one", 1))
    _write_jsonl(sessions / "two.jsonl", _codex_session("second prompt", "two", 11))

    result = parse_codex_sessions(
        sessions,
        agent="codex-acp",
        session_id="rollout-1",
        started_at=STARTED_AT,
        configured_model="gpt-5.6",
    )

    assert result is not None
    assert len(result.trajectory.exchanges) == 2
    first_input = result.trajectory.exchanges[0].request.body["input"]
    second_input = result.trajectory.exchanges[1].request.body["input"]
    assert [item["content"][0]["text"] for item in first_input] == ["first prompt"]
    assert [item["content"][0]["text"] for item in second_input] == ["second prompt"]
