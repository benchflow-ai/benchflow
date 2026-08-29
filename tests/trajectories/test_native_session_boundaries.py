"""Session-boundary regressions for native OAuth trajectory capture."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchflow.rollout import Rollout
from benchflow.trajectories.llm_capture import (
    LLMTrajectoryCapture,
    _download_bound_session_files,
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
async def test_first_named_role_removes_actual_unbound_provisional_target(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against retaining a different provisional primary."""

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="codex-acp",
        model="gpt-primary",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    await capture.prepare_agent(
        None,
        agent="codex-acp",
        model="gpt-primary",
        agent_env={
            "CODEX_AUTH_JSON": '{"auth_mode":"chatgpt","tokens":{"refresh_token":"test"}}'
        },
        credential_home="/home/agent",
        sandbox_user="agent",
    )
    await capture.prepare_agent(
        None,
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        agent_env={"ANTHROPIC_API_KEY": "test-key"},
        credential_home="/home/agent",
        sandbox_user="agent",
        role_name="reviewer",
    )

    targets = list(capture._targets.values())
    assert [(target.role, target.agent) for target in targets] == [
        ("reviewer", "claude-agent-acp")
    ]
    assert capture._provisional_target_key is None


@pytest.mark.asyncio
async def test_executed_primary_is_not_removed_by_later_named_role(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against deleting an activated primary capture target."""

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="codex-acp",
        model="gpt-primary",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    await capture.prepare_agent(
        None,
        agent="codex-acp",
        model="gpt-primary",
        agent_env={"OPENAI_API_KEY": "test-key"},
        credential_home="/home/agent",
        sandbox_user="agent",
    )
    capture.bind_native_session(
        agent="codex-acp",
        model="gpt-primary",
        credential_home="/home/agent",
        native_session_id="primary-session",
    )
    await capture.prepare_agent(
        None,
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        agent_env={"ANTHROPIC_API_KEY": "test-key"},
        credential_home="/home/agent",
        sandbox_user="agent",
        role_name="reviewer",
    )

    assert {target.role for target in capture._targets.values()} == {
        "primary",
        "reviewer",
    }


@pytest.mark.asyncio
async def test_claude_capture_setup_clears_reused_raw_attempt(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against importing raw bodies from a crashed retry."""

    commands: list[str] = []

    class FailingCollectorEnv:
        async def exec(self, command, **_kwargs):
            commands.append(command)
            if "nohup" in command:
                return SimpleNamespace(return_code=1, stdout="", stderr="no node")
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def upload_file(self, *_args, **_kwargs):
            return None

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        session_id="reused-rollout",
        started_at=STARTED_AT,
    )
    await capture.prepare_agent(
        FailingCollectorEnv(),
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        agent_env={"CLAUDE_CODE_OAUTH_TOKEN": "test-token"},
        credential_home="/home/agent",
        sandbox_user="agent",
    )

    stop, setup = commands[:2]
    assert 'case "$old_pid"' in stop
    assert "otel_sink.mjs*" in stop
    clear_at = setup.index("-mindepth 1 -delete")
    create_at = setup.index("mkdir -p")
    assert clear_at < create_at


@pytest.mark.asyncio
async def test_claude_startup_timeout_still_releases_owned_collector(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against leaking a collector after startup timeout."""

    commands: list[str] = []

    class StartupTimeoutEnv:
        async def exec(self, command, **_kwargs):
            commands.append(command)
            if "nohup" in command:
                return SimpleNamespace(
                    return_code=1,
                    stdout="",
                    stderr="collector port was not observed",
                )
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def upload_file(self, *_args, **_kwargs):
            return None

        async def download_dir(self, _remote, local):
            Path(local).mkdir(parents=True)

    env = StartupTimeoutEnv()
    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    await capture.prepare_agent(
        env,
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        agent_env={"CLAUDE_CODE_OAUTH_TOKEN": "test-token"},
        credential_home="/home/agent",
        sandbox_user="agent",
    )

    assert capture._collector_started is False
    assert capture._collector_owned is True

    await capture.finalize(env, acp_events=[], model_call_seen=False)

    stop_commands = [command for command in commands if "read -r old_pid" in command]
    assert len(stop_commands) == 2
    assert capture._collector_owned is False
    assert any("-depth -delete" in command for command in commands)


@pytest.mark.asyncio
async def test_native_download_selects_recent_files_before_copying(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against copying unrelated concurrent native sessions."""

    commands: list[str] = []
    downloads: list[tuple[str, Path]] = []
    native_session_id = "019effaf-3966-75d3-b61a-2916c84b0ac8"

    class DockerLikeEnv:
        async def exec(self, command, **_kwargs):
            commands.append(command)
            return SimpleNamespace(
                return_code=0,
                stdout=f"2026/08/28/rollout-now-{native_session_id}.jsonl\n",
                stderr="",
            )

        async def download_file(self, remote, local):
            destination = Path(local)
            downloads.append((remote, destination))
            assert destination.parent.is_dir()
            destination.write_text("{}\n")

    destination = tmp_path / "missing-parent" / "sessions"
    downloaded = await _download_bound_session_files(
        DockerLikeEnv(),
        "/home/agent/.codex/sessions",
        destination,
        started_at=STARTED_AT,
        session_ids=(native_session_id,),
    )

    assert downloaded is True
    assert len(downloads) == 1
    assert downloads[0][0].endswith(
        f"/2026/08/28/rollout-now-{native_session_id}.jsonl"
    )
    assert downloads[0][1].is_file()
    assert "-newermt" in commands[0]
    assert f"-name {native_session_id}.jsonl" in commands[0]
    assert f"-name '*-{native_session_id}.jsonl'" in commands[0]


@pytest.mark.asyncio
async def test_native_capture_binds_only_returned_acp_session_ids(
    tmp_path: Path,
) -> None:
    """Guards PR #1057's exact ACP-to-native-session ownership boundary."""

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
        agent_env={"CODEX_AUTH_JSON": '{"tokens":{"access_token":"test"}}'},
        credential_home="/home/agent",
        sandbox_user="agent",
    )

    capture.bind_native_session(
        agent="codex-acp",
        model="gpt-5.6",
        credential_home="/home/agent",
        native_session_id="019effaf-3966-75d3-b61a-2916c84b0ac8",
    )
    capture.bind_native_session(
        agent="codex-acp",
        model="gpt-5.6",
        credential_home="/home/agent",
        native_session_id="019effaf-3ab7-71f1-8ff3-fdecf66b551e",
    )

    target = capture._native_targets()[0]
    assert target.native_session_ids == (
        "019effaf-3966-75d3-b61a-2916c84b0ac8",
        "019effaf-3ab7-71f1-8ff3-fdecf66b551e",
    )


@pytest.mark.asyncio
async def test_replaced_native_target_still_releases_collector(tmp_path: Path) -> None:
    """Guards PR #1057 against leaking a replaced OAuth collector."""

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
    capture.configure({"ANTHROPIC_API_KEY": "test-key"})
    capture._collector_started = True
    capture._collector_owned = True
    capture._capture_root_prepared = True

    await capture.finalize(
        CleanupEnv(),
        acp_events=[],
        model_call_seen=False,
    )

    assert any("kill -TERM" in command for command in commands)
    assert any("-depth -delete" in command for command in commands)
    assert capture._collector_started is False
    assert capture._collector_owned is False


@pytest.mark.asyncio
async def test_raw_capture_cleanup_failure_marks_manifest_failed(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against hiding retained raw provider bodies."""

    commands: list[str] = []

    class CleanupFailureEnv:
        async def exec(self, command, **_kwargs):
            commands.append(command)
            return SimpleNamespace(
                return_code=1,
                stdout="",
                stderr="capture directory deletion failed",
            )

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="codex-acp",
        model="gpt-5.6",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    capture.configure({"OPENAI_API_KEY": "test-key"})
    capture._capture_root_prepared = True
    capture.trajectory_path.write_text(
        json.dumps(
            {
                "request": {"body": {"model": "gpt-5.6", "input": "hello"}},
                "response": {"status_code": 200, "body": {"output": []}},
            }
        )
        + "\n"
    )

    await capture.finalize(
        CleanupFailureEnv(),
        acp_events=[],
        model_call_seen=True,
    )

    manifest = json.loads(
        (tmp_path / "trajectory" / "llm_trajectory.manifest.json").read_text()
    )
    assert "for attempt in 1 2 3" in commands[0]
    assert manifest["status"] == "capture_failed"
    assert manifest["exchange_count"] == 1
    assert any("cleanup failed" in error for error in manifest["errors"])
    assert capture._capture_root_prepared is True


def test_rollout_binds_the_acp_session_after_connect() -> None:
    """Guards PR #1057's rollout-to-capture session binding seam."""

    rollout = object.__new__(Rollout)
    capture = SimpleNamespace(bind_native_session=Mock())
    rollout._llm_capture = capture
    rollout._session = SimpleNamespace(session_id="native-session-1")

    rollout._bind_llm_capture_session(
        agent="codex-acp",
        model="gpt-5.6",
        credential_home="/home/agent",
        role_name="solver",
    )

    capture.bind_native_session.assert_called_once_with(
        agent="codex-acp",
        model="gpt-5.6",
        credential_home="/home/agent",
        native_session_id="native-session-1",
        role_name="solver",
    )


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
