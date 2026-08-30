"""Session-boundary regressions for native OAuth trajectory capture."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchflow.rollout import Rollout
from benchflow.trajectories import (
    native_capture_collection as native_capture_module,
)
from benchflow.trajectories.llm_capture import (
    LLMTrajectoryCapture,
    _CaptureTarget,
    _NativeCaptureBundle,
)
from benchflow.trajectories.llm_capture_manifest import (
    AuthMode,
    CaptureFidelity,
    CaptureSource,
)
from benchflow.trajectories.native_capture_collection import (
    download_bound_session_files,
)
from benchflow.trajectories.native_capture_parsers import (
    NativeParseResult,
    parse_codex_sessions,
)
from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
)

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


def _native_result(
    *,
    session_id: str,
    source: CaptureSource,
    fidelity: CaptureFidelity,
    model: str,
    request_id: str | None = None,
    response_id: str | None = None,
) -> NativeParseResult:
    response_body: dict[str, object] = {"output": []}
    if response_id is not None:
        response_body["id"] = response_id
    metadata: dict[str, object] = {
        "capture_source": source.value,
        "capture_fidelity": fidelity.value,
        "auth_mode": AuthMode.OAUTH_SUBSCRIPTION.value,
        "native_session_id": session_id,
        "request_complete": fidelity is CaptureFidelity.PROVIDER_WIRE,
        "response_complete": True,
    }
    if request_id is not None:
        metadata["provider_request_id"] = request_id
    exchange = LLMExchange(
        request=LLMRequest(body={"model": model, "input": "hello"}),
        response=LLMResponse(body=response_body),
        metadata=metadata,
    )
    return NativeParseResult(
        trajectory=Trajectory(
            session_id="rollout-1",
            agent_name="agent",
            exchanges=[exchange],
        ),
        source=source,
        fidelity=fidelity,
        request_complete=fidelity is CaptureFidelity.PROVIDER_WIRE,
        response_complete=True,
        missing_fields=(
            [] if fidelity is CaptureFidelity.PROVIDER_WIRE else ["request"]
        ),
    )


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
async def test_repeated_role_preserves_auth_distinct_capture_targets(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against API auth replacing a bound OAuth role target."""

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="codex-acp",
        model="gpt-5.6",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    common = {
        "env": None,
        "agent": "codex-acp",
        "model": "gpt-5.6",
        "credential_home": "/home/agent",
        "sandbox_user": "agent",
        "role_name": "reviewer",
    }
    await capture.prepare_agent(
        **common,
        agent_env={
            "CODEX_AUTH_JSON": (
                '{"auth_mode":"chatgpt","tokens":{"refresh_token":"test"}}'
            )
        },
    )
    capture.bind_native_session(
        agent="codex-acp",
        model="gpt-5.6",
        credential_home="/home/agent",
        native_session_id="oauth-session",
        role_name="reviewer",
    )
    await capture.prepare_agent(
        **common,
        agent_env={"OPENAI_API_KEY": "test-key"},
    )

    targets = list(capture._targets.values())
    assert {target.auth_mode for target in targets} == {
        AuthMode.OAUTH_SUBSCRIPTION,
        AuthMode.API_KEY,
    }
    oauth_target = next(
        target for target in targets if target.auth_mode is AuthMode.OAUTH_SUBSCRIPTION
    )
    assert oauth_target.native_session_ids == ("oauth-session",)


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

    assert capture._otel_collector.started is False
    assert capture._otel_collector.owned is True

    await capture.finalize(env, acp_events=[], model_call_seen=False)

    stop_commands = [command for command in commands if "read -r old_pid" in command]
    assert len(stop_commands) == 2
    assert capture._otel_collector.owned is False
    assert any("-depth -delete" in command for command in commands)


@pytest.mark.asyncio
async def test_claude_capture_setup_error_is_cleared_after_recovery(
    tmp_path: Path,
) -> None:
    """Guards PR #1057 against retaining a recovered setup failure."""

    class RecoveringCollectorEnv:
        def __init__(self) -> None:
            self.launch_attempts = 0

        async def exec(self, command, **_kwargs):
            if "nohup" not in command:
                return SimpleNamespace(return_code=0, stdout="", stderr="")
            self.launch_attempts += 1
            if self.launch_attempts == 1:
                return SimpleNamespace(
                    return_code=1,
                    stdout="",
                    stderr="collector port was not observed",
                )
            return SimpleNamespace(return_code=0, stdout="43123\n", stderr="")

        async def upload_file(self, *_args, **_kwargs):
            return None

    env = RecoveringCollectorEnv()
    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    first_env = await capture.prepare_agent(
        env,
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        agent_env={"CLAUDE_CODE_OAUTH_TOKEN": "test-token"},
        credential_home="/home/agent",
        sandbox_user="agent",
    )
    assert "OTEL_LOGS_EXPORTER" not in first_env
    assert capture._otel_setup_error is not None

    recovered_env = await capture.prepare_agent(
        env,
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        agent_env={"CLAUDE_CODE_OAUTH_TOKEN": "test-token"},
        credential_home="/home/agent",
        sandbox_user="agent",
        role_name="solver",
    )

    assert recovered_env["OTEL_LOGS_EXPORTER"] == "otlp"
    assert capture._otel_setup_error is None


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
    downloaded = await download_bound_session_files(
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
async def test_native_role_reprepare_preserves_prior_session_ids(
    tmp_path: Path,
) -> None:
    """Guards PR #1057's prepare-bind reconnect sequence across rounds."""

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="codex-acp",
        model="gpt-5.6",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    prepare_kwargs = {
        "agent": "codex-acp",
        "model": "gpt-5.6",
        "agent_env": {
            "CODEX_AUTH_JSON": '{"auth_mode":"chatgpt","tokens":{"refresh_token":"test"}}'
        },
        "credential_home": "/home/agent",
        "sandbox_user": "agent",
        "role_name": "solver",
    }
    await capture.prepare_agent(None, **prepare_kwargs)
    capture.bind_native_session(
        agent="codex-acp",
        model="gpt-5.6",
        credential_home="/home/agent",
        native_session_id="019effaf-3966-75d3-b61a-2916c84b0ac8",
        role_name="solver",
    )

    await capture.prepare_agent(None, **prepare_kwargs)
    capture.bind_native_session(
        agent="codex-acp",
        model="gpt-5.6",
        credential_home="/home/agent",
        native_session_id="019effaf-3ab7-71f1-8ff3-fdecf66b551e",
        role_name="solver",
    )

    target = capture._native_targets()[0]
    assert target.native_session_ids == (
        "019effaf-3966-75d3-b61a-2916c84b0ac8",
        "019effaf-3ab7-71f1-8ff3-fdecf66b551e",
    )


@pytest.mark.asyncio
async def test_claude_fallback_collects_only_raw_uncovered_exchanges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards PR #1057 against losing a partially raw-captured Claude session."""

    first_id = "019effaf-3966-75d3-b61a-2916c84b0ac8"
    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    target = _CaptureTarget(
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        credential_home="/home/agent",
        auth_mode=AuthMode.OAUTH_SUBSCRIPTION,
        native=True,
        role="solver",
        native_session_ids=(first_id,),
    )
    capture._targets[
        (target.role, target.agent, target.model, target.credential_home)
    ] = target
    capture._otel_collector.root_prepared = True
    raw_result = _native_result(
        session_id=first_id,
        source=CaptureSource.CLAUDE_OTEL_RAW_BODY,
        fidelity=CaptureFidelity.PROVIDER_WIRE,
        model="claude-sonnet-4-6",
        request_id="request-covered",
        response_id="message-covered",
    )
    covered_fallback = _native_result(
        session_id=first_id,
        source=CaptureSource.CLAUDE_NATIVE_SESSION,
        fidelity=CaptureFidelity.AGENT_SESSION,
        model="claude-sonnet-4-6",
        request_id="request-covered",
        response_id="message-covered",
    )
    missing_fallback = _native_result(
        session_id=first_id,
        source=CaptureSource.CLAUDE_NATIVE_SESSION,
        fidelity=CaptureFidelity.AGENT_SESSION,
        model="claude-sonnet-4-6",
        request_id="request-missing",
        response_id="message-missing",
    )
    fallback_result = replace(
        covered_fallback,
        trajectory=covered_fallback.trajectory.model_copy(
            update={
                "exchanges": [
                    *covered_fallback.trajectory.exchanges,
                    *missing_fallback.trajectory.exchanges,
                ]
            }
        ),
    )
    fallback_calls: list[tuple[str, ...]] = []

    class CaptureEnv:
        async def download_dir(self, _remote, local):
            Path(local).mkdir(parents=True)

    async def download_bound(_env, _remote, local, *, started_at, session_ids):
        del started_at
        fallback_calls.append(session_ids)
        Path(local).mkdir(parents=True)
        return True

    monkeypatch.setattr(
        native_capture_module,
        "parse_claude_raw_capture",
        lambda *_args, **_kwargs: raw_result,
    )
    monkeypatch.setattr(
        native_capture_module,
        "download_bound_session_files",
        download_bound,
    )
    monkeypatch.setattr(
        native_capture_module,
        "parse_claude_sessions",
        lambda *_args, **_kwargs: fallback_result,
    )

    collection = await capture._native_collector.collect(CaptureEnv(), targets=[target])

    assert len(collection.bundles) == 2
    assert fallback_calls == [(first_id,)]
    fallback_exchanges = collection.bundles[1].result.trajectory.exchanges
    assert len(fallback_exchanges) == 1
    assert fallback_exchanges[0].metadata["provider_request_id"] == "request-missing"
    assert fallback_exchanges[0].metadata["native_session_id"] == first_id
    assert collection.errors == ()


@pytest.mark.asyncio
async def test_later_native_target_failure_preserves_prior_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards PR #1057 against discarding earlier native role evidence."""

    capture = LLMTrajectoryCapture(
        tmp_path,
        agent="codex-acp",
        model="gpt-one",
        session_id="rollout-1",
        started_at=STARTED_AT,
    )
    first = _CaptureTarget(
        agent="codex-acp",
        model="gpt-one",
        credential_home="/home/agent",
        auth_mode=AuthMode.OAUTH_SUBSCRIPTION,
        native=True,
        role="solver",
        native_session_ids=("session-one",),
    )
    second = _CaptureTarget(
        agent="codex-acp",
        model="gpt-two",
        credential_home="/home/agent",
        auth_mode=AuthMode.OAUTH_SUBSCRIPTION,
        native=True,
        role="reviewer",
        native_session_ids=("session-two",),
    )
    for target in (first, second):
        capture._targets[
            (target.role, target.agent, target.model, target.credential_home)
        ] = target
    first_result = _native_result(
        session_id="session-one",
        source=CaptureSource.CODEX_NATIVE_SESSION,
        fidelity=CaptureFidelity.AGENT_SESSION,
        model="gpt-one",
    )

    async def collect_codex(_env, *, local_root, index, target):
        del local_root, index
        if target is second:
            raise RuntimeError("second target download failed")
        return _NativeCaptureBundle(targets=(first,), result=first_result)

    monkeypatch.setattr(
        capture._native_collector, "_collect_codex_session", collect_codex
    )

    await capture.finalize(object(), acp_events=[], model_call_seen=True)

    manifest = json.loads(
        (tmp_path / "trajectory" / "llm_trajectory.manifest.json").read_text()
    )
    rows = [
        json.loads(line) for line in capture.trajectory_path.read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["metadata"]["role"] == "solver"
    assert manifest["exchange_count"] == 1
    assert manifest["status"] == "partial"
    assert any("role reviewer" in error for error in manifest["errors"])


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
    capture._otel_collector.started = True
    capture._otel_collector.owned = True
    capture._otel_collector.root_prepared = True

    await capture.finalize(
        CleanupEnv(),
        acp_events=[],
        model_call_seen=False,
    )

    assert any("kill -TERM" in command for command in commands)
    assert any("-depth -delete" in command for command in commands)
    assert capture._otel_collector.started is False
    assert capture._otel_collector.owned is False


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
    capture._otel_collector.root_prepared = True
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
    assert capture._otel_collector.root_prepared is True


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
