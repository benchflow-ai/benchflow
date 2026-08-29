"""Native OpenRouter Ori harness adapter tests."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.acp.types import StopReason
from benchflow.agents.ori import ORI_BINARY, OriAgent, OriSession
from benchflow.agents.registry import AGENTS
from benchflow.rollout.session_factory_runtime import SessionFactorySandbox
from benchflow.usage_tracking import USAGE_SOURCE_AGENT_NATIVE


def _jsonl(*documents: dict) -> str:
    return "".join(json.dumps(document) + "\n" for document in documents)


def _runtime(event_type: str, payload: dict, *, session_id: str = "ori-session") -> dict:
    return {
        "kind": "event",
        "event": {
            "type": "runtime.event",
            "event": {
                "createdAt": "2026-08-29T00:00:00Z",
                "eventId": f"run:{event_type}",
                "harness": "ori",
                "model": "benchflow-alias",
                "runId": "run",
                "sessionId": session_id,
                "turnId": "turn",
                "payload": payload,
                "type": event_type,
            },
        },
    }


class _FakeSandbox:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = list(outputs or [])
        self.exec_calls: list[tuple[str, dict]] = []
        self.uploads: dict[str, str] = {}
        self.files: dict[str, str] = {}

    async def exec(self, command: str, **kwargs):
        self.exec_calls.append((command, kwargs))
        if command.startswith(ORI_BINARY):
            parts = shlex.split(command)
            output_path = parts[parts.index(">") + 1]
            self.files[output_path] = self.outputs.pop(0)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source: Path, destination: str) -> None:
        self.uploads[destination] = source.read_text()

    async def download_file(self, source: str, destination: Path) -> None:
        destination.write_text(self.files[source])


def _first_turn() -> str:
    return _jsonl(
        {
            "kind": "event",
            "event": {
                "type": "audit.event",
                "audit": {"message": "accepted agent command"},
            },
        },
        _runtime("session.started", {"sessionId": "ori-session"}),
        _runtime(
            "tool.started",
            {
                "input": {"command": "pwd"},
                "name": "bash",
                "toolCallId": "call-1",
            },
        ),
        _runtime(
            "tool.progress",
            {
                "name": "bash",
                "partialResult": "/workspace\n",
                "toolCallId": "call-1",
            },
        ),
        _runtime(
            "tool.succeeded",
            {"durationMs": 12, "name": "bash", "toolCallId": "call-1"},
        ),
        _runtime("assistant.text.delta", {"delta": "done"}),
        _runtime("assistant.text.delta", {"delta": "!"}),
        _runtime(
            "turn.succeeded",
            {
                "usage": {
                    "cacheCreationTokens": 2,
                    "cacheReadTokens": 3,
                    "contextTokens": 999,
                    "inputTokens": 10,
                    "outputTokens": 6,
                }
            },
        ),
        {"kind": "result", "ok": True, "sessionId": "ori-session"},
    )


def _second_turn() -> str:
    return _jsonl(
        _runtime("assistant.text.delta", {"delta": "again"}),
        _runtime(
            "turn.succeeded",
            {
                "usage": {
                    "cacheCreationTokens": 0,
                    "cacheReadTokens": 1,
                    "contextTokens": 500,
                    "inputTokens": 4,
                    "outputTokens": 2,
                }
            },
        ),
        {"kind": "result", "ok": True, "sessionId": "ori-session"},
    )


@pytest.mark.asyncio
async def test_ori_session_runs_jsonl_tools_usage_and_resumes() -> None:
    sandbox = _FakeSandbox([_first_turn(), _second_turn()])
    session = OriSession(
        sandbox,
        agent_env={
            "ORI_MODEL": "benchflow-alias",
            "ORI_OPENROUTER_BASE_URL": "http://proxy/v1",
            "OPENROUTER_API_KEY": "proxy-key",
        },
        cwd="/workspace",
        exec_user="agent",
        reasoning_effort="max",
        command_timeout=123,
        runtime_dir="/tmp/benchflow-ori-test",
    )

    assert await session.prompt("first") is StopReason.END_TURN
    assert await session.prompt("second") is StopReason.END_TURN

    ori_commands = [call for call in sandbox.exec_calls if call[0].startswith(ORI_BINARY)]
    assert len(ori_commands) == 2
    first_command, first_kwargs = ori_commands[0]
    second_command, second_kwargs = ori_commands[1]
    assert "--harness ori" in first_command
    assert "--model benchflow-alias" in first_command
    assert "--reasoning-effort max" in first_command
    assert "--approvals self-drive" in first_command
    assert "--output jsonl" in first_command
    assert "--session" not in first_command
    assert "--session ori-session" in second_command
    assert first_kwargs["cwd"] == second_kwargs["cwd"] == "/workspace"
    assert first_kwargs["user"] == second_kwargs["user"] == "agent"
    assert first_kwargs["timeout_sec"] == second_kwargs["timeout_sec"] == 123
    assert first_kwargs["env"]["ORI_TELEMETRY"] == "0"
    assert first_kwargs["env"]["CI"] == "true"

    tool = next(step for step in session.steps if step.get("type") == "tool_call")
    assert tool["tool_call_id"] == "call-1"
    assert tool["kind"] == "bash"
    assert tool["title"] == "pwd"
    assert tool["status"] == "completed"
    assert tool["content"][0]["content"]["text"] == "/workspace\n"
    assert len(tool["ori_events"]) == 3
    assert session.tool_call_count == 1
    assert session.session_id == "ori-session"
    assert [
        step["text"] for step in session.steps if step.get("type") == "agent_message"
    ] == ["done!", "again"]
    assert [
        step["text"] for step in session.steps if step.get("type") == "user_message"
    ] == ["first", "second"]
    assert session.latest_usage_totals() == {
        "input_tokens": 14,
        "output_tokens": 8,
        "cached_read_tokens": 4,
        "cached_write_tokens": 2,
        "thought_tokens": 0,
        "total_tokens": 22,
    }
    assert session.usage_source == USAGE_SOURCE_AGENT_NATIVE


@pytest.mark.asyncio
async def test_ori_session_surfaces_terminal_cli_failure() -> None:
    failed = _jsonl(
        _runtime("turn.failed", {"failure": {"code": "ORI_PROVIDER_FAILURE"}}),
        {
            "kind": "result",
            "ok": False,
            "error": {"message": "provider rejected the request"},
        },
    )
    sandbox = _FakeSandbox([failed])

    async def failing_exec(command: str, **kwargs):
        result = await _FakeSandbox.exec(sandbox, command, **kwargs)
        if command.startswith(ORI_BINARY):
            result.return_code = 1
        return result

    sandbox.exec = failing_exec
    session = OriSession(
        sandbox,
        agent_env={"ORI_MODEL": "benchflow-alias"},
        cwd="/workspace",
        exec_user=None,
        reasoning_effort=None,
        command_timeout=30,
        runtime_dir="/tmp/benchflow-ori-test",
    )

    with pytest.raises(RuntimeError, match="provider rejected the request"):
        await session.prompt("fail")
    assert any(step.get("type") == "ori_event" for step in session.steps)


@pytest.mark.asyncio
async def test_ori_agent_prepares_minimal_offline_workspace() -> None:
    sandbox = _FakeSandbox()
    wrapped = SessionFactorySandbox(
        sandbox,
        {"ORI_MODEL": "anthropic/claude-sonnet-4.6"},
        "/workspace",
        "xhigh",
        456,
    )

    session = await OriAgent(exec_user="agent").connect(wrapped, "agent")

    assert isinstance(session, OriSession)
    setup_command, setup_kwargs = sandbox.exec_calls[0]
    assert "/home/agent/.ori/global/ori.md" in setup_command
    assert "/home/agent/.ori/global/package.json" in setup_command
    assert "base64 -d" in setup_command
    assert setup_kwargs["user"] == "agent"


def test_ori_registry_contract_is_pinned_and_routable() -> None:
    cfg = AGENTS["ori"]

    assert cfg.protocol == "session-factory"
    assert cfg.session_factory == "benchflow.agents.ori:build_ori_agent"
    assert cfg.default_model == "openrouter/openrouter/auto"
    assert cfg.api_protocol == "openai-completions"
    assert cfg.env_mapping == {
        "BENCHFLOW_PROVIDER_BASE_URL": "ORI_OPENROUTER_BASE_URL",
        "BENCHFLOW_PROVIDER_API_KEY": "OPENROUTER_API_KEY",
        "BENCHFLOW_PROVIDER_MODEL": "ORI_MODEL",
    }
    assert "cli-0.12.0-68f9a36" in cfg.install_cmd
    assert "2dffa9f311f8b65f" in cfg.install_cmd
    assert "d3ee260046c313a7" in cfg.install_cmd
    assert "ori-releases/releases/download" in cfg.install_cmd
    assert "install.sh" not in cfg.install_cmd
    assert cfg.subscription_auth is not None
    assert cfg.subscription_auth.detect_files == [
        "~/.ori/credentials.json",
        "~/.openrouter/credentials.json",
    ]


def test_ori_rejects_acpx_wrapper() -> None:
    from benchflow.agents.registry import resolve_agent

    with pytest.raises(KeyError, match="cannot be wrapped by ACPX"):
        resolve_agent("acpx/ori")


def test_ori_native_login_gate_only_applies_to_openrouter_models() -> None:
    from benchflow.agents.env import uses_native_subscription_auth

    marker = {"_BENCHFLOW_SUBSCRIPTION_AUTH": "1"}
    assert uses_native_subscription_auth(
        "ori", "openrouter/anthropic/claude-sonnet-4.6", marker
    )
    assert not uses_native_subscription_auth(
        "ori",
        "openrouter/anthropic/claude-sonnet-4.6",
        {**marker, "OPENROUTER_API_KEY": "api-key-wins"},
    )
    assert not uses_native_subscription_auth(
        "ori", "deepseek/deepseek-v4-flash", marker
    )


def test_ori_login_detects_openrouter_fallback_file(monkeypatch, tmp_path: Path) -> None:
    from benchflow.agents.env import check_subscription_auth

    fallback = tmp_path / ".openrouter" / "credentials.json"
    fallback.parent.mkdir()
    fallback.write_text('{"key":"saved"}')
    auth = AGENTS["ori"].subscription_auth
    assert auth is not None
    monkeypatch.setattr(
        auth,
        "detect_files",
        [str(tmp_path / ".ori" / "credentials.json"), str(fallback)],
    )

    assert check_subscription_auth("ori", "OPENROUTER_API_KEY")


def test_ori_openrouter_provider_maps_native_cli_environment() -> None:
    from benchflow.agents.env import resolve_agent_env

    resolved = resolve_agent_env(
        "ori",
        "openrouter/anthropic/claude-sonnet-4.6",
        {"OPENROUTER_API_KEY": "test-openrouter-key"},
    )

    assert resolved["ORI_OPENROUTER_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert resolved["OPENROUTER_API_KEY"] == "test-openrouter-key"
    assert resolved["ORI_MODEL"] == "anthropic/claude-sonnet-4.6"


def test_ori_usage_is_recorded_as_trusted_non_acp_native_usage() -> None:
    from benchflow.rollout import Rollout

    session = SimpleNamespace(
        usage_source=USAGE_SOURCE_AGENT_NATIVE,
        latest_usage_totals=lambda: {
            "input_tokens": 11,
            "output_tokens": 7,
            "cached_read_tokens": 2,
            "cached_write_tokens": 1,
            "thought_tokens": 0,
            "total_tokens": 18,
        },
    )
    rollout = Rollout.__new__(Rollout)
    rollout._session = session
    rollout._native_usage_checkpoint = None

    rollout._collect_native_acp_usage()

    assert rollout._native_usage_metrics["usage_source"] == (
        USAGE_SOURCE_AGENT_NATIVE
    )
    assert rollout._native_usage_metrics["n_input_tokens"] == 11
    assert rollout._native_usage_metrics["n_output_tokens"] == 7
    assert rollout._native_usage_metrics["total_tokens"] == 18
