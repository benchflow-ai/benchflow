"""OpenRouter Ori ACP-shim integration tests."""

from __future__ import annotations

import json
from pathlib import Path

from benchflow.agents import ori_acp_shim as shim
from benchflow.agents.ori_events import TurnTranslator
from benchflow.agents.ori_jsonl import OriUsage, decode_line
from benchflow.agents.registry import AGENTS


def _runtime(event_type: str, payload: dict, *, session_id: str = "native-ori"):
    return {
        "kind": "event",
        "event": {
            "type": "runtime.event",
            "event": {
                "type": event_type,
                "payload": payload,
                "sessionId": session_id,
            },
        },
    }


def test_ori_decoder_preserves_diagnostic_before_structured_result() -> None:
    """Guards review feedback on PR #1067: plain prelude must not lose JSON evidence."""
    diagnostic = decode_line("provider bootstrap warning\n", 1)
    result = decode_line('{"kind":"result","ok":false}\n', 2)

    assert diagnostic is not None
    assert diagnostic.kind == "diagnostic"
    assert diagnostic.raw == "provider bootstrap warning"
    assert result is not None
    assert result.kind == "document"
    assert result.document == {"kind": "result", "ok": False}


def test_ori_usage_total_includes_cache_components() -> None:
    """Guards review feedback on PR #1067: ACP total includes every component."""
    usage = OriUsage.from_ori(
        {
            "inputTokens": 10,
            "outputTokens": 6,
            "cacheReadTokens": 3,
            "cacheCreationTokens": 2,
            "contextTokens": 999,
        }
    )

    assert usage is not None
    assert usage.as_acp() == {
        "inputTokens": 10,
        "outputTokens": 6,
        "cachedReadTokens": 3,
        "cachedWriteTokens": 2,
        "thoughtTokens": 0,
        "totalTokens": 21,
    }


def test_ori_failed_result_remains_available_after_plain_diagnostic() -> None:
    """Guards review feedback on PR #1067: failure diagnostics and result both survive."""
    messages: list[dict] = []
    translator = TurnTranslator("acp-session", messages.append)
    translator.consume_diagnostic(1, "provider rejected request")
    translator.consume_document(
        {
            "kind": "result",
            "ok": False,
            "error": {"message": "invalid model"},
            "sessionId": "native-failed",
        }
    )

    assert translator.result is not None
    assert translator.result["error"]["message"] == "invalid model"
    assert translator.native_session_id == "native-failed"
    evidence = "".join(
        message["params"]["update"]["content"]["text"] for message in messages
    )
    assert "provider rejected request" in evidence
    assert "invalid model" in evidence


def test_ori_command_is_argv_and_resumes_native_session() -> None:
    """Guards PR #1067: prompt text stays in a file and follow-up turns resume."""
    command = shim.build_ori_command(
        model="anthropic/claude-sonnet-4.6",
        prompt_path="/tmp/prompt with spaces.txt",
        reasoning_effort="max",
        native_session_id="native-session",
    )

    assert command[:5] == [
        shim.ORI_BINARY,
        "code",
        "--harness",
        "ori",
        "--model",
    ]
    assert command[command.index("--prompt-file") + 1] == (
        "/tmp/prompt with spaces.txt"
    )
    assert command[command.index("--session") + 1] == "native-session"
    assert command[command.index("--reasoning-effort") + 1] == "max"


def test_ori_acp_server_streams_tools_diagnostics_usage_and_resume(
    monkeypatch, tmp_path: Path
) -> None:
    """Guards PR #1067: Ori is an honest multi-turn ACP adapter with evidence."""
    fake_ori = tmp_path / "fake-ori"
    argv_log = tmp_path / "argv.jsonl"
    fake_ori.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "prompt = pathlib.Path(args[args.index('--prompt-file') + 1]).read_text()\n"
        "with open(os.environ['ORI_FAKE_ARGV_LOG'], 'a', encoding='utf-8') as f:\n"
        "    f.write(json.dumps({'args': args, 'prompt': prompt}) + '\\n')\n"
        "print('provider bootstrap warning')\n"
        "def runtime(kind, payload):\n"
        "    return {'kind':'event','event':{'type':'runtime.event','event':"
        "{'type':kind,'payload':payload,'sessionId':'native-ori'}}}\n"
        "print(json.dumps(runtime('tool.started', {'toolCallId':'call-1',"
        "'name':'bash','input':{'command':'pwd'}})))\n"
        "print(json.dumps(runtime('tool.succeeded', {'toolCallId':'call-1',"
        "'name':'bash','result':'/workspace'})))\n"
        "print(json.dumps(runtime('assistant.text.delta', {'delta':'done'})))\n"
        "print(json.dumps(runtime('turn.succeeded', {'usage':{'inputTokens':10,"
        "'outputTokens':6,'cacheReadTokens':3,'cacheCreationTokens':2}})))\n"
        "print(json.dumps({'kind':'result','ok':True,'sessionId':'native-ori'}))\n",
        encoding="utf-8",
    )
    fake_ori.chmod(0o755)
    monkeypatch.setattr(shim, "ORI_BINARY", str(fake_ori))
    monkeypatch.setenv("ORI_FAKE_ARGV_LOG", str(argv_log))
    messages: list[dict] = []
    monkeypatch.setattr(shim, "send", messages.append)

    server = shim.OriACPServer()
    server.sessions["acp-session"] = shim.SessionState(
        cwd=str(tmp_path), model="anthropic/claude-sonnet-4.6", reasoning_effort="max"
    )
    prompt_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "session/prompt",
        "params": {
            "sessionId": "acp-session",
            "prompt": [{"type": "text", "text": "fix $(not-shell)"}],
        },
    }

    server.handle(prompt_request)
    prompt_request["id"] = 2
    server.handle(prompt_request)

    responses = [message for message in messages if "result" in message]
    assert responses[-1]["result"] == {
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 20,
            "outputTokens": 12,
            "cachedReadTokens": 6,
            "cachedWriteTokens": 4,
            "thoughtTokens": 0,
            "totalTokens": 42,
        },
    }
    from benchflow.acp.types import PromptResult

    assert PromptResult.model_validate(responses[-1]["result"]).stop_reason == (
        "end_turn"
    )
    updates = [
        message["params"]["update"]
        for message in messages
        if message.get("method") == "session/update"
    ]
    assert any(update["sessionUpdate"] == "tool_call" for update in updates)
    assert any(
        update["sessionUpdate"] == "tool_call_update"
        and update["status"] == "completed"
        for update in updates
    )
    assert any(
        update["sessionUpdate"] == "agent_message_chunk"
        and update["content"]["text"] == "done"
        for update in updates
    )
    thought_text = "".join(
        update["content"]["text"]
        for update in updates
        if update["sessionUpdate"] == "agent_thought_chunk"
    )
    assert "provider bootstrap warning" in thought_text
    assert '"kind": "result"' in thought_text

    invocations = [json.loads(line) for line in argv_log.read_text().splitlines()]
    assert invocations[0]["prompt"] == "fix $(not-shell)"
    assert "--session" not in invocations[0]["args"]
    assert invocations[1]["args"][invocations[1]["args"].index("--session") + 1] == (
        "native-ori"
    )


def test_ori_registry_contract_is_pinned_acp_and_manifest_expressible() -> None:
    """Guards review feedback on PR #1067: Ori uses the canonical ACP contract."""
    cfg = AGENTS["ori"]

    assert cfg.protocol == "acp"
    assert cfg.session_factory == ""
    assert cfg.launch_cmd == "/opt/benchflow/bin/ori-acp-shim"
    assert cfg.default_model == "openrouter/openrouter/auto"
    assert cfg.api_protocol == "openai-completions"
    assert cfg.acp_effort_config_id == "reasoning_effort"
    assert cfg.env_mapping == {
        "BENCHFLOW_PROVIDER_BASE_URL": "ORI_OPENROUTER_BASE_URL",
        "BENCHFLOW_PROVIDER_API_KEY": "OPENROUTER_API_KEY",
        "BENCHFLOW_PROVIDER_MODEL": "ORI_MODEL",
    }
    assert "cli-0.12.0-68f9a36" in cfg.install_cmd
    assert "2dffa9f311f8b65f" in cfg.install_cmd
    assert "d3ee260046c313a7" in cfg.install_cmd
    assert "ori-releases/releases/download" in cfg.install_cmd
    assert "ori-acp-shim" in cfg.install_cmd
    assert "ori_events.py" in cfg.install_cmd
    assert "ori_jsonl.py" in cfg.install_cmd
    assert "install.sh" not in cfg.install_cmd
    assert cfg.subscription_auth is not None
    assert cfg.subscription_auth.native_policy is not None
    assert cfg.subscription_auth.detect_files == [
        "~/.ori/credentials.json",
        "~/.openrouter/credentials.json",
    ]


def test_ori_native_login_policy_only_applies_to_openrouter_models() -> None:
    """Guards PR #1067: typed subscription policy cannot bypass another provider."""
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


def test_ori_login_detects_openrouter_fallback_file(
    monkeypatch, tmp_path: Path
) -> None:
    """Guards PR #1067: either official Ori credential location is detected."""
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
    """Guards PR #1067: OpenRouter provider routing reaches Ori's native env."""
    from benchflow.agents.env import resolve_agent_env

    resolved = resolve_agent_env(
        "ori",
        "openrouter/anthropic/claude-sonnet-4.6",
        {"OPENROUTER_API_KEY": "test-openrouter-key"},
    )

    assert resolved["ORI_OPENROUTER_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert resolved["OPENROUTER_API_KEY"] == "test-openrouter-key"
    assert resolved["ORI_MODEL"] == "anthropic/claude-sonnet-4.6"
