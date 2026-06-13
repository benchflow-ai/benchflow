from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from benchflow.agents.registry import AGENTS
from benchflow.evaluation import effective_model
from benchflow.providers.litellm_runtime import needs_litellm_runtime

_SHIM = (
    Path(__file__).parents[1]
    / "src"
    / "benchflow"
    / "agents"
    / "computer_use_agent_acp_shim.py"
)


def test_computer_use_agent_registry_and_routing() -> None:
    cfg = AGENTS["computer-use-agent"]

    assert cfg.name == "computer-use-agent"
    assert cfg.protocol == "acp"
    assert cfg.requires_env == ["GEMINI_API_KEY"]
    assert cfg.default_model == "gemini-3.5-flash"
    assert effective_model("computer-use-agent", None) == "gemini-3.5-flash"
    # The shim speaks Gemini natively; it does not need the LiteLLM runtime.
    assert not needs_litellm_runtime("computer-use-agent", "gemini-3.5-flash")


def _fake_bins(bin_dir: Path, xdotool_log: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    screenshot = bin_dir / "gnome-screenshot"
    screenshot.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-f" ]; then shift; out="$1"; fi\n'
        "  shift || true\n"
        "done\n"
        "printf 'fake-png' > \"$out\"\n"
    )
    screenshot.chmod(0o755)
    xdotool = bin_dir / "xdotool"
    xdotool.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{xdotool_log}"\nexit 0\n')
    xdotool.chmod(0o755)


def test_computer_use_agent_acp_loop_executes_actions_and_writes_artifact(
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "app"
    artifact_dir = tmp_path / "artifacts"
    bin_dir = tmp_path / "bin"
    app_dir.mkdir()
    artifact_dir.mkdir()
    xdotool_log = tmp_path / "xdotool.log"
    _fake_bins(bin_dir, xdotool_log)

    # Scripted "model": click, type, then finish with the required answer.
    expected = "computer-use-agent: ok"
    actions = [
        {"action": "click", "x": 10, "y": 20},
        {"action": "type", "text": "hello"},
        {"action": "done", "result": expected},
    ]
    fake_actions = tmp_path / "actions.json"
    fake_actions.write_text(json.dumps(actions))

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": {"cwd": str(app_dir), "mcpServers": []},
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/set_model",
            "params": {"sessionId": "s", "modelId": "gemini-3.5-flash"},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "session/prompt",
            "params": {
                "sessionId": "s",
                "prompt": [
                    {
                        "type": "text",
                        "text": f"Final answer must be exactly: {expected}",
                    }
                ],
            },
        },
    ]
    stdin = "".join(json.dumps(request) + "\n" for request in requests)
    env = {
        **os.environ,
        "BENCHFLOW_COMPUTER_USE_ARTIFACT_DIR": str(artifact_dir),
        "BENCHFLOW_CUA_AGENT_FAKE_ACTIONS": str(fake_actions),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    result = subprocess.run(
        [sys.executable, str(_SHIM)],
        input=stdin,
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    responses = {message["id"]: message for message in messages if "id" in message}
    updates = [m for m in messages if m.get("method") == "session/update"]

    assert responses[1]["result"]["agentInfo"]["name"] == "computer-use-agent"
    assert responses[2]["result"]["sessionId"].startswith("computer-use-agent-")
    assert responses[3]["result"] == {}
    assert responses[4]["result"]["stopReason"] == "end_turn"

    # One tool_call + tool_call_update per action (3), then a final message chunk.
    assert [u["params"]["update"]["sessionUpdate"] for u in updates] == [
        "tool_call",
        "tool_call_update",
        "tool_call",
        "tool_call_update",
        "tool_call",
        "tool_call_update",
        "agent_message_chunk",
    ]

    # The two non-terminal actions actually drove xdotool inside the sandbox.
    xdotool_calls = xdotool_log.read_text()
    assert "mousemove --sync 10 20 click 1" in xdotool_calls
    assert "type --clearmodifiers -- hello" in xdotool_calls

    assert app_dir.joinpath("computer_use_result.txt").read_text() == expected + "\n"

    artifact = json.loads(
        artifact_dir.joinpath("computer-use-agent-trace.json").read_text()
    )
    assert artifact["schema"] == "benchflow.desktop-runtime-trace.v1"
    assert artifact["framework"] == "benchflow-computer-use-agent"
    assert artifact["final_result"] == expected
    assert artifact["environment"]["adapter"] == "desktop"
    assert artifact["environment"]["sandbox_provider"] == "cua"
    assert artifact["screenshot_method"] == "gnome-screenshot"
    assert artifact["model"] == "gemini-3.5-flash"
    assert artifact["history_steps"] == 3
    assert len(artifact["steps"]) == 3
    assert len(artifact["screenshots_b64"]) == 3
    assert [step["action"] for step in artifact["steps"]] == ["click", "type", "done"]


def test_computer_use_agent_errors_when_step_budget_exhausted(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    artifact_dir = tmp_path / "artifacts"
    bin_dir = tmp_path / "bin"
    app_dir.mkdir()
    artifact_dir.mkdir()
    _fake_bins(bin_dir, tmp_path / "xdotool.log")

    # Never emits "done" within the budget — the loop must fail closed, not
    # silently pass. Supply max_steps worth of non-terminal actions so the
    # scripted policy never reaches its exhaustion fallback.
    fake_actions = tmp_path / "actions.json"
    fake_actions.write_text(json.dumps([{"action": "wait", "ms": 0}] * 3))

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": {"cwd": str(app_dir)},
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": "s",
                "prompt": [{"type": "text", "text": "do something"}],
            },
        },
    ]
    stdin = "".join(json.dumps(request) + "\n" for request in requests)
    env = {
        **os.environ,
        "BENCHFLOW_COMPUTER_USE_ARTIFACT_DIR": str(artifact_dir),
        "BENCHFLOW_CUA_AGENT_FAKE_ACTIONS": str(fake_actions),
        "BENCHFLOW_CUA_AGENT_MAX_STEPS": "3",
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    result = subprocess.run(
        [sys.executable, str(_SHIM)],
        input=stdin,
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    chunks = [
        m["params"]["update"]["content"]["text"]
        for m in messages
        if m.get("method") == "session/update"
        and m["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
    ]
    assert any("did not finish within 3 steps" in text for text in chunks)
    # No durable success artifact when the loop never finished.
    assert not artifact_dir.joinpath("computer-use-agent-trace.json").exists()
