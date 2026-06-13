from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from benchflow.agents.registry import AGENTS
from benchflow.evaluation import effective_model
from benchflow.providers.litellm_runtime import needs_litellm_runtime


def test_stagehand_agent_registry_and_routing() -> None:
    cfg = AGENTS["stagehand-agent"]

    assert cfg.name == "stagehand-agent"
    assert cfg.protocol == "acp"
    assert cfg.requires_env == ["GEMINI_API_KEY"]
    assert cfg.default_model == "google/gemini-2.5-flash"
    assert "@browserbasehq/stagehand@3.5.0" in cfg.install_cmd
    assert "@ai-sdk/google@2.0.74" in cfg.install_cmd
    assert "playwright@1.55.1" in cfg.install_cmd
    assert "stagehand-agent-acp-shim" in cfg.install_cmd
    assert "/opt/benchflow/benchflow/environment/browser_runtime.py" in cfg.install_cmd
    assert cfg.launch_cmd.startswith("PYTHONPATH=/opt/benchflow")
    assert effective_model("stagehand-agent", None) == "google/gemini-2.5-flash"
    assert not needs_litellm_runtime("stagehand-agent", "google/gemini-2.5-flash")


def test_stagehand_agent_acp_shim_runs_node_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "app"
    fixture_dir = app_dir / "browser_fixture"
    artifact_dir = tmp_path / "artifacts"
    fixture_dir.mkdir(parents=True)
    fixture_dir.joinpath("index.html").write_text(
        "<main>browser-use-smoke: ready</main>\n"
    )
    fake_node = _write_fake_node(tmp_path)

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
            "params": {"sessionId": "s", "modelId": "gemini-2.5-flash"},
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
                        "text": (
                            "Final answer must be exactly: browser-use-smoke: ready"
                        ),
                    }
                ],
            },
        },
    ]
    stdin = "".join(json.dumps(request) + "\n" for request in requests)
    env = {
        **os.environ,
        "BENCHFLOW_STAGEHAND_ARTIFACT_DIR": str(artifact_dir),
        "GEMINI_API_KEY": "fake-gemini-key",
        "STAGEHAND_AGENT_NODE": str(fake_node),
        "STAGEHAND_AGENT_NODE_PATH": str(tmp_path / "fake_node_modules"),
    }
    shim = (
        Path(__file__).parents[1]
        / "src"
        / "benchflow"
        / "agents"
        / "stagehand_agent_acp_shim.py"
    )
    result = subprocess.run(
        [sys.executable, str(shim)],
        input=stdin,
        text=True,
        capture_output=True,
        env=env,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    responses = {message["id"]: message for message in messages if "id" in message}
    updates = [
        message for message in messages if message.get("method") == "session/update"
    ]
    update_types = [update["params"]["update"]["sessionUpdate"] for update in updates]

    assert responses[1]["result"]["agentInfo"]["name"] == "stagehand-agent"
    assert responses[2]["result"]["sessionId"].startswith("stagehand-agent-")
    assert responses[3]["result"] == {}
    assert responses[4]["result"]["stopReason"] == "end_turn"
    assert update_types == ["tool_call", "tool_call_update", "agent_message_chunk"]
    assert app_dir.joinpath("final_result.txt").read_text() == (
        "browser-use-smoke: ready\n"
    )

    artifact = json.loads(
        artifact_dir.joinpath("browser-use-smoke-trace.json").read_text()
    )
    assert artifact["framework"] == "benchflow-stagehand-agent"
    assert artifact["final_result"] == "browser-use-smoke: ready"
    assert artifact["stagehand_model"] == "google/gemini-2.5-flash"
    assert artifact["stagehand_success"] is True
    assert artifact["stagehand_completed"] is True
    assert artifact["stagehand_usage"]["input_tokens"] == 10
    assert artifact["screenshots_b64"] == ["fake-stagehand-screenshot-b64"]
    assert [step["type"] for step in artifact["steps"]] == ["ariaTree", "done"]
    assert artifact["environment"]["readiness"]["status"] == "ready"
    assert len(artifact["environment"]["readiness"]["content_sha256"]) == 64

    fake_state = json.loads((tmp_path / "fake_state.json").read_text())
    assert fake_state["model"] == "google/gemini-2.5-flash"
    assert fake_state["expected"] == "browser-use-smoke: ready"
    assert "http://127.0.0.1:" in fake_state["url"]
    assert "http://127.0.0.1:" in fake_state["instruction"]
    assert "file://" not in fake_state["instruction"]


def test_stagehand_agent_acp_shim_accepts_freeform_upstream_prompt(
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "app"
    artifact_dir = tmp_path / "artifacts"
    app_dir.mkdir()
    fake_node = _write_fake_node(tmp_path)

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
            "params": {"sessionId": "s", "modelId": "gemini-2.5-flash"},
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
                        "text": "Browse example.com and summarize the page.",
                    }
                ],
            },
        },
    ]
    env = {
        **os.environ,
        "BENCHFLOW_STAGEHAND_ARTIFACT_DIR": str(artifact_dir),
        "FAKE_STAGEHAND_MESSAGE": "Example Domain summary",
        "GEMINI_API_KEY": "fake-gemini-key",
        "STAGEHAND_AGENT_NODE": str(fake_node),
        "STAGEHAND_AGENT_NODE_PATH": str(tmp_path / "fake_node_modules"),
    }
    shim = (
        Path(__file__).parents[1]
        / "src"
        / "benchflow"
        / "agents"
        / "stagehand_agent_acp_shim.py"
    )

    result = subprocess.run(
        [sys.executable, str(shim)],
        input="".join(json.dumps(request) + "\n" for request in requests),
        text=True,
        capture_output=True,
        env=env,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    artifact = json.loads(
        artifact_dir.joinpath("browser-use-smoke-trace.json").read_text()
    )
    assert artifact["final_result"] == "Example Domain summary"
    assert artifact["environment"]["readiness"]["status"] == "not-applicable"
    assert app_dir.joinpath("final_result.txt").read_text() == (
        "Example Domain summary\n"
    )
    fake_state = json.loads((tmp_path / "fake_state.json").read_text())
    assert fake_state["expected"] is None
    assert fake_state["url"] is None
    assert fake_state["instruction"] == "Browse example.com and summarize the page."


def _write_fake_node(tmp_path: Path) -> Path:
    fake_node = tmp_path / "fake-node"
    fake_node.write_text(
        f"""#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

payload = json.loads(sys.stdin.read())
message = os.environ.get("FAKE_STAGEHAND_MESSAGE", "browser-use-smoke: ready")
Path({str(tmp_path / "fake_state.json")!r}).write_text(json.dumps(payload))
sys.stdout.write(json.dumps({{
    "model": payload["model"],
    "extracted": {{"extraction": message}},
    "result": {{
        "success": True,
        "completed": True,
        "message": message,
        "actions": [
            {{"type": "ariaTree", "taskCompleted": False}},
            {{"type": "done", "taskCompleted": True}},
        ],
        "usage": {{"input_tokens": 10, "output_tokens": 2}},
        "messages_count": 6,
    }},
    "screenshots_b64": ["fake-stagehand-screenshot-b64"],
}}))
""",
    )
    fake_node.chmod(0o755)
    return fake_node
