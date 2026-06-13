from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from benchflow.agents.registry import AGENTS
from benchflow.evaluation import effective_model
from benchflow.providers.litellm_runtime import needs_litellm_runtime


def test_browser_use_smoke_agent_registry_and_routing() -> None:
    cfg = AGENTS["browser-use-smoke"]

    assert cfg.name == "browser-use-smoke"
    assert cfg.protocol == "acp"
    assert cfg.requires_env == []
    assert cfg.default_model == "browser-use-smoke"
    assert "/opt/benchflow/benchflow/environment/browser_runtime.py" in cfg.install_cmd
    assert "PYTHONPATH=/opt/benchflow" in cfg.launch_cmd
    assert effective_model("browser-use-smoke", None) == "browser-use-smoke"
    assert not needs_litellm_runtime("browser-use-smoke", "browser-use-smoke")


def test_browser_use_smoke_acp_shim_writes_result_and_artifact(
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "app"
    fixture_dir = app_dir / "browser_fixture"
    artifact_dir = tmp_path / "artifacts"
    fixture_dir.mkdir(parents=True)
    fixture_dir.joinpath("index.html").write_text(
        "<main>browser-use-smoke: ready</main>\n"
    )

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
            "params": {"sessionId": "s", "modelId": "browser-use-smoke"},
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
        "BENCHFLOW_BROWSER_USE_ARTIFACT_DIR": str(artifact_dir),
    }
    shim = (
        Path(__file__).parents[1]
        / "src"
        / "benchflow"
        / "agents"
        / "browser_use_smoke_acp_shim.py"
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

    assert responses[1]["result"]["agentInfo"]["name"] == "browser-use-smoke"
    assert responses[2]["result"]["sessionId"].startswith("browser-use-smoke-")
    assert responses[3]["result"] == {}
    assert responses[4]["result"]["stopReason"] == "end_turn"
    assert [u["params"]["update"]["sessionUpdate"] for u in updates] == [
        "tool_call",
        "tool_call_update",
        "agent_message_chunk",
    ]
    assert app_dir.joinpath("final_result.txt").read_text() == (
        "browser-use-smoke: ready\n"
    )

    artifact = json.loads(
        artifact_dir.joinpath("browser-use-smoke-trace.json").read_text()
    )
    assert artifact["framework"] == "benchflow-browser-use-smoke-agent"
    assert artifact["final_result"] == "browser-use-smoke: ready"
    assert artifact["screenshots_b64"] == []
    assert len(artifact["steps"]) == 4
    assert artifact["environment"]["adapter"] == "browser"
    assert artifact["environment"]["readiness"]["status"] == "ready"
    assert len(artifact["environment"]["readiness"]["content_sha256"]) == 64
