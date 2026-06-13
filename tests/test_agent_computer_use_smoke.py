from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from benchflow.agents.registry import AGENTS
from benchflow.evaluation import effective_model
from benchflow.providers.litellm_runtime import needs_litellm_runtime


def test_computer_use_smoke_agent_registry_and_routing() -> None:
    cfg = AGENTS["computer-use-smoke"]

    assert cfg.name == "computer-use-smoke"
    assert cfg.protocol == "acp"
    assert cfg.requires_env == []
    assert cfg.default_model == "computer-use-smoke"
    assert effective_model("computer-use-smoke", None) == "computer-use-smoke"
    assert not needs_litellm_runtime("computer-use-smoke", "computer-use-smoke")


def test_computer_use_smoke_acp_shim_writes_result_and_artifact(
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "app"
    artifact_dir = tmp_path / "artifacts"
    bin_dir = tmp_path / "bin"
    app_dir.mkdir()
    artifact_dir.mkdir()
    bin_dir.mkdir()
    fake_screenshot = bin_dir / "gnome-screenshot"
    fake_screenshot.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-f" ]; then shift; out="$1"; fi\n'
        "  shift || true\n"
        "done\n"
        "printf 'fake-png' > \"$out\"\n"
    )
    fake_screenshot.chmod(0o755)

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
            "params": {"sessionId": "s", "modelId": "computer-use-smoke"},
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
                            "Final answer must be exactly: computer-use-smoke: ready"
                        ),
                    }
                ],
            },
        },
    ]
    stdin = "".join(json.dumps(request) + "\n" for request in requests)
    env = {
        **os.environ,
        "BENCHFLOW_COMPUTER_USE_ARTIFACT_DIR": str(artifact_dir),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    shim = (
        Path(__file__).parents[1]
        / "src"
        / "benchflow"
        / "agents"
        / "computer_use_smoke_acp_shim.py"
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

    assert responses[1]["result"]["agentInfo"]["name"] == "computer-use-smoke"
    assert responses[2]["result"]["sessionId"].startswith("computer-use-smoke-")
    assert responses[3]["result"] == {}
    assert responses[4]["result"]["stopReason"] == "end_turn"
    assert [u["params"]["update"]["sessionUpdate"] for u in updates] == [
        "tool_call",
        "tool_call",
        "tool_call",
        "tool_call_update",
        "tool_call_update",
        "tool_call_update",
        "agent_message_chunk",
    ]
    assert app_dir.joinpath("computer_use_result.txt").read_text() == (
        "computer-use-smoke: ready\n"
    )
    assert app_dir.joinpath("computer_use_roundtrip.txt").read_text() == (
        "computer-use-smoke: ready\n"
    )

    artifact = json.loads(
        artifact_dir.joinpath("computer-use-smoke-trace.json").read_text()
    )
    assert artifact["schema"] == "benchflow.desktop-runtime-trace.v1"
    assert artifact["framework"] == "benchflow-computer-use-smoke-agent"
    assert artifact["final_result"] == "computer-use-smoke: ready"
    assert artifact["environment"]["adapter"] == "desktop"
    assert artifact["environment"]["sandbox_provider"] == "cua"
    assert artifact["screenshot_method"] == "gnome-screenshot"
    assert len(artifact["screenshots_b64"]) == 1
    assert len(artifact["steps"]) == 3
