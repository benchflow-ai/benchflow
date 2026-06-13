from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from benchflow.agents.registry import AGENTS
from benchflow.evaluation import effective_model
from benchflow.providers.litellm_runtime import needs_litellm_runtime


def test_browser_use_cli_agent_registry_and_routing() -> None:
    cfg = AGENTS["browser-use-cli"]

    assert cfg.name == "browser-use-cli"
    assert cfg.protocol == "acp"
    assert cfg.requires_env == []
    assert cfg.default_model == "browser-use-cli"
    assert "browser-use==0.13.1" in cfg.install_cmd
    assert "PLAYWRIGHT_BROWSERS_PATH=/opt/benchflow/ms-playwright" in cfg.install_cmd
    assert "/opt/benchflow/benchflow/environment/browser_runtime.py" in cfg.install_cmd
    assert cfg.launch_cmd.startswith("PYTHONPATH=/opt/benchflow")
    assert effective_model("browser-use-cli", None) == "browser-use-cli"
    assert not needs_litellm_runtime("browser-use-cli", "browser-use-cli")


def test_browser_use_cli_acp_shim_uses_cli_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "app"
    fixture_dir = app_dir / "browser_fixture"
    artifact_dir = tmp_path / "artifacts"
    browser_home_root = tmp_path / "browser-use-home"
    fixture_dir.mkdir(parents=True)
    fixture_dir.joinpath("index.html").write_text(
        "<main>browser-use-smoke: ready</main>\n"
    )
    fake_cli = _write_fake_browser_use_cli(tmp_path)

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
            "params": {"sessionId": "s", "modelId": "browser-use-cli"},
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
        "BROWSER_USE_BIN": str(fake_cli),
        "BENCHFLOW_BROWSER_USE_ARTIFACT_DIR": str(artifact_dir),
        "BENCHFLOW_BROWSER_USE_HOME_ROOT": str(browser_home_root),
    }
    shim = (
        Path(__file__).parents[1]
        / "src"
        / "benchflow"
        / "agents"
        / "browser_use_cli_acp_shim.py"
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

    assert responses[1]["result"]["agentInfo"]["name"] == "browser-use-cli"
    assert responses[2]["result"]["sessionId"].startswith("browser-use-cli-")
    assert responses[3]["result"] == {}
    assert responses[4]["result"]["stopReason"] == "end_turn"
    assert update_types.count("tool_call") == 4
    assert update_types.count("tool_call_update") == 4
    assert update_types[-1] == "agent_message_chunk"
    assert app_dir.joinpath("final_result.txt").read_text() == (
        "browser-use-smoke: ready\n"
    )

    artifact = json.loads(
        artifact_dir.joinpath("browser-use-smoke-trace.json").read_text()
    )
    assert artifact["framework"] == "benchflow-browser-use-cli-agent"
    assert artifact["final_result"] == "browser-use-smoke: ready"
    assert len(artifact["screenshots_b64"]) == 1
    assert [step["action"] for step in artifact["steps"]] == [
        "open",
        "get_html",
        "screenshot",
        "close",
    ]
    assert artifact["steps"][0]["environment"]["adapter"] == "browser"
    assert "http://127.0.0.1:" in artifact["steps"][0]["environment"]["served_url"]
    assert artifact["environment"]["readiness"]["status"] == "ready"
    assert len(artifact["environment"]["readiness"]["content_sha256"]) == 64


def _write_fake_browser_use_cli(tmp_path: Path) -> Path:
    script = tmp_path / "browser-use"
    script.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

args = sys.argv[1:]
if len(args) >= 2 and args[0] == "--session":
    args = args[2:]

cmd = args[0] if args else ""
if cmd == "open":
    print(f"url: {args[1]}")
elif cmd == "get" and args[1:] == ["html"]:
    print("html: <main>browser-use-smoke: ready</main>")
elif cmd == "screenshot":
    path = Path(args[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake png bytes")
    print(f"saved: {path}")
    print("size: 14")
elif cmd == "close":
    print("Browser closed")
else:
    print(f"unexpected command: {sys.argv}", file=sys.stderr)
    raise SystemExit(2)
""",
    )
    script.chmod(0o755)
    return script
