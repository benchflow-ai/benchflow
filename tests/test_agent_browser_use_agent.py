from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from benchflow.agents.registry import AGENTS
from benchflow.evaluation import effective_model
from benchflow.providers.litellm_runtime import needs_litellm_runtime


def test_browser_use_agent_registry_and_routing() -> None:
    cfg = AGENTS["browser-use-agent"]

    assert cfg.name == "browser-use-agent"
    assert cfg.protocol == "acp"
    assert cfg.requires_env == ["GEMINI_API_KEY"]
    assert cfg.default_model == "gemini-2.5-flash"
    assert "browser-use==0.13.1" in cfg.install_cmd
    assert "browser-use-agent-acp-shim" in cfg.install_cmd
    assert "/opt/benchflow/benchflow/environment/browser_runtime.py" in cfg.install_cmd
    assert cfg.launch_cmd.startswith("PYTHONPATH=/opt/benchflow")
    assert effective_model("browser-use-agent", None) == "gemini-2.5-flash"
    assert not needs_litellm_runtime("browser-use-agent", "gemini-2.5-flash")


def test_browser_use_agent_acp_shim_runs_agent_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "app"
    fixture_dir = app_dir / "browser_fixture"
    artifact_dir = tmp_path / "artifacts"
    fixture_dir.mkdir(parents=True)
    fixture_dir.joinpath("index.html").write_text(
        "<main>browser-use-smoke: ready</main>\n"
    )
    fake_pkg = _write_fake_browser_use_package(tmp_path)

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
            "params": {"sessionId": "s", "modelId": "google/gemini-2.5-flash"},
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
        "GEMINI_API_KEY": "fake-gemini-key",
        "PYTHONPATH": str(fake_pkg),
    }
    shim = (
        Path(__file__).parents[1]
        / "src"
        / "benchflow"
        / "agents"
        / "browser_use_agent_acp_shim.py"
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

    assert responses[1]["result"]["agentInfo"]["name"] == "browser-use-agent"
    assert responses[2]["result"]["sessionId"].startswith("browser-use-agent-")
    assert responses[3]["result"] == {}
    assert responses[4]["result"]["stopReason"] == "end_turn"
    assert update_types == ["tool_call", "tool_call_update", "agent_message_chunk"]
    assert app_dir.joinpath("final_result.txt").read_text() == (
        "browser-use-smoke: ready\n"
    )

    artifact = json.loads(
        artifact_dir.joinpath("browser-use-smoke-trace.json").read_text()
    )
    assert artifact["framework"] == "benchflow-browser-use-agent"
    assert artifact["final_result"] == "browser-use-smoke: ready"
    assert artifact["history_final_result"] == "browser-use-smoke: ready"
    assert artifact["history_steps"] == 2
    assert artifact["screenshots_b64"] == ["fake-screenshot-b64"]
    assert [step["action"] for step in artifact["steps"]] == ["navigate", "done"]
    assert artifact["environment"]["readiness"]["status"] == "ready"
    assert len(artifact["environment"]["readiness"]["content_sha256"]) == 64

    fake_state = json.loads((tmp_path / "fake_state.json").read_text())
    assert fake_state["model"] == "gemini-2.5-flash"
    assert fake_state["api_key"] == "fake-gemini-key"
    assert "http://127.0.0.1:" in fake_state["task"]
    assert "file://" not in fake_state["task"]


def test_browser_use_agent_acp_shim_accepts_freeform_upstream_prompt(
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "app"
    artifact_dir = tmp_path / "artifacts"
    app_dir.mkdir()
    fake_pkg = _write_fake_browser_use_package(tmp_path)

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
        "BENCHFLOW_BROWSER_USE_ARTIFACT_DIR": str(artifact_dir),
        "FAKE_BROWSER_USE_FINAL_RESULT": "Example Domain summary",
        "GEMINI_API_KEY": "fake-gemini-key",
        "PYTHONPATH": str(fake_pkg),
    }
    shim = (
        Path(__file__).parents[1]
        / "src"
        / "benchflow"
        / "agents"
        / "browser_use_agent_acp_shim.py"
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
    assert fake_state["task"] == "Browse example.com and summarize the page."
    assert fake_state["allowed_domains"] is None


def _write_fake_browser_use_package(tmp_path: Path) -> Path:
    root = tmp_path / "fake_pkg"
    package = root / "browser_use"
    llm = package / "llm"
    llm.mkdir(parents=True)
    (package / "__init__.py").write_text(
        f"""from __future__ import annotations

import json
import os
from pathlib import Path

STATE = Path({str(tmp_path / "fake_state.json")!r})


class BrowserProfile:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class Agent:
    def __init__(self, task, llm, browser_profile, **kwargs):
        self.task = task
        self.llm = llm
        self.browser_profile = browser_profile
        self.kwargs = kwargs

    async def run(self, max_steps=6):
        STATE.write_text(json.dumps({{
            "task": self.task,
            "model": self.llm.model,
            "api_key": self.llm.api_key,
            "allowed_domains": self.browser_profile.kwargs.get("allowed_domains"),
            "max_steps": max_steps,
        }}))
        return History()


class History:
    def final_result(self):
        return os.environ.get("FAKE_BROWSER_USE_FINAL_RESULT", "browser-use-smoke: ready")

    def action_names(self):
        return ["navigate", "done"]

    def screenshots(self):
        return ["fake-screenshot-b64"]

    def errors(self):
        return []

    def number_of_steps(self):
        return 2

    def is_successful(self):
        return True
""",
    )
    (llm / "__init__.py").write_text("")
    (llm / "google.py").write_text(
        """from __future__ import annotations


class ChatGoogle:
    def __init__(self, model, api_key, temperature=0):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
""",
    )
    return root
