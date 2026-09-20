"""Google Antigravity CLI (``antigravity`` agent) — registry shape, shim protocol,
LiteLLM wiring, and web-tool policy.

The Antigravity CLI (``agy``) replaced the hosted Gemini CLI in mid-2026 and
ships without an ACP mode, so BenchFlow drives it through
``benchflow/agents/antigravity_acp_shim.py``. These tests pin the contract
introduced by the antigravity-native PR: the registry entry mirrors the
``gemini`` agent (Gemini API-key auth, GenerateContent pass-through through the
LiteLLM gateway, skills discovery), and the shim maps agy's ``stream-json``
events onto ACP updates with raw tool I/O and turn usage.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from benchflow.agents import antigravity_config as cfgmod
from benchflow.agents.registry import AGENT_ALIASES, AGENTS, get_agent, resolve_agent
from benchflow.evaluation import effective_model

_SHIM = (
    Path(__file__).resolve().parents[1] / "src/benchflow/agents/antigravity_acp_shim.py"
)

# ── registry shape ───────────────────────────────────────────────────────────


class TestRegistryEntry:
    def test_aliases_resolve(self):
        assert AGENT_ALIASES["agy"] == "antigravity"
        assert resolve_agent("agy").name == "antigravity"
        assert resolve_agent("acp:antigravity").name == "antigravity"

    def test_default_model_is_gemini_family_with_gemini_key(self):
        """Like gemini (#343): ``--agent antigravity`` must never fall back to a
        Claude default or demand ANTHROPIC_API_KEY."""
        from benchflow.agents.registry import infer_env_key_for_model

        cfg, default_model = get_agent("antigravity")
        assert "gemini" in default_model
        assert infer_env_key_for_model(default_model) == "GEMINI_API_KEY"
        assert cfg.requires_env == ["GEMINI_API_KEY"]
        assert effective_model("antigravity", None) == default_model

    def test_env_mapping_mirrors_gemini_native_routing(self):
        cfg = AGENTS["antigravity"]
        assert (
            cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "GOOGLE_GEMINI_BASE_URL"
        )
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "GEMINI_API_KEY"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_MODEL"] == "ANTIGRAVITY_MODEL"
        assert cfg.api_protocol == ""
        assert cfgmod.routes_gemini_natively(cfg)
        assert cfgmod.routes_gemini_natively(AGENTS["gemini"])
        assert not cfgmod.routes_gemini_natively(AGENTS["codex-acp"])

    def test_model_and_effort_go_through_acp(self):
        """The shim implements session/set_model and a ``thinking`` config
        option, so BenchFlow drives both over ACP (no env-owned model)."""
        cfg = AGENTS["antigravity"]
        assert cfg.supports_acp_set_model is True
        assert cfg.acp_model_format == "bare"
        assert cfg.acp_model_config_id == ""
        assert cfg.acp_effort_config_id == "thinking"

    def test_skills_and_home(self):
        cfg = AGENTS["antigravity"]
        assert cfg.skill_paths == [
            "$HOME/.gemini/config/skills",
            "$WORKSPACE/.agents/skills",
        ]
        assert cfg.home_dirs == [".gemini"]
        assert cfg.disallow_web_tools_owned_paths == ["$HOME/.gemini"]
        # Google sign-in lives in the OS keyring: no host login file to copy.
        assert cfg.subscription_auth is None
        assert cfg.credential_files == []

    def test_task_mcp_uses_agy_native_config(self):
        cfg = AGENTS["antigravity"]
        assert cfg.task_mcp_transport == "native-config"
        assert cfg.task_mcp_config_path == ".gemini/config/mcp_config.json"

    def test_install_pins_the_agy_release_and_verifies_it(self):
        cfg = AGENTS["antigravity"]
        for _arch, (url, sha) in cfgmod.AGY_RELEASES.items():
            assert url in cfg.install_cmd
            assert sha in cfg.install_cmd
            assert "1.2.7-6731160148115456" in url
        assert "sha512sum -c" in cfg.install_cmd
        # The download stage proves the binary runs on the task image before
        # the shim is deployed after it.
        assert "/opt/benchflow/antigravity/agy --version" in cfg.install_cmd
        assert cfg.install_cmd.index(
            "/opt/benchflow/antigravity/agy --version"
        ) < cfg.install_cmd.index("antigravity-acp-shim")
        # No Node.js: the CLI is a native binary and the shim is stdlib Python.
        assert "npm install" not in cfg.install_cmd
        assert "/opt/benchflow/bin/antigravity-acp-shim" in cfg.install_cmd
        assert cfg.launch_cmd == "/opt/benchflow/bin/antigravity-acp-shim"

    def test_install_cmd_is_idempotent_on_matching_version(self, tmp_path):
        """A binary already reporting AGY_VERSION short-circuits the download."""
        fake = tmp_path / "opt" / "benchflow" / "antigravity" / "agy"
        fake.parent.mkdir(parents=True)
        fake.write_text(f"#!/bin/sh\necho {cfgmod.AGY_VERSION}\n")
        fake.chmod(0o755)
        cmd = cfgmod.agy_install_cmd("exit 99").replace(
            "/opt/benchflow/antigravity", str(fake.parent)
        )
        result = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == cfgmod.AGY_VERSION


# ── model / effort helpers (host and shim copies must agree) ─────────────────


def _load_shim_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("antigravity_acp_shim", _SHIM)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gemini-3.8-flash", ("gemini-3.8-flash", None)),
        ("gemini-3.8-flash-high", ("gemini-3.8-flash", "high")),
        ("google/gemini-3.1-pro-low", ("gemini-3.1-pro", "low")),
        ("gemini/gemini-3.8-flash", ("gemini-3.8-flash", None)),
        ("gemini-3.5-flash-lite", ("gemini-3.5-flash-lite", None)),
    ],
)
def test_split_model_effort_matches_between_host_and_shim(model, expected):
    shim = _load_shim_module()
    assert cfgmod.split_model_effort(model) == expected
    assert shim.split_model_effort(model) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "high"),
        ("max", "high"),
        ("minimal", "low"),
        (None, None),
        ("", None),
    ],
)
def test_effort_normalization_matches_between_host_and_shim(value, expected):
    shim = _load_shim_module()
    assert cfgmod.normalize_antigravity_effort(value) == expected
    assert shim.normalize_effort(value) == expected


def test_effort_normalization_rejects_unknown_labels():
    with pytest.raises(ValueError):
        cfgmod.normalize_antigravity_effort("ultra")


# ── web-tool policy: PreToolUse deny hooks ───────────────────────────────────


def _run_policy_cmd(cmd: str, home: Path) -> dict:
    result = subprocess.run(
        ["bash", "-c", cmd],
        env={"BENCHFLOW_AGENT_HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return json.loads((home / ".gemini/antigravity-cli/hooks.json").read_text())


def _matchers(hook: dict) -> list[str]:
    return [group["matcher"] for group in hook["PreToolUse"]]


def _hook_verdict(hook: dict) -> dict:
    command = hook["PreToolUse"][0]["hooks"][0]["command"]
    out = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
    assert out.returncode == 0
    return json.loads(out.stdout)


def test_no_web_policy_denies_every_web_tool(tmp_path):
    hooks = _run_policy_cmd(
        AGENTS["antigravity"].disallow_web_tools_setup_cmd, tmp_path
    )
    hook = hooks["benchflow-no-web"]
    assert set(_matchers(hook)) == set(cfgmod.ANTIGRAVITY_WEB_TOOL_MATCHERS)
    assert {"search_web", "read_url_content", ".*browser.*"} <= set(_matchers(hook))
    assert _hook_verdict(hook)["decision"] == "deny"


def test_hosted_search_policy_denies_search_and_fetch_only(tmp_path):
    hooks = _run_policy_cmd(
        AGENTS["antigravity"].disallow_hosted_search_setup_cmd, tmp_path
    )
    hook = hooks["benchflow-no-hosted-search"]
    assert set(_matchers(hook)) == {"search_web", "read_url_content"}
    assert _hook_verdict(hook)["decision"] == "deny"


def test_policy_hooks_merge_with_existing_hooks(tmp_path):
    hooks_path = tmp_path / ".gemini/antigravity-cli/hooks.json"
    hooks_path.parent.mkdir(parents=True)
    hooks_path.write_text(json.dumps({"task-hook": {"Stop": [{"command": "true"}]}}))
    hooks = _run_policy_cmd(
        AGENTS["antigravity"].disallow_web_tools_setup_cmd, tmp_path
    )
    assert "task-hook" in hooks and "benchflow-no-web" in hooks


# ── shim protocol against a fake agy ─────────────────────────────────────────

_FAKE_AGY = textwrap.dedent(
    r"""
    #!/usr/bin/env python3
    import json, os, sys
    args = sys.argv[1:]
    log = open(os.environ["FAKE_AGY_LOG"], "a")
    log.write(json.dumps({"args": args, "cwd": os.getcwd()}) + "\n"); log.flush()
    if "--version" in args:
        print("9.9.9"); sys.exit(0)
    def emit(obj):
        sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()
    conv = "conv-1234-5678"
    emit({"event": "init", "conversation_id": conv, "init": {"model": "x", "cwd": os.getcwd(), "tools": ["run_command"], "permission_mode": "always-proceed"}})
    for line in sys.stdin:
        msg = json.loads(line)
        text = msg["message"]["content"]
        log.write(json.dumps({"prompt": text}) + "\n"); log.flush()
        if text == "explode":
            emit({"event": "result", "result": {"conversation_id": conv, "status": "ERROR", "error": "model exploded", "usage": {}}})
            continue
        if text == "die":
            sys.stderr.write('AGY_ERROR: {"short_error":"boom"}\n'); sys.stderr.flush()
            sys.exit(3)
        su = lambda s: emit({"event": "step_update", "step_update": {"conversation_id": conv, **s}})
        su({"step_index": 0, "state": "DONE", "step_type": "user_input"})
        su({"step_index": 1, "state": "ACTIVE", "step_type": "tool", "tool_name": "run_command", "tool_info": {"name": "run_command", "parameters": {"CommandLine": "ls -la", "Cwd": os.getcwd()}}})
        su({"step_index": 1, "state": "DONE", "step_type": "tool", "tool_name": "run_command", "tool_info": {"name": "run_command", "parameters": {"CommandLine": "ls -la"}, "output": "total 0\n"}})
        su({"step_index": 2, "state": "ACTIVE", "step_type": "tool", "tool_name": "view_file", "tool_info": {"name": "view_file", "parameters": {"AbsolutePath": "/app/missing.txt"}}})
        su({"step_index": 2, "state": "ERROR", "step_type": "tool", "tool_name": "view_file", "tool_info": {"name": "view_file", "parameters": {"AbsolutePath": "/app/missing.txt"}, "error": {"type": "TOOL_ERROR", "message": "no such file"}}})
        su({"step_index": 5, "state": "ACTIVE", "step_type": "tool", "tool_name": "view_file", "tool_info": {"name": "view_file", "parameters": {"AbsolutePath": "/app/.agents/skills/citation-management/SKILL.md"}}})
        su({"step_index": 5, "state": "DONE", "step_type": "tool", "tool_name": "view_file", "tool_info": {"name": "view_file", "parameters": {"AbsolutePath": "/app/.agents/skills/citation-management/SKILL.md"}, "output": "# Citation management\n"}})
        su({"step_index": 3, "state": "ACTIVE", "step_type": "agent_response", "text_delta": "hello "})
        su({"step_index": 3, "state": "DONE", "step_type": "agent_response", "text_delta": "world", "usage": {"input_tokens": 10, "output_tokens": 2, "thinking_tokens": 1, "cache_read_tokens": 0, "total_tokens": 12}})
        emit({"event": "result", "result": {"conversation_id": conv, "status": "SUCCESS", "response": "hello world", "duration_seconds": 0.1, "num_turns": 1, "usage": {"input_tokens": 100, "output_tokens": 20, "thinking_tokens": 5, "cache_read_tokens": 7, "total_tokens": 120}}})
    """
).lstrip()


class _ShimClient:
    """Minimal JSON-RPC driver for the shim subprocess."""

    def __init__(self, tmp_path: Path, env_extra: dict | None = None):
        self.fake = tmp_path / "fake-agy"
        self.fake.write_text(_FAKE_AGY)
        self.fake.chmod(self.fake.stat().st_mode | stat.S_IXUSR)
        self.log = tmp_path / "fake-agy.log"
        self.home = tmp_path / "home"
        self.home.mkdir()
        env = {
            **os.environ,
            "BENCHFLOW_AGY_BIN": str(self.fake),
            "FAKE_AGY_LOG": str(self.log),
            "HOME": str(self.home),
            "BENCHFLOW_AGENT_HOME": str(self.home),
            "GEMINI_API_KEY": "test-key",
        }
        env.pop("ANTIGRAVITY_MODEL", None)
        env.pop("ANTIGRAVITY_EFFORT", None)
        env.update(env_extra or {})
        self.proc = subprocess.Popen(
            [sys.executable, str(_SHIM)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        self._id = 0
        self.notifications: list[dict] = []

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": self._id,
                    "method": method,
                    "params": params or {},
                }
            )
            + "\n"
        )
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            assert line, (
                f"shim closed stdout; stderr={self.proc.stderr.read() if self.proc.stderr else ''}"
            )
            msg = json.loads(line)
            if msg.get("id") == self._id:
                return msg
            self.notifications.append(msg)

    def notify(self, method: str, params: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n"
        )
        self.proc.stdin.flush()

    def close(self) -> str:
        # communicate() closes stdin (EOF ends the shim's read loop) and
        # drains stdout/stderr until the shim exits.
        _out, err = self.proc.communicate(timeout=20)
        return err

    def agy_log(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [
            json.loads(line)
            for line in self.log.read_text().splitlines()
            if line.strip()
        ]

    def updates(self) -> list[dict]:
        return [
            n["params"]["update"]
            for n in self.notifications
            if n.get("method") == "session/update"
        ]


@pytest.fixture
def shim(tmp_path):
    client = _ShimClient(tmp_path)
    yield client
    if client.proc.poll() is None:
        client.proc.kill()


def _handshake(shim: _ShimClient, cwd: str) -> str:
    init = shim.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    assert init["result"]["agentInfo"]["name"] == "antigravity"
    assert init["result"]["agentInfo"]["version"] == "9.9.9"
    new = shim.request("session/new", {"cwd": cwd, "mcpServers": []})
    session_id = new["result"]["sessionId"]
    assert [o["id"] for o in new["result"]["configOptions"]] == ["thinking"]
    return session_id


def test_shim_streams_tool_calls_message_and_usage(shim, tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    session_id = _handshake(shim, str(workspace))
    assert (
        shim.request(
            "session/set_model",
            {"sessionId": session_id, "modelId": "gemini-3.8-flash"},
        )["result"]
        == {}
    )
    cfg = shim.request(
        "session/set_config_option",
        {"sessionId": session_id, "configId": "thinking", "value": "xhigh"},
    )
    assert cfg["result"]["configOptions"][0]["currentValue"] == "high"

    reply = shim.request(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "do the thing"}]},
    )
    assert reply["result"]["stopReason"] == "end_turn"
    assert reply["result"]["usage"] == {
        "inputTokens": 100,
        "outputTokens": 20,
        "totalTokens": 120,
        "cachedReadTokens": 7,
        "thoughtTokens": 5,
    }

    # agy was launched headless with the workspace, model and effort.
    launches = [e for e in shim.agy_log() if "args" in e]
    assert launches[-1]["cwd"] == str(workspace)
    args = launches[-1]["args"]
    assert (
        "--input-format=stream-json" in args and "--output-format=stream-json" in args
    )
    assert "--dangerously-skip-permissions" in args
    assert f"--add-dir={workspace}" in args
    assert "--model=gemini-3.8-flash" in args and "--effort=high" in args
    assert [e for e in shim.agy_log() if "prompt" in e][-1]["prompt"] == "do the thing"

    updates = shim.updates()
    tool_calls = [u for u in updates if u["sessionUpdate"] == "tool_call"]
    assert [tc["kind"] for tc in tool_calls] == ["execute", "read", "skill"]
    # Reading .../skills/<name>/SKILL.md is how agy loads a skill; it is
    # reported with the canonical ACP kind so BenchFlow counts the invocation.
    assert tool_calls[2]["title"] == "skill: citation-management"
    assert tool_calls[2]["rawInput"]["AbsolutePath"].endswith("/SKILL.md")
    assert tool_calls[0]["title"] == "run_command: ls -la"
    assert tool_calls[0]["rawInput"]["CommandLine"] == "ls -la"
    assert tool_calls[1]["locations"] == [{"path": "/app/missing.txt"}]
    tool_updates = [u for u in updates if u["sessionUpdate"] == "tool_call_update"]
    assert [u["status"] for u in tool_updates] == ["completed", "failed", "completed"]
    assert tool_updates[0]["status"] == "completed"
    assert tool_updates[0]["content"][0]["content"]["text"] == "total 0\n"
    assert tool_updates[0]["rawOutput"] == "total 0\n"
    assert tool_updates[1]["status"] == "failed"
    assert tool_updates[1]["rawOutput"] == {"error": "no such file"}
    text = "".join(
        u["content"]["text"]
        for u in updates
        if u["sessionUpdate"] == "agent_message_chunk"
    )
    assert text == "hello world"

    # settings.json was prepared for headless Gemini API-key mode.
    settings = json.loads(
        (shim.home / ".gemini/antigravity-cli/settings.json").read_text()
    )
    assert settings["modelProvider"] == "gemini"
    assert settings["enableTelemetry"] == "off"
    assert settings["enableTerminalSandbox"] == "off"
    shim.close()


def test_shim_takes_effort_from_catalog_style_model_ids(shim, tmp_path):
    session_id = _handshake(shim, str(tmp_path))
    shim.request(
        "session/set_model",
        {"sessionId": session_id, "modelId": "gemini-3.8-flash-low"},
    )
    shim.request(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]},
    )
    args = [e for e in shim.agy_log() if "args" in e][-1]["args"]
    assert "--model=gemini-3.8-flash" in args and "--effort=low" in args
    shim.close()


def test_shim_defaults_model_and_effort_from_env(tmp_path):
    client = _ShimClient(
        tmp_path,
        {"ANTIGRAVITY_MODEL": "google/gemini-3.1-pro", "ANTIGRAVITY_EFFORT": "medium"},
    )
    try:
        session_id = _handshake(client, str(tmp_path))
        client.request(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]},
        )
        args = [e for e in client.agy_log() if "args" in e][-1]["args"]
        assert "--model=gemini-3.1-pro" in args and "--effort=medium" in args
        client.close()
    finally:
        if client.proc.poll() is None:
            client.proc.kill()


def test_shim_rejects_prompt_without_a_model(shim, tmp_path):
    session_id = _handshake(shim, str(tmp_path))
    reply = shim.request(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]},
    )
    assert (
        "error" in reply
        and "No Antigravity model configured" in reply["error"]["message"]
    )
    shim.close()


def test_shim_surfaces_agy_turn_errors_and_process_death(shim, tmp_path):
    session_id = _handshake(shim, str(tmp_path))
    shim.request(
        "session/set_model", {"sessionId": session_id, "modelId": "gemini-3.8-flash"}
    )
    reply = shim.request(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "explode"}]},
    )
    assert "error" in reply and "model exploded" in reply["error"]["message"]
    reply = shim.request(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "die"}]},
    )
    assert "error" in reply
    assert "rc=3" in reply["error"]["message"] and "boom" in reply["error"]["message"]
    # The session recovers: a fresh agy is spawned for the next prompt.
    reply = shim.request(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "again"}]},
    )
    assert reply["result"]["stopReason"] == "end_turn"
    shim.close()


def test_shim_rejects_unknown_config_option(shim, tmp_path):
    session_id = _handshake(shim, str(tmp_path))
    reply = shim.request(
        "session/set_config_option",
        {"sessionId": session_id, "configId": "model", "value": "x"},
    )
    assert reply["error"]["code"] == -32602
    shim.close()


def test_shim_initialize_fails_closed_without_binary(tmp_path):
    client = _ShimClient(tmp_path, {"BENCHFLOW_AGY_BIN": str(tmp_path / "nope")})
    try:
        reply = client.request("initialize", {"protocolVersion": 1})
        assert "error" in reply and "not found" in reply["error"]["message"]
        client.close()
    finally:
        if client.proc.poll() is None:
            client.proc.kill()
