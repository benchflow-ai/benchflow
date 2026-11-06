"""Meta Muse Code (``muse-code`` agent) — registry shape, install pin, and shim protocol.

Muse Code ships without an ACP mode, so BenchFlow drives ``muse exec --json``
through ``benchflow/agents/muse_acp_shim.py``. These tests pin that contract:
the registry entry (pinned binary, subscription login file, web-policy launch
flags), the shim's credential handling (a login wins over META_API_KEY), the
mapping of Muse's session log onto ACP updates with turn usage, and the wait
on an exhausted subscription window.
"""

from __future__ import annotations

import datetime
import json
import os
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from benchflow.agents import muse_config as cfgmod
from benchflow.agents.credentials import CREDENTIAL_EVIDENCE_PATHS
from benchflow.agents.registry import AGENT_ALIASES, AGENTS, get_agent, resolve_agent
from benchflow.errors import MissingCredentialError

_SHIM = Path(__file__).resolve().parents[1] / "src/benchflow/agents/muse_acp_shim.py"

# ── registry shape ───────────────────────────────────────────────────────────


class TestRegistryEntry:
    def test_aliases_resolve(self):
        assert AGENT_ALIASES["muse"] == "muse-code"
        assert resolve_agent("muse").name == "muse-code"
        assert resolve_agent("acp:muse-code").name == "muse-code"

    def test_default_model_needs_only_the_login(self):
        """The credential is the subscription login, never inferred from the model."""
        from benchflow.agents.registry import infer_env_key_for_model

        cfg, default_model = get_agent("muse-code")
        assert default_model == "muse-spark-1.3"
        assert infer_env_key_for_model(default_model) is None
        assert cfg.requires_env == ["MUSE_AUTH_JSON"]
        assert cfg.api_protocol == ""
        assert cfg.env_mapping == {}

    def test_login_reaches_the_sandbox_as_a_private_file(self):
        cfg = AGENTS["muse-code"]
        assert [(c.path, c.env_source) for c in cfg.credential_files] == [
            ("{home}/.config/muse/auth.json", "MUSE_AUTH_JSON")
        ]
        assert cfg.subscription_auth is None
        # Evidence capture must never ship the login to a reviewer or a bucket.
        assert ".config/muse/auth.json" in CREDENTIAL_EVIDENCE_PATHS

    def test_model_effort_and_web_policy_go_through_the_shim(self):
        cfg = AGENTS["muse-code"]
        assert cfg.supports_acp_set_model is True
        assert cfg.acp_model_format == "bare"
        assert cfg.acp_effort_config_id == "effort"
        assert cfg.disallow_web_tools_launch_suffix == " --disable-web-tools"
        assert cfg.disallow_hosted_search_launch_suffix == " --disable-hosted-search"
        assert cfg.skill_paths == ["$HOME/.config/muse/skills"]

    def test_install_pins_the_muse_release_and_verifies_it(self):
        cfg = AGENTS["muse-code"]
        for url, sha, _size in cfgmod.MUSE_RELEASES.values():
            assert url in cfg.install_cmd
            assert sha in cfg.install_cmd
            assert f"version={cfgmod.MUSE_VERSION}" in url
        assert "sha256sum -c" in cfg.install_cmd
        assert "/opt/benchflow/muse/muse --version" in cfg.install_cmd
        assert cfg.install_cmd.index(
            "/opt/benchflow/muse/muse --version"
        ) < cfg.install_cmd.index("muse-acp-shim")
        assert "npm install" not in cfg.install_cmd
        assert "bf_shim_python=/usr/bin/python3" in cfg.launch_cmd
        assert "bf_shim_python=/usr/local/bin/python3" in cfg.launch_cmd
        assert (
            'exec "$bf_shim_python" -I -S /opt/benchflow/bin/muse-acp-shim'
            in cfg.launch_cmd
        )

    def test_install_cmd_is_idempotent_on_matching_version(self, tmp_path):
        """A binary already reporting MUSE_VERSION short-circuits the download."""
        fake = tmp_path / "opt" / "benchflow" / "muse" / "muse"
        fake.parent.mkdir(parents=True)
        fake.write_text(f"#!/bin/sh\necho 'Muse Code 1.4.7 ({cfgmod.MUSE_VERSION})'\n")
        fake.chmod(0o755)
        cmd = cfgmod.muse_install_cmd("exit 99").replace(
            "/opt/benchflow/muse", str(fake.parent)
        )
        result = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert cfgmod.MUSE_VERSION in result.stdout


# ── shim helpers ─────────────────────────────────────────────────────────────


def _load_shim_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("muse_acp_shim", _SHIM)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("muse-wonder-full-1.0", "muse-wonder-full-1.0"),
        ("meta/muse-spark-1.3", "muse-spark-1.3"),
        ("  muse-spark-1.2 ", "muse-spark-1.2"),
    ],
)
def test_shim_normalizes_model_ids(model, expected):
    assert _load_shim_module().normalize_model(model) == expected


def test_shim_refuses_contributor_model_ids():
    with pytest.raises(ValueError, match="contributor"):
        _load_shim_module().normalize_model("muse-spark-1.3-contributor")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("none", "minimal"),
        ("max", "max"),
        ("XHIGH", "xhigh"),
        ("ultra", "ultra"),
        ("", None),
    ],
)
def test_shim_maps_benchflow_efforts_onto_muse_tiers(value, expected):
    assert _load_shim_module().normalize_effort(value) == expected


def test_shim_rejects_unknown_effort_labels():
    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        _load_shim_module().normalize_effort("extreme")


def test_shim_reads_the_reset_time_of_a_quota_error():
    shim = _load_shim_module()
    text = (
        "run ended with Failed: API error 429 [request_id=abc]: Subscription quota "
        "exhausted. Your usage window resets at 2026-10-13T05:31:21Z. (rate_limit_error)"
    )
    expected = datetime.datetime(2026, 10, 13, 5, 31, 21, tzinfo=datetime.UTC)
    assert shim.parse_quota_reset(text) == expected.timestamp()
    assert shim.parse_quota_reset("API error 500: upstream unavailable") is None


# ── shim protocol against a fake muse ────────────────────────────────────────

_FAKE_MUSE = textwrap.dedent(
    r"""
    #!/usr/bin/env python3
    import datetime, json, os, sys, time
    args = sys.argv[1:]
    log = open(os.environ["FAKE_MUSE_LOG"], "a")
    if "--version" in args:
        print("Muse Code 9.9.9 (9.9.9-R1)"); sys.exit(0)
    def opt(name):
        return args[args.index(name) + 1] if name in args else None
    prompt = open(opt("--prompt-file")).read()
    sid = opt("--session-id")
    log.write(json.dumps({
        "args": args, "cwd": os.getcwd(), "prompt": prompt,
        "has_api_key": "META_API_KEY" in os.environ,
        "has_auth_json_env": "MUSE_AUTH_JSON" in os.environ,
        "backend": os.environ.get("TBH_CREDENTIAL_BACKEND"),
    }) + "\n"); log.flush()
    data = os.environ.get("XDG_DATA_HOME") or os.path.join(os.environ["HOME"], ".local", "share")
    d = os.path.join(data, "muse", "sessions", "2026", "10", "13", sid)
    os.makedirs(os.path.join(d, "subagent", "r1"), exist_ok=True)
    def ev(path, event):
        with open(path, "a") as f:
            f.write(json.dumps({"payload": {"kind": "run", "event": event}}) + "\n")
    main = os.path.join(d, "session.jsonl")
    def terminal(state, reason=None, text=""):
        sys.stdout.write(json.dumps({"stream": {"kind": "session", "id": sid}, "payload": {"kind": "run_terminal", "terminal": state, "reason": reason, "text": text}}) + "\n")
        sys.stdout.flush()
    if prompt == "die":
        sys.stderr.write("fatal: boom\n"); sys.exit(3)
    if prompt == "explode":
        ev(main, {"kind": "terminal", "terminal": "failed", "reason": "model exploded"})
        terminal("failed", "model exploded"); sys.exit(1)
    if prompt == "quota":
        reset = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=float(os.environ.get("FAKE_RESET_IN", "1")))
        msg = ("API error 429 [request_id=r1]: Subscription quota exhausted. Your usage window resets at "
               + reset.strftime("%Y-%m-%dT%H:%M:%SZ") + ". (rate_limit_error)")
        ev(main, {"kind": "model_completed", "usage": {"input_tokens": 5, "cached_tokens": 0, "output_tokens": 1, "reasoning_tokens": 0}})
        sys.stderr.write("run ended with Failed: " + msg + "\n"); sys.stderr.flush()
        terminal("failed", msg); sys.exit(1)
    ev(main, {"kind": "assistant_tool_calls_committed", "tool_calls": [
        {"call_id": "call_1", "name": "bash", "args": json.dumps({"command": "ls -la", "description": "list"})},
        {"call_id": "call_2", "name": "read_skill", "args": json.dumps({"name": "citation-management"})},
    ]})
    time.sleep(0.4)
    ev(main, {"kind": "tool_result_batch_committed", "results": [
        {"tool_call_id": "call_1", "tool_call_index": 0, "text": json.dumps({"command": "ls -la", "exit_code": 0, "terminal_status": "completed", "output": "total 0"})},
        {"tool_call_id": "call_2", "tool_call_index": 1, "text": json.dumps({"error": "no such skill"})},
    ]})
    ev(main, {"kind": "todo_snapshot_updated", "items": [{"status": "in_progress", "text": "measure"}, {"status": "pending", "text": "write paper"}]})
    ev(main, {"kind": "reasoning_summary_committed", "text": "thinking about it"})
    ev(main, {"kind": "model_completed", "usage": {"input_tokens": 100, "cached_tokens": 40, "output_tokens": 20, "reasoning_tokens": 5}})
    ev(os.path.join(d, "subagent", "r1", "session.jsonl"), {"kind": "model_completed", "usage": {"input_tokens": 10, "cached_tokens": 0, "output_tokens": 2, "reasoning_tokens": 1}})
    ev(main, {"kind": "assistant_message_committed", "text": "done: " + prompt[:40]})
    ev(main, {"kind": "model_completed", "usage": {"input_tokens": 50, "cache_read_tokens": 30, "output_tokens": 10, "reasoning_tokens": 0}})
    ev(main, {"kind": "terminal", "terminal": "completed", "reason": None})
    terminal("completed", None, "done")
    """
).lstrip()


class _ShimClient:
    """Minimal JSON-RPC driver for the shim subprocess."""

    def __init__(self, tmp_path: Path, env_extra: dict | None = None, args=()):
        self.fake = tmp_path / "fake-muse"
        self.fake.write_text(_FAKE_MUSE)
        self.fake.chmod(self.fake.stat().st_mode | stat.S_IXUSR)
        self.log = tmp_path / "fake-muse.log"
        self.home = tmp_path / "home"
        self.home.mkdir(exist_ok=True)
        env = {
            **os.environ,
            "BENCHFLOW_MUSE_BIN": str(self.fake),
            "FAKE_MUSE_LOG": str(self.log),
            "HOME": str(self.home),
            "BENCHFLOW_AGENT_HOME": str(self.home),
            "MUSE_AUTH_JSON": json.dumps(
                {"providers": {"meta": {"mechanism": "oauth"}}}
            ),
            "META_API_KEY": "test-meta-api-key",
        }
        for key in (
            "MUSE_MODEL",
            "MUSE_EFFORT",
            "XDG_DATA_HOME",
            "XDG_CONFIG_HOME",
            "HTTPS_PROXY",
            "https_proxy",
            "SSL_CERT_FILE",
            "BENCHFLOW_DISALLOW_WEB_TOOLS",
            "BENCHFLOW_EGRESS_DENYLIST",
            "BENCHFLOW_MUSE_QUOTA_WAIT_SEC",
        ):
            env.pop(key, None)
        env.update(env_extra or {})
        for key, value in list(env.items()):
            if value is None:
                env.pop(key)
        self.proc = subprocess.Popen(
            [sys.executable, str(_SHIM), *args],
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

    def close(self) -> str:
        _out, err = self.proc.communicate(timeout=20)
        return err

    def muse_log(self) -> list[dict]:
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


def _start(tmp_path, env_extra=None, args=()):
    client = _ShimClient(tmp_path, env_extra, args)
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    init = client.request(
        "initialize", {"protocolVersion": 1, "clientCapabilities": {}}
    )
    return client, workspace, init


def _session(client, workspace) -> str:
    new = client.request("session/new", {"cwd": str(workspace), "mcpServers": []})
    session_id = new["result"]["sessionId"]
    assert [o["id"] for o in new["result"]["configOptions"]] == ["effort"]
    return session_id


@pytest.fixture
def shim(tmp_path):
    client, workspace, init = _start(tmp_path)
    assert init["result"]["agentInfo"] == {
        "name": "muse-code",
        "version": "Muse Code 9.9.9 (9.9.9-R1)",
        "title": "Meta Muse Code",
    }
    client.workspace = workspace
    yield client
    if client.proc.poll() is None:
        client.proc.kill()


def test_shim_streams_tool_calls_messages_plan_and_usage(shim):
    session_id = _session(shim, shim.workspace)
    assert (
        shim.request(
            "session/set_model",
            {"sessionId": session_id, "modelId": "muse-wonder-full-1.0"},
        )["result"]
        == {}
    )
    cfg = shim.request(
        "session/set_config_option",
        {"sessionId": session_id, "configId": "effort", "value": "max"},
    )
    assert cfg["result"]["configOptions"][0]["currentValue"] == "max"

    resp = shim.request(
        "session/prompt",
        {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": "measure the gap"}],
        },
    )
    assert resp["result"]["stopReason"] == "end_turn"
    # Main session (two model calls) plus the reminder sub-session; input counts
    # include the cached prefix, OpenAI style.
    assert resp["result"]["usage"] == {
        "inputTokens": 160,
        "cachedReadTokens": 70,
        "outputTokens": 32,
        "thoughtTokens": 6,
        "totalTokens": 192,
    }

    updates = shim.updates()
    calls = [u for u in updates if u["sessionUpdate"] == "tool_call"]
    assert [(c["toolCallId"], c["kind"], c["title"]) for c in calls] == [
        ("call_1", "execute", "bash: ls -la"),
        ("call_2", "skill", "skill: citation-management"),
    ]
    assert calls[0]["rawInput"] == {"command": "ls -la", "description": "list"}
    results = {
        u["toolCallId"]: u for u in updates if u["sessionUpdate"] == "tool_call_update"
    }
    assert results["call_1"]["status"] == "completed"
    assert '"output": "total 0"' in results["call_1"]["rawOutput"]
    assert results["call_2"]["status"] == "failed"
    plan = [u for u in updates if u["sessionUpdate"] == "plan"]
    assert plan[0]["entries"] == [
        {"content": "measure", "status": "in_progress", "priority": "medium"},
        {"content": "write paper", "status": "pending", "priority": "medium"},
    ]
    thoughts = [
        u["content"]["text"]
        for u in updates
        if u["sessionUpdate"] == "agent_thought_chunk"
    ]
    assert thoughts == ["thinking about it"]
    messages = [
        u["content"]["text"]
        for u in updates
        if u["sessionUpdate"] == "agent_message_chunk"
    ]
    assert messages == ["done: measure the gap"]

    (call,) = shim.muse_log()
    args = call["args"]
    assert args[:2] == ["exec", "--json"]
    for flag in (
        "--yolo",
        "--user-input-auto-resolve",
        "--no-foreign-personal-context",
    ):
        assert flag in args
    assert args[args.index("--contributor-mode") + 1] == "off"
    assert args[args.index("--model") + 1] == "muse-wonder-full-1.0"
    assert args[args.index("--reasoning-effort") + 1] == "max"
    assert "telemetry.enabled=false" in args
    assert "--disable-web-tools" not in args
    assert call["cwd"] == str(shim.workspace)
    assert call["prompt"] == "measure the gap"
    assert call["backend"] == "file"


def test_shim_continues_the_same_muse_session_on_the_next_prompt(shim):
    session_id = _session(shim, shim.workspace)
    for text in ("first", "second"):
        resp = shim.request(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]},
        )
        assert resp["result"]["stopReason"] == "end_turn"
    first, second = shim.muse_log()
    sid = first["args"][first["args"].index("--session-id") + 1]
    assert second["args"][second["args"].index("--session-id") + 1] == sid


def test_shim_writes_the_login_and_withholds_the_api_key(shim, tmp_path):
    """A subscription login wins: Muse would otherwise bill META_API_KEY."""
    session_id = _session(shim, shim.workspace)
    shim.request("session/prompt", {"sessionId": session_id, "prompt": "hello"})
    auth = shim.home / ".config" / "muse" / "auth.json"
    assert json.loads(auth.read_text())["providers"]["meta"]["mechanism"] == "oauth"
    assert stat.S_IMODE(auth.stat().st_mode) == 0o600
    (call,) = shim.muse_log()
    assert call["has_api_key"] is False
    assert call["has_auth_json_env"] is False


@pytest.mark.parametrize("api_key", ["test-meta-api-key", None])
def test_shim_initialize_fails_closed_without_a_login(tmp_path, api_key):
    """An API key alone is not a credential: pay-as-you-go has no proxy route."""
    client, _workspace, init = _start(
        tmp_path, {"MUSE_AUTH_JSON": None, "META_API_KEY": api_key}
    )
    assert "No Muse Code login" in init["error"]["message"]
    client.proc.kill()


class TestRouting:
    """A login run is native subscription auth; nothing else may start."""

    def test_login_runs_skip_the_proxy(self):
        from benchflow.agents.env import uses_native_subscription_auth

        assert uses_native_subscription_auth(
            "muse-code", "muse-wonder-full-1.0", {"MUSE_AUTH_JSON": "{}"}
        )
        assert not uses_native_subscription_auth(
            "muse-code", "muse-wonder-full-1.0", {"META_API_KEY": "k"}
        )

    def test_runs_without_a_login_are_refused(self, monkeypatch):
        from benchflow.agents import env as envmod

        monkeypatch.delenv("MUSE_AUTH_JSON", raising=False)
        monkeypatch.setattr(envmod, "load_dotenv_env", lambda: {})
        with pytest.raises(MissingCredentialError, match="subscription login") as exc:
            envmod.resolve_agent_env(
                "muse-code", "muse-wonder-full-1.0", {"META_API_KEY": "k"}
            )
        assert "muse login" in (exc.value.hint or "")

    def test_the_preflight_stops_a_run_without_a_login(self, monkeypatch):
        """bench eval run's preflight (check_credentials) re-raises only a
        MissingCredentialError and skips anything else, so a bare ValueError
        let the job start and every trial fail at setup with a traceback
        (PY8 AG review round 1, AG-R1-A-2)."""
        from types import SimpleNamespace

        from benchflow.agents import env as envmod
        from benchflow.runtime import check_credentials

        monkeypatch.delenv("MUSE_AUTH_JSON", raising=False)
        monkeypatch.setattr(envmod, "load_dotenv_env", lambda: {})
        with pytest.raises(MissingCredentialError, match="MUSE_AUTH_JSON"):
            check_credentials(
                [SimpleNamespace(agent="muse-code", model=None, agent_env={})]
            )
        # With the login the same preflight passes.
        check_credentials(
            [
                SimpleNamespace(
                    agent="muse-code", model=None, agent_env={"MUSE_AUTH_JSON": "{}"}
                )
            ]
        )

    def test_no_other_inherited_credential_reaches_the_sandbox(self, monkeypatch):
        """A reviewer's Azure key or another provider's key in the same process
        stays out of the Muse sandbox; explicit agent env stays."""
        from benchflow.agents import env as envmod

        monkeypatch.setattr(envmod, "load_dotenv_env", lambda: {})
        monkeypatch.setenv("MUSE_AUTH_JSON", "{}")
        for key in ("AZURE_API_KEY", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
            monkeypatch.setenv(key, "secret")
        resolved = envmod.resolve_agent_env(
            "muse-code", "muse-wonder-full-1.0", {"TZ": "UTC"}
        )
        for key in ("AZURE_API_KEY", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
            assert key not in resolved
        assert resolved["TZ"] == "UTC"
        assert resolved["MUSE_AUTH_JSON"] == "{}"

    def test_the_quota_wait_reaches_the_shim(self, monkeypatch):
        """The campaign sets the shim's quota wait in the runner's environment; it
        must reach the sandbox, or a run that meets a quota wall fails instead of
        waiting for the reset (three rollouts lost on 2026-10-13)."""
        from benchflow.agents import env as envmod
        from benchflow.agents import muse_acp_shim

        monkeypatch.setattr(envmod, "load_dotenv_env", lambda: {})
        monkeypatch.setenv("MUSE_AUTH_JSON", "{}")
        monkeypatch.setenv(muse_acp_shim.QUOTA_WAIT_ENV, "18000")
        monkeypatch.setenv(muse_acp_shim.QUOTA_SLACK_ENV, "45")
        monkeypatch.setenv("AZURE_API_KEY", "secret")
        resolved = envmod.resolve_agent_env("muse-code", "muse-wonder-full-1.0", {})
        assert resolved[muse_acp_shim.QUOTA_WAIT_ENV] == "18000"
        assert resolved[muse_acp_shim.QUOTA_SLACK_ENV] == "45"
        assert "AZURE_API_KEY" not in resolved

    @pytest.mark.parametrize(
        ("agent", "model"),
        [("claude-agent-acp", None), ("opencode", "openai/gpt-5.5")],
    )
    def test_other_agents_never_inherit_the_muse_login(self, monkeypatch, agent, model):
        """A host that exports the Muse login (or keeps it in .env) for muse-code
        must not hand Meta's subscription login to another agent's sandbox.
        Guards PY8 AG's fix: the 0.7.8 side branch (c0eceb9a, d2866142) added
        MUSE_AUTH_JSON and the quota settings to every agent's inherited names."""
        from benchflow.agents import env as envmod
        from benchflow.agents import muse_acp_shim

        dotenv = {"MUSE_AUTH_JSON": '{"from": "dotenv"}'}
        monkeypatch.setattr(envmod, "load_dotenv_env", lambda: dict(dotenv))
        monkeypatch.setenv("MUSE_AUTH_JSON", '{"from": "host"}')
        monkeypatch.setenv(muse_acp_shim.QUOTA_WAIT_ENV, "18000")
        monkeypatch.setenv(muse_acp_shim.QUOTA_SLACK_ENV, "45")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        resolved = envmod.resolve_agent_env(agent, model, {})
        assert "MUSE_AUTH_JSON" not in resolved
        assert muse_acp_shim.QUOTA_WAIT_ENV not in resolved
        assert muse_acp_shim.QUOTA_SLACK_ENV not in resolved

    def test_the_login_comes_from_dotenv_before_the_host(self, monkeypatch):
        """The muse-code run still finds its login where the other names come
        from: .env first, then the host environment; explicit agent env wins."""
        from benchflow.agents import env as envmod

        monkeypatch.setattr(
            envmod, "load_dotenv_env", lambda: {"MUSE_AUTH_JSON": '{"from": "dotenv"}'}
        )
        monkeypatch.setenv("MUSE_AUTH_JSON", '{"from": "host"}')
        resolved = envmod.resolve_agent_env("muse-code", "muse-wonder-full-1.0", {})
        assert resolved["MUSE_AUTH_JSON"] == '{"from": "dotenv"}'
        explicit = envmod.resolve_agent_env(
            "muse-code", "muse-wonder-full-1.0", {"MUSE_AUTH_JSON": '{"x": 1}'}
        )
        assert explicit["MUSE_AUTH_JSON"] == '{"x": 1}'

    def test_network_policies_that_need_a_gateway_refuse_a_login_run(self):
        """0.8 enforces "no internet" and allowlists in the network, and a Muse
        login run talks to Meta with no gateway in between: both refuse it
        before the agent starts (as for every native login but direct Claude),
        instead of trusting the client-side --disable-web-tools flag 0.7.8 used.
        A denylist keeps the run, with the shim's egress-proxy route."""
        from benchflow.sandbox.egress_denylist import EgressDenylist
        from benchflow.sandbox.native_oauth import (
            allowlist_model_transport,
            native_oauth_egress_policy,
        )

        env = {"MUSE_AUTH_JSON": "{}"}
        model = "muse-wonder-full-1.0"
        with pytest.raises(ValueError, match="only for direct Claude ACP"):
            native_oauth_egress_policy("muse-code", model, env, no_web=True)
        assert native_oauth_egress_policy("muse-code", model, env, no_web=False) is None
        allow = EgressDenylist((), (), allowed_hosts=("example.com",))
        with pytest.raises(ValueError, match="native subscription endpoint"):
            allowlist_model_transport(allow, "muse-code", model, env)
        deny = EgressDenylist((), ("example.com",))
        assert allowlist_model_transport(deny, "muse-code", model, env) is deny

    def test_the_login_is_inherited_and_the_api_key_is_not(self, monkeypatch):
        from benchflow.agents import env as envmod

        monkeypatch.setattr(envmod, "load_dotenv_env", lambda: {})
        monkeypatch.setenv("MUSE_AUTH_JSON", "{}")
        monkeypatch.setenv("META_API_KEY", "should-not-travel")
        resolved = envmod.resolve_agent_env("muse-code", "muse-wonder-full-1.0", {})
        assert resolved["MUSE_AUTH_JSON"] == "{}"
        assert "META_API_KEY" not in resolved


def test_shim_initialize_fails_closed_without_binary(tmp_path):
    client, _workspace, init = _start(
        tmp_path, {"BENCHFLOW_MUSE_BIN": str(tmp_path / "nope")}
    )
    assert "binary not found" in init["error"]["message"]
    client.proc.kill()


def test_shim_refuses_contributor_models(shim):
    session_id = _session(shim, shim.workspace)
    resp = shim.request(
        "session/set_model",
        {"sessionId": session_id, "modelId": "muse-spark-1.3-contributor"},
    )
    assert "contributor" in resp["error"]["message"]


@pytest.mark.parametrize(
    ("args", "env", "expected", "absent"),
    [
        (
            ["--disable-web-tools"],
            {},
            ["--disable-web-tools"],
            ['tools.web_search.mode="off"'],
        ),
        (
            ["--disable-hosted-search"],
            {},
            ['tools.web_search.mode="off"'],
            ["--disable-web-tools"],
        ),
        ([], {"BENCHFLOW_EGRESS_DENYLIST": "1"}, ['tools.web_search.mode="off"'], []),
        ([], {}, [], ["--disable-web-tools", 'tools.web_search.mode="off"']),
    ],
)
def test_shim_maps_benchflow_web_policies(tmp_path, args, env, expected, absent):
    client, workspace, _init = _start(tmp_path, env, args)
    session_id = _session(client, workspace)
    client.request("session/prompt", {"sessionId": session_id, "prompt": "hello"})
    (call,) = client.muse_log()
    for item in expected:
        assert item in call["args"]
    for item in absent:
        assert item not in call["args"]
    client.proc.kill()


def test_shim_routes_model_traffic_through_the_egress_proxy(tmp_path):
    bundle = tmp_path / "ca-bundle.crt"
    bundle.write_text("cert")
    client, workspace, _init = _start(
        tmp_path,
        {"HTTPS_PROXY": "http://127.0.0.1:18628", "SSL_CERT_FILE": str(bundle)},
    )
    session_id = _session(client, workspace)
    client.request("session/prompt", {"sessionId": session_id, "prompt": "hello"})
    (call,) = client.muse_log()
    assert 'endpoint_transport.proxy="http://127.0.0.1:18628"' in call["args"]
    assert f'endpoint_transport.ca_bundle="{bundle}"' in call["args"]
    client.proc.kill()


def test_shim_waits_out_an_exhausted_window_and_continues(tmp_path):
    client, workspace, _init = _start(
        tmp_path,
        {
            "BENCHFLOW_MUSE_QUOTA_WAIT_SEC": "60",
            "BENCHFLOW_MUSE_QUOTA_SLACK_SEC": "0",
            "FAKE_RESET_IN": "2",
        },
    )
    session_id = _session(client, workspace)
    started = time.monotonic()
    resp = client.request(
        "session/prompt", {"sessionId": session_id, "prompt": "quota"}
    )
    assert resp["result"]["stopReason"] == "end_turn"
    assert time.monotonic() - started >= 1.0
    first, second = client.muse_log()
    assert first["prompt"] == "quota"
    assert second["prompt"].startswith("The previous model request stopped")
    sid = first["args"][first["args"].index("--session-id") + 1]
    assert second["args"][second["args"].index("--session-id") + 1] == sid
    waits = [
        u
        for u in client.updates()
        if u.get("toolCallId", "").startswith("muse-quota-wait-")
    ]
    assert waits[0]["sessionUpdate"] == "tool_call"
    assert waits[0]["title"].startswith("wait for the Muse usage window to reset at ")
    assert waits[-1]["status"] == "completed"
    # The quota-failed call's tokens still count.
    assert resp["result"]["usage"]["inputTokens"] == 165
    client.proc.kill()


def test_shim_never_waits_past_the_prompt_budget(tmp_path):
    """A reset that lands past BenchFlow's prompt budget (less the working
    margin) fails the prompt at once with the usage-limit text, instead of
    holding the turn until BenchFlow's wall-clock timeout scores a partial
    workspace as an agent timeout (PY8 AG review round 1, AG-R1-A-3)."""
    client, workspace, _init = _start(
        tmp_path,
        {
            "BENCHFLOW_MUSE_QUOTA_WAIT_SEC": "18000",
            "BENCHFLOW_MUSE_QUOTA_SLACK_SEC": "0",
            # The margin is 300 s: a 330 s budget leaves 30 s to wait in.
            "BENCHFLOW_MUSE_PROMPT_BUDGET_SEC": "330",
            "FAKE_RESET_IN": "120",
        },
    )
    session_id = _session(client, workspace)
    started = time.monotonic()
    resp = client.request(
        "session/prompt", {"sessionId": session_id, "prompt": "quota"}
    )
    assert time.monotonic() - started < 30
    assert resp["error"]["message"].startswith(
        "Muse subscription quota exhausted; the usage window resets at "
    )
    assert len(client.muse_log()) == 1
    assert not [
        u
        for u in client.updates()
        if u.get("toolCallId", "").startswith("muse-quota-wait-")
    ]
    client.proc.kill()


def test_shim_still_waits_for_a_reset_inside_the_prompt_budget(tmp_path):
    client, workspace, _init = _start(
        tmp_path,
        {
            "BENCHFLOW_MUSE_QUOTA_WAIT_SEC": "60",
            "BENCHFLOW_MUSE_QUOTA_SLACK_SEC": "0",
            "BENCHFLOW_MUSE_PROMPT_BUDGET_SEC": "900",
            "FAKE_RESET_IN": "2",
        },
    )
    session_id = _session(client, workspace)
    resp = client.request(
        "session/prompt", {"sessionId": session_id, "prompt": "quota"}
    )
    assert resp["result"]["stopReason"] == "end_turn"
    assert len(client.muse_log()) == 2
    client.proc.kill()


def test_only_muse_gets_the_prompt_budget():
    from benchflow.agents import muse_acp_shim
    from benchflow.agents.env import with_prompt_budget

    env = {"MUSE_AUTH_JSON": "{}"}
    assert with_prompt_budget("muse-code", env, 1800) == {
        "MUSE_AUTH_JSON": "{}",
        muse_acp_shim.PROMPT_BUDGET_ENV: "1800",
    }
    assert env == {"MUSE_AUTH_JSON": "{}"}
    assert with_prompt_budget("muse-code", env, 0) is env
    assert with_prompt_budget("claude-agent-acp", env, 1800) is env


def _connect_planes():
    from unittest.mock import AsyncMock, MagicMock

    planes = MagicMock()
    planes.ensure_litellm_runtime = AsyncMock(
        side_effect=lambda **kwargs: (kwargs["agent_env"], None)
    )
    planes.connect_acp = AsyncMock(
        return_value=(MagicMock(), MagicMock(), MagicMock(), "agent")
    )
    # Per-connection credential staging and web-tool policy (C23, C30).
    planes.write_credential_files = AsyncMock(return_value=[])
    planes.upload_subscription_auth = AsyncMock(return_value=[])
    planes.apply_web_tool_policy = AsyncMock()
    planes.quiesce_agent = AsyncMock()
    return planes


def _muse_rollout(tmp_path):
    from types import SimpleNamespace

    from benchflow.rollout import Rollout, RolloutConfig

    trial = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task", agent="muse-code", model="muse-spark-1.3"
        )
    )
    from unittest.mock import AsyncMock, MagicMock

    from benchflow.sandbox.protocol import ExecResult

    trial._env = MagicMock()
    trial._env.exec = AsyncMock(return_value=ExecResult(0, "", ""))
    trial._rollout_dir = tmp_path
    trial._agent_launch = "/opt/benchflow/bin/muse-acp-shim"
    trial._agent_env = {"MUSE_AUTH_JSON": "{}"}
    trial._timeout = 1800
    trial._phase = "installed"
    trial._task = SimpleNamespace(
        config=SimpleNamespace(sandbox=SimpleNamespace(mcp_servers=[]))
    )
    trial._planes = _connect_planes()
    trial._planes.write_credential_files = AsyncMock()
    return trial


async def test_the_rollout_hands_the_shim_its_prompt_budget(tmp_path):
    """The shim can only stay inside BenchFlow's prompt timeout if it is told
    it (PY8 AG review round 1, AG-R1-A-3)."""
    from benchflow.agents import muse_acp_shim

    trial = _muse_rollout(tmp_path)
    await trial.connect()
    env = trial._planes.connect_acp.await_args.kwargs["agent_env"]
    assert env[muse_acp_shim.PROMPT_BUDGET_ENV] == "1800"
    assert env["MUSE_AUTH_JSON"] == "{}"


async def test_a_role_hands_the_shim_its_own_prompt_budget(tmp_path):
    from unittest.mock import AsyncMock

    from benchflow.agents import muse_acp_shim
    from benchflow.rollout import Role

    trial = _muse_rollout(tmp_path)
    trial._agent_cfg = AGENTS["muse-code"]
    trial._planes.agent_launch.return_value = "/opt/benchflow/bin/muse-acp-shim"
    trial._planes.resolve_agent_env.side_effect = lambda _agent, _model, env: {
        "MUSE_AUTH_JSON": "{}",
        **(env or {}),
    }
    trial._planes.install_agent = AsyncMock(return_value=AGENTS["muse-code"])
    trial._planes.write_credential_files = AsyncMock()
    trial._planes.upload_subscription_auth = AsyncMock()
    trial._planes.apply_web_tool_policy = AsyncMock()

    await trial.connect_as(
        Role(name="solver", agent="muse-code", model="muse-spark-1.3", timeout_sec=600)
    )
    env = trial._planes.connect_acp.await_args.kwargs["agent_env"]
    assert env[muse_acp_shim.PROMPT_BUDGET_ENV] == "600"


def test_shim_fails_with_the_reset_time_when_waiting_is_off(tmp_path):
    client, workspace, _init = _start(tmp_path, {"FAKE_RESET_IN": "3600"})
    session_id = _session(client, workspace)
    resp = client.request(
        "session/prompt", {"sessionId": session_id, "prompt": "quota"}
    )
    message = resp["error"]["message"]
    assert message.startswith(
        "Muse subscription quota exhausted; the usage window resets at "
    )
    assert len(client.muse_log()) == 1
    # BenchFlow reads the answer as a typed usage limit with the reset time.
    from benchflow.acp.client import acp_error
    from benchflow.agents.errors import UsageLimitError

    err = acp_error(resp["error"]["code"], message)
    assert isinstance(err, UsageLimitError)
    assert err.resets_at is not None
    left = (err.resets_at - datetime.datetime.now(datetime.UTC)).total_seconds()
    assert left > 3500
    client.proc.kill()


def test_a_spent_muse_login_is_an_unscored_usage_limit_not_a_retried_acp_error():
    """The shim's quota answer was a plain ACPError, classified acp_error and
    retried, so a job kept starting trials (each installing the 363 MB
    binary) on a login that could not answer until the reset (PY8 AG review
    round 1, AG-R1-A-1)."""
    from benchflow._utils.scoring import USAGE_LIMIT, classify_error
    from benchflow.acp.client import acp_error
    from benchflow.agents.errors import UsageLimitError

    message = (
        "Muse subscription quota exhausted; the usage window resets at "
        "2026-10-13T05:31:21Z"
    )
    err = acp_error(-32603, message)
    assert isinstance(err, UsageLimitError)
    assert err.resets_at == datetime.datetime(
        2026, 10, 13, 5, 31, 21, tzinfo=datetime.UTC
    )
    assert err.detail == message
    assert classify_error(str(err)) == USAGE_LIMIT
    assert classify_error(f"ACP error -32603: {message}") == USAGE_LIMIT


def test_shim_surfaces_failed_runs_and_process_death(shim):
    session_id = _session(shim, shim.workspace)
    resp = shim.request(
        "session/prompt", {"sessionId": session_id, "prompt": "explode"}
    )
    assert "model exploded" in resp["error"]["message"]
    resp = shim.request("session/prompt", {"sessionId": session_id, "prompt": "die"})
    assert "without a terminal state" in resp["error"]["message"]
    assert "fatal: boom" in resp["error"]["message"]


def test_shim_rejects_unknown_config_option(shim):
    session_id = _session(shim, shim.workspace)
    resp = shim.request(
        "session/set_config_option",
        {"sessionId": session_id, "configId": "temperature", "value": "1"},
    )
    assert resp["error"]["code"] == -32602
