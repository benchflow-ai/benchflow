"""Egress blocklist (``network_mode = "blocklist"``) enforcement tests.

Guards the egress-blocklist PR: policy resolution, agent-env shaping, the
per-harness web-tool knobs, the LiteLLM server-tool rewrite, the lockdown
gate, the mocked start/verify flows, and — on Linux with openssl — a real
run of the in-sandbox proxy covering plain HTTP, CONNECT tunnels, and TLS
inspection with path rules.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox.egress import (
    BLOCKED_STATUS,
    EGRESS_BLOCKED_URLS_ENV,
    EGRESS_LOG_PATH,
    EgressBlocklist,
    apply_blocklist_env,
    blocklist_active,
    egress_proxy_source,
    match_blocked,
    start_egress_proxy,
    strip_blocklist_secret,
    strip_proxy_env,
    verify_egress_blocklist,
)
from benchflow.task.config import TaskConfig

LINUX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="needs bash/Linux")
NEEDS_OPENSSL = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("openssl") is None,
    reason="needs openssl + Linux sockets",
)


# ------------------------------------------------------------------ matching


@pytest.mark.parametrize(
    ("host", "path", "expected"),
    [
        ("arxiv.org", "/abs/2401.12345", "arxiv.org/abs/2401.12345"),
        ("ARXIV.ORG", "/abs/2401.12345v2", "arxiv.org/abs/2401.12345"),
        ("export.arxiv.org", "/abs/2401.12345?x=1", "arxiv.org/abs/2401.12345"),
        ("arxiv.org", "/abs/2402.00001", None),
        ("arxiv.org", "/", None),
        ("notarxiv.org", "/abs/2401.12345", None),
        ("openreview.net", "/forum?id=abc", "openreview.net"),
        ("api.openreview.net", "/", "openreview.net"),
        ("openreview.net.evil.com", "/", None),
    ],
)
def test_match_blocked_host_suffix_and_path_prefix(host, path, expected):
    rules = ("arxiv.org/abs/2401.12345", "openreview.net")
    assert match_blocked(rules, host, path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "/abs/%32%34%30%31%2e%31%32%33%34%35",  # percent-encoded digits
        "//abs///2401.12345",  # slash runs
        "/other/../abs/2401.12345",  # dot segments
        "/./abs/./2401.12345v2?download=1#x",  # dot-segments + query + fragment
        "/ABS/2401.12345",  # case
        "/abs/2401.12345/",  # trailing slash
    ],
)
def test_match_blocked_normalizes_like_an_upstream_server(path):
    """Review round 5 (WAF bypass): matching must see the path the upstream
    server will route, not the raw bytes the agent typed."""
    rules = ("arxiv.org/abs/2401.12345",)
    assert match_blocked(rules, "arxiv.org", path) == "arxiv.org/abs/2401.12345"
    assert match_blocked(rules, "arxiv.org", "/abs/%32%34%30%32.1") is None


def test_normalize_request_path_edge_cases():
    from benchflow.sandbox.egress import normalize_request_path

    assert normalize_request_path("/") == ""
    assert normalize_request_path("") == ""
    assert normalize_request_path("/../../etc") == "etc"
    assert normalize_request_path("/a/%2e%2e/b") == "b"
    # Double / triple encoding decodes until stable (Devin review).
    assert normalize_request_path("/abs%252f2401.12345") == "abs/2401.12345"
    assert normalize_request_path("/abs/%2532%2534") == "abs/24"
    assert match_blocked(
        ("arxiv.org/abs/2401.12345",), "arxiv.org", "/abs%252f2401.12345"
    )


def test_blocklist_tls_inspection_hosts_are_only_hosts_with_path_rules():
    b = EgressBlocklist(rules=("arxiv.org/abs/1", "arxiv.org/pdf/1", "openreview.net"))
    assert b.tls_inspection_hosts == ("arxiv.org",)
    assert b.needs_tls_inspection is True
    assert EgressBlocklist(rules=("openreview.net",)).needs_tls_inspection is False


def test_blocklist_requires_rules():
    with pytest.raises(ValueError, match="at least one rule"):
        EgressBlocklist(rules=())


# ------------------------------------------------------------------ resolution


def test_from_task_config_uses_sandbox_then_agent_override():
    sandbox_only = TaskConfig.model_validate(
        {"sandbox": {"network_mode": "blocklist", "blocked_urls": ["openreview.net"]}}
    )
    assert EgressBlocklist.from_task_config(sandbox_only).rules == ("openreview.net",)

    agent_override = TaskConfig.model_validate(
        {
            "agent": {"network_mode": "blocklist", "blocked_urls": ["arxiv.org/abs/1"]},
            "sandbox": {
                "network_mode": "blocklist",
                "blocked_urls": ["openreview.net"],
            },
        }
    )
    assert EgressBlocklist.from_task_config(agent_override).rules == (
        "arxiv.org/abs/1",
    )

    agent_public = TaskConfig.model_validate(
        {
            "agent": {"network_mode": "public"},
            "sandbox": {
                "network_mode": "blocklist",
                "blocked_urls": ["openreview.net"],
            },
        }
    )
    # Devin review: an explicit agent-level mode must not silently shadow a
    # sandbox blocklist (which is where --block-url lands) — it is a conflict.
    with pytest.raises(ValueError, match="overridden by the agent-level"):
        EgressBlocklist.from_task_config(agent_public)
    assert EgressBlocklist.from_task_config(TaskConfig.model_validate({})) is None
    agent_blocklist_only = TaskConfig.model_validate(
        {"agent": {"network_mode": "blocklist", "blocked_urls": ["x.org"]}}
    )
    assert EgressBlocklist.from_task_config(agent_blocklist_only).rules == ("x.org",)


def test_agent_env_routes_through_proxy_without_revealing_rules():
    b = EgressBlocklist(rules=("arxiv.org/abs/2401.12345",), proxy_port=61399)
    env = b.agent_env()
    assert env["HTTP_PROXY"] == env["HTTPS_PROXY"] == "http://127.0.0.1:61399"
    assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert env["NODE_USE_ENV_PROXY"] == "1"
    assert env["SSL_CERT_FILE"] == env["NODE_EXTRA_CA_CERTS"]
    # The hidden URLs must never be readable from the agent process env.
    assert EGRESS_BLOCKED_URLS_ENV not in env
    assert "2401.12345" not in json.dumps(env)

    no_tls = EgressBlocklist(rules=("openreview.net",)).agent_env()
    assert "SSL_CERT_FILE" not in no_tls


def test_apply_and_strip_blocklist_env_round_trip():
    b = EgressBlocklist(rules=("openreview.net",))
    applied = apply_blocklist_env({"FOO": "1"}, b)
    assert blocklist_active(applied)
    assert EgressBlocklist.from_env(applied) == b
    assert apply_blocklist_env({"FOO": "1"}, None) == {"FOO": "1"}

    process_env = strip_blocklist_secret(applied)
    assert EGRESS_BLOCKED_URLS_ENV not in process_env
    assert process_env["HTTP_PROXY"] == b.proxy_url
    assert process_env["FOO"] == "1"
    assert not blocklist_active(process_env)


def test_strip_proxy_env_keeps_rules_for_model_proxy():
    b = EgressBlocklist(rules=("arxiv.org/abs/1",))
    env = strip_proxy_env(apply_blocklist_env({"OPENAI_API_KEY": "k"}, b))
    assert "HTTP_PROXY" not in env and "SSL_CERT_FILE" not in env
    assert env[EGRESS_BLOCKED_URLS_ENV] == b.to_env_value()
    assert env["OPENAI_API_KEY"] == "k"


# ------------------------------------------------------------------ rollout policy


def test_apply_web_policy_marks_blocklist_and_no_web_independently():
    from benchflow.rollout._setup import _apply_web_policy, _task_egress_blocklist

    b = EgressBlocklist(rules=("openreview.net",))
    env = _apply_web_policy({}, disallow=False, blocklist=b)
    assert "BENCHFLOW_DISALLOW_WEB_TOOLS" not in env
    assert blocklist_active(env)
    both = _apply_web_policy({}, disallow=True, blocklist=None)
    assert both == {"BENCHFLOW_DISALLOW_WEB_TOOLS": "1"}

    task = SimpleNamespace(
        config=TaskConfig.model_validate(
            {
                "sandbox": {
                    "network_mode": "blocklist",
                    "blocked_urls": ["openreview.net"],
                }
            }
        )
    )
    assert _task_egress_blocklist(task) == b
    assert _task_egress_blocklist(SimpleNamespace()) is None


def test_agent_launch_applies_blocklist_suffix_only_for_server_side_search():
    from benchflow.agents.registry import AGENT_LAUNCH
    from benchflow.rollout._setup import _agent_launch_with_web_policy

    codex = _agent_launch_with_web_policy("codex-acp", disallow=False, blocklist=True)
    assert codex == AGENT_LAUNCH["codex-acp"] + " -c tools.web_search=false"
    # A no-web run keeps its own (identical here) knob and wins over blocklist.
    assert (
        _agent_launch_with_web_policy("codex-acp", disallow=True, blocklist=True)
        == codex
    )
    # Client-side fetch tools go through the proxy, so nothing is appended.
    for agent in ("claude-agent-acp", "opencode", "openhands", "gemini"):
        assert (
            _agent_launch_with_web_policy(agent, disallow=False, blocklist=True)
            == AGENT_LAUNCH[agent]
        )


def test_registry_blocklist_knobs_cover_only_unfilterable_server_tools():
    from benchflow.agents.registry import AGENTS

    assert (
        AGENTS["codex-acp"].blocklist_web_tools_launch_suffix
        == " -c tools.web_search=false"
    )
    assert "google_web_search" in AGENTS["gemini"].blocklist_web_tools_setup_cmd
    # Anthropic server tools are filtered in the LiteLLM hook; client-side
    # fetchers (OpenCode/MiMo webfetch, OpenHands browsing) stay enabled.
    for agent in ("claude-agent-acp", "opencode", "mimo", "openhands", "pi-acp"):
        cfg = AGENTS[agent]
        assert cfg.blocklist_web_tools_setup_cmd == ""
        assert cfg.blocklist_web_tools_launch_suffix == ""


def test_manifest_contract_keeps_blocklist_knobs_shim_only():
    from benchflow.agents.manifest import _SHIM_ONLY

    assert {
        "blocklist_web_tools_setup_cmd",
        "blocklist_web_tools_launch_suffix",
    } <= _SHIM_ONLY


@LINUX_ONLY
def test_gemini_blocklist_setup_cmd_excludes_server_side_tools(tmp_path):
    from benchflow.agents.registry import AGENTS

    home = tmp_path / "home"
    subprocess.run(
        ["bash", "-c", AGENTS["gemini"].blocklist_web_tools_setup_cmd],
        check=True,
        env={**os.environ, "BENCHFLOW_AGENT_HOME": str(home)},
    )
    settings = json.loads((home / ".gemini" / "settings.json").read_text())
    assert settings["tools"]["exclude"] == ["google_web_search", "web_fetch"]


@pytest.mark.asyncio
async def test_apply_web_tool_policy_blocklist_runs_narrow_cmd_and_no_web_wins():
    from benchflow.agents.install import apply_web_tool_policy
    from benchflow.agents.registry import AgentConfig

    cfg = AgentConfig(
        name="x",
        install_cmd="true",
        launch_cmd="x",
        disallow_web_tools_setup_cmd="echo no-web",
        blocklist_web_tools_setup_cmd="echo blocklist",
    )
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0))

    await apply_web_tool_policy(env, "x", cfg, "/root", disallow=False, blocklist=True)
    assert "echo blocklist" in env.exec.await_args.args[0]

    env.exec.reset_mock()
    await apply_web_tool_policy(env, "x", cfg, "/root", disallow=True, blocklist=True)
    assert "echo no-web" in env.exec.await_args.args[0]

    env.exec.reset_mock()
    await apply_web_tool_policy(env, "x", cfg, "/root", disallow=False, blocklist=False)
    env.exec.assert_not_awaited()


# ------------------------------------------------------------------ lockdown gate


@pytest.mark.asyncio
async def test_firewall_fires_under_blocklist_and_requires_loopback_model_proxy():
    from benchflow.sandbox.lockdown import (
        agent_network_policy_active,
        enforce_agent_egress_firewall,
    )

    b = EgressBlocklist(rules=("openreview.net",))
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0))
    agent_env = apply_blocklist_env(
        {"BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:4000/v1"}, b
    )
    assert agent_network_policy_active(agent_env)
    assert not agent_network_policy_active({"HTTP_PROXY": b.proxy_url})

    await enforce_agent_egress_firewall(env, "agent", agent_env)
    env.exec.assert_awaited_once()
    assert '--uid-owner "$agent_uid" -j REJECT' in env.exec.await_args.args[0]

    with pytest.raises(RuntimeError, match="loopback provider base URL"):
        await enforce_agent_egress_firewall(
            env,
            "agent",
            apply_blocklist_env({"OPENAI_BASE_URL": "https://api.openai.com/v1"}, b),
        )


# ------------------------------------------------------------------ start / verify (mocked)


def _exec_recorder(responses):
    """Build an ``env.exec`` mock whose result depends on the command text."""
    calls: list[tuple[str, dict]] = []

    async def _exec(cmd, **kwargs):
        calls.append((cmd, kwargs))
        for needle, result in responses:
            if needle in cmd:
                return result
        return MagicMock(return_code=0, stdout="", stderr="")

    return AsyncMock(side_effect=_exec), calls


@pytest.mark.asyncio
async def test_start_egress_proxy_uploads_policy_and_waits_for_health():
    b = EgressBlocklist(rules=("arxiv.org/abs/2401.12345",))
    health_states = iter(
        [1, 1, 0]
    )  # first probe (already-running check) fails, then up

    async def _exec(cmd, **kwargs):
        if "/healthz" in cmd:
            return MagicMock(return_code=next(health_states), stdout="", stderr="")
        return MagicMock(return_code=0, stdout="", stderr="")

    env = MagicMock()
    env.exec = AsyncMock(side_effect=_exec)
    env.upload_file = AsyncMock()

    await start_egress_proxy(env, b)

    uploaded = {
        str(call.args[1]): Path(call.args[0])
        for call in env.upload_file.await_args_list
    }
    assert "/opt/benchflow/egress/egress_proxy.py" in uploaded
    assert "/opt/benchflow/egress/policy.json" in uploaded
    root_cmds = [
        c.args[0] for c in env.exec.await_args_list if c.kwargs.get("user") == "root"
    ]
    assert all(c.kwargs.get("user") == "root" for c in env.exec.await_args_list)
    # TLS inspection (path rule) provisions openssl + the per-run CA.
    assert any("openssl req -x509" in c for c in root_cmds)
    assert any("chmod 600 /opt/benchflow/egress/policy.json" in c for c in root_cmds)
    assert any(
        "nohup python3 /opt/benchflow/egress/egress_proxy.py" in c for c in root_cmds
    )


@pytest.mark.asyncio
async def test_start_egress_proxy_is_idempotent_when_already_healthy():
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
    env.upload_file = AsyncMock()

    await start_egress_proxy(env, EgressBlocklist(rules=("openreview.net",)))

    env.upload_file.assert_not_awaited()
    env.exec.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_egress_proxy_fails_loudly_without_python3():
    async def _exec(cmd, **kwargs):
        if "/healthz" in cmd:
            return MagicMock(return_code=1, stdout="", stderr="")
        return MagicMock(return_code=87, stdout="", stderr="python3 is required")

    env = MagicMock()
    env.exec = AsyncMock(side_effect=_exec)
    env.upload_file = AsyncMock()
    with pytest.raises(RuntimeError, match="python3 is required"):
        await start_egress_proxy(env, EgressBlocklist(rules=("openreview.net",)))


@pytest.mark.asyncio
async def test_verify_egress_blocklist_runs_probe_as_sandbox_user_and_logs_it():
    b = EgressBlocklist(rules=("openreview.net",))
    probe = {"via_proxy_status": BLOCKED_STATUS, "direct_egress": "blocked", "ok": True}
    env = MagicMock()
    env.exec, calls = _exec_recorder(
        [
            (
                "python3 - <<'PY'",
                MagicMock(return_code=0, stdout=json.dumps(probe), stderr=""),
            )
        ]
    )

    result = await verify_egress_blocklist(env, "agent", apply_blocklist_env({}, b))

    assert result == probe
    probe_call = next(c for c in calls if "python3 - <<'PY'" in c[0])
    assert probe_call[1]["user"] == "agent"
    assert "target = '/'" in probe_call[0]
    log_call = next(c for c in calls if EGRESS_LOG_PATH in c[0] and "printf" in c[0])
    assert log_call[1]["user"] == "root"
    assert '"event": "probe"' in log_call[0]


def test_probe_targets_the_rule_path_not_the_host_root():
    """Review bug: a path-only rule leaves the host root open, so the self-check
    must probe host/path (which the proxy hides) — probing "/" reported 200
    and aborted correctly configured rollouts."""
    from benchflow.sandbox.egress import _probe_cmd

    cmd = _probe_cmd(EgressBlocklist(rules=("arxiv.org/abs/2401.12345", "x.org")))
    assert "host = 'arxiv.org'" in cmd
    assert "target = '/abs/2401.12345'" in cmd
    assert "opener.open('http://' + host + target" in cmd
    assert "'/'" not in cmd.split("target = ")[1].split("\n")[0]


@NEEDS_OPENSSL
def test_probe_passes_end_to_end_against_real_proxy_for_path_only_rule(
    running_proxy, monkeypatch
):
    """Run the actual self-check script against the live proxy: the first rule
    is path-only, so only the exact blocked path yields the blocked status."""
    from benchflow.sandbox.egress import _probe_cmd

    p = running_proxy
    # The fixture's first rule is a host rule; build a path-only blocklist that
    # points at the same proxy and matches the fixture's "localhost/secret".
    b = EgressBlocklist(rules=("localhost/secret",), proxy_port=p.proxy_port)
    script = _probe_cmd(b).split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    # The direct-egress leg needs the UID firewall; emulate it by making the
    # raw socket connect fail the way REJECT does.
    script = script.replace(
        "socket.create_connection(('1.1.1.1', 443), timeout=5)",
        "(_ for _ in ()).throw(OSError('Connection refused'))",
    )
    # localhost:<tls_port> is the inspected origin; plain-HTTP probe goes to the
    # HTTP origin on the same host name so the path rule applies.
    script = script.replace(
        "'http://' + host + target", f"'http://' + host + ':{p.http_port}' + target"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["blocked_target"] == "localhost/secret"
    assert out["via_proxy_status"] == BLOCKED_STATUS
    assert out["ok"] is True, out
    assert result.returncode == 0


@pytest.mark.asyncio
async def test_verify_egress_blocklist_fails_when_hidden_host_is_reachable():
    b = EgressBlocklist(rules=("openreview.net",))
    probe = {"via_proxy_status": 200, "direct_egress": "open", "ok": False}
    env = MagicMock()
    env.exec, _ = _exec_recorder(
        [
            (
                "python3 - <<'PY'",
                MagicMock(return_code=1, stdout=json.dumps(probe), stderr=""),
            )
        ]
    )
    with pytest.raises(RuntimeError, match="self-check failed"):
        await verify_egress_blocklist(env, "agent", apply_blocklist_env({}, b))


@pytest.mark.asyncio
async def test_verify_egress_blocklist_is_noop_without_policy_and_needs_sandbox_user():
    env = MagicMock()
    env.exec = AsyncMock()
    assert await verify_egress_blocklist(env, "agent", {}) is None
    env.exec.assert_not_awaited()
    with pytest.raises(RuntimeError, match="requires a sandbox_user"):
        await verify_egress_blocklist(
            env, None, apply_blocklist_env({}, EgressBlocklist(rules=("x.org",)))
        )


# ------------------------------------------------------------------ LiteLLM hook


def _hook_namespace(monkeypatch, rules):
    from benchflow.providers.litellm_logging import callback_module_source

    monkeypatch.setenv(EGRESS_BLOCKED_URLS_ENV, json.dumps(rules))
    namespace: dict[str, object] = {}
    exec(callback_module_source(), namespace)
    return namespace


@pytest.mark.asyncio
async def test_pre_call_hook_injects_anthropic_blocked_domains(monkeypatch):
    ns = _hook_namespace(monkeypatch, ["arxiv.org/abs/2401.12345", "openreview.net"])
    logger = ns["proxy_handler_instance"]
    data = {
        "model": "claude-fable-5-1",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            {
                "type": "web_fetch_20250910",
                "name": "web_fetch",
                "blocked_domains": ["x.org"],
            },
            {"name": "bash", "input_schema": {"type": "object"}},
        ],
    }

    cleaned = await logger.async_pre_call_hook(None, None, data, "anthropic_messages")

    assert cleaned is not data
    search, fetch, bash = cleaned["tools"]
    assert search["blocked_domains"] == ["arxiv.org/abs/2401.12345", "openreview.net"]
    assert fetch["blocked_domains"] == [
        "x.org",
        "arxiv.org/abs/2401.12345",
        "openreview.net",
    ]
    assert bash == {"name": "bash", "input_schema": {"type": "object"}}
    # The original request is not mutated in place.
    assert "blocked_domains" not in data["tools"][0]


@pytest.mark.asyncio
async def test_pre_call_hook_narrows_anthropic_allowed_domains_instead_of_mixing(
    monkeypatch,
):
    ns = _hook_namespace(monkeypatch, ["openreview.net"])
    logger = ns["proxy_handler_instance"]
    data = {
        "model": "claude-fable-5-1",
        "messages": [],
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "allowed_domains": ["arxiv.org", "api.openreview.net"],
            }
        ],
    }
    cleaned = await logger.async_pre_call_hook(None, None, data, "anthropic_messages")
    tool = cleaned["tools"][0]
    assert tool["allowed_domains"] == ["arxiv.org"]
    assert "blocked_domains" not in tool


@pytest.mark.asyncio
async def test_pre_call_hook_strips_openai_hosted_search_under_blocklist(monkeypatch):
    ns = _hook_namespace(monkeypatch, ["openreview.net"])
    logger = ns["proxy_handler_instance"]
    data = {
        "model": "gpt-5.5",
        "input": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "web_search"},
            {"type": "function", "name": "shell", "parameters": {}},
        ],
    }
    cleaned = await logger.async_pre_call_hook(None, None, data, "aresponses")
    assert [t["type"] for t in cleaned["tools"]] == ["function"]


@pytest.mark.asyncio
async def test_pre_call_hook_still_drops_anthropic_server_tools_without_blocklist(
    monkeypatch,
):
    """Review fix: the blocklist exemption must not relax the pure no-web mode.

    Without an active blocklist the pre-existing filter drops every non-function
    tool — Anthropic ``web_search_*`` included — exactly as before this PR.
    """
    from benchflow.providers.litellm_logging import callback_module_source

    monkeypatch.delenv(EGRESS_BLOCKED_URLS_ENV, raising=False)
    monkeypatch.setenv("BENCHFLOW_DISALLOW_WEB_TOOLS", "1")
    ns: dict[str, object] = {}
    exec(callback_module_source(), ns)
    logger = ns["proxy_handler_instance"]
    data = {
        "model": "claude-fable-5-1",
        "messages": [],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search"},
            {"name": "bash", "input_schema": {"type": "object"}},
        ],
    }
    cleaned = await logger.async_pre_call_hook(None, None, data, "anthropic_messages")
    assert [t["name"] for t in cleaned["tools"]] == ["bash"]


# ------------------------------------------------------------------ rollout seams


def test_session_factory_agents_are_refused_under_blocklist():
    """Review fix: a host-side session-factory agent is outside the sandbox
    proxy and firewall, so a blocklist run must refuse it, not run open."""
    from benchflow.rollout._setup import _refuse_session_factory_under_blocklist

    b = EgressBlocklist(rules=("openreview.net",))
    _refuse_session_factory_under_blocklist("claude-agent-acp", None, b)
    _refuse_session_factory_under_blocklist("omnigent", "omnigent.acp:factory", None)
    with pytest.raises(RuntimeError, match="runs on the host"):
        _refuse_session_factory_under_blocklist("omnigent", "omnigent.acp:factory", b)


@pytest.mark.asyncio
async def test_install_phase_starts_proxy_after_web_policy(tmp_path, monkeypatch):
    """Review fix: the proxy is started in the generic install phase (before
    any connect protocol), right after the harness web-tool policy."""
    import benchflow.rollout as rollout_mod
    from benchflow.rollout import Rollout, RolloutConfig

    config = RolloutConfig.from_legacy(
        task_path=tmp_path / "task",
        agent="claude-agent-acp",
        prompts=[None],
        sandbox_user="agent",
    )
    trial = Rollout(config)
    trial._env = MagicMock()
    trial._env.exec = AsyncMock(return_value=MagicMock(stdout="/workspace\n"))
    trial._rollout_dir = tmp_path / "trial"
    trial._rollout_dir.mkdir()
    trial._rollout_paths = MagicMock()
    trial._task = MagicMock()
    trial._effective_locked = []
    trial._agent_cwd = "/app"
    trial._agent_env = {}
    trial._disallow_web_tools = False
    trial._egress_blocklist = EgressBlocklist(rules=("openreview.net",))
    order: list[str] = []

    async def _policy(*args, **kwargs):
        order.append("web-policy")

    async def _start(env, blocklist):
        assert blocklist == trial._egress_blocklist
        order.append("egress-proxy")

    monkeypatch.setattr(rollout_mod, "start_egress_proxy", _start)
    planes = trial._planes
    monkeypatch.setattr(planes, "install_agent", AsyncMock(return_value=MagicMock()))
    monkeypatch.setattr(planes, "setup_sandbox_user", AsyncMock(return_value="/app"))
    monkeypatch.setattr(planes, "write_credential_files", AsyncMock())
    monkeypatch.setattr(planes, "upload_subscription_auth", AsyncMock())
    monkeypatch.setattr(planes, "apply_web_tool_policy", _policy)
    monkeypatch.setattr(planes, "snapshot_build_config", AsyncMock())
    monkeypatch.setattr(planes, "seed_verifier_workspace", AsyncMock())
    monkeypatch.setattr(planes, "deploy_skills", AsyncMock())
    monkeypatch.setattr(planes, "lockdown_paths", AsyncMock())
    monkeypatch.setattr(planes, "link_skill_paths", AsyncMock())

    await trial.install_agent()

    assert order == ["web-policy", "egress-proxy"]


def test_container_policy_follows_task_even_when_primary_is_oracle():
    """Review P0 #3: an oracle primary is exempt from the blocklist itself, but
    the container must still be provisioned (NET_ADMIN, sandbox-local model
    proxy) for the role agents that connect_as() later."""
    from benchflow.rollout._setup import _resolve_agent_network_policy

    task = SimpleNamespace(
        config=TaskConfig.model_validate(
            {
                "sandbox": {
                    "network_mode": "blocklist",
                    "blocked_urls": ["openreview.net"],
                }
            }
        )
    )
    b = EgressBlocklist(rules=("openreview.net",))

    assert _resolve_agent_network_policy(
        task, primary_agent="claude-agent-acp", disallow_web_tools=False
    ) == (b, True)
    # Oracle primary: no proxy routing for the oracle, container still provisioned.
    assert _resolve_agent_network_policy(
        task, primary_agent="oracle", disallow_web_tools=False
    ) == (None, True)
    # A no-web run wins over the blocklist and is itself a container policy.
    assert _resolve_agent_network_policy(
        task, primary_agent="claude-agent-acp", disallow_web_tools=True
    ) == (None, True)
    # No policy declared anywhere.
    plain = SimpleNamespace(config=TaskConfig.model_validate({}))
    assert _resolve_agent_network_policy(
        plain, primary_agent="oracle", disallow_web_tools=False
    ) == (None, False)


def test_proxy_source_caps_concurrent_connections():
    """Review P1 #5: the accept loop is bounded by a connection semaphore."""
    source = egress_proxy_source()
    assert 'MAX_CONNECTIONS = int(CONFIG.get("max_connections", 256))' in source
    assert "_SLOTS = threading.BoundedSemaphore(MAX_CONNECTIONS)" in source
    assert "_SLOTS.acquire()" in source and "_SLOTS.release()" in source


# ------------------------------------------------------------------ batch preflight + resume


def _write_task(tasks_dir: Path, name: str, sandbox_toml: str = "") -> Path:
    task_dir = tasks_dir / name
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("do it\n")
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n[agent]\ntimeout_sec = 60\n'
        f"[environment]\n{sandbox_toml}"
    )
    return task_dir


def test_preflight_names_every_task_that_conflicts_with_block_url(tmp_path):
    """Review round 3: --block-url against a no-network/allowlist task must fail
    before ANY rollout starts, naming the offenders, not after 49 tasks ran."""
    from benchflow.evaluation import (
        EvaluationConfig,
        NetworkPolicyPreflightError,
        _expected_network_policies,
    )

    tasks = tmp_path / "tasks"
    open_task = _write_task(tasks, "task-open")
    closed = _write_task(tasks, "task-closed", 'network_mode = "no-network"\n')
    listed = _write_task(
        tasks, "task-allow", 'network_mode = "allowlist"\nallowed_hosts = ["x.org"]\n'
    )
    overlay = {
        "sandbox": {"network_mode": "blocklist", "blocked_urls": ["openreview.net"]}
    }
    cfg = EvaluationConfig(agent="claude-agent-acp", config_override=overlay)

    with pytest.raises(NetworkPolicyPreflightError) as info:
        _expected_network_policies([open_task, closed, listed], cfg)
    message = str(info.value)
    assert "2 task(s)" in message
    assert "task-closed" in message and "task-allow" in message
    assert "task-open" not in message

    # Devin review: a task that pins agent.network_mode would have swallowed
    # the run-level blocklist silently; preflight must name it instead.
    pinned = _write_task(tasks, "task-agent-public")
    (pinned / "task.toml").write_text(
        (pinned / "task.toml")
        .read_text()
        .replace(
            "[agent]\ntimeout_sec = 60\n",
            '[agent]\ntimeout_sec = 60\nnetwork_mode = "public"\n',
        )
    )
    with pytest.raises(NetworkPolicyPreflightError, match="task-agent-public"):
        _expected_network_policies([open_task, pinned], cfg)

    # Without the conflicting tasks the overlay resolves to a per-task policy.
    expected = _expected_network_policies([open_task], cfg)
    assert expected["task-open"]["mode"] == "blocklist"
    assert expected["task-open"]["blocked_urls"] == ["openreview.net"]
    # The oracle records no policy of its own (it is exempt), like the rollout.
    oracle = EvaluationConfig(agent="oracle", config_override=overlay)
    assert _expected_network_policies([open_task], oracle) == {"task-open": None}
    # A no-web run wins over the blocklist in Rollout, so the expectation must
    # be null too — otherwise resume would refuse a perfectly consistent job.
    no_web = EvaluationConfig(
        agent="claude-agent-acp", config_override=overlay, self_gen_no_internet=True
    )
    assert _expected_network_policies([open_task], no_web) == {"task-open": None}


def test_preflight_refuses_blocklist_task_on_non_enforcing_backend(tmp_path):
    from benchflow.evaluation import (
        EvaluationConfig,
        NetworkPolicyPreflightError,
        _expected_network_policies,
    )

    tasks = tmp_path / "tasks"
    task = _write_task(
        tasks,
        "task-bl",
        'network_mode = "blocklist"\nblocked_urls = ["openreview.net"]\n',
    )
    with pytest.raises(NetworkPolicyPreflightError, match="not enforced by modal"):
        _expected_network_policies([task], EvaluationConfig(environment="modal"))
    assert _expected_network_policies(
        [task], EvaluationConfig(environment="docker")
    ) == {"task-bl": EgressBlocklist(rules=("openreview.net",)).config_metadata()}


def test_resume_refuses_mixing_open_and_blocklisted_scores(tmp_path):
    """Review round 3: a job resumed with a different network policy must
    refuse, exactly like an agent mismatch."""
    from benchflow.evaluation import (
        EvaluationConfig,
        ResumeMismatchError,
        _check_resume_mismatch,
    )

    job_dir = tmp_path / "jobs" / "job"
    rollout = job_dir / "task-a__r1"
    rollout.mkdir(parents=True)
    policy = EgressBlocklist(rules=("openreview.net",)).config_metadata()
    (rollout / "config.json").write_text(
        json.dumps(
            {
                "agent": "claude-agent-acp",
                "task_path": "task-a",
                "network_policy": policy,
            }
        )
    )
    cfg = EvaluationConfig(agent="claude-agent-acp")

    # Same posture: fine. Open network on resume: refused. Unknown task: ignored.
    _check_resume_mismatch(job_dir, cfg, {"task-a": policy})
    with pytest.raises(ResumeMismatchError, match="network_policy"):
        _check_resume_mismatch(job_dir, cfg, {"task-a": None})
    _check_resume_mismatch(job_dir, cfg, {"task-other": None})

    # A pre-feature config.json (no key) counts as open network.
    (rollout / "config.json").write_text(
        json.dumps({"agent": "claude-agent-acp", "task_path": "task-a"})
    )
    _check_resume_mismatch(job_dir, cfg, {"task-a": None})
    with pytest.raises(ResumeMismatchError, match="different experiments"):
        _check_resume_mismatch(job_dir, cfg, {"task-a": policy})
    # A provenance-recorded task_path ("benchmarks/physics/task-a") still maps
    # to the expectation keyed by directory name (review micro-fix).
    (rollout / "config.json").write_text(
        json.dumps(
            {
                "agent": "claude-agent-acp",
                "task_path": "benchmarks/physics/task-a",
                "network_policy": policy,
            }
        )
    )
    _check_resume_mismatch(job_dir, cfg, {"task-a": policy})
    with pytest.raises(ResumeMismatchError, match="task-a"):
        _check_resume_mismatch(job_dir, cfg, {"task-a": None})
    # No expectations supplied: the legacy agent/loop-only behaviour is kept.
    _check_resume_mismatch(job_dir, cfg)


@pytest.mark.asyncio
async def test_disconnect_downloads_egress_log_whenever_proxy_ran(
    tmp_path, monkeypatch
):
    """Review #1: an oracle primary's env carries no blocklist marker, yet a
    role agent may have run under the blocklist via connect_as(); the audit
    log download is gated on the proxy having started, not on the env."""
    import benchflow.rollout as rollout_mod
    from benchflow.rollout import Rollout

    downloads: list[Path] = []

    async def _download(env, target_dir):
        downloads.append(target_dir)

    monkeypatch.setattr(rollout_mod, "download_egress_log", _download)
    trial = Rollout.__new__(Rollout)
    trial._is_session_factory = False
    trial._capture_partial_acp_trajectory = lambda: None
    trial._collect_native_acp_usage = None
    trial._acp_client = None
    trial._session = None
    trial._session_adapter = None
    trial._agent_launch = "agent-binary"
    trial._env = MagicMock()
    trial._env.exec = AsyncMock(return_value=MagicMock(return_code=0))
    trial._rollout_paths = SimpleNamespace(agent_dir=tmp_path / "agent")
    trial._agent_env = {}  # oracle primary: no blocklist marker here
    trial._active_role = None
    trial._phase = "connected"

    trial._egress_proxy_started = True
    await trial.disconnect()
    assert downloads == [tmp_path / "agent"]

    downloads.clear()
    trial._egress_proxy_started = False
    await trial.disconnect()
    assert downloads == []


def test_proxy_source_tries_ipv4_first_and_falls_back_across_candidates():
    """Review #2: docker's default bridge has no IPv6 route; a v6-first
    resolve must not turn into a 502 for dual-stack hosts."""
    source = egress_proxy_source()
    assert (
        "candidates.sort(key=lambda item: 0 if item[0] == socket.AF_INET else 1)"
        in source
    )
    assert "for family, sockaddr in resolve_upstream(host, port):" in source
    assert "last_err = exc" in source


def test_proxy_source_uses_ip_san_for_ip_literal_hosts():
    """Review #5: RFC 5280 requires IP: SANs for IP-literal names."""
    source = egress_proxy_source()
    assert 'san = "IP:%s" % host' in source
    assert 'san = "DNS:%s" % host' in source
    assert 'open(sys.argv[1], encoding="utf-8")' in source


@pytest.mark.asyncio
async def test_connect_acp_applies_firewall_before_launch_under_blocklist(tmp_path):
    """Devin review: with the proxy already up, the UID firewall must precede
    the agent process so startup traffic cannot escape before the handshake."""
    from unittest.mock import patch

    from benchflow.acp.client import ACPClient
    from benchflow.acp.runtime import connect_acp

    events: list[str] = []
    mock_session = MagicMock(session_id="s1")
    mock_init = MagicMock(agent_info=None)
    mock_acp = AsyncMock(spec=ACPClient)
    mock_acp.connect = AsyncMock(side_effect=lambda: events.append("connect"))
    mock_acp.initialize = AsyncMock(return_value=mock_init)
    mock_acp.session_new = AsyncMock(
        side_effect=lambda *a, **k: events.append("session_new") or mock_session
    )
    mock_acp.set_config_option = AsyncMock()
    mock_acp.close = AsyncMock()

    async def enforce(*args, **kwargs):
        events.append("firewall")

    async def verify(*args, **kwargs):
        events.append("verify")

    agent_env = apply_blocklist_env(
        {"BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:4000/v1"},
        EgressBlocklist(rules=("openreview.net",)),
    )
    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        patch(
            "benchflow.acp.runtime.enforce_agent_egress_firewall",
            new_callable=AsyncMock,
            side_effect=enforce,
        ) as mock_firewall,
        patch(
            "benchflow.acp.runtime.verify_egress_blocklist",
            new_callable=AsyncMock,
            side_effect=verify,
        ),
    ):
        await connect_acp(
            env=AsyncMock(),
            agent="openhands",
            agent_launch="openhands acp",
            agent_env=agent_env,
            sandbox_user="agent",
            model=None,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )

    assert events == ["firewall", "connect", "session_new", "verify"]
    mock_firewall.assert_awaited_once()


# ------------------------------------------------------------------ CLI / docker / config


def test_blocklist_override_folds_urls_into_c_axis_overlay(tmp_path):
    from benchflow._utils.config_override import blocklist_override

    assert blocklist_override(None, None, None) is None
    assert blocklist_override('{"agent":{"timeout_sec":5}}', [], None) == (
        '{"agent":{"timeout_sec":5}}'
    )
    listing = tmp_path / "hidden.txt"
    listing.write_text(
        "# hidden papers\nhttps://arxiv.org/abs/2401.12345  # v1\n\nopenreview.net\n"
    )

    merged = json.loads(
        blocklist_override(
            '{"agent":{"timeout_sec":5}}',
            ["openreview.net", " arxiv.org/pdf/2401.12345 "],
            listing,
        )
    )
    assert merged["agent"] == {"timeout_sec": 5}
    assert merged["sandbox"]["network_mode"] == "blocklist"
    assert merged["sandbox"]["blocked_urls"] == [
        "openreview.net",
        "arxiv.org/pdf/2401.12345",
        "https://arxiv.org/abs/2401.12345",
    ]
    # The overlay re-validates through the task schema at rollout time.
    cfg = TaskConfig.model_validate({"sandbox": merged["sandbox"]})
    assert cfg.sandbox.blocked_urls[-1] == "arxiv.org/abs/2401.12345"


def test_docker_stacks_net_admin_overlay_only_under_agent_network_policy(tmp_path):
    from benchflow.sandbox._compose import COMPOSE_NET_ADMIN_PATH
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.config import SandboxConfig

    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM scratch\n")

    def _paths(policy: bool):
        sandbox = DockerSandbox(
            environment_dir=env_dir,
            environment_name="t",
            session_id="s",
            rollout_paths=None,
            task_env_config=SandboxConfig(),
            agent_network_policy=policy,
        )
        return sandbox._docker_compose_paths

    assert COMPOSE_NET_ADMIN_PATH in _paths(True)
    assert COMPOSE_NET_ADMIN_PATH not in _paths(False)
    assert "NET_ADMIN" in COMPOSE_NET_ADMIN_PATH.read_text()


def test_write_config_records_network_policy(tmp_path):
    from benchflow.rollout._results import _write_config
    from benchflow.skill_policy import resolve_task_skill_policy

    b = EgressBlocklist(rules=("arxiv.org/abs/2401.12345",))
    task = tmp_path / "task"
    task.mkdir()
    _write_config(
        tmp_path,
        task_path=task,
        agent="claude-agent-acp",
        model="m",
        environment="docker",
        skill_policy=resolve_task_skill_policy(
            task_path=task,
            skill_mode="no-skill",
            runtime_skills_dir=None,
            declared_sandbox_skills_dir=None,
        ),
        sandbox_user="agent",
        context_root=None,
        timeout=60,
        started_at=datetime(2026, 1, 1),
        agent_env={},
        network_policy=b.config_metadata(),
    )
    recorded = json.loads((tmp_path / "config.json").read_text())["network_policy"]
    assert recorded["mode"] == "blocklist"
    assert recorded["blocked_urls"] == ["arxiv.org/abs/2401.12345"]
    assert recorded["tls_inspection_hosts"] == ["arxiv.org"]
    assert recorded["blocked_status"] == BLOCKED_STATUS


def test_proxy_source_defaults_to_refusing_private_networks_outside_local_subnets():
    """Devin review: RFC1918 is refused by default; only the container's own
    on-link subnets (compose network) are allowed, plus an explicit opt-in."""
    source = egress_proxy_source()
    assert (
        'ALLOW_PRIVATE_NETWORKS = bool(CONFIG.get("allow_private_networks", False))'
        in source
    )
    assert "_local_ipv4_subnets" in source and "/proc/net/route" in source
    assert "any(ip in net for net in _LOCAL_SUBNETS)" in source
    assert 'header(headers, "expect") or "").lower() == "100-continue"' in source


def test_local_subnet_parser_reads_on_link_routes_only():
    """Exercise the embedded route parser on a captured /proc/net/route."""
    ns: dict = {}
    src = egress_proxy_source()
    start = src.index("def _local_ipv4_subnets(")
    end = src.index("\n_LOCAL_SUBNETS = ")
    exec("import ipaddress\n" + src[start:end], ns)
    table = (
        "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
        "eth0\t00000000\t010011AC\t0003\t0\t0\t0\t00000000\t0\t0\t0\n"  # default via gw
        "eth0\t000011AC\t00000000\t0001\t0\t0\t0\t0000FFFF\t0\t0\t0\n"  # 172.17.0.0/16 on-link
        "lo\t0000007F\t00000000\t0001\t0\t0\t0\t000000FF\t0\t0\t0\n"  # 127.0.0.0/8
    )
    nets = ns["_local_ipv4_subnets"](table)
    assert [str(n) for n in nets] == ["172.17.0.0/16"]


# ------------------------------------------------------------------ real proxy (Linux)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Origin(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = f"origin:{self.path}".encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        received = self.rfile.read(length)
        body = f"origin:{self.path}:{len(received)}".encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        pass


def _serve(handler, port, tls_cert=None, tls_key=None):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    if tls_cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(tls_cert, tls_key)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _openssl_selfsigned(
    directory: Path, name: str, subj: str, *ext: str
) -> tuple[Path, Path]:
    key, crt = directory / f"{name}.key", directory / f"{name}.crt"
    cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(key),
        "-out",
        str(crt),
        "-days",
        "2",
        "-subj",
        subj,
    ]
    for e in ext:
        cmd += ["-addext", e]
    subprocess.run(cmd, check=True, capture_output=True)
    return crt, key


def _wait_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"port {port} never opened")


@pytest.fixture
def running_proxy(tmp_path):
    """Start the real proxy script with a CA, a plain origin, and a TLS origin."""
    ca_dir = tmp_path / "ca"
    ca_dir.mkdir()
    _openssl_selfsigned(
        ca_dir,
        "ca",
        "/CN=BenchFlow Test CA",
        "basicConstraints=critical,CA:TRUE",
        "keyUsage=critical,keyCertSign,cRLSign",
    )
    srv_crt, srv_key = _openssl_selfsigned(
        tmp_path, "origin", "/CN=localhost", "subjectAltName=DNS:localhost,IP:127.0.0.1"
    )
    http_port, tls_port, proxy_port = _free_port(), _free_port(), _free_port()
    origin = _serve(_Origin, http_port)
    tls_origin = _serve(_Origin, tls_port, srv_crt, srv_key)

    script = tmp_path / "egress_proxy.py"
    script.write_text(egress_proxy_source())
    log_path = tmp_path / "egress.jsonl"
    config = tmp_path / "policy.json"
    config.write_text(
        json.dumps(
            {
                "rules": ["blocked.example", "localhost/secret"],
                "port": proxy_port,
                "log_path": str(log_path),
                "blocked_status": BLOCKED_STATUS,
                "ca_dir": str(ca_dir),
                "upstream_ca_file": str(srv_crt),
                # Test origins live on 127.0.0.1; production refuses loopback.
                "allow_loopback_upstream": True,
                # Small cap: every request in these tests must release its slot.
                "max_connections": 3,
            }
        )
    )
    proc = subprocess.Popen(
        [sys.executable, str(script), str(config)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_port(proxy_port)
        yield SimpleNamespace(
            proxy_port=proxy_port,
            http_port=http_port,
            tls_port=tls_port,
            ca_crt=ca_dir / "ca.crt",
            origin_crt=srv_crt,
            log_path=log_path,
        )
    finally:
        proc.terminate()
        try:
            stderr = proc.communicate(timeout=5)[1].decode()
        except subprocess.TimeoutExpired:
            proc.kill()
            stderr = ""
        origin.shutdown()
        tls_origin.shutdown()
        if stderr.strip():
            print("proxy stderr:", stderr)


def _via_proxy(p, url: str, *, trust: Path | None = None) -> tuple[int, bytes]:
    proxy = f"http://127.0.0.1:{p.proxy_port}"
    handlers: list = [urllib.request.ProxyHandler({"http": proxy, "https": proxy})]
    if trust is not None:
        ctx = ssl.create_default_context(cafile=str(trust))
        handlers.append(urllib.request.HTTPSHandler(context=ctx))
    opener = urllib.request.build_opener(*handlers)
    try:
        with opener.open(url, timeout=15) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


@NEEDS_OPENSSL
def test_proxy_filters_plain_http_by_host_and_path(running_proxy):
    p = running_proxy
    assert _via_proxy(p, f"http://127.0.0.1:{p.http_port}/ok") == (200, b"origin:/ok")
    assert _via_proxy(p, f"http://localhost:{p.http_port}/ok")[0] == 200
    # Path rule on localhost: /secret* hidden, everything else served.
    assert (
        _via_proxy(p, f"http://localhost:{p.http_port}/secret/paper.pdf")[0]
        == BLOCKED_STATUS
    )
    # Host rule: no DNS lookup is even attempted for a blocked host.
    status, body = _via_proxy(p, "http://blocked.example/anything")
    assert status == BLOCKED_STATUS
    assert body == b"Not Found\n"


@NEEDS_OPENSSL
def test_proxy_rejects_connect_to_blocked_host(running_proxy):
    p = running_proxy
    with socket.create_connection(("127.0.0.1", p.proxy_port), timeout=5) as s:
        s.sendall(
            b"CONNECT sub.blocked.example:443 HTTP/1.1\r\nHost: sub.blocked.example:443\r\n\r\n"
        )
        reply = s.recv(4096)
    assert reply.startswith(f"HTTP/1.1 {BLOCKED_STATUS} ".encode())


@NEEDS_OPENSSL
def test_proxy_tunnels_https_without_inspection_for_hosts_without_path_rules(
    running_proxy,
):
    p = running_proxy
    # 127.0.0.1 carries no path rule => opaque tunnel; the client sees the
    # ORIGIN certificate (trusting it directly), not a proxy-minted one, and
    # a /secret path is NOT filtered because the rule is bound to "localhost".
    status, body = _via_proxy(
        p, f"https://127.0.0.1:{p.tls_port}/secret", trust=p.origin_crt
    )
    assert (status, body) == (200, b"origin:/secret")


@NEEDS_OPENSSL
def test_proxy_inspects_tls_for_hosts_with_path_rules(running_proxy):
    p = running_proxy
    # localhost carries a path rule => TLS is terminated with a leaf signed by
    # the run CA; the client trusts that CA (as the sandbox bundle would).
    assert _via_proxy(p, f"https://localhost:{p.tls_port}/ok", trust=p.ca_crt) == (
        200,
        b"origin:/ok",
    )
    status, body = _via_proxy(
        p, f"https://localhost:{p.tls_port}/secret/x", trust=p.ca_crt
    )
    assert (status, body) == (BLOCKED_STATUS, b"Not Found\n")
    # Without the CA the inspected host must fail verification — proof the
    # tunnel really was terminated rather than passed through.
    with pytest.raises(urllib.error.URLError):
        _via_proxy(p, f"https://localhost:{p.tls_port}/ok", trust=p.origin_crt)

    events = [json.loads(line) for line in p.log_path.read_text().splitlines()]
    blocked = [e for e in events if e["event"] == "block"]
    assert {(e["host"], e.get("rule")) for e in blocked} >= {
        ("localhost", "localhost/secret")
    }
    assert all("secret" not in json.dumps(e) or e["event"] == "block" for e in events)


@NEEDS_OPENSSL
def test_proxy_refuses_link_local_and_loopback_upstreams(tmp_path):
    """Review fix: the root-run proxy must not be an SSRF hop. Link-local
    (cloud metadata) is always refused; loopback is refused unless a test
    explicitly allows it."""
    port = _free_port()
    origin_port = _free_port()
    origin = _serve(_Origin, origin_port)
    script = tmp_path / "egress_proxy.py"
    script.write_text(egress_proxy_source())
    log_path = tmp_path / "egress.jsonl"
    config = tmp_path / "policy.json"
    config.write_text(
        json.dumps(
            {
                "rules": ["blocked.example"],
                "port": port,
                "log_path": str(log_path),
                "blocked_status": BLOCKED_STATUS,
                "ca_dir": None,
            }
        )
    )
    proc = subprocess.Popen(
        [sys.executable, str(script), str(config)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_port(port)
        p = SimpleNamespace(proxy_port=port)
        status, body = _via_proxy(p, "http://169.254.169.254/latest/meta-data/")
        assert status == 403
        assert b"link-local" in body
        status, body = _via_proxy(p, f"http://127.0.0.1:{origin_port}/ok")
        assert status == 403
        assert b"loopback" in body
        with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
            s.sendall(b"CONNECT 169.254.169.254:443 HTTP/1.1\r\n\r\n")
            assert s.recv(4096).startswith(b"HTTP/1.1 403 ")
        events = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert {e["event"] for e in events} >= {"refuse"}
        assert all(e["event"] != "allow" for e in events)
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        origin.shutdown()


def test_proxy_source_pins_http11_alpn_and_vets_upstream_addresses():
    """Review fixes #2/#3: ALPN is limited to HTTP/1.1 on both TLS legs and
    every resolved upstream address is vetted before connecting."""
    source = egress_proxy_source()
    assert source.count('set_alpn_protocols(["http/1.1"])') == 2
    assert "ipaddress.ip_address(sockaddr[0])" in source
    assert "is_link_local" in source and "is_loopback" in source


@NEEDS_OPENSSL
def test_proxy_reuses_one_leaf_key_across_inspected_hosts(running_proxy):
    """Review fix #5: leaf certs share one key; only the x509 signing is per host."""
    p = running_proxy
    assert _via_proxy(p, f"https://localhost:{p.tls_port}/ok", trust=p.ca_crt)[0] == 200
    leaf_dir = p.ca_crt.parent / "leaf"
    assert (leaf_dir / "leaf.key").exists()
    assert (leaf_dir / "localhost.crt").exists()
    assert not (leaf_dir / "localhost.key").exists()


@NEEDS_OPENSSL
def test_proxy_blocks_encoded_and_dotted_paths_and_host_header_spoofs(running_proxy):
    """Review round 5 (WAF bypass): the live proxy must hide ``localhost/secret``
    through percent-encoding, slash runs, dot segments, and an IP-literal URL
    that smuggles the real host in the Host header."""
    p = running_proxy
    for path in (
        "/%73ecret/paper.pdf",
        "//secret///paper.pdf",
        "/other/../secret/paper.pdf",
        "/SECRET/x",
    ):
        status, _ = _via_proxy(p, f"http://localhost:{p.http_port}{path}")
        assert status == BLOCKED_STATUS, path
    # Control: a neighbouring path is still served.
    assert _via_proxy(p, f"http://localhost:{p.http_port}/secre/t")[0] == 200

    # Host-header spoof: URL names the IP (no rule), header names localhost.
    with socket.create_connection(("127.0.0.1", p.proxy_port), timeout=5) as sock:
        sock.sendall(
            f"GET http://127.0.0.1:{p.http_port}/secret/paper.pdf HTTP/1.1\r\n"
            "Host: localhost\r\nConnection: close\r\n\r\n".encode()
        )
        reply = sock.recv(4096)
    assert reply.startswith(f"HTTP/1.1 {BLOCKED_STATUS} ".encode())
    # Same request with an honest Host header is not affected by the rule.
    with socket.create_connection(("127.0.0.1", p.proxy_port), timeout=5) as sock:
        sock.sendall(
            f"GET http://127.0.0.1:{p.http_port}/secret/paper.pdf HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{p.http_port}\r\nConnection: close\r\n\r\n".encode()
        )
        reply = sock.recv(4096)
    # The stdlib test origin answers HTTP/1.0; only the status matters here.
    assert reply.split(b" ", 2)[1] == b"200"


@NEEDS_OPENSSL
def test_proxy_inspected_tls_also_normalizes_paths(running_proxy):
    p = running_proxy
    for path in ("/%73ecret/x", "/a/../secret/x", "//secret//x"):
        status, _ = _via_proxy(
            p, f"https://localhost:{p.tls_port}{path}", trust=p.ca_crt
        )
        assert status == BLOCKED_STATUS, path
    assert _via_proxy(p, f"https://localhost:{p.tls_port}/ok", trust=p.ca_crt)[0] == 200


@NEEDS_OPENSSL
def test_inspected_tls_negotiates_http11_with_an_h2_capable_client(running_proxy):
    """Functional ALPN check: a client that offers h2 first must be negotiated
    down to HTTP/1.1 inside the inspected tunnel, and a plain HTTP/1.1 request
    over that session must succeed. Guards the review fix for h2 clients."""
    p = running_proxy
    ctx = ssl.create_default_context(cafile=str(p.ca_crt))
    ctx.set_alpn_protocols(["h2", "http/1.1"])
    with socket.create_connection(("127.0.0.1", p.proxy_port), timeout=10) as raw:
        raw.sendall(
            f"CONNECT localhost:{p.tls_port} HTTP/1.1\r\n"
            f"Host: localhost:{p.tls_port}\r\n\r\n".encode()
        )
        assert raw.recv(4096).startswith(b"HTTP/1.1 200 ")
        with ctx.wrap_socket(raw, server_hostname="localhost") as tls:
            assert tls.selected_alpn_protocol() == "http/1.1"
            tls.sendall(
                b"GET /ok HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
            )
            reply = b""
            while True:
                chunk = tls.recv(65536)
                if not chunk:
                    break
                reply += chunk
    assert reply.split(b" ", 2)[1] == b"200"
    assert reply.endswith(b"origin:/ok")


@NEEDS_OPENSSL
def test_proxy_refuses_private_addresses_outside_local_subnets(tmp_path):
    """Devin review: a private address that is not on one of the container's
    own subnets is refused without any connect attempt."""
    port = _free_port()
    script = tmp_path / "egress_proxy.py"
    script.write_text(egress_proxy_source())
    config = tmp_path / "policy.json"
    config.write_text(
        json.dumps(
            {
                "rules": ["blocked.example"],
                "port": port,
                "log_path": str(tmp_path / "egress.jsonl"),
                "blocked_status": BLOCKED_STATUS,
                "ca_dir": None,
                # TEST-NET-3 style private target that no lab subnet uses.
                "allow_local_subnets": True,
            }
        )
    )
    proc = subprocess.Popen(
        [sys.executable, str(script), str(config)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_port(port)
        p = SimpleNamespace(proxy_port=port)
        status, body = _via_proxy(p, "http://10.255.255.254/")
        assert status == 403 and b"private" in body
        status, body = _via_proxy(p, "http://192.168.255.254/")
        assert status == 403 and b"private" in body
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@NEEDS_OPENSSL
def test_proxy_answers_expect_100_continue_and_relays_the_body(running_proxy):
    """Devin review: a client that waits for 100 Continue must get it from the
    proxy (with Expect stripped upstream) and then have its body relayed."""
    p = running_proxy
    body = b"x" * 4096
    with socket.create_connection(("127.0.0.1", p.proxy_port), timeout=10) as sock:
        sock.sendall(
            f"POST http://localhost:{p.http_port}/upload HTTP/1.1\r\n"
            f"Host: localhost:{p.http_port}\r\nContent-Length: {len(body)}\r\n"
            "Expect: 100-continue\r\nConnection: close\r\n\r\n".encode()
        )
        interim = sock.recv(4096)
        assert interim.startswith(b"HTTP/1.1 100 Continue")
        sock.sendall(body)
        reply = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            reply += chunk
    assert reply.split(b" ", 2)[1] == b"200"
    assert reply.endswith(b"origin:/upload:4096")
