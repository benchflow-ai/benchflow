from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.agents import registry
from benchflow.providers.litellm_runtime import ensure_litellm_runtime
from benchflow.rollout import Role, Rollout, RolloutConfig
from benchflow.rollout_planes import DefaultRolloutPlanes
from benchflow.sandbox import _egress_denylist_proxy as proxy
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._egress_denylist_proxy import Policy
from benchflow.sandbox.native_oauth import (
    native_oauth_egress_policy,
    validate_native_oauth_transport,
)

# Reuse the existing actual TLS proxy fixture; no public network involved.
from tests.test_egress_denylist import stack as _tls_stack

stack = _tls_stack

# In-sandbox metadata: ACP version, the SDK it pins, installed SDK, the Claude
# Code release that SDK bundles, and the native binary's own --version.
VERSIONS = {
    "acp": "0.73.0",
    "acp_sdk": "0.3.257",
    "sdk": "0.3.257",
    "sdk_native": "2.1.257",
    "native": "2.1.257 (Claude Code)",
}
LAUNCHER = "/opt/benchflow/bin/claude-agent-acp"
TOKEN_ENV = {"CLAUDE_CODE_OAUTH_TOKEN": "fake-oauth"}


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages?beta=true"])
def test_exact_native_model_request(path):
    p = Policy([], [], native_claude_model_only=True)
    assert p.host_rule("api.anthropic.com", 443) is None
    assert p.request_rule("api.anthropic.com", 443, path, "POST", True) is None


@pytest.mark.parametrize(
    "host,port,path,method,secure",
    [
        ("api.anthropic.com", 80, "/v1/messages", "POST", False),
        ("api.anthropic.com", 444, "/v1/messages", "POST", True),
        ("www.api.anthropic.com", 443, "/v1/messages", "POST", True),
        ("elsewhere.example", 443, "/v1/messages", "POST", True),
        ("api.anthropic.com", 443, "/v1/messages", "GET", True),
        ("api.anthropic.com", 443, "/v1/messages", "OPTIONS", True),
        ("api.anthropic.com", 443, "/v1/messages?beta=true&other=1", "POST", True),
        ("api.anthropic.com", 443, "/v1/messages/", "POST", True),
        ("api.anthropic.com", 443, "/v1/%6dessages", "POST", True),
        ("api.anthropic.com", 443, "/x/../v1/messages", "POST", True),
        ("api.anthropic.com", 443, "/v1/messages/count_tokens", "POST", True),
        ("api.anthropic.com", 443, "/api/hello", "HEAD", True),
        ("api.anthropic.com", 443, "/api/claude_code/settings", "GET", True),
    ],
)
def test_native_model_gate_rejects_other_requests(host, port, path, method, secure):
    assert Policy([], [], native_claude_model_only=True).request_rule(
        host, port, path, method, secure
    )


def test_no_native_opaque_gateway_or_arbitrary_policy_value():
    with pytest.raises(ValueError):
        Policy([], [], 12345, True)
    with pytest.raises(ValueError):
        Policy([], [], native_claude_model_only="true")
    assert Policy([], [], native_claude_model_only=True).host_rule("127.0.0.1", 12345)


def test_policy_selection_preserves_api_key_and_open_native_paths():
    assert (
        native_oauth_egress_policy(
            "claude-agent-acp", "claude-sonnet-4-6", TOKEN_ENV, no_web=False
        )
        is None
    )
    assert (
        native_oauth_egress_policy(
            "claude-agent-acp",
            "claude-sonnet-4-6",
            {"ANTHROPIC_API_KEY": "key"},
            no_web=True,
        )
        is None
    )
    assert native_oauth_egress_policy(
        "claude-agent-acp", "claude-sonnet-4-6", TOKEN_ENV, no_web=True
    ).native_claude_model_only
    with pytest.raises(ValueError, match="canonical"):
        native_oauth_egress_policy(
            "claude-agent-acp",
            "claude-sonnet-4-6",
            {**TOKEN_ENV, "ANTHROPIC_BASE_URL": "https://other.example"},
            no_web=True,
        )
    with pytest.raises(ValueError, match="only"):
        native_oauth_egress_policy(
            "codex-acp", "gpt-5", {"CODEX_AUTH_JSON": "{}"}, no_web=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("uid", ["0", "not-a-uid", ""])
async def test_actual_uid_is_checked_even_for_nonroot_name(uid):
    env = SimpleNamespace(
        exec=AsyncMock(return_value=ExecResult(stdout=uid, return_code=0))
    )
    with pytest.raises(ValueError, match="nonzero"):
        await validate_native_oauth_transport(
            env, "agent-alias", "/opt/benchflow/bin/claude-agent-acp"
        )
    assert env.exec.await_count == 1


def _client_env(versions):
    async def execute(command, **kwargs):
        if command.startswith("id -u"):
            return ExecResult(stdout="1500\n", return_code=0)
        return ExecResult(stdout=json.dumps(versions), return_code=0)

    return SimpleNamespace(exec=execute)


@pytest.mark.asyncio
async def test_native_client_gate_follows_registry_claude_pin(monkeypatch):
    """Guards this fix against the duplicated pins from the native Claude OAuth reviewer transport.

    The no-web transport hard-coded ACP 0.73.0 / SDK 0.3.257 / Claude 2.1.257
    beside the registry pin, so PR #1139's bump to claude-agent-acp 0.81.0 left
    every test green and refused every native Claude no-web run.
    """
    monkeypatch.setattr(
        registry,
        "_CLAUDE_AGENT_ACP_PACKAGE",
        "@agentclientprotocol/claude-agent-acp@0.81.0",
    )
    bumped = {
        "acp": "0.81.0",
        "acp_sdk": "0.3.280",
        "sdk": "0.3.280",
        "sdk_native": "2.1.280",
        "native": "2.1.280 (Claude Code)",
    }
    admission = await validate_native_oauth_transport(
        _client_env(bumped), "agent", LAUNCHER
    )
    assert admission["versions"] == {
        "acp": "0.81.0",
        "sdk": "0.3.280",
        "native": "2.1.280 (Claude Code)",
    }
    with pytest.raises(ValueError, match="not verified"):
        await validate_native_oauth_transport(_client_env(VERSIONS), "agent", LAUNCHER)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"sdk": "0.3.258"},
        {"acp_sdk": "^0.3.257"},
        {"native": "2.1.258 (Claude Code)"},
        {"sdk_native": None},
        {"acp": None},
    ],
)
async def test_native_client_must_match_what_the_pinned_acp_installs(change):
    """Guards this fix for the native Claude OAuth reviewer transport: SDK and native versions are derived from the
    pinned ACP's exact SDK dependency and that SDK's bundled Claude Code release."""
    with pytest.raises(ValueError, match="not verified"):
        await validate_native_oauth_transport(
            _client_env({**VERSIONS, **change}), "agent", LAUNCHER
        )


def make_rollout(tmp_path):
    r = Rollout(
        RolloutConfig(
            task_path=tmp_path,
            agent="claude-agent-acp",
            model="claude-sonnet-4-6",
            environment="docker",
            sandbox_user="agent",
        )
    )
    r._rollout_dir = tmp_path
    r._rollout_name = "run"
    r._agent_env = dict(TOKEN_ENV)
    r._agent_launch = "/opt/benchflow/bin/claude-agent-acp"
    r._disallow_web_tools = True
    r._reapply_ask_user_handler = lambda: None
    r._attach_trajectory_writer = lambda _: None
    calls = []

    async def execute(command, **kwargs):
        if command.startswith("id -u"):
            calls.append("uid")
            return ExecResult(stdout="1500\n", return_code=0)
        if "/opt/benchflow/node/bin/node -e " in command:
            calls.append("version")
            return ExecResult(stdout=json.dumps(VERSIONS), return_code=0)
        calls.append("firewall")
        return ExecResult(stdout="", return_code=0)

    r._env = SimpleNamespace(exec=execute)

    async def runtime(**kwargs):
        # Actual provider entry point must retain the native skip without pretending to be a gateway.
        calls.append("runtime")
        assert kwargs["force_sandbox_local"] is True
        return await ensure_litellm_runtime(**kwargs)

    async def start(*args, **kwargs):
        async def start_proxy(*args, **kwargs):
            calls.append("proxy")

        with patch("benchflow.rollout_planes.start_egress_denylist", start_proxy):
            await DefaultRolloutPlanes().start_egress_denylist(*args, **kwargs)

    async def stop(*args, **kwargs):
        calls.append("stop")

    async def connect(**kwargs):
        calls.append("acp")
        assert calls.index("firewall") < len(calls) - 1
        assert kwargs["agent_env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "fake-oauth"
        assert "BENCHFLOW_PROVIDER_BASE_URL" not in kwargs["agent_env"]
        assert kwargs["agent_env"]["HTTPS_PROXY"] == "http://127.0.0.1:18628"
        return AsyncMock(), AsyncMock(), AsyncMock(), "claude-agent-acp"

    r._planes = SimpleNamespace(
        ensure_litellm_runtime=runtime,
        start_egress_denylist=start,
        stop_egress_denylist=stop,
        connect_acp=connect,
    )
    return r, calls


@pytest.mark.asyncio
async def test_real_native_runtime_skip_gets_truthful_transport_before_acp(tmp_path):
    r, calls = make_rollout(tmp_path)
    await r.connect()
    assert calls == ["runtime", "uid", "version", "proxy", "firewall", "acp"]
    assert (
        json.loads((tmp_path / "native-oauth-network.json").read_text())["admitted"]
        is True
    )
    assert getattr(r, "_egress_denylist", None) is None
    await r.connect()
    assert calls[-7:] == [
        "runtime",
        "uid",
        "version",
        "stop",
        "proxy",
        "firewall",
        "acp",
    ]
    await r._stop_active_egress()
    assert r._active_egress_policy is None


@pytest.mark.asyncio
async def test_failed_firewall_never_publishes_admission_and_remains_cleanup_eligible(
    tmp_path,
):
    r, calls = make_rollout(tmp_path)
    real_exec = r._env.exec

    async def failing_exec(command, **kwargs):
        if command.startswith("id -u") or "/opt/benchflow/node/bin/node -e " in command:
            return await real_exec(command, **kwargs)
        return ExecResult(stdout="", return_code=1)

    r._env.exec = failing_exec
    (tmp_path / "native-oauth-network.json").write_text('{"admitted":true}')
    with pytest.raises(RuntimeError, match="firewall"):
        await r.connect()
    assert not (tmp_path / "native-oauth-network.json").exists()
    assert "acp" not in calls
    assert r._active_egress_policy is not None
    await r._stop_active_egress()
    assert calls[-1] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key",
    [
        "NODE_OPTIONS",
        "CLAUDE_CODE_EXECUTABLE",
        "CLAUDE_CODE_CUSTOM_OAUTH_URL",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_MANTLE",
        "CLAUDE_CODE_USE_ANTHROPIC_AWS",
        "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
    ],
)
async def test_failed_reconnect_clears_previous_admission(tmp_path, key):
    r, calls = make_rollout(tmp_path)
    await r.connect()
    r._agent_env[key] = "unverified"
    before = len(calls)
    with pytest.raises(ValueError, match="conflicts"):
        await r.connect()
    assert len(calls) == before
    assert not (tmp_path / "native-oauth-network.json").exists()


@pytest.mark.asyncio
async def test_native_to_api_key_to_native_role_switch(tmp_path):
    r, calls = make_rollout(tmp_path)
    await r.connect()
    r._config.agent_env = dict(TOKEN_ENV)
    r._planes.agent_launch = lambda *args, **kwargs: r._agent_launch
    r._planes.resolve_agent_env = lambda agent, model, env: dict(env)
    r._planes.write_credential_files = AsyncMock()
    r._planes.apply_web_tool_policy = AsyncMock()

    async def runtime(**kwargs):
        return kwargs["agent_env"], None

    async def acp(**kwargs):
        calls.append("role-acp")
        return AsyncMock(), AsyncMock(), AsyncMock(), "claude-agent-acp"

    r._planes.ensure_litellm_runtime = runtime
    r._planes.connect_acp = acp
    await r.connect_as(
        Role(
            name="api",
            agent="claude-agent-acp",
            model="claude-sonnet-4-6",
            env={"ANTHROPIC_API_KEY": "fake-key"},
        )
    )
    assert r._active_egress_policy is None
    assert not (tmp_path / "native-oauth-network.json").exists()
    assert calls[-2:] == ["stop", "role-acp"]
    await r.connect_as(
        Role(name="native", agent="claude-agent-acp", model="claude-sonnet-4-6")
    )
    assert r._active_egress_policy.native_claude_model_only
    assert calls[-5:] == ["uid", "version", "proxy", "firewall", "role-acp"]
    assert json.loads((tmp_path / "native-oauth-network.json").read_text())["admitted"]


def _delayed_body():
    time.sleep(0.05)
    yield b"{}"


@pytest.mark.parametrize("body", [b"{}", iter([b"{", b"}"]), _delayed_body()])
def test_denied_tls_post_body_returns_explicit_403(stack, monkeypatch, body):
    def forbidden_upstream(*args, **kwargs):
        raise AssertionError("denied POST must never open an upstream")

    monkeypatch.setattr(proxy, "_connect_upstream", forbidden_upstream)
    request = urllib.request.Request(
        "https://paper.test/abs/2401.12345", data=body, method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        stack.opener.open(request, timeout=3)
    assert error.value.code == 403
    assert error.value.headers["X-BenchFlow-Blocked"] == "1"


def test_denied_body_drain_has_total_byte_cap_and_restores_timeout():
    class Input:
        timeout = 7
        received = 0

        def gettimeout(self):
            return self.timeout

        def settimeout(self, value):
            self.timeout = value

        def recv(self, size):
            self.received += size
            return b"x" * size

    client = Input()
    proxy._drain_denied_body(client, b"buffered", 10**9, False)
    assert client.received + len(b"buffered") == 65536
    assert client.timeout == 7


def test_denied_body_drain_uses_one_deadline_for_slow_sender(monkeypatch):
    times = iter([0.0, 0.3, 0.6])
    monkeypatch.setattr(proxy.time, "monotonic", lambda: next(times))

    class Input:
        timeout = 7
        reads = 0

        def gettimeout(self):
            return self.timeout

        def settimeout(self, value):
            self.timeout = value

        def recv(self, size):
            self.reads += 1
            return b"x"

    client = Input()
    proxy._drain_denied_body(client, b"", 100, False)
    assert client.reads == 1
    assert client.timeout == 7
