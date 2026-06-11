"""Regression coverage for the LiteLLM-runtime hardening pass.

Covers the cleanup/robustness/security/routing fixes layered on top of the
provider-proxy -> LiteLLM migration, plus the previously-untested
sandbox-orchestration, Bedrock-patch, and embedded-callback-logger code paths.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from benchflow.providers import litellm_bedrock_preflight as preflight_mod
from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.litellm_config import resolve_litellm_route
from benchflow.providers.litellm_logging import (
    callback_module_source,
    extract_usage_from_trajectory,
    trajectory_from_litellm_callback_log,
)
from benchflow.providers.runtime import ensure_litellm_runtime


class _FakeHostServer:
    def __init__(self, base_url: str, route):
        self._base_url = base_url
        self.route = route
        self.trajectory = None
        self.stopped = False

    @property
    def base_url(self) -> str:
        return self._base_url

    async def is_running(self) -> bool:
        return not self.stopped

    async def stop(self) -> None:
        self.stopped = True


# --------------------------------------------------------------------------- #
# Routing: native-protocol agents excluded; opencode hits a registered route   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "agent,model,expected",
    [
        ("gemini", "gemini-3.5-flash", False),
        ("oracle", "openai/gpt-4.1-mini", False),
        ("openhands", "gemini-3.5-flash", True),
        ("codex-acp", "openai/gpt-4.1-mini", True),
        ("openhands", None, False),
    ],
)
def test_needs_litellm_runtime_excludes_native_protocol_agents(agent, model, expected):
    assert runtime_mod.needs_litellm_runtime(agent, model) is expected


def test_opencode_litellm_alias_formats_to_registered_openai_route():
    from benchflow.acp.runtime import _format_acp_model

    # The proxy registers only "<alias>" and "openai/<alias>"; provider/model
    # agents must send the openai/ form, not a guessed anthropic/ provider.
    out = _format_acp_model("benchflow-minimax-MiniMax-M3", "opencode")
    assert out == "openai/benchflow-minimax-MiniMax-M3"


def test_format_acp_model_passes_through_existing_provider_prefix():
    from benchflow.acp.runtime import _format_acp_model

    assert (
        _format_acp_model("google/gemini-3.1-pro", "opencode")
        == "google/gemini-3.1-pro"
    )


def test_vllm_route_honors_runtime_supplied_base_url():
    route = resolve_litellm_route(
        "vllm/Qwen/Qwen3-Coder",
        {
            "OPENAI_API_KEY": "sk-x",
            "BENCHFLOW_PROVIDER_BASE_URL": "http://my-vllm:8000/v1",
        },
    )
    assert route.litellm_params.get("api_base") == "http://my-vllm:8000/v1"
    assert route.litellm_params["model"] == "openai/Qwen/Qwen3-Coder"


# --------------------------------------------------------------------------- #
# Security: provider keys are isolated from the agent env                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_env_strips_raw_provider_secrets(monkeypatch):
    async def fake_start(**kwargs):
        return _FakeHostServer("http://127.0.0.1:4000", kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)

    real = {
        "GEMINI_API_KEY": "real-gemini-secret",
        "OPENAI_API_KEY": "real-openai-secret",
        "AWS_BEARER_TOKEN_BEDROCK": "real-bedrock-secret",
        "AWS_REGION": "us-west-2",
    }
    updated, provider_runtime = await ensure_litellm_runtime(
        agent="openhands",
        agent_env=real,
        model="gemini/gemini-3.5-flash",
        runtime=None,
        environment="local",
        session_id="s",
    )

    # Raw provider creds must not reach the agent; only proxy creds do.
    for secret_key in ("GEMINI_API_KEY", "OPENAI_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"):
        assert secret_key not in updated
    blob = " ".join(updated.values())
    assert "real-gemini-secret" not in blob
    assert "real-openai-secret" not in blob
    assert updated["LLM_API_KEY"] == provider_runtime.master_key
    assert updated["LLM_BASE_URL"] == "http://127.0.0.1:4000/v1"


# --------------------------------------------------------------------------- #
# Networking: host proxy binds loopback locally, bridge IP for docker          #
# --------------------------------------------------------------------------- #


def test_host_bind_address_local_is_loopback():
    assert runtime_mod._host_bind_address("local") == "127.0.0.1"
    assert runtime_mod._host_bind_address("modal") == "127.0.0.1"


def test_host_bind_address_docker_uses_bridge_ip(monkeypatch):
    monkeypatch.setattr(runtime_mod, "_docker_host_address", lambda: "172.17.0.1")
    assert runtime_mod._host_bind_address("docker") == "172.17.0.1"


def test_host_bind_address_docker_hostname_falls_back_to_all_ifaces(monkeypatch):
    monkeypatch.setattr(
        runtime_mod, "_docker_host_address", lambda: "host.docker.internal"
    )
    assert runtime_mod._host_bind_address("docker") == "0.0.0.0"


def test_agent_endpoint_docker_health_uses_loopback_when_all_ifaces(monkeypatch):
    monkeypatch.setattr(
        runtime_mod, "_docker_host_address", lambda: "host.docker.internal"
    )
    endpoint = runtime_mod._agent_endpoint_for_environment(4000, "docker", "0.0.0.0")
    assert endpoint.agent_base_url == "http://host.docker.internal:4000"
    assert endpoint.local_base_url == "http://127.0.0.1:4000"


def test_agent_endpoint_docker_bridge_ip_is_used_for_both():
    endpoint = runtime_mod._agent_endpoint_for_environment(4000, "docker", "172.17.0.1")
    assert endpoint.agent_base_url == "http://172.17.0.1:4000"
    assert endpoint.local_base_url == "http://172.17.0.1:4000"


# --------------------------------------------------------------------------- #
# Robustness: deterministic callback-log flush wait (no fixed sleep)           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_await_log_stable_returns_after_size_settles():
    sizes = iter([10, 40, 80, 80, 80, 80, 80, 80, 80, 80])

    def get_size() -> int:
        return next(sizes, 80)

    await runtime_mod._await_log_stable(get_size, deadline_s=5.0, quiet_s=0.3)


@pytest.mark.asyncio
async def test_await_log_stable_supports_async_size_getter():
    calls = {"n": 0}

    async def get_size() -> int:
        calls["n"] += 1
        return 123

    await runtime_mod._await_log_stable(get_size, deadline_s=5.0, quiet_s=0.3)
    assert calls["n"] >= 2


@pytest.mark.asyncio
async def test_await_log_stable_bails_at_deadline_while_growing():
    counter = {"v": 0}

    def get_size() -> int:
        counter["v"] += 100
        return counter["v"]

    # Never settles; must still return at the deadline rather than hang.
    await runtime_mod._await_log_stable(get_size, deadline_s=0.6, quiet_s=0.3)
    assert counter["v"] > 0


# --------------------------------------------------------------------------- #
# Sandbox orchestration: real _start_sandbox_litellm + SandboxLiteLLMProcess   #
# --------------------------------------------------------------------------- #


class _ExecResult:
    def __init__(self, return_code: int = 0, stdout: str = "", stderr: str = ""):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


_SUCCESS_LOG = json.dumps(
    {
        "event": "success",
        "request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {
                "model": "MiniMax-M3",
                "messages": [{"role": "user", "content": "hi"}],
            },
        },
        "response": {
            "model": "openai/MiniMax-M3",
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        },
        "response_cost": 0.001,
        "start_time": "2026-06-04T10:00:00",
        "end_time": "2026-06-04T10:00:01",
        "duration_ms": 1000,
    }
)


class _FakeSandbox:
    """Drives the real sandbox-local LiteLLM launch/teardown code paths."""

    def __init__(
        self,
        *,
        fail_launch: bool = False,
        fail_preflight: bool = False,
        log_content: str = _SUCCESS_LOG,
    ):
        self.uploaded: dict[str, str] = {}
        self.exec_calls: list[str] = []
        self.fail_launch = fail_launch
        self.fail_preflight = fail_preflight
        self.log_content = log_content
        self._started = False

    async def upload_file(self, local_path, remote_path) -> None:
        self.uploaded[str(remote_path)] = Path(local_path).read_text()

    async def exec(self, command: str, timeout_sec: int | None = None) -> _ExecResult:
        self.exec_calls.append(command)
        if "stat -c %s" in command:
            return _ExecResult(0, stdout=str(len(self.log_content)))
        if "urllib.request" in command:
            return _ExecResult(0)
        if "bedrock_patch_preflight.py" in command:
            if self.fail_preflight:
                return _ExecResult(1, stdout="adaptive-thinking gate inactive")
            return _ExecResult(0)
        if "launcher.py" in command:
            if self.fail_launch:
                return _ExecResult(1, stderr="boom")
            self._started = True
            return _ExecResult(0)
        if "kill -0" in command:
            return _ExecResult(0, stdout="yes")
        if "kill -TERM" in command:
            return _ExecResult(0)
        if command.strip().startswith("rm -rf"):
            return _ExecResult(0)
        if command.strip().startswith("cat") and "state.json" in command:
            if self._started:
                return _ExecResult(0, stdout=json.dumps({"pid": 4242, "port": 45999}))
            return _ExecResult(0, stdout="")
        if command.strip().startswith("cat") and "callback.jsonl" in command:
            return _ExecResult(0, stdout=self.log_content)
        return _ExecResult(0)


@pytest.mark.asyncio
async def test_sandbox_litellm_launch_keeps_secrets_off_command_line():
    secret = "minimax-super-secret-key"
    route = resolve_litellm_route(
        "minimax/MiniMax-M3",
        {"MINIMAX_API_KEY": secret, "MINIMAX_BASE_URL": "https://api.minimax.io/v1"},
    )
    sandbox = _FakeSandbox()

    proc = await runtime_mod._start_sandbox_litellm(
        sandbox=sandbox,
        route=route,
        master_key="sk-master",
        agent_env={
            "MINIMAX_API_KEY": secret,
            "MINIMAX_BASE_URL": "https://api.minimax.io/v1",
        },
        session_id="s",
        agent_name="openhands",
    )

    # launch_config is uploaded as a file (proxy needs the key)...
    launch_files = [k for k in sandbox.uploaded if k.endswith("launch_config.json")]
    assert launch_files, "launch_config.json should be uploaded"
    assert secret in sandbox.uploaded[launch_files[0]]
    # ...and the secret never appears on any exec command line (/proc exposure).
    assert all(secret not in call for call in sandbox.exec_calls)
    # config.yaml uses os.environ/ refs, so the secret is not inlined there either.
    config_files = [k for k in sandbox.uploaded if k.endswith("config.yaml")]
    assert config_files and secret not in sandbox.uploaded[config_files[0]]

    assert proc.base_url == "http://127.0.0.1:45999"
    assert await proc.is_running() is True


@pytest.mark.asyncio
async def test_sandbox_litellm_stop_imports_usage_and_cleans_up():
    route = resolve_litellm_route(
        "minimax/MiniMax-M3",
        {"MINIMAX_API_KEY": "k", "MINIMAX_BASE_URL": "https://api.minimax.io/v1"},
    )
    sandbox = _FakeSandbox()
    proc = await runtime_mod._start_sandbox_litellm(
        sandbox=sandbox,
        route=route,
        master_key="sk-master",
        agent_env={
            "MINIMAX_API_KEY": "k",
            "MINIMAX_BASE_URL": "https://api.minimax.io/v1",
        },
        session_id="s",
        agent_name="openhands",
    )

    await proc.stop()

    assert proc.trajectory is not None
    usage = extract_usage_from_trajectory(
        proc.trajectory, fallback_model="openai/MiniMax-M3"
    )
    assert usage["usage_source"] == "provider_response"
    assert usage["n_input_tokens"] == 11
    assert any(call.strip().startswith("rm -rf") for call in sandbox.exec_calls)


@pytest.mark.asyncio
async def test_sandbox_litellm_startup_failure_tears_down():
    route = resolve_litellm_route(
        "minimax/MiniMax-M3",
        {"MINIMAX_API_KEY": "k", "MINIMAX_BASE_URL": "https://api.minimax.io/v1"},
    )
    sandbox = _FakeSandbox(fail_launch=True)

    with pytest.raises(RuntimeError):
        await runtime_mod._start_sandbox_litellm(
            sandbox=sandbox,
            route=route,
            master_key="sk-master",
            agent_env={
                "MINIMAX_API_KEY": "k",
                "MINIMAX_BASE_URL": "https://api.minimax.io/v1",
            },
            session_id="s",
            agent_name="openhands",
        )

    # A half-started proxy (provider keys + master_key on disk) must be removed.
    assert any(call.strip().startswith("rm -rf") for call in sandbox.exec_calls)


# --------------------------------------------------------------------------- #
# Bedrock adaptive-thinking patch actually applies (no silent no-op)           #
# --------------------------------------------------------------------------- #


def test_bedrock_patch_recognizes_bedrock_opus_48_as_adaptive_thinking():
    import benchflow.providers.litellm_bedrock_patch as patch  # importing applies the patch

    assert patch._is_new_bedrock_claude("us.anthropic.claude-opus-4-8") is True
    assert patch._is_new_bedrock_claude("us.anthropic.claude-opus-4-7") is False

    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    # The whole point of the patch: the Bedrock opus-4-8 inference-profile ID,
    # which stock litellm does NOT recognize, is gated ON. (litellm already
    # recognizes 4-6/4-7 natively, so the patch must not regress those — it
    # delegates to the original — and must not over-trigger on plain models.)
    assert (
        AnthropicConfig._is_adaptive_thinking_model("us.anthropic.claude-opus-4-8")
        is True
    )
    assert AnthropicConfig._is_adaptive_thinking_model("gpt-4o") is False


def test_bedrock_patch_flags_cost_map_when_entry_present():
    import litellm

    import benchflow.providers.litellm_bedrock_patch as patch  # importing applies the patch

    flagged = [
        key
        for key, meta in litellm.model_cost.items()
        if patch._is_new_bedrock_claude(key)
        and isinstance(meta, dict)
        and meta.get("supports_adaptive_thinking")
    ]
    bedrock_48 = [k for k in litellm.model_cost if patch._is_new_bedrock_claude(k)]
    # If litellm ships any 4.8+ bedrock cost entry, the patch must have flagged it.
    if bedrock_48:
        assert flagged, "bedrock 4.8+ cost entries should be flagged adaptive-thinking"


# --------------------------------------------------------------------------- #
# Bedrock 4.8+ patch preflight fails closed (issue #602)                       #
# --------------------------------------------------------------------------- #

_BEDROCK_ENV = {"AWS_BEARER_TOKEN_BEDROCK": "tok", "AWS_REGION": "us-east-1"}


@pytest.mark.parametrize(
    "model,required",
    [
        ("aws-bedrock/us.anthropic.claude-opus-4-8-20251101-v1:0", True),
        ("aws-bedrock/eu.anthropic.claude-sonnet-4-9-20260301-v1:0", True),
        ("aws-bedrock/us.anthropic.claude-opus-4-7-20251101-v1:0", False),
    ],
)
def test_route_requires_bedrock_patch_gating(model, required):
    """Guards PR #668's fail-closed gating for issue #602: only Bedrock Claude 4.8+
    routes require the thinking patch; 4.7 and below stay best-effort."""
    route = resolve_litellm_route(model, dict(_BEDROCK_ENV))
    assert preflight_mod.route_requires_bedrock_patch(route) is required


def test_route_requires_bedrock_patch_ignores_non_bedrock_providers():
    """Guards PR #668 / issue #602 scope: a 4.8 model on a non-Bedrock provider does not
    trigger the fail-closed preflight (direct Anthropic handles 4.8 natively)."""
    route = resolve_litellm_route(
        "minimax/MiniMax-M3",
        {"MINIMAX_API_KEY": "k", "MINIMAX_BASE_URL": "https://api.minimax.io/v1"},
    )
    assert preflight_mod.route_requires_bedrock_patch(route) is False


def test_host_bedrock_preflight_uses_litellm_shebang_python(tmp_path):
    """Guards PR #668 against checking a different Python than the LiteLLM CLI
    will actually run under when the executable is a script with a shebang."""
    litellm = tmp_path / "litellm"
    python = tmp_path / "tools" / "python"
    python.parent.mkdir()
    litellm.write_text(f"#!{python}\n")

    assert preflight_mod._host_python_for_litellm(str(litellm)) == str(python)


def test_host_bedrock_preflight_resolves_env_shebang_with_proxy_path(tmp_path):
    """Guards PR #668 against resolving `/usr/bin/env python*` with a different
    PATH than the LiteLLM proxy receives."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    python = bindir / "python3"
    python.write_text("")
    python.chmod(0o755)
    litellm = tmp_path / "litellm"
    litellm.write_text("#!/usr/bin/env python3\n")

    assert preflight_mod._host_python_for_litellm(
        str(litellm), env={"PATH": str(bindir)}
    ) == str(python)


def test_bedrock_patch_preflight_passes_when_runtime_files_on_pythonpath(tmp_path):
    """End-to-end happy path for the PR #668 preflight (issue #602): a fresh interpreter
    with the runtime dir on PYTHONPATH loads sitecustomize -> patch module, so
    the behavioral probe sees the patched litellm and exits 0."""
    import os
    import subprocess
    import sys

    runtime_mod._write_runtime_files(tmp_path, config={"model_list": []})
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", preflight_mod.BEDROCK_PATCH_PREFLIGHT_SOURCE],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_bedrock_patch_preflight_fails_closed_when_patch_not_loaded(tmp_path):
    """THE regression test for issue #602's fail-open (fixed in PR #668): when the patch never
    loads (sitecustomize missing from PYTHONPATH — the exact silent-failure
    mode), the behavioral probe against stock litellm must exit non-zero so
    setup fails before the agent launches, instead of regressing mid-task."""
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", preflight_mod.BEDROCK_PATCH_PREFLIGHT_SOURCE],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "inactive" in result.stdout


@pytest.mark.asyncio
async def test_sandbox_litellm_bedrock_route_runs_preflight():
    """Guards PR #668 / issue #602 wiring: a Bedrock 4.8+ sandbox route must run the
    patch preflight after health and succeed when the patch is active."""
    route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-opus-4-8-20251101-v1:0", dict(_BEDROCK_ENV)
    )
    sandbox = _FakeSandbox()

    proc = await runtime_mod._start_sandbox_litellm(
        sandbox=sandbox,
        route=route,
        master_key="sk-master",
        agent_env=dict(_BEDROCK_ENV),
        session_id="s",
        agent_name="openhands",
    )

    preflights = [c for c in sandbox.exec_calls if "bedrock_patch_preflight.py" in c]
    assert preflights, "Bedrock 4.8+ route must run the patch preflight"
    assert proc.base_url == "http://127.0.0.1:45999"


@pytest.mark.asyncio
async def test_sandbox_litellm_bedrock_preflight_failure_fails_closed():
    """THE sandbox fail-closed regression for issue #602 (PR #668): an inactive patch on
    a Daytona Bedrock 4.8+ route must abort startup (RuntimeError) and tear the
    half-started proxy down — never continue into the task fail-open."""
    route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-opus-4-8-20251101-v1:0", dict(_BEDROCK_ENV)
    )
    sandbox = _FakeSandbox(fail_preflight=True)

    with pytest.raises(RuntimeError, match="NOT active"):
        await runtime_mod._start_sandbox_litellm(
            sandbox=sandbox,
            route=route,
            master_key="sk-master",
            agent_env=dict(_BEDROCK_ENV),
            session_id="s",
            agent_name="openhands",
        )

    assert any(call.strip().startswith("rm -rf") for call in sandbox.exec_calls)


@pytest.mark.asyncio
async def test_sandbox_litellm_non_bedrock_route_skips_preflight():
    """Guards PR #668 / issue #602 scope: non-Bedrock routes never run the preflight, so
    a broken patch can not fail runs that do not depend on it."""
    route = resolve_litellm_route(
        "minimax/MiniMax-M3",
        {"MINIMAX_API_KEY": "k", "MINIMAX_BASE_URL": "https://api.minimax.io/v1"},
    )
    sandbox = _FakeSandbox(fail_preflight=True)  # would fail IF it ran

    proc = await runtime_mod._start_sandbox_litellm(
        sandbox=sandbox,
        route=route,
        master_key="sk-master",
        agent_env={
            "MINIMAX_API_KEY": "k",
            "MINIMAX_BASE_URL": "https://api.minimax.io/v1",
        },
        session_id="s",
        agent_name="openhands",
    )

    assert not [c for c in sandbox.exec_calls if "bedrock_patch_preflight.py" in c]
    assert proc.base_url == "http://127.0.0.1:45999"


# --------------------------------------------------------------------------- #
# Embedded callback logger actually produces importable JSONL (no drift)       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_embedded_callback_logger_round_trips_to_provider_usage(
    tmp_path, monkeypatch
):
    namespace: dict[str, object] = {}
    exec(callback_module_source(), namespace)
    logger = namespace["proxy_handler_instance"]

    log_path = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log_path))

    response = {
        "model": "gpt-4.1-mini",
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }
    kwargs = {
        "model": "benchflow-openai-gpt-4.1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_params": {"model": "openai/gpt-4.1-mini"},
        "optional_params": {},
        "call_type": "acompletion",
    }
    start = datetime(2026, 6, 4, 10, 0, 0)
    end = datetime(2026, 6, 4, 10, 0, 1)

    await logger.async_log_success_event(kwargs, response, start, end)

    text = log_path.read_text()
    assert text.strip(), "callback logger should have written a JSONL record"

    trajectory = trajectory_from_litellm_callback_log(
        text, session_id="s", agent_name="codex-acp"
    )
    usage = extract_usage_from_trajectory(
        trajectory, fallback_model="openai/gpt-4.1-mini"
    )
    assert usage["usage_source"] == "provider_response"
    assert usage["n_input_tokens"] == 12
    assert usage["n_output_tokens"] == 4


def test_gemini_usage_metadata_is_detected_as_provider_usage():
    # LiteLLM normally normalizes to OpenAI shape, but a raw Gemini passthrough
    # reports usageMetadata; it must not silently degrade to 'unavailable'.
    record = {
        "event": "success",
        "request": {"method": "POST", "path": "/v1/chat/completions", "body": {}},
        "response": {"model": "gemini/gemini-3.5-flash"},
        "usage": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 7,
            "totalTokenCount": 27,
        },
        "start_time": "2026-06-04T10:00:00",
        "end_time": "2026-06-04T10:00:01",
    }
    trajectory = trajectory_from_litellm_callback_log(
        json.dumps(record), session_id="s", agent_name="openhands"
    )
    usage = extract_usage_from_trajectory(
        trajectory, fallback_model="gemini/gemini-3.5-flash"
    )
    assert usage["usage_source"] == "provider_response"


# --------------------------------------------------------------------------- #
# Trajectory.to_jsonl redaction (coverage lost when test_atif was deleted)     #
# --------------------------------------------------------------------------- #


def test_to_jsonl_redacts_provider_secrets():
    from benchflow.trajectories.types import (
        LLMExchange,
        LLMRequest,
        LLMResponse,
        Trajectory,
    )

    anthropic_key = "sk-ant-api03-" + "Z" * 40
    openai_key = "sk-" + "Y" * 40
    trajectory = Trajectory(session_id="s", agent_name="a")
    trajectory.exchanges.append(
        LLMExchange(
            request=LLMRequest(
                body={
                    "messages": [{"role": "user", "content": f"key {anthropic_key}"}],
                    "authorization": f"Bearer {openai_key}",
                }
            ),
            response=LLMResponse(body={"content": "ok"}),
        )
    )

    redacted = trajectory.to_jsonl(redact_keys=True)
    assert anthropic_key not in redacted
    assert openai_key not in redacted
    assert "***REDACTED***" in redacted
    # Disabling redaction keeps the raw content (guards against over-redaction).
    assert anthropic_key in trajectory.to_jsonl(redact_keys=False)


# --------------------------------------------------------------------------- #
# Cost: LiteLLM is the single source; custom prices injected per-route          #
# --------------------------------------------------------------------------- #


def test_custom_cost_per_token_substring_match(monkeypatch):
    from benchflow.providers import litellm_config

    monkeypatch.setattr(
        litellm_config, "MODEL_COST_PER_TOKEN", {"minimax-m3": (3e-7, 1.2e-6)}
    )
    assert litellm_config.custom_cost_per_token("openai/MiniMax-M3") == (3e-7, 1.2e-6)
    assert litellm_config.custom_cost_per_token("gemini/gemini-3.5-flash") is None


def test_litellm_config_injects_custom_cost_into_route(monkeypatch):
    from benchflow.providers import litellm_config
    from benchflow.providers.litellm_config import (
        litellm_proxy_config,
        resolve_litellm_route,
    )

    monkeypatch.setattr(
        litellm_config, "MODEL_COST_PER_TOKEN", {"minimax-m3": (3e-7, 1.2e-6)}
    )
    route = resolve_litellm_route(
        "minimax/MiniMax-M3",
        {"MINIMAX_API_KEY": "k", "MINIMAX_BASE_URL": "https://api.minimax.io/v1"},
    )
    config = litellm_proxy_config(route, master_key="sk-master")
    for entry in config["model_list"]:
        params = entry["litellm_params"]
        assert params["input_cost_per_token"] == 3e-7
        assert params["output_cost_per_token"] == 1.2e-6


def test_litellm_config_no_cost_injection_for_litellm_priced_model():
    # gemini is priced by litellm.model_cost; benchflow must not inject anything.
    from benchflow.providers.litellm_config import (
        litellm_proxy_config,
        resolve_litellm_route,
    )

    route = resolve_litellm_route("gemini/gemini-3.5-flash", {"GEMINI_API_KEY": "k"})
    config = litellm_proxy_config(route, master_key="sk-master")
    for entry in config["model_list"]:
        assert "input_cost_per_token" not in entry["litellm_params"]


@pytest.mark.asyncio
async def test_callback_prefers_proxy_computed_response_cost(tmp_path, monkeypatch):
    # The injected logger captures the proxy's already-computed response_cost
    # (which honors per-deployment input_cost_per_token for custom models).
    from types import SimpleNamespace

    namespace: dict[str, object] = {}
    exec(callback_module_source(), namespace)
    logger = namespace["proxy_handler_instance"]
    log_path = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log_path))

    response = SimpleNamespace(
        model="openai/MiniMax-M3",
        usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        _hidden_params={"response_cost": 0.00075},
    )
    response.model_dump = lambda mode=None: {  # type: ignore[attr-defined]
        "model": "openai/MiniMax-M3",
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        },
    }
    kwargs = {
        "model": "benchflow-minimax-MiniMax-M3",
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_params": {"model": "openai/MiniMax-M3"},
        "optional_params": {},
        "call_type": "acompletion",
    }
    start = datetime(2026, 6, 4, 10, 0, 0)
    end = datetime(2026, 6, 4, 10, 0, 1)

    await logger.async_log_success_event(kwargs, response, start, end)

    record = json.loads(log_path.read_text().splitlines()[0])
    assert record["response_cost"] == 0.00075


@pytest.mark.asyncio
async def test_callback_records_unpriced_cost_as_null(tmp_path, monkeypatch):
    # The proxy reports response_cost=0.0 for models it cannot price; that must
    # be recorded as null (unknown), not a misleading $0.00, so downstream
    # cost_usd ends up None rather than 0.0.
    from types import SimpleNamespace

    namespace: dict[str, object] = {}
    exec(callback_module_source(), namespace)
    logger = namespace["proxy_handler_instance"]
    log_path = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log_path))

    response = SimpleNamespace(
        model="openai/SomeUnpricedModel",
        usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        _hidden_params={"response_cost": 0.0},
    )
    response.model_dump = lambda mode=None: {  # type: ignore[attr-defined]
        "model": "openai/SomeUnpricedModel",
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        },
    }
    kwargs = {
        "model": "benchflow-some-unpriced",
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_params": {"model": "openai/SomeUnpricedModel"},
        "optional_params": {},
        "call_type": "acompletion",
    }
    start = datetime(2026, 6, 4, 10, 0, 0)
    end = datetime(2026, 6, 4, 10, 0, 1)

    await logger.async_log_success_event(kwargs, response, start, end)

    text = log_path.read_text()
    record = json.loads(text.splitlines()[0])
    assert record["response_cost"] is None

    trajectory = trajectory_from_litellm_callback_log(
        text, session_id="s", agent_name="openhands"
    )
    usage = extract_usage_from_trajectory(trajectory)
    assert usage["usage_source"] == "provider_response"
    assert usage["n_input_tokens"] == 1000
    assert usage["cost_usd"] is None
    assert usage["price_source"] is None
