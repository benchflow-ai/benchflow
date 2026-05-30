"""Tests for sandbox-local provider usage telemetry."""

from __future__ import annotations

import base64
import json
import re
from types import SimpleNamespace

import pytest

from benchflow.trajectories.types import Trajectory


def test_agent_kill_pattern_excludes_usage_proxy_agent_name_argument():
    from benchflow.rollout import _agent_process_kill_pattern

    pattern = _agent_process_kill_pattern("/opt/benchflow/bin/codex-acp")

    assert pattern is not None
    assert re.search(pattern, "/opt/benchflow/bin/codex-acp")
    assert re.search(pattern, "node /opt/benchflow/js-agents/bin/codex-acp --flag")
    assert not re.search(pattern, "node /tmp/benchflow-usage-proxy/proxy.js")
    assert not re.search(pattern, "proxy.js --agent-name=codex-acp")


@pytest.mark.asyncio
async def test_daytona_uses_sandbox_local_proxy_not_host_proxy(monkeypatch):
    """Daytona must not point agents at a host-local proxy address."""
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ensure_usage_proxy_runtime

    class FakeSandboxUsageProxy:
        target = "https://api.anthropic.com"
        base_url = "http://127.0.0.1:49000"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.started = False
            self.trajectory = Trajectory(
                session_id=kwargs["session_id"], agent_name=kwargs["agent_name"]
            )

        async def start(self):
            self.started = True

        async def stop(self):
            return None

    monkeypatch.setattr(
        provider_runtime_mod,
        "TrajectoryProxy",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("host proxy must not start")
        ),
    )
    monkeypatch.setattr(
        provider_runtime_mod, "SandboxUsageProxy", FakeSandboxUsageProxy
    )

    env = {
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        "ANTHROPIC_API_KEY": "sk-real-key",
    }
    updated, runtime = await ensure_usage_proxy_runtime(
        agent="claude-agent-acp",
        agent_env=env,
        model="claude-haiku-4-5-20251001",
        runtime=None,
        environment="daytona",
        session_id="rollout-1",
        sandbox=object(),
    )

    assert runtime is not None
    assert runtime.server.started is True
    assert updated["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:49000"


@pytest.mark.asyncio
async def test_sandbox_usage_proxy_imports_raw_captures():
    """Sandbox-local proxy captures should reuse the canonical usage parser."""
    from benchflow.providers.runtime import ProviderRuntime, extract_usage
    from benchflow.providers.sandbox_usage_proxy import SandboxUsageProxy

    capture = {
        "duration_ms": 12,
        "request": {
            "method": "POST",
            "path": "/v1/messages",
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(
                json.dumps({"model": "claude-haiku-4-5-20251001"}).encode()
            ).decode(),
        },
        "response": {
            "status_code": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(
                json.dumps(
                    {
                        "model": "claude-haiku-4-5-20251001",
                        "usage": {"input_tokens": 13, "output_tokens": 5},
                    }
                ).encode()
            ).decode(),
        },
    }

    class FakeSandbox:
        def __init__(self):
            self.uploads = []
            self.commands = []
            self.state_reads = 0

        async def upload_file(self, source_path, target_path):
            assert any(command.startswith("mkdir -p ") for command in self.commands)
            self.uploads.append((source_path, target_path))

        async def exec(self, command, timeout_sec=None):
            self.commands.append(command)
            if command.startswith("mkdir -p "):
                return SimpleNamespace(return_code=0, stdout="", stderr="")
            if "command -v node" in command:
                return SimpleNamespace(
                    return_code=0, stdout="/usr/bin/node\n", stderr=""
                )
            if "node -e" in command or "node' -e" in command:
                assert "nohup" not in command
                assert "--agent-name" not in command
                return SimpleNamespace(return_code=0, stdout="123\n", stderr="")
            if "state.json" in command and command.strip().startswith("cat "):
                self.state_reads += 1
                if self.state_reads == 1:
                    return SimpleNamespace(return_code=0, stdout="{", stderr="")
                return SimpleNamespace(
                    return_code=0,
                    stdout='{"port":49000,"pid":123}\n',
                    stderr="",
                )
            if "captures.jsonl" in command and command.strip().startswith("cat "):
                return SimpleNamespace(
                    return_code=0,
                    stdout=json.dumps(capture) + "\n",
                    stderr="",
                )
            if "kill -TERM" in command:
                return SimpleNamespace(return_code=0, stdout="", stderr="")
            return SimpleNamespace(return_code=1, stdout="", stderr=command)

    sandbox = FakeSandbox()
    proxy = SandboxUsageProxy(
        sandbox=sandbox,
        target="https://api.anthropic.com",
        session_id="rollout-1",
        agent_name="claude-agent-acp",
    )
    await proxy.start()
    await proxy.stop()

    runtime = ProviderRuntime(
        kind="usage-proxy",
        agent_base_url=proxy.base_url,
        backend_model="claude-haiku-4-5-20251001",
        server=proxy,
    )
    usage = extract_usage(runtime)

    assert proxy.base_url == "http://127.0.0.1:49000"
    assert sandbox.state_reads == 2
    assert usage["usage_source"] == "provider_response"
    assert usage["n_input_tokens"] == 13
    assert usage["n_output_tokens"] == 5


@pytest.mark.asyncio
async def test_sandbox_usage_proxy_liveness_reports_pid_status():
    from benchflow.providers.sandbox_usage_proxy import SandboxUsageProxy

    class FakeSandbox:
        async def exec(self, command, timeout_sec=None):
            assert "kill -0" in command
            return SimpleNamespace(return_code=0, stdout="yes\n", stderr="")

    proxy = SandboxUsageProxy(
        sandbox=FakeSandbox(),
        target="https://api.anthropic.com",
        session_id="rollout-1",
        agent_name="claude-agent-acp",
    )

    assert await proxy.is_running() is True


@pytest.mark.asyncio
async def test_usage_runtime_recreated_when_sandbox_proxy_is_dead(monkeypatch):
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ProviderRuntime, ensure_usage_proxy_runtime

    stopped = []
    started = []

    class DeadServer:
        target = "https://api.anthropic.com"

        async def is_running(self):
            return False

        async def stop(self):
            stopped.append("dead")

    class FakeSandboxUsageProxy:
        target = "https://api.anthropic.com"
        base_url = "http://127.0.0.1:49001"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.trajectory = Trajectory(
                session_id=kwargs["session_id"], agent_name=kwargs["agent_name"]
            )

        async def start(self):
            started.append(self.target)

        async def stop(self):
            stopped.append("new")

    monkeypatch.setattr(
        provider_runtime_mod, "SandboxUsageProxy", FakeSandboxUsageProxy
    )

    stale_runtime = ProviderRuntime(
        kind="usage-proxy",
        agent_base_url="http://127.0.0.1:49000",
        backend_model="claude-haiku-4-5-20251001",
        server=DeadServer(),
    )

    updated, runtime = await ensure_usage_proxy_runtime(
        agent="claude-agent-acp",
        agent_env={"ANTHROPIC_BASE_URL": "https://api.anthropic.com"},
        model="claude-haiku-4-5-20251001",
        runtime=stale_runtime,
        environment="daytona",
        session_id="rollout-1",
        sandbox=object(),
    )

    assert stopped == ["dead"]
    assert started == ["https://api.anthropic.com"]
    assert runtime is not None
    assert runtime is not stale_runtime
    assert updated["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:49001"


@pytest.mark.asyncio
async def test_daytona_openhands_bedrock_usage_proxy_skips_direct_bedrock(monkeypatch):
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import (
        ensure_bedrock_proxy_runtime,
        ensure_usage_proxy_runtime,
    )
    from benchflow.usage_tracking import UsageTrackingConfig

    monkeypatch.setattr(
        provider_runtime_mod,
        "SandboxUsageProxy",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("native Bedrock calls cannot use generic usage proxy")
        ),
    )

    agent_env = {
        "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
        "AWS_REGION": "us-west-2",
        "LLM_BASE_URL": "",
        "LLM_MODEL": "anthropic/us.anthropic.claude-opus-4-7",
    }
    bedrock_env, bedrock_runtime = await ensure_bedrock_proxy_runtime(
        agent="openhands",
        agent_env=agent_env,
        model="aws-bedrock/us.anthropic.claude-opus-4-7",
        runtime=None,
        environment="daytona",
    )

    assert bedrock_runtime is None
    assert "LLM_BASE_URL" not in bedrock_env
    assert bedrock_env["LLM_MODEL"] == "bedrock/us.anthropic.claude-opus-4-7"

    usage_env, usage_runtime = await ensure_usage_proxy_runtime(
        agent="openhands",
        agent_env=bedrock_env,
        model="aws-bedrock/us.anthropic.claude-opus-4-7",
        runtime=None,
        environment="daytona",
        usage_tracking=UsageTrackingConfig(mode="auto"),
        sandbox=object(),
    )

    assert usage_runtime is None
    assert usage_env == bedrock_env
    assert "LLM_BASE_URL" not in usage_env

    with pytest.raises(RuntimeError, match="Remote Bedrock-direct"):
        await ensure_usage_proxy_runtime(
            agent="openhands",
            agent_env=bedrock_env,
            model="aws-bedrock/us.anthropic.claude-opus-4-7",
            runtime=None,
            environment="daytona",
            usage_tracking=UsageTrackingConfig(mode="required"),
            sandbox=object(),
        )
