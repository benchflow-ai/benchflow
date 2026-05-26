"""Tests for provider token/cost telemetry capture and serialization."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
)


def _build_result(rollout_dir: Path, **overrides):
    from benchflow.rollout import _build_rollout_result

    defaults = dict(
        task_name="usage-task",
        rollout_name="usage-rollout",
        agent="claude-agent-acp",
        agent_name="Claude",
        model="claude-haiku-4-5-20251001",
        n_tool_calls=3,
        prompts=["solve"],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime(2026, 5, 18, 12, 0),
        timing={},
    )
    defaults.update(overrides)
    return _build_rollout_result(rollout_dir, **defaults)


def _result_json(rollout_dir: Path) -> dict:
    return json.loads((rollout_dir / "result.json").read_text())


def test_result_json_contains_unavailable_usage_defaults(tmp_path):
    result = _build_result(tmp_path)
    data = _result_json(tmp_path)

    assert result.usage_source == "unavailable"
    assert data["agent_result"] == {
        "n_tool_calls": 3,
        "n_skill_invocations": 0,
        "n_prompts": 1,
        "n_input_tokens": None,
        "n_output_tokens": None,
        "n_cache_read_tokens": None,
        "n_cache_creation_tokens": None,
        "total_tokens": None,
        "cost_usd": None,
        "usage_source": "unavailable",
        "price_source": None,
    }


def test_usage_source_never_absent(tmp_path):
    _build_result(tmp_path, error="agent failed", rewards=None)
    data = _result_json(tmp_path)

    assert "agent_result" in data
    assert data["agent_result"]["usage_source"] in {
        "provider_response",
        "unavailable",
    }


def test_result_json_contains_provider_usage_when_supplied(tmp_path):
    result = _build_result(
        tmp_path,
        n_input_tokens=100,
        n_output_tokens=20,
        n_cache_read_tokens=7,
        n_cache_creation_tokens=3,
        total_tokens=130,
        cost_usd=0.0012,
        usage_source="provider_response",
        price_source="pricing_table_2026-05",
    )
    data = _result_json(tmp_path)

    assert result.total_tokens == 130
    assert data["agent_result"]["n_input_tokens"] == 100
    assert data["agent_result"]["n_output_tokens"] == 20
    assert data["agent_result"]["n_cache_read_tokens"] == 7
    assert data["agent_result"]["n_cache_creation_tokens"] == 3
    assert data["agent_result"]["total_tokens"] == 130
    assert data["agent_result"]["cost_usd"] == 0.0012
    assert data["agent_result"]["usage_source"] == "provider_response"
    assert data["agent_result"]["price_source"] == "pricing_table_2026-05"


def test_telemetry_not_silently_dropped(tmp_path):
    _build_result(
        tmp_path,
        n_input_tokens=9,
        n_output_tokens=4,
        n_cache_read_tokens=2,
        n_cache_creation_tokens=1,
        total_tokens=16,
        cost_usd=0.0005,
        usage_source="provider_response",
        price_source="pricing_table_2026-05",
    )
    data = _result_json(tmp_path)
    agent_result = data["agent_result"]

    assert agent_result["usage_source"] == "provider_response"
    for field in (
        "n_input_tokens",
        "n_output_tokens",
        "n_cache_read_tokens",
        "n_cache_creation_tokens",
        "total_tokens",
    ):
        assert field in agent_result
        assert agent_result[field] != 0


def test_telemetry_unavailable_is_explicit(tmp_path):
    _build_result(tmp_path)
    agent_result = _result_json(tmp_path)["agent_result"]

    assert agent_result["usage_source"] == "unavailable"
    assert agent_result["n_input_tokens"] is None
    assert agent_result["n_output_tokens"] is None
    assert agent_result["n_cache_read_tokens"] is None
    assert agent_result["n_cache_creation_tokens"] is None
    assert agent_result["total_tokens"] is None
    assert agent_result["cost_usd"] is None


class _ProxyLike:
    def __init__(self, trajectory: Trajectory):
        self.trajectory = trajectory


def _trajectory(*bodies: dict, model: str = "claude-haiku-4-5-20251001") -> Trajectory:
    traj = Trajectory(session_id="s1", agent_name="agent")
    for body in bodies:
        traj.exchanges.append(
            LLMExchange(
                request=LLMRequest(body={"model": model, "messages": []}),
                response=LLMResponse(body=body),
            )
        )
    return traj


def test_extract_usage_none_proxy():
    from benchflow.providers.runtime import extract_usage

    assert extract_usage(None) == {
        "n_input_tokens": None,
        "n_output_tokens": None,
        "n_cache_read_tokens": None,
        "n_cache_creation_tokens": None,
        "total_tokens": None,
        "cost_usd": None,
        "usage_source": "unavailable",
        "price_source": None,
    }


def test_extract_usage_with_anthropic_exchanges():
    from benchflow.providers.runtime import ProviderRuntime, extract_usage

    runtime = ProviderRuntime(
        kind="usage-proxy",
        host="host.docker.internal",
        port=12345,
        backend_model="claude-haiku-4-5-20251001",
        server=_ProxyLike(
            _trajectory(
                {
                    "model": "claude-haiku-4-5-20251001",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 7,
                        "cache_creation_input_tokens": 3,
                    },
                },
                {
                    "model": "claude-haiku-4-5-20251001",
                    "usage": {
                        "input_tokens": 50,
                        "output_tokens": 10,
                        "cache_read_input_tokens": 2,
                        "cache_creation_input_tokens": 1,
                    },
                },
            )
        ),
    )

    usage = extract_usage(runtime)

    assert usage["n_input_tokens"] == 150
    assert usage["n_output_tokens"] == 30
    assert usage["n_cache_read_tokens"] == 9
    assert usage["n_cache_creation_tokens"] == 4
    assert usage["total_tokens"] == 193
    assert usage["cost_usd"] > 0
    assert usage["usage_source"] == "provider_response"
    assert str(usage["price_source"]).startswith("https://www.anthropic.com/pricing@")


def test_extract_usage_with_openai_exchanges():
    from benchflow.providers.runtime import ProviderRuntime, extract_usage

    runtime = ProviderRuntime(
        kind="usage-proxy",
        host="host.docker.internal",
        port=12345,
        backend_model="gpt-4.1-mini",
        server=_ProxyLike(
            _trajectory(
                {
                    "model": "gpt-4.1-mini",
                    "usage": {
                        "prompt_tokens": 25,
                        "completion_tokens": 5,
                        "total_tokens": 30,
                        "prompt_tokens_details": {"cached_tokens": 20},
                    },
                },
                {
                    "model": "gpt-4.1-mini",
                    "usage": {
                        "prompt_tokens": 75,
                        "completion_tokens": 15,
                        "total_tokens": 90,
                        "prompt_tokens_details": {"cached_tokens": 60},
                    },
                },
                model="gpt-4.1-mini",
            )
        ),
    )

    usage = extract_usage(runtime)

    assert usage["n_input_tokens"] == 100
    assert usage["n_output_tokens"] == 20
    assert usage["n_cache_read_tokens"] == 80
    assert usage["n_cache_creation_tokens"] == 0
    assert usage["total_tokens"] == 120
    assert usage["cost_usd"] == 0.000048
    assert usage["usage_source"] == "provider_response"
    assert str(usage["price_source"]).startswith("https://openai.com/api/pricing/@")


def test_extract_usage_with_gemini_exchanges():
    from benchflow.providers.runtime import ProviderRuntime, extract_usage

    runtime = ProviderRuntime(
        kind="usage-proxy",
        host="host.docker.internal",
        port=12345,
        backend_model="gemini-2.5-flash",
        server=_ProxyLike(
            _trajectory(
                {
                    "model": "gemini-2.5-flash",
                    "usageMetadata": {
                        "promptTokenCount": 11,
                        "candidatesTokenCount": 4,
                    },
                },
                model="gemini-2.5-flash",
            )
        ),
    )

    usage = extract_usage(runtime)

    assert usage["n_input_tokens"] == 11
    assert usage["n_output_tokens"] == 4
    assert usage["n_cache_read_tokens"] == 0
    assert usage["n_cache_creation_tokens"] == 0
    assert usage["total_tokens"] == 15
    assert usage["usage_source"] == "provider_response"


@pytest.mark.asyncio
async def test_start_proxy_rewrites_env(monkeypatch):
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ensure_usage_proxy_runtime

    monkeypatch.setattr(
        provider_runtime_mod, "_docker_host_address", lambda: "host.docker.internal"
    )

    class FakeTrajectoryProxy:
        def __init__(
            self,
            target,
            session_id="",
            agent_name="",
            host="127.0.0.1",
            port=0,
            prompt_cache_retention=None,
        ):
            self.target = target
            self.session_id = session_id
            self.agent_name = agent_name
            self.host = host
            self.port = 32124
            self.prompt_cache_retention = prompt_cache_retention
            self.trajectory = Trajectory(session_id=session_id, agent_name=agent_name)
            self.started = False

        async def start(self):
            self.started = True

        async def stop(self):
            return None

    monkeypatch.setattr(provider_runtime_mod, "TrajectoryProxy", FakeTrajectoryProxy)

    updated, runtime = await ensure_usage_proxy_runtime(
        agent="claude-agent-acp",
        agent_env={
            "BENCHFLOW_PROVIDER_BASE_URL": "https://api.anthropic.com",
            "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        },
        model="claude-haiku-4-5-20251001",
        runtime=None,
        environment="docker",
        session_id="rollout-1",
    )

    assert runtime is not None
    assert runtime.server.target == "https://api.anthropic.com"
    assert runtime.server.started is True
    assert runtime.host == "host.docker.internal"
    assert updated["BENCHFLOW_PROVIDER_BASE_URL"] == "http://host.docker.internal:32124"
    assert updated["ANTHROPIC_BASE_URL"] == "http://host.docker.internal:32124"


@pytest.mark.asyncio
async def test_usage_proxy_dials_loopback_for_host_bound_provider_proxy(monkeypatch):
    """Guards v0.5-integration@e55219d against host-side proxy chains dialing Docker aliases."""
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ensure_usage_proxy_runtime

    monkeypatch.setattr(
        provider_runtime_mod, "_docker_host_address", lambda: "host.docker.internal"
    )

    class FakeTrajectoryProxy:
        def __init__(
            self,
            target,
            session_id="",
            agent_name="",
            host="127.0.0.1",
            port=0,
            prompt_cache_retention=None,
        ):
            self.target = target
            self.host = host
            self.port = 32124
            self.trajectory = Trajectory(session_id=session_id, agent_name=agent_name)

        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(provider_runtime_mod, "TrajectoryProxy", FakeTrajectoryProxy)

    updated, runtime = await ensure_usage_proxy_runtime(
        agent="openhands",
        agent_env={
            "BENCHFLOW_PROVIDER_BASE_URL": "http://host.docker.internal:32123",
            "LLM_BASE_URL": "http://host.docker.internal:32123",
        },
        model="aws-bedrock/anthropic.claude-opus-4-7",
        runtime=None,
        environment="docker",
        session_id="rollout-1",
    )

    assert runtime is not None
    assert runtime.server.target == "http://127.0.0.1:32123"
    assert updated["LLM_BASE_URL"] == "http://host.docker.internal:32124"


@pytest.mark.asyncio
async def test_usage_proxy_can_be_disabled_for_operator_recovery(monkeypatch):
    """Guards v0.5-integration@e55219d recovery runs when telemetry proxying blocks rollouts."""
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ensure_usage_proxy_runtime

    def _fail_start(*_args, **_kwargs):
        raise AssertionError("TrajectoryProxy must not start when disabled")

    monkeypatch.setenv("BENCHFLOW_DISABLE_USAGE_PROXY", "1")
    monkeypatch.setattr(provider_runtime_mod, "TrajectoryProxy", _fail_start)

    env = {
        "BENCHFLOW_PROVIDER_BASE_URL": "http://host.docker.internal:32123",
        "LLM_BASE_URL": "http://host.docker.internal:32123",
    }
    updated, runtime = await ensure_usage_proxy_runtime(
        agent="openhands",
        agent_env=env,
        model="aws-bedrock/us.anthropic.claude-opus-4-7",
        runtime=None,
        environment="docker",
        session_id="rollout-1",
    )

    assert runtime is None
    assert updated == env


@pytest.mark.asyncio
async def test_start_proxy_uses_openai_v1_default_for_codex(monkeypatch):
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ensure_usage_proxy_runtime

    monkeypatch.setattr(
        provider_runtime_mod, "_docker_host_address", lambda: "host.docker.internal"
    )

    class FakeTrajectoryProxy:
        def __init__(
            self,
            target,
            session_id="",
            agent_name="",
            host="127.0.0.1",
            port=0,
            prompt_cache_retention=None,
        ):
            self.target = target
            self.session_id = session_id
            self.agent_name = agent_name
            self.host = host
            self.port = 32124
            self.prompt_cache_retention = prompt_cache_retention
            self.trajectory = Trajectory(session_id=session_id, agent_name=agent_name)

        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(provider_runtime_mod, "TrajectoryProxy", FakeTrajectoryProxy)

    updated, runtime = await ensure_usage_proxy_runtime(
        agent="codex-acp",
        agent_env={"OPENAI_API_KEY": "sk-test"},
        model="gpt-4.1-mini",
        runtime=None,
        environment="docker",
        session_id="rollout-1",
    )

    assert runtime is not None
    assert runtime.server.target == "https://api.openai.com/v1"
    assert updated["OPENAI_BASE_URL"] == "http://host.docker.internal:32124"


@pytest.mark.asyncio
async def test_start_proxy_passes_prompt_cache_retention(monkeypatch):
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ensure_usage_proxy_runtime

    monkeypatch.setattr(
        provider_runtime_mod, "_docker_host_address", lambda: "host.docker.internal"
    )

    class FakeTrajectoryProxy:
        def __init__(
            self,
            target,
            session_id="",
            agent_name="",
            host="127.0.0.1",
            port=0,
            prompt_cache_retention=None,
        ):
            self.target = target
            self.session_id = session_id
            self.agent_name = agent_name
            self.host = host
            self.port = 32124
            self.prompt_cache_retention = prompt_cache_retention
            self.trajectory = Trajectory(session_id=session_id, agent_name=agent_name)

        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(provider_runtime_mod, "TrajectoryProxy", FakeTrajectoryProxy)

    _updated, runtime = await ensure_usage_proxy_runtime(
        agent="codex-acp",
        agent_env={
            "OPENAI_API_KEY": "sk-test",
            "BENCHFLOW_PROVIDER_PROMPT_CACHE_RETENTION": "24h",
        },
        model="gpt-5.5",
        runtime=None,
        environment="docker",
        session_id="rollout-1",
    )

    assert runtime is not None
    assert runtime.server.prompt_cache_retention == "24h"


@pytest.mark.asyncio
async def test_no_proxy_for_oracle():
    from benchflow.providers.runtime import ensure_usage_proxy_runtime

    env = {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}
    updated, runtime = await ensure_usage_proxy_runtime(
        agent="oracle",
        agent_env=env,
        model=None,
        runtime=None,
        environment="docker",
        session_id="rollout-1",
    )

    assert updated == env
    assert runtime is None


@pytest.mark.asyncio
async def test_no_proxy_for_daytona_remote_sandbox(monkeypatch):
    """Daytona runs the agent on a remote host the host proxy cannot reach.

    Guards the fix from PR #327: the usage proxy must be skipped so the agent
    talks to the provider directly instead of being pointed at an unreachable
    127.0.0.1 address (the regression that produced ACP ECONNREFUSED errors).
    """
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ensure_usage_proxy_runtime

    def _fail_start(*_args, **_kwargs):
        raise AssertionError("TrajectoryProxy must not start for daytona")

    monkeypatch.setattr(provider_runtime_mod, "TrajectoryProxy", _fail_start)

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
    )

    # Proxy skipped: env left untouched (no loopback rewrite), no runtime.
    assert runtime is None
    assert updated == env
    assert updated["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"


@pytest.mark.asyncio
async def test_daytona_runtime_retired_when_environment_unreachable(monkeypatch):
    """Guards the fix from PR #327: a stale runtime from an earlier env must be
    stopped, not reused."""
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import (
        ProviderRuntime,
        ensure_usage_proxy_runtime,
    )

    stopped = []

    class _StaleServer:
        target = "https://api.anthropic.com"

        async def stop(self):
            stopped.append(True)

    stale = ProviderRuntime(
        kind="usage-proxy", host="127.0.0.1", port=999, server=_StaleServer()
    )

    monkeypatch.setattr(
        provider_runtime_mod,
        "TrajectoryProxy",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not start")),
    )

    updated, runtime = await ensure_usage_proxy_runtime(
        agent="claude-agent-acp",
        agent_env={"ANTHROPIC_BASE_URL": "https://api.anthropic.com"},
        model="claude-haiku-4-5-20251001",
        runtime=stale,
        environment="daytona",
        session_id="rollout-1",
    )

    assert runtime is None
    assert stopped == [True]
    assert updated["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"


@pytest.mark.asyncio
async def test_daytona_openhands_bedrock_uses_direct_agent_mapping(monkeypatch):
    """Guards v0.5-integration@e55219d against rejecting remote OpenHands Bedrock runs."""
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ensure_bedrock_proxy_runtime

    def _fail_start(*_args, **_kwargs):
        raise AssertionError("BedrockProxyServer must not start for daytona")

    monkeypatch.setattr(provider_runtime_mod, "BedrockProxyServer", _fail_start)

    env = {
        "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
        "AWS_REGION": "us-west-2",
        "BENCHFLOW_PROVIDER_BASE_URL": "",
        "BENCHFLOW_PROVIDER_API_KEY": "bedrock-proxy",
        "LLM_BASE_URL": "",
        "LLM_API_KEY": "bedrock-proxy",
        "LLM_MODEL": "anthropic/us.anthropic.claude-opus-4-7",
    }
    updated, runtime = await ensure_bedrock_proxy_runtime(
        agent="openhands",
        agent_env=env,
        model="aws-bedrock/us.anthropic.claude-opus-4-7",
        runtime=None,
        environment="daytona",
    )

    assert runtime is None
    assert "BENCHFLOW_PROVIDER_BASE_URL" not in updated
    assert "LLM_BASE_URL" not in updated
    assert updated["LLM_MODEL"] == "bedrock/us.anthropic.claude-opus-4-7"
    assert updated["LLM_API_KEY"] == "bedrock-token"
    assert updated["AWS_BEARER_TOKEN_BEDROCK"] == "bedrock-token"


@pytest.mark.asyncio
async def test_daytona_bedrock_rejects_agents_without_direct_support():
    """Guards v0.5-integration@e55219d against silently wiring unreachable Bedrock proxy URLs."""
    from benchflow.providers.runtime import ensure_bedrock_proxy_runtime

    with pytest.raises(RuntimeError, match="direct Bedrock support"):
        await ensure_bedrock_proxy_runtime(
            agent="codex-acp",
            agent_env={"AWS_BEARER_TOKEN_BEDROCK": "bedrock-token"},
            model="aws-bedrock/us.anthropic.claude-opus-4-7",
            runtime=None,
            environment="daytona",
        )


def test_host_proxy_reachable_only_for_local_environments():
    from benchflow.providers.runtime import host_proxy_reachable_from_agent

    # docker shares the host's network namespace via the docker bridge.
    assert host_proxy_reachable_from_agent("docker") is True
    # Remote cloud sandboxes run the agent on a separate machine.
    assert host_proxy_reachable_from_agent("daytona") is False
    assert host_proxy_reachable_from_agent("modal") is False
    # Unrecognized environments are treated conservatively as reachable.
    assert host_proxy_reachable_from_agent("") is True
    assert host_proxy_reachable_from_agent("some-future-local-env") is True


def test_total_tokens_is_sum_of_parts():
    from benchflow.providers.runtime import ProviderRuntime, extract_usage

    runtime = ProviderRuntime(
        kind="usage-proxy",
        host="host.docker.internal",
        port=12345,
        backend_model="unknown-model",
        server=_ProxyLike(
            _trajectory(
                {
                    "model": "unknown-model",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 2,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 4,
                    },
                },
                model="unknown-model",
            )
        ),
    )

    usage = extract_usage(runtime)

    assert usage["total_tokens"] == 1 + 2 + 3 + 4


@pytest.mark.asyncio
async def test_rollout_connect_wires_usage_proxy_before_acp(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from benchflow.rollout import Rollout, RolloutConfig

    calls: list[str] = []
    cfg = RolloutConfig(
        task_path=tmp_path / "task",
        agent="codex-acp",
        model="gpt-4.1-mini",
        agent_env={"OPENAI_API_KEY": "sk-test"},
    )
    rollout = Rollout.__new__(Rollout)
    rollout._config = cfg
    rollout._env = SimpleNamespace()
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "rollout-1"
    rollout._timing = {}
    rollout._agent_launch = "codex-acp"
    rollout._agent_cwd = "/app"
    rollout._agent_env = dict(cfg.agent_env)

    async def fake_bedrock(**kwargs):
        calls.append("bedrock")
        return (
            {
                **kwargs["agent_env"],
                "BENCHFLOW_PROVIDER_BASE_URL": "https://api.openai.com",
                "OPENAI_BASE_URL": "https://api.openai.com",
            },
            None,
        )

    async def fake_usage(**kwargs):
        calls.append("usage")
        return (
            {
                **kwargs["agent_env"],
                "BENCHFLOW_PROVIDER_BASE_URL": "http://host.docker.internal:32124",
                "OPENAI_BASE_URL": "http://host.docker.internal:32124",
            },
            object(),
        )

    async def fake_connect_acp(**kwargs):
        calls.append("acp")
        assert kwargs["agent_env"]["OPENAI_BASE_URL"] == (
            "http://host.docker.internal:32124"
        )
        return (AsyncMock(), AsyncMock(), AsyncMock(), "codex-acp")

    rollout._planes = SimpleNamespace(
        ensure_bedrock_proxy_runtime=fake_bedrock,
        ensure_usage_proxy_runtime=fake_usage,
        connect_acp=fake_connect_acp,
    )

    await rollout.connect()

    assert calls == ["bedrock", "usage", "acp"]


@pytest.mark.asyncio
async def test_rollout_cleanup_extracts_usage_and_writes_llm_trajectory(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from benchflow.providers.runtime import ProviderRuntime, extract_usage
    from benchflow.rollout import Rollout, RolloutConfig

    class FakeServer:
        def __init__(self):
            self.stopped = False
            self.trajectory = _trajectory(
                {
                    "model": "claude-haiku-4-5-20251001",
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            )

        async def stop(self):
            self.stopped = True

    server = FakeServer()
    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(task_path=tmp_path / "task")
    rollout._trajectory = []
    rollout._acp_client = None
    rollout._agent_launch = ""
    rollout._env = SimpleNamespace(stop=AsyncMock())
    rollout._environment = None
    rollout._usage_runtime = ProviderRuntime(
        kind="usage-proxy",
        host="host.docker.internal",
        port=32124,
        backend_model="claude-haiku-4-5-20251001",
        server=server,
    )
    rollout._planes = SimpleNamespace(
        stop_provider_runtime=lambda runtime: runtime.server.stop(),
        extract_usage=extract_usage,
    )
    rollout._rollout_dir = tmp_path

    await rollout.cleanup()

    assert server.stopped is True
    assert rollout._usage_metrics["usage_source"] == "provider_response"
    assert rollout._usage_metrics["n_input_tokens"] == 10
    assert rollout._usage_metrics["n_output_tokens"] == 2
    llm_traj = tmp_path / "trajectory" / "llm_trajectory.jsonl"
    assert llm_traj.exists()
    assert json.loads(llm_traj.read_text().splitlines()[0])["response"]["body"][
        "usage"
    ] == {"input_tokens": 10, "output_tokens": 2}


def test_cache_read_tokens_handles_null_token_details():
    """Guards PR #307: OpenAI may return `prompt_tokens_details: null`."""
    traj = _trajectory(
        {
            "model": "gpt-4.1-mini",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 5,
                "total_tokens": 55,
                "prompt_tokens_details": None,
                "input_tokens_details": None,
            },
        },
        model="gpt-4.1-mini",
    )

    assert traj.total_cache_read_tokens == 0


def test_extract_usage_prefers_captured_model_over_backend_model():
    """Guards PR #307: cost uses the model reported in captured exchanges."""
    from benchflow.providers.runtime import ProviderRuntime, extract_usage

    # backend_model is stale (haiku) but the exchange reports gpt-4.1-mini.
    runtime = ProviderRuntime(
        kind="usage-proxy",
        host="host.docker.internal",
        port=12345,
        backend_model="claude-haiku-4-5-20251001",
        server=_ProxyLike(
            _trajectory(
                {
                    "model": "gpt-4.1-mini",
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 100,
                        "total_tokens": 1100,
                    },
                },
                model="gpt-4.1-mini",
            )
        ),
    )

    usage = extract_usage(runtime)

    assert str(usage["price_source"]).startswith("https://openai.com/api/pricing/@")
    # gpt-4.1-mini pricing (0.4 in / 1.6 out per Mtok), not haiku's 1.0 / 5.0.
    assert usage["cost_usd"] == round((1000 * 0.4 + 100 * 1.6) / 1_000_000, 10)


@pytest.mark.asyncio
async def test_usage_proxy_recreated_when_target_changes(monkeypatch):
    """Guards PR #307: a multi-role provider switch must repoint the proxy."""
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ensure_usage_proxy_runtime

    monkeypatch.setattr(
        provider_runtime_mod, "_docker_host_address", lambda: "host.docker.internal"
    )
    started: list[str] = []
    stopped: list[str] = []

    class FakeTrajectoryProxy:
        def __init__(
            self,
            target,
            session_id="",
            agent_name="",
            host="127.0.0.1",
            port=0,
            prompt_cache_retention=None,
        ):
            self._target = target.rstrip("/")
            self.host = host
            self.port = 40000 + len(started)
            self.trajectory = Trajectory(session_id=session_id, agent_name=agent_name)

        @property
        def target(self) -> str:
            return self._target

        async def start(self):
            started.append(self._target)

        async def stop(self):
            stopped.append(self._target)

    monkeypatch.setattr(provider_runtime_mod, "TrajectoryProxy", FakeTrajectoryProxy)

    _env1, runtime1 = await ensure_usage_proxy_runtime(
        agent="codex-acp",
        agent_env={"OPENAI_API_KEY": "sk-test"},
        model="gpt-4.1-mini",
        runtime=None,
        environment="docker",
        session_id="r1",
    )
    assert runtime1 is not None
    assert runtime1.server.target == "https://api.openai.com/v1"

    # Same rollout, role switches to an Anthropic model — different target.
    _env2, runtime2 = await ensure_usage_proxy_runtime(
        agent="claude-agent-acp",
        agent_env={"ANTHROPIC_API_KEY": "sk-test"},
        model="claude-haiku-4-5-20251001",
        runtime=runtime1,
        environment="docker",
        session_id="r1",
    )
    assert runtime2 is not None and runtime2 is not runtime1
    assert runtime2.server.target == "https://api.anthropic.com"
    assert stopped == ["https://api.openai.com/v1"]

    # Same target again — the proxy is reused, not recreated.
    _env3, runtime3 = await ensure_usage_proxy_runtime(
        agent="claude-agent-acp",
        agent_env={"ANTHROPIC_API_KEY": "sk-test"},
        model="claude-haiku-4-5-20251001",
        runtime=runtime2,
        environment="docker",
        session_id="r1",
    )
    assert runtime3 is runtime2
    assert started == ["https://api.openai.com/v1", "https://api.anthropic.com"]
