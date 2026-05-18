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


def _build_result(trial_dir: Path, **overrides):
    from benchflow.sdk import SDK

    defaults = dict(
        task_name="usage-task",
        trial_name="usage-trial",
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
    return SDK._build_result(trial_dir, **defaults)


def _result_json(trial_dir: Path) -> dict:
    return json.loads((trial_dir / "result.json").read_text())


def test_result_json_contains_unavailable_usage_defaults(tmp_path):
    result = _build_result(tmp_path)
    data = _result_json(tmp_path)

    assert result.usage_source == "unavailable"
    assert data["agent_result"] == {
        "n_tool_calls": 3,
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
    from benchflow._provider_runtime import extract_usage

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
    from benchflow._provider_runtime import ProviderRuntime, extract_usage

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
    from benchflow._provider_runtime import ProviderRuntime, extract_usage

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
    from benchflow._provider_runtime import ProviderRuntime, extract_usage

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
    from benchflow import _provider_runtime as provider_runtime_mod
    from benchflow._provider_runtime import ensure_usage_proxy_runtime

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
        session_id="trial-1",
    )

    assert runtime is not None
    assert runtime.server.target == "https://api.anthropic.com"
    assert runtime.server.started is True
    assert runtime.host == "host.docker.internal"
    assert updated["BENCHFLOW_PROVIDER_BASE_URL"] == "http://host.docker.internal:32124"
    assert updated["ANTHROPIC_BASE_URL"] == "http://host.docker.internal:32124"


@pytest.mark.asyncio
async def test_start_proxy_uses_openai_v1_default_for_codex(monkeypatch):
    from benchflow import _provider_runtime as provider_runtime_mod
    from benchflow._provider_runtime import ensure_usage_proxy_runtime

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
        session_id="trial-1",
    )

    assert runtime is not None
    assert runtime.server.target == "https://api.openai.com/v1"
    assert updated["OPENAI_BASE_URL"] == "http://host.docker.internal:32124"


@pytest.mark.asyncio
async def test_start_proxy_passes_prompt_cache_retention(monkeypatch):
    from benchflow import _provider_runtime as provider_runtime_mod
    from benchflow._provider_runtime import ensure_usage_proxy_runtime

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
        session_id="trial-1",
    )

    assert runtime is not None
    assert runtime.server.prompt_cache_retention == "24h"


@pytest.mark.asyncio
async def test_no_proxy_for_oracle():
    from benchflow._provider_runtime import ensure_usage_proxy_runtime

    env = {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}
    updated, runtime = await ensure_usage_proxy_runtime(
        agent="oracle",
        agent_env=env,
        model=None,
        runtime=None,
        environment="docker",
        session_id="trial-1",
    )

    assert updated == env
    assert runtime is None


def test_total_tokens_is_sum_of_parts():
    from benchflow._provider_runtime import ProviderRuntime, extract_usage

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

    from benchflow.trial import Trial, TrialConfig

    calls: list[str] = []
    cfg = TrialConfig(
        task_path=tmp_path / "task",
        agent="codex-acp",
        model="gpt-4.1-mini",
        agent_env={"OPENAI_API_KEY": "sk-test"},
    )
    trial = Trial.__new__(Trial)
    trial._config = cfg
    trial._env = SimpleNamespace()
    trial._trial_dir = tmp_path
    trial._trial_name = "trial-1"
    trial._timing = {}
    trial._agent_launch = "codex-acp"
    trial._agent_cwd = "/app"
    trial._agent_env = dict(cfg.agent_env)

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
        return (AsyncMock(), AsyncMock(), "codex-acp")

    monkeypatch.setattr("benchflow.rollout.ensure_bedrock_proxy_runtime", fake_bedrock)
    monkeypatch.setattr("benchflow.rollout.ensure_usage_proxy_runtime", fake_usage)
    monkeypatch.setattr("benchflow.rollout.connect_acp", fake_connect_acp)

    await trial.connect()

    assert calls == ["bedrock", "usage", "acp"]


@pytest.mark.asyncio
async def test_rollout_cleanup_extracts_usage_and_writes_llm_trajectory(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from benchflow._provider_runtime import ProviderRuntime
    from benchflow.trial import Trial, TrialConfig

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
    trial = Trial.__new__(Trial)
    trial._config = TrialConfig(task_path=tmp_path / "task")
    trial._trajectory = []
    trial._acp_client = None
    trial._agent_launch = ""
    trial._env = SimpleNamespace(stop=AsyncMock())
    trial._usage_runtime = ProviderRuntime(
        kind="usage-proxy",
        host="host.docker.internal",
        port=32124,
        backend_model="claude-haiku-4-5-20251001",
        server=server,
    )
    trial._trial_dir = tmp_path

    await trial.cleanup()

    assert server.stopped is True
    assert trial._usage_metrics["usage_source"] == "provider_response"
    assert trial._usage_metrics["n_input_tokens"] == 10
    assert trial._usage_metrics["n_output_tokens"] == 2
    llm_traj = tmp_path / "trajectory" / "llm_trajectory.jsonl"
    assert llm_traj.exists()
    assert json.loads(llm_traj.read_text().splitlines()[0])["response"]["body"][
        "usage"
    ] == {"input_tokens": 10, "output_tokens": 2}
