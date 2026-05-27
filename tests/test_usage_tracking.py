"""Tests for remote token usage tracking policy and config wiring."""

from __future__ import annotations

import asyncio
import json
import socket

import httpx
import pytest

from benchflow.trajectories.types import Trajectory


def _unused_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _read_http_request(
    reader: asyncio.StreamReader,
) -> tuple[str, str, bytes]:
    request_line = (await reader.readline()).decode()
    method, path, _version = request_line.strip().split(" ", 2)
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        key, _, value = line.decode().partition(":")
        headers[key.lower().strip()] = value.strip()
    content_length = int(headers.get("content-length", "0"))
    body = await reader.readexactly(content_length) if content_length else b""
    return method, path, body


@pytest.mark.asyncio
async def test_daytona_required_usage_tracking_requires_external_endpoint():
    """Guards PR #568: required remote tracking must fail closed."""
    from benchflow.providers.runtime import ensure_usage_proxy_runtime
    from benchflow.usage_tracking import UsageTrackingConfig

    with pytest.raises(RuntimeError, match="Token usage tracking is required"):
        await ensure_usage_proxy_runtime(
            agent="claude-agent-acp",
            agent_env={
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
                "ANTHROPIC_API_KEY": "sk-real-key",
            },
            model="claude-haiku-4-5-20251001",
            runtime=None,
            environment="daytona",
            session_id="rollout-1",
            usage_tracking=UsageTrackingConfig(mode="required"),
        )


@pytest.mark.asyncio
async def test_daytona_external_usage_proxy_advertises_tunnel_url(monkeypatch):
    """Guards PR #568: remote tracking must not inject local-only addresses."""
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import (
        ensure_usage_proxy_runtime,
        stop_provider_runtime,
    )
    from benchflow.usage_tracking import UsageTrackingConfig

    class FakeTrajectoryProxy:
        def __init__(
            self,
            target=None,
            session_id="",
            agent_name="",
            host="127.0.0.1",
            port=0,
            prompt_cache_retention=None,
            path_prefix="",
        ):
            self.target = target
            self.session_id = session_id
            self.agent_name = agent_name
            self.host = host
            self.port = port
            self.prompt_cache_retention = prompt_cache_retention
            self.path_prefix = path_prefix
            self.trajectory = Trajectory(session_id=session_id, agent_name=agent_name)
            self.routes = {}
            self.started = False

        async def start(self):
            self.started = True

        async def stop(self):
            return None

        @property
        def has_routes(self):
            return bool(self.routes)

        def register_route(
            self,
            *,
            target,
            session_id="",
            agent_name="",
            prompt_cache_retention=None,
            path_prefix="",
        ):
            self.target = target
            self.session_id = session_id
            self.agent_name = agent_name
            self.prompt_cache_retention = prompt_cache_retention
            self.path_prefix = path_prefix
            self.trajectory = Trajectory(session_id=session_id, agent_name=agent_name)
            self.routes[path_prefix] = self.trajectory
            return self.trajectory

        def unregister_route(self, path_prefix):
            self.routes.pop(path_prefix, None)

    async def reachable(_url):
        return True

    monkeypatch.setattr(provider_runtime_mod, "TrajectoryProxy", FakeTrajectoryProxy)
    monkeypatch.setattr(
        provider_runtime_mod, "_external_usage_proxy_reachable", reachable
    )

    updated, runtime = await ensure_usage_proxy_runtime(
        agent="openhands",
        agent_env={
            "LLM_BASE_URL": "https://llm-proxy.example.test",
            "LLM_API_KEY": "sk-real-key",
        },
        model="gpt-4.1-mini",
        runtime=None,
        environment="daytona",
        session_id="rollout-1",
        usage_tracking=UsageTrackingConfig(
            mode="required",
            advertised_base_url="https://usage-proxy.example.test",
            port=18081,
        ),
    )

    assert runtime is not None
    assert runtime.server.proxy.started is True
    assert runtime.server.proxy.host == "127.0.0.1"
    assert runtime.server.port == 18081
    assert runtime.server.path_prefix.startswith("/__benchflow/")
    assert runtime.base_url.startswith("https://usage-proxy.example.test/__benchflow/")
    assert updated["LLM_BASE_URL"] == runtime.base_url
    assert updated["BENCHFLOW_PROVIDER_BASE_URL"] == runtime.base_url
    await stop_provider_runtime(runtime)


@pytest.mark.asyncio
async def test_external_usage_proxy_multiplexes_concurrent_rollouts_on_one_port():
    """Guards PR #568 follow-up: fixed-port Daytona tracking multiplexes rollouts."""
    from benchflow.providers.runtime import (
        ensure_usage_proxy_runtime,
        stop_provider_runtime,
    )
    from benchflow.usage_tracking import UsageTrackingConfig

    upstream_requests: list[tuple[str, str, bytes]] = []

    async def upstream_handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        method, path, body = await _read_http_request(reader)
        upstream_requests.append((method, path, body))
        response_body = json.dumps(
            {
                "id": f"chatcmpl-{len(upstream_requests)}",
                "model": "gpt-4.1-mini",
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                },
            },
            separators=(",", ":"),
        ).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"content-type: application/json\r\n"
            + f"content-length: {len(response_body)}\r\n\r\n".encode()
            + response_body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    upstream = await asyncio.start_server(upstream_handler, "127.0.0.1", 0)
    upstream_port = upstream.sockets[0].getsockname()[1]
    proxy_port = _unused_local_port()
    usage_tracking = UsageTrackingConfig(
        mode="required",
        advertised_base_url=f"http://127.0.0.1:{proxy_port}",
        port=proxy_port,
    )
    runtime1 = None
    runtime2 = None

    try:
        _env1, runtime1 = await ensure_usage_proxy_runtime(
            agent="openhands",
            agent_env={
                "LLM_BASE_URL": f"http://127.0.0.1:{upstream_port}",
                "LLM_API_KEY": "sk-test",
            },
            model="gpt-4.1-mini",
            runtime=None,
            environment="daytona",
            session_id="rollout-1",
            usage_tracking=usage_tracking,
        )
        _env2, runtime2 = await ensure_usage_proxy_runtime(
            agent="openhands",
            agent_env={
                "LLM_BASE_URL": f"http://127.0.0.1:{upstream_port}",
                "LLM_API_KEY": "sk-test",
            },
            model="gpt-4.1-mini",
            runtime=None,
            environment="daytona",
            session_id="rollout-2",
            usage_tracking=usage_tracking,
        )

        assert runtime1 is not None
        assert runtime2 is not None
        assert runtime1.base_url != runtime2.base_url
        assert runtime1.server.proxy is runtime2.server.proxy

        async with httpx.AsyncClient() as client:
            response1, response2 = await asyncio.gather(
                client.post(
                    f"{runtime1.base_url}/chat/completions",
                    json={"model": "gpt-4.1-mini", "messages": []},
                ),
                client.post(
                    f"{runtime2.base_url}/chat/completions",
                    json={"model": "gpt-4.1-mini", "messages": []},
                ),
            )
            assert response1.status_code == 200
            assert response2.status_code == 200
            assert len(runtime1.server.trajectory.exchanges) == 1
            assert len(runtime2.server.trajectory.exchanges) == 1

            await stop_provider_runtime(runtime1)
            runtime1 = None

            health = await client.get(f"{runtime2.base_url}/__benchflow_health")
            assert health.status_code == 200

        assert [request[1] for request in upstream_requests] == [
            "/chat/completions",
            "/chat/completions",
        ]
    finally:
        if runtime1 is not None:
            await stop_provider_runtime(runtime1)
        if runtime2 is not None:
            await stop_provider_runtime(runtime2)
        upstream.close()
        await upstream.wait_closed()


def test_evaluation_yaml_loads_required_usage_tracking(tmp_path):
    """Guards PR #568: eval YAML should preserve required usage tracking."""
    from benchflow.evaluation import Evaluation

    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    config = tmp_path / "eval.yaml"
    config.write_text(
        "\n".join(
            [
                f"tasks_dir: {tasks_dir}",
                "agent: openhands",
                "model: gpt-4.1-mini",
                "environment: daytona",
                "usage_tracking: required",
                "usage_proxy:",
                "  advertised_base_url: https://usage-proxy.example.test",
                "  port: 18081",
            ]
        )
    )

    evaluation = Evaluation.from_yaml(config)

    assert evaluation._config.usage_tracking.mode == "required"
    assert (
        evaluation._config.usage_tracking.advertised_base_url
        == "https://usage-proxy.example.test"
    )
    assert evaluation._config.usage_tracking.port == 18081


def test_evaluation_preflight_fails_required_daytona_without_endpoint(tmp_path):
    """Guards PR #568: required Daytona tracking fails before agent launch."""
    from benchflow.evaluation import Evaluation, EvaluationConfig
    from benchflow.usage_tracking import UsageTrackingConfig

    evaluation = Evaluation(
        tasks_dir=tmp_path,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            environment="daytona",
            usage_tracking=UsageTrackingConfig(mode="required"),
        ),
    )

    with pytest.raises(RuntimeError, match="no external usage proxy endpoint"):
        evaluation._preflight_usage_tracking()


def test_evaluation_preflight_rejects_external_proxy_port_zero(tmp_path):
    """Guards PR #568: external proxy tracking needs a stable local port."""
    from benchflow.evaluation import Evaluation, EvaluationConfig
    from benchflow.usage_tracking import UsageTrackingConfig

    evaluation = Evaluation(
        tasks_dir=tmp_path,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            concurrency=1,
            environment="daytona",
            usage_tracking=UsageTrackingConfig(
                mode="required",
                advertised_base_url="https://usage-proxy.example.test",
                port=0,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="fixed positive local proxy port"):
        evaluation._preflight_usage_tracking()


def test_evaluation_preflight_allows_external_proxy_concurrency(tmp_path):
    """Guards PR #568 follow-up: one fixed external proxy port hosts concurrent rollouts."""
    from benchflow.evaluation import Evaluation, EvaluationConfig
    from benchflow.usage_tracking import UsageTrackingConfig

    evaluation = Evaluation(
        tasks_dir=tmp_path,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            concurrency=94,
            environment="daytona",
            usage_tracking=UsageTrackingConfig(
                mode="required",
                advertised_base_url="https://usage-proxy.example.test",
                port=18081,
            ),
        ),
    )

    evaluation._preflight_usage_tracking()


def test_explicit_auto_usage_tracking_beats_env_default(monkeypatch):
    """Guards PR #568: explicit auto should override env-level required."""
    from benchflow.usage_tracking import USAGE_TRACKING_ENV, UsageTrackingConfig

    monkeypatch.setenv(USAGE_TRACKING_ENV, "required")

    assert UsageTrackingConfig().with_env_defaults().mode == "required"
    assert UsageTrackingConfig(mode="auto").with_env_defaults().mode == "auto"


def test_usage_tracking_shard_payload_preserves_implicit_env_mode(monkeypatch):
    """Guards PR #568: sharded workers must still inherit env-level required."""
    from benchflow.eval_sharding import EvalShard, _config_payload
    from benchflow.eval_worker import _evaluation_config
    from benchflow.evaluation import EvaluationConfig
    from benchflow.usage_tracking import USAGE_TRACKING_ENV, UsageTrackingConfig

    monkeypatch.setenv(USAGE_TRACKING_ENV, "required")
    parent_config = EvaluationConfig(
        environment="daytona",
        usage_tracking=UsageTrackingConfig(),
    )

    payload = _config_payload(
        parent_config,
        shard=EvalShard(index=0, task_names=("task-a",), concurrency=1),
        environment_manifest_path=None,
    )
    worker_config = _evaluation_config(payload)

    assert "usage_tracking" not in payload
    assert worker_config.usage_tracking.mode_is_explicit is False
    assert worker_config.usage_tracking.with_env_defaults().mode == "required"


def test_usage_tracking_shard_payload_uses_flat_yaml_shape():
    """Guards PR #568: worker payload must not nest usage_tracking twice."""
    from benchflow.eval_sharding import EvalShard, _config_payload
    from benchflow.eval_worker import _evaluation_config
    from benchflow.evaluation import EvaluationConfig
    from benchflow.usage_tracking import UsageTrackingConfig

    parent_config = EvaluationConfig(
        environment="daytona",
        usage_tracking=UsageTrackingConfig(
            mode="required",
            advertised_base_url="https://usage-proxy.example.test",
            port=18081,
        ),
    )

    payload = _config_payload(
        parent_config,
        shard=EvalShard(index=0, task_names=("task-a",), concurrency=1),
        environment_manifest_path=None,
    )
    worker_config = _evaluation_config(payload)

    assert payload["usage_tracking"] == "required"
    assert payload["usage_proxy"] == {
        "advertised_base_url": "https://usage-proxy.example.test",
        "port": 18081,
    }
    assert worker_config.usage_tracking.mode == "required"
    assert (
        worker_config.usage_tracking.advertised_base_url
        == "https://usage-proxy.example.test"
    )
    assert worker_config.usage_tracking.port == 18081


def test_usage_tracking_overlay_preserves_yaml_fields_for_partial_cli_override():
    """Guards PR #568: partial CLI usage overrides must not erase YAML policy."""
    from benchflow.usage_tracking import UsageTrackingConfig

    yaml_config = UsageTrackingConfig(
        mode="required",
        advertised_base_url="https://old-proxy.example.test",
        port=18081,
    )
    cli_override = UsageTrackingConfig(
        advertised_base_url="https://new-proxy.example.test",
    )

    merged = yaml_config.overlay(cli_override)

    assert merged.mode == "required"
    assert merged.advertised_base_url == "https://new-proxy.example.test"
    assert merged.port == 18081


def test_external_usage_tracking_rejects_multiple_shard_workers():
    """Guards PR #568: sharded workers cannot share one fixed proxy port."""
    from benchflow.usage_tracking import UsageTrackingConfig

    config = UsageTrackingConfig(
        advertised_base_url="https://usage-proxy.example.test",
        port=18081,
    )

    with pytest.raises(ValueError, match="sharded workers cannot share"):
        config.validate_parallelism(concurrency=1, worker_count=2)


def test_usage_proxy_advertised_base_url_rejects_path():
    """Guards PR #568: advertised proxy URLs must be root base URLs."""
    from benchflow.usage_tracking import UsageTrackingConfig

    with pytest.raises(ValueError, match="must not include a path"):
        UsageTrackingConfig(
            advertised_base_url="https://usage-proxy.example.test/benchflow"
        )


def test_usage_tracking_mapping_preserves_zero_port():
    """Guards PR #568: config sharding must preserve explicit port=0."""
    from benchflow.usage_tracking import UsageTrackingConfig

    config = UsageTrackingConfig.from_mapping(
        {
            "usage_tracking": "required",
            "usage_proxy": {
                "advertised_base_url": "https://usage-proxy.example.test",
                "port": 0,
            },
        }
    )

    assert config.port == 0
    assert config.has_fixed_proxy_port is False


@pytest.mark.asyncio
async def test_completed_eval_resume_skips_usage_preflight(tmp_path, monkeypatch):
    """Guards PR #568: completed resumes should not require a live usage proxy."""
    from benchflow.evaluation import Evaluation, EvaluationConfig
    from benchflow.usage_tracking import UsageTrackingConfig

    task_dir = tmp_path / "tasks" / "done-task"
    task_dir.mkdir(parents=True)
    evaluation = Evaluation(
        tasks_dir=tmp_path / "tasks",
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            environment="daytona",
            usage_tracking=UsageTrackingConfig(mode="required"),
        ),
    )

    def fail_preflight():
        raise AssertionError("usage preflight should not run for completed jobs")

    async def no_fresh_runs(remaining):
        assert remaining == []
        return []

    monkeypatch.setattr(evaluation, "_preflight_usage_tracking", fail_preflight)
    monkeypatch.setattr(evaluation, "_prune_docker", lambda: None)
    monkeypatch.setattr(evaluation, "_get_task_dirs", lambda: [task_dir])
    monkeypatch.setattr(
        evaluation,
        "_get_completed_tasks",
        lambda: {
            "done-task": {
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
                "n_tool_calls": 0,
                "agent_result": {"usage_source": "unavailable"},
            }
        },
    )
    monkeypatch.setattr(evaluation, "_run_parallel_independent", no_fresh_runs)

    result = await evaluation.run()

    assert result.total == 1
    assert result.passed == 1
