"""Tests for remote token usage tracking policy and config wiring."""

from __future__ import annotations

import pytest

from benchflow.trajectories.types import Trajectory


@pytest.mark.asyncio
async def test_daytona_required_usage_tracking_requires_sandbox_handle():
    """Guards the Daytona sandbox-local proxy path: required still fails closed."""
    from benchflow.providers.runtime import ensure_usage_proxy_runtime
    from benchflow.usage_tracking import UsageTrackingConfig

    with pytest.raises(RuntimeError, match="sandbox-local usage proxy"):
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
async def test_daytona_usage_tracking_starts_sandbox_local_proxy(monkeypatch):
    """Daytona auto telemetry should use a proxy inside the agent sandbox."""
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ensure_usage_proxy_runtime
    from benchflow.usage_tracking import UsageTrackingConfig

    class FakeSandboxUsageProxy:
        def __init__(
            self,
            sandbox,
            target,
            session_id,
            agent_name,
            prompt_cache_retention=None,
        ):
            self.sandbox = sandbox
            self.target = target
            self.session_id = session_id
            self.agent_name = agent_name
            self.prompt_cache_retention = prompt_cache_retention
            self.trajectory = Trajectory(session_id=session_id, agent_name=agent_name)
            self.started = False
            self.base_url = "http://127.0.0.1:49152"

        async def start(self):
            self.started = True

        async def stop(self):
            return None

    monkeypatch.setattr(
        provider_runtime_mod, "SandboxUsageProxy", FakeSandboxUsageProxy
    )
    sandbox = object()

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
        usage_tracking=UsageTrackingConfig(mode="required"),
        sandbox=sandbox,
    )

    assert runtime is not None
    assert runtime.server.started is True
    assert runtime.server.sandbox is sandbox
    assert runtime.server.target == "https://llm-proxy.example.test"
    assert runtime.base_url == "http://127.0.0.1:49152"
    assert updated["LLM_BASE_URL"] == runtime.base_url
    assert updated["BENCHFLOW_PROVIDER_BASE_URL"] == runtime.base_url


def test_evaluation_yaml_loads_required_usage_tracking(tmp_path):
    """Usage policy should round-trip through eval YAML."""
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
            ]
        )
    )

    evaluation = Evaluation.from_yaml(config)

    assert evaluation._config.usage_tracking.mode == "required"


def test_evaluation_preflight_allows_required_daytona(tmp_path):
    """Daytona required tracking is checked when the sandbox proxy is started."""
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
    """Worker payload must preserve the flat usage_tracking policy shape."""
    from benchflow.eval_sharding import EvalShard, _config_payload
    from benchflow.eval_worker import _evaluation_config
    from benchflow.evaluation import EvaluationConfig
    from benchflow.usage_tracking import UsageTrackingConfig

    parent_config = EvaluationConfig(
        environment="daytona",
        usage_tracking=UsageTrackingConfig(mode="required"),
    )

    payload = _config_payload(
        parent_config,
        shard=EvalShard(index=0, task_names=("task-a",), concurrency=1),
        environment_manifest_path=None,
    )
    worker_config = _evaluation_config(payload)

    assert payload["usage_tracking"] == "required"
    assert worker_config.usage_tracking.mode == "required"


def test_usage_tracking_overlay_preserves_existing_mode_for_partial_cli_override():
    """A partial CLI override should not erase YAML usage policy."""
    from benchflow.usage_tracking import UsageTrackingConfig

    yaml_config = UsageTrackingConfig(mode="required")
    cli_override = UsageTrackingConfig()

    merged = yaml_config.overlay(cli_override)

    assert merged.mode == "required"


def test_sandbox_local_usage_tracking_allows_multiple_shard_workers():
    """Per-sandbox proxies should not impose a global fixed-port worker limit."""
    from benchflow.usage_tracking import UsageTrackingConfig

    config = UsageTrackingConfig()

    config.validate_parallelism(concurrency=1, worker_count=2)


def test_usage_tracking_mapping_ignores_legacy_usage_proxy_section():
    """Legacy usage_proxy sections should not affect the default policy."""
    from benchflow.usage_tracking import UsageTrackingConfig

    config = UsageTrackingConfig.from_mapping(
        {
            "usage_tracking": "required",
            "usage_proxy": {
                "ignored": "value",
            },
        }
    )

    assert config.mode == "required"


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
