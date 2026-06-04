"""Tests for remote token usage tracking policy and config wiring."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_daytona_required_usage_tracking_requires_sandbox_handle():
    """Guards the LiteLLM sandbox-local path: required still fails closed."""
    from benchflow.providers.runtime import ensure_litellm_runtime
    from benchflow.usage_tracking import UsageTrackingConfig

    with pytest.raises(RuntimeError, match="sandbox-local LiteLLM"):
        await ensure_litellm_runtime(
            agent="claude-agent-acp",
            agent_env={"ANTHROPIC_API_KEY": "sk-real-key"},
            model="claude-haiku-4-5-20251001",
            runtime=None,
            environment="daytona",
            session_id="rollout-1",
            usage_tracking=UsageTrackingConfig(mode="required"),
        )


@pytest.mark.asyncio
async def test_daytona_usage_tracking_starts_sandbox_local_litellm(monkeypatch):
    """Daytona auto telemetry should use LiteLLM inside the agent sandbox."""
    from benchflow.providers import litellm_runtime as runtime_mod
    from benchflow.providers.runtime import ensure_litellm_runtime
    from benchflow.usage_tracking import UsageTrackingConfig

    class FakeSandboxLiteLLM:
        def __init__(self, sandbox, route):
            self.sandbox = sandbox
            self.route = route
            self.trajectory = None
            self.base_url = "http://127.0.0.1:49152"

        async def is_running(self):
            return True

        async def stop(self):
            return None

    async def fake_start(**kwargs):
        return FakeSandboxLiteLLM(kwargs["sandbox"], kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_sandbox_litellm", fake_start)
    sandbox = object()

    updated, runtime = await ensure_litellm_runtime(
        agent="openhands",
        agent_env={"OPENAI_API_KEY": "sk-real-key"},
        model="openai/gpt-4.1-mini",
        runtime=None,
        environment="daytona",
        session_id="rollout-1",
        usage_tracking=UsageTrackingConfig(mode="required"),
        sandbox=sandbox,
    )

    assert runtime is not None
    assert runtime.server.sandbox is sandbox
    assert runtime.server.route.upstream_model == "openai/gpt-4.1-mini"
    assert runtime.base_url == "http://127.0.0.1:49152"
    assert updated["LLM_BASE_URL"] == "http://127.0.0.1:49152/v1"
    assert updated["BENCHFLOW_PROVIDER_BASE_URL"] == "http://127.0.0.1:49152/v1"


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
    """Daytona required tracking is checked when LiteLLM starts in the sandbox."""
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


@pytest.mark.parametrize(
    "legacy_key",
    [
        "usage_proxy",
        "usage_proxy_advertised_base_url",
        "usage_proxy_bind_host",
        "usage_proxy_port",
        "usage_proxy_url",
    ],
)
def test_usage_tracking_mapping_rejects_legacy_usage_proxy_keys(legacy_key):
    """Guards PR #587: legacy proxy config keys fail instead of being ignored."""
    from benchflow.usage_tracking import UsageTrackingConfig

    with pytest.raises(ValueError, match=f"{legacy_key} is no longer supported"):
        UsageTrackingConfig.from_mapping(
            {
                "usage_tracking": "required",
                legacy_key: {
                    "ignored": "value",
                },
            }
        )


@pytest.mark.asyncio
async def test_completed_eval_resume_skips_usage_preflight(tmp_path, monkeypatch):
    """Guards PR #568: completed resumes should not require a live LiteLLM preflight."""
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
