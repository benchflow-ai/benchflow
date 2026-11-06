from __future__ import annotations

import json

import pytest

from benchflow.eval_sharding import (
    EvalShard,
    EvalShardPlan,
    _aggregate_result,
    _config_payload,
    _worker_payload_artifact,
    plan_eval_shards,
)
from benchflow.eval_worker import _evaluation_config
from benchflow.evaluation import EvaluationConfig
from benchflow.loop_strategies import LoopStrategySpec


def test_plan_eval_shards_caps_worker_concurrency() -> None:
    """Guards the fix from PR #567 against one process owning all Daytona sessions."""
    plan = plan_eval_shards(
        [f"task-{i}" for i in range(5)],
        total_concurrency=5,
        worker_concurrency=2,
    )

    assert [shard.concurrency for shard in plan.shards] == [2, 2, 1]
    assert sum(shard.concurrency for shard in plan.shards) == 5
    assert max(shard.concurrency for shard in plan.shards) == 2


def test_plan_eval_shards_assigns_each_task_once() -> None:
    """Guards the fix from PR #567 against duplicate or dropped shard tasks."""
    tasks = [f"task-{i}" for i in range(9)]

    plan = plan_eval_shards(tasks, total_concurrency=6, worker_concurrency=2)

    assigned = [task for shard in plan.shards for task in shard.task_names]
    assert sorted(assigned) == sorted(tasks)
    assert len(assigned) == len(set(assigned))


def test_plan_eval_shards_rejects_invalid_concurrency() -> None:
    """Guards the fix from PR #567 against silently creating unusable workers."""
    with pytest.raises(ValueError, match="total_concurrency"):
        plan_eval_shards(["task"], total_concurrency=0, worker_concurrency=1)

    with pytest.raises(ValueError, match="worker_concurrency"):
        plan_eval_shards(["task"], total_concurrency=1, worker_concurrency=0)


def test_worker_payload_round_trips_loop_strategy() -> None:
    config = EvaluationConfig(loop_strategy="verify-retry:k=2,feedback=raw")
    shard = EvalShard(index=0, task_names=("task-a",), concurrency=1)

    payload = _config_payload(config, shard=shard)
    restored = _evaluation_config(json.loads(json.dumps(payload)))

    assert restored.loop_strategy == config.loop_strategy
    assert restored.loop_strategy == LoopStrategySpec(
        "verify-retry", {"k": 2, "feedback": "raw"}
    )


def test_worker_payload_without_loop_strategy_stays_none() -> None:
    config = EvaluationConfig()
    shard = EvalShard(index=0, task_names=("task-a",), concurrency=1)

    payload = _config_payload(config, shard=shard)

    assert payload["loop_strategy"] is None
    assert _evaluation_config(json.loads(json.dumps(payload))).loop_strategy is None


def test_worker_payload_artifact_redacts_agent_env_secrets() -> None:
    config = EvaluationConfig(
        agent_env={
            "OPENAI_API_KEY": "sk-secret",
            "BENCHFLOW_PROVIDER_API_KEY": "provider-secret",
            "NORMAL_VAR": "keep-me",
            "OPENAI_BASE_URL": "http://127.0.0.1:30000/v1",
        }
    )
    shard = EvalShard(index=0, task_names=("task-a",), concurrency=1)
    raw = {
        "tasks_dir": "/tasks",
        "jobs_dir": "/jobs",
        "result_path": "/result.json",
        "config": _config_payload(config, shard=shard),
    }

    artifact = _worker_payload_artifact(raw)
    artifact_text = json.dumps(artifact)

    assert raw["config"]["agent_env"]["OPENAI_API_KEY"] == "sk-secret"
    assert "sk-secret" not in artifact_text
    assert "provider-secret" not in artifact_text
    assert artifact["config"]["agent_env"] == {
        "NORMAL_VAR": "keep-me",
        "OPENAI_BASE_URL": "http://127.0.0.1:30000/v1",
    }
    assert artifact["config"]["agent_env_keys"] == [
        "BENCHFLOW_PROVIDER_API_KEY",
        "NORMAL_VAR",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ]


def test_worker_payload_rejects_unparsed_loop_strategy() -> None:
    """A spec string here means EvaluationConfig validation was bypassed —
    fail loudly instead of silently dropping the strategy from the payload."""
    config = EvaluationConfig()
    config.loop_strategy = "verify-retry:k=2"  # bypasses __post_init__ parsing
    shard = EvalShard(index=0, task_names=("task-a",), concurrency=1)

    with pytest.raises(TypeError, match="LoopStrategySpec"):
        _config_payload(config, shard=shard)


def test_worker_payload_round_trips_resolved_environment_manifest() -> None:
    """Guards against the PR #790 regression where sharded workers lost the S axis.

    PR #790 added ``--state`` but the sharded payload serialized only a manifest
    *path* (``request.environment_manifest``), which is ``None`` for a ``--state``
    run — so every worker booted with no Environment plane, and an inline tool
    subset from ``resolve_state`` was dropped even when a path was present. The
    fix serializes the already-resolved manifest object. This round-trips it
    through the JSON payload boundary (exactly what the worker subprocess reads)
    and asserts the filtered service subset survives.
    """
    from benchflow.environment.manifest import EnvironmentManifest, ServiceSpec

    full = EnvironmentManifest(
        name="clawsbench",
        base_image="clawsbench/base:latest",
        owns_lifecycle=False,
        services=[
            ServiceSpec(name="gmail", command="run-gmail", port=8001),
            ServiceSpec(name="slack", command="run-slack", port=8002),
        ],
    )
    # Mirror resolve_state's inline-JSON tool subset: keep only gmail.
    subset = full.model_copy(update={"services": full.services[:1]})
    config = EvaluationConfig(environment_manifest=subset)
    shard = EvalShard(index=0, task_names=("task-a",), concurrency=1)

    payload = _config_payload(config, shard=shard)
    assert payload["environment_manifest"] is not None
    # The path-only key is gone: the object is the single source of truth.
    assert "environment_manifest_path" not in payload

    restored = _evaluation_config(json.loads(json.dumps(payload)))
    assert restored.environment_manifest == subset
    assert [s.name for s in restored.environment_manifest.services] == ["gmail"]


def test_worker_payload_environment_manifest_none_stays_none() -> None:
    """A run with no S-axis binding must serialize a null manifest, not crash."""
    payload = _config_payload(
        EvaluationConfig(),
        shard=EvalShard(index=0, task_names=("task-a",), concurrency=1),
    )
    assert payload["environment_manifest"] is None
    restored = _evaluation_config(json.loads(json.dumps(payload)))
    assert restored.environment_manifest is None


def test_worker_sharded_summary_includes_numeric_score_ratios(tmp_path) -> None:
    """Guards the fix from PR #778 against sharded summary schema drift."""
    plan = EvalShardPlan(
        total_concurrency=3,
        worker_concurrency=2,
        shards=(
            EvalShard(index=0, task_names=("task-a", "task-c"), concurrency=2),
            EvalShard(index=1, task_names=("task-b",), concurrency=1),
        ),
    )

    result = _aggregate_result(
        jobs_dir=tmp_path,
        config=EvaluationConfig(),
        plan=plan,
        shard_results=[
            {
                "total": 2,
                "passed": 1,
                "failed": 0,
                "errored": 1,
                "verifier_errored": 0,
            },
            {
                "total": 1,
                "passed": 0,
                "failed": 1,
                "errored": 0,
                "verifier_errored": 0,
            },
        ],
        elapsed_sec=1.25,
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    advertised_summary = json.loads(
        (tmp_path / "worker-sharded" / "summary.json").read_text()
    )
    assert result.score == pytest.approx(1 / 3)
    assert summary["score_ratio"] == pytest.approx(1 / 3)
    assert summary["score_excl_errors_ratio"] == pytest.approx(1 / 2)
    assert advertised_summary == summary


def test_a_run_check_that_failed_in_a_worker_fails_the_sharded_run(tmp_path) -> None:
    """A worker's run checks reach the parent: one that failed fails the run."""
    from types import SimpleNamespace

    import typer

    from benchflow.cli.main import _finish_run

    plan = EvalShardPlan(
        total_concurrency=2,
        worker_concurrency=1,
        shards=(
            EvalShard(index=0, task_names=("task-a",), concurrency=1),
            EvalShard(index=1, task_names=("task-b",), concurrency=1),
        ),
    )
    failure = "max_output_tokens: requested 128000 (from --agent-env); ..."
    result = _aggregate_result(
        jobs_dir=tmp_path,
        config=EvaluationConfig(),
        plan=plan,
        shard_results=[
            {
                "job_name": "shard-a",
                "total": 1,
                "passed": 0,
                "failed": 1,
                "errored": 0,
                "verifier_errored": 0,
                "run_checks": {"failed": True, "failures": [failure], "warnings": []},
            },
            {"job_name": "shard-b", "total": 1, "passed": 1, "failed": 0},
        ],
        elapsed_sec=1.0,
    )
    checks = result.run_checks
    # A worker's lines go by its shard's name, as its folder does.
    assert (checks["failed"], checks["failures"], checks["warnings"]) == (
        True,
        [f"shard-000: {failure}"],
        [],
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["run_checks"] == result.run_checks
    request = SimpleNamespace(fail_under=None, fail_on=None, summary_out=None)
    with pytest.raises(typer.Exit) as stop:
        _finish_run(request, result, tmp_path)
    assert stop.value.exit_code == 1


async def test_a_sharded_run_is_refused_only_as_the_whole_job_would_be(tmp_path):
    """Guards the run checks merged at 47a52fdf7: 10 tasks at --concurrency
    10 --worker-concurrency 2 run as five workers of two tasks. One trial cut
    at the cap is 10% of the job, a warning, but its worker saw 1 of 2
    (50%), refused itself, and the parent failed the run. The parent judges
    the workers' figures together, as the whole job would be judged."""
    from tests.test_run_checks import _job, _proxy_call, _summary, _trials_that

    names = [f"task-{i:03d}" for i in range(10)]
    plan = plan_eval_shards(names, total_concurrency=10, worker_concurrency=2)
    assert [len(s.task_names) for s in plan.shards] == [2, 2, 2, 2, 2]

    def calls(name):
        cut = name == "task-000"
        return [_proxy_call("tool_calls"), _proxy_call("length" if cut else "stop")]

    whole = _job(tmp_path / "whole", 10)
    whole._run_single_task = _trials_that(whole, calls)
    await whole.run()
    assert _summary(whole)["run_checks"]["warnings"] == [
        "truncation: 1 of 10 trials with a known ending were cut at the output cap "
        "(10.0%), at or above 10%"
    ]

    shard_results = []
    for shard in plan.shards:
        worker = _job(
            tmp_path / f"shard-{shard.index}", 10, include_tasks=set(shard.task_names)
        )
        worker._run_single_task = _trials_that(worker, calls)
        result = await worker.run()
        shard_results.append(
            {
                "job_name": result.job_name,
                "total": result.total,
                "run_checks": result.run_checks,
            }
        )
    parent = _aggregate_result(
        jobs_dir=tmp_path / "parent",
        config=EvaluationConfig(agent="claude-agent-acp"),
        plan=plan,
        shard_results=shard_results,
        elapsed_sec=1.0,
    )
    checks = parent.run_checks
    assert (checks["failed"], checks["truncation"]["known"]) == (False, 10)
    assert checks["warnings"] == _summary(whole)["run_checks"]["warnings"]


async def test_a_sharded_run_reports_the_trials_a_worker_did_not_start(tmp_path):
    """Guards the run checks merged at 47a52fdf7: a worker whose run check
    stopped it keeps trials from starting, but its result payload had no
    not_run and the parent kept only its failure lines, prefixed with the
    worker's timestamped job name. The parent's result and summary.json now
    count the trials that never ran, and name each worker by its shard."""
    from benchflow.eval_worker import _result_payload
    from tests.test_run_checks import _job, _proxy_call, _trials_that

    worker = _job(tmp_path / "shard-0", 10)
    worker._run_single_task = _trials_that(
        worker, lambda name: [_proxy_call("tool_calls"), _proxy_call("length")]
    )
    result = await worker.run()
    assert result.not_run == 5 and len(result.run_checks["not_started"]) == 5
    plan = plan_eval_shards(
        [f"task-{i:03d}" for i in range(10)], total_concurrency=1, worker_concurrency=1
    )
    parent = _aggregate_result(
        jobs_dir=tmp_path / "parent",
        config=EvaluationConfig(agent="claude-agent-acp"),
        plan=plan,
        shard_results=[_result_payload(result)],
        elapsed_sec=1.0,
    )
    summary = json.loads((tmp_path / "parent" / "summary.json").read_text())
    assert (parent.total, parent.not_run, parent.planned) == (5, 5, 10)
    assert (summary["not_run"], summary["planned"]) == (5, 10)
    checks = summary["run_checks"]
    assert checks["not_started"] == [f"shard-000/task-{i:03d}" for i in range(5, 10)]
    assert checks["failures"] == [
        "truncation: 5 of 5 trials with a known ending were cut at the output cap "
        "(100.0%), at or above 50%"
    ]
    assert checks["reason"].startswith("shard-000: 5 of 5 trials")
