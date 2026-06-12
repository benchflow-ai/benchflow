from __future__ import annotations

import json

import pytest

from benchflow.eval_sharding import EvalShard, _config_payload, plan_eval_shards
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

    payload = _config_payload(config, shard=shard, environment_manifest_path=None)
    restored = _evaluation_config(json.loads(json.dumps(payload)))

    assert restored.loop_strategy == config.loop_strategy
    assert restored.loop_strategy == LoopStrategySpec(
        "verify-retry", {"k": 2, "feedback": "raw"}
    )


def test_worker_payload_without_loop_strategy_stays_none() -> None:
    config = EvaluationConfig()
    shard = EvalShard(index=0, task_names=("task-a",), concurrency=1)

    payload = _config_payload(config, shard=shard, environment_manifest_path=None)

    assert payload["loop_strategy"] is None
    assert _evaluation_config(json.loads(json.dumps(payload))).loop_strategy is None


def test_worker_payload_rejects_unparsed_loop_strategy() -> None:
    """A spec string here means EvaluationConfig validation was bypassed —
    fail loudly instead of silently dropping the strategy from the payload."""
    config = EvaluationConfig()
    config.loop_strategy = "verify-retry:k=2"  # bypasses __post_init__ parsing
    shard = EvalShard(index=0, task_names=("task-a",), concurrency=1)

    with pytest.raises(TypeError, match="LoopStrategySpec"):
        _config_payload(config, shard=shard, environment_manifest_path=None)
