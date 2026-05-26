from __future__ import annotations

import pytest

from benchflow.eval_sharding import plan_eval_shards


def test_plan_eval_shards_caps_worker_concurrency() -> None:
    """Guards bry/eval-worker-shards against one process owning all Daytona sessions."""
    plan = plan_eval_shards(
        [f"task-{i}" for i in range(5)],
        total_concurrency=5,
        worker_concurrency=2,
    )

    assert [shard.concurrency for shard in plan.shards] == [2, 2, 1]
    assert sum(shard.concurrency for shard in plan.shards) == 5
    assert max(shard.concurrency for shard in plan.shards) == 2


def test_plan_eval_shards_assigns_each_task_once() -> None:
    """Guards bry/eval-worker-shards against duplicate or dropped shard tasks."""
    tasks = [f"task-{i}" for i in range(9)]

    plan = plan_eval_shards(tasks, total_concurrency=6, worker_concurrency=2)

    assigned = [task for shard in plan.shards for task in shard.task_names]
    assert sorted(assigned) == sorted(tasks)
    assert len(assigned) == len(set(assigned))


def test_plan_eval_shards_rejects_invalid_concurrency() -> None:
    """Guards bry/eval-worker-shards against silently creating unusable workers."""
    with pytest.raises(ValueError, match="total_concurrency"):
        plan_eval_shards(["task"], total_concurrency=0, worker_concurrency=1)

    with pytest.raises(ValueError, match="worker_concurrency"):
        plan_eval_shards(["task"], total_concurrency=1, worker_concurrency=0)
