"""summary.json names the cause of a fresh run's integration failure.

A batch whose trial failed with ``agent integration failure
[agent_model]`` wrote the cause into the trial's result.json, but summary.json
counted it as ``integration_failures.by_cause == {"unknown": 1}``: the summary
of a fresh run is built from in-memory rows that carry no
``integration_failure_info``, while a resumed job (rows read from disk) got
the real cause. The evaluation already reads each fresh trial's result.json
back for ``timing``; it now carries the integration record too.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RolloutResult

ERROR = "agent integration failure [agent_auth]: invalid api key"


@pytest.mark.asyncio
async def test_fresh_run_summary_counts_the_recorded_cause(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    (tasks / "task-0").mkdir(parents=True)
    (tasks / "task-0" / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n[agent]\ntimeout_sec = 60\n[environment]\n'
    )
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="j",
        config=EvaluationConfig(concurrency=1, retry=RetryConfig(max_retries=0)),
    )
    trial = tmp_path / "jobs" / "j" / "task-0__x"

    async def run_one(*_a, **_k) -> RolloutResult:
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "task-0",
                    "rollout_name": "task-0__x",
                    "rewards": None,
                    "error": ERROR,
                    "error_category": "agent_integration",
                    "integration_failure_info": {"cause": "agent_auth"},
                }
            )
        )
        return RolloutResult(
            task_name="task-0",
            rollout_name="task-0__x",
            error=ERROR,
            error_category="agent_integration",
        )

    job._run_single_task = AsyncMock(side_effect=run_one)

    await job.run()

    summary = json.loads((tmp_path / "jobs" / "j" / "summary.json").read_text())
    assert summary["integration_failures"] == {
        "total": 1,
        "by_cause": {"agent_auth": 1},
    }
