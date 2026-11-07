"""GAP-ORACLE-READABLE through PERF-C's warm pool, on a real Docker daemon.

GAPS-1 uploads a task's solution only for an oracle run (``upload_solution``
from the trial's own agent). The warm pool prepares a waiting trial's sandbox
through the same ``setup()``/``start()``/``install_agent()`` of that trial's
own rollout (``_prestart_steps``), so a prepared sandbox must take
``upload_solution`` from its own trial exactly as a cold start does: no
solution in a non-oracle trial's sandbox, the solution in an oracle's. Merge
wave batch 31 (gaps-1 with perf-pipe). Skipped without a Docker daemon (the
gate container has none; it runs on a Docker VM).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from tests.test_warm_pool_docker import _docker_ready, _immutable_image

pytestmark = [
    pytest.mark.skipif(not _docker_ready(), reason="a Docker daemon is required"),
]
# Whole trials on real Docker: the exec channel and the side-by-side steps
# before the verifier stay on, as in a run (tests/conftest.py).
DOCKER_RUN_DEFAULTS = True

# test.sh: reward 1 when the sandbox holds the solution (``present``) or when
# it holds none (``absent``); the oracle's own solve.sh does nothing.
_TEST = """#!/bin/sh
if [ -e /solution ] || [ -e /oracle ]; then has=present; else has=absent; fi
if [ "$has" = "{want}" ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
"""


def _tasks(root: Path, prefix: str, count: int, want: str) -> Path:
    image = _immutable_image()
    for i in range(count):
        task = root / f"{prefix}-{i}"
        (task / "tests").mkdir(parents=True)
        (task / "solution").mkdir()
        (task / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 120\n[agent]\n'
            f'timeout_sec = 60\n[environment]\ndocker_image = "{image}"\n'
        )
        (task / "instruction.md").write_text("Do nothing.\n")
        (task / "solution" / "solve.sh").write_text("#!/bin/sh\ntrue\n")
        (task / "tests" / "test.sh").write_text(_TEST.format(want=want))
    return root


async def _run(tmp_path: Path, monkeypatch, *, agent: str, want: str):
    monkeypatch.delenv("BENCHFLOW_WARM_POOL", raising=False)
    monkeypatch.setenv("BENCHFLOW_PREPULL", "off")
    prefix = f"wps{uuid.uuid4().hex[:6]}"
    job = Evaluation(
        tasks_dir=_tasks(tmp_path / "tasks", prefix, 5, want),
        jobs_dir=tmp_path / "jobs",
        job_name="pool",
        config=EvaluationConfig(
            agent=agent,
            environment="docker",
            concurrency=2,
            retry=RetryConfig(max_retries=0),
            integrity="off",
        ),
        preflight=False,
    )
    result = await job.run()
    timings = [
        json.loads(p.read_text()).get("timing") or {}
        for p in (tmp_path / "jobs" / "pool").glob("*/result.json")
    ]
    return result, timings


@pytest.mark.asyncio
async def test_a_warm_prepared_non_oracle_trial_has_no_solution(tmp_path, monkeypatch):
    result, timings = await _run(tmp_path, monkeypatch, agent="nop", want="absent")
    assert result.total == 5 and result.errored == 0
    # Every trial, the warm-prepared ones included, found no solution.
    assert result.passed == 5
    assert any("warm_pool_wait" in t for t in timings), "the pool was never used"


@pytest.mark.asyncio
async def test_a_warm_prepared_oracle_trial_has_its_solution(tmp_path, monkeypatch):
    """Control: the oracle's own trial, prepared warm, still gets it."""
    result, timings = await _run(tmp_path, monkeypatch, agent="oracle", want="present")
    assert result.total == 5 and result.errored == 0
    assert result.passed == 5
    assert any("warm_pool_wait" in t for t in timings), "the pool was never used"
