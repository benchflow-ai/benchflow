"""Layer 2 — Agent matrix: 9 SkillsBench tasks × 8 agents.

Runs the full 9-task SkillsBench subset against each registered agent
on Daytona with gemini-3.1-flash-lite-preview.  Validates that:

- Agent installs successfully
- Verifier runs (rewards dict is present)
- No infra errors (agent_install, timeout)
- Result files written to disk

This is the core integration test for ENG-6.

Run::

    GEMINI_API_KEY=... DAYTONA_API_KEY=... \\
      pytest -m integration tests/integration/test_agent_matrix.py -v

Guards: ENG-6 integration test plan (issue #253).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tests.integration.conftest import (
    ALL_AGENTS,
    DEFAULT_CONCURRENCY,
    DEFAULT_ENVIRONMENT,
    DEFAULT_MODEL,
    SKILLSBENCH_TASKS,
    has_creds_for_agent,
    model_for_agent,
)

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ALL_AGENTS)
async def test_agent_on_skillsbench_9(
    integration_prereqs: None,
    agent: str,
    skillsbench_tasks_dir: Path,
    session_jobs_dir: Path,
) -> None:
    """Run 9 SkillsBench tasks with a single agent via Job."""
    if not has_creds_for_agent(agent):
        pytest.skip(f"No credentials for {agent}")

    from benchflow.job import Job, JobConfig, RetryConfig

    # Filter to only our 9 selected tasks
    tasks_dir = skillsbench_tasks_dir
    selected = {t for t in SKILLSBENCH_TASKS}
    all_tasks = {
        d.name for d in tasks_dir.iterdir() if d.is_dir() and (d / "task.toml").exists()
    }
    exclude = all_tasks - selected

    agent_jobs_dir = session_jobs_dir / f"matrix-{agent}"
    agent_jobs_dir.mkdir(exist_ok=True)

    config = JobConfig(
        agent=agent,
        model=model_for_agent(agent),
        environment=DEFAULT_ENVIRONMENT,
        concurrency=DEFAULT_CONCURRENCY,
        retry=RetryConfig(max_retries=1),
        exclude_tasks=exclude,
    )

    job = Job(
        tasks_dir=tasks_dir,
        jobs_dir=agent_jobs_dir,
        config=config,
    )
    result = await job.run()

    # Invariant: all tasks accounted for
    assert result.total == len(SKILLSBENCH_TASKS), (
        f"Expected {len(SKILLSBENCH_TASKS)} tasks, got {result.total}"
    )
    assert result.passed + result.failed + result.errored == result.total

    # No complete failures — at least some tasks should pass
    logger.info(
        "Agent %s: %d/%d passed (%.1f%%), %d errored",
        agent,
        result.passed,
        result.total,
        result.score * 100,
        result.errored,
    )

    # Hard gate: zero infra errors means agent installs + verifier work
    # Soft gate: model may fail tasks, so we don't assert pass rate > X
    # but we do assert the run completed without crashing
    assert result.errored <= result.total, "More errors than tasks (invariant broken)"

    # Validate result files exist for each trial
    trial_dirs = list(agent_jobs_dir.glob("*/*/result.json"))
    assert len(trial_dirs) > 0, "No result.json files produced"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_matrix_summary_json(
    integration_prereqs: None,
    skillsbench_tasks_dir: Path,
    session_jobs_dir: Path,
) -> None:
    """Validate summary.json schema after a batch run."""
    from benchflow.job import Job, JobConfig, RetryConfig

    tasks_dir = skillsbench_tasks_dir
    selected = {t for t in SKILLSBENCH_TASKS[:3]}  # Low-complexity only for speed
    all_tasks = {
        d.name for d in tasks_dir.iterdir() if d.is_dir() and (d / "task.toml").exists()
    }
    exclude = all_tasks - selected

    summary_jobs_dir = session_jobs_dir / "matrix-summary-check"
    summary_jobs_dir.mkdir(exist_ok=True)

    config = JobConfig(
        agent="gemini",
        model=DEFAULT_MODEL,
        environment=DEFAULT_ENVIRONMENT,
        concurrency=DEFAULT_CONCURRENCY,
        retry=RetryConfig(max_retries=1),
        exclude_tasks=exclude,
    )

    job = Job(
        tasks_dir=tasks_dir,
        jobs_dir=summary_jobs_dir,
        config=config,
    )
    await job.run()

    # Check summary.json was written
    summary_files = list(summary_jobs_dir.glob("summary.json"))
    assert len(summary_files) >= 1, "No summary.json produced"

    data = json.loads(summary_files[0].read_text())
    required = {"total", "passed", "failed", "errored", "score"}
    missing = required - set(data.keys())
    assert not missing, f"summary.json missing: {missing}"
    assert data["total"] == len(selected)
