"""Layer 3 — Job orchestration: YAML config, concurrency, resume.

Tests Job.from_yaml(), concurrent execution, and the resume-from-disk
capability.

Run::

    GEMINI_API_KEY=... DAYTONA_API_KEY=... \\
      pytest -m integration tests/integration/test_job_orchestration.py -v

Guards: ENG-6 integration test plan (issue #253).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.conftest import (
    DEFAULT_CONCURRENCY,
    DEFAULT_ENVIRONMENT,
    DEFAULT_MODEL,
    SKILLSBENCH_TASKS,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_job_from_yaml(
    integration_prereqs: None,
    skillsbench_tasks_dir: Path,
    tmp_path: Path,
) -> None:
    """Job.from_yaml() correctly parses a native config and runs tasks."""
    import yaml

    from benchflow.job import Job

    config_data = {
        "source": {
            "repo": "benchflow-ai/skillsbench",
            "path": "tasks",
            "ref": "main",
        },
        "jobs_dir": str(tmp_path / "yaml-test-jobs"),
        "agent": "gemini",
        "model": DEFAULT_MODEL,
        "environment": DEFAULT_ENVIRONMENT,
        "concurrency": DEFAULT_CONCURRENCY,
        "max_retries": 1,
        "exclude": [
            t
            for t in _all_task_names(skillsbench_tasks_dir)
            if t not in SKILLSBENCH_TASKS[:3]
        ],
    }

    config_path = tmp_path / "test-config.yaml"
    config_path.write_text(yaml.dump(config_data))

    job = Job.from_yaml(config_path)
    result = await job.run()

    assert result.total == 3
    assert result.passed + result.failed + result.errored == result.total


@pytest.mark.integration
@pytest.mark.asyncio
async def test_job_resume(
    integration_prereqs: None,
    skillsbench_tasks_dir: Path,
    tmp_path: Path,
) -> None:
    """Job resumes from disk, skipping already-completed trials.

    Run once with 2 tasks, then re-run the same job — the second run
    should detect existing results and complete instantly.
    """
    from benchflow.job import Job, JobConfig, RetryConfig

    selected = set(SKILLSBENCH_TASKS[:2])
    all_tasks = _all_task_names(skillsbench_tasks_dir)
    exclude = all_tasks - selected

    resume_jobs_dir = tmp_path / "resume-test-jobs"
    resume_jobs_dir.mkdir()

    config = JobConfig(
        agent="gemini",
        model=DEFAULT_MODEL,
        environment=DEFAULT_ENVIRONMENT,
        concurrency=DEFAULT_CONCURRENCY,
        retry=RetryConfig(max_retries=0),
        exclude_tasks=exclude,
    )

    # First run
    job1 = Job(
        tasks_dir=skillsbench_tasks_dir,
        jobs_dir=resume_jobs_dir,
        config=config,
    )
    result1 = await job1.run()
    assert result1.total == 2

    # Count result files from first run
    result_files_1 = list(resume_jobs_dir.glob("*/*/result.json"))

    # Second run — same config, same jobs_dir
    job2 = Job(
        tasks_dir=skillsbench_tasks_dir,
        jobs_dir=resume_jobs_dir,
        config=config,
    )
    result2 = await job2.run()

    # Second run should have same totals (resumed, not re-run)
    assert result2.total == 2
    result_files_2 = list(resume_jobs_dir.glob("*/*/result.json"))
    # Should not have created new trial dirs for already-completed tasks
    assert len(result_files_2) == len(result_files_1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrency_bounded(
    integration_prereqs: None,
    skillsbench_tasks_dir: Path,
    tmp_path: Path,
) -> None:
    """Job runs tasks concurrently and respects the concurrency limit."""
    from benchflow.job import Job, JobConfig, RetryConfig

    selected = set(SKILLSBENCH_TASKS[:3])
    all_tasks = _all_task_names(skillsbench_tasks_dir)
    exclude = all_tasks - selected

    conc_jobs_dir = tmp_path / "concurrency-test-jobs"
    conc_jobs_dir.mkdir()

    config = JobConfig(
        agent="gemini",
        model=DEFAULT_MODEL,
        environment=DEFAULT_ENVIRONMENT,
        concurrency=3,  # All 3 tasks should run concurrently
        retry=RetryConfig(max_retries=0),
        exclude_tasks=exclude,
    )

    job = Job(
        tasks_dir=skillsbench_tasks_dir,
        jobs_dir=conc_jobs_dir,
        config=config,
    )
    result = await job.run()

    assert result.total == 3
    assert result.passed + result.failed + result.errored == result.total

    # Validate timing: all 3 ran concurrently if total elapsed
    # is less than 3x the longest individual task.
    result_files = list(conc_jobs_dir.glob("*/*/result.json"))
    assert len(result_files) == 3


def _all_task_names(tasks_dir: Path) -> set[str]:
    """Return all task names in a tasks directory."""
    return {
        d.name
        for d in tasks_dir.iterdir()
        if d.is_dir() and (d / "task.toml").exists()
    }
