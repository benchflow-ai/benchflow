"""DX-e round 3: the review of 43d41593.

R2-M1: a sequential-shared job reuses terminal errors on resume, so the trial
folder the job writes for a task that failed before its own folder existed
(a task.toml typo) kept that task failed after the user fixed it. R2-L1: that
folder's config.json lacked the rollout's loop and digest, so resume set it
aside with a false reason. R2-L2: the BusyBox listing compared whole seconds
strictly and said nothing about having no ctime.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.loop_strategies import loop_block
from benchflow.models import RolloutResult

AGENT = "claude-agent-acp"


def _tasks(tmp_path: Path) -> Path:
    tasks = tmp_path / "tasks"
    for name in ("ok", "typo"):
        (tasks / name).mkdir(parents=True, exist_ok=True)
        (tasks / name / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    return tasks


def _job(tmp_path: Path, *, fixed: bool, **config):
    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig

    job = Evaluation(
        tasks_dir=_tasks(tmp_path),
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent=AGENT, concurrency=1, retry=RetryConfig(max_retries=0), **config
        ),
        job_name="j",
        preflight=False,
    )

    async def fake_run(task_path, _cfg):
        if task_path.name == "typo" and not fixed:
            # Refused before any rollout folder exists.
            return RolloutResult(
                task_name="typo",
                error="TaskConfigKeyError: unknown task.toml key 'agent.timout_sec'",
                agent=AGENT,
            )
        trial = job.job_dir / f"{task_path.name}__aaaa1111"
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_path.name,
                    "rollout_name": trial.name,
                    "rewards": {"reward": 1.0},
                    "error": None,
                    "verifier_error": None,
                    "agent": AGENT,
                }
            )
        )
        # What a real trial records: its loop among the rest.
        (trial / "config.json").write_text(
            json.dumps({"agent": AGENT, "loop": loop_block(job._config.loop_strategy)})
        )
        return RolloutResult(
            task_name=task_path.name,
            rollout_name=trial.name,
            rewards={"reward": 1.0},
            agent=AGENT,
            rollout_dir=trial,
        )

    job._run_single_task = AsyncMock(side_effect=fake_run)
    return job


def _ran(job) -> list[str]:
    return [call.args[0].name for call in job._run_single_task.call_args_list]


def test_a_fixed_task_runs_on_resume_of_a_sequential_shared_job(tmp_path) -> None:
    first = _job(tmp_path, fixed=False, job_mode="sequential-shared")
    asyncio.run(first.run())
    assert sorted(_ran(first)) == ["ok", "typo"]

    # The user fixes the typo and runs the same command again.
    again = _job(tmp_path, fixed=True, job_mode="sequential-shared")
    result = asyncio.run(again.run())

    assert _ran(again) == ["typo"]
    assert result.passed == 2


def test_a_fixed_task_runs_on_resume_of_a_parallel_job(tmp_path) -> None:
    asyncio.run(_job(tmp_path, fixed=False).run())
    again = _job(tmp_path, fixed=True)
    asyncio.run(again.run())
    assert _ran(again) == ["typo"]


def test_the_written_trial_records_what_the_rollout_records(tmp_path) -> None:
    from benchflow._utils.task_authoring import task_digest

    job = _job(tmp_path, fixed=False)
    asyncio.run(job.run())

    folder = next(p for p in job.job_dir.iterdir() if p.name.startswith("typo__"))
    result = json.loads((folder / "result.json").read_text())
    config = json.loads((folder / "config.json").read_text())
    assert result["no_rollout_folder"] is True
    assert config["task_digest"] == task_digest(tmp_path / "tasks" / "typo")
    assert config["loop"] == {"strategy": "single-shot"}
    assert config["agent"] == AGENT


def test_a_resume_with_a_loop_strategy_sets_nothing_aside(tmp_path, caplog) -> None:
    loop = "verify-retry:k=2,feedback=raw"
    asyncio.run(_job(tmp_path, fixed=False, loop_strategy=loop).run())

    again = _job(tmp_path, fixed=True, loop_strategy=loop)
    with caplog.at_level(logging.WARNING):
        asyncio.run(again.run())

    assert "Setting aside" not in caplog.text
    assert not (again.job_dir / ".set-aside").exists()
    assert _ran(again) == ["typo"]


# --- R2-L2: the BusyBox listing --------------------------------------------------


def test_the_busybox_stamp_is_a_second_early_and_the_basis_is_recorded() -> None:
    from benchflow.rollout._separate_verifier import (
        _outside_writes_find,
        _outside_writes_report,
    )

    script = _outside_writes_find("/", 1792170000.5, ["/app"], [])
    assert 'touch -d @1792169999 "$ref"' in script
    assert "basis=mtime\n" in script and "basis=mtime+ctime\n" in script

    busybox = _outside_writes_report("/", "basis=mtime\0/root/notes.txt\0")
    assert busybox["basis"] == "mtime" and busybox["count"] == 1
    gnu = _outside_writes_report("/", "basis=mtime+ctime\0")
    assert gnu["basis"] == "mtime+ctime" and gnu["count"] == 0
    assert not gnu["truncated"]


@pytest.mark.asyncio
async def test_the_gnu_listing_records_its_basis(tmp_path) -> None:
    import time

    from benchflow.rollout._separate_verifier import scan_outside_workspace_writes
    from tests.test_upgrade_shell_only_dx import _ShellOnly

    root = tmp_path / "box"
    (root / "srv").mkdir(parents=True)
    since = time.time()
    time.sleep(1.1)
    (root / "srv" / "fit.csv").write_text("new\n")

    report = await scan_outside_workspace_writes(
        _ShellOnly(), since=since, keep=["/app"], root=str(root)
    )
    assert report is not None and report["basis"] == "mtime+ctime"
    assert report["sample"] == ["/srv/fit.csv"]
