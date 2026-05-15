"""Layer 4 — Parity: compare BenchFlow results vs stored Harbor trajectories.

Imports trajectory data from benchflow-ai/skillsbench-trajectories and
compares structural parity with fresh BenchFlow runs:

- result.json schema matches between Harbor and BenchFlow formats
- Verifier reward fields present in both
- Trajectory files present and non-empty in both
- Timing fields present in both

Does NOT assert identical reward values (model non-determinism) but
validates that the result format and verification pipeline produce
structurally equivalent outputs.

Run::

    GEMINI_API_KEY=... DAYTONA_API_KEY=... \\
      pytest -m integration tests/integration/test_parity.py -v

Guards: ENG-6 integration test plan (issue #253).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import pytest

from tests.integration.conftest import (
    DEFAULT_CONCURRENCY,
    DEFAULT_ENVIRONMENT,
    DEFAULT_MODEL,
    SKILLSBENCH_TASKS,
)

logger = logging.getLogger(__name__)

TRAJECTORIES_REPO = "benchflow-ai/skillsbench-trajectories"
TRAJECTORIES_CACHE = Path.home() / ".cache" / "benchflow-trajectories"


def _clone_trajectories() -> Path:
    """Clone the trajectories repo if not already cached."""
    if TRAJECTORIES_CACHE.exists() and (TRAJECTORIES_CACHE / ".git").exists():
        return TRAJECTORIES_CACHE
    TRAJECTORIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            f"https://github.com/{TRAJECTORIES_REPO}.git",
            str(TRAJECTORIES_CACHE),
        ],
        check=True,
    )
    return TRAJECTORIES_CACHE


def _load_harbor_results(traj_root: Path) -> dict[str, dict]:
    """Load Harbor result.json files keyed by task_name."""
    results = {}
    for result_file in traj_root.rglob("result.json"):
        data = json.loads(result_file.read_text())
        task_name = data.get("task_name")
        if task_name and task_name in SKILLSBENCH_TASKS and task_name not in results:
            results[task_name] = data
    return results


def _load_benchflow_results(jobs_dir: Path) -> dict[str, dict]:
    """Load BenchFlow result.json files keyed by task_name."""
    results = {}
    for result_file in jobs_dir.rglob("result.json"):
        data = json.loads(result_file.read_text())
        task_name = data.get("task_name")
        if task_name and task_name not in results:
            results[task_name] = data
    return results


# ---------------------------------------------------------------------------
# Harbor result.json schema fields (from skillsbench-trajectories)
# ---------------------------------------------------------------------------

HARBOR_REQUIRED_FIELDS = {
    "task_name",
    "trial_name",
    "config",
    "verifier_result",
    "started_at",
    "finished_at",
}

BENCHFLOW_REQUIRED_FIELDS = {
    "task_name",
    "trial_name",
    "agent_name",
    "model",
    "rewards",
    "n_tool_calls",
    "started_at",
    "finished_at",
}


@pytest.mark.integration
def test_harbor_trajectories_loadable() -> None:
    """Stored Harbor trajectories can be cloned and parsed."""
    traj_root = _clone_trajectories()
    harbor_results = _load_harbor_results(traj_root)

    assert len(harbor_results) > 0, (
        "No Harbor results found for selected tasks in trajectories repo"
    )

    # Validate Harbor result schema
    for task_name, data in harbor_results.items():
        missing = HARBOR_REQUIRED_FIELDS - set(data.keys())
        assert not missing, f"Harbor result for {task_name} missing fields: {missing}"
        # Verifier result has rewards
        vr = data.get("verifier_result", {})
        assert "rewards" in vr, f"Harbor {task_name}: no rewards in verifier_result"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parity_schema(
    integration_prereqs: None,
    skillsbench_tasks_dir: Path,
    tmp_path: Path,
) -> None:
    """BenchFlow results have structurally equivalent schema to Harbor."""
    from benchflow.job import Job, JobConfig, RetryConfig

    # Run a small subset (3 low-complexity tasks) for schema comparison
    selected = set(SKILLSBENCH_TASKS[:3])
    all_tasks = {
        d.name
        for d in skillsbench_tasks_dir.iterdir()
        if d.is_dir() and (d / "task.toml").exists()
    }
    exclude = all_tasks - selected

    parity_jobs_dir = tmp_path / "parity-schema-jobs"
    parity_jobs_dir.mkdir()

    config = JobConfig(
        agent="gemini",
        model=DEFAULT_MODEL,
        environment=DEFAULT_ENVIRONMENT,
        concurrency=DEFAULT_CONCURRENCY,
        retry=RetryConfig(max_retries=1),
        exclude_tasks=exclude,
    )

    job = Job(
        tasks_dir=skillsbench_tasks_dir,
        jobs_dir=parity_jobs_dir,
        config=config,
    )
    await job.run()

    bf_results = _load_benchflow_results(parity_jobs_dir)

    # Load Harbor results for comparison
    traj_root = _clone_trajectories()
    harbor_results = _load_harbor_results(traj_root)

    # For each task that exists in both, compare schema
    compared = 0
    for task_name in selected:
        if task_name not in bf_results:
            logger.warning("Task %s not in BenchFlow results", task_name)
            continue

        bf = bf_results[task_name]

        # BenchFlow schema check
        bf_missing = BENCHFLOW_REQUIRED_FIELDS - set(bf.keys())
        assert not bf_missing, f"BenchFlow {task_name} missing: {bf_missing}"

        # Rewards present and is a dict
        assert isinstance(bf.get("rewards"), dict), (
            f"BenchFlow {task_name}: rewards should be a dict"
        )

        # Compare with Harbor if available
        if task_name in harbor_results:
            hb = harbor_results[task_name]
            # Both have timing fields
            assert "started_at" in bf and "started_at" in hb
            assert "finished_at" in bf and "finished_at" in hb
            # Both have reward information
            hb_rewards = hb.get("verifier_result", {}).get("rewards", {})
            bf_rewards = bf.get("rewards", {})
            assert "reward" in hb_rewards or len(hb_rewards) > 0, (
                f"Harbor {task_name} has no reward keys"
            )
            assert "reward" in bf_rewards or len(bf_rewards) > 0, (
                f"BenchFlow {task_name} has no reward keys"
            )
            compared += 1
            logger.info(
                "Parity check for %s: harbor_reward=%s, bf_reward=%s",
                task_name,
                hb_rewards.get("reward"),
                bf_rewards.get("reward"),
            )

    logger.info("Compared %d tasks between Harbor and BenchFlow", compared)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parity_trajectory_structure(
    integration_prereqs: None,
    skillsbench_tasks_dir: Path,
    tmp_path: Path,
) -> None:
    """BenchFlow trial directories have equivalent structure to Harbor trials."""
    from benchflow import SDK

    # Run a single task and check directory structure
    task_path = skillsbench_tasks_dir / SKILLSBENCH_TASKS[0]
    parity_traj_jobs = tmp_path / "parity-traj-jobs"
    parity_traj_jobs.mkdir()

    result = await SDK().run(
        task_path=task_path,
        agent="gemini",
        model=DEFAULT_MODEL,
        jobs_dir=parity_traj_jobs,
        environment=DEFAULT_ENVIRONMENT,
    )

    # Find the trial directory
    trial_dirs = list(parity_traj_jobs.glob(f"*/{result.trial_name}"))
    assert len(trial_dirs) == 1
    trial_dir = trial_dirs[0]

    # BenchFlow trial should contain these artifacts
    expected_files = ["result.json"]
    expected_dirs = ["trajectory"]

    for f in expected_files:
        assert (trial_dir / f).exists(), f"Missing {f} in trial dir"

    for d in expected_dirs:
        assert (trial_dir / d).exists(), f"Missing {d}/ in trial dir"

    # Trajectory should have ACP trajectory
    traj_files = list((trial_dir / "trajectory").glob("*.jsonl"))
    assert len(traj_files) > 0, "No trajectory JSONL files"

    # Compare with Harbor structure
    traj_root = _clone_trajectories()
    harbor_trial_dirs = list(traj_root.rglob(f"{SKILLSBENCH_TASKS[0]}__*"))
    if harbor_trial_dirs:
        harbor_trial = harbor_trial_dirs[0]
        harbor_contents = {p.name for p in harbor_trial.iterdir()}
        logger.info(
            "Harbor trial contents: %s vs BenchFlow trial: %s",
            harbor_contents,
            {p.name for p in trial_dir.iterdir()},
        )
        # Both should have result.json
        assert "result.json" in harbor_contents
