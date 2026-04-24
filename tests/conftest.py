"""Test fixtures."""

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

REF_TASKS = REPO_ROOT / ".ref" / "harbor" / "examples" / "tasks"


@pytest.fixture
def hello_world_task_dir() -> Path:
    path = REF_TASKS / "hello-world"
    if not path.exists():
        pytest.skip("Harbor reference tasks not available")
    return path


@pytest.fixture
def build_result_json(tmp_path):
    """Factory: call SDK._build_result with 14-field defaults and return parsed result.json.

    Accepts `trajectory_source` override so callers that care about that field can
    set it; otherwise defaults to None. The `trial_dir` used is `tmp_path / "trial"`
    (created if missing). Any override via kwargs is merged on top of the defaults.
    """

    def _build(**overrides) -> dict:
        from benchflow.sdk import SDK

        trial_dir = tmp_path / "trial"
        trial_dir.mkdir(exist_ok=True)
        defaults = dict(
            task_name="t1",
            trial_name="trial-1",
            agent="test",
            agent_name="",
            model="",
            n_tool_calls=0,
            prompts=["x"],
            error=None,
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            trajectory_source=None,
            rewards={"reward": 1.0},
            started_at=datetime.now(),
            timing={},
        )
        defaults.update(overrides)
        SDK._build_result(trial_dir, **defaults)
        return json.loads((trial_dir / "result.json").read_text())

    return _build


@pytest.fixture
def job_factory(tmp_path):
    """Create a Job with n task directories and a mocked SDK."""
    from benchflow.job import Job, JobConfig, RetryConfig

    def _make(n_tasks=1, max_retries=0):
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir(exist_ok=True)
        for i in range(n_tasks):
            td = tasks_dir / f"task-{i}"
            td.mkdir(exist_ok=True)
            (td / "task.toml").write_text(
                'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
                "[agent]\ntimeout_sec = 60\n[environment]\n"
            )
        cfg = JobConfig(retry=RetryConfig(max_retries=max_retries))
        job = Job(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=cfg)
        return job, tasks_dir

    return _make
