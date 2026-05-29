"""Test fixtures."""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def docker_daemon_unavailable_reason() -> str | None:
    """Why a Docker-daemon-backed test can't run, or ``None`` if it can.

    The **canonical** docker gate, shared by every daemon-backed test (the
    smoke test and the env-snapshot integration test) so there is one answer,
    not a per-file reinvention. Checks the CLI is present *and* the daemon is
    reachable (3s timeout to kill hangs on a misconfigured ``DOCKER_HOST``) —
    a CLI-only check would let a test run (and fail on a real image build) when
    the daemon is down.

    Pure function — call it inside a fixture/skipif body, never at decorator or
    collection time, so the subprocess fires only when the test is selected.
    """
    if shutil.which("docker") is None:
        return "docker CLI not installed"
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            timeout=3,
            capture_output=True,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"docker daemon unreachable: {e}"
    if result.returncode != 0:
        return "docker daemon unreachable"
    return None


@pytest.fixture
def docker_daemon() -> None:
    """Skip the test unless a Docker daemon is reachable (see the helper above)."""
    reason = docker_daemon_unavailable_reason()
    if reason:
        pytest.skip(reason)

REF_TASKS = REPO_ROOT / ".cache" / "datasets" / "benchflow" / "examples" / "tasks"


@pytest.fixture(autouse=True)
def isolate_local_dotenv(monkeypatch, tmp_path) -> None:
    """Keep developer-machine `.env` secrets out of unit tests."""
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / ".env"))


@pytest.fixture
def hello_world_task_dir() -> Path:
    path = REF_TASKS / "hello-world"
    if not path.exists():
        pytest.skip("Reference tasks not available")
    return path


@pytest.fixture
def build_result_json(tmp_path):
    """Factory: call SDK._build_result with 14-field defaults and return parsed result.json.

    Accepts `trajectory_source` override so callers that care about that field can
    set it; otherwise defaults to None. The `rollout_dir` used is `tmp_path / "trial"`
    (created if missing). Any override via kwargs is merged on top of the defaults.
    """

    def _build(**overrides) -> dict:
        from benchflow.sdk import SDK

        rollout_dir = tmp_path / "trial"
        rollout_dir.mkdir(exist_ok=True)
        defaults = dict(
            task_name="t1",
            rollout_name="trial-1",
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
        SDK._build_result(rollout_dir, **defaults)
        return json.loads((rollout_dir / "result.json").read_text())

    return _build


@pytest.fixture
def job_factory(tmp_path):
    """Create a Evaluation with n task directories and a mocked SDK."""
    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig

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
        cfg = EvaluationConfig(retry=RetryConfig(max_retries=max_retries))
        job = Evaluation(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=cfg)
        return job, tasks_dir

    return _make
