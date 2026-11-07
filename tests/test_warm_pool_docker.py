"""The warm pool on a real Docker daemon: live sandboxes stay within
``--concurrency`` + N, and a ``--concurrency 1`` job keeps
one sandbox.

Real ``Evaluation`` jobs, real sandboxes (``python:3.12-slim``), the nop
agent; a thread counts the job's running containers every 0.2 s. Skipped
when no Docker daemon answers (the gate container has none, so
these run on a Docker VM).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import uuid
from pathlib import Path

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


pytestmark = [
    pytest.mark.skipif(not _docker_ready(), reason="a Docker daemon is required"),
]
# Whole trials on real Docker: the exec channel and the side-by-side steps
# before the verifier stay on, as in a run (tests/conftest.py).
DOCKER_RUN_DEFAULTS = True

IMAGE = "python:3.12-slim"


def _immutable_image() -> str:
    """Guards 9b0320f: warm trials bind the prepared image's immutable ID."""
    inspect = ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE]
    found = subprocess.run(inspect, capture_output=True, text=True, timeout=60)
    if found.returncode:
        subprocess.run(
            ["docker", "pull", IMAGE], check=True, capture_output=True, timeout=600
        )
        found = subprocess.run(inspect, capture_output=True, text=True, timeout=60)
    assert found.returncode == 0, found.stderr
    image = found.stdout.strip()
    assert image.startswith("sha256:") and len(image) == 71
    return image


def _tasks(root: Path, prefix: str, count: int) -> Path:
    image = _immutable_image()
    for i in range(count):
        task = root / f"{prefix}-{i}"
        (task / "tests").mkdir(parents=True)
        (task / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 120\n[agent]\n'
            f'timeout_sec = 60\n[environment]\ndocker_image = "{image}"\n'
        )
        (task / "instruction.md").write_text("Do nothing.\n")
        (task / "tests" / "test.sh").write_text(
            "#!/bin/sh\necho 0 > /logs/verifier/reward.txt\n"
        )
    return root


class LiveCount:
    """The most running containers of this job's tasks seen at once."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _count(self) -> int:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        return sum(1 for name in out.split() if name.startswith(self.prefix))

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._count())
            self._stop.wait(0.2)

    def __enter__(self) -> LiveCount:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def left(self) -> list[str]:
        out = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        return [name for name in out.split() if name.startswith(self.prefix)]


async def _run(tmp_path: Path, monkeypatch, *, tasks: int, concurrency: int):
    monkeypatch.delenv("BENCHFLOW_WARM_POOL", raising=False)
    monkeypatch.setenv("BENCHFLOW_PREPULL", "off")
    prefix = f"wpd{uuid.uuid4().hex[:6]}"
    job = Evaluation(
        tasks_dir=_tasks(tmp_path / "tasks", prefix, tasks),
        jobs_dir=tmp_path / "jobs",
        job_name="pool",
        config=EvaluationConfig(
            agent="nop",
            environment="docker",
            concurrency=concurrency,
            retry=RetryConfig(max_retries=0),
            integrity="off",
        ),
        preflight=False,
    )
    with LiveCount(prefix) as live:
        result = await job.run()
    return result, live


def _timings(tmp_path: Path) -> list[dict]:
    import json

    return [
        json.loads(p.read_text()).get("timing") or {}
        for p in (tmp_path / "jobs" / "pool").glob("*/result.json")
    ]


@pytest.mark.asyncio
async def test_live_sandboxes_stay_within_concurrency_plus_one(tmp_path, monkeypatch):
    result, live = await _run(tmp_path, monkeypatch, tasks=5, concurrency=2)
    assert live.peak <= 3, live.peak
    assert live.left() == []
    assert result.total == 5 and result.errored == 0 and result.passed == 0
    # The pool was used: some trial took a prepared sandbox.
    assert any("warm_pool_wait" in t for t in _timings(tmp_path))


@pytest.mark.asyncio
async def test_concurrency_one_keeps_one_sandbox(tmp_path, monkeypatch):
    result, live = await _run(tmp_path, monkeypatch, tasks=3, concurrency=1)
    assert live.peak == 1, live.peak
    assert live.left() == []
    assert result.total == 3 and result.errored == 0 and result.passed == 0
    assert not any("warm_pool_wait" in t for t in _timings(tmp_path))
