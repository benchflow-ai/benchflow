"""Sandbox teardown off a trial's critical path (PERF-B,
rollout/_teardown_queue.py).

A trial's result and the next trial no longer wait for its sandbox's stop
(1-3 s of Docker work); the job waits for every stop before it sweeps Docker
and returns, Ctrl-C included. Outside a job, and for a sandbox that holds a
verifier-recovery lease, the stop runs inline as before.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.sandbox.protocol import ExecResult

STOP_SEC = 0.6


def _trial(task_path: Path, events: list[tuple[str, str, float]]) -> Rollout:
    trial = Rollout(RolloutConfig(task_path=task_path, agent="oracle"))
    trial._env = MagicMock()
    trial._env.exec = AsyncMock(return_value=ExecResult(0, "", ""))
    task_tmp = task_path.parent / f"{task_path.name}-tmp"
    task_tmp.mkdir()
    trial._task_tmp = task_tmp

    async def stop(delete: bool) -> None:
        events.append(("stop-begin", task_path.name, time.monotonic()))
        await asyncio.sleep(STOP_SEC)
        # The task copy the stop's Compose files live in is still there.
        assert task_tmp.is_dir()
        events.append(("stop-end", task_path.name, time.monotonic()))

    trial._env.stop = AsyncMock(side_effect=stop)
    trial._env.background_stop = True  # a Docker sandbox
    return trial


def _job(tmp_path: Path, n_tasks: int, concurrency: int, budget=None) -> Evaluation:
    tasks_dir = tmp_path / "tasks"
    for i in range(n_tasks):
        task = tasks_dir / f"task-{i}"
        task.mkdir(parents=True)
        (task / "task.toml").write_text('version = "1.0"\n')
        (task / "instruction.md").write_text("do it\n")
    cfg = EvaluationConfig(
        agent="oracle",
        concurrency=concurrency,
        retry=RetryConfig(max_retries=0),
        budget=budget,
    )
    return Evaluation(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=cfg)


@pytest.mark.asyncio
async def test_the_next_trial_starts_while_the_last_sandbox_stops(tmp_path):
    """Guards py8/perf-exec (PERF-B): each trial's sandbox stop (kill, rm,
    down, the image listing: 1-3 s of Docker work) held its result and the
    next trial. Now the next trial starts at once, and the job still
    returns only after every stop has ended and every task copy is gone."""
    from benchflow.evaluation import RunResult

    events: list[tuple[str, str, float]] = []
    job = _job(tmp_path, n_tasks=3, concurrency=1)
    trials: list[Rollout] = []

    async def run_one(task_dir: Path, _cfg) -> RunResult:
        events.append(("start", task_dir.name, time.monotonic()))
        trial = _trial(tmp_path / "work" / task_dir.name, events)
        trials.append(trial)
        await trial._teardown_sandbox()
        assert trial._phase == "cleaned"
        events.append(("result", task_dir.name, time.monotonic()))
        return RunResult(task_name=task_dir.name, rewards={"reward": 1.0})

    (tmp_path / "work").mkdir()
    for i in range(3):
        (tmp_path / "work" / f"task-{i}").mkdir()
    job._run_single_task = AsyncMock(side_effect=run_one)
    began = time.monotonic()
    result = await job.run()
    took = time.monotonic() - began

    assert result.passed == 3
    when = {(kind, name): t for kind, name, t in events}
    # Each result came before its sandbox finished stopping, and the next
    # trial started meanwhile.
    for i in range(3):
        assert when[("result", f"task-{i}")] < when[("stop-end", f"task-{i}")]
    for i in range(2):
        assert when[("start", f"task-{i + 1}")] < when[("stop-end", f"task-{i}")]
    # The job waited for all of them, and removed every task copy after.
    assert sum(1 for kind, _, _ in events if kind == "stop-end") == 3
    assert all(not t._task_tmp.exists() for t in trials)
    assert took < 3 * STOP_SEC + 1.5


@pytest.mark.asyncio
async def test_stops_are_bounded_and_drained_through_ctrl_c(tmp_path):
    from benchflow.rollout._teardown_queue import background_teardown

    finished: list[int] = []
    submitted: list[int] = []
    running = 0
    peak = 0

    async def stop(i: int) -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.3)
        running -= 1
        finished.append(i)

    async def job() -> None:
        async with background_teardown(limit=2) as queue:
            for i in range(5):
                await queue.submit(lambda i=i: stop(i), what=f"t{i}")
                submitted.append(i)

    task = asyncio.ensure_future(job())
    await asyncio.sleep(0.4)
    task.cancel()  # Ctrl-C while the stops run
    with pytest.raises(asyncio.CancelledError):
        await task
    # Ctrl-C stopped the job from queueing more; every queued stop ended.
    assert len(submitted) < 5
    assert sorted(finished) == submitted
    assert peak == 2


@pytest.mark.asyncio
async def test_a_failed_stop_is_logged_and_the_job_goes_on(caplog):
    from benchflow.rollout._teardown_queue import background_teardown

    async def broken() -> None:
        raise RuntimeError("daemon said no")

    with caplog.at_level(logging.WARNING):
        async with background_teardown(limit=2) as queue:
            await queue.submit(broken, what="task-x")
    assert any("Cleanup of task-x failed: daemon said no" in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_outside_a_job_and_with_a_recovery_lease_the_stop_is_inline(
    tmp_path, monkeypatch
):
    from benchflow.rollout import _verifier_recovery
    from benchflow.rollout._teardown_queue import background_teardown

    events: list[tuple[str, str, float]] = []
    (tmp_path / "solo").mkdir()
    solo = _trial(tmp_path / "solo", events)
    await solo._teardown_sandbox()
    assert [kind for kind, _, _ in events] == ["stop-begin", "stop-end"]
    assert not solo._task_tmp.exists()

    released: list[str] = []

    async def release(rollout) -> None:
        # The lease goes only after the sandbox that ran on it has stopped.
        assert ("stop-end", "leased") in {(k, n) for k, n, _ in events}
        released.append("lease")

    monkeypatch.setattr(_verifier_recovery, "release_lease_at_teardown", release)
    (tmp_path / "leased").mkdir()
    leased = _trial(tmp_path / "leased", events)
    leased._recovery_lease_owned = True
    async with background_teardown(limit=2):
        await leased._teardown_sandbox()
        assert ("stop-end", "leased") in {(k, n) for k, n, _ in events}
    assert released == ["lease"]


@pytest.mark.asyncio
async def test_the_built_image_listing_runs_beside_the_container_removal():
    """Guards py8/perf-exec (PERF-B): stop() listed the images the project
    built (0.3-1.2 s on a busy daemon) only after kill, rm and down. It
    now lists while they run; every image removal still comes after down."""
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = object.__new__(DockerSandbox)
    sandbox.logger = logging.getLogger("test")
    sandbox._keep_containers = False
    sandbox._compose_up_attempted = True
    sandbox._recovery_baseline = None
    sandbox._chown_to_host_user = AsyncMock()
    sandbox._release_layers = AsyncMock()
    sandbox._remove_project_volumes = AsyncMock()
    sandbox._snapshot_credentials = MagicMock()
    sandbox._deferred_snapshot_refs = set()
    sandbox._staged = None
    sandbox._release_subnets = lambda: None
    sandbox._compose_project = "perf-b-overlap"
    events: list[tuple[str, float]] = []

    async def compose(command, **_kwargs):
        events.append((f"{command[0]}-begin", time.monotonic()))
        await asyncio.sleep(0.3)
        events.append((command[0], time.monotonic()))
        return ExecResult(0, "", "")

    async def listing():
        events.append(("list-begin", time.monotonic()))
        await asyncio.sleep(0.6)
        events.append(("list", time.monotonic()))
        return ExecResult(0, "", "")

    async def remove(listed=None):
        assert listed is not None
        await listed
        events.append(("remove", time.monotonic()))

    sandbox._run_docker_compose_command = compose
    sandbox._list_built_images = listing
    sandbox._remove_built_images = remove
    began = time.monotonic()
    await DockerSandbox.stop(sandbox, delete=True)
    took = time.monotonic() - began
    when = dict(events)
    assert when["list-begin"] < when["kill-begin"] + 0.1
    assert when["remove"] > when["down"]
    assert took < 0.3 * 3 + 0.6 - 0.3


@pytest.mark.asyncio
async def test_a_provider_sandbox_stops_inline_inside_a_job(tmp_path):
    """Guards PERF-B-L1 (review of fc2563ec): every backend's stop
    was queued, so a Daytona/Modal/AgentCore job could hold up to twice its
    concurrency in live, billed sandboxes. Only Docker-family sandboxes
    (``background_stop``) are queued now."""
    from benchflow.rollout._teardown_queue import background_teardown

    events: list[tuple[str, str, float]] = []
    (tmp_path / "remote").mkdir()
    trial = _trial(tmp_path / "remote", events)
    trial._env.background_stop = False
    async with background_teardown(limit=2):
        await trial._teardown_sandbox()
        assert [kind for kind, _, _ in events] == ["stop-begin", "stop-end"]


@pytest.mark.asyncio
async def test_a_jobs_budget_counts_each_queued_stop(tmp_path):
    """Guards PERF-B-L1: a trial's budgeted sandbox seconds ended
    with its result, before its queued stop; the stop's seconds now count,
    added when it ends, before the job's drain ends."""
    from benchflow.budget import Budget
    from benchflow.evaluation import RunResult

    events: list[tuple[str, str, float]] = []
    job = _job(
        tmp_path, n_tasks=3, concurrency=1, budget=Budget(max_sandbox_seconds=10_000)
    )
    (tmp_path / "work").mkdir()

    async def run_one(task_dir: Path, _cfg) -> RunResult:
        (tmp_path / "work" / task_dir.name).mkdir()
        trial = _trial(tmp_path / "work" / task_dir.name, events)
        await trial._teardown_sandbox()
        return RunResult(task_name=task_dir.name, rewards={"reward": 1.0})

    job._run_single_task = AsyncMock(side_effect=run_one)
    result = await job.run()
    assert result.passed == 3
    assert result.budget is not None
    assert result.budget["spent"]["sandbox_seconds"] >= 3 * STOP_SEC - 0.05


class _ImageListClient:
    def __init__(self, rows=None, error=None):
        self.rows, self.error, self.calls = rows, error, []

    async def call(self, method, path, *, query=None, body=None, timeout=None):
        self.calls.append((method, path, query))
        if self.error is not None:
            raise self.error

        class _Reply:
            def json(_self):
                return self.rows

        return _Reply()


def _listing_sandbox(client):
    from types import SimpleNamespace

    from benchflow.sandbox.docker import DockerSandbox

    sandbox = object.__new__(DockerSandbox)
    sandbox._compose_project = "perf-sbx-list"
    sandbox._direct = SimpleNamespace(client=client)
    return sandbox


@pytest.mark.asyncio
async def test_the_direct_path_lists_built_images_through_the_engine_api():
    """PERF-SBX: the listing beside the removal was a ``docker image ls``
    process (0.7 s, longer than the faster removal); on the admitted direct
    path the daemon answers it in one request, in the CLI's format."""
    a, b = "sha256:" + "a" * 64, "sha256:" + "b" * 64
    client = _ImageListClient([{"Id": a}, {"Id": b}, {"Id": a}])
    listed = await _listing_sandbox(client)._list_built_images()
    assert (listed.return_code, listed.stdout) == (0, f"{a}\n{b}\n{a}\n")
    assert client.calls == [
        (
            "GET",
            "/images/json",
            {"filters": {"label": {"com.docker.compose.project=perf-sbx-list": True}}},
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows, error",
    [
        (None, OSError("socket gone")),
        ({"message": "not a list"}, None),
        ([{"Id": "sha256:short"}], None),
        ([{"Id": "sha256:" + "a" * 64}, "x"], None),
    ],
)
async def test_an_unreadable_engine_image_list_is_a_failed_listing(rows, error):
    """Never "nothing built": _remove_built_images warns on a failed listing."""
    listed = await _listing_sandbox(_ImageListClient(rows, error))._list_built_images()
    assert listed.return_code != 0 and listed.stdout in ("", None)
