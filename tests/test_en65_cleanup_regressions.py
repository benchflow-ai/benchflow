"""Cleanup ownership regressions from ac3bedaa1."""

from __future__ import annotations

import asyncio
import logging
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._warm_pool import task_shares_nothing
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RolloutResult
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout._teardown_queue import background_teardown
from benchflow.sandbox.process import modal as modal_process
from benchflow.sandbox.process.modal import ModalProcess
from tests.test_modal_live_process import _Aio, _Chunks, _Proc, _Sandbox
from tests.test_sv_overlap import _scenario, _trial, _verify


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_fails", [True, False])
async def test_modal_verifier_requires_confirmed_agent_release(
    tmp_path, monkeypatch, stop_fails
):
    """Guards ac3bedaa1: never allocate past a failed release check."""
    from benchflow.sandbox.modal_impl import ModalSandbox

    sc = await _scenario(tmp_path, monkeypatch, environment="modal", trajectory=[])
    live = True
    terminations = 0

    async def terminate():
        nonlocal live, terminations
        terminations += 1
        # Exhaust the provider's own retry. The final cleanup can release it.
        if stop_fails and terminations <= 2:
            raise ConnectionError("terminate RPC unavailable")
        live = False

    remote = SimpleNamespace(
        object_id="sb-agent-gpu",
        terminate=SimpleNamespace(aio=terminate),
        wait=SimpleNamespace(aio=AsyncMock(return_value=None)),
        poll=SimpleNamespace(aio=AsyncMock(side_effect=lambda: None if live else 137)),
    )
    env = object.__new__(ModalSandbox)
    env._sandbox = remote
    env.logger = logging.getLogger(__name__)
    sc.rollout._env = env
    create = sc.rollout._planes.create_environment
    overlaps = []

    def create_verifier(*args, **kwargs):
        verifier = create(*args, **kwargs)
        start = verifier.start

        async def start_with_one_gpu(*args, **kwargs):
            overlaps.append(live)
            if live:
                raise RuntimeError("agent still owns the only GPU")
            await start(*args, **kwargs)

        verifier.start = start_with_one_gpu
        return verifier

    sc.rollout._planes.create_environment = create_verifier
    rewards = await _trial(sc)
    record = sc.record()
    assert live is False
    if stop_fails:
        assert overlaps == []
        assert rewards is None
        assert "agent sandbox release" in sc.rollout._verifier_error
        assert record["error"] == sc.rollout._verifier_error
        assert "still running" in record["custody"]["agent_stop_error"]
        assert record["status"] == "abandoned"
        assert "verifier.test_sh" not in sc.log
        assert terminations == 3
    else:
        assert overlaps == [False]
        assert rewards == {"reward": 1.0}
        assert sc.rollout._verifier_error is None
        assert record["status"] == "complete"
        assert terminations == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", ["modal", "daytona"])
async def test_after_stop_verifier_does_not_start_after_pre_stop_failure(
    tmp_path, monkeypatch, environment
):
    """Guards ac3bedaa1 when teardown fails before stop is attempted."""
    sc = await _scenario(tmp_path, monkeypatch, environment=environment, trajectory=[])
    sc.rollout._steps_before_agent_stop = AsyncMock(return_value=False)
    try:
        assert await _verify(sc) is None
        assert "verifier.create" not in sc.log
        assert "agent.stop" not in sc.log
        assert "agent sandbox release" in sc.rollout._verifier_error
    finally:
        await sc.rollout.cleanup()
    assert "agent.stopped" in sc.log


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_stage", ["stdin", "stderr"])
async def test_modal_close_cancellation_preserves_cleanup_for_retry(
    monkeypatch, cancel_stage
):
    """Guards ac3bedaa1: cancelled close must not orphan readers."""
    hold = asyncio.Event()
    entered = asyncio.Event()
    release = asyncio.Event()
    proc = _Proc(hold=hold)
    proc.stderr = _Chunks([], hold=hold)
    drains = 0

    async def drain():
        nonlocal drains
        drains += 1
        if cancel_stage == "stdin":
            entered.set()
            await release.wait()

    proc.stdin.drain = _Aio(drain)
    live = ModalProcess(_Sandbox(proc))
    # The existing fixture supplies a fake Sandbox; no provider SDK call occurs.
    live._proc = proc
    live._stdout_task = asyncio.create_task(live._read_stdout())
    live._stderr_task = asyncio.create_task(live._drain_stderr())
    readers = [live._stdout_task, live._stderr_task]
    monkeypatch.setattr(modal_process, "_STDERR_DRAIN_TIMEOUT_SEC", 0.01)
    finish_drain = live._finish_stderr_drain

    async def finish_stderr(*, cancel_on_timeout=False):
        if cancel_stage == "stderr":
            entered.set()
            await release.wait()
        await finish_drain(cancel_on_timeout=cancel_on_timeout)

    live._finish_stderr_drain = finish_stderr
    closing = asyncio.create_task(live.close())
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert not live.is_running
        release.set()
        await asyncio.wait_for(live.close(), timeout=1)
        await live.close()
        assert all(task.done() for task in readers)
        assert proc.stdin.eof
        assert drains == 1
    finally:
        release.set()
        for task in readers:
            task.cancel()
        await asyncio.gather(*readers, return_exceptions=True)


@pytest.mark.asyncio
async def test_serial_trials_release_exclusive_port_before_the_next_starts(
    tmp_path: Path,
):
    """Guards ac3bedaa1: queued cleanup must not exceed exclusive ownership."""
    events = []
    with socket.socket() as seed:
        seed.bind(("127.0.0.1", 0))
        port = seed.getsockname()[1]
    tasks = [tmp_path / "task-a", tmp_path / "task-b"]
    for task in tasks:
        (task / "environment").mkdir(parents=True)
        (task / "task.toml").write_text('version="1.0"\n')
        (task / "environment/docker-compose.yaml").write_text(
            f'services:\n  main:\n    ports: ["{port}:8080"]\n'
        )

    class Sandbox:
        background_stop = True

        def __init__(self, held, name):
            self.held, self.name = held, name

        async def stop(self, delete=True):
            await asyncio.sleep(0.05)
            self.held.close()
            events.append(f"released {self.name}")

    async def run_one(task, _config):
        held = socket.socket()
        try:
            held.bind(("127.0.0.1", port))
            held.listen()
        except OSError as exc:
            held.close()
            events.append(f"collision {task.name}")
            return RolloutResult(task_name=task.name, error=f"startup failed: {exc}")
        events.append(f"acquired {task.name}")
        rollout = Rollout(RolloutConfig(task_path=task, agent="oracle"))
        rollout._env = Sandbox(held, task.name)
        rollout._cleanup_before_teardown = AsyncMock()
        await rollout.cleanup()
        return RolloutResult(task_name=task.name, rewards={"reward": 1.0})

    config = EvaluationConfig(
        agent="oracle", concurrency=1, retry=RetryConfig(max_retries=0)
    )
    job = Evaluation(tmp_path, tmp_path / "jobs", config, job_name="exclusive-port")
    job._run_single_task = run_one
    async with background_teardown(limit=1):
        pairs = await job._run_parallel_independent(tasks)
    assert all(result.rewards == {"reward": 1.0} for _, result in pairs)
    assert events == [
        "acquired task-a",
        "released task-a",
        "acquired task-b",
        "released task-b",
    ]


@pytest.mark.parametrize(
    ("compose", "independent"),
    [
        ("services:\n  main:\n    image: ubuntu\n", True),
        ("include: shared.yaml\nservices: {}\n", False),
        (
            "services:\n  main:\n    extends: {file: shared.yaml, service: base}\n",
            False,
        ),
        ("services:\n  main:\n    gpus: all\n", False),
        ("services:\n  main:\n    ports: ['8123:80']\n", False),
    ],
)
def test_composed_or_exclusive_resources_cannot_overlap(tmp_path, compose, independent):
    """Guards ac3bedaa1: unresolved Compose inheritance may reserve host state."""
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "docker-compose.yaml").write_text(compose)
    (environment / "shared.yaml").write_text(
        "services:\n  base:\n    container_name: exclusive-name\n"
    )
    assert task_shares_nothing(tmp_path) is independent
