"""The host's hard deadline doesn't count the time checkpoints take.

A checkpoint runs between prompts, outside every prompt's own budget, but
inside the rollout's hard deadline: the sum of the phase budgets plus a fixed
margin. A Daytona snapshot of a sandbox with an agent's install takes minutes
(up to BENCHFLOW_DAYTONA_SNAPSHOT_TIMEOUT_SEC, 600 s, and the look after it),
so with ``--checkpoints every-prompt`` the margin could run out and the
backstop would stop the agent and report the agent's timeout. The deadline now
moves out by ``checkpoints.seconds_spent``, the checkpoint in progress
included, and each checkpoint row records its seconds.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

from benchflow import checkpoints
from benchflow.checkpoints import after_prompt, parse_checkpoint_policy
from benchflow.models import RolloutResult
from benchflow.rollout import Rollout, _deadline
from benchflow.sandbox.protocol import SandboxImage

CONFIG = SimpleNamespace(task_path=Path("primes"))


def _checkpointing_rollout() -> SimpleNamespace:
    return SimpleNamespace(_checkpoint_seconds=0.0, _checkpoint_started=None)


async def _checkpoint(rollout, seconds: float) -> None:
    rollout._checkpoint_started = time.monotonic()
    await asyncio.sleep(seconds)
    rollout._checkpoint_seconds = checkpoints.seconds_spent(rollout)
    rollout._checkpoint_started = None


# K24: a lifecycle whose checkpoint outlasts the whole deadline still ends
# itself: the deadline moved out by the checkpoint's seconds.
async def test_the_hard_deadline_moves_out_by_the_time_checkpoints_take(
    monkeypatch,
):
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.3")
    rollout = _checkpointing_rollout()

    async def lifecycle():
        await asyncio.sleep(0.1)
        await _checkpoint(rollout, 1.5)
        await asyncio.sleep(0.1)
        return RolloutResult(task_name="primes", rollout_name="primes__abc")

    result = await _deadline.enforce_hard_deadline(
        lifecycle(),
        config=CONFIG,
        paused_sec=lambda: checkpoints.seconds_spent(rollout),
    )
    assert result.error is None
    assert result.rollout_name == "primes__abc"


# K24b (guard): time that isn't a checkpoint still trips the deadline, and its
# message names the checkpoint seconds it added.
async def test_a_wedge_after_a_checkpoint_still_trips_the_deadline(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.3")
    monkeypatch.setattr(_deadline, "ABANDON_CLEANUP_BOUND_SEC", 0.1)
    rollout = _checkpointing_rollout()
    reasons: list[str] = []

    async def lifecycle():
        await _checkpoint(rollout, 1.2)
        await asyncio.sleep(30)
        return RolloutResult(task_name="primes", rollout_name="primes__abc")

    started = time.monotonic()
    result = await _deadline.enforce_hard_deadline(
        lifecycle(),
        config=CONFIG,
        stop_agent=lambda reason: reasons.append(reason) or False,
        paused_sec=lambda: checkpoints.seconds_spent(rollout),
    )
    assert 1.4 <= time.monotonic() - started < 5
    assert result.error is not None and "hard deadline" in result.error
    assert reasons and "plus 1s of checkpoints" in reasons[0]


# K24c: the rollout passes its checkpoint seconds to the deadline.
async def test_the_rollout_gives_the_deadline_its_checkpoint_seconds(monkeypatch):
    rollout = Rollout.__new__(Rollout)
    rollout._config = CONFIG
    rollout._solver_execution_complete = False
    rollout._rollout_dir = None
    rollout._checkpoint_seconds = 42.0
    seen: dict = {}

    async def lifecycle():
        return RolloutResult(task_name="primes", rollout_name="primes__abc")

    async def enforce(coro, *, config, stop_agent=None, paused_sec=None):
        seen["paused"] = paused_sec() if paused_sec is not None else None
        return await coro

    monkeypatch.setattr(rollout, "_run_lifecycle", lifecycle, raising=False)
    monkeypatch.setattr(_deadline, "enforce_hard_deadline", enforce)
    result = await rollout._run_with_scoring()
    assert result.rollout_name == "primes__abc"
    assert seen["paused"] == 42.0


class SlowSandbox:
    supports_snapshot = True

    def __init__(self, rollout, fail: bool) -> None:
        self.rollout = rollout
        self.fail = fail
        self.during: float | None = None

    async def snapshot(self, name=None) -> SandboxImage:
        await asyncio.sleep(0.2)
        self.during = checkpoints.seconds_spent(self.rollout)
        if self.fail:
            raise RuntimeError("provider refused")
        return SandboxImage(provider="daytona", ref="bf-snap-1")


# K25: the checkpoint in progress counts, the total stays after it, and a
# failed checkpoint's row records its seconds too (0.10.0 left them null).
async def test_checkpoint_seconds_count_the_one_in_progress_and_every_row(
    tmp_path,
):
    for fail in (False, True):
        trial = tmp_path / f"t{fail}"
        trial.mkdir()
        rollout = SimpleNamespace(
            _config=SimpleNamespace(checkpoints=parse_checkpoint_policy("prompt:1")),
            _rollout_dir=trial,
            _rollout_name="primes__abc",
            _cursor=SimpleNamespace(id="n1"),
        )
        rollout._env = SlowSandbox(rollout, fail)
        await after_prompt(rollout, 1)
        assert rollout._env.during is not None and rollout._env.during >= 0.2
        assert checkpoints.seconds_spent(rollout) >= 0.2
        assert rollout._checkpoint_started is None
        [row] = json.loads((trial / "checkpoints.json").read_text())["checkpoints"]
        assert row["status"] == ("failed" if fail else "kept")
        assert row["seconds"] is not None and row["seconds"] >= 0.2


class HangingSandbox:
    """A snapshot that never ends (an SDK call with no timeout of its own),
    on a sandbox whose worst case for one snapshot is 1 s."""

    supports_snapshot = True
    _strategy = SimpleNamespace(snapshot_budget_sec=lambda: 1.0)

    async def snapshot(self, name=None) -> SandboxImage:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


# K24d: a checkpoint wedged past its own worst case trips the deadline: its
# credit is capped there, so the backstop still fires.
async def test_a_wedged_checkpoint_still_trips_the_deadline(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.3")
    monkeypatch.setattr(_deadline, "ABANDON_CLEANUP_BOUND_SEC", 0.1)
    rollout = SimpleNamespace(
        _config=SimpleNamespace(checkpoints=parse_checkpoint_policy("prompt:1")),
        _env=HangingSandbox(),
        _rollout_dir=tmp_path,
        _rollout_name="primes__abc",
        _cursor=SimpleNamespace(id="n1"),
    )

    async def lifecycle():
        await after_prompt(rollout, 1)
        return RolloutResult(task_name="primes", rollout_name="primes__abc")

    result = await asyncio.wait_for(
        _deadline.enforce_hard_deadline(
            lifecycle(),
            config=CONFIG,
            paused_sec=lambda: checkpoints.seconds_spent(rollout),
        ),
        timeout=10,
    )
    assert result.error is not None and "hard deadline" in result.error


# K24e: a provider that gives no worst case gets 600 s of credit at most.
def test_a_checkpoint_without_a_worst_case_gets_600_s_at_most():
    rollout = SimpleNamespace()
    with checkpoints.clock(rollout, SimpleNamespace()):
        rollout._checkpoint_started -= 5000
        assert checkpoints.seconds_spent(rollout) == 600.0


# K24f: while a checkpoint that began with almost no time left runs, the
# deadline's wait is asked again about once a second, not in a busy loop.
async def test_the_deadline_rechecks_about_once_a_second(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.01")
    rollout = _checkpointing_rollout()
    calls = {"n": 0}
    real = _deadline._finished_within

    async def counted(inner, timeout):
        calls["n"] += 1
        return await real(inner, timeout)

    monkeypatch.setattr(_deadline, "_finished_within", counted)

    async def lifecycle():
        await _checkpoint(rollout, 1.5)
        return RolloutResult(task_name="primes", rollout_name="primes__abc")

    result = await _deadline.enforce_hard_deadline(
        lifecycle(),
        config=CONFIG,
        paused_sec=lambda: checkpoints.seconds_spent(rollout),
    )
    assert result.error is None
    assert calls["n"] <= 4, calls


# K24g: an in-run branch's snapshot is checkpoint time too.
async def test_a_branch_snapshot_counts_as_checkpoint_time(tmp_path):
    from benchflow.rollout import RolloutConfig, Scene

    class SlowSnapSandbox:
        supports_snapshot = True

        async def snapshot(self, name=None) -> SandboxImage:
            await asyncio.sleep(0.3)
            return SandboxImage(provider="fake", ref="bf-snap-1")

        async def restore(self, image) -> None:
            return None

    (tmp_path / "task").mkdir()
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy --agent")]
        )
    )
    rollout._env = SlowSnapSandbox()

    async def run_child(child):
        return 1.0

    await rollout.branch(2, run_child=run_child, snapshot_layers={"sandbox"})
    assert checkpoints.seconds_spent(rollout) >= 0.3
