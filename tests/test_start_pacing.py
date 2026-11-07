"""Trial starts are paced by a token bucket outside the slot (round PY8, U9).

Above ``--concurrency 16`` each trial used to sleep ``uniform(0, max(c/2, 8))``
seconds while it held its concurrency slot, so every slot idled about c/4
seconds per trial. Guards py8/start-pacing: starts stay at or below
2 per second, no slot is held while a trial waits for its token, a usage stop
still skips a trial whose wait ended after the stop, and at 16 or less
nothing changes.

The tests run the real ``Evaluation`` schedule on a shared virtual clock:
``asyncio.sleep`` parks the caller until the clock reaches its wake-up, and
the clock moves only once every other coroutine has run, so overlapping
sleeps overlap as they do on a real clock and every time is exact.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import json
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest

from benchflow._utils import timing_spans
from benchflow._utils.scoring import USAGE_LIMIT
from benchflow.agents.errors import UsageLimitError
from benchflow.evaluation import Evaluation, EvaluationConfig, UsageLimitStop
from benchflow.models import RolloutResult
from benchflow.rollout import Rollout, RolloutConfig

_real_sleep = asyncio.sleep
_real_semaphore = asyncio.Semaphore

# Rounds of the event loop the driver lets pass before it moves the clock.
_SETTLE_ROUNDS = 200


class VirtualTime:
    def __init__(self) -> None:
        self.now = 0.0
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = 0

    def clock(self) -> float:
        return self.now

    async def sleep(self, delay: float, result: Any = None) -> Any:
        if delay <= 0:
            await _real_sleep(0)
            return result
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._seq += 1
        heapq.heappush(self._sleepers, (self.now + delay, self._seq, future))
        await future
        return result

    async def run(
        self,
        coro: Coroutine[Any, Any, Any],
        on_settled: Callable[[], None] | None = None,
    ) -> Any:
        main = asyncio.ensure_future(coro)
        while True:
            for _ in range(_SETTLE_ROUNDS):
                await _real_sleep(0)
                if main.done():
                    return main.result()
            if on_settled is not None:
                on_settled()
            while self._sleepers and self._sleepers[0][2].done():
                heapq.heappop(self._sleepers)
            if not self._sleepers:
                await _real_sleep(0.001)  # something off the loop (a thread)
                continue
            self.now = max(self.now, self._sleepers[0][0])
            while self._sleepers and self._sleepers[0][0] <= self.now:
                future = heapq.heappop(self._sleepers)[2]
                if not future.done():
                    future.set_result(None)


class CountingSemaphore(_real_semaphore):
    """A semaphore that counts the slots taken (the schedule's slots)."""

    made: ClassVar[list[CountingSemaphore]] = []

    def __init__(self, value: int = 1) -> None:
        super().__init__(value)
        self.size = value
        self.held = 0
        CountingSemaphore.made.append(self)

    async def acquire(self) -> bool:
        taken = await super().acquire()
        self.held += 1
        return taken

    def release(self) -> None:
        self.held -= 1
        super().release()


@pytest.fixture
def vt(monkeypatch) -> VirtualTime:
    virtual = VirtualTime()
    monkeypatch.setattr(timing_spans, "_clock", virtual.clock)
    monkeypatch.setattr(asyncio, "sleep", virtual.sleep)
    CountingSemaphore.made = []
    monkeypatch.setattr(asyncio, "Semaphore", CountingSemaphore)
    return virtual


def _job(tmp_path: Path, n: int, concurrency: int) -> Evaluation:
    tasks_dir = tmp_path / "tasks"
    for i in range(n):
        (tasks_dir / f"task-{i:03d}").mkdir(parents=True)
        (tasks_dir / f"task-{i:03d}" / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    job = Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent="claude-agent-acp",
            model="claude-haiku-4-5",
            concurrency=concurrency,
        ),
        job_name="job",
        preflight=False,
    )
    job._sweep_docker = AsyncMock()  # no Docker daemon in a unit test
    return job


class Trials:
    """Fake trials: each records its start and timing, then runs ``seconds``."""

    def __init__(
        self,
        job: Evaluation,
        vt: VirtualTime,
        seconds: Callable[[int], float],
        limited: frozenset[str] = frozenset(),
    ) -> None:
        self.job, self.vt, self.seconds, self.limited = job, vt, seconds, limited
        self.started: dict[str, float] = {}
        self.timing: dict[str, dict[str, Any]] = {}
        self.running = 0
        self.idle_slots: list[tuple[float, int, int]] = []
        job._run_single_task = AsyncMock(side_effect=self._run)

    async def _run(self, task_path: Path, _cfg: Any) -> RolloutResult:
        name = task_path.name
        rollout = Rollout(RolloutConfig(task_path=task_path, agent="nop"))
        self.started[name] = self.vt.now
        self.timing[name] = dict(rollout.timing)
        self.running += 1
        try:
            await asyncio.sleep(self.seconds(int(name.rsplit("-", 1)[1])))
        finally:
            self.running -= 1
        rollout_dir = self.job._jobs_dir / self.job._job_name / f"{name}__abcd1234"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        limited = name in self.limited
        text = "usage limit reached on login L: 7-day window (limit)"
        payload = {
            "task_name": name,
            "rollout_name": rollout_dir.name,
            "rewards": None if limited else {"reward": 1.0},
            "error": text if limited else None,
            "error_category": USAGE_LIMIT if limited else None,
            "verifier_error": None,
            "usage_limit_info": {"login": "L", "window": "7-day"} if limited else None,
        }
        (rollout_dir / "result.json").write_text(json.dumps(payload))
        return RolloutResult(
            task_name=name,
            rollout_name=rollout_dir.name,
            rewards=payload["rewards"],
            error=payload["error"],
            error_category=payload["error_category"],
            rollout_dir=rollout_dir,
        )

    def check_slots(self) -> None:
        """Called whenever the loop is settled: every slot taken is a running
        trial's (a held slot whose trial is not running is an idle slot)."""
        slots = [
            s for s in CountingSemaphore.made if s.size == self.job._config.concurrency
        ]
        for slot in slots:
            if slot.held != self.running:
                self.idle_slots.append((self.vt.now, slot.held, self.running))


async def _paced_run(tmp_path: Path, vt: VirtualTime, concurrency: int) -> Trials:
    """2c trials of mixed lengths, so later ones wait for slots as well as
    for tokens."""
    job = _job(tmp_path, 2 * concurrency, concurrency)
    trials = Trials(job, vt, lambda i: (3.0, 11.0, 47.0, 0.2)[i % 4])
    await vt.run(job.run(), on_settled=trials.check_slots)
    assert len(trials.started) == 2 * concurrency
    return trials


@pytest.mark.parametrize("concurrency", [32, 64])
async def test_starts_stay_at_or_below_the_bucket_rate(tmp_path, vt, concurrency):
    trials = await _paced_run(tmp_path, vt, concurrency)
    times = sorted(trials.started.values())
    # 2 starts/s with one token in hand: no two starts closer than 0.5 s, so
    # at most 1 + 2w starts in any w seconds.
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert min(gaps) >= 0.5 - 1e-9
    # And not slower than that while slots are free: the first wave starts at
    # exactly the bucket rate.
    assert times[:concurrency] == pytest.approx([0.5 * k for k in range(concurrency)])


@pytest.mark.parametrize("concurrency", [32, 64])
async def test_no_slot_is_held_while_a_trial_waits_to_start(tmp_path, vt, concurrency):
    trials = await _paced_run(tmp_path, vt, concurrency)
    # Whenever the loop is settled, every slot taken is a running trial's: the
    # token wait happens before the slot is taken.
    assert trials.idle_slots == []
    # The instrument agrees: slot_wait, slot taken to trial start,
    # is 0 for every trial; the wait moved to start_pace_wait, outside it.
    assert {t["slot_wait"] for t in trials.timing.values()} == {0.0}
    pace = [t.get("start_pace_wait") for t in trials.timing.values()]
    assert None not in pace
    assert sum(pace) >= 0.5 * (concurrency - 1) - 1e-9
    assert sum(pace) <= 0.5 * (2 * concurrency - 1) + 1e-9


async def test_slots_that_free_together_still_start_at_the_bucket_rate(vt):
    """A token is spent only when its slot is taken, and the bucket holds one:
    a trial waiting for a busy slot cannot bank a burst for the trials behind
    it, so two slots that free at once start their next trials 0.5 s apart."""
    from benchflow._utils.start_pacing import StartPacer

    pacer = StartPacer(rate=2.0)
    slots = _real_semaphore(2)
    release_all = asyncio.Event()
    started: dict[int, tuple[float, float]] = {}

    async def trial(i: int) -> None:
        waited = await pacer.take_slot(slots, lambda: False)
        started[i] = (vt.now, waited)
        try:
            if i < 2:
                await release_all.wait()
        finally:
            slots.release()

    async def free_both_at_ten() -> None:
        await asyncio.sleep(10.0)
        release_all.set()

    async def main() -> None:
        await asyncio.gather(free_both_at_ten(), *(trial(i) for i in range(4)))

    await vt.run(main())
    assert started == {
        0: (0.0, 0.0),
        1: (0.5, 0.5),
        2: (10.0, 0.5),  # its token came at 1.0, then it waited for a slot
        3: (10.5, 0.5),  # not 10.0: no second token was saved up
    }


async def test_a_usage_stop_skips_a_trial_whose_token_wait_ended_after_it(
    tmp_path, vt, monkeypatch
):
    """task-000 hits the usage limit at 1.2 s, while task-003 waits for the
    token due at 1.5 s: task-003 and every trial behind it are left out, and
    none of them waits for a token after the stop."""
    n, concurrency = 64, 32
    job = _job(tmp_path, n, concurrency)
    trials = Trials(
        job, vt, lambda i: 1.2 if i == 0 else 30.0, limited=frozenset({"task-000"})
    )
    skipped_at: dict[str, float] = {}
    skip = UsageLimitStop.skip

    def timed_skip(self: UsageLimitStop, name: str) -> bool:
        if skip(self, name):
            skipped_at[name] = vt.now
            return True
        return False

    monkeypatch.setattr(UsageLimitStop, "skip", timed_skip)

    with pytest.raises(UsageLimitError) as caught:
        await vt.run(job.run(), on_settled=trials.check_slots)

    assert trials.started == {"task-000": 0.0, "task-001": 0.5, "task-002": 1.0}
    left_out = [f"task-{i:03d}" for i in range(3, n)]
    assert sorted(skipped_at) == left_out
    # task-003's wait ended after the stop (1.2 s) and it still did not start;
    # the 60 behind it were skipped at once, not one token apart.
    assert skipped_at["task-003"] == pytest.approx(1.5)
    assert max(skipped_at.values()) == pytest.approx(1.5)
    assert trials.idle_slots == []
    result = caught.value.result
    assert result is not None and result.job_dir is not None
    summary = json.loads((result.job_dir / "summary.json").read_text())
    assert sorted(summary["usage_limit"]["not_started"]) == left_out


@pytest.mark.parametrize("concurrency", [1, 8, 16])
async def test_at_concurrency_16_or_less_nothing_changes(
    tmp_path, vt, monkeypatch, concurrency
):
    """No pacer at 16 or less: a full first wave starts at once, each later
    trial as soon as a slot frees, and no trial records a pacing wait."""
    import benchflow.evaluation as evaluation

    class NoPacer:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("no pacer at --concurrency 16 or less")

    monkeypatch.setattr(evaluation, "StartPacer", NoPacer, raising=False)
    n = 2 * concurrency + 1
    job = _job(tmp_path, n, concurrency)
    trials = Trials(job, vt, lambda i: 5.0 + i)
    await vt.run(job.run(), on_settled=trials.check_slots)

    assert len(trials.started) == n
    order = sorted(trials.started, key=lambda name: int(name.rsplit("-", 1)[1]))
    starts = [trials.started[name] for name in order]
    assert starts[:concurrency] == [0.0] * concurrency
    # Trial c starts when the first slot frees (task 0 runs 5 s), and so on.
    assert starts[concurrency] == pytest.approx(5.0)
    assert trials.idle_slots == []
    assert {t["slot_wait"] for t in trials.timing.values()} == {0.0}
    assert all("start_pace_wait" not in t for t in trials.timing.values())
