"""One training episode on one BenchFlow task sandbox, and the training rule.

This half of the Tinker cookbook uses BenchFlow only (no Tinker import), so its
logic is testable anywhere. `tinker_env.py` wraps it for tinker-cookbook's RL
framework.

An `Episode` starts the task's sandbox with `bf.TaskRuntime`, runs the
policy's `run_bash` commands and `submit` answer in it the way the shared RL
harness does (docs/examples/rl/common/harness.py: BenchFlow's TRL adapter and
the held-out evaluator use the same tools, instructions and limits), runs the
task's verifier, and always closes the sandbox. The training rule of
`benchflow.integrations.rewards` turns each ending into a `RewardDecision`:

- dropped (reward None): an infrastructure failure the policy could not have
  caused: the sandbox never started, the model endpoint failed, or the
  verifier gave no reward on a clean run (the policy never acted). Dropped
  episodes leave the batch and are counted by reason.
- scored: the verifier's reward (a fraction of checks passed, for partial
  credit).
- 0, with a named reason, for every other failure: `timeout` (including the
  episode's wall-clock budget), `verifier_error`, `run_error`, `no_reward`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shlex
import shutil
import signal
import sys
import time
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import benchflow as bf
from benchflow.integrations.rewards import (
    DROP_REASONS,
    MODEL_ENDPOINT,
    SCORED,
    TIMEOUT,
    VERIFIER_CRASH_CLEAN_RUN,
    VERIFIER_ERROR,
    ZERO_REASONS,
    RewardDecision,
    dropped,
    reward_from_verify,
    sandbox_start_failure,
    zero,
)
from benchflow.integrations.trl import write_rollout_record

COMMON = Path(__file__).resolve().parents[1] / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
from harness import HARNESS_MESSAGE, MAX_TURNS, harness_config  # noqa: E402

log = logging.getLogger(__name__)

# The shared harness's limits (docs/examples/rl/common/harness.py).
_HARNESS = harness_config()
BASH_TIMEOUT_SEC = _HARNESS.bash_timeout_sec
MAX_OUTPUT_CHARS = _HARNESS.max_output_chars
SUBMIT_PATH = _HARNESS.submit_path
# What the TRL adapter's run_bash appends to output it cuts.
TRUNCATION_MARKER = "\n[benchflow output truncated]\n"

Decision = RewardDecision
# Reasons a kept episode can have, the verifier's reward first.
KEPT_REASONS = (SCORED, *sorted(ZERO_REASONS))

__all__ = [
    "BASH_TIMEOUT_SEC",
    "DROP_REASONS",
    "HARNESS_MESSAGE",
    "KEPT_REASONS",
    "LIVE",
    "RunStopped",
    "MAX_OUTPUT_CHARS",
    "MAX_TURNS",
    "SUBMIT_PATH",
    "TRUNCATION_MARKER",
    "Decision",
    "DropLog",
    "Episode",
    "EpisodeSettings",
    "GroupLog",
    "InfrastructureError",
    "SandboxSlots",
    "TooManyInfrastructureFailures",
    "close_all_live",
    "describe",
    "free_gib",
    "group_kind",
    "run_guarded",
    "solved",
    "truncate",
]


def solved(decision: RewardDecision | None) -> bool:
    """Every check passed: the verifier's full reward."""
    return (
        decision is not None and decision.reward is not None and decision.reward >= 1.0
    )


class InfrastructureError(RuntimeError):
    """A failure the policy could not have caused: drop the episode and count it."""

    def __init__(self, decision: Decision, *, task: str = "") -> None:
        if not decision.dropped:
            raise ValueError("InfrastructureError needs a dropped decision")
        where = f" (task {task})" if task else ""
        super().__init__(f"{decision.reason}: {decision.detail or ''}{where}")
        self.decision = decision
        self.reason = decision.reason
        self.task = task


def describe(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def truncate(text: str, limit: int) -> str:
    """Cut like the TRL adapter: keep the head, end with a marker."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(TRUNCATION_MARKER))] + TRUNCATION_MARKER


# -- sandboxes -----------------------------------------------------------------


class SandboxSlots:
    """A cap on live sandboxes, shared by every episode in this process.

    An episode takes a slot before its sandbox is created and gives it back
    after the sandbox is closed, whichever way the episode ended.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("the sandbox limit must be at least 1")
        self.limit = limit
        self.in_use = 0
        self.peak = 0
        self._semaphores: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = weakref.WeakKeyDictionary()

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        semaphore = self._semaphores.get(loop)
        if semaphore is None:
            semaphore = self._semaphores[loop] = asyncio.Semaphore(self.limit)
        return semaphore

    async def acquire(self) -> None:
        await self._semaphore().acquire()
        self.in_use += 1
        self.peak = max(self.peak, self.in_use)

    def release(self) -> None:
        self.in_use -= 1
        self._semaphore().release()


@dataclass(frozen=True)
class EpisodeSettings:
    """How episodes run their sandboxes. Picklable, as tinker-cookbook needs."""

    sandbox: str = "daytona"  # any BenchFlow backend: daytona, docker, modal, ...
    sandbox_user: str | None = "agent"  # the policy's commands never run as root
    jobs_dir: str = "trials"  # BenchFlow's trial layout: <jobs_dir>/<job>/<task>__<id>/
    command_timeout_sec: int = BASH_TIMEOUT_SEC  # per run_bash call
    max_output_chars: int = MAX_OUTPUT_CHARS  # per run_bash result shown to the model
    submit_path: str = SUBMIT_PATH  # where submit(answer) writes the answer
    episode_timeout_sec: float = 900.0  # wall clock once the sandbox is up; 0 = none
    verify_timeout_sec: float = 900.0  # guard around BenchFlow's verify and finalize
    # Reward integrity (BenchShield, benchflow.integrity): "audit" or "strict"
    # need a BenchFlow with TaskRuntimeConfig.integrity; an exploit the audit
    # finds scores 0 and is flagged (rewards.apply_integrity).
    integrity: str = "off"


RuntimeFactory = Callable[[Any], Awaitable[Any]]

# Every episode whose sandbox may be live, so a shutdown can close them all.
LIVE: set[Episode] = set()


class Episode:
    """One task sandbox, driven by an outside policy: start, act, finish, close.

    `close()` is idempotent and runs on every path: after `finish()` (which
    verifies), after `time_out()` (which does not), when `start()` fails, and
    from the caller's cleanup when the rollout was cut off from outside.
    """

    def __init__(
        self,
        task_dir: Path | str,
        settings: EpisodeSettings,
        *,
        slots: SandboxSlots,
        job_name: str,
        runtime_factory: RuntimeFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.task_dir = Path(task_dir)
        self.settings = settings
        self.slots = slots
        self.job_name = job_name
        self._factory = runtime_factory or bf.TaskRuntime.create
        self._clock = clock
        self.runtime: Any | None = None
        self.rollout_dir: Path | None = None
        self.decision: Decision | None = None
        self.ended: str | None = None  # how the episode ended (set by the caller)
        self.policy_acted = False
        self.commands_run = 0
        self.submitted = False
        self.timings: dict[str, float] = {}
        self._deadline: float | None = None
        self._created_at: float | None = None
        self._start_called = False
        self._closed = False
        self._slot_held = False

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Take a sandbox slot and start the task sandbox.

        Raises InfrastructureError (reason sandbox_start) when the sandbox
        does not start; the slot is given back and nothing is left running.
        """
        if self._start_called:
            raise RuntimeError("an Episode starts once")
        self._start_called = True
        await self.slots.acquire()
        self._slot_held = True
        integrity = {}
        if self.settings.integrity != "off":
            if "integrity" not in bf.TaskRuntimeConfig.__dataclass_fields__:
                self._release_slot()
                self._closed = True
                raise ValueError(
                    "this BenchFlow has no reward-integrity option; it comes with "
                    "benchflow.integrity (BenchShield)"
                )
            integrity = {"integrity": self.settings.integrity}
        config = bf.TaskRuntimeConfig(
            task_path=self.task_dir,
            environment=self.settings.sandbox,
            sandbox_user=self.settings.sandbox_user,
            jobs_dir=self.settings.jobs_dir,
            job_name=self.job_name,
            **integrity,
        )
        t0 = self._created_at = self._clock()
        try:
            self.runtime = await self._factory(config)
        except BaseException as exc:
            # TaskRuntime.start() already cleaned up whatever it created.
            self._closed = True
            self._release_slot()
            if not isinstance(exc, Exception):
                raise
            self.decision = sandbox_start_failure(exc)
            raise InfrastructureError(self.decision, task=self.task_dir.name) from exc
        LIVE.add(self)
        self.timings["sandbox_start_sec"] = round(self._clock() - t0, 3)
        with contextlib.suppress(Exception):
            self.rollout_dir = Path(self.runtime.rollout_dir)
        if self.settings.episode_timeout_sec > 0:
            self._deadline = self._clock() + self.settings.episode_timeout_sec

    def past_deadline(self) -> bool:
        return self._deadline is not None and self._clock() >= self._deadline

    async def run_bash(self, command: str) -> str:
        """Run one policy command in the task workspace; return what the model sees.

        As in the shared harness: stdout then stderr, cut past
        `max_output_chars`; a command that fails to run (a timeout included)
        comes back as `{"error": ...}` and the episode goes on.
        """
        if self.runtime is None or self._closed:
            return json.dumps({"error": "the sandbox is closed"})
        self.policy_acted = True
        self.commands_run += 1
        t0 = self._clock()
        try:
            result = await self.runtime.bash(
                command, timeout_sec=self.settings.command_timeout_sec
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})
        finally:
            self.timings["commands_sec"] = round(
                self.timings.get("commands_sec", 0.0) + self._clock() - t0, 3
            )
        output = result.stdout or ""
        if result.stderr:
            output = f"{output}{result.stderr}"
        return truncate(output, self.settings.max_output_chars)

    async def submit(self, answer: str) -> str | None:
        """Write the answer file, as the shared harness's submit does.

        Returns None when written (the episode should end and be verified),
        or the error text to show the model when the write failed.
        """
        if self.runtime is None or self._closed:
            return json.dumps({"error": "the sandbox is closed"})
        self.policy_acted = True
        path = self.settings.submit_path
        quoted = _quote(path)
        try:
            await self.runtime.bash(
                f"mkdir -p $(dirname {quoted}) && printf %s {_quote(str(answer))} > {quoted}",
                timeout_sec=self.settings.command_timeout_sec,
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})
        self.submitted = True
        return None

    async def finish(self) -> Decision:
        """Run the verifier on the sandbox as it is, decide, close. Idempotent."""
        if self.decision is not None:
            await self.close()
            return self.decision
        if self.runtime is None:
            raise RuntimeError("finish() before start()")
        t0 = self._clock()
        try:
            result = await asyncio.wait_for(
                self.runtime.verify(), timeout=self.settings.verify_timeout_sec
            )
        except Exception as exc:  # a crash, or the guard's TimeoutError
            # As the TRL adapter decides a verify() that raises.
            decision = (
                zero(VERIFIER_ERROR, exc)
                if self.policy_acted
                else dropped(VERIFIER_CRASH_CLEAN_RUN, exc)
            )
        else:
            with contextlib.suppress(Exception):
                self.rollout_dir = Path(result.rollout_dir)
            # With an integrity audit, an exploit scores 0 and is flagged.
            decision = reward_from_verify(
                result,
                policy_acted=self.policy_acted,
                integrity=getattr(result, "integrity", None),
            )
        finally:
            self.timings["verify_sec"] = round(self._clock() - t0, 3)
            await self.close()
        self.decision = decision
        return decision

    async def time_out(self) -> Decision:
        """End at the wall-clock budget: 0 (reason timeout), no verifier run."""
        if self.decision is None:
            budget = self.settings.episode_timeout_sec
            self.decision = zero(TIMEOUT, f"the episode ran past {budget:g} s")
        await self.close()
        return self.decision

    async def close(self) -> None:
        """Close the sandbox (idempotent) and give the slot back."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.runtime is not None:
                # Shielded: a cancellation must not cut a sandbox delete short.
                await asyncio.shield(self.runtime.close())
        except Exception as exc:
            log.warning(
                "closing the sandbox of %s: %s", self.task_dir.name, describe(exc)
            )
        finally:
            if self.runtime is not None and self._created_at is not None:
                # From the create call to the end of the delete: the sandbox's life.
                self.timings["sandbox_sec"] = round(self._clock() - self._created_at, 3)
            LIVE.discard(self)
            self._release_slot()

    def _release_slot(self) -> None:
        if self._slot_held:
            self._slot_held = False
            self.slots.release()

    @property
    def started(self) -> bool:
        """True once the sandbox started (it may be closed since)."""
        return self.runtime is not None

    @property
    def closed(self) -> bool:
        return self._closed

    # -- audit record --------------------------------------------------------

    def record(self, messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
        """The shared harness's rollout record (see trl.rollout_record), plus extras."""
        decision = self.decision
        return {
            "task_id": self.task_dir.name,
            "rollout_dir": str(self.rollout_dir) if self.rollout_dir else None,
            "step": self.job_name,
            "policy_acted": self.policy_acted,
            **(
                decision.as_dict()
                if decision is not None
                else {"reward": None, "reason": None}
            ),
            "ended": self.ended,
            "commands_run": self.commands_run,
            "timings": dict(self.timings),
            **extra,
            "messages": messages,
        }

    def write_record(self, record: dict[str, Any]) -> None:
        """Keep the rollout for audit as the shared harness does:
        `policy/messages.json` in its rollout folder, a line in `rollouts.jsonl`."""
        try:
            write_rollout_record(record, self.settings.jobs_dir)
        except OSError as exc:  # never let bookkeeping stop training
            log.warning("could not record rollout of %s: %s", self.task_dir.name, exc)


def _quote(text: str) -> str:
    return shlex.quote(text)


async def close_all_live() -> int:
    """Close every sandbox this process still holds (for shutdown paths)."""
    episodes = list(LIVE)
    await asyncio.gather(*(e.close() for e in episodes), return_exceptions=True)
    return len(episodes)


class RunStopped(RuntimeError):
    """A run stopped from outside: a signal, or its disk running low."""


def free_gib(path: Path | str) -> float:
    return shutil.disk_usage(path).free / 2**30


async def run_guarded(
    coro: Awaitable[Any],
    *,
    disk_path: Path | str | None = None,
    min_free_gib: float = 0.0,
    check_every_sec: float = 30.0,
) -> Any:
    """Run `coro`; on SIGINT, SIGTERM or low disk, cancel it cleanly.

    Cancelling lets every cleanup path run (groups close their sandboxes),
    then any sandbox still open is closed and RunStopped is raised. A job
    started in the background from a script ignores SIGINT; the handlers
    installed here take both signals regardless.
    """
    task = asyncio.ensure_future(coro)
    loop = asyncio.get_running_loop()
    reasons: list[str] = []

    def stop(reason: str) -> None:
        if not reasons:
            reasons.append(reason)
            log.error("stopping: %s", reason)
            task.cancel()

    installed = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            loop.add_signal_handler(sig, stop, f"signal {sig.name}")
            installed.append(sig)

    async def watch_disk() -> None:
        while True:
            free = free_gib(disk_path)  # type: ignore[arg-type]
            if free < min_free_gib:
                stop(f"{free:.1f} GiB free under {disk_path}, below {min_free_gib:g}")
                return
            await asyncio.sleep(check_every_sec)

    watcher = (
        asyncio.create_task(watch_disk())
        if disk_path is not None and min_free_gib > 0
        else None
    )
    try:
        return await task
    except asyncio.CancelledError:
        if reasons:
            raise RunStopped(reasons[0]) from None
        raise
    finally:
        if watcher is not None:
            watcher.cancel()
        for sig in installed:
            loop.remove_signal_handler(sig)
        left = await close_all_live()
        if left:
            log.warning("closed %d sandboxes left open", left)


# -- drops -----------------------------------------------------------------------


class TooManyInfrastructureFailures(RuntimeError):
    """Consecutive drops past the limit: something systematic is wrong."""


@dataclass
class DropLog:
    """Infrastructure drops, counted by reason and kept as JSON lines.

    Drops are for rare failures. `max_consecutive` drops in a row with no
    episode finishing in between is not rare: a misconfigured sandbox, a broken
    task image or an outage fails every episode the same way, so the run stops
    instead of skipping every group.
    """

    path: Path | None = None
    counts: dict[str, int] = field(default_factory=dict)
    max_consecutive: int = 8
    consecutive: int = 0
    last: str = ""

    def completed(self) -> None:
        self.consecutive = 0

    def add(self, exc: BaseException, *, task: str, where: str) -> None:
        reason = getattr(exc, "reason", None) or MODEL_ENDPOINT
        self.counts[reason] = self.counts.get(reason, 0) + 1
        self.consecutive += 1
        self.last = describe(exc)
        log.warning("dropped (%s, %s) %s: %s", reason, where, task, self.last)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                row = {
                    "time": time.time(),
                    "where": where,
                    "task": task,
                    "reason": reason,
                    "error": self.last[:2000],
                }
                f.write(json.dumps(row) + "\n")
        if self.consecutive >= self.max_consecutive:
            raise TooManyInfrastructureFailures(
                f"{self.consecutive} infrastructure failures in a row, the last: {self.last}"
            )


# -- groups ------------------------------------------------------------------------


def group_kind(rewards: list[float]) -> str:
    """`mixed` when the group's rewards differ; otherwise what they all were."""
    if len(set(rewards)) > 1:
        return "mixed"
    if not rewards:
        return "empty"
    if rewards[0] >= 1.0:
        return "all_solved"
    if rewards[0] <= 0.0:
        return "all_failed"
    return "constant_partial"


@dataclass
class GroupLog:
    """Every finished group's rewards, counted by kind and kept as JSON lines.

    A group whose episodes all got the same reward has zero advantage for
    every episode, so it teaches nothing; training drops it and counts it here.
    """

    path: Path | None = None
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, rewards: list[float], *, task: str, where: str, dropped: bool) -> str:
        kind = group_kind(rewards)
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                row = {
                    "where": where,
                    "task": task,
                    "kind": kind,
                    "rewards": rewards,
                    "dropped": dropped,
                }
                f.write(json.dumps(row) + "\n")
        return kind
