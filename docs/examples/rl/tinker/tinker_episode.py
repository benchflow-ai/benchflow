"""One training episode on one BenchFlow task sandbox, and the training rule.

This half of the Tinker cookbook uses BenchFlow only (no Tinker import), so its
logic is testable anywhere. `tinker_env.py` wraps it for tinker-cookbook's RL
framework.

An `Episode` starts the task's sandbox with `bf.TaskRuntime`, runs the
policy's `run_bash` commands and `submit` answer in it the way the shared RL
harness does (BenchFlow's TRL adapter and the held-out evaluator), runs the
task's verifier, and always closes the sandbox. The training rule turns each
ending into a `Decision`:

- dropped (reward None): an infrastructure failure the policy could not have
  caused: the sandbox never started, the model endpoint failed, or the
  verifier gave no reward on a clean run (the policy never acted). Dropped
  episodes leave the batch and are counted by reason.
- scored: the verifier's reward.
- 0, with a named reason, for every other failure: `timeout` (including the
  episode's wall-clock budget), `verifier_error`, `run_error`, `no_reward`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import shlex
import time
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import benchflow as bf

log = logging.getLogger(__name__)

# -- the shared harness --------------------------------------------------------
# The same values as docs/examples/rl/common/harness.py, so that a policy trained
# here is evaluated with the tools, instructions and limits it trained with.

HARNESS_MESSAGE = (
    "\n\nYou are working in a Linux sandbox. Use the run_bash tool to run shell "
    "commands; each call starts in /workdir. When you are done, call the submit tool "
    "once with your final answer: it writes the answer to /workdir/answer.txt for you "
    "and ends the task. For a code fix, submit the word done."
)
BASH_TIMEOUT_SEC = 30
MAX_OUTPUT_CHARS = 2000
MAX_TURNS = 10
SUBMIT_PATH = "/workdir/answer.txt"
TRUNCATION_MARKER = "\n[benchflow output truncated]\n"

# -- the training rule ---------------------------------------------------------
# The reasons of benchflow.integrations.rewards (branch cookbook/rl-core).

SCORED = "scored"
TIMEOUT = "timeout"
VERIFIER_ERROR = "verifier_error"
RUN_ERROR = "run_error"
NO_REWARD = "no_reward"
ZERO_REASONS = (TIMEOUT, VERIFIER_ERROR, RUN_ERROR, NO_REWARD)
SANDBOX_START = "sandbox_start"
MODEL_ENDPOINT = "model_endpoint"
VERIFIER_CRASH_CLEAN_RUN = "verifier_crash_clean_run"
DROP_REASONS = (SANDBOX_START, MODEL_ENDPOINT, VERIFIER_CRASH_CLEAN_RUN)


@dataclass(frozen=True)
class Decision:
    """A training reward, or a drop (reward None) with its reason."""

    reward: float | None
    reason: str
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.reward is None and self.reason not in DROP_REASONS:
            raise ValueError(
                f"a drop needs an infrastructure reason, got {self.reason!r}"
            )
        if self.reward is not None and self.reason in DROP_REASONS:
            raise ValueError(f"{self.reason!r} is a drop; its reward must be None")

    @property
    def dropped(self) -> bool:
        return self.reward is None

    @property
    def solved(self) -> bool:
        return self.reward is not None and self.reward >= 1.0


def _short(detail: object) -> str | None:
    if detail is None:
        return None
    if isinstance(detail, BaseException):
        text = f"{type(detail).__name__}: {detail}"
    else:
        text = str(detail)
    text = " ".join(text.split())
    return (text[:497] + "...") if len(text) > 500 else (text or None)


def scored(reward: float) -> Decision:
    return Decision(float(reward), SCORED)


def zero(reason: str, detail: object = None) -> Decision:
    if reason not in ZERO_REASONS:
        raise ValueError(f"unknown zero reason {reason!r}")
    return Decision(0.0, reason, _short(detail))


def dropped(reason: str, detail: object = None) -> Decision:
    return Decision(None, reason, _short(detail))


def _finite(value: Any) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def decide(result: Any, *, policy_acted: bool) -> Decision:
    """A verify result (TaskRuntimeResult) as a reward or a drop.

    Mirrors benchflow.integrations.rewards.reward_from_verify: a finite reward
    is the score; without one, a sandbox that failed to start is a drop, a
    verifier that gave nothing on a clean run is a drop, and everything after
    the policy acted scores 0 by what went wrong.
    """
    reward = _finite(getattr(result, "reward", None))
    if reward is None:
        rewards = getattr(result, "rewards", None)
        if isinstance(rewards, dict):
            reward = _finite(rewards.get("reward"))
    if reward is not None:
        return scored(reward)
    error = getattr(result, "error", None) or ""
    verifier_error = getattr(result, "verifier_error", None) or ""
    detail = verifier_error or error or None
    if "sandbox startup" in error.lower() or "sandbox creation" in error.lower():
        return dropped(SANDBOX_START, detail)
    if not policy_acted:
        return dropped(VERIFIER_CRASH_CLEAN_RUN, detail)
    if "verifier timed out" in verifier_error or "timed out" in error.lower():
        return zero(TIMEOUT, detail)
    if verifier_error:
        return zero(VERIFIER_ERROR, detail)
    if error:
        return zero(RUN_ERROR, detail)
    return zero(NO_REWARD)


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
        config = bf.TaskRuntimeConfig(
            task_path=self.task_dir,
            environment=self.settings.sandbox,
            sandbox_user=self.settings.sandbox_user,
            jobs_dir=self.settings.jobs_dir,
            job_name=self.job_name,
        )
        t0 = self._clock()
        try:
            self.runtime = await self._factory(config)
        except BaseException as exc:
            # TaskRuntime.start() already cleaned up whatever it created.
            self._closed = True
            self._release_slot()
            if not isinstance(exc, Exception):
                raise
            self.decision = dropped(SANDBOX_START, exc)
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
            decision = (
                zero(VERIFIER_ERROR, exc)
                if self.policy_acted
                else dropped(VERIFIER_CRASH_CLEAN_RUN, exc)
            )
        else:
            with contextlib.suppress(Exception):
                self.rollout_dir = Path(result.rollout_dir)
            decision = decide(result, policy_acted=self.policy_acted)
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
        """The rollout record of the shared harness (rollout_record), plus extras."""
        decision = self.decision
        return {
            "task_id": self.task_dir.name,
            "rollout_dir": str(self.rollout_dir) if self.rollout_dir else None,
            "step": self.job_name,
            "policy_acted": self.policy_acted,
            **(
                {
                    "reward": decision.reward,
                    "dropped": decision.dropped,
                    "reason": decision.reason,
                    "detail": decision.detail,
                    "flagged": False,
                }
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
        """Keep the rollout for audit, like the shared harness's write_rollout_record:
        `policy/messages.json` in its rollout folder, a line in `rollouts.jsonl`."""
        text = json.dumps(record, default=str)
        try:
            if self.rollout_dir is not None:
                policy_dir = self.rollout_dir / "policy"
                policy_dir.mkdir(parents=True, exist_ok=True)
                (policy_dir / "messages.json").write_text(text + "\n")
            jobs = Path(self.settings.jobs_dir)
            jobs.mkdir(parents=True, exist_ok=True)
            with (jobs / "rollouts.jsonl").open("a") as f:
                f.write(text + "\n")
        except OSError as exc:  # never let bookkeeping stop training
            log.warning("could not record rollout of %s: %s", self.task_dir.name, exc)


def _quote(text: str) -> str:
    return shlex.quote(text)


async def close_all_live() -> int:
    """Close every sandbox this process still holds (for shutdown paths)."""
    episodes = list(LIVE)
    await asyncio.gather(*(e.close() for e in episodes), return_exceptions=True)
    return len(episodes)


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


def decision_dict(decision: Decision | None) -> dict[str, Any] | None:
    return asdict(decision) if decision is not None else None


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
