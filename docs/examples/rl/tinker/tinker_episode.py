"""One training episode on one BenchFlow task sandbox, and the training rule.

This half of the Tinker cookbook uses BenchFlow only (no Tinker import), so its
logic is testable anywhere. `tinker_env.py` wraps it for tinker-cookbook's RL
framework.

An `Episode` starts the task's sandbox with `bf.TaskRuntime`, runs the
policy's bash commands in it, runs the task's verifier, and always closes the
sandbox. The training rule decides what each ending is worth:

- infrastructure: the policy could not have caused the failure (the sandbox
  never started, or the verifier failed on a clean run, where the policy ran
  no command). The episode is dropped from the batch and counted; its
  `Outcome.reward` is None.
- scored: the verifier ran and returned a reward; that reward is the score.
- policy failure: everything else scores 0 and keeps a named exit status
  (`verifier_error`, `sandbox_lost`, `timeout`, `parse_error`, ...).
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

SCORED = "scored"
POLICY_FAILURE = "policy_failure"
INFRASTRUCTURE = "infrastructure"

# Exit statuses. Only "verified" carries the verifier's reward; the rest are
# policy failures and score 0.
EXIT_VERIFIED = "verified"
EXIT_VERIFIER_ERROR = (
    "verifier_error"  # the verifier gave no reward after the policy acted
)
EXIT_SANDBOX_LOST = "sandbox_lost"  # the sandbox stopped answering a policy command
EXIT_TIMEOUT = "timeout"  # the episode ran past its wall-clock budget
EXITS = (
    EXIT_VERIFIED,
    EXIT_VERIFIER_ERROR,
    EXIT_SANDBOX_LOST,
    EXIT_TIMEOUT,
    "parse_error",  # malformed tool calls past the retry budget, or broken framing
    "max_tokens",  # a turn cut off at the per-turn token limit
    "context_overflow",  # the conversation outgrew max_trajectory_tokens
    "prompt_too_long",  # the task prompt alone did not fit
    "cut_off",  # the rollout was stopped from outside the episode
)


@dataclass(frozen=True)
class Outcome:
    """How an episode ended, after the training rule."""

    status: str  # SCORED, POLICY_FAILURE or INFRASTRUCTURE
    exit: str  # named exit status
    reward: float | None  # None only for INFRASTRUCTURE (the episode is dropped)
    detail: str = ""

    @property
    def solved(self) -> bool:
        return self.reward is not None and self.reward >= 1.0


class InfrastructureError(RuntimeError):
    """A failure the policy could not have caused: drop the episode, count it."""

    exit = INFRASTRUCTURE

    def __init__(self, detail: str, *, task: str = "") -> None:
        super().__init__(f"{self.exit}: {detail}" + (f" (task {task})" if task else ""))
        self.detail = detail
        self.task = task


class SandboxStartError(InfrastructureError):
    """The task's sandbox never started; the policy had not acted yet."""

    exit = "sandbox_start"


class VerifierCrashOnCleanRun(InfrastructureError):
    """The verifier gave no reward although the policy ran no command."""

    exit = "verifier_crash_clean_run"


def score_verifier_run(
    *, reward: Any, verifier_error: str | None, commands_run: int
) -> Outcome:
    """The training rule for an episode whose verifier ran.

    A finite numeric reward is the score, whatever else was reported. Without
    one, the verifier crashed, timed out or wrote nothing: that is
    infrastructure only on a clean run (the policy ran no command, so it could
    not have broken the sandbox or the tests); otherwise it scores 0.
    """
    if (
        isinstance(reward, int | float)
        and not isinstance(reward, bool)
        and math.isfinite(reward)
    ):
        return Outcome(SCORED, EXIT_VERIFIED, float(reward))
    detail = verifier_error or "the verifier produced no reward"
    if commands_run == 0:
        return Outcome(INFRASTRUCTURE, VerifierCrashOnCleanRun.exit, None, detail)
    return Outcome(POLICY_FAILURE, EXIT_VERIFIER_ERROR, 0.0, detail)


def policy_failure(exit: str, detail: str = "") -> Outcome:
    """An ending the policy caused without a verifier run: it scores 0."""
    return Outcome(POLICY_FAILURE, exit, 0.0, detail)


def describe(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


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
    """How episodes run their sandboxes. Picklable, like tinker-cookbook needs."""

    sandbox: str = (
        "daytona"  # any BenchFlow sandbox backend: daytona, docker, modal, ...
    )
    sandbox_user: str | None = "agent"  # the policy's commands never run as root
    jobs_dir: str = "trials"  # BenchFlow's trial layout: <jobs_dir>/<job>/<task>__<id>/
    command_timeout_sec: int = 60  # per run_bash call
    max_output_chars: int = 6000  # per run_bash result shown to the model
    episode_timeout_sec: float = 900.0  # wall clock once the sandbox is up; 0 = none
    verify_timeout_sec: float = 900.0  # guard around BenchFlow's verify and finalize


RuntimeFactory = Callable[[Any], Awaitable[Any]]

# Every episode whose sandbox may be live, so a shutdown can close them all.
LIVE: set[Episode] = set()


class Episode:
    """One task sandbox, driven by an external policy: start, bash, finish, close.

    `close()` is idempotent and runs on every path: after `finish()` (which
    verifies), after `abandon()` (which does not), when `start()` fails, and
    from the caller's cleanup when the episode was cut off from outside.
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
        self.outcome: Outcome | None = None
        self.commands_run = 0
        self.submitted = False
        self.lost = False
        self.lost_detail = ""
        self.timings: dict[str, float] = {}
        self._deadline: float | None = None
        self._started = False
        self._closed = False
        self._slot_held = False

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Take a sandbox slot and start the task sandbox.

        Raises SandboxStartError (infrastructure) when the sandbox does not
        start; the slot is returned and nothing is left running.
        """
        if self._started or self._closed:
            raise RuntimeError("an Episode starts once")
        self._started = True
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
            self.outcome = Outcome(
                INFRASTRUCTURE, SandboxStartError.exit, None, describe(exc)
            )
            raise SandboxStartError(describe(exc), task=self.task_dir.name) from exc
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

        The command runs in a fresh `bash -c` under `timeout`, with stdin from
        /dev/null, as the sandbox user. A timeout is reported to the model and
        the episode goes on. A sandbox that stops answering marks the episode
        lost: the caller should end it (it scores 0, since the policy's own
        commands could have caused it).
        """
        if self.runtime is None or self._closed:
            return "[not run: the sandbox is closed]"
        if self.submitted:
            return "[not run: the task was already submitted]"
        if self.lost:
            return "[not run: the sandbox stopped responding]"
        limit = self.settings.command_timeout_sec
        # The inner timeout fires well before the exec deadline, so a slow
        # command comes back as exit 124 rather than as an exec error.
        wrapped = f"timeout -k 5 {limit} bash -c {shlex.quote(command)} </dev/null"
        self.commands_run += 1
        t0 = self._clock()
        try:
            result = await self.runtime.bash(wrapped, timeout_sec=limit + 30)
        except Exception as exc:
            if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
                return (
                    f"[the command did not return within {limit} s: a process it "
                    "started may still hold its output open; run background jobs "
                    "as `cmd >log 2>&1 &`]"
                )
            self.lost = True
            self.lost_detail = describe(exc)
            log.warning("sandbox lost in %s: %s", self.task_dir.name, self.lost_detail)
            return "[the sandbox stopped responding; the episode ends]"
        finally:
            self.timings["commands_sec"] = round(
                self.timings.get("commands_sec", 0.0) + self._clock() - t0, 3
            )
        text = format_command_output(
            result.return_code,
            result.stdout,
            result.stderr,
            self.settings.max_output_chars,
        )
        if result.return_code in (124, 137):
            text += f"\n[the command timed out after {limit} s]"
        return text

    async def finish(self) -> Outcome:
        """Run the verifier on the sandbox as it is, apply the rule, close.

        Called when the policy submits, stops calling tools, or runs out of
        turns. Idempotent: a second call returns the first outcome.
        """
        if self.outcome is not None:
            await self.close()
            return self.outcome
        if self.runtime is None:
            raise RuntimeError("finish() before start()")
        if self.lost:
            self.outcome = policy_failure(EXIT_SANDBOX_LOST, self.lost_detail)
            await self.close()
            return self.outcome
        t0 = self._clock()
        try:
            result = await asyncio.wait_for(
                self.runtime.verify(), timeout=self.settings.verify_timeout_sec
            )
        except Exception as exc:  # a crash, or the guard's TimeoutError
            outcome = score_verifier_run(
                reward=None,
                verifier_error=f"verify raised {describe(exc)}",
                commands_run=self.commands_run,
            )
        else:
            with contextlib.suppress(Exception):
                self.rollout_dir = Path(result.rollout_dir)
            outcome = score_verifier_run(
                reward=result.reward,
                verifier_error=result.verifier_error or result.error,
                commands_run=self.commands_run,
            )
        finally:
            self.timings["verify_sec"] = round(self._clock() - t0, 3)
            await self.close()
        self.outcome = outcome
        return outcome

    async def abandon(self, exit: str, detail: str = "") -> Outcome:
        """End without a verifier run (a policy failure: it scores 0), close."""
        if self.outcome is None:
            self.outcome = policy_failure(exit, detail)
        await self.close()
        return self.outcome

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
                "closing the sandbox of %s failed: %s",
                self.task_dir.name,
                describe(exc),
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

    # -- record ------------------------------------------------------------

    def record(self, **extra: Any) -> dict[str, Any]:
        return {
            "task": self.task_dir.name,
            "job": self.job_name,
            "rollout_dir": str(self.rollout_dir) if self.rollout_dir else None,
            "outcome": asdict(self.outcome) if self.outcome else None,
            "commands_run": self.commands_run,
            "submitted": self.submitted,
            "timings": dict(self.timings),
            **extra,
        }

    def write_record(self, record: dict[str, Any]) -> Path | None:
        """Write the episode record next to BenchFlow's own trial files."""
        if self.rollout_dir is None:
            return None
        path = self.rollout_dir / "tinker_episode.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1, default=str))
        except OSError as exc:
            log.warning("could not write %s: %s", path, exc)
            return None
        return path


async def close_all_live() -> int:
    """Close every sandbox this process still holds (for shutdown paths)."""
    episodes = list(LIVE)
    await asyncio.gather(*(e.close() for e in episodes), return_exceptions=True)
    return len(episodes)


def format_command_output(code: int, stdout: str, stderr: str, limit: int) -> str:
    """Exit code plus stdout and stderr, cut in the middle past `limit` chars."""
    parts = [f"exit code {code}"]
    if stdout:
        parts.append(f"stdout:\n{stdout.rstrip()}")
    if stderr:
        parts.append(f"stderr:\n{stderr.rstrip()}")
    if not stdout and not stderr:
        parts.append("(no output)")
    return cut_middle("\n".join(parts), limit)


def cut_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = f"\n[... {len(text) - limit} characters cut ...]\n"
    keep = max(limit - len(marker), 0)
    head = keep * 2 // 3
    return text[:head] + marker + text[len(text) - (keep - head) :]


class TooManyInfrastructureFailures(RuntimeError):
    """Consecutive drops past the limit: something systematic is wrong."""


@dataclass
class DropLog:
    """Infrastructure drops, counted per exit status and kept as JSON lines.

    Drops are for rare failures. `max_consecutive` drops in a row, with no
    episode completing in between, is not rare: a misconfigured sandbox, a
    broken task image or an outage fails every episode the same way, so the
    run stops instead of skipping every group.
    """

    path: Path | None = None
    counts: dict[str, int] = field(default_factory=dict)
    max_consecutive: int = 8
    consecutive: int = 0
    last: str = ""

    def completed(self) -> None:
        self.consecutive = 0

    def add(self, exc: BaseException, *, task: str, where: str) -> None:
        exit = getattr(exc, "exit", type(exc).__name__)
        self.counts[exit] = self.counts.get(exit, 0) + 1
        self.consecutive += 1
        self.last = describe(exc)
        log.warning("dropped (%s) %s: %s", where, task, self.last)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                record = {
                    "time": time.time(),
                    "where": where,
                    "task": task,
                    "exit": exit,
                    "error": self.last[:2000],
                }
                f.write(json.dumps(record) + "\n")
        if self.consecutive >= self.max_consecutive:
            raise TooManyInfrastructureFailures(
                f"{self.consecutive} infrastructure failures in a row, the last: {self.last}"
            )
