"""Sync and async entry points for one rollout or many, sharing one code path.

Every function here ends in :func:`arun`, the async entry point
(``bf.run`` is the same function, kept for existing ``await bf.run(...)``
code). The synchronous variants drive the async ones on a private event loop:

- ``await bf.arun(config)`` / ``bf.run_sync(config)``: one rollout.
- ``async for done in bf.as_completed(configs, concurrency=4)``: many rollouts,
  yielded as each finishes.
- ``await bf.arun_batch(configs)`` / ``bf.run_batch(configs)``: many rollouts,
  collected in input order into a :class:`Results` list that exports to
  records, CSV and JSONL without pandas.

For every task under a directory, with retries, resume and ``summary.json``,
use :class:`benchflow.Evaluation` instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import csv
import json
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from benchflow.models import RolloutResult
from benchflow.runtime import run as arun

if TYPE_CHECKING:
    from benchflow.jobs import Denominators
    from benchflow.rollout import RolloutConfig

logger = logging.getLogger(__name__)

__all__ = [
    "Completed",
    "Results",
    "arun",
    "arun_batch",
    "as_completed",
    "run_batch",
    "run_blocking",
    "run_sync",
    "write_csv",
    "write_jsonl",
]


def run_blocking[T](factory: Callable[[], Awaitable[T]]) -> T:
    """Run the awaitable ``factory()`` returns to completion from sync code.

    Without a running event loop this is ``asyncio.run``. Inside one (a
    Jupyter cell, an async framework's sync callback) the work runs on a
    fresh loop in a helper thread and this call blocks until it finishes, so
    the same script works in both places.
    """

    async def _main() -> T:
        try:
            return await factory()
        finally:
            await _close_loop_clients()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_main())

    outcome: dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["value"] = asyncio.run(_main())
        except BaseException as exc:  # re-raised in the caller's thread
            outcome["error"] = exc

    worker = threading.Thread(target=_target, name="benchflow-run-sync")
    worker.start()
    worker.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


async def _close_loop_clients() -> None:
    """Close a Daytona client created on this (private, about to close) loop.

    The client is a process-wide singleton bound to the loop that created it;
    left open, it is garbage-collected after the loop closes and asyncio logs
    "Unclosed client session". The next loop creates a fresh one.
    """
    import sys

    daytona = sys.modules.get("benchflow.sandbox.daytona")
    manager = getattr(getattr(daytona, "DaytonaClientManager", None), "_instance", None)
    if (
        manager is not None
        and manager._client is not None
        and manager._client_loop is asyncio.get_running_loop()
    ):
        with contextlib.suppress(Exception):
            await manager._cleanup()


def run_sync(*args: Any, **kwargs: Any) -> RolloutResult:
    """Blocking form of :func:`arun`; takes exactly the same arguments.

    >>> import benchflow as bf
    >>> bf.run_sync.__doc__.splitlines()[0]
    'Blocking form of :func:`arun`; takes exactly the same arguments.'
    """
    return run_blocking(lambda: arun(*args, **kwargs))


class Completed(NamedTuple):
    """One finished rollout from :func:`as_completed`.

    ``index`` is the config's position in the input, so results can be matched
    to configs even when several share a task.
    """

    index: int
    config: RolloutConfig
    result: RolloutResult


def _check_configs(items: list[Any]) -> None:
    """Refuse a bad batch before any rollout starts."""
    from benchflow.rollout import RolloutConfig
    from benchflow.runtime import check_rollout_config

    for index, item in enumerate(items):
        if not isinstance(item, RolloutConfig):
            raise TypeError(
                f"batch item {index} is a {type(item).__name__}, not a "
                "RolloutConfig; build one per run, e.g. "
                "bf.RolloutConfig(task_path=..., agent=..., model=...)"
            )
        check_rollout_config(item)


def _with_job_name(config: RolloutConfig, job_name: str) -> RolloutConfig:
    if config.job_name is not None:
        return config
    shared = copy.copy(config)
    shared.job_name = job_name
    return shared


async def _run_one(config: RolloutConfig) -> RolloutResult:
    """Run one config; an exception becomes an errored result, not a crash."""
    try:
        return await arun(config)
    except Exception as exc:
        logger.exception("rollout for %s raised", config.task_path)
        return RolloutResult(
            task_name=Path(config.task_path).name,
            agent=config.primary_agent or "",
            model=config.primary_model,
            error=f"unexpected exception: {type(exc).__name__}: {exc}",
            error_category="other",
        )


async def as_completed(
    configs: Iterable[RolloutConfig], *, concurrency: int = 4
) -> AsyncIterator[Completed]:
    """Run rollouts with at most ``concurrency`` at a time, yielding each as it finishes.

    Configs without a ``job_name`` share one, so the batch's rollouts land in
    one job directory under their ``jobs_dir``.

    A rollout that raises is yielded as a result with ``error`` set, so one bad
    config does not stop the others. Leaving the loop early cancels the
    rollouts still running.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    items = list(configs)
    _check_configs(items)
    from benchflow.runtime import _HOST_CHECKED, check_host

    check_host(items)
    host_checked = _HOST_CHECKED.set(True)
    # One job directory per batch: configs without a job_name share one, in
    # the same timestamp format a single rollout uses. Callers' configs are
    # not modified.
    shared_job = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    runnable = [_with_job_name(c, shared_job) for c in items]
    gate = asyncio.Semaphore(concurrency)
    finished: asyncio.Queue[Completed] = asyncio.Queue()

    async def _one(index: int, config: RolloutConfig) -> None:
        async with gate:
            result = await _run_one(runnable[index])
        await finished.put(Completed(index, config, result))

    tasks = [asyncio.create_task(_one(i, c)) for i, c in enumerate(items)]
    try:
        for _ in items:
            yield await finished.get()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # An async generator finalized by the loop's shutdown hook runs in
        # another context, where the token cannot be reset; nothing leaks then.
        with contextlib.suppress(ValueError):
            _HOST_CHECKED.reset(host_checked)


_RECORD_FIELDS = [
    "task_name",
    "rollout_name",
    "agent",
    "model",
    "reward",
    "passed",
    "score_outcome",
    "execution",
    "assessment",
    "error",
    "error_category",
    "verifier_error",
    "verifier_error_category",
    "n_tool_calls",
    "n_prompts",
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_read_tokens",
    "n_cache_creation_tokens",
    "total_tokens",
    "cost_usd",
    "usage_source",
    "started_at",
    "finished_at",
    "duration_sec",
    "rollout_dir",
]


def _record(result: RolloutResult) -> dict[str, Any]:
    """``to_record`` plus the viewer's execution and assessment."""
    from benchflow.jobs import Trial

    trial = Trial(path=result.rollout_dir or Path("."), result=result, raw={})
    return {
        **result.to_record(),
        "execution": trial.execution,
        "assessment": trial.assessment,
    }


def write_csv(results: Iterable[RolloutResult], path: str | Path) -> Path:
    """Write one CSV row per result (the :meth:`RolloutResult.to_record` columns)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_RECORD_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(_record(result))
    return path


def write_jsonl(results: Iterable[RolloutResult], path: str | Path) -> Path:
    """Write one JSON object per line per result (the ``to_record`` fields)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for result in results:
            handle.write(json.dumps(_record(result)) + "\n")
    return path


class Results(list[RolloutResult]):
    """A list of :class:`RolloutResult` with summary numbers and exports.

    >>> from benchflow import RolloutResult
    >>> rs = Results([RolloutResult("a", rewards={"reward": 1.0}),
    ...               RolloutResult("b", rewards={"reward": 0.0})])
    >>> rs.n_passed, rs.mean_reward
    (1, 0.5)
    >>> [r["task_name"] for r in rs.to_records()]
    ['a', 'b']
    """

    @property
    def n_passed(self) -> int:
        """How many results passed."""
        return sum(1 for r in self if r.passed)

    @property
    def n_scored(self) -> int:
        """How many results have a reward."""
        return self.denominators(include_controls=True).scored

    @property
    def n_errored(self) -> int:
        """How many results ended with an agent-side error (errored or timed out)."""
        return self.denominators(include_controls=True).execution_errors

    def denominators(self, *, include_controls: bool = False) -> Denominators:
        """Attempted, scored, errors and pass rates, counted as ``Job.denominators``.

        Control runs (oracle, empty) are left out unless ``include_controls=True``.
        """
        from benchflow.jobs import Denominators, Trial

        trials = [
            Trial(path=r.rollout_dir or Path("."), result=r, raw={}) for r in self
        ]
        kept = [t for t in trials if include_controls or t.control is None]
        return Denominators.of(kept, controls_excluded=len(trials) - len(kept))

    @property
    def mean_reward(self) -> float | None:
        """Mean ``reward`` over scored results; None when nothing was scored."""
        rewards = [r.reward for r in self if r.reward is not None]
        return sum(rewards) / len(rewards) if rewards else None

    def to_records(self) -> list[dict[str, Any]]:
        """One flat, JSON-safe dict per result, e.g. for ``pandas.DataFrame(...)``."""
        return [_record(r) for r in self]

    def to_csv(self, path: str | Path) -> Path:
        """Write the records to a CSV file and return its path."""
        return write_csv(self, path)

    def to_jsonl(self, path: str | Path) -> Path:
        """Write the records to a JSONL file and return its path."""
        return write_jsonl(self, path)


async def arun_batch(
    configs: Iterable[RolloutConfig],
    *,
    concurrency: int = 4,
    on_result: Callable[[Completed], None] | None = None,
) -> Results:
    """Run rollouts with bounded concurrency; return their results in input order.

    ``on_result`` is called with each :class:`Completed` as it finishes (for
    progress output). A rollout that raises comes back with ``error`` set.
    """
    items = list(configs)
    ordered: list[RolloutResult | None] = [None] * len(items)
    async for done in as_completed(items, concurrency=concurrency):
        ordered[done.index] = done.result
        if on_result is not None:
            on_result(done)
    return Results(r for r in ordered if r is not None)


def run_batch(
    configs: Iterable[RolloutConfig],
    *,
    concurrency: int = 4,
    on_result: Callable[[Completed], None] | None = None,
) -> Results:
    """Blocking form of :func:`arun_batch`."""
    items = list(configs)
    return run_blocking(
        lambda: arun_batch(items, concurrency=concurrency, on_result=on_result)
    )
