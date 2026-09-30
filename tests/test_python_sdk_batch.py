"""Sync and async entry points and batches.

``bf.run`` was async only and there was no way to run a list of configs with
bounded concurrency short of writing a task directory for ``Evaluation``.
``benchflow.batch`` adds ``arun``/``run_sync``, ``as_completed``,
``arun_batch``/``run_batch`` and a ``Results`` list with pandas-free exports,
all through the one ``arun`` path.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import json
from pathlib import Path

import pytest

import benchflow as bf
from benchflow import batch
from benchflow.models import RolloutResult
from benchflow.rollout import RolloutConfig

TASK = Path(__file__).parent / "examples" / "hello-world-task"


def _cfg(name: str) -> RolloutConfig:
    return RolloutConfig(task_path=TASK, agent="oracle", rollout_name=name)


@pytest.fixture
def fake_arun(monkeypatch):
    """Replace the one shared path; each config's rollout_name drives the fake."""
    state = {"in_flight": 0, "max_in_flight": 0, "started": []}

    async def fake(config, *a, **kw):
        name = config.rollout_name
        state["started"].append(name)
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        try:
            if name == "boom":
                raise RuntimeError("bad config")
            # Later names finish first, so completion order != input order.
            await asyncio.sleep({"a": 0.06, "b": 0.03}.get(name, 0.01))
            return RolloutResult(
                task_name="hello-world-task",
                rollout_name=name,
                rewards={"reward": 1.0 if name != "b" else 0.0},
            )
        finally:
            state["in_flight"] -= 1

    monkeypatch.setattr(batch, "arun", fake)
    return state


def test_public_names() -> None:
    for name in (
        "arun",
        "run_sync",
        "as_completed",
        "arun_batch",
        "run_batch",
        "Results",
    ):
        assert name in bf.__all__ and getattr(bf, name) is getattr(batch, name)
    assert bf.arun is bf.run  # bf.run stays the (async) same function


def test_run_sync_outside_a_loop(fake_arun) -> None:
    assert bf.run_sync(_cfg("c")).rollout_name == "c"


@pytest.mark.asyncio
async def test_run_sync_inside_a_running_loop(fake_arun) -> None:
    """As in a Jupyter cell: a running loop, but a blocking call still works."""
    assert bf.run_sync(_cfg("c")).rollout_name == "c"


@pytest.mark.asyncio
async def test_as_completed_yields_in_finish_order_with_bounded_concurrency(
    fake_arun,
) -> None:
    order = []
    async with contextlib.aclosing(
        bf.as_completed([_cfg("a"), _cfg("b"), _cfg("c")], concurrency=2)
    ) as stream:
        async for done in stream:
            order.append((done.index, done.result.rollout_name))
    assert order == [(1, "b"), (2, "c"), (0, "a")]
    assert fake_arun["max_in_flight"] == 2


@pytest.mark.asyncio
async def test_arun_batch_keeps_input_order_and_reports_progress(fake_arun) -> None:
    seen = []
    results = await bf.arun_batch(
        [_cfg("a"), _cfg("b"), _cfg("c")], concurrency=3, on_result=seen.append
    )
    assert [r.rollout_name for r in results] == ["a", "b", "c"]
    assert sorted(d.index for d in seen) == [0, 1, 2]
    assert results.n_passed == 2
    assert results.mean_reward == pytest.approx(2 / 3)


def test_run_batch_turns_an_exception_into_an_errored_result(fake_arun) -> None:
    results = bf.run_batch([_cfg("boom"), _cfg("c")], concurrency=1)
    assert results[0].error == "unexpected exception: RuntimeError: bad config"
    assert results[0].score_outcome == "errored"
    assert results[1].passed


@pytest.mark.asyncio
async def test_leaving_early_cancels_the_rest(fake_arun) -> None:
    async with contextlib.aclosing(
        bf.as_completed([_cfg("c"), _cfg("a"), _cfg("a")], concurrency=3)
    ) as stream:
        async for _ in stream:
            break
    await asyncio.sleep(0)
    assert fake_arun["in_flight"] == 0


def test_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        bf.run_batch([_cfg("c")], concurrency=0)


def _results() -> bf.Results:
    return bf.Results(
        [
            RolloutResult(
                "t1",
                rollout_name="t1__1",
                agent="oracle",
                rewards={"reward": 1.0},
                n_input_tokens=3,
                rollout_dir=Path("/tmp/x/t1__1"),
            ),
            RolloutResult("t2", error="agent crashed", error_category="other"),
        ]
    )


def test_to_record_is_flat_and_json_safe() -> None:
    record = _results()[0].to_record()
    assert record["reward"] == 1.0 and record["passed"] is True
    assert record["score_outcome"] == "passed"
    assert record["rollout_dir"] == "/tmp/x/t1__1"
    json.dumps(record)


def test_csv_and_jsonl_exports(tmp_path: Path) -> None:
    rs = _results()
    rows = list(csv.DictReader(rs.to_csv(tmp_path / "r.csv").open()))
    assert [r["task_name"] for r in rows] == ["t1", "t2"]
    assert rows[1]["error"] == "agent crashed" and rows[1]["passed"] == "False"
    lines = rs.to_jsonl(tmp_path / "r.jsonl").read_text().splitlines()
    assert [json.loads(line)["task_name"] for line in lines] == ["t1", "t2"]


def test_a_batch_shares_one_job_directory(monkeypatch) -> None:
    """Regression test: run_batch put its two rollouts under two timestamped job
    directories (their start times differed by a second)."""
    seen: list[RolloutConfig] = []

    async def fake(config, *a, **kw):
        seen.append(config)
        await asyncio.sleep(0)
        return RolloutResult("hello-world-task", rollout_name=config.rollout_name)

    monkeypatch.setattr(batch, "arun", fake)
    mine = [_cfg("a"), _cfg("b")]
    pinned = RolloutConfig(task_path=TASK, agent="oracle", job_name="kept")
    bf.run_batch([*mine, pinned], concurrency=3)

    by_name = {c.rollout_name: c.job_name for c in seen}
    assert by_name["a"] == by_name["b"] is not None
    assert [c.job_name for c in seen if c.rollout_name is None] == ["kept"]
    assert all(c.job_name is None for c in mine)  # callers' configs untouched


def test_run_sync_closes_the_daytona_client_its_loop_created(monkeypatch) -> None:
    """Every bf.run_sync/run_batch call used to leave the Daytona client of its private event loop unclosed, and asyncio
    logged 'ERROR:asyncio:Unclosed client session' after each call."""
    from benchflow.sandbox import daytona

    closed: list[bool] = []

    class _Manager:
        _client = None
        _client_loop = None

        async def _cleanup(self) -> None:
            closed.append(True)
            self._client = None

    manager = _Manager()
    monkeypatch.setattr(daytona.DaytonaClientManager, "_instance", manager)

    async def fake(config, *a, **kw):
        manager._client = object()
        manager._client_loop = asyncio.get_running_loop()
        return RolloutResult("hello-world-task", rollout_name=config.rollout_name)

    monkeypatch.setattr(batch, "arun", fake)
    bf.run_sync(_cfg("c"))
    assert closed == [True] and manager._client is None
    bf.run_batch([_cfg("a"), _cfg("b")], concurrency=2)
    assert closed == [True, True]


def test_run_sync_shows_the_same_arguments_as_arun() -> None:
    """help(bf.run_sync) and editors showed ``(*args, **kwargs)``: the blocking
    form now declares arun's parameters, so both read the same."""
    import inspect

    sync, async_ = inspect.signature(bf.run_sync), inspect.signature(bf.arun)
    assert list(sync.parameters) == list(async_.parameters)
    for name, param in async_.parameters.items():
        assert sync.parameters[name].kind == param.kind
        assert sync.parameters[name].default == param.default


@pytest.mark.asyncio
async def test_regrade_works_inside_a_running_loop(tmp_path) -> None:
    """``bf.regrade`` called ``asyncio.run`` and failed in a Jupyter cell,
    where ``bf.run_sync``, ``bf.run_batch`` and ``Evaluation.run_sync``
    work; it now blocks the same way, and takes aregrade's ``runner``."""
    import inspect

    trial = tmp_path / "hello__1"  # a trial that froze no workspace
    trial.mkdir()
    (trial / "result.json").write_text('{"task_name": "hello", "rewards": null}')
    (trial / "config.json").write_text('{"agent": "oracle"}')
    summary = bf.regrade(tmp_path)
    assert [t.status for t in summary.trials] == ["not_regradable"]
    assert list(inspect.signature(bf.regrade).parameters) == list(
        inspect.signature(bf.aregrade).parameters
    )
