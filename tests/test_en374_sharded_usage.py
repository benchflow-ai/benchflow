"""A sharded run's summary.json has the job's tokens and cost.

Before: ``bench eval run --worker-concurrency`` wrote an aggregate
summary.json with no token or cost field at all (eval_sharding.py
``_aggregate_result``); each worker's summary had them and nothing pooled
them. Now each worker passes its summary's usage keys to the parent, which
adds them up: the aggregate has the same keys, with the same numbers, as one
unsharded summary over the same trials, and a total no shard knows stays
null.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils.evaluation_results import (
    USAGE_SUMMARY_KEYS,
    attempt_spend_summary,
    judge_usage_summary,
    usage_summary,
)
from benchflow.eval_sharding import EvalShard, EvalShardPlan, _aggregate_result
from benchflow.eval_worker import _result_payload
from benchflow.evaluation import EvaluationConfig


def _row(
    name: str,
    *,
    tokens: int | None,
    cost: float | None,
    source: str | None,
    judge_cost: float | None = None,
    no_usage: str | None = None,
) -> dict:
    """A result.json row: agent_result as rollout/_results.py writes it."""
    return {
        "task_name": name,
        "rewards": {"reward": 1.0},
        "error": None,
        "verifier_error": None,
        "agent_result": {
            "n_input_tokens": None if tokens is None else tokens // 2,
            "n_output_tokens": None if tokens is None else tokens // 4,
            "n_cache_read_tokens": None if tokens is None else tokens // 4,
            "n_cache_write_tokens": None if tokens is None else 0,
            "n_cache_write_5m_tokens": None,
            "n_cache_write_1h_tokens": None,
            "n_cache_creation_tokens": None if tokens is None else 0,
            "total_tokens": tokens,
            "cost_usd": cost,
            "usage_source": source,
            "cost_source": None if cost is None else "list_price_estimate",
            "price_table_date": None,
            "cost_usd_provider": None,
            "n_calls_cost_estimated": None if cost is None else 2,
            "price_source": None if cost is None else "list_price",
            "usage_source_legacy": "agent_native_acp" if source else "unavailable",
            "no_usage_reason": no_usage,
        },
        "judge_usage": (
            None
            if judge_cost is None
            else {
                "calls": 1,
                "calls_without_usage": 0,
                "models": {"claude-haiku-5-5": 1},
                "n_input_tokens": 1000,
                "n_output_tokens": 100,
                "n_cache_read_tokens": 0,
                "n_cache_write_tokens": 0,
                "n_cache_write_5m_tokens": 0,
                "n_cache_write_1h_tokens": 0,
                "total_tokens": 1100,
                "cost_usd": judge_cost,
                "usage_source": "provider",
                "cost_source": "list_price_estimate",
                "price_source": "list_price",
                "price_table_date": "2026-11-04",
            }
        ),
    }


SHARD_A = {
    "task-a": _row("task-a", tokens=4000, cost=0.04, source="session_log"),
    "task-b": _row("task-b", tokens=None, cost=None, source=None),
}
SHARD_B = {
    "task-c": _row("task-c", tokens=8000, cost=0.08, source="proxy", judge_cost=0.0002),
    "task-d": _row(
        "task-d", tokens=None, cost=None, source=None, no_usage="no_model_call"
    ),
}


def _usage(rows: dict[str, dict]) -> dict:
    return {
        **usage_summary(rows),
        **judge_usage_summary(rows),
        **attempt_spend_summary(rows.values()),
    }


def _plan() -> EvalShardPlan:
    return EvalShardPlan(
        total_concurrency=2,
        worker_concurrency=1,
        shards=(
            EvalShard(index=0, task_names=("task-a", "task-b"), concurrency=1),
            EvalShard(index=1, task_names=("task-c",), concurrency=1),
        ),
    )


def _aggregate(tmp_path: Path, shards: list[dict[str, dict]]) -> dict:
    _aggregate_result(
        jobs_dir=tmp_path,
        config=EvaluationConfig(),
        plan=_plan(),
        shard_results=[
            {"total": len(rows), "passed": len(rows), "usage": _usage(rows)}
            for rows in shards
        ],
        elapsed_sec=1.0,
    )
    return json.loads((tmp_path / "summary.json").read_text())


def _same(got, want) -> bool:
    if isinstance(want, dict):
        return (
            isinstance(got, dict)
            and got.keys() == want.keys()
            and all(_same(got[k], want[k]) for k in want)
        )
    if isinstance(want, float) and isinstance(got, int | float):
        return got == pytest.approx(want)
    return got == want


def test_the_sharded_summary_has_the_unsharded_jobs_usage(tmp_path):
    summary = _aggregate(tmp_path, [SHARD_A, SHARD_B])
    unsharded = _usage({**SHARD_A, **SHARD_B})
    for key in USAGE_SUMMARY_KEYS:
        assert _same(summary[key], unsharded[key]), key
    assert summary["total_tokens"] == 12_000
    assert summary["total_cost_usd"] == pytest.approx(0.12)
    assert summary["usage_sources"] == {"proxy": 1, "session_log": 1}
    assert summary["trials_without_usage"] == 1
    assert summary["trials_without_model_call"] == 1
    assert summary["n_calls_cost_estimated"] == 4
    assert summary["judge_totals"]["total_cost_usd"] == pytest.approx(0.0002)
    assert summary["spend_all_attempts"]["cost_usd"] == pytest.approx(0.1202)


def test_every_usage_key_of_a_summary_is_merged():
    """A usage key added to summary.json later is added up for a sharded
    run too, or this fails."""
    assert set(USAGE_SUMMARY_KEYS) == set(_usage({**SHARD_A, **SHARD_B}))


def test_a_total_no_shard_knows_stays_null(tmp_path):
    nothing = {"task-b": SHARD_A["task-b"]}
    summary = _aggregate(tmp_path, [nothing, nothing])
    assert summary["total_tokens"] is None
    assert summary["total_cost_usd"] is None
    assert summary["avg_cost_per_trial_usd"] is None
    assert summary["trials_without_cost"] == 2
    assert summary["spend_all_attempts"]["cost_usd"] is None
    assert summary["judge_totals"]["total_cost_usd"] is None

    # One shard unknown, one known: the known one's total, never None + 0.
    summary = _aggregate(tmp_path, [nothing, {"task-c": SHARD_B["task-c"]}])
    assert summary["total_tokens"] == 8000
    assert summary["total_cost_usd"] == pytest.approx(0.08)
    assert summary["avg_cost_per_trial_usd"] == pytest.approx(0.08)


def test_a_worker_passes_its_summarys_usage_to_the_parent(tmp_path):
    job_dir = tmp_path / "jobs" / "shard-job"
    job_dir.mkdir(parents=True)
    usage = _usage(SHARD_A)
    (job_dir / "summary.json").write_text(
        json.dumps({"job_name": "shard-job", "score": "50.0%", **usage})
    )
    result = SimpleNamespace(
        job_name="shard-job",
        job_dir=job_dir,
        total=2,
        passed=1,
        failed=0,
        errored=1,
        verifier_errored=0,
        elapsed_sec=1.0,
        score=0.5,
        score_excl_errors=1.0,
    )
    payload = _result_payload(result)
    assert payload["usage"] == usage
    assert (
        _result_payload(SimpleNamespace(**{**vars(result), "job_dir": None}))["usage"]
        == {}
    )


def test_an_unknown_shard_spend_is_unknown_not_nothing():
    """Review 2, #1: a shard whose spend_all_attempts is null (its records
    could not be read) merged as nothing: two of them gave {}, one beside a
    known shard a sum with no marker. All unknown is null; a sum with an
    unknown part says so (parts_unknown), and a shard whose summary could
    not be read at all is counted (shards_without_usage)."""
    from benchflow._utils.evaluation_results import merge_usage_summaries

    unknown = {**_usage(SHARD_A), "spend_all_attempts": None}
    assert merge_usage_summaries([unknown, unknown])["spend_all_attempts"] is None
    merged = merge_usage_summaries([unknown, _usage(SHARD_B)])
    spend = merged["spend_all_attempts"]
    assert spend["parts_unknown"] == 1
    assert spend["cost_usd"] == pytest.approx(
        _usage(SHARD_B)["spend_all_attempts"]["cost_usd"]
    )
    unread = merge_usage_summaries([{}, _usage(SHARD_B)])
    assert unread["shards_without_usage"] == 1
    assert unread["spend_all_attempts"]["parts_unknown"] == 1


def test_a_shard_without_the_incomplete_counts_counts_none():
    """Review of PR #147 (F2b-R147, N1): a shard summary from before #122 has
    no ``trials_usage_incomplete`` / ``trials_cost_incomplete``. Beside a
    shard that has one incomplete trial the merge says 1, and with no shard
    that knows the keys it says 0, never null."""
    from benchflow._utils.evaluation_results import merge_usage_summaries

    def shard(*rows: dict, old: bool = False) -> dict:
        usage = usage_summary({row["task_name"]: row for row in rows})
        if old:
            usage.pop("trials_usage_incomplete")
            usage.pop("trials_cost_incomplete")
        return usage

    partial = _row("cut", tokens=4620, cost=0.0234, source="end_of_turn")
    partial["agent_result"].update(
        usage_incomplete=True,
        usage_incomplete_reason="a turn ended without its usage report",
        cost_incomplete=True,
    )
    whole = _row("whole", tokens=1000, cost=0.01, source="end_of_turn")
    older = _row("older", tokens=1000, cost=0.01, source="end_of_turn")

    merged = merge_usage_summaries([shard(older, old=True), shard(partial, whole)])
    assert (merged["trials_usage_incomplete"], merged["trials_cost_incomplete"]) == (
        1,
        1,
    )
    merged = merge_usage_summaries([shard(older, old=True), shard(whole, old=True)])
    assert (merged["trials_usage_incomplete"], merged["trials_cost_incomplete"]) == (
        0,
        0,
    )
