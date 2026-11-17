"""A cache-write total says how many trials it leaves out (EN-624).

summary.json's ``total_cache_creation_tokens`` sums the trials whose
cache-write count is known (EN-374's rule: a trial that did not count a
field adds nothing to it, and a field no trial counted is unknown, not 0).
An answer that names no cache-write count leaves it null (EN-645), so the
known sum leaves that trial out. ``trials_without_cache_write_count`` beside
it counts the trials the total covers whose count is not known, so a total
beside a count above 0 is a floor, and a null total with a count says every
trial's count is unknown. ``bench eval metrics --json`` gives the same.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow._utils.evaluation_results import usage_summary
from benchflow.cli.main import app


def _row(writes: object, *, source: str = "proxy", tokens: int = 1050) -> dict:
    return {
        "agent_result": {
            "usage_source": source,
            "n_input_tokens": 1000,
            "n_output_tokens": 50,
            "n_cache_read_tokens": 0,
            "n_cache_creation_tokens": writes,
            "total_tokens": tokens,
            "cost_usd": 0.00125,
        }
    }


def _summary(*writes: object) -> dict:
    return usage_summary({f"t{i}": _row(w) for i, w in enumerate(writes)})


def test_every_count_unknown_gives_a_null_total_and_counts_them_all():
    summary = _summary(None, None, None)
    assert summary["total_cache_creation_tokens"] is None
    assert summary["trials_without_cache_write_count"] == 3


def test_a_mix_gives_the_known_sum_and_counts_the_unknown():
    summary = _summary(0, None, 5)
    assert summary["total_cache_creation_tokens"] == 5
    assert summary["trials_without_cache_write_count"] == 1


def test_every_count_known_gives_the_sum_and_counts_none():
    summary = _summary(0, 7, 5)
    assert summary["total_cache_creation_tokens"] == 12
    assert summary["trials_without_cache_write_count"] == 0


def test_only_the_trials_the_total_covers_are_counted():
    """A trial no trusted source counted is in no total, so in no count."""
    rows = {
        "a": _row(4),
        "b": _row(None, source="unavailable"),
        "c": {"agent_result": {"no_usage_reason": "no_model_call"}},
    }
    summary = usage_summary(rows)
    assert summary["total_cache_creation_tokens"] == 4
    assert summary["trials_without_cache_write_count"] == 0


def _job(tmp_path: Path, *writes: object) -> Path:
    job = tmp_path / "jobs" / "run"
    job.mkdir(parents=True)
    rows = {}
    for i, w in enumerate(writes):
        folder = job / f"t{i}__abcd123{i}"
        folder.mkdir()
        row = {
            "task_name": f"t{i}",
            "rollout_name": folder.name,
            "agent": "claude-agent-acp",
            "model": "claude-haiku-4-5",
            "rewards": {"reward": 1.0},
            **_row(w),
        }
        (folder / "result.json").write_text(json.dumps(row))
        rows[folder.name] = row
    (job / "summary.json").write_text(json.dumps(usage_summary(rows)))
    return job


@pytest.mark.parametrize(
    ("writes", "total", "unknown"),
    [((None, None), None, 2), ((0, None, 5), 5, 1), ((0, 3), 3, 0)],
)
def test_bench_eval_metrics_json_agrees_with_summary_json(
    tmp_path, writes, total, unknown
):
    job = _job(tmp_path, *writes)
    result = CliRunner().invoke(app, ["eval", "metrics", str(job), "--json"])
    assert result.exit_code == 0, result.output
    metrics = json.loads(result.output)
    summary = json.loads((job / "summary.json").read_text())
    for doc in (metrics, summary):
        assert doc["total_cache_creation_tokens"] == total
        assert doc["trials_without_cache_write_count"] == unknown
