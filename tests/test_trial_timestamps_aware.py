"""Trial times are timezone-aware everywhere BenchFlow writes them, and every reader that subtracts two of them
survives a naive one (a result.json written before 0.10.0 holds naive times). `bench run` exited 1 with a TypeError
printing its summary when a rubric-reviewed trial's started_at was aware and its finished_at naive."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

MIXED = [
    (datetime.now(UTC) - timedelta(seconds=5), datetime.now()),
    (datetime.now() - timedelta(seconds=5), datetime.now(UTC)),
]


@pytest.mark.parametrize(("start", "end"), MIXED)
def test_the_run_summary_s_seconds_survive_mixed_times(start, end):
    from benchflow.cli._output.run_summary import _seconds

    seconds = _seconds(SimpleNamespace(started_at=start, finished_at=end))
    assert seconds is not None and 4 <= seconds <= 30


@pytest.mark.parametrize(("start", "end"), MIXED)
def test_a_job_trial_s_duration_survives_mixed_times(start, end):
    from benchflow.jobs import Trial

    trial = SimpleNamespace(result=SimpleNamespace(started_at=start, finished_at=end))
    duration = Trial.duration_sec.fget(trial)
    assert duration is not None and 4 <= duration <= 30


def test_a_stored_review_duration_survives_a_naive_finish():
    from benchflow.review import persistence

    reader = persistence._execution_seconds
    start = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
    end = datetime.now().isoformat()
    duration = reader({"started_at": start, "finished_at": end, "timing": {}})
    assert duration is not None and 4 <= duration <= 30


def test_every_time_a_built_result_holds_is_aware(tmp_path):
    from benchflow.rollout._results import _build_rollout_result

    result = _build_rollout_result(
        tmp_path,
        task_name="t",
        rollout_name="r",
        agent="pi-acp",
        agent_name="pi-acp",
        model=None,
        n_tool_calls=0,
        prompts=["task"],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime.now(),  # the rollout's own clock is naive
        timing={},
    )
    assert (
        result.started_at.tzinfo is not None and result.finished_at.tzinfo is not None
    )
    recorded = json.loads((tmp_path / "result.json").read_text())
    stamps = {
        k: v for k, v in recorded.items() if k.endswith("_at") and isinstance(v, str)
    }
    assert {"started_at", "finished_at"} <= set(stamps)
    for key, value in stamps.items():
        assert datetime.fromisoformat(value).tzinfo is not None, (key, value)


@pytest.mark.parametrize(("start", "end"), MIXED)
def test_a_result_s_record_row_survives_mixed_times(start, end):
    """``RolloutResult.to_record`` (the SDK's CSV and JSONL rows) subtracts the same two times."""
    from benchflow.models import RolloutResult

    record = RolloutResult(
        "t", rewards={"reward": 1.0}, started_at=start, finished_at=end
    ).to_record()
    duration = next(
        v for k, v in record.items() if "duration" in k or k.endswith("_sec")
    )
    assert duration is not None and 4 <= duration <= 30
