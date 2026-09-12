"""Tests for metrics collection and aggregation."""

import json

import pytest

from benchflow.metrics import collect_metrics


@pytest.fixture
def results_dir(tmp_path):
    """Create a mock results directory with result.json files."""
    # Task A: passed
    trial_a = tmp_path / "job1" / "task-a__abc123"
    trial_a.mkdir(parents=True)
    (trial_a / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "rewards": {"reward": 1.0},
                "error": None,
                "n_tool_calls": 10,
                "started_at": "2026-03-24 10:00:00.000000",
                "finished_at": "2026-03-24 10:01:00.000000",
            }
        )
    )

    # Task B: failed
    trial_b = tmp_path / "job1" / "task-b__def456"
    trial_b.mkdir(parents=True)
    (trial_b / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-b",
                "rewards": {"reward": 0.0},
                "error": None,
                "n_tool_calls": 25,
                "started_at": "2026-03-24 10:00:00.000000",
                "finished_at": "2026-03-24 10:02:00.000000",
            }
        )
    )

    # Task C: errored
    trial_c = tmp_path / "job1" / "task-c__ghi789"
    trial_c.mkdir(parents=True)
    (trial_c / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-c",
                "rewards": None,
                "error": "Agent timed out after 900.0s",
                "n_tool_calls": 0,
                "started_at": "2026-03-24 10:00:00.000000",
                "finished_at": "2026-03-24 10:15:00.000000",
            }
        )
    )

    return tmp_path


@pytest.fixture
def results_dir_with_retries(tmp_path):
    """Results dir where task-a failed first, then passed on retry."""
    # Task A: first attempt failed
    trial_a1 = tmp_path / "attempt1" / "task-a__first"
    trial_a1.mkdir(parents=True)
    (trial_a1 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "rewards": {"reward": 0.0},
                "error": None,
                "n_tool_calls": 5,
                "started_at": "2026-03-24 10:00:00.000000",
                "finished_at": "2026-03-24 10:01:00.000000",
            }
        )
    )

    # Task A: retry passed
    trial_a2 = tmp_path / "attempt2" / "task-a__second"
    trial_a2.mkdir(parents=True)
    (trial_a2 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "rewards": {"reward": 1.0},
                "error": None,
                "n_tool_calls": 15,
                "started_at": "2026-03-24 10:02:00.000000",
                "finished_at": "2026-03-24 10:03:00.000000",
            }
        )
    )

    return tmp_path


def test_collect_metrics_basic(results_dir):
    """Test basic pass/fail/error counting."""
    metrics = collect_metrics(str(results_dir))
    s = metrics.summary()

    assert s["total"] == 3
    assert s["passed"] == 1
    assert s["failed"] == 1
    assert s["errored"] == 1
    assert s["score"].startswith("33")
    assert s["score_ratio"] == pytest.approx(1 / 3)
    assert s["score_excl_errors_ratio"] == pytest.approx(0.5)
    assert "task-a" in s["passed_tasks"]
    assert "task-b" in s["failed_tasks"]
    assert "task-c" in s["errored_tasks"]


def test_collect_metrics_tool_calls(results_dir):
    """Test average tool call calculation (excludes errored tasks)."""
    metrics = collect_metrics(str(results_dir))
    s = metrics.summary()
    # (10 + 25) / 2 = 17.5 — errored task-c excluded
    assert abs(s["avg_tool_calls"] - 17.5) < 0.1


def test_collect_metrics_duration(results_dir):
    """Test average duration calculation (excludes errored tasks)."""
    metrics = collect_metrics(str(results_dir))
    s = metrics.summary()
    # (60 + 120) / 2 = 90 — errored task-c excluded
    assert abs(s["avg_duration_sec"] - 90.0) < 1.0


def test_collect_metrics_best_result_picking(results_dir_with_retries):
    """Test that best result per task is picked (higher reward wins)."""
    base = results_dir_with_retries

    # Both errored (no rewards): first seen is kept, counted once as errored
    trial_b1 = base / "attempt1" / "task-b__err1"
    trial_b1.mkdir(parents=True)
    (trial_b1 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-b",
                "rewards": None,
                "error": "install failed",
                "n_tool_calls": 0,
                "started_at": "2026-03-24 10:00:00.000000",
                "finished_at": "2026-03-24 10:01:00.000000",
            }
        )
    )
    trial_b2 = base / "attempt2" / "task-b__err2"
    trial_b2.mkdir(parents=True)
    (trial_b2 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-b",
                "rewards": None,
                "error": "pipe closed",
                "n_tool_calls": 0,
                "started_at": "2026-03-24 10:02:00.000000",
                "finished_at": "2026-03-24 10:03:00.000000",
            }
        )
    )

    # Equal rewards: deterministic pick (counted once)
    trial_c1 = base / "attempt1" / "task-c__eq1"
    trial_c1.mkdir(parents=True)
    (trial_c1 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-c",
                "rewards": {"reward": 0.5},
                "error": None,
                "n_tool_calls": 3,
                "started_at": "2026-03-24 10:00:00.000000",
                "finished_at": "2026-03-24 10:01:00.000000",
            }
        )
    )
    trial_c2 = base / "attempt2" / "task-c__eq2"
    trial_c2.mkdir(parents=True)
    (trial_c2 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-c",
                "rewards": {"reward": 0.5},
                "error": None,
                "n_tool_calls": 4,
                "started_at": "2026-03-24 10:02:00.000000",
                "finished_at": "2026-03-24 10:03:00.000000",
            }
        )
    )

    metrics = collect_metrics(str(base))
    s = metrics.summary()

    # task-a: passed (higher reward picked), task-b: errored, task-c: failed (0.5 != 1.0)
    assert s["total"] == 3
    assert s["passed"] == 1
    assert "task-a" in s["passed_tasks"]
    assert s["errored"] == 1
    assert "task-b" in s["errored_tasks"]
    # Determinism: when both attempts errored, first-seen wins (install_failure
    # from attempt1, not "other" from attempt2's "pipe closed").
    assert s["error_breakdown"] == {"install_failure": 1}
    assert s["failed"] == 1
    assert "task-c" in s["failed_tasks"]


def test_collect_metrics_empty_dir(tmp_path):
    """Test with no results."""
    metrics = collect_metrics(str(tmp_path))
    s = metrics.summary()

    assert s["total"] == 0
    assert s["passed"] == 0


def test_collect_metrics_corrupt_file(tmp_path):
    """Test that corrupt result.json files are skipped."""
    trial = tmp_path / "job" / "task__abc"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text("not json")

    metrics = collect_metrics(str(tmp_path))
    s = metrics.summary()
    assert s["total"] == 0


def test_collect_metrics_metadata(results_dir):
    """Test that benchmark/agent/model metadata is passed through."""
    metrics = collect_metrics(
        str(results_dir), benchmark="TB2", agent="claude", model="haiku"
    )
    s = metrics.summary()

    assert s["benchmark"] == "TB2"
    assert s["agent"] == "claude"
    assert s["model"] == "haiku"


def test_collect_metrics_partial_reward(tmp_path):
    """Test handling of partial rewards (not 0 or 1)."""
    trial = tmp_path / "job" / "task-a__abc"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "rewards": {"reward": 0.5},
                "error": None,
                "n_tool_calls": 10,
                "started_at": "2026-03-24 10:00:00.000000",
                "finished_at": "2026-03-24 10:01:00.000000",
            }
        )
    )

    metrics = collect_metrics(str(tmp_path))
    s = metrics.summary()
    assert s["total"] == 1
    # 0.5 is not 1.0, so not passed
    assert s["passed"] == 0
    assert s["failed"] == 1


def test_collect_metrics_aggregates_memory_scores_without_changing_output_score(
    tmp_path,
):
    """Guards OPEN-3 memory-space metrics as additive, not pass/fail signal."""
    rows = [
        ("task-a", 1.0, 0.5),
        ("task-b", 0.0, 1.0),
        ("task-c", 1.0, None),
    ]
    for task_name, output_reward, memory_score in rows:
        trial = tmp_path / "job" / f"{task_name}__trial"
        trial.mkdir(parents=True)
        data = {
            "task_name": task_name,
            "rewards": {"reward": output_reward},
            "error": None,
            "verifier_error": None,
            "n_tool_calls": 1,
            "started_at": "2026-03-24 10:00:00.000000",
            "finished_at": "2026-03-24 10:01:00.000000",
        }
        if memory_score is not None:
            data["memory_score"] = memory_score
            data["reward_events"] = [
                {
                    "type": "terminal",
                    "reward": memory_score,
                    "source": "memory",
                    "space": "memory",
                    "granularity": "terminal",
                }
            ]
        (trial / "result.json").write_text(json.dumps(data))

    metrics = collect_metrics(str(tmp_path))
    summary = metrics.summary()

    assert summary["score"] == "66.7%"
    assert summary["score_ratio"] == pytest.approx(2 / 3)
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["memory"] == {
        "scored": 2,
        "avg_score": 0.75,
        "score": "75.0%",
    }
    assert summary["memory_score"] == 0.75
    assert summary["memory_score_coverage"] == 2 / 3
    assert summary["memory_scores"] == {"task-a": 0.5, "task-b": 1.0}


def test_collect_metrics_error_and_verifier_error_counted_once(tmp_path):
    """A result with BOTH error and verifier_error (rewards=None) must land in
    exactly one bucket — errored — so the count buckets stay disjoint and
    passed+failed+errored+verifier_errored == total.

    Guards the fix from PR #320 for audit Finding 6 (metrics.py side).
    """
    trial = tmp_path / "job" / "task-x__abc123"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-x",
                "rewards": None,
                "error": "agent crashed",
                "verifier_error": "verifier also failed",
                "n_tool_calls": 0,
                "started_at": "2026-03-24 10:00:00.000000",
                "finished_at": "2026-03-24 10:05:00.000000",
            }
        )
    )

    metrics = collect_metrics(str(tmp_path))
    s = metrics.summary()

    assert s["total"] == 1
    # Agent error takes precedence — counted once as errored, never as both.
    assert s["errored"] == 1
    assert s["verifier_errored"] == 0
    assert (
        s["passed"] + s["failed"] + s["errored"] + s["verifier_errored"] == s["total"]
    )


def test_collect_metrics_usage_aggregation_mixed_telemetry(tmp_path):
    rows = [
        (
            "task-a",
            {
                "usage_source": "provider_response",
                "n_input_tokens": 100,
                "n_output_tokens": 10,
                "n_cache_read_tokens": 5,
                "n_cache_creation_tokens": 1,
                "total_tokens": 116,
                "cost_usd": 0.001,
            },
        ),
        (
            "task-b",
            {
                "usage_source": "provider_response",
                "n_input_tokens": 200,
                "n_output_tokens": 20,
                "n_cache_read_tokens": 0,
                "n_cache_creation_tokens": 4,
                "total_tokens": 224,
                "cost_usd": 0.002,
            },
        ),
        (
            "task-c",
            {
                "usage_source": "unavailable",
                "n_input_tokens": None,
                "n_output_tokens": None,
                "n_cache_read_tokens": None,
                "n_cache_creation_tokens": None,
                "total_tokens": None,
                "cost_usd": None,
            },
        ),
    ]
    for task_name, agent_result in rows:
        trial = tmp_path / "job" / f"{task_name}__trial"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_name,
                    "rewards": {"reward": 1.0},
                    "error": None,
                    "verifier_error": None,
                    "n_tool_calls": 1,
                    "started_at": "2026-03-24 10:00:00.000000",
                    "finished_at": "2026-03-24 10:01:00.000000",
                    "agent_result": {"n_tool_calls": 1, "n_prompts": 1, **agent_result},
                }
            )
        )

    metrics = collect_metrics(str(tmp_path))
    summary = metrics.summary()

    assert summary["total_input_tokens"] == 300
    assert summary["total_output_tokens"] == 30
    assert summary["total_cache_read_tokens"] == 5
    assert summary["total_cache_creation_tokens"] == 5
    assert summary["total_tokens"] == 340
    assert summary["total_cost_usd"] == 0.003
    assert summary["avg_cost_per_trial_usd"] == 0.0015
    assert summary["telemetry_coverage"] == 2 / 3


def test_usage_source_type_contract_tracks_trusted_sources():
    """Guards the fix from PR #858 against usage_source_type drifting from the trusted source contract."""
    from typing import get_args

    from benchflow.usage_tracking import (
        TRUSTED_USAGE_SOURCES,
        USAGE_SOURCE_AGENT_NATIVE_ACP,
        USAGE_SOURCE_PROVIDER_RESPONSE,
        USAGE_SOURCE_UNAVAILABLE,
        UsageSource,
        normalize_usage_source,
    )

    assert set(get_args(UsageSource)) == {
        USAGE_SOURCE_PROVIDER_RESPONSE,
        USAGE_SOURCE_AGENT_NATIVE_ACP,
        USAGE_SOURCE_UNAVAILABLE,
    }
    assert {
        USAGE_SOURCE_PROVIDER_RESPONSE,
        USAGE_SOURCE_AGENT_NATIVE_ACP,
    } == TRUSTED_USAGE_SOURCES
    assert normalize_usage_source(USAGE_SOURCE_AGENT_NATIVE_ACP) == (
        USAGE_SOURCE_AGENT_NATIVE_ACP
    )
    with pytest.raises(ValueError, match="usage_source must be one of"):
        normalize_usage_source("new_unregistered_source")


def test_collect_metrics_malformed_rewards_does_not_win_over_none(tmp_path):
    """Guards the follow-up noted in PR #1096.

    A malformed rewards value (truthy non-dict, e.g. 1.0, True, []) must NOT
    win clause 2 of the selection predicate over a well-formed rewards=None
    result. Before the fix, `r.get("rewards") is not None` passed for any
    truthy non-dict, so rewards:1.0 beat rewards:None and then crashed
    _safe_reward (which called .get() on a float) in clause 3.

    After the fix, only isinstance(..., dict) values are treated as
    having a scoreable reward — malformed artifacts are treated as errored
    (no reward), consistent with _classify_completed_outcomes and evaluation.py.
    """
    import json

    # attempt1: malformed rewards (bare float — the exact shape from PR #1096 comment)
    t1 = tmp_path / "attempt1" / "task-x__t1"
    t1.mkdir(parents=True)
    (t1 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-x",
                "rewards": 1.0,
                "error": None,
                "n_tool_calls": 0,
                "started_at": "2026-01-01 00:00:00.000000",
                "finished_at": "2026-01-01 00:01:00.000000",
            }
        )
    )
    # attempt2: well-formed rewards=None (errored)
    t2 = tmp_path / "attempt2" / "task-x__t2"
    t2.mkdir(parents=True)
    (t2 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-x",
                "rewards": None,
                "error": "install failed",
                "n_tool_calls": 0,
                "started_at": "2026-01-01 00:02:00.000000",
                "finished_at": "2026-01-01 00:03:00.000000",
            }
        )
    )

    # Must not crash, and malformed rewards must NOT be treated as a scored result
    metrics = collect_metrics(str(tmp_path))
    s = metrics.summary()

    assert s["total"] == 1
    # The malformed artifact must not win as a scored (passed/failed) result.
    # Either the errored artifact wins (errored=1) or the malformed one is
    # kept as-is but still counted as errored (no valid reward extracted).
    assert s["passed"] == 0, "malformed rewards:1.0 must not count as passed"
    assert s["failed"] == 0, "malformed rewards:1.0 must not count as failed"
    assert s["errored"] == 1, "task with malformed rewards must count as errored"


def test_collect_metrics_malformed_rewards_does_not_beat_zero_reward(tmp_path):
    """Guards clause 2 tie-breaking: malformed rewards must not displace a
    well-formed rewards={'reward': 0.0} result.

    The PR #1096 comment noted: 'a malformed artifact scores 0.0 in the
    selection, so it can still tie out a well-formed artifact whose reward
    is 0.0'. With the isinstance guard, the malformed artifact never enters
    clause 2 or 3 as a winner — the well-formed dict is always preferred.
    """
    import json

    # attempt1: well-formed rewards with reward=0.0 (scored, failed)
    t1 = tmp_path / "attempt1" / "task-y__t1"
    t1.mkdir(parents=True)
    (t1 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-y",
                "rewards": {"reward": 0.0},
                "error": None,
                "n_tool_calls": 2,
                "started_at": "2026-01-01 00:00:00.000000",
                "finished_at": "2026-01-01 00:01:00.000000",
            }
        )
    )
    # attempt2: malformed rewards (bare float)
    t2 = tmp_path / "attempt2" / "task-y__t2"
    t2.mkdir(parents=True)
    (t2 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-y",
                "rewards": 1.0,
                "error": None,
                "n_tool_calls": 0,
                "started_at": "2026-01-01 00:02:00.000000",
                "finished_at": "2026-01-01 00:03:00.000000",
            }
        )
    )

    metrics = collect_metrics(str(tmp_path))
    s = metrics.summary()

    assert s["total"] == 1
    # The well-formed reward=0.0 result must win — task is failed, not errored
    assert s["failed"] == 1, "well-formed rewards:{'reward':0.0} must be kept as failed"
    assert s["passed"] == 0
    assert s["errored"] == 0


def test_collect_metrics_boolean_reward_treated_as_invalid(tmp_path):
    """Guards Devin review: isinstance(True, (int,float)) is True in Python,
    so _safe_reward previously accepted boolean rewards. is_valid_reward_number
    explicitly rejects booleans — selection and reporting must agree."""
    import json

    # attempt1: boolean reward (True) — invalid per is_valid_reward_number
    t1 = tmp_path / "a1" / "task__t1"
    t1.mkdir(parents=True)
    (t1 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "rewards": {"reward": True},
                "error": None,
                "n_tool_calls": 0,
                "started_at": "2026-01-01 00:00:00.000000",
                "finished_at": "2026-01-01 00:01:00.000000",
            }
        )
    )
    # attempt2: well-formed reward=0.0 (scored, failed)
    t2 = tmp_path / "a2" / "task__t2"
    t2.mkdir(parents=True)
    (t2 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "rewards": {"reward": 0.0},
                "error": None,
                "n_tool_calls": 1,
                "started_at": "2026-01-01 00:02:00.000000",
                "finished_at": "2026-01-01 00:03:00.000000",
            }
        )
    )

    metrics = collect_metrics(str(tmp_path))
    s = metrics.summary()
    assert s["failed"] == 1, "well-formed reward=0.0 must win over boolean reward=True"
    assert s["passed"] == 0
    assert s["errored"] == 0


def test_collect_metrics_nonfinite_reward_treated_as_invalid(tmp_path):
    """Guards Devin review: inf/-inf/nan are rejected by is_valid_reward_number
    but isinstance(inf, float) is True, so old _safe_reward accepted them."""
    import json

    t1 = tmp_path / "a1" / "task__t1"
    t1.mkdir(parents=True)
    (t1 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "rewards": {"reward": None},
                "error": "agent error",
                "n_tool_calls": 0,
                "started_at": "2026-01-01 00:00:00.000000",
                "finished_at": "2026-01-01 00:01:00.000000",
            }
        )
    )
    # inf cannot be serialised directly to JSON; simulate via a post-load patch
    # by writing a valid file then checking the validator directly
    from benchflow.rewards.validation import is_valid_reward_number

    assert not is_valid_reward_number(float("inf")), "inf must be invalid"
    assert not is_valid_reward_number(float("nan")), "nan must be invalid"
    assert not is_valid_reward_number(True), "True must be invalid"
    assert not is_valid_reward_number(False), "False must be invalid"
    assert not is_valid_reward_number(-1.0), "-1.0 must be invalid (below range)"
    assert is_valid_reward_number(0.0), "0.0 must be valid"
    assert is_valid_reward_number(0.5), "0.5 must be valid"
    assert is_valid_reward_number(1.0), "1.0 must be valid"


def test_collect_metrics_empty_rewards_dict_does_not_beat_valid_zero(tmp_path):
    """Guards Devin review: {} has no valid reward scalar; reward=0.0 must win."""
    import json

    # attempt1: empty dict — no reward key at all
    t1 = tmp_path / "a1" / "task__t1"
    t1.mkdir(parents=True)
    (t1 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "rewards": {},
                "error": None,
                "n_tool_calls": 0,
                "started_at": "2026-01-01 00:00:00.000000",
                "finished_at": "2026-01-01 00:01:00.000000",
            }
        )
    )
    # attempt2: valid reward=0.0
    t2 = tmp_path / "a2" / "task__t2"
    t2.mkdir(parents=True)
    (t2 / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "rewards": {"reward": 0.0},
                "error": None,
                "n_tool_calls": 2,
                "started_at": "2026-01-01 00:02:00.000000",
                "finished_at": "2026-01-01 00:03:00.000000",
            }
        )
    )

    metrics = collect_metrics(str(tmp_path))
    s = metrics.summary()
    assert s["failed"] == 1, "valid reward=0.0 must beat empty rewards dict"
    assert s["errored"] == 0
