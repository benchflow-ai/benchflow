"""One vocabulary for where a trial's tokens and cost came from, and null for
what nobody counted.

result.json's ``agent_result``, ATIF's ``final_metrics``, summary.json and
the run card name a trial's ``usage_source`` (``stream_log``,
``session_log``, ``end_of_turn``, ``proxy``, ``provider``), its
``cost_source`` (``provider`` or ``list_price_estimate``) and the
``price_table_date``. A trial nothing counted has null tokens and a null
cost in every one of them, and no summary counts it as 0. Guards
against frontier-design-benchmark PR #17, whose errored Claude trials had no
output tokens and no cost.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

TOKENS = (
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_read_tokens",
    "n_cache_creation_tokens",
    "total_tokens",
)


def _write_result(folder: Path, **usage) -> dict:
    """result.json for one trial, as the rollout writes it."""
    from benchflow.rollout import _build_rollout_result

    trajectory = [{"type": "agent_message", "text": "done"}]
    _build_rollout_result(
        folder,
        task_name="task",
        rollout_name="task__1",
        agent="claude-agent-acp",
        agent_name="claude",
        model="claude-opus-5-5",
        n_tool_calls=0,
        prompts=["solve"],
        error=usage.pop("error", None),
        verifier_error=None,
        trajectory=trajectory,
        partial_trajectory=False,
        rewards=usage.pop("rewards", {"reward": 1.0}),
        started_at=datetime(2026, 11, 13, 6, 0),
        timing={},
        **usage,
    )
    return json.loads((folder / "result.json").read_text())


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("provider_response", "proxy"),
        ("agent_native_acp", "end_of_turn"),
        ("agent-reported", "end_of_turn"),
        ("unavailable", None),
        ("none", None),
        ("stream_log", "stream_log"),
        ("session_log", "session_log"),
        (None, None),
    ],
)
def test_every_usage_source_maps_to_the_result_vocabulary(old, new):
    from benchflow.usage_tracking import result_usage_source

    assert result_usage_source(old) == new


def test_a_trial_nobody_counted_is_null_everywhere(tmp_path):
    """No usage: result.json, ATIF (trainer/atif.json and final_metrics) and
    the summaries say null, never 0, and count it as no trial's 0."""
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow.metrics import collect_metrics
    from benchflow.usage_tracking import usage_unavailable

    unknown = usage_unavailable()
    assert all(unknown[field] is None for field in TOKENS)
    data = _write_result(tmp_path / "trial", **unknown)

    agent = data["agent_result"]
    assert all(agent[field] is None for field in TOKENS)
    assert agent["cost_usd"] is None
    assert agent["usage_source"] is None
    assert agent["cost_source"] is None
    assert agent["price_table_date"] is None
    assert agent["usage_source_legacy"] == "unavailable"
    final = data["final_metrics"]
    assert final["total_prompt_tokens"] is None
    assert final["total_completion_tokens"] is None
    assert final["total_cost_usd"] is None
    assert final["extra"] == {
        "usage_source": None,
        "cost_source": None,
        "price_table_date": None,
        "cost_usd_provider": None,
        "n_calls_cost_estimated": None,
        "n_input_tokens": None,
        "n_output_tokens": None,
        "n_cache_read_tokens": None,
        "n_cache_write_tokens": None,
        "n_cache_write_5m_tokens": None,
        "n_cache_write_1h_tokens": None,
    }
    assert agent["n_cache_write_tokens"] is None
    assert agent["no_usage_reason"] is None
    atif = json.loads((tmp_path / "trial" / "trainer" / "atif.json").read_text())
    metrics = atif["final_metrics"]
    for key in (
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_cached_tokens",
        "total_cost_usd",
    ):
        assert key in metrics and metrics[key] is None
    assert metrics["extra"] == final["extra"]

    summary = usage_summary({"task__1": data})
    for key in (
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
        "total_cost_usd",
        "avg_cost_per_trial_usd",
    ):
        assert summary[key] is None, key
    assert summary["usage_sources"] == {}
    assert summary["cost_sources"] == {}
    assert summary["trials_without_usage"] == 1
    assert summary["trials_without_cost"] == 1
    assert summary["telemetry_coverage"] == 0.0

    job = tmp_path / "job"
    job.mkdir()
    (job / "task__1").mkdir()
    (job / "task__1" / "result.json").write_text(json.dumps(data))
    totals = collect_metrics(str(job)).summary()
    assert totals["total_input_tokens"] is None
    assert totals["total_tokens"] is None
    assert totals["total_cost_usd"] is None


def test_a_counted_trial_carries_the_same_labels_everywhere(tmp_path):
    data = _write_result(
        tmp_path / "trial",
        n_input_tokens=1200,
        n_output_tokens=300,
        n_cache_read_tokens=9000,
        n_cache_creation_tokens=400,
        total_tokens=10900,
        cost_usd=0.0123,
        usage_source="stream_log",
        usage_source_legacy="agent_native_acp",
        price_source="list_price",
        cost_source="list_price_estimate",
        price_table_date="2026-11-04",
    )
    agent = data["agent_result"]
    assert agent["usage_source"] == "stream_log"
    assert agent["cost_source"] == "list_price_estimate"
    assert agent["price_table_date"] == "2026-11-04"
    assert agent["usage_source_legacy"] == "agent_native_acp"
    assert agent["n_cache_write_tokens"] == agent["n_cache_creation_tokens"] == 400
    assert agent["total_tokens"] == 1200 + 300 + 9000 + 400
    atif = json.loads((tmp_path / "trial" / "trainer" / "atif.json").read_text())
    final = atif["final_metrics"]
    # ATIF's own meaning: the prompt counts the cache, cached is the reads.
    assert final["total_prompt_tokens"] == 1200 + 9000 + 400
    assert final["total_cached_tokens"] == 9000
    assert final["total_completion_tokens"] == agent["n_output_tokens"]
    assert final["total_cost_usd"] == agent["cost_usd"]
    assert final["extra"] == {
        "usage_source": "stream_log",
        "cost_source": "list_price_estimate",
        "price_table_date": "2026-11-04",
        "cost_usd_provider": None,
        "n_calls_cost_estimated": None,
        "n_input_tokens": 1200,
        "n_output_tokens": 300,
        "n_cache_read_tokens": 9000,
        "n_cache_write_tokens": 400,
        "n_cache_write_5m_tokens": None,
        "n_cache_write_1h_tokens": None,
    }
    assert data["final_metrics"]["extra"] == final["extra"]


def test_an_old_result_reads_in_the_new_vocabulary():
    """A 0.10.0rc3 result.json keeps loading: its usage_source maps, and the
    old name stays readable as usage_source_legacy."""
    from benchflow.models import RolloutResult

    old = RolloutResult.from_dict(
        {
            "task_name": "t",
            "agent_result": {
                "n_input_tokens": 5,
                "usage_source": "agent_native_acp",
                "cost_usd": None,
            },
        }
    )
    assert old.usage_source == "end_of_turn"
    assert old.usage_source_legacy == "agent_native_acp"
    routed = RolloutResult.from_dict(
        {"task_name": "t", "agent_result": {"usage_source": "agent-reported"}}
    )
    assert routed.usage_source == "end_of_turn"
    assert routed.usage_source_legacy == "agent-reported"


def _trial(source, legacy, *, tokens, cost, cost_source=None, rewards=True, error=None):
    return {
        "rewards": {"reward": 1.0} if rewards else None,
        "error": error,
        "verifier_error": None,
        "agent_result": {
            "usage_source": source,
            "usage_source_legacy": legacy,
            "n_input_tokens": tokens,
            "n_output_tokens": tokens,
            "n_cache_read_tokens": None if tokens is None else 0,
            "n_cache_creation_tokens": None if tokens is None else 0,
            "total_tokens": None if tokens is None else tokens * 2,
            "cost_usd": cost,
            "cost_source": cost_source,
        },
    }


def test_summary_totals_count_errored_trials_and_name_their_sources():
    """An errored trial (a timeout with no reward) spent what it spent: the
    totals count it. A trial nobody counted is not a 0; an agent-routed
    trial is named but not pooled."""
    from benchflow._utils.evaluation_results import usage_summary

    summary = usage_summary(
        {
            "proxied": _trial(
                "proxy",
                "provider_response",
                tokens=100,
                cost=0.5,
                cost_source="list_price_estimate",
            ),
            "timed_out": _trial(
                "stream_log",
                "agent_native_acp",
                tokens=40,
                cost=0.2,
                cost_source="list_price_estimate",
                rewards=False,
                error="Agent prompt exceeded wall-clock budget 14400s",
            ),
            "billed": _trial(
                "proxy",
                "provider_response",
                tokens=10,
                cost=0.05,
                cost_source="provider",
            ),
            "nothing": _trial(None, "unavailable", tokens=None, cost=None),
            "routed": _trial(
                "end_of_turn",
                "agent-reported",
                tokens=1000,
                cost=9.0,
                cost_source="list_price_estimate",
            ),
        }
    )
    assert summary["total_input_tokens"] == 150
    assert summary["total_tokens"] == 300
    assert summary["total_cost_usd"] == pytest.approx(0.75)
    assert summary["usage_sources"] == {"end_of_turn": 1, "proxy": 2, "stream_log": 1}
    assert summary["cost_sources"] == {"list_price_estimate": 3, "provider": 1}
    assert summary["trials_without_usage"] == 1
    assert summary["trials_without_cost"] == 1
    assert summary["telemetry_coverage"] == pytest.approx(3 / 5)


@pytest.mark.parametrize(
    ("agent", "model", "env", "harness", "expected"),
    [
        (
            "claude-agent-acp",
            "anthropic/claude-opus-5-5",
            {"ANTHROPIC_API_KEY": "sk-test-key-000000"},
            "acp",
            ("api_key", "proxy", None, "list_price_estimate"),
        ),
        (
            "opencode",
            "openrouter/moonshotai/kimi-k3",
            {"OPENROUTER_API_KEY": "sk-or-test-000000"},
            "acp",
            ("api_key", "proxy", None, "provider"),
        ),
        (
            "claude-agent-acp",
            "anthropic/claude-opus-5-5",
            {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-test-0000"},
            "native",
            ("subscription", "end_of_turn", "stream_log", "list_price_estimate"),
        ),
        (
            "claude-agent-acp",
            "anthropic/claude-opus-5-5",
            {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-test-0000"},
            "acp",
            ("subscription", "end_of_turn", "session_log", "list_price_estimate"),
        ),
        (
            "codex-acp",
            "openai/gpt-6-astra",
            {"CODEX_ACCESS_TOKEN": "codex-test-token-0000"},
            "acp",
            ("subscription", "session_log", "session_log", "list_price_estimate"),
        ),
        (
            "codex-acp",
            "openai/gpt-6-astra",
            {"CODEX_ACCESS_TOKEN": "codex-test-token-0000"},
            "native",
            ("subscription", "session_log", "session_log", "list_price_estimate"),
        ),
    ],
)
def test_the_run_card_states_each_trial_s_capture_path(
    agent, model, env, harness, expected
):
    from benchflow.run_card import _capture_text, capture_path

    capture = capture_path(agent, model, env, harness)
    assert capture is not None
    assert (
        capture["auth"],
        capture["usage_source"],
        capture["salvage"],
        capture["cost_source"],
    ) == expected
    assert capture["harness"] == harness
    assert "cost from" in _capture_text(capture)


def test_a_control_has_no_capture_path():
    from benchflow.run_card import capture_path

    assert capture_path("oracle", None, {}, "acp") is None


def test_a_run_with_no_model_call_says_so_and_is_no_zero(tmp_path):
    """An oracle made no model call: its tokens and cost are null with
    no_usage_reason no_model_call, and summary.json counts it apart, not as a
    trial with 0 tokens."""
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow.rollout import _build_rollout_result

    _build_rollout_result(
        tmp_path / "oracle",
        task_name="task",
        rollout_name="task__oracle",
        agent="oracle",
        agent_name="oracle",
        model=None,
        n_tool_calls=0,
        prompts=["solve"],
        error=None,
        verifier_error=None,
        trajectory=[{"type": "agent_message", "text": "ran solve.sh"}],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime(2026, 11, 13, 6, 0),
        timing={},
        n_input_tokens=0,
        n_output_tokens=0,
        total_tokens=0,
    )
    data = json.loads((tmp_path / "oracle" / "result.json").read_text())
    agent = data["agent_result"]
    assert agent["no_usage_reason"] == "no_model_call"
    assert all(agent[field] is None for field in TOKENS)
    assert agent["cost_usd"] is None
    summary = usage_summary(
        {
            "oracle": data,
            "model": _trial(
                "proxy",
                "provider_response",
                tokens=10,
                cost=0.01,
                cost_source="list_price_estimate",
            ),
        }
    )
    assert summary["trials_without_model_call"] == 1
    assert summary["trials_without_usage"] == 0
    assert summary["telemetry_coverage"] == 1.0
    assert summary["total_input_tokens"] == 10
