"""One vocabulary for where a trial's tokens and cost came from, and null for
what nobody counted (EN-374, backported to 0.9.3).

result.json's ``agent_result`` names a trial's ``usage_source``
(``stream_log``, ``session_log``, ``end_of_turn``, ``proxy``, ``provider``),
its ``cost_source`` (``provider`` or ``list_price_estimate``) and the
``price_table_date``, and keeps the pre-EN-374 name as
``usage_source_legacy``. A trial nothing counted has null tokens and a null
cost, never 0. Guards EN-374 against frontier-design-benchmark PR #17, whose
errored Claude trials had no output tokens and no cost.
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

    _build_rollout_result(
        folder,
        task_name="task",
        rollout_name="task__1",
        agent="claude-agent-acp",
        agent_name="claude",
        model="claude-opus-5-5",
        n_tool_calls=0,
        prompts=["solve"],
        error=None,
        verifier_error=None,
        trajectory=[{"type": "agent_message", "text": "done"}],
        partial_trajectory=False,
        rewards={"reward": 1.0},
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
        ("unavailable", None),
        ("stream_log", "stream_log"),
        ("session_log", "session_log"),
        (None, None),
    ],
)
def test_every_usage_source_maps_to_the_result_vocabulary(old, new):
    from benchflow.usage_tracking import result_usage_source

    assert result_usage_source(old) == new


def test_a_trial_nobody_counted_is_null_not_zero(tmp_path):
    from benchflow.usage_tracking import usage_unavailable

    unknown = usage_unavailable()
    assert all(unknown[field] is None for field in TOKENS)
    agent = _write_result(tmp_path / "trial", **unknown)["agent_result"]
    assert all(agent[field] is None for field in TOKENS)
    assert agent["cost_usd"] is None
    assert agent["usage_source"] is None
    assert agent["cost_source"] is None
    assert agent["price_table_date"] is None
    assert agent["usage_source_legacy"] == "unavailable"


def test_a_counted_trial_carries_its_labels(tmp_path):
    agent = _write_result(
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
    )["agent_result"]
    assert agent["usage_source"] == "stream_log"
    assert agent["cost_source"] == "list_price_estimate"
    assert agent["price_table_date"] == "2026-11-04"
    assert agent["usage_source_legacy"] == "agent_native_acp"


def test_an_old_result_reads_in_the_new_vocabulary():
    """A 0.9.2 result.json keeps loading: its usage_source maps, and the old
    name stays readable as usage_source_legacy."""
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
    proxied = RolloutResult.from_dict(
        {"task_name": "t", "agent_result": {"usage_source": "provider_response"}}
    )
    assert proxied.usage_source == "proxy"
    assert proxied.usage_source_legacy == "provider_response"


def _write_agent_result(folder: Path, agent: str, **usage) -> dict:
    from benchflow.rollout import _build_rollout_result

    _build_rollout_result(
        folder,
        task_name="task",
        rollout_name=folder.name,
        agent=agent,
        agent_name=agent,
        model=None,
        n_tool_calls=0,
        prompts=["solve"],
        error=None,
        verifier_error=None,
        trajectory=[{"type": "agent_message", "text": "done"}],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime(2026, 11, 13, 6, 0),
        timing={},
        **usage,
    )
    return json.loads((folder / "result.json").read_text())


@pytest.mark.parametrize("agent", ["oracle", "nop", "script:/tmp/solve.sh"])
def test_a_trial_that_called_no_model_says_so(tmp_path, agent):
    """A control or an operator script calls no model: its tokens and cost
    are null with no_usage_reason "no_model_call", and summary.json counts it
    apart, neither as a 0 nor as missing usage (EN-374)."""
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow.usage_tracking import usage_unavailable

    control = _write_agent_result(tmp_path / "control", agent, **usage_unavailable())
    agent_result = control["agent_result"]
    assert agent_result["no_usage_reason"] == "no_model_call"
    assert all(agent_result[field] is None for field in TOKENS)
    assert agent_result["cost_usd"] is None
    model = _write_agent_result(
        tmp_path / "model", "claude-agent-acp", **usage_unavailable()
    )
    assert model["agent_result"]["no_usage_reason"] is None

    summary = usage_summary({"control": control, "model": model})
    assert summary["trials_without_model_call"] == 1
    assert summary["trials_without_usage"] == 1
    assert summary["trials_without_cost"] == 1
    assert summary["total_tokens"] is None
    assert summary["usage_sources"] == {}
