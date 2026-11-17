"""The 0.9.3 review's findings on the token split, unknown counts and the
partial-count marker (EN-374)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

ANTHROPIC_USAGE = {
    "input_tokens": 100,
    "cache_read_input_tokens": 900,
    "cache_creation_input_tokens": 300,
    "output_tokens": 20,
    "cache_creation": {
        "ephemeral_5m_input_tokens": 300,
        "ephemeral_1h_input_tokens": 0,
    },
}


def _litellm_chat_usage() -> dict:
    """Anthropic's usage as LiteLLM 1.103.3's chat-completions shape carries
    it (Anthropic and Bedrock through the gateway): ``prompt_tokens`` is the
    input plus the cache reads and writes."""
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    usage = AnthropicConfig().calculate_usage(
        usage_object=dict(ANTHROPIC_USAGE), reasoning_content=None
    )
    return json.loads(usage.model_dump_json())


def _usage_of(body_usage: dict):
    from benchflow.trajectories.types import (
        LLMExchange,
        LLMRequest,
        LLMResponse,
        _exchange_token_usage,
    )

    exchange = LLMExchange(
        request=LLMRequest(), response=LLMResponse(body={"usage": body_usage})
    )
    return _exchange_token_usage(exchange)


def _record(usage: dict) -> str:
    return json.dumps(
        {
            "event": "success",
            "request": {"body": {"model": "anthropic/claude-opus-5-5"}},
            "response": {
                "model": "claude-opus-5-5",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": usage,
            },
            "response_cost": None,
        }
    )


def _extract(*usages: dict) -> dict:
    from benchflow.providers.litellm_logging import (
        extract_usage_from_trajectory,
        trajectory_from_litellm_callback_log,
    )

    trajectory = trajectory_from_litellm_callback_log(
        "\n".join(_record(u) for u in usages), session_id="s", agent_name="opencode"
    )
    return extract_usage_from_trajectory(trajectory)


def test_litellm_s_chat_shape_takes_the_cache_writes_out_of_the_input():
    """100 uncached, 900 read, 300 written: the input is 100, not 400, and
    the 5-minute/1-hour split survives LiteLLM's shape."""
    from benchflow._utils.result_metadata import atif_prompt_tokens

    shaped = _litellm_chat_usage()
    assert shaped["prompt_tokens"] == 1300
    usage = _usage_of(shaped)
    assert usage.input_tokens == 100
    assert usage.cache_read_tokens == 900
    assert usage.cache_creation_tokens == 300
    assert usage.cache_write_5m_tokens == 300
    assert usage.cache_write_1h_tokens == 0
    counts = _extract(shaped)
    assert counts["n_input_tokens"] == 100
    assert counts["n_cache_write_5m_tokens"] == 300
    assert counts["total_tokens"] == 1320
    assert atif_prompt_tokens({**counts, "n_cache_write_tokens": 300}) == 1300
    # Anthropic's own Messages shape is uncached already.
    assert _usage_of(dict(ANTHROPIC_USAGE)).input_tokens == 100


def test_an_output_only_response_leaves_the_input_unknown():
    counts = _extract({"completion_tokens": 20})
    assert counts["n_output_tokens"] == 20
    assert counts["n_input_tokens"] is None
    assert counts["total_tokens"] is None


def test_atif_prompt_total_with_an_unknown_part_is_unknown():
    from benchflow._utils.result_metadata import atif_prompt_tokens

    known = {"n_input_tokens": 10, "n_cache_read_tokens": 5, "n_cache_write_tokens": 0}
    assert atif_prompt_tokens(known) == 15
    assert atif_prompt_tokens({**known, "n_cache_read_tokens": None}) is None
    assert (
        atif_prompt_tokens(
            {
                "n_input_tokens": 10,
                "n_cache_read_tokens": 5,
                "n_cache_write_tokens": None,
            }
        )
        is None
    )


def _acp_rollout(tmp_path, monkeypatch):
    from benchflow.acp.session import ACPSession
    from tests.test_en374_salvage import _rollout

    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._session = ACPSession("s")
    rollout._native_usage_checkpoint = None
    return rollout, trial


def test_an_end_of_turn_report_without_cache_counts_leaves_them_unknown(
    tmp_path, monkeypatch
):
    rollout, _ = _acp_rollout(tmp_path, monkeypatch)
    rollout._session.record_prompt_usage(
        SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120)
    )
    rollout._collect_native_acp_usage()
    usage = rollout._native_usage_metrics
    assert usage["n_input_tokens"] == 100
    assert usage["n_cache_read_tokens"] is None
    assert usage["n_cache_creation_tokens"] is None
    # Its price would pass for the whole: the cost stays unknown too.
    from benchflow.rollout._cost import apply_list_price

    rollout._session_reported_models = ["claude-opus-5-5"]  # a priced model
    rollout._usage_metrics = dict(usage)
    apply_list_price(rollout)
    assert rollout._usage_metrics["cost_usd"] is None


@pytest.mark.asyncio
async def test_a_turn_that_never_reported_marks_the_count_incomplete(
    tmp_path, monkeypatch
):
    """Turn one reported, turn two was cut off and no session log could be
    read: the count is turn one's, marked so it is not read as whole."""
    from benchflow.rollout import _build_rollout_result

    rollout, trial = _acp_rollout(tmp_path, monkeypatch)
    rollout._session.record_prompt_usage(
        SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            total_tokens=1120,
            cached_read_tokens=1000,
            cached_write_tokens=0,
        )
    )
    rollout._collect_native_acp_usage()
    rollout._usage_report_missing = True  # turn two ended without its report
    await rollout._cleanup_before_teardown()
    usage = rollout._usage_metrics
    assert usage["n_input_tokens"] == 100
    assert usage["usage_incomplete"] is True
    assert "without its usage report" in usage["usage_incomplete_reason"]
    _build_rollout_result(
        trial,
        task_name="task",
        rollout_name="task__1",
        agent="claude-agent-acp",
        agent_name="claude-agent-acp",
        model="claude-opus-5-5",
        n_tool_calls=0,
        prompts=["a", "b"],
        error="Agent timed out",
        verifier_error=None,
        trajectory=[],
        partial_trajectory=True,
        rewards=None,
        started_at=rollout._started_at,
        timing={},
        **usage,
    )
    agent = json.loads((trial / "result.json").read_text())["agent_result"]
    assert agent["usage_incomplete"] is True


def test_metrics_totals_match_summary_json(tmp_path):
    """bench eval metrics counts the tokens of every trial a trusted source
    counted, errored ones included, as summary.json does; unknown is None."""
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow.metrics import collect_metrics

    rows = {
        "ok": {"rewards": {"reward": 1.0}, "error": None, "tokens": 100},
        "crash": {"rewards": None, "error": "ACP error", "tokens": 40},
    }
    results = {}
    for name, row in rows.items():
        trial = tmp_path / f"{name}__1"
        trial.mkdir()
        data = {
            "task_name": name,
            "rollout_name": trial.name,
            "agent": "codex-acp",
            "model": "openai/gpt-6.1-sol",
            "rewards": row["rewards"],
            "error": row["error"],
            "agent_result": {
                "n_input_tokens": row["tokens"],
                "n_output_tokens": 1,
                "n_cache_read_tokens": None,
                "total_tokens": row["tokens"] + 1,
                "cost_usd": None,
                "usage_source": "proxy",
            },
        }
        (trial / "result.json").write_text(json.dumps(data))
        results[name] = data
    metrics = collect_metrics(tmp_path)
    summary = usage_summary(results)
    assert metrics.total_input_tokens == summary["total_input_tokens"] == 140
    assert metrics.total_tokens == summary["total_tokens"] == 142
    assert metrics.total_cache_read_tokens is None
    assert summary["total_cache_read_tokens"] is None
    assert metrics.telemetry_coverage == summary["telemetry_coverage"] == 1.0


def test_a_codex_count_has_its_atif_prompt_total(tmp_path):
    """Codex logs cache_write_input_tokens (always 0): a known count, so the
    trial's ATIF prompt total is Codex's own prompt, 12,894 tokens, not null
    beside a total of 12,960. A log without the field leaves it unknown."""
    import shutil

    from benchflow._utils.result_metadata import atif_prompt_tokens
    from benchflow.rollout._cost import codex_session_tokens
    from tests.test_en374_salvage import _codex_session
    from tests.test_en374_stopped_trials import CTRL_C_CALL

    home = tmp_path / "home"
    _codex_session(home, [CTRL_C_CALL])
    trial = tmp_path / "trial"
    shutil.copytree(home / ".codex" / "sessions", trial / "agent" / "codex-sessions")
    counts = codex_session_tokens(trial)
    assert counts is not None
    assert counts["n_cache_creation_tokens"] == 0
    assert counts["total_tokens"] == 12894 + 66
    assert atif_prompt_tokens(counts) == 12894

    without = {k: v for k, v in CTRL_C_CALL.items() if k != "cache_write_input_tokens"}
    other = tmp_path / "other"
    shutil.rmtree(home)
    _codex_session(home, [without])
    shutil.copytree(home / ".codex" / "sessions", other / "agent" / "codex-sessions")
    assert codex_session_tokens(other)["n_cache_creation_tokens"] is None
