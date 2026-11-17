"""One token split in every record (EN-374, the lead's ruling, 0.9.3).

``n_input_tokens`` is the uncached input only; ``n_cache_read_tokens`` and
``n_cache_write_tokens`` (``n_cache_creation_tokens`` kept as its alias) sit
beside it, the writes split into ``n_cache_write_5m_tokens`` and
``n_cache_write_1h_tokens`` where the provider reports it; ``n_output_tokens``
includes reasoning. Unknown is null. 0.9.2's proxy folded the cache into the
input for every provider, so a priced cache read looked like fresh input to
any reader that took ``n_input_tokens`` at its word. ATIF keeps ATIF's
meaning (``total_prompt_tokens`` counts every prompt token); the split goes
in ``final_metrics.extra``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

# The same call in three providers' words: 100 fresh input tokens, 900 read
# from the cache, 20 output tokens of which 5 are reasoning.
ANTHROPIC = {
    "usage": {
        "input_tokens": 100,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 0,
        "output_tokens": 20,
    }
}
OPENAI = {
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 20,
        "total_tokens": 1020,
        "prompt_tokens_details": {"cached_tokens": 900},
        "completion_tokens_details": {"reasoning_tokens": 5},
    }
}
GEMINI = {
    "usageMetadata": {
        "promptTokenCount": 1000,
        "cachedContentTokenCount": 900,
        "candidatesTokenCount": 15,
        "thoughtsTokenCount": 5,
        "totalTokenCount": 1020,
    }
}


def _usage(body: dict):
    from benchflow.trajectories.types import _exchange_token_usage

    return _exchange_token_usage(SimpleNamespace(response=SimpleNamespace(body=body)))


@pytest.mark.parametrize(
    "body", [ANTHROPIC, OPENAI, GEMINI], ids=["anthropic", "openai", "gemini"]
)
def test_every_provider_s_input_is_uncached(body):
    usage = _usage(body)
    assert usage.input_tokens == 100
    assert usage.cache_read_tokens == 900
    assert usage.cache_creation_tokens == 0
    assert usage.output_tokens == 20
    assert usage.total_tokens == 1020


def test_anthropic_reports_its_cache_writes_by_lifetime():
    usage = _usage(
        {
            "usage": {
                "input_tokens": 2,
                "cache_read_input_tokens": 18362,
                "cache_creation_input_tokens": 8526,
                "cache_creation": {
                    "ephemeral_1h_input_tokens": 8526,
                    "ephemeral_5m_input_tokens": 0,
                },
                "output_tokens": 127,
            }
        }
    )
    assert usage.input_tokens == 2
    assert usage.cache_creation_tokens == 8526
    assert (usage.cache_write_5m_tokens, usage.cache_write_1h_tokens) == (0, 8526)
    assert usage.total_tokens == 2 + 18362 + 8526 + 127
    assert (
        _usage(OPENAI).cache_write_5m_tokens,
        _usage(OPENAI).cache_write_1h_tokens,
    ) == (
        None,
        None,
    )


@pytest.mark.parametrize(
    "body", [ANTHROPIC, OPENAI, GEMINI], ids=["anthropic", "openai", "gemini"]
)
def test_the_same_call_costs_the_same_in_every_shape(body):
    """Cost parity: priced from the split, each token once, at its own rate
    (gpt-6.1-sol: $2/M input, $0.10/M cache read, $10/M output)."""
    from benchflow._utils.pricing import TokenSlice, price_slices

    usage = _usage(body)
    priced = price_slices(
        [
            TokenSlice(
                model="gpt-6.1-sol",
                input=usage.input_tokens,
                output=usage.output_tokens,
                cache_read=usage.cache_read_tokens,
                cache_write=usage.cache_creation_tokens,
            )
        ]
    )
    assert priced is not None
    assert priced.usd == pytest.approx(100 * 2e-6 + 900 * 0.1e-6 + 20 * 10e-6)


def _exchange(body: dict):
    from benchflow.trajectories.types import LLMExchange, LLMRequest, LLMResponse

    return LLMExchange(
        request=LLMRequest(
            timestamp=datetime(2026, 11, 13), method="POST", path="/", body={}
        ),
        response=LLMResponse(
            timestamp=datetime(2026, 11, 13), status_code=200, body=body
        ),
        duration_ms=1.0,
        metadata={},
    )


def test_the_proxy_s_usage_carries_the_split():
    from benchflow.providers.litellm_logging import extract_usage_from_trajectory
    from benchflow.trajectories.types import Trajectory

    trajectory = Trajectory(session_id="s", agent_name="a")
    trajectory.exchanges = [
        _exchange(OPENAI),
        _exchange(
            {
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 300,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 300,
                        "ephemeral_1h_input_tokens": 0,
                    },
                    "output_tokens": 4,
                }
            }
        ),
    ]
    usage = extract_usage_from_trajectory(trajectory)
    assert usage["n_input_tokens"] == 110
    assert usage["n_cache_read_tokens"] == 900
    # The OpenAI answer names no cache writes: the trial's writes, and their
    # 5-minute and 1-hour shares, are unknown, not the other call's 300.
    assert usage["n_cache_creation_tokens"] is None
    assert usage["n_cache_write_5m_tokens"] is None
    assert usage["n_cache_write_1h_tokens"] is None
    assert usage["n_output_tokens"] == 24
    assert usage["total_tokens"] == 1020 + 314


def test_a_cache_write_without_its_lifetimes_leaves_the_split_unknown():
    from benchflow.providers.litellm_logging import extract_usage_from_trajectory
    from benchflow.trajectories.types import Trajectory

    trajectory = Trajectory(session_id="s", agent_name="a")
    trajectory.exchanges = [
        _exchange(
            {
                "usage": {
                    "input_tokens": 1,
                    "cache_creation_input_tokens": 9,
                    "output_tokens": 1,
                }
            }
        )
    ]
    usage = extract_usage_from_trajectory(trajectory)
    assert usage["n_cache_creation_tokens"] == 9
    assert usage["n_cache_write_5m_tokens"] is None
    assert usage["n_cache_write_1h_tokens"] is None


# OpenRouter's documented usage object (https://openrouter.ai/docs/use-cases/
# usage-accounting, read 2026-11-13): "prompt_tokens": 194, "completion_tokens":
# 2, "total_tokens": 196, "prompt_tokens_details": {"cached_tokens": 0,
# "cache_write_tokens": 100}. 194 + 2 = 196: the 100 tokens written to the
# cache are inside prompt_tokens, as the cached tokens read are. Prompt 10,000
# with 6,000 cached and 3,000 written is 1,000 uncached input, 6,000 cache
# reads and 3,000 cache writes.
OPENROUTER = {
    "id": "gen-123",
    "object": "chat.completion",
    "model": "anthropic/claude-sonnet-5-5",
    "created": 1,
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
        }
    ],
    "usage": {
        "prompt_tokens": 10000,
        "completion_tokens": 100,
        "total_tokens": 10100,
        "prompt_tokens_details": {"cached_tokens": 6000, "cache_write_tokens": 3000},
        "cost": 0.0123,
    },
}


def _openrouter_body(shape: str) -> dict:
    """OpenRouter's answer, raw or as the pinned LiteLLM hands it to the
    gateway's logger: its chat answer (``prompt_tokens_details`` gains
    ``cache_creation_tokens``) or its Responses-API bridge's, the shape a
    Codex run records (``input_tokens`` and ``input_tokens_details``)."""
    if shape == "openrouter":
        return json.loads(json.dumps(OPENROUTER))
    import litellm
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_model_response_object,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    response = convert_to_model_response_object(
        response_object=json.loads(json.dumps(OPENROUTER)),
        model_response_object=litellm.ModelResponse(),
    )
    if shape == "litellm-chat":
        return json.loads(json.dumps(response.model_dump(), default=str))
    usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
        response
    )
    return {"usage": json.loads(json.dumps(usage.model_dump(), default=str))}


@pytest.mark.parametrize("shape", ["openrouter", "litellm-chat", "litellm-responses"])
def test_openrouter_s_cache_writes_leave_the_uncached_input(shape):
    from benchflow.providers.litellm_logging import extract_usage_from_trajectory
    from benchflow.trajectories.types import Trajectory

    body = _openrouter_body(shape)
    usage = _usage(body)
    assert usage.input_tokens == 1000
    assert usage.cache_read_tokens == 6000
    assert usage.cache_creation_tokens == 3000
    assert usage.output_tokens == 100
    assert usage.total_tokens == 10100
    # OpenRouter does not split its writes by lifetime: unknown, not 0.
    assert (usage.cache_write_5m_tokens, usage.cache_write_1h_tokens) == (None, None)

    trajectory = Trajectory(session_id="s", agent_name="a")
    trajectory.exchanges = [_exchange(body)]
    trial = extract_usage_from_trajectory(trajectory)
    assert trial["n_input_tokens"] == 1000
    assert trial["n_cache_read_tokens"] == 6000
    assert trial["n_cache_creation_tokens"] == 3000
    assert trial["n_cache_write_5m_tokens"] is None
    assert trial["n_cache_write_1h_tokens"] is None
    assert trial["total_tokens"] == 10100


def test_openrouter_s_cache_writes_are_priced_as_writes():
    """Priced from the split, each token once at its own rate (Claude Sonnet
    5.5's snapshot: $2/M input, $0.10/M cache read, $2.50/M cache write,
    $10/M output): $0.0111. Left in the input, the 3,000 written tokens were
    priced at the input rate: $0.0096."""
    from benchflow._utils.pricing import TokenSlice, price_slices

    usage = _usage(_openrouter_body("openrouter"))
    priced = price_slices(
        [
            TokenSlice(
                model="claude-sonnet-5-5",
                input=usage.input_tokens,
                output=usage.output_tokens,
                cache_read=usage.cache_read_tokens,
                cache_write=usage.cache_creation_tokens,
                prompt=10000,
            )
        ]
    )
    assert priced is not None
    assert priced.usd == pytest.approx(
        1000 * 2e-6 + 6000 * 0.1e-6 + 3000 * 2.5e-6 + 100 * 10e-6
    )


def test_codex_s_native_report_leaves_the_cache_out_of_its_input():
    from benchflow.native_harness.codex import codex_usage

    snapshot = codex_usage(
        {
            "input_tokens": 12883,
            "cached_input_tokens": 11264,
            "cache_write_input_tokens": 0,
            "output_tokens": 46,
            "reasoning_output_tokens": 0,
        }
    )
    assert snapshot is not None
    assert snapshot["input_tokens"] == 12883 - 11264
    assert snapshot["cached_read_tokens"] == 11264
    assert snapshot["total_tokens"] == 12929


def _write_result(folder: Path, **usage) -> dict:
    from benchflow.rollout import _build_rollout_result

    _build_rollout_result(
        folder,
        task_name="task",
        rollout_name="task__1",
        agent="claude-agent-acp",
        agent_name="claude",
        model="claude-sonnet-5-5",
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


def test_result_json_atif_and_summary_carry_the_split(tmp_path):
    """The 0.9.2 native timeout's session log (x2 evidence): 2 fresh, 18,362
    cache read, 8,526 cache write (all 1-hour), 127 output."""
    from benchflow._utils.evaluation_results import usage_summary

    data = _write_result(
        tmp_path / "trial",
        n_input_tokens=2,
        n_output_tokens=127,
        n_cache_read_tokens=18362,
        n_cache_creation_tokens=8526,
        n_cache_write_5m_tokens=0,
        n_cache_write_1h_tokens=8526,
        total_tokens=27017,
        cost_usd=0.01,
        usage_source="session_log",
        cost_source="list_price_estimate",
        price_table_date="2026-11-04",
    )
    agent = data["agent_result"]
    assert agent["n_input_tokens"] == 2
    assert agent["n_cache_write_tokens"] == agent["n_cache_creation_tokens"] == 8526
    assert (agent["n_cache_write_5m_tokens"], agent["n_cache_write_1h_tokens"]) == (
        0,
        8526,
    )
    for final in (
        data["final_metrics"],
        json.loads((tmp_path / "trial" / "trainer" / "atif.json").read_text())[
            "final_metrics"
        ],
    ):
        assert final["total_prompt_tokens"] == 2 + 18362 + 8526
        assert final["total_cached_tokens"] == 18362
        assert final["total_completion_tokens"] == 127
        assert final["extra"] == {
            "n_input_tokens": 2,
            "n_cache_read_tokens": 18362,
            "n_cache_write_tokens": 8526,
            "n_cache_write_5m_tokens": 0,
            "n_cache_write_1h_tokens": 8526,
            "n_output_tokens": 127,
            "usage_source": "session_log",
            "cost_source": "list_price_estimate",
            "price_table_date": "2026-11-04",
        }
    summary = usage_summary({"t": data})
    assert summary["total_input_tokens"] == 2
    assert summary["total_cache_write_tokens"] == 8526
    assert summary["total_cache_write_1h_tokens"] == 8526
    assert summary["total_tokens"] == 27017


def test_unknown_parts_are_null_everywhere(tmp_path):
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow.usage_tracking import usage_unavailable

    data = _write_result(tmp_path / "trial", **usage_unavailable())
    agent = data["agent_result"]
    for field in (
        "n_input_tokens",
        "n_cache_read_tokens",
        "n_cache_write_tokens",
        "n_cache_write_5m_tokens",
        "n_cache_write_1h_tokens",
        "n_output_tokens",
        "total_tokens",
    ):
        assert agent[field] is None, field
    assert data["final_metrics"]["total_prompt_tokens"] is None
    assert data["final_metrics"]["extra"]["n_input_tokens"] is None
    summary = usage_summary({"t": data})
    assert summary["total_input_tokens"] is None
    assert summary["total_cache_write_tokens"] is None


def test_a_response_without_its_split_leaves_it_unknown():
    """Guards EN-386 (0.9.2 and rc3): Harvey's MiMo responses carry only
    total_tokens (6,027 and 6,099). result.json said input, output and cache
    0 beside a total of 12,126. The split is unknown, so every count is null
    (a total of unknown parts too); the providers' own total stays on record."""
    from benchflow.providers.litellm_logging import extract_usage_from_trajectory
    from benchflow.trajectories.types import Trajectory

    trajectory = Trajectory(session_id="s", agent_name="harvey-lab-harness")
    trajectory.exchanges = [
        _exchange({"usage": {"total_tokens": 6027}}),
        _exchange({"usage": {"total_tokens": 6099}}),
    ]
    usage = extract_usage_from_trajectory(trajectory)
    for field in (
        "n_input_tokens",
        "n_output_tokens",
        "n_cache_read_tokens",
        "n_cache_creation_tokens",
        "total_tokens",
    ):
        assert usage[field] is None, field
    assert usage["usage_source"] == "provider_response"
    assert usage["usage_details"] == {
        "provider_total_tokens": 12126,
        "split": "unreported",
    }
