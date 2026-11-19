"""One token split in every BenchFlow record (the lead's ruling).

``n_input_tokens`` is the uncached input only, whatever the provider's shape:
the proxy path folded the cache into it (claude-acp on MiMo recorded
``in=22001``) while Claude's session log wrote ``in=8`` beside 65,177 cache
reads. Cache reads and writes are their own fields (writes with their
5-minute and 1-hour parts where Anthropic splits them), output counts
reasoning, an unreported count is null (OpenAI reports no cache writes), and
``total_tokens`` is the sum of the known parts, null when input or output is
unknown (harvey-lab-harness recorded a measured total beside zero
splits). The same logical call priced from each shape costs the same.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
)

MODEL = "claude-opus-5-5"


def _trajectory(body: dict) -> Trajectory:
    now = datetime(2026, 11, 13, 7, 0)
    trajectory = Trajectory(session_id="s", agent_name="agent")
    trajectory.exchanges.append(
        LLMExchange(
            request=LLMRequest(timestamp=now, method="POST", path="/v1", body={}),
            response=LLMResponse(timestamp=now, status_code=200, body=body),
            duration_ms=1.0,
        )
    )
    return trajectory


# One call: 1,000 uncached input tokens, 4,000 cache reads, 200 output.
SHAPES = {
    "anthropic": {
        "usage": {
            "input_tokens": 1000,
            "cache_read_input_tokens": 4000,
            "cache_creation_input_tokens": 0,
            "output_tokens": 200,
        }
    },
    "openai-chat": {
        "usage": {
            "prompt_tokens": 5000,
            "completion_tokens": 200,
            "total_tokens": 5200,
            "prompt_tokens_details": {"cached_tokens": 4000},
        }
    },
    "openai-responses": {
        "usage": {
            "input_tokens": 5000,
            "input_tokens_details": {"cached_tokens": 4000},
            "output_tokens": 200,
            "total_tokens": 5200,
        }
    },
    "gemini": {
        "usageMetadata": {
            "promptTokenCount": 5000,
            "cachedContentTokenCount": 4000,
            "candidatesTokenCount": 150,
            "thoughtsTokenCount": 50,
            "totalTokenCount": 5200,
        }
    },
    "bedrock": {
        "usage": {
            "inputTokens": 1000,
            "cacheReadInputTokens": 4000,
            "outputTokens": 200,
            "totalTokens": 5200,
        }
    },
}
# The same call with 500 cache writes, 200 of them for an hour: Anthropic's
# own shape, and LiteLLM's OpenAI-shaped answer for an Anthropic model, whose
# prompt_tokens counts the reads and the writes.
WITH_WRITES = {
    "anthropic": {
        "usage": {
            "input_tokens": 1000,
            "cache_read_input_tokens": 4000,
            "cache_creation_input_tokens": 500,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 300,
                "ephemeral_1h_input_tokens": 200,
            },
            "output_tokens": 200,
        }
    },
    "litellm-openai-for-anthropic": {
        "usage": {
            "prompt_tokens": 5500,
            "completion_tokens": 200,
            "total_tokens": 5700,
            "prompt_tokens_details": {"cached_tokens": 4000},
            "cache_read_input_tokens": 4000,
            "cache_creation_input_tokens": 500,
        }
    },
}


def _usage(body: dict) -> dict:
    from benchflow.providers.litellm_logging import extract_usage_from_trajectory

    return extract_usage_from_trajectory(_trajectory(body))


def _price(usage: dict) -> float:
    from benchflow._utils.pricing import TokenSlice, price_slices

    priced = price_slices(
        [
            TokenSlice(
                model=MODEL,
                input=usage["n_input_tokens"],
                output=usage["n_output_tokens"],
                cache_read=usage["n_cache_read_tokens"] or 0,
                cache_write=usage["n_cache_creation_tokens"] or 0,
                cache_write_1h=usage["n_cache_write_1h_tokens"] or 0,
            )
        ]
    )
    assert priced is not None
    return priced.usd


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_every_shape_gives_the_uncached_input(shape):
    usage = _usage(SHAPES[shape])
    assert usage["n_input_tokens"] == 1000
    assert usage["n_cache_read_tokens"] == 4000
    assert usage["n_output_tokens"] == 200
    if shape in ("openai-chat", "openai-responses", "gemini", "bedrock"):
        # These shapes report no cache writes: unknown, not 0.
        assert usage["n_cache_creation_tokens"] is None
    assert usage["n_cache_write_1h_tokens"] is None


def test_the_same_call_costs_the_same_from_every_shape():
    """Input at the input rate, reads at the read rate, output at the output
    rate, once each: no shape counts its cache twice."""
    costs = {shape: _price(_usage(body)) for shape, body in SHAPES.items()}
    expected = 1000 * 4e-6 + 4000 * 0.2e-6 + 200 * 20e-6
    for shape, cost in costs.items():
        assert cost == pytest.approx(expected), shape


@pytest.mark.parametrize("shape", sorted(WITH_WRITES))
def test_cache_writes_are_their_own_count_and_price(shape):
    usage = _usage(WITH_WRITES[shape])
    assert usage["n_input_tokens"] == 1000
    assert usage["n_cache_read_tokens"] == 4000
    assert usage["n_cache_creation_tokens"] == 500
    base = 1000 * 4e-6 + 4000 * 0.2e-6 + 200 * 20e-6
    if shape == "anthropic":
        assert usage["n_cache_write_5m_tokens"] == 300
        assert usage["n_cache_write_1h_tokens"] == 200
        assert _price(usage) == pytest.approx(base + 300 * 5e-6 + 200 * 8e-6)
    else:
        # LiteLLM's OpenAI shape gives no lifetime split: all at the 5m rate.
        assert usage["n_cache_write_1h_tokens"] is None
        assert _price(usage) == pytest.approx(base + 500 * 5e-6)


def test_a_total_without_its_split_is_no_count(tmp_path):
    """harvey-lab-harness's provider answered with a total and no
    split (12,080 tokens on rc3). The split is unknown, so is the total."""
    from benchflow.rollout import _build_rollout_result

    usage = _usage({"usage": {"total_tokens": 12080}})
    assert usage["n_input_tokens"] is None
    assert usage["n_output_tokens"] is None
    result = _build_rollout_result(
        tmp_path,
        task_name="hello",
        rollout_name="hello__1",
        agent="harvey-lab-harness",
        agent_name="harvey",
        model="xiaomi/mimo-v2.6-flash",
        n_tool_calls=0,
        prompts=["hello"],
        error=None,
        verifier_error=None,
        trajectory=[{"type": "agent_message", "text": "ok"}],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime(2026, 11, 13, 7, 0),
        timing={},
        **usage,
    )
    assert result.total_tokens is None
    assert result.n_input_tokens is None


def test_an_acp_report_without_a_split_keeps_it_unknown(tmp_path, monkeypatch):
    """An ACP agent that reports a total and no input/output split: those
    counts are None, never 0."""
    from benchflow.acp.session import ACPSession
    from benchflow.rollout._usage import _NATIVE_ACP_USAGE_SNAPSHOT_TO_RESULT
    from tests.test_en374_salvage import _rollout

    rollout, _ = _rollout(
        tmp_path, monkeypatch, agent="harvey-lab-harness", harness="acp", home=tmp_path
    )
    session = ACPSession("s")
    session.record_prompt_usage({"totalTokens": 12080, "outputTokens": 40})
    rollout._session = session
    rollout._collect_native_acp_usage()
    metrics = rollout._native_usage_metrics
    assert _NATIVE_ACP_USAGE_SNAPSHOT_TO_RESULT["input_tokens"] == "n_input_tokens"
    assert metrics["n_input_tokens"] is None
    assert metrics["n_cache_read_tokens"] is None
    assert metrics["n_output_tokens"] == 40


def test_codex_s_native_report_takes_the_cache_out_of_its_input():
    """Codex's turn.completed input counts the cached tokens (OpenAI's
    convention); its always-0 cache_write_input_tokens is no count."""
    from benchflow.native_harness.codex import codex_usage

    usage = codex_usage(
        {
            "input_tokens": 12883,
            "cached_input_tokens": 11264,
            "cache_write_input_tokens": 0,
            "output_tokens": 46,
            "reasoning_output_tokens": 0,
        }
    )
    assert usage is not None
    assert usage["input_tokens"] == 12883 - 11264
    assert usage["cached_read_tokens"] == 11264
    assert usage["cached_write_tokens"] is None
    assert usage["total_tokens"] == 12929
