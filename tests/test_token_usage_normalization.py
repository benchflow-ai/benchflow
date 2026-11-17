"""Cross-provider token-usage normalization (apples-to-apples).

`_exchange_token_usage` normalizes every provider's per-call usage to one split
(the lead's ruling of 2026-11-13, which replaced the cache-inclusive
input this test pinned before): `input_tokens` is the UNCACHED input only,
cache reads and cache writes are their own counts beside it, and output counts
reasoning. Providers report this differently:

- Anthropic Messages / Bedrock Converse report `input_tokens`/`inputTokens` as
  the UNCACHED input, with cache reads/writes as SEPARATE additive fields: the
  input is taken as reported.
- OpenAI (`prompt_tokens`/`input_tokens` + `*_tokens_details.cached_tokens`) and
  Gemini (`promptTokenCount` + `cachedContentTokenCount`) report the
  cache-INCLUSIVE prompt: the cache reads are taken out of it. They report no
  cache writes, which are then None, not 0.

Gemini's `thoughtsTokenCount` (reasoning, billed as output) is folded into
output, and its `toolUsePromptTokenCount` into input. With no provider total,
the total is the sum of the four disjoint parts.
"""

from __future__ import annotations

import pytest

from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    _exchange_token_usage,
)


def _exchange(body: dict) -> LLMExchange:
    return LLMExchange(request=LLMRequest(), response=LLMResponse(body=body))


# (label, response body, expected input, output, cache_read, cache_creation, total)
_CASES = [
    (
        "anthropic_messages_uncached_delta",
        {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 7,
                "cache_creation_input_tokens": 3,
            }
        },
        100,
        20,
        7,
        3,
        130,  # input as reported; total = 100 + 20 + 7 + 3
    ),
    (
        "bedrock_converse_uncached_delta",
        {
            "usage": {
                "inputTokens": 34,
                "outputTokens": 13,
                "cacheReadInputTokens": 100,
                "cacheWriteInputTokens": 200,
                "totalTokens": 347,
            }
        },
        34,
        13,
        100,
        200,
        347,  # input as reported; provider total 347 == 34 + 13 + 100 + 200
    ),
    (
        "openai_responses_cache_is_subset",  # input_tokens here is the TOTAL
        {
            "usage": {
                "input_tokens": 123,
                "output_tokens": 45,
                "total_tokens": 168,
                "input_tokens_details": {"cached_tokens": 12},
            }
        },
        111,
        45,
        12,
        None,
        168,  # 123 less its 12 cached; OpenAI reports no cache writes
    ),
    (
        "openai_chat_cache_is_subset",
        {
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 80},
            }
        },
        70,
        20,
        80,
        None,
        170,
    ),
    (
        "gemini_total_prompt_thoughts_as_output",
        {
            "usageMetadata": {
                "promptTokenCount": 66608,
                "candidatesTokenCount": 695,
                "cachedContentTokenCount": 11652,
                "thoughtsTokenCount": 62,
                "totalTokenCount": 67365,
            }
        },
        66608 - 11652,
        757,
        11652,
        None,
        67365,  # output = 695 + 62 thoughts; the cached part taken out
    ),
    (
        "gemini_tool_use_prompt_folded_into_input",
        {
            "usageMetadata": {
                "promptTokenCount": 1000,
                "candidatesTokenCount": 50,
                "cachedContentTokenCount": 200,
                "thoughtsTokenCount": 10,
                "toolUsePromptTokenCount": 300,
                "totalTokenCount": 1360,
            }
        },
        1100,  # 1000 promptTokenCount less 200 cached, plus 300 tool-use prompt
        60,  # output = 50 candidates + 10 thoughts
        200,  # cache_read was a subset of the prompt
        None,
        1360,  # provider total == input + output + cache read
    ),
]


@pytest.mark.parametrize(
    "label,body,exp_in,exp_out,exp_cr,exp_cc,exp_total",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_input_tokens_are_uncached_with_the_cache_beside_them(
    label, body, exp_in, exp_out, exp_cr, exp_cc, exp_total
):
    u = _exchange_token_usage(_exchange(body))
    assert u.input_tokens == exp_in
    assert u.output_tokens == exp_out
    assert u.cache_read_tokens == exp_cr
    assert u.cache_creation_tokens == exp_cc
    assert u.total_tokens == exp_total
    # Core invariant: the four parts are disjoint and add up to the total.
    assert (
        u.input_tokens
        + u.output_tokens
        + u.cache_read_tokens
        + (u.cache_creation_tokens or 0)
        == exp_total
    )
