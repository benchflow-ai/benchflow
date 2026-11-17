"""Cross-provider token-usage normalization (apples-to-apples).

`_exchange_token_usage` normalizes every provider's per-call usage to
BenchFlow's split (EN-374, the lead's ruling): `input_tokens` is the
**uncached** input only, beside the cache reads and writes, and `total_tokens`
is input + cache read + cache write + output. Providers report this
differently:

- Anthropic Messages / Bedrock Converse report `input_tokens`/`inputTokens` as
  the UNCACHED input, with cache reads/writes as SEPARATE additive fields: kept
  as reported.
- OpenAI (`prompt_tokens`/`input_tokens` + `*_tokens_details.cached_tokens`) and
  Gemini (`promptTokenCount` + `cachedContentTokenCount`) report a
  cache-INCLUSIVE prompt, with the cache as a subset: the cache read comes out.

Gemini's `thoughtsTokenCount` (reasoning, billed as output) is folded into
output for parity with Anthropic/OpenAI, which already include reasoning in
output; its `toolUsePromptTokenCount` is additive uncached input. (Before
EN-374 the input was normalized the other way, cache included.)
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
        130,  # 100 uncached as reported; total = 100 + 7 + 3 + 20
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
        347,  # 34 uncached; provider total 347 == 34 + 100 + 200 + 13
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
        111,  # 123 - 12 cached: the cache was inside input_tokens
        45,
        12,
        0,
        168,
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
        70,  # 150 - 80 cached
        20,
        80,
        0,
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
        54956,  # 66608 - 11652 cached
        757,  # output = 695 + 62 thoughts
        11652,
        0,
        67365,
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
        1100,  # input = 1000 prompt - 200 cached + 300 toolUsePromptTokenCount
        60,  # output = 50 candidates + 10 thoughts
        200,  # cache_read was a subset of the prompt
        0,
        1360,  # provider total == 1100 + 200 + 60
    ),
]


@pytest.mark.parametrize(
    "label,body,exp_in,exp_out,exp_cr,exp_cc,exp_total",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_input_tokens_are_uncached(
    label, body, exp_in, exp_out, exp_cr, exp_cc, exp_total
):
    u = _exchange_token_usage(_exchange(body))
    assert u.input_tokens == exp_in
    assert u.output_tokens == exp_out
    assert u.cache_read_tokens == exp_cr
    assert u.cache_creation_tokens == exp_cc
    assert u.total_tokens == exp_total
    # Core invariant: each token counted once, in exactly one part.
    assert u.total_tokens == (
        u.input_tokens + u.cache_read_tokens + u.cache_creation_tokens + u.output_tokens
    )
