"""Provider-routing policy for the mini-swe ACP shim.

mini-swe drives ``litellm.completion`` (chat-completions / anthropic-messages),
which can't speak the OpenAI Responses API. ``_litellm_prefix`` reconstructs the
litellm provider prefix from BenchFlow's resolved ``BENCHFLOW_PROVIDER_PROTOCOL``
so Azure (openai-completions) and Bedrock (openai-responses → anthropic-messages
surface for Claude models) route correctly. These are the cases the PR's
headline Azure/Bedrock support rides on, so they get a regression test.

The shim module is import-safe (no minisweagent import, no stdout redirect at
import time), so this runs in CI without the sandbox runtime.
"""

import pytest

from benchflow.agents.mini_swe_acp_shim import _is_anthropic_model, _litellm_prefix


@pytest.mark.parametrize(
    "protocol,model,expected",
    [
        # Azure Foundry OpenAI — OpenAI-compatible /openai/v1 endpoint.
        ("openai-completions", "gpt-5.5", "openai"),
        # Bedrock Opus — responses-primary provider, but its proxy serves an
        # anthropic-messages surface; Claude models must route there.
        ("openai-responses", "us.anthropic.claude-opus-4-7", "anthropic"),
        # Bedrock non-Claude — best-effort openai.
        ("openai-responses", "openai.gpt-oss-20b-1:0", "openai"),
        # Native anthropic-messages provider.
        ("anthropic-messages", "claude-sonnet-4-6", "anthropic"),
        # Protocol wins over model family for chat-completions endpoints.
        ("openai-completions", "claude-via-openai-gateway", "openai"),
        # Unknown/empty protocol — let litellm infer from the model name.
        ("", "gpt-4o-mini", ""),
    ],
)
def test_litellm_prefix(protocol: str, model: str, expected: str) -> None:
    assert _litellm_prefix(protocol, model) == expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("us.anthropic.claude-opus-4-7", True),
        ("claude-sonnet-4-6", True),
        ("gpt-5.5", False),
        ("openai.gpt-oss-20b-1:0", False),
        ("gemini-3.1-flash", False),
    ],
)
def test_is_anthropic_model(model: str, expected: bool) -> None:
    assert _is_anthropic_model(model) is expected
