"""Bedrock Claude 4.8+ adaptive-thinking normalization (host proxy) + shim parity.

The proxy must (a) convert the thinking SHAPE to the adaptive contract Bedrock
4.8+ requires *only when thinking was requested*, (b) resolve the effort from the
request / BENCHFLOW override rather than forcing MAX on every 4.8 call, and (c)
match model ids with an anchored regex kept identical to the Daytona sandbox shim.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import benchflow
from benchflow.providers.bedrock_runtime import (
    BEDROCK_ADAPTIVE_THINKING_RE,
    BEDROCK_THINKING_EFFORT_ENV,
    _normalize_bedrock_thinking_for_opus_4_8,
    _resolve_bedrock_thinking_effort,
    anthropic_request_to_bedrock_converse,
    openai_responses_request_to_bedrock_converse,
)

_MODEL = "bedrock/us.anthropic.claude-opus-4-8"


@pytest.fixture(autouse=True)
def _clear_effort_env(monkeypatch):
    monkeypatch.delenv(BEDROCK_THINKING_EFFORT_ENV, raising=False)


# --- model matcher (anchored) ------------------------------------------------
@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-8",
        "claude-opus-4-8-20260301",
        "us.anthropic.claude-opus-4-8",
        "bedrock/eu.anthropic.claude-opus-4-8",
        "claude-sonnet-4-9",
        "claude-haiku-4-10",
    ],
)
def test_regex_matches_4_8_plus(model):
    assert BEDROCK_ADAPTIVE_THINKING_RE.search(model.lower())


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-80",  # the un-anchored bug: must NOT match 4-8
        "claude-opus-4-100",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-3-5-sonnet",
        "gpt-4",
    ],
)
def test_regex_rejects_non_4_8(model):
    assert not BEDROCK_ADAPTIVE_THINKING_RE.search(model.lower())


# --- effort resolution -------------------------------------------------------
def test_effort_defaults_to_high_not_max(monkeypatch):
    assert _resolve_bedrock_thinking_effort(None) == "high"


def test_effort_honors_request_when_no_override():
    assert _resolve_bedrock_thinking_effort("xhigh") == "xhigh"


def test_effort_override_beats_request(monkeypatch):
    monkeypatch.setenv(BEDROCK_THINKING_EFFORT_ENV, "max")
    assert _resolve_bedrock_thinking_effort("low") == "max"


def test_effort_ignores_garbage_override(monkeypatch):
    monkeypatch.setenv(BEDROCK_THINKING_EFFORT_ENV, "turbo")
    assert _resolve_bedrock_thinking_effort(None) == "high"


# --- normalize behavior ------------------------------------------------------
def test_normalize_injects_adaptive_when_thinking_requested():
    payload = {"modelId": _MODEL}
    _normalize_bedrock_thinking_for_opus_4_8(
        payload, thinking_requested=True, requested_effort=None
    )
    fields = payload["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "adaptive"}
    assert fields["output_config"] == {"effort": "high"}  # NOT max


def test_normalize_noop_when_thinking_not_requested():
    payload = {"modelId": _MODEL}
    _normalize_bedrock_thinking_for_opus_4_8(
        payload, thinking_requested=False, requested_effort=None
    )
    assert "additionalModelRequestFields" not in payload


def test_normalize_noop_for_non_4_8_model():
    payload = {"modelId": "bedrock/us.anthropic.claude-opus-4-7"}
    _normalize_bedrock_thinking_for_opus_4_8(
        payload, thinking_requested=True, requested_effort="max"
    )
    assert "additionalModelRequestFields" not in payload


def test_normalize_uses_override_effort(monkeypatch):
    monkeypatch.setenv(BEDROCK_THINKING_EFFORT_ENV, "max")
    payload = {"modelId": _MODEL}
    _normalize_bedrock_thinking_for_opus_4_8(
        payload, thinking_requested=True, requested_effort="low"
    )
    assert payload["additionalModelRequestFields"]["output_config"] == {"effort": "max"}


# --- end-to-end caller wiring ------------------------------------------------
def test_anthropic_request_converts_thinking_to_adaptive():
    body = {
        "model": "us.anthropic.claude-opus-4-8",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }
    payload = anthropic_request_to_bedrock_converse(body)
    assert payload["additionalModelRequestFields"]["thinking"] == {"type": "adaptive"}


def test_anthropic_request_without_thinking_is_untouched():
    body = {
        "model": "us.anthropic.claude-opus-4-8",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    payload = anthropic_request_to_bedrock_converse(body)
    assert "additionalModelRequestFields" not in payload


def test_openai_responses_honors_requested_effort():
    body = {
        "model": "us.anthropic.claude-opus-4-8",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "reasoning": {"effort": "high"},
    }
    payload = openai_responses_request_to_bedrock_converse(body)
    fields = payload["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "adaptive"}
    assert fields["output_config"] == {"effort": "high"}


# --- effort override forwarded into the remote (Daytona) sandbox -------------
def test_direct_bedrock_mapping_forwards_effort_override(monkeypatch):
    from benchflow.providers.runtime import _apply_direct_bedrock_agent_mapping

    monkeypatch.setenv(BEDROCK_THINKING_EFFORT_ENV, "max")
    out = _apply_direct_bedrock_agent_mapping(
        {"AWS_REGION": "us-east-1"},
        agent="openhands",
        backend_model="us.anthropic.claude-opus-4-8",
        environment="daytona",
    )
    assert out[BEDROCK_THINKING_EFFORT_ENV] == "max"


def test_direct_bedrock_mapping_omits_effort_when_unset(monkeypatch):
    from benchflow.providers.runtime import _apply_direct_bedrock_agent_mapping

    monkeypatch.delenv(BEDROCK_THINKING_EFFORT_ENV, raising=False)
    out = _apply_direct_bedrock_agent_mapping(
        {"AWS_REGION": "us-east-1"},
        agent="openhands",
        backend_model="us.anthropic.claude-opus-4-8",
        environment="daytona",
    )
    assert BEDROCK_THINKING_EFFORT_ENV not in out


def test_direct_bedrock_mapping_preserves_run_config_effort(monkeypatch):
    from benchflow.providers.runtime import _apply_direct_bedrock_agent_mapping

    monkeypatch.setenv(BEDROCK_THINKING_EFFORT_ENV, "max")
    out = _apply_direct_bedrock_agent_mapping(
        {BEDROCK_THINKING_EFFORT_ENV: "high"},
        agent="openhands",
        backend_model="us.anthropic.claude-opus-4-8",
        environment="daytona",
    )
    # an explicit value already in the agent env wins over the host override
    assert out[BEDROCK_THINKING_EFFORT_ENV] == "high"


# --- proxy <-> shim regex parity ---------------------------------------------
def test_proxy_and_shim_share_the_same_matcher():
    shim = (
        Path(benchflow.__file__).parent / "agents" / "oh_bedrock_opus_patch.py"
    ).read_text()
    m = re.search(r'_NEW = re\.compile\(r"([^"]+)"\)', shim)
    assert m is not None, "shim _NEW pattern not found"
    assert m.group(1) == BEDROCK_ADAPTIVE_THINKING_RE.pattern, (
        "the Daytona shim and host proxy model matchers have drifted"
    )
