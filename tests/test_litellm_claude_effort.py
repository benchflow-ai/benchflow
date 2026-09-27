"""Regression coverage: the requested effort reaches the provider for new Claude ids.

LiteLLM 1.91.0 removes ``output_config`` (the effort setting) when its model map
does not advertise effort support for the model, and without
``supports_adaptive_thinking`` it maps ``reasoning_effort`` to nothing. The
proxy uses LiteLLM's pinned backup map whenever it cannot fetch the upstream map
from GitHub, and that backup has no Claude id newer than Fable 5 / Opus 4.8, so
ids such as ``us.anthropic.claude-fable-5-1`` lost the effort on every proxied
call and fell back to the provider's default effort (see #1135). These tests pin the backup map, send requests through
BenchFlow's route and LiteLLM's real handlers, and capture the body LiteLLM
would send; no network calls are made.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import httpx
import pytest

import benchflow.providers.litellm_bedrock_patch as patch
import benchflow.providers.litellm_bedrock_preflight as preflight_mod
import benchflow.providers.litellm_runtime as runtime_mod
from benchflow.providers.litellm_config import (
    litellm_proxy_config,
    resolve_litellm_route,
)

_ENV = {
    "AWS_BEARER_TOKEN_BEDROCK": "test-bedrock-key",
    "AWS_REGION": "us-east-2",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "BENCHFLOW_BEDROCK_THINKING_EFFORT": "max",
}
_ANTHROPIC_RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
_BEDROCK_MODELS = (
    "aws-bedrock/us.anthropic.claude-fable-5-1",
    "aws-bedrock/global.anthropic.claude-fable-5-1",
    "aws-bedrock/us.anthropic.claude-opus-5-5",
    "aws-bedrock/us.anthropic.claude-sonnet-5",
)
_ANTHROPIC_MODELS = (
    "anthropic/claude-fable-5-1",
    "anthropic/claude-opus-5-5",
    "anthropic/claude-sonnet-5",
)


@pytest.fixture
def model_map(monkeypatch):
    """Pin LiteLLM's backup model map, the one a proxy without GitHub access uses.

    Registrations are isolated to the test; the process-wide map (possibly the
    upstream one fetched at import) is restored afterwards.
    """
    import litellm
    from litellm.constants import BEDROCK_CONVERSE_MODELS
    from litellm.litellm_core_utils.get_model_cost_map import (
        GetModelCostMap,
        _expand_model_aliases,
    )
    from litellm.utils import _invalidate_model_cost_lowercase_map

    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("AWS_REGION_NAME", _ENV["AWS_REGION"])
    backup = _expand_model_aliases(GetModelCostMap.load_local_model_cost_map())
    # Bedrock chat routing (Converse vs Invoke) is derived from the loaded map.
    monkeypatch.setattr(
        litellm,
        "bedrock_converse_models",
        set(BEDROCK_CONVERSE_MODELS)
        | {
            k
            for k, v in backup.items()
            if v.get("litellm_provider") == "bedrock_converse"
        },
    )
    original = litellm.model_cost
    litellm.model_cost = backup
    _invalidate_model_cost_lowercase_map()
    try:
        yield litellm
    finally:
        litellm.model_cost = original
        _invalidate_model_cost_lowercase_map()


async def _captured_body(monkeypatch, litellm, model: str, *, wire: str):
    """Start-up registration as the proxy does it, then one captured request."""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    route = resolve_litellm_route(model, dict(_ENV))
    config = litellm_proxy_config(route, master_key="sk-test")
    patch.register_claude_effort_capabilities(
        runtime_mod._config_upstream_models(config)
    )
    settings = config["litellm_settings"]
    assert isinstance(settings, dict)
    monkeypatch.setattr(litellm, "drop_params", settings["drop_params"])
    monkeypatch.setattr(
        litellm,
        "use_chat_completions_url_for_anthropic_messages",
        settings["use_chat_completions_url_for_anthropic_messages"],
    )

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_ANTHROPIC_RESPONSE)

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    params = {**route.litellm_params, "api_key": "test-key"}
    messages = [{"role": "user", "content": "Fix the failing test."}]
    if wire == "messages":
        # Claude Code's /v1/messages shape with ``--effort max``.
        await litellm.anthropic_messages(
            max_tokens=1024,
            messages=messages,
            thinking={"type": "adaptive", "display": "omitted"},
            output_config={"effort": "max"},
            client=client,
            stream=False,
            **params,
        )
    else:
        # A chat-completions agent (e.g. OpenHands). Bedrock routes carry the
        # effort in the route params; direct Anthropic gets it from the agent.
        params.setdefault("reasoning_effort", "max")
        await litellm.acompletion(
            messages=messages, max_tokens=1024, client=client, **params
        )
    assert len(captured) == 1
    return captured[0], json.loads(captured[0].content)


@pytest.mark.parametrize("wire", ["messages", "chat"])
@pytest.mark.parametrize("model", _BEDROCK_MODELS)
async def test_bedrock_request_carries_requested_effort(
    monkeypatch, model_map, model, wire
):
    """Guards the effort-registration fix against LiteLLM 1.91.0 dropping
    ``output_config`` for Claude ids missing from its model map (#1135 follow-up):
    the Bedrock InvokeModel body must carry adaptive thinking and effort ``max``."""
    request, sent = await _captured_body(monkeypatch, model_map, model, wire=wire)

    bare = model.split("/", 1)[1]
    assert request.url.path == f"/model/{bare}/invoke"
    assert sent["thinking"]["type"] == "adaptive"
    assert sent["output_config"] == {"effort": "max"}


@pytest.mark.parametrize("wire", ["messages", "chat"])
@pytest.mark.parametrize("model", _ANTHROPIC_MODELS)
async def test_anthropic_request_carries_requested_effort(
    monkeypatch, model_map, model, wire
):
    """Guards the effort-registration fix (#1135 follow-up): a chat-completions
    ``reasoning_effort=max`` for a new Claude id must become adaptive thinking
    plus ``output_config.effort=max`` on the Anthropic Messages wire, matching
    what Claude Code sends on /v1/messages."""
    request, sent = await _captured_body(monkeypatch, model_map, model, wire=wire)

    assert request.url.path == "/v1/messages"
    assert sent["thinking"]["type"] == "adaptive"
    assert sent["output_config"] == {"effort": "max"}


def test_registration_keeps_litellm_entries_and_skips_other_models(model_map):
    """Guards the effort-registration fix (#1135 follow-up) against overriding
    LiteLLM's own data: known ids and non-Claude models are left alone."""
    litellm = model_map
    fable5 = dict(litellm.model_cost["us.anthropic.claude-fable-5"])

    updated = patch.register_claude_effort_capabilities(
        [
            "bedrock/us.anthropic.claude-fable-5",
            "bedrock/us.anthropic.claude-opus-4-7",
            "openai/gpt-5.5",
            "bedrock/us.anthropic.claude-fable-5-1",
        ]
    )

    assert updated == ["us.anthropic.claude-fable-5-1", "anthropic.claude-fable-5-1"]
    assert litellm.model_cost["us.anthropic.claude-fable-5"] == fable5
    new_entry = litellm.model_cost["us.anthropic.claude-fable-5-1"]
    # Capabilities only: cost stays unknown instead of borrowing another price.
    assert not any("cost" in key for key in new_entry)
    assert (
        patch.register_claude_effort_capabilities(
            ["bedrock/us.anthropic.claude-fable-5-1"]
        )
        == []
    )


@pytest.mark.parametrize(
    ("model", "matches"),
    [
        ("us.anthropic.claude-fable-5-1", True),
        ("claude-opus-5-5", True),
        ("claude-sonnet-5", True),
        ("anthropic.claude-opus-5", True),
        ("us.anthropic.claude-opus-4-8", True),
        ("us.anthropic.claude-opus-4-7", False),
        ("anthropic.claude-opus-4-5-20251101-v1:0", False),
        ("claude-3-7-sonnet", False),
    ],
)
def test_adaptive_thinking_matcher_covers_claude_5_family(model, matches):
    assert bool(patch.BEDROCK_ADAPTIVE_THINKING_RE.search(model)) is matches


def test_bedrock_claude_5_routes_carry_the_requested_effort():
    """Guards the effort-registration fix (#1135 follow-up): Claude 5-family
    Bedrock routes honor ``BENCHFLOW_BEDROCK_THINKING_EFFORT`` like Fable 5."""
    for model in _BEDROCK_MODELS:
        route = resolve_litellm_route(model, dict(_ENV))
        assert route.litellm_params["reasoning_effort"] == "max", model
        assert preflight_mod.route_requires_bedrock_patch(route) is True


def test_sitecustomize_registers_the_route_models(tmp_path):
    route = resolve_litellm_route(_BEDROCK_MODELS[0], dict(_ENV))
    config = litellm_proxy_config(route, master_key="sk-test")

    runtime_mod._write_runtime_files(tmp_path, config=config)

    source = (tmp_path / "sitecustomize.py").read_text()
    assert (
        "benchflow_litellm_bedrock_patch.register_claude_effort_capabilities("
        "['bedrock/us.anthropic.claude-fable-5-1'])"
    ) in source


def _run_preflight(env: dict[str, str], *models: str):
    return subprocess.run(
        [sys.executable, "-c", preflight_mod.BEDROCK_PATCH_PREFLIGHT_SOURCE, *models],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_preflight_passes_when_sitecustomize_registers_effort(tmp_path):
    """Guards the effort-registration fix (#1135 follow-up): in a fresh
    interpreter the generated sitecustomize makes the route model keep
    ``output_config``, and the preflight proves it."""
    route = resolve_litellm_route(_BEDROCK_MODELS[0], dict(_ENV))
    runtime_mod._write_runtime_files(
        tmp_path, config=litellm_proxy_config(route, master_key="sk-test")
    )
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

    result = _run_preflight(env, route.upstream_model)

    assert result.returncode == 0, result.stdout + result.stderr


def test_preflight_fails_closed_when_effort_is_not_registered(tmp_path):
    """Guards the effort-registration fix (#1135 follow-up): a proxy whose
    sitecustomize registered nothing for the route model fails the preflight
    instead of silently running at the provider's default effort."""
    runtime_mod._write_runtime_files(tmp_path, config={"model_list": []})
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

    result = _run_preflight(env, "bedrock/us.anthropic.claude-fable-5-1")

    assert result.returncode != 0
    assert "effort capability missing" in result.stdout
