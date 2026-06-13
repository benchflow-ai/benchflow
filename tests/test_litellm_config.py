from __future__ import annotations

import pytest

from benchflow.providers.litellm_config import (
    litellm_proxy_config,
    resolve_litellm_route,
)


def test_bedrock_model_maps_to_litellm_bedrock_route():
    route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-opus-4-8",
        {"AWS_BEARER_TOKEN_BEDROCK": "token", "AWS_REGION": "us-west-2"},
    )

    assert route.model_alias == "benchflow-aws-bedrock-us.anthropic.claude-opus-4-8"
    assert route.upstream_model == "bedrock/us.anthropic.claude-opus-4-8"
    assert route.required_env == ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION")
    assert route.litellm_params["reasoning_effort"] == "high"


def test_bedrock_model_honors_max_thinking_effort_env():
    route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-opus-4-8",
        {
            "AWS_BEARER_TOKEN_BEDROCK": "token",
            "AWS_REGION": "us-west-2",
            "BENCHFLOW_BEDROCK_THINKING_EFFORT": "max",
        },
    )

    assert route.litellm_params["reasoning_effort"] == "max"


def test_azure_openai_route_uses_resource_and_preview_version():
    route = resolve_litellm_route(
        "azure-foundry-openai/gpt-5.5",
        {"AZURE_API_KEY": "key", "AZURE_RESOURCE": "benchflow"},
    )

    assert route.upstream_model == "azure/gpt-5.5"
    assert route.litellm_params["api_key"] == "os.environ/AZURE_API_KEY"
    assert route.litellm_params["api_base"] == "https://benchflow.openai.azure.com/"
    assert route.litellm_params["api_version"] == "preview"


def test_azure_anthropic_route_uses_azure_ai_anthropic_surface():
    route = resolve_litellm_route(
        "azure-foundry-anthropic/claude-opus-4-5",
        {"AZURE_API_KEY": "key", "AZURE_RESOURCE": "benchflow"},
    )

    assert route.upstream_model == "azure_ai/claude-opus-4-5"
    assert (
        route.litellm_params["api_base"]
        == "https://benchflow.services.ai.azure.com/anthropic"
    )


@pytest.mark.parametrize(
    ("model", "upstream", "required_env"),
    [
        ("openai/gpt-4.1-mini", "openai/gpt-4.1-mini", ("OPENAI_API_KEY",)),
        ("claude-sonnet-4-6", "anthropic/claude-sonnet-4-6", ("ANTHROPIC_API_KEY",)),
        ("gemini-3.5-flash", "gemini/gemini-3.5-flash", ("GEMINI_API_KEY",)),
        ("minimax/MiniMax-M3", "openai/MiniMax-M3", ("MINIMAX_API_KEY",)),
    ],
)
def test_common_provider_routes(model, upstream, required_env):
    route = resolve_litellm_route(
        model,
        {
            "OPENAI_API_KEY": "openai",
            "ANTHROPIC_API_KEY": "anthropic",
            "GEMINI_API_KEY": "gemini",
            "MINIMAX_API_KEY": "minimax",
            "MINIMAX_BASE_URL": "https://api.minimax.io/v1",
        },
    )

    assert route.upstream_model == upstream
    assert route.required_env == required_env


def test_native_gemini_route_honors_provider_base_url():
    """A native ``gemini/`` model with BENCHFLOW_PROVIDER_BASE_URL set routes
    through the gateway with api_base = that URL, keeping the ``gemini/``
    upstream so LiteLLM speaks GenerateContent against the custom host
    (parity with the vllm base-URL override)."""
    base = "https://gateway.example.test/v1beta"
    route = resolve_litellm_route(
        "gemini/gemini-3.1-flash-lite",
        {"GEMINI_API_KEY": "key", "BENCHFLOW_PROVIDER_BASE_URL": base},
    )

    assert route.provider_name == "native"
    assert route.upstream_model == "gemini/gemini-3.1-flash-lite"
    assert route.litellm_params["api_base"] == base
    assert route.litellm_params["api_key"] == "os.environ/GEMINI_API_KEY"
    assert route.required_env == ("GEMINI_API_KEY",)


def test_native_bare_gemini_route_honors_provider_base_url():
    """A bare gemini model id (no prefix) also picks up the base URL and keeps
    the ``gemini/`` GenerateContent upstream."""
    base = "https://gateway.example.test/v1beta"
    route = resolve_litellm_route(
        "gemini-2.5-flash",
        {"GEMINI_API_KEY": "key", "BENCHFLOW_PROVIDER_BASE_URL": base},
    )

    assert route.upstream_model == "gemini/gemini-2.5-flash"
    assert route.litellm_params["api_base"] == base


def test_native_gemini_route_without_base_url_is_unchanged():
    """Without BENCHFLOW_PROVIDER_BASE_URL the gemini route sets no api_base —
    the baseline behavior must be preserved exactly."""
    route = resolve_litellm_route(
        "gemini/gemini-2.5-flash",
        {"GEMINI_API_KEY": "key"},
    )

    assert route.provider_name == "native"
    assert route.upstream_model == "gemini/gemini-2.5-flash"
    assert "api_base" not in route.litellm_params
    assert route.litellm_params == {
        "model": "gemini/gemini-2.5-flash",
        "api_key": "os.environ/GEMINI_API_KEY",
    }


def test_native_non_gemini_route_ignores_provider_base_url():
    """The gemini base-URL override must not leak into non-gemini native routes
    (e.g. a bare anthropic claude id)."""
    route = resolve_litellm_route(
        "claude-sonnet-4-6",
        {"ANTHROPIC_API_KEY": "key", "BENCHFLOW_PROVIDER_BASE_URL": "https://x.test"},
    )

    assert route.upstream_model == "anthropic/claude-sonnet-4-6"
    assert "api_base" not in route.litellm_params


def test_google_ai_studio_provider_routes_openai_compatible_with_base_url():
    """The google-ai-studio provider targets an OpenAI-compatible Gemini proxy:
    ``openai/`` upstream + api_base + GEMINI_API_KEY auth (vllm parity)."""
    base = "https://gateway.example.test/v1beta/openai"
    route = resolve_litellm_route(
        "google-ai-studio/gemini-3.1-flash-lite",
        {"GEMINI_API_KEY": "key", "BENCHFLOW_PROVIDER_BASE_URL": base},
    )

    assert route.provider_name == "google-ai-studio"
    assert route.upstream_model == "openai/gemini-3.1-flash-lite"
    assert route.litellm_params["api_base"] == base
    assert route.litellm_params["api_key"] == "os.environ/GEMINI_API_KEY"
    assert route.required_env == ("GEMINI_API_KEY",)


def test_google_ai_studio_matches_vllm_base_url_shape():
    """google-ai-studio and vllm produce the same param shape (openai/ upstream
    + api_base) for a runtime-supplied base URL — they differ only in auth env."""
    base = "https://gateway.example.test/v1"
    gemini = resolve_litellm_route(
        "google-ai-studio/m",
        {"GEMINI_API_KEY": "k", "BENCHFLOW_PROVIDER_BASE_URL": base},
    )
    vllm = resolve_litellm_route(
        "vllm/m",
        {"OPENAI_API_KEY": "k", "BENCHFLOW_PROVIDER_BASE_URL": base},
    )

    assert gemini.litellm_params["api_base"] == vllm.litellm_params["api_base"]
    assert gemini.litellm_params["model"] == vllm.litellm_params["model"]
    assert gemini.litellm_params["api_key"] == "os.environ/GEMINI_API_KEY"
    assert vllm.litellm_params["api_key"] == "os.environ/OPENAI_API_KEY"


def test_proxy_config_registers_plain_and_openai_aliases():
    route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-opus-4-8",
        {"AWS_BEARER_TOKEN_BEDROCK": "token", "AWS_REGION": "us-west-2"},
    )
    config = litellm_proxy_config(route, master_key="sk-local")

    assert config["general_settings"] == {"master_key": "sk-local"}
    names = [entry["model_name"] for entry in config["model_list"]]
    assert route.model_alias in names
    assert f"openai/{route.model_alias}" in names
    assert config["litellm_settings"]["callbacks"] == [
        "benchflow_litellm_callback.proxy_handler_instance"
    ]
