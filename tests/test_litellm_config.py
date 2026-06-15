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


def test_registered_provider_route_honors_explicit_generic_proxy_env():
    """Guards PR #780: external LiteLLM proxies can back registered providers."""
    route = resolve_litellm_route(
        "deepseek/deepseek-v4-flash",
        {
            "BENCHFLOW_PROVIDER_BASE_URL": "https://llm-proxy.example.test/v1",
            "BENCHFLOW_PROVIDER_API_KEY": "sk-proxy",
        },
    )

    assert route.upstream_model == "openai/deepseek-v4-flash"
    assert route.litellm_params["api_base"] == "https://llm-proxy.example.test/v1"
    assert route.litellm_params["api_key"] == ("os.environ/BENCHFLOW_PROVIDER_API_KEY")
    assert route.required_env == ("BENCHFLOW_PROVIDER_API_KEY",)


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
    assert "us.anthropic.claude-opus-4-8" in names
    assert "openai/us.anthropic.claude-opus-4-8" in names
    assert config["litellm_settings"]["callbacks"] == [
        "benchflow_litellm_callback.proxy_handler_instance"
    ]


def test_proxy_config_registers_requested_bare_model_name():
    """Codex ACP sends the bare selected model name to the proxy."""
    route = resolve_litellm_route("openai/gpt-5.4-mini", {"OPENAI_API_KEY": "key"})
    config = litellm_proxy_config(route, master_key="sk-local")

    names = [entry["model_name"] for entry in config["model_list"]]
    assert "gpt-5.4-mini" in names
    assert "openai/gpt-5.4-mini" in names
