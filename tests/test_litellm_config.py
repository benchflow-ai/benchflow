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
    """Guards PR #739 against #737's route-config effort ceiling regression."""
    route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-opus-4-8",
        {
            "AWS_BEARER_TOKEN_BEDROCK": "token",
            "AWS_REGION": "us-west-2",
            "BENCHFLOW_BEDROCK_THINKING_EFFORT": "max",
        },
    )

    # `max` is honored as "the highest supported effort": LiteLLM 1.88.0rc1
    # rejects `max`/`xhigh` for opus-4-8, so BenchFlow clamps to the accepted
    # ceiling `high` rather than erroring at request time (#737).
    assert route.litellm_params["reasoning_effort"] == "high"


def test_azure_openai_route_uses_resource_and_preview_version():
    route = resolve_litellm_route(
        "azure-foundry-openai/gpt-5.5",
        {"AZURE_API_KEY": "key", "AZURE_RESOURCE": "benchflow"},
    )

    assert route.upstream_model == "azure/gpt-5.5"
    assert route.litellm_params["api_key"] == "os.environ/AZURE_API_KEY"
    assert route.litellm_params["api_base"] == "https://benchflow.openai.azure.com/"
    assert route.litellm_params["api_version"] == "preview"


def test_azure_openai_route_honors_openhands_reasoning_effort_env():
    """Guards PR #911 against proxy-alias capability guessing."""
    route = resolve_litellm_route(
        "azure-foundry-openai/gpt-5.6-sol",
        {
            "AZURE_API_KEY": "key",
            "AZURE_RESOURCE": "benchflow",
            "LLM_REASONING_EFFORT": "xhigh",
        },
    )

    assert route.litellm_params["reasoning_effort"] == "xhigh"


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


@pytest.mark.parametrize("model", ["gemini/gemini-2.5-flash", "gemini-2.5-flash"])
def test_gemini_native_route_honors_explicit_base_url(model):
    """Guards the fix from PR #881 for issue #672."""
    route = resolve_litellm_route(
        model,
        {
            "BENCHFLOW_PROVIDER_BASE_URL": "https://gemini-proxy.example.test/v1",
            "GEMINI_API_KEY": "sk-gemini",
        },
    )

    assert route.upstream_model == "gemini/gemini-2.5-flash"
    assert route.provider_name == "native"
    assert route.litellm_params["api_base"] == "https://gemini-proxy.example.test/v1"
    assert route.litellm_params["api_key"] == "os.environ/GEMINI_API_KEY"
    assert route.required_env == ("GEMINI_API_KEY",)


def test_gemini_native_route_honors_generic_proxy_key():
    """Guards the fix from PR #881 for issue #672."""
    route = resolve_litellm_route(
        "gemini/gemini-2.5-flash",
        {
            "BENCHFLOW_PROVIDER_BASE_URL": "https://gemini-proxy.example.test/v1",
            "BENCHFLOW_PROVIDER_API_KEY": "sk-proxy",
            "GEMINI_API_KEY": "sk-gemini",
        },
    )

    assert route.litellm_params["api_base"] == "https://gemini-proxy.example.test/v1"
    assert route.litellm_params["api_key"] == "os.environ/BENCHFLOW_PROVIDER_API_KEY"
    assert route.required_env == ("BENCHFLOW_PROVIDER_API_KEY",)


def test_gemini_native_route_without_explicit_base_url_is_unchanged():
    """Guards the fix from PR #881 for issue #672."""
    route = resolve_litellm_route(
        "gemini/gemini-2.5-flash",
        {"GEMINI_API_KEY": "sk-gemini"},
    )

    assert route.upstream_model == "gemini/gemini-2.5-flash"
    assert "api_base" not in route.litellm_params
    assert route.litellm_params["api_key"] == "os.environ/GEMINI_API_KEY"
    assert route.required_env == ("GEMINI_API_KEY",)


def test_openrouter_route_uses_openai_compatible_endpoint():
    route = resolve_litellm_route(
        "openrouter/qwen/qwen3.5-397b-a17b",
        {"OPENROUTER_API_KEY": "sk-openrouter"},
    )

    assert route.upstream_model == "openai/qwen/qwen3.5-397b-a17b"
    assert route.litellm_params["api_base"] == "https://openrouter.ai/api/v1"
    assert route.litellm_params["api_key"] == "os.environ/OPENROUTER_API_KEY"
    assert route.required_env == ("OPENROUTER_API_KEY",)


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
    assert config["router_settings"] == {
        "num_retries": 0,
        "disable_cooldowns": True,
    }


def test_proxy_config_forces_chat_completions_for_anthropic_messages():
    """Streaming /v1/messages must bridge via /chat/completions so the
    LiteLLM success callback fires and llm_trajectory.jsonl is written for
    claude-agent-acp. LiteLLM's Responses-API streaming adapter (used for
    openai/-prefixed upstreams such as the vllm provider) skips the success
    callback; this flag opts out of it (#833).
    """
    route = resolve_litellm_route("openai/gpt-5.4-mini", {"OPENAI_API_KEY": "key"})
    config = litellm_proxy_config(route, master_key="sk-local")

    assert (
        config["litellm_settings"]["use_chat_completions_url_for_anthropic_messages"]
        is True
    )


def test_litellm_exposes_anthropic_messages_chat_completions_flag():
    """Guard against a LiteLLM upgrade silently dropping the flag.

    litellm_proxy_config sets use_chat_completions_url_for_anthropic_messages
    via LiteLLM's generic litellm_settings -> setattr(litellm, ...) path, which
    does NOT raise on an unknown key. If a future LiteLLM renames or removes
    this attribute the fix would become a silent no-op and regress #833, so
    assert the attribute still exists.
    """
    import litellm

    assert hasattr(litellm, "use_chat_completions_url_for_anthropic_messages")


def test_proxy_config_registers_requested_bare_model_name():
    """Codex ACP sends the bare selected model name to the proxy."""
    route = resolve_litellm_route("openai/gpt-5.4-mini", {"OPENAI_API_KEY": "key"})
    config = litellm_proxy_config(route, master_key="sk-local")

    names = [entry["model_name"] for entry in config["model_list"]]
    assert "gpt-5.4-mini" in names
    assert "openai/gpt-5.4-mini" in names


def test_proxy_config_registers_responses_bridge_for_openai_upstream():
    """A responses-only CLI (codex) hits /v1/responses; a chat-only OpenAI-
    compatible backend has no /responses endpoint, so the proxy must expose a
    bridge deployment named ``<model>-responses-bridge`` whose upstream carries
    the ``openai/chat_completions/`` prefix (LiteLLM strips it and bridges
    responses→chat). The bridge name is non-slashed (codex mis-parses a slashed
    model id and then sends no request)."""
    route = resolve_litellm_route("openai/gpt-5.4-mini", {"OPENAI_API_KEY": "key"})
    config = litellm_proxy_config(route, master_key="sk-local")

    by_name = {e["model_name"]: e for e in config["model_list"]}
    bridge_name = "gpt-5.4-mini-responses-bridge"
    assert bridge_name in by_name
    assert f"{route.model_alias}-responses-bridge" in by_name
    # the bridge entry's UPSTREAM carries the chat-completions prefix
    assert (
        by_name[bridge_name]["litellm_params"]["model"]
        == "openai/chat_completions/gpt-5.4-mini"
    )
    # never slashed (would break codex's model parsing)
    assert "/" not in bridge_name


def test_proxy_config_no_responses_bridge_for_non_openai_upstream():
    """The bridge is openai/-upstream only; native-responses/anthropic/bedrock
    providers are untouched."""
    route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-opus-4-8",
        {"AWS_BEARER_TOKEN_BEDROCK": "token", "AWS_REGION": "us-west-2"},
    )
    config = litellm_proxy_config(route, master_key="sk-local")
    names = [entry["model_name"] for entry in config["model_list"]]
    assert not any(n.endswith("-responses-bridge") for n in names)
