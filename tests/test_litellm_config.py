from __future__ import annotations

import pytest

from benchflow.agents.env import resolve_agent_env, resolve_provider_env
from benchflow.providers.litellm_config import (
    litellm_proxy_config,
    resolve_litellm_route,
)


@pytest.mark.asyncio
async def test_gemini_vertex_passthrough_exchanges_gateway_auth_for_adc(monkeypatch):
    """Guards Vertex proxy auth against the regression in commit 28b82e33."""
    from unittest.mock import AsyncMock, Mock

    import litellm
    from litellm.proxy import proxy_server
    from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as vendor
    from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
        PassthroughEndpointRouter,
    )
    from starlette.requests import Request
    from starlette.responses import Response

    route = resolve_litellm_route(
        "google-vertex/gemini-3.1-flash-lite",
        {"GOOGLE_CLOUD_PROJECT": "dummy-project", "GOOGLE_CLOUD_LOCATION": "global"},
    )
    config = litellm_proxy_config(route, master_key="gateway-key")
    monkeypatch.setattr(
        vendor, "passthrough_endpoint_router", PassthroughEndpointRouter()
    )
    model_router = litellm.Router(model_list=config["model_list"])
    monkeypatch.setattr(proxy_server, "llm_router", model_router)
    # Native registration must leave other harnesses' translated routes usable.
    deployment = model_router.get_available_deployment(model=route.model_alias)
    assert deployment["litellm_params"]["model"] == route.upstream_model

    auth = AsyncMock()
    adc = AsyncMock(return_value=("mock-oauth-token", "dummy-project"))
    forward = Mock(return_value=AsyncMock())
    monkeypatch.setattr(vendor, "user_api_key_auth", auth)
    monkeypatch.setattr(vendor.VertexBase, "_ensure_access_token_async", adc)
    monkeypatch.setattr(vendor, "create_pass_through_route", forward)
    request = Request(
        {
            "type": "http",
            "headers": [
                (b"authorization", b"Bearer gateway-key"),
                (b"x-goog-api-key", b"gateway-key"),
            ],
        }
    )
    # Actual CLI 0.42.0 Express-style path, captured using a dummy gateway key.
    await vendor._base_vertex_proxy_route(
        endpoint="v1beta1/publishers/google/models/gemini-3.1-flash-lite:streamGenerateContent",
        request=request,
        fastapi_response=Response(),
        get_vertex_pass_through_handler=vendor.get_vertex_pass_through_handler(
            call_type="aiplatform"
        ),
    )
    auth.assert_awaited_once_with(request=request, api_key="Bearer gateway-key")
    adc.assert_awaited_once_with(
        credentials=None,
        project_id="dummy-project",
        custom_llm_provider="vertex_ai_beta",
    )
    sent = forward.call_args.kwargs
    assert sent["target"] == (
        "https://aiplatform.googleapis.com/v1beta1/projects/dummy-project/locations/global/"
        "publishers/google/models/gemini-3.1-flash-lite:streamGenerateContent?alt=sse"
    )
    assert sent["is_streaming_request"] is True
    # Use the forwarding layer's real merge rule, including incoming key headers.
    headers = vendor.HttpPassThroughEndpointHelpers.forward_headers_from_request(
        dict(request.headers),
        sent["custom_headers"],
        sent.get("_forward_headers", False),
    )
    assert headers["Authorization"] == "Bearer mock-oauth-token"
    assert "gateway-key" not in str(headers)


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


def test_vertex_route_passes_project_and_location_to_litellm():
    """Guards PR #985 Vertex project/location forwarding to LiteLLM."""
    route = resolve_litellm_route(
        "anthropic-vertex/claude-sonnet-4-6",
        {"GOOGLE_CLOUD_PROJECT": "skillsbench", "GOOGLE_CLOUD_LOCATION": "global"},
    )

    assert route.litellm_params["vertex_project"] == "skillsbench"
    assert route.litellm_params["vertex_location"] == "global"


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

    # deepseek routes via LiteLLM's NATIVE provider prefix (reasoning-param
    # passthrough) even behind an explicit proxy — the proxy override is
    # honored through api_base, which is what PR #780 actually guards.
    assert route.upstream_model == "deepseek/deepseek-v4-flash"
    assert route.litellm_params["api_base"] == "https://llm-proxy.example.test/v1"
    assert route.litellm_params["api_key"] == ("os.environ/BENCHFLOW_PROVIDER_API_KEY")
    assert route.required_env == ("BENCHFLOW_PROVIDER_API_KEY",)


@pytest.mark.parametrize(
    ("agent", "agent_base"),
    [
        ("claude-agent-acp", "https://api.z.ai/api/anthropic"),
        ("openclaw", "https://api.z.ai/api/coding/paas/v4"),
    ],
)
def test_zai_coding_clawsbench_routes(agent, agent_base):
    """Guards PR #1074: ClawsBench agents use each supported Z.AI surface."""
    env = resolve_agent_env(agent, "zai-coding/glm-5.3", {"ZAI_API_KEY": "native-key"})
    route = resolve_litellm_route("zai-coding/glm-5.3", env)

    assert env["BENCHFLOW_PROVIDER_BASE_URL"] == agent_base
    assert route.litellm_params["api_base"] == ("https://api.z.ai/api/coding/paas/v4")
    assert route.litellm_params["api_key"] == "os.environ/ZAI_API_KEY"
    assert route.required_env == ("ZAI_API_KEY",)
    assert route.upstream_model == "openai/glm-5.3"


@pytest.mark.parametrize("model", ["glm-5.4", "glm-5.4-flash"])
def test_zai_coding_registry_base_preserves_explicit_generic_key(model):
    """Guards PR #1074: mixed provenance must not retain Anthropic upstream URL."""
    env = {"BENCHFLOW_PROVIDER_API_KEY": "generic-key"}
    model_id = f"zai-coding/{model}"
    resolve_provider_env(env, model_id, "claude-agent-acp")
    route = resolve_litellm_route(model_id, env)

    assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.z.ai/api/anthropic"
    assert route.litellm_params["api_base"] == ("https://api.z.ai/api/coding/paas/v4")
    assert route.litellm_params["api_key"] == ("os.environ/BENCHFLOW_PROVIDER_API_KEY")
    assert route.required_env == ("BENCHFLOW_PROVIDER_API_KEY",)
    assert route.upstream_model == f"openai/{model}"


def test_zai_coding_preserves_explicit_proxy_route():
    """Guards PR #1074: explicit Z.AI-compatible proxies remain authoritative."""
    route = resolve_litellm_route(
        "zai-coding/glm-5.3-flash",
        {
            "BENCHFLOW_PROVIDER_BASE_URL": "https://proxy.example.test/v1",
            "BENCHFLOW_PROVIDER_API_KEY": "proxy-key",
        },
    )

    assert route.litellm_params["api_base"] == "https://proxy.example.test/v1"
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


@pytest.mark.parametrize(
    "model,expected",
    [
        (
            "google/gemini-3.1-flash-lite-preview",
            "gemini/gemini-3.1-flash-lite-preview",
        ),
        ("google/gemma-3-27b-it", "gemini/gemma-3-27b-it"),
        ("google/geminix-1", "gemini/google/geminix-1"),
    ],
)
def test_google_model_normalizes_only_gemini_families(model, expected):
    """Google Gemini/Gemma IDs get one LiteLLM provider prefix."""
    route = resolve_litellm_route(
        model,
        {"GEMINI_API_KEY": "key"},
    )
    assert route.upstream_model == expected
    assert route.litellm_params["model"] == expected
