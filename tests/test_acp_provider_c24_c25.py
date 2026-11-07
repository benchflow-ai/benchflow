"""Guards C24/C25 on integ 85b87bc9: admit an entire route.

Dummy endpoints and credentials only. No provider calls or model hooks.
"""

import json
from pathlib import Path

import pytest

from benchflow.providers import litellm_runtime as runtime
from benchflow.providers import restricted_models as rm
from benchflow.providers.litellm_config import resolve_litellm_route
from tests.test_litellm_runtime import FakeLiteLLMServer

# Every listed router/gateway entry, including chat-mode Switchpoint, is
# outside the first-party provider set regardless of its spelling or mode.
GATEWAY_MODELS = sorted(
    key
    for key, info in rm.pinned_model_metadata().items()
    if isinstance(info, dict)
    and info.get("litellm_provider") in {"openrouter", "vercel_ai_gateway"}
    and info.get("mode") in {"chat", "responses", "completion"}
)


@pytest.mark.parametrize("model", GATEWAY_MODELS)
def test_c24_gateway_metadata_never_auto_admits(model):
    with pytest.raises(rm.RestrictedModelRefused):
        rm.admit_restricted_model(model, environ={})


BAD_ROUTES = [
    ("openrouter/switchpoint/router", {"OPENROUTER_API_KEY": "dummy"}),
    (
        "litellm/gpt-4o",
        {"LITELLM_BASE_URL": "https://gateway.invalid/v1", "LITELLM_API_KEY": "dummy"},
    ),
    (
        "vllm/gpt-4o",
        {
            "BENCHFLOW_PROVIDER_BASE_URL": "https://gateway.invalid/v1",
            "BENCHFLOW_PROVIDER_API_KEY": "dummy",
        },
    ),
    (
        "sglang/gpt-4o",
        {
            "BENCHFLOW_PROVIDER_BASE_URL": "https://gateway.invalid/v1",
            "BENCHFLOW_PROVIDER_API_KEY": "dummy",
        },
    ),
    (
        "openai/gpt-4o",
        {
            "OPENAI_API_KEY": "dummy",
            "BENCHFLOW_PROVIDER_BASE_URL": "https://gateway.invalid/v1",
        },
    ),
    (
        "anthropic/claude-sonnet-4-5",
        {"ANTHROPIC_API_KEY": "dummy", "ANTHROPIC_API_BASE": "https://gateway.invalid"},
    ),
    (
        "gemini/gemini-2.5-flash",
        {
            "GEMINI_API_KEY": "dummy",
            "BENCHFLOW_PROVIDER_BASE_URL": "https://gateway.invalid",
        },
    ),
    (
        "aws-bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0",
        {
            "AWS_BEARER_TOKEN_BEDROCK": "dummy",
            "AWS_REGION": "us-east-1",
            "AWS_BEDROCK_RUNTIME_ENDPOINT": "https://gateway.invalid",
        },
    ),
    (
        "google-vertex/gemini-2.5-flash",
        {
            "GOOGLE_CLOUD_PROJECT": "dummy",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
            "VERTEXAI_API_BASE": "https://gateway.invalid",
        },
    ),
    (
        "azure-foundry-openai/gpt-4o",
        {"AZURE_API_KEY": "dummy", "AZURE_RESOURCE": "dummy"},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", ["local", "daytona"])
@pytest.mark.parametrize("model,env", BAD_ROUTES)
async def test_c24_c25_launch_refuses_the_actual_deployment(
    monkeypatch, environment, model, env
):
    started = []

    async def start(**kwargs):
        started.append(kwargs)
        return FakeLiteLLMServer("http://dummy", kwargs["route"])

    monkeypatch.setattr(runtime, "_start_host_litellm", start)
    monkeypatch.setattr(runtime, "_start_sandbox_litellm", start)
    monkeypatch.delenv(rm.OPERATOR_ENV, raising=False)
    with pytest.raises(RuntimeError, match=rm.OPERATOR_ENV):
        await runtime.ensure_litellm_runtime(
            agent="codex-acp",
            model=model,
            agent_env=env,
            runtime=None,
            environment=environment,
            sandbox=object(),
            disallow_web_tools=True,
        )
    assert not started


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", ["local", "daytona"])
@pytest.mark.parametrize("model,env", BAD_ROUTES)
async def test_operator_entry_is_required_and_recorded(
    monkeypatch, environment, model, env
):
    async def start(**kwargs):
        return FakeLiteLLMServer("http://dummy", kwargs["route"])

    monkeypatch.setattr(runtime, "_start_host_litellm", start)
    monkeypatch.setattr(runtime, "_start_sandbox_litellm", start)
    route = resolve_litellm_route(model, env)
    monkeypatch.setenv(rm.OPERATOR_ENV, route.upstream_model)
    _, current = await runtime.ensure_litellm_runtime(
        agent="codex-acp",
        model=model,
        agent_env=env,
        runtime=None,
        environment=environment,
        sandbox=object(),
        disallow_web_tools=True,
    )
    assert current is not None
    assert current.restricted_web_model["admitted_by"] == "operator"
    assert current.restricted_web_model["model"] == route.upstream_model
    expected = {"aws-bedrock": "bedrock", "google-vertex": "vertex_ai"}.get(
        route.provider_name, route.provider_name
    )
    if expected == "native":
        expected = route.upstream_model.split("/", 1)[0]
    assert current.restricted_web_model["provider"] == expected


def test_snapshot_is_generated_from_the_pin():
    from tools.restricted_model_allowlist import DEST, snapshot

    assert json.loads(DEST.read_text()) == snapshot()
    lock = (Path(__file__).resolve().parents[1] / "uv.lock").read_text()
    assert f'name = "litellm"\nversion = "{snapshot()["litellm_version"]}"' in lock


def test_every_admitted_entry_has_provider_endpoint_and_owned_chat_metadata():
    from tools.restricted_model_allowlist import MODES, OWNERS

    metadata = rm.pinned_model_metadata()
    entries = rm.first_party_models()
    assert len(entries) > 300
    for (provider, endpoint, base), key in entries.items():
        info = metadata[key]
        assert OWNERS[info["litellm_provider"]] == provider
        assert info["mode"] in MODES
        assert endpoint in rm.OFFICIAL_ENDPOINTS[provider]
        for suffix in (
            ("", ":nitro")
            if provider in {"openai", "anthropic", "deepseek", "zai"}
            else ("",)
        ):
            admission = rm.admit_restricted_model(
                f"{provider}/{base}{suffix}",
                api_base=endpoint if endpoint.startswith("https://") else None,
                environ={},
            )
            assert admission.admitted_by == "metadata"
            assert admission.base == base
            assert admission.provider == provider
            assert admission.endpoint == endpoint
        for alias in (
            "litellm",
            "openrouter",
            "vercel_ai_gateway",
            "vllm",
            "sglang",
            "azure",
            "private",
        ):
            with pytest.raises(rm.RestrictedModelRefused):
                rm.admit_restricted_model(
                    f"{provider}/{base}", provider_name=alias, environ={}
                )
        for custom in (
            "https://gateway.invalid/v1",
            "https://api.openai.com.evil.invalid/v1",
            "http://api.openai.com/v1",
            "https://api.openai.com/v1/alias",
            "https://api.openai.com/v1?deployment=search",
            "bedrock-sdk-default",
            "vertex-sdk-default",
        ):
            with pytest.raises(rm.RestrictedModelRefused):
                rm.admit_restricted_model(
                    f"{provider}/{base}", api_base=custom, environ={}
                )


HONEST = [
    ("openai/gpt-4o", {"OPENAI_API_KEY": "dummy"}),
    ("us-openai/gpt-4.1", {"OPENAI_API_KEY": "dummy"}),
    ("anthropic/claude-sonnet-4-5", {"ANTHROPIC_API_KEY": "dummy"}),
    ("gemini/gemini-2.5-flash", {"GEMINI_API_KEY": "dummy"}),
    (
        "google-vertex/gemini-2.5-flash",
        {"GOOGLE_CLOUD_PROJECT": "dummy", "GOOGLE_CLOUD_LOCATION": "us-central1"},
    ),
    (
        "anthropic-vertex/claude-sonnet-4-5@20250929",
        {"GOOGLE_CLOUD_PROJECT": "dummy", "GOOGLE_CLOUD_LOCATION": "us-east5"},
    ),
    (
        "aws-bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0",
        {"AWS_BEARER_TOKEN_BEDROCK": "dummy", "AWS_REGION": "us-east-1"},
    ),
    (
        "aws-bedrock/invoke/amazon.nova-pro-v1:0",
        {"AWS_BEARER_TOKEN_BEDROCK": "dummy", "AWS_REGION": "us-east-1"},
    ),
    (
        "aws-bedrock/converse/meta.llama3-1-70b-instruct-v1:0",
        {"AWS_BEARER_TOKEN_BEDROCK": "dummy", "AWS_REGION": "us-east-1"},
    ),
    ("deepseek/deepseek-chat", {"DEEPSEEK_API_KEY": "dummy"}),
    ("zai/glm-4.6", {"ZAI_API_KEY": "dummy"}),
    (
        "openai/ft:gpt-4o-mini-2024-07-18:example:bench:abc123",
        {"OPENAI_API_KEY": "dummy"},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", ["local", "daytona"])
@pytest.mark.parametrize("model,env", HONEST)
async def test_honest_models_on_all_protocols_keep_their_dispatch(
    monkeypatch, environment, model, env
):
    captured = []

    async def start(**kwargs):
        captured.append(kwargs)
        return FakeLiteLLMServer("http://dummy", kwargs["route"])

    monkeypatch.setattr(runtime, "_start_host_litellm", start)
    monkeypatch.setattr(runtime, "_start_sandbox_litellm", start)
    monkeypatch.delenv(rm.OPERATOR_ENV, raising=False)
    _, current = await runtime.ensure_litellm_runtime(
        agent="codex-acp",
        model=model,
        agent_env=env,
        runtime=None,
        environment=environment,
        sandbox=object(),
        disallow_web_tools=True,
    )
    assert current is not None
    assert current.restricted_web_model["admitted_by"] == "metadata"
    assert (
        captured[0]["route"].upstream_model
        == resolve_litellm_route(model, env).upstream_model
    )
    assert current.restricted_web_model["endpoint"]
    if current.restricted_web_model["endpoint"].startswith("https://"):
        assert (
            captured[0]["route"].litellm_params["api_base"]
            == current.restricted_web_model["endpoint"]
        )
    else:
        for name in rm.ENDPOINT_ENV[current.restricted_web_model["provider"]]:
            assert captured[0]["agent_env"][name] == ""


@pytest.mark.parametrize(
    "provider", ["openai", "anthropic", "gemini", "vertex_ai", "bedrock"]
)
def test_every_endpoint_fallback_needs_operator_admission(monkeypatch, provider):
    from dataclasses import replace

    model, env = next(
        (m, e)
        for m, e in HONEST
        if rm._PROVIDER_ALIASES.get(
            resolve_litellm_route(m, e).provider_name,
            resolve_litellm_route(m, e).upstream_model.split("/", 1)[0],
        )
        == provider
    )
    route = resolve_litellm_route(model, env)
    # Deployment api_base has precedence. Without it, enumerate every fallback
    # from each process source: role env and inherited controller env.
    route = replace(
        route,
        litellm_params={
            k: v for k, v in route.litellm_params.items() if k != "api_base"
        },
    )
    monkeypatch.delenv(rm.OPERATOR_ENV, raising=False)
    for name in rm.ENDPOINT_ENV[provider]:
        for inherited in (False, True):
            with monkeypatch.context() as patch:
                extra = {name: "https://gateway.invalid/alias"}
                if inherited:
                    patch.setenv(name, extra.pop(name))
                with pytest.raises(rm.RestrictedModelRefused, match=rm.OPERATOR_ENV):
                    rm.admit_restricted_route(route, {**env, **extra})


@pytest.mark.parametrize(
    "mode",
    ["router", "embedding", "completion", "image_generation", "chat", "responses"],
)
@pytest.mark.parametrize(
    "owner", ["openai", "openrouter", "vercel_ai_gateway", "private"]
)
def test_provider_and_mode_select_the_allowlist(monkeypatch, mode, owner):
    from tools import restricted_model_allowlist as generator

    monkeypatch.setattr(
        generator,
        "pinned_model_metadata",
        lambda: {"gpt-4o": {"litellm_provider": owner, "mode": mode}},
    )
    assert bool(generator.snapshot()["entries"]) == (
        owner == "openai" and mode in {"chat", "responses"}
    )


def test_new_metadata_cannot_extend_the_runtime_allowlist(monkeypatch):
    # An added alias cannot acquire admission by changing the live metadata
    # map. A reviewed regeneration of the frozen snapshot is required.
    monkeypatch.setattr(
        rm,
        "pinned_model_metadata",
        lambda: {"gpt-new-alias": {"litellm_provider": "openai", "mode": "chat"}},
    )
    with pytest.raises(rm.RestrictedModelRefused):
        rm.admit_restricted_model("openai/gpt-new-alias", environ={})


@pytest.mark.parametrize(
    "model",
    [
        "vertex_ai/publishers/private/models/gemini-2.5-flash",
        "vertex_ai/publishers/google/models/claude-sonnet-4-5@20250929",
        "vertex_ai/publishers/anthropic/models/gemini-2.5-flash",
        "bedrock/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/gpt-4o",
        "openai/container",
    ],
)
def test_known_model_names_do_not_identify_opaque_resources(model):
    with pytest.raises(rm.RestrictedModelRefused):
        rm.admit_restricted_model(model, environ={})
