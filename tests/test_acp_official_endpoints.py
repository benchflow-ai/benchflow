"""Guards FF-ACP-CONF-1/2/3, endpoint regressions introduced by 5e4204fb.

Launch through both proxy placements, then drive pinned LiteLLM's URL builders
with the admitted deployment. All HTTP sends stop at a recorder; only dummy
credentials and cloud components are used.
"""

from contextlib import suppress
from dataclasses import replace
from typing import Any, cast

import httpx
import pytest

from benchflow.providers import litellm_runtime as runtime
from benchflow.providers import restricted_models as rm
from benchflow.providers.litellm_config import litellm_proxy_config
from benchflow.providers.restricted_endpoints import ENDPOINT_TABLE
from tests.test_litellm_runtime import FakeLiteLLMServer

# Independent expected URLs, rather than deriving them from the policy table.
# (model, environment, configured forms, translation URL)
CASES = [
    (
        "openai/gpt-4o",
        {"OPENAI_API_KEY": "dummy", "BASE_ENV": "BENCHFLOW_PROVIDER_BASE_URL"},
        ("https://api.openai.com", "https://api.openai.com/v1"),
        "https://api.openai.com/v1/chat/completions",
    ),
    (
        "us-openai/gpt-4o",
        {"OPENAI_API_KEY": "dummy", "BASE_ENV": "BENCHFLOW_PROVIDER_BASE_URL"},
        ("https://us.api.openai.com", "https://us.api.openai.com/v1"),
        "https://us.api.openai.com/v1/chat/completions",
    ),
    (
        "anthropic/claude-sonnet-4-5",
        {"ANTHROPIC_API_KEY": "dummy", "BASE_ENV": "ANTHROPIC_API_BASE"},
        (
            "https://api.anthropic.com",
            "https://api.anthropic.com/v1",
            "https://api.anthropic.com/v1/messages",
        ),
        "https://api.anthropic.com/v1/messages",
    ),
    (
        "gemini/gemini-2.5-flash",
        {"GEMINI_API_KEY": "dummy", "BASE_ENV": "BENCHFLOW_PROVIDER_BASE_URL"},
        (
            "https://generativelanguage.googleapis.com",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
    ),
    (
        "gemini/gemini-2.5-flash",
        {"GEMINI_API_KEY": "dummy", "BASE_ENV": "BENCHFLOW_PROVIDER_BASE_URL"},
        ("https://generativelanguage.googleapis.com/v1",),
        "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent",
    ),
    (
        "deepseek/deepseek-chat",
        {"DEEPSEEK_API_KEY": "dummy", "BASE_ENV": "DEEPSEEK_BASE_URL"},
        ("https://api.deepseek.com", "https://api.deepseek.com/v1"),
        "https://api.deepseek.com/v1/chat/completions",
    ),
    (
        "deepseek/deepseek-chat",
        {"DEEPSEEK_API_KEY": "dummy", "BASE_ENV": "DEEPSEEK_BASE_URL"},
        ("https://api.deepseek.com/beta",),
        "https://api.deepseek.com/beta/chat/completions",
    ),
    (
        "zai/glm-4.6",
        {"ZAI_API_KEY": "dummy", "BASE_ENV": "BENCHFLOW_PROVIDER_BASE_URL"},
        ("https://api.z.ai", "https://api.z.ai/api/paas/v4"),
        "https://api.z.ai/api/paas/v4/chat/completions",
    ),
    (
        "zai-coding/glm-4.6",
        {"ZAI_API_KEY": "dummy", "BASE_ENV": "BENCHFLOW_PROVIDER_BASE_URL"},
        ("https://api.z.ai/api/coding/paas/v4",),
        "https://api.z.ai/api/coding/paas/v4/chat/completions",
    ),
    (
        "google-vertex/gemini-2.5-flash",
        {
            "GOOGLE_CLOUD_PROJECT": "dummy-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
        },
        (None,),
        "https://us-central1-aiplatform.googleapis.com/v1/projects/dummy-project/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent",
    ),
    (
        "anthropic-vertex/claude-sonnet-4-5@20250929",
        {"GOOGLE_CLOUD_PROJECT": "dummy-project", "GOOGLE_CLOUD_LOCATION": "us-east5"},
        (None,),
        "https://us-east5-aiplatform.googleapis.com/v1/projects/dummy-project/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict",
    ),
    (
        "aws-bedrock/invoke/amazon.nova-pro-v1:0",
        {"AWS_BEARER_TOKEN_BEDROCK": "dummy", "AWS_REGION": "us-east-1"},
        (None,),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-pro-v1:0/invoke",
    ),
    (
        "aws-bedrock/converse/meta.llama3-1-70b-instruct-v1:0",
        {"AWS_BEARER_TOKEN_BEDROCK": "dummy", "AWS_REGION": "us-east-1"},
        (None,),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/meta.llama3-1-70b-instruct-v1%3A0/converse",
    ),
]
# Independent inventory from the official data-residency and mTLS docs:
# https://developers.openai.com/api/docs/guides/your-data
# https://help.openai.com/en/articles/10876024-openai-mutual-tls-beta-program
# Explicit hosts, never a wildcard over an apparent provider-owned suffix.
CASES += [
    (
        "openai/gpt-4o",
        {"OPENAI_API_KEY": "dummy", "BASE_ENV": "BENCHFLOW_PROVIDER_BASE_URL"},
        (f"https://{host}", f"https://{host}/v1"),
        f"https://{host}/v1/chat/completions",
    )
    for host in (
        "eu.api.openai.com",
        "au.api.openai.com",
        "ca.api.openai.com",
        "jp.api.openai.com",
        "in.api.openai.com",
        "sg.api.openai.com",
        "kr.api.openai.com",
        "gb.api.openai.com",
        "ae.api.openai.com",
        "mtls.api.openai.com",
        "mtls-eu.api.openai.com",
    )
]
# Keep the original independent URL controls above. Admission and the complete
# matrix below share the official inventory, so adding a form cannot leave it
# unexercised. Expected chat URLs are explicit table facts, not computed from
# the dispatch base: a bad normalization still fails at the pinned translator.
SAMPLES = {
    "openai": CASES[0][:2],
    "anthropic": CASES[2][:2],
    "gemini": (
        CASES[3][0],
        {"GEMINI_API_KEY": "dummy", "BASE_ENV": "GEMINI_API_BASE"},
    ),
    "deepseek": CASES[5][:2],
    "zai": CASES[7][:2],
}
FORMS = [(model, env, form, url) for model, env, forms, url in CASES for form in forms]
FORMS += [
    (
        SAMPLES[provider][0],
        SAMPLES[provider][1],
        form.format(model=SAMPLES[provider][0].split("/", 1)[1]),
        row.chat_url.format(model=SAMPLES[provider][0].split("/", 1)[1]),
    )
    for provider, rows in ENDPOINT_TABLE.items()
    if provider in SAMPLES
    for row in rows
    for form in row.forms
]


@pytest.fixture
def launcher(monkeypatch):
    """Capture the same sealed deployment consumed by either real starter."""
    captured = []

    async def start(**kwargs):
        captured.append(kwargs)
        return FakeLiteLLMServer("http://dummy", kwargs["route"])

    monkeypatch.setattr(runtime, "_start_host_litellm", start)
    monkeypatch.setattr(runtime, "_start_sandbox_litellm", start)
    monkeypatch.delenv(rm.OPERATOR_ENV, raising=False)

    async def launch(model, env, placement):
        _, current = await runtime.ensure_litellm_runtime(
            agent="openhands",
            model=model,
            agent_env=env,
            runtime=None,
            environment=placement,
            sandbox=object(),
            disallow_web_tools=True,
        )
        assert current is not None
        assert current.restricted_web_model["admitted_by"] == "metadata"
        return captured[-1]["route"], captured[-1]["agent_env"]

    return launch


@pytest.fixture
def recorder(monkeypatch):
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

    sent = []

    async def send(self, request, **kwargs):
        sent.append(str(request.url))
        return httpx.Response(
            400, json={"error": {"message": "dummy recorder"}}, request=request
        )

    def sync_send(self, request, **kwargs):
        sent.append(str(request.url))
        return httpx.Response(
            400, json={"error": {"message": "dummy recorder"}}, request=request
        )

    async def token(*args, **kwargs):
        return "dummy", "dummy-project"

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(httpx.Client, "send", sync_send)
    monkeypatch.setattr(VertexBase, "_ensure_access_token_async", token)
    monkeypatch.setattr(
        VertexBase, "_ensure_access_token", lambda *a, **kw: ("dummy", "dummy-project")
    )
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    return sent


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["local", "daytona"])
@pytest.mark.parametrize("slash", ["", "/", "//"])
@pytest.mark.parametrize("endpoint_setting", ["provider", "benchflow", "deployment"])
@pytest.mark.parametrize("model,env,form,expected", FORMS)
async def test_official_forms_generate_correct_upstream_urls(
    launcher,
    recorder,
    monkeypatch,
    placement,
    slash,
    endpoint_setting,
    model,
    env,
    form,
    expected,
):
    import litellm

    env = dict(env)
    base_env = env.pop("BASE_ENV", "")
    if form and endpoint_setting == "deployment":
        resolved = runtime.resolve_litellm_route(model, env)
        configured = replace(
            resolved,
            litellm_params={**resolved.litellm_params, "api_base": form + slash},
        )
        monkeypatch.setattr(runtime, "resolve_litellm_route", lambda *a: configured)
    elif form:
        if endpoint_setting == "benchflow":
            base_env = "BENCHFLOW_PROVIDER_BASE_URL"
        env[base_env] = form + slash
    route, proxy_env = await launcher(model, env, placement)
    for name, value in {**env, **proxy_env}.items():
        monkeypatch.setenv(name, value)
    config = litellm_proxy_config(route, master_key="dummy")
    models = cast(list[dict[str, Any]], config["model_list"])
    params = dict(models[0]["litellm_params"])
    params["api_key"] = "dummy"
    # The recorder intentionally returns a provider error after recording URL.
    with suppress(Exception):
        await litellm.acompletion(
            **params,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            num_retries=0,
        )
    assert recorder == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["local", "daytona"])
@pytest.mark.parametrize("slash", ["", "/", "//"])
@pytest.mark.parametrize(
    "model,env,form,expected",
    [case for case in FORMS if case[0] in {"openai/gpt-4o", "us-openai/gpt-4o"}],
)
async def test_official_openai_forms_generate_responses_urls(
    launcher, recorder, monkeypatch, placement, slash, model, env, form, expected
):
    import litellm

    env = {**env, env["BASE_ENV"]: form + slash}
    route, _ = await launcher(model, env, placement)
    with suppress(Exception):
        await litellm.aresponses(
            model=route.upstream_model,
            api_base=route.litellm_params["api_base"],
            api_key="dummy",
            input="hi",
            num_retries=0,
        )
    assert recorder == [expected.removesuffix("chat/completions") + "responses"]


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["local", "daytona"])
@pytest.mark.parametrize("version", ["v1", "v1beta"])
@pytest.mark.parametrize(
    "operation", ["generateContent", "streamGenerateContent", "countTokens"]
)
@pytest.mark.parametrize("form", CASES[3][2] + CASES[4][2])
async def test_official_gemini_native_origin_is_separate(
    launcher, monkeypatch, placement, version, operation, form
):
    from fastapi import FastAPI, Request, Response
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as vendor
    from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
        PassthroughEndpointRouter,
    )

    env = {"GEMINI_API_KEY": "dummy", "BENCHFLOW_PROVIDER_BASE_URL": form}
    route, proxy_env = await launcher("gemini/gemini-2.5-flash", env, placement)
    for name, value in runtime._native_passthrough_env(
        route, {**env, **proxy_env}
    ).items():
        monkeypatch.setenv(name, value)
    router = PassthroughEndpointRouter()
    monkeypatch.setattr(vendor, "passthrough_endpoint_router", router)
    observed = []

    async def auth(**kwargs):
        return UserAPIKeyAuth()

    def forwarding(**kwargs):
        observed.append(kwargs["target"])

        async def response(*args):
            return {"ok": True}

        return response

    monkeypatch.setattr(vendor, "user_api_key_auth", auth)
    monkeypatch.setattr(vendor, "create_pass_through_route", forwarding)
    app = FastAPI()

    @app.post("/gemini/{endpoint:path}")
    async def native(endpoint: str, request: Request):
        return await vendor.gemini_proxy_route(endpoint, request, Response())

    path = f"/{version}/models/gemini-2.5-flash:{operation}"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://dummy"
    ) as client:
        response = await client.post("/gemini" + path, json={"contents": []})
    assert response.status_code == 200
    assert observed == ["https://generativelanguage.googleapis.com" + path]


def test_endpoint_table_has_no_untested_forms():
    covered = {"vertex_ai": {"vertex-sdk-default"}, "bedrock": {"bedrock-sdk-default"}}
    for model, env, form, _ in FORMS:
        route = runtime.resolve_litellm_route(model, env)
        provider = rm._PROVIDER_ALIASES.get(route.provider_name, route.provider_name)
        if provider == "native":
            provider = route.upstream_model.split("/", 1)[0]
        if form:
            covered.setdefault(provider, set()).add(form)
    expected = {
        p: {
            f.format(model=SAMPLES[p][0].split("/", 1)[1] if p in SAMPLES else "")
            for row in rows
            for f in row.forms
        }
        for p, rows in ENDPOINT_TABLE.items()
    }
    assert covered == expected


@pytest.mark.parametrize("source", ["controller", "role"])
@pytest.mark.parametrize("model,env,form,expected", FORMS)
def test_official_forms_normalize_through_every_endpoint_fallback(
    monkeypatch, source, model, env, form, expected
):
    if form is None:
        return  # SDK-only providers have no admitted custom base
    route = runtime.resolve_litellm_route(model, env)
    admission, sealed, _ = rm.admit_restricted_route(
        replace(route, litellm_params={**route.litellm_params, "api_base": form}), env
    )
    provider = admission.provider
    params = {k: v for k, v in route.litellm_params.items() if k != "api_base"}
    for name in rm.ENDPOINT_ENV.get(provider, ()):
        # Independently exercise every fallback without an earlier variable
        # shadowing the one being tested.
        for other in rm.ENDPOINT_ENV[provider]:
            monkeypatch.delenv(other, raising=False)
        if source == "controller":
            monkeypatch.setenv(name, form + "/")
        admitted, resolved, _ = rm.admit_restricted_route(
            replace(route, litellm_params=params),
            {**env, **({name: form + "/"} if source == "role" else {})},
        )
        assert admitted.endpoint == admission.endpoint
        assert resolved.litellm_params["api_base"] == sealed.litellm_params["api_base"]


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["local", "daytona"])
@pytest.mark.parametrize("model,env,form,expected", FORMS)
async def test_nonofficial_host_and_path_refused(
    launcher, monkeypatch, placement, model, env, form, expected
):
    route = runtime.resolve_litellm_route(model, env)
    origin = str(httpx.URL(form or expected).copy_with(path="/")).rstrip("/")
    host = httpx.URL(origin).host
    for bad in (
        origin + "/private",
        origin.replace(host, host + ".evil.invalid"),
        origin + "/v1/../private",
        origin + "/%76%31",
        origin + "?deployment=search",
        origin + "#fragment",
        origin.replace("https://", "http://"),
        origin.replace(host, "dummy@" + host),
        origin + ":444",
        origin + "/v1//messages",
        (form or origin) + "/private",
        (form or origin) + "/../private",
        origin + "/Beta",
        origin + "/beta/v1",
        origin + "/beta/./",
        origin + "/%62eta",
    ):
        bad_route = replace(
            route, litellm_params={**route.litellm_params, "api_base": bad}
        )
        monkeypatch.setattr(runtime, "resolve_litellm_route", lambda *a, r=bad_route: r)
        with pytest.raises(RuntimeError, match=rm.OPERATOR_ENV) as refused:
            await launcher(model, env, placement)
        provider = rm._PROVIDER_ALIASES.get(route.provider_name, route.provider_name)
        if provider == "native":
            provider = route.upstream_model.split("/", 1)[0]
        row = ENDPOINT_TABLE[provider][0]
        if row.base.startswith("https://"):
            assert any(
                "base URL " + allowed.base in str(refused.value)
                for allowed in ENDPOINT_TABLE[provider]
            )
        else:
            assert "SDK default; remove custom endpoint settings" in str(refused.value)


def test_full_gemini_endpoints_are_bound_to_the_identified_model():
    """Guards FF-ACP-CONF-3 / 5e4204fb: no wildcard model path admission."""
    models = [
        base for provider, _, base in rm.first_party_models() if provider == "gemini"
    ]
    for base in models:
        for row in ENDPOINT_TABLE["gemini"]:
            for form in row.forms:
                if "{model}" not in form:
                    continue
                admission = rm.admit_restricted_model(
                    "gemini/" + base, api_base=form.format(model=base), environ={}
                )
                assert admission.endpoint == row.base
                with pytest.raises(rm.RestrictedModelRefused):
                    rm.admit_restricted_model(
                        "gemini/" + base,
                        api_base=form.format(model="other-model"),
                        environ={},
                    )


@pytest.mark.parametrize("provider,rows", ENDPOINT_TABLE.items())
def test_endpoint_inventory_is_unambiguous_and_dispatch_bases_are_fixed(provider, rows):
    """Guards FF-ACP-CONF-3 / 5e4204fb: rows cannot shadow another mapping."""
    forms = [form for row in rows for form in row.forms]
    assert len(forms) == len(set(forms))
    for row in rows:
        assert row.base in row.forms
        assert "{" not in row.base


@pytest.mark.parametrize("provider", SAMPLES)
@pytest.mark.parametrize(
    "path",
    [
        "/embeddings",
        "/files",
        "/search",
        "/fetch",
        "/completions",
        "/v1beta/openai",
        "/anthropic",
    ],
)
def test_nonchat_protocols_and_hosted_retrieval_paths_stay_refused(provider, path):
    """Guards FF-ACP-CONF-3 / 5e4204fb without broadening the wire identity."""
    model, _ = SAMPLES[provider]
    for row in ENDPOINT_TABLE[provider]:
        origin = str(httpx.URL(row.base).copy_with(path="/")).rstrip("/")
        for base in {origin, row.base}:
            with pytest.raises(rm.RestrictedModelRefused):
                rm.admit_restricted_model(model, api_base=base + path, environ={})


@pytest.mark.parametrize("provider,rows", ENDPOINT_TABLE.items())
def test_endpoint_rows_never_override_hosted_retrieval_refusal(provider, rows):
    """Guards FF-ACP-CONF-3 / 5e4204fb: endpoint aliases grant no search rights."""
    model = "openai/gpt-4o-search-preview"
    for row in rows:
        for form in row.forms:
            with pytest.raises(rm.RestrictedModelRefused, match="choose a model"):
                rm.admit_restricted_model(
                    model,
                    provider_name=provider,
                    api_base=form.format(model="gpt-4o-search-preview"),
                    environ={rm.OPERATOR_ENV: model},
                )


@pytest.mark.parametrize("provider", SAMPLES)
def test_endpoint_refusal_suggests_a_public_base_without_echoing_private_values(
    provider,
):
    """Guards FF-ACP-CONF-3 / 5e4204fb's useful but non-reflecting refusal."""
    model, _ = SAMPLES[provider]
    with pytest.raises(rm.RestrictedModelRefused) as refused:
        rm.admit_restricted_model(
            model,
            api_base="https://dummy-private@gateway.invalid/secret-marker?dummy-private",
            environ={},
        )
    assert "base URL " + ENDPOINT_TABLE[provider][0].base in str(refused.value)
    assert "dummy-private" not in str(refused.value)
    assert "secret-marker" not in str(refused.value)
    assert "gateway.invalid" not in str(refused.value)
