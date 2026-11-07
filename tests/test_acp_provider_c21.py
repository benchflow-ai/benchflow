"""A restricted run's model is identified.

C21 (OpenRouter presets), C22 (router aliases) and C23 (OpenAI fine-tunes)
were the third round of catalog spellings. With web access restricted the
controller now admits only a model it can positively identify in pinned
LiteLLM metadata, at launch; everything else needs the operator entry, and
the catalog stays as a second check. Dummy values only.
"""

import json
from pathlib import Path
from typing import Any, cast

import litellm
import pytest

from benchflow.providers import litellm_runtime as runtime
from benchflow.providers.litellm_config import (
    litellm_proxy_config,
    resolve_litellm_route,
)
from benchflow.providers.litellm_logging import callback_module_source
from tests import test_acp_provider_c13 as c13

try:
    from benchflow.providers import restricted_models as rm
except ImportError:  # the failing-first run on e4420414, before admission
    rm = cast(Any, None)

deployed = c13.deployed
OPERATOR_ENV = "BENCHFLOW_RESTRICTED_WEB_MODELS"  # documented operator key


def test_operator_key_is_the_documented_one():
    assert rm.OPERATOR_ENV == OPERATOR_ENV
    # Docs v2: the restricted-network model check is on this page.
    doc = (
        Path(__file__).resolve().parents[1] / "docs/tasks/network-and-secrets.md"
    ).read_text()
    assert f"{OPERATOR_ENV}=" in doc


MODES = {"chat", "responses", "completion"}


def _guard(source):
    namespace: dict[str, Any] = {}
    exec(compile(source, "request_guard.py", "exec"), namespace)
    return namespace


def _admit(model, **kwargs):
    return rm.admit_restricted_model(model, environ=kwargs.pop("environ", {}), **kwargs)


# --- (1) dispatch is pinned to the configured model --------------------------


@pytest.mark.parametrize(
    "model,env",
    [
        ("openai/gpt-4.1", {"OPENAI_API_KEY": "dummy"}),
        ("anthropic/claude-sonnet-4-5", {"ANTHROPIC_API_KEY": "dummy"}),
        ("gemini/gemini-2.5-flash", {"GEMINI_API_KEY": "dummy"}),
        (
            "openai/gpt-4.1",
            {
                "OPENAI_API_KEY": "dummy",
                "BENCHFLOW_PROVIDER_BASE_URL": "http://dummy.invalid/v1",
            },
        ),
    ],
)
def test_every_served_name_dispatches_to_the_configured_model(model, env):
    """litellm_config.litellm_proxy_config builds every entry from one route.

    The request's model can choose only an alias of that route; the callback
    gate refuses any other name before LiteLLM routes it (next test).
    """
    route = resolve_litellm_route(model, env)
    config = litellm_proxy_config(route, master_key="dummy")
    entries: list[dict[str, Any]] = cast(Any, config["model_list"])
    assert isinstance(entries, list)
    upstream = route.upstream_model
    bridge = f"openai/chat_completions/{upstream.split('/', 1)[-1]}"
    for entry in entries:
        params = dict(entry["litellm_params"])
        assert params.pop("model") in {upstream, bridge}, entry
        expected = dict(route.litellm_params)
        expected.pop("model")
        assert {k: v for k, v in params.items() if k in expected} == expected
        assert "*" not in entry["model_name"]


@pytest.mark.asyncio
async def test_a_body_naming_another_model_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(tmp_path / "log.jsonl"))
    monkeypatch.setenv(
        "BENCHFLOW_LITELLM_SERVED_MODELS",
        json.dumps(["runtime-openai-gpt-4.1", "openai/gpt-4.1"]),
    )
    namespace: dict = {}
    exec(callback_module_source(), namespace)
    hook = namespace["proxy_handler_instance"].async_pre_call_hook
    for foreign in (
        "gpt-4o-search-preview",
        "openrouter/fusion",
        "openai/gpt-4.1:online",
    ):
        with pytest.raises(namespace["ModelNotServed"]):
            await hook(None, None, {"model": foreign, "messages": []}, "acompletion")


# --- (2)/(4) admission is generated over pinned LiteLLM metadata ------------


def _configured(key, info):
    """The configured upstream id a controller would use for one entry."""
    from benchflow.providers.litellm_request_guard import split_provider

    if split_provider(key)[0] is not None and "/" in key:
        return key
    return f"{str(info['litellm_provider']).split('-', 1)[0]}/{key}"


@pytest.fixture
def no_provider_hooks(monkeypatch):
    """get_llm_provider runs provider hooks (github_copilot logs in over the
    network); neither admission nor the guard may call it."""

    def refuse(*args, **kwargs):
        raise AssertionError("get_llm_provider called")

    monkeypatch.setattr(litellm, "get_llm_provider", refuse)


def test_every_admitted_model_comes_from_pinned_metadata(no_provider_hooks):
    metadata = rm.pinned_model_metadata()
    admitted = refused = 0
    for key, info in metadata.items():
        if not isinstance(info, dict) or info.get("mode") not in MODES:
            continue
        model = _configured(key, info)
        try:
            admission = _admit(model)
        except rm.RestrictedModelRefused as exc:
            refused += 1
            assert "web access restricted" in str(exc)
            continue
        admitted += 1
        assert admission.admitted_by == "metadata"
        found = metadata[admission.metadata_key]
        assert found.get("mode") in MODES, (model, admission)
        guard = __import__("benchflow.providers.litellm_request_guard", fromlist=["x"])
        assert guard.hosted_retrieval_model(model) is None, model
        assert not model.startswith(("azure/", "azure_ai/")), model
    # C24/C25 (85b87bc9): only curated first-party providers and chat modes
    # now qualify; gateway metadata must no longer expand this population.
    assert 0 < admitted < refused, (admitted, refused)


@pytest.mark.parametrize(
    "model,provider_name",
    [
        ("openai/gpt-4.1", ""),  # OpenAI chat and responses
        ("openai/gpt-5.5", ""),
        ("anthropic/claude-sonnet-4-5", ""),  # Anthropic messages
        ("anthropic/claude-haiku-4-5", ""),
        ("gemini/gemini-2.5-flash", ""),  # Gemini
        ("vertex_ai/gemini-2.5-flash", ""),  # Vertex Gemini
        ("vertex_ai/claude-sonnet-4-5@20250929", ""),  # Vertex rawPredict
        ("bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0", ""),  # Bedrock
        ("bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", ""),
        ("bedrock/invoke/amazon.nova-pro-v1:0", ""),
        ("bedrock/converse/meta.llama3-1-70b-instruct-v1:0", ""),
        (
            "bedrock/arn:aws:bedrock:us-east-1:123456789012:inference-profile/"
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "",
        ),
        (
            "bedrock/arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0",
            "",
        ),
        ("deepseek/deepseek-chat", ""),
        ("openrouter/openai/gpt-4o", ""),
        ("openai/glm-4.6", "zai"),  # a registered OpenAI-compatible provider
        ("openai/ft:gpt-4o-mini-2024-07-18:example:bench:abc123", ""),  # C23
    ],
)
def test_honest_models_on_every_protocol_are_admitted(
    no_provider_hooks, model, provider_name
):
    # C24/C25: fixed gateway routes remain usable with the operator entry.
    gateway = model.startswith("openrouter/")
    admission = _admit(
        model,
        provider_name=provider_name,
        environ={OPERATOR_ENV: model} if gateway else {},
    )
    assert admission.admitted_by == ("operator" if gateway else "metadata")
    if not gateway:
        assert admission.metadata_key in rm.pinned_model_metadata()


def test_finetune_is_classified_by_its_base_and_dispatched_whole(deployed, monkeypatch):
    model = "openai/ft:gpt-4o-mini-2024-07-18:example:bench:abc123"
    assert _admit(model).base == "gpt-4o-mini-2024-07-18"
    guard = _guard(deployed)
    monkeypatch.setenv("BENCHFLOW_LITELLM_UPSTREAM_MODEL", model)
    body = {
        "model": "ft:gpt-4o-mini-2024-07-18:example:bench:abc123",
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert (
        guard["outbound_refusal"](
            "POST", "/v1/chat/completions", json.dumps(body).encode()
        )
        is None
    )


# --- (3) everything else is refused at launch, with the fix -----------------

# Routers and presets choose models and tools, web search included, per
# request: catalogued, so no operator entry admits them.
_ROUTED = [
    "openrouter/openai/gpt-4o-search-preview@preset/plain",  # C21
    "openrouter/@preset/plain",
    "openrouter/openrouter/fusion",  # C22
    "openrouter/openrouter/fusion-flash",
    "openrouter/openrouter/auto",
    "openai/openrouter/fusion",
]
_REFUSED = [
    "openrouter/openai/gpt-4o:turbo",  # unknown variant
    "openai/gpt-4.1:search",
    "azure/my-deployment",  # opaque Azure deployment
    "azure_ai/claude-sonnet-4-5",
    "hosted_vllm/fake-policy",  # operator-run server
    "bedrock/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc",
    "bedrock/invoke/llama/arn:aws:bedrock:us-east-1:123456789012:imported-model/abc",
    "vertex_ai/1234567890",  # a Vertex endpoint id
    "notaprovider/some-model",  # unrecognised provider
]


@pytest.mark.parametrize("model", _REFUSED)
def test_unidentified_models_are_refused_with_the_operator_entry(
    no_provider_hooks, model
):
    with pytest.raises(rm.RestrictedModelRefused) as refused:
        _admit(model)
    message = str(refused.value)
    assert f"{OPERATOR_ENV}={model}" in message
    assert "config.json" in message


@pytest.mark.parametrize("model", _ROUTED)
def test_routers_and_presets_are_refused_even_with_an_operator_entry(model):
    for environ in ({}, {OPERATOR_ENV: model}):
        with pytest.raises(rm.RestrictedModelRefused, match=r"per request|preset"):
            _admit(model, environ=environ)


@pytest.mark.parametrize("model", _REFUSED)
def test_operator_entry_admits_an_unidentified_model_and_says_so(model):
    admission = _admit(model, environ={OPERATOR_ENV: f"other/x, {model}"})
    assert admission.admitted_by == "operator"
    assert admission.record()["model"] == model


def test_provider_wildcard_entry_admits_an_operator_run_server():
    admission = _admit(
        "hosted_vllm/fake-policy", environ={OPERATOR_ENV: "hosted_vllm/*"}
    )
    assert admission.admitted_by == "operator"


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4o-search-preview",
        "openrouter/openrouter/fusion",
        "openrouter/openai/gpt-4o-search-preview@preset/plain",
        "openrouter/openai/gpt-4o:online",
    ],
)
def test_no_operator_entry_admits_a_catalogued_model(model):
    with pytest.raises(rm.RestrictedModelRefused, match="hosted web search"):
        _admit(model, environ={OPERATOR_ENV: model})


@pytest.mark.parametrize(
    "model",
    [
        "openrouter/openai/gpt-4o-search-preview@preset/plain",
        "openai/gpt-4o-search-preview@preset/plain",
        "openrouter/openrouter/fusion",
        "openrouter/fusion",
        "openrouter/fusion-flash",
        "openai/openrouter/auto",
    ],
)
def test_guard_second_check_refuses_presets_and_routers(
    deployed, no_provider_hooks, model
):
    assert _guard(deployed)["hosted_retrieval_model"](model)


@pytest.mark.parametrize(
    "model",
    [
        "vertex_ai/claude-sonnet-4-5@20250929",
        "openai/ft:gpt-4o-mini-2024-07-18:example:bench:abc123",
        "ft:gpt-4o-mini-2024-07-18:example:bench:abc123",
    ],
)
def test_guard_second_check_admits_vertex_versions_and_finetunes(deployed, model):
    assert _guard(deployed)["hosted_retrieval_model"](model) is None


# --- launch: refused before any proxy starts; overrides are recorded --------


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", ["local", "docker"])
async def test_launch_refuses_before_any_proxy_starts(monkeypatch, environment):
    started = []

    async def start(**kwargs):
        started.append(kwargs)
        raise AssertionError("a refused model must not start a proxy")

    monkeypatch.setattr(runtime, "_start_host_litellm", start)
    monkeypatch.setattr(runtime, "_start_sandbox_litellm", start)
    monkeypatch.delenv(OPERATOR_ENV, raising=False)
    for model, expected in (
        ("openai/my-private-model", OPERATOR_ENV),
        ("openrouter/openrouter/fusion", "per request"),  # C22: no override
    ):
        with pytest.raises(RuntimeError, match=expected):
            await runtime.ensure_litellm_runtime(
                agent="codex-acp",
                model=model,
                agent_env={"OPENAI_API_KEY": "dummy", "OPENROUTER_API_KEY": "dummy"},
                runtime=None,
                environment=environment,
                sandbox=object(),
                disallow_web_tools=True,
            )
    assert not started


@pytest.mark.asyncio
async def test_agent_env_cannot_supply_the_operator_entry(monkeypatch):
    async def start(**kwargs):
        raise AssertionError("must not start")

    monkeypatch.setattr(runtime, "_start_host_litellm", start)
    monkeypatch.delenv(OPERATOR_ENV, raising=False)
    with pytest.raises(RuntimeError, match=OPERATOR_ENV):
        await runtime.ensure_litellm_runtime(
            agent="codex-acp",
            model="openai/my-private-model",
            agent_env={
                "OPENAI_API_KEY": "dummy",
                OPERATOR_ENV: "openai/my-private-model",
            },
            runtime=None,
            environment="local",
            disallow_web_tools=True,
        )


@pytest.mark.asyncio
async def test_override_and_metadata_admissions_are_recorded(monkeypatch, tmp_path):
    from benchflow.rollout import Rollout, RolloutConfig
    from tests.test_litellm_runtime import FakeLiteLLMServer

    async def start(**kwargs):
        return FakeLiteLLMServer("http://dummy", kwargs["route"])

    monkeypatch.setattr(runtime, "_start_host_litellm", start)
    monkeypatch.setenv(OPERATOR_ENV, "openai/my-private-model")
    trial = Rollout(RolloutConfig(task_path=tmp_path / "task"))
    trial._rollout_dir = tmp_path
    (tmp_path / "config.json").write_text("{}")
    for model in ("openai/my-private-model", "openai/gpt-4.1"):
        _, trial._usage_runtime = await runtime.ensure_litellm_runtime(
            agent="codex-acp",
            model=model,
            agent_env={"OPENAI_API_KEY": "dummy"},
            runtime=trial._usage_runtime,
            environment="local",
            disallow_web_tools=True,
        )
        trial._record_proxy_layer_use()
    records = json.loads((tmp_path / "config.json").read_text())["restricted_web_model"]
    assert [(r["model"], r["admitted_by"]) for r in records] == [
        ("openai/my-private-model", "operator"),
        ("openai/gpt-4.1", "metadata"),
    ]


@pytest.mark.asyncio
async def test_unrestricted_runs_are_not_screened(monkeypatch):
    from tests.test_litellm_runtime import FakeLiteLLMServer

    async def start(**kwargs):
        return FakeLiteLLMServer("http://dummy", kwargs["route"])

    monkeypatch.setattr(runtime, "_start_host_litellm", start)
    _, current = await runtime.ensure_litellm_runtime(
        agent="codex-acp",
        model="openrouter/openrouter/fusion",
        agent_env={"OPENROUTER_API_KEY": "dummy"},
        runtime=None,
        environment="local",
        disallow_web_tools=False,
    )
    assert current is not None and current.restricted_web_model is None
