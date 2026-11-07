"""Identity, not spelling, decides.

C18: a catalogued model behind a routing variant (``:nitro``) or another
provider prefix is the same model; a variant that searches is retrieval; an
unknown variant is refused under restrictions. C19/C20: Anthropic's client
tools, memory included, with every option the pinned SDK defines, from a
table generated from that SDK. Dummy values only.
"""

import json
import tomllib
from pathlib import Path
from typing import Any, Literal, Required, TypedDict

import httpx
import pytest

from benchflow.providers import litellm_runtime as runtime
from tests import test_acp_provider_c13 as c13
from tools import anthropic_client_tools as generator

deployed = c13.deployed
REPO = Path(__file__).resolve().parents[1]
UPSTREAM = "BENCHFLOW_LITELLM_UPSTREAM_MODEL"
DISALLOW = "BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS"
BY_ITSELF = "reaches the web by itself"


def _guard(source):
    namespace: dict[str, Any] = {}
    exec(compile(source, "request_guard.py", "exec"), namespace)
    return namespace


def _spellings(model):
    """The ways a controller or a request can name one catalogued model."""
    *path, base = model.split("/")
    vendor = path[-1] if path and path[0] not in {"openai", "azure"} else "openai"
    names = {
        model,
        base,
        f"openrouter/{vendor}/{base}",
        f"openai/{vendor}/{base}",  # an OpenAI-compatible gateway route
        f"vercel_ai_gateway/{vendor}/{base}",
    }
    if path and path[0] not in {"openai", "azure"}:
        names.add(f"{path[0]}/{base}")
    return sorted(names)


def _suffixes(guard):
    known = [v for v, why in guard["MODEL_VARIANTS"].items() if why is None]
    return ["", *(f":{v}" for v in known), ":nitro:floor", ":online", ":free:online"]


def test_every_catalog_family_is_refused_in_every_spelling_and_variant(deployed):
    guard = _guard(deployed)
    checked = 0
    for kind, _pattern, _why, examples in guard["HOSTED_RETRIEVAL_MODELS"]:
        for example in examples:
            for name in _spellings(example) if kind == "model" else [example]:
                for suffix in _suffixes(guard):
                    reason = guard["hosted_retrieval_model"](name + suffix)
                    assert reason and BY_ITSELF in reason, name + suffix
                    checked += 1
    assert checked > 300


def test_reported_openrouter_variant_is_refused_at_the_boundary(deployed, monkeypatch):
    guard = _guard(deployed)
    body = {
        "model": "openai/gpt-4o-search-preview:nitro",
        "messages": [{"role": "user", "content": "Find today's news."}],
    }
    for configured in (
        "openai/openai/gpt-4o-search-preview:nitro",
        "openrouter/openai/gpt-4o-search-preview:nitro",
        "openrouter/openai/gpt-4o-search-preview:floor",
    ):
        monkeypatch.setenv(UPSTREAM, configured)
        reason = guard["outbound_refusal"](
            "POST", "/api/v1/chat/completions", json.dumps(body).encode()
        )
        assert reason and BY_ITSELF in reason, configured
    # The model the request names decides even when the route looks plain.
    monkeypatch.setenv(UPSTREAM, "openai/openai/gpt-4o")
    reason = guard["outbound_refusal"](
        "POST", "/api/v1/chat/completions", json.dumps(body).encode()
    )
    assert reason and BY_ITSELF in reason


@pytest.mark.parametrize(
    "model",
    [
        "openrouter/openai/gpt-4o:nitro",
        "openrouter/anthropic/claude-sonnet-4.5:floor",
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "openai/openai/gpt-4o:extended",
        "openrouter/deepseek/deepseek-r1:thinking",
        "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0",
        "bedrock/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc",
        "bedrock/invoke/llama/arn:aws:bedrock:us-east-1:123456789012:imported-model/abc",
        "ollama/llama3:8b",
        "ollama_chat/qwen3:14b",
        "vertex_ai/claude-sonnet-4-5@20250929",
        "gemini/gemini-2.5-flash",
        "hosted_vllm/fake-policy",
    ],
)
def test_ordinary_models_and_their_variants_pass(deployed, model):
    assert _guard(deployed)["hosted_retrieval_model"](model) is None


@pytest.mark.parametrize(
    "model", ["openrouter/openai/gpt-4o:turbo", "openai/gpt-4.1:search", "x:y:z"]
)
def test_unknown_variant_is_refused_at_launch_with_what_to_do(deployed, model):
    """Unknown variants are refused when the run starts.

    The guard's second check keeps only what is known to reach the web, so
    an operator entry that admits the configured model is not undone.
    """
    from benchflow.providers import restricted_models as rm

    with pytest.raises(rm.RestrictedModelRefused) as refused:
        rm.admit_restricted_model(model, environ={})
    assert f"{rm.OPERATOR_ENV}={model}" in str(refused.value)
    assert _guard(deployed)["hosted_retrieval_model"](model) is None


def test_identity_uses_litellms_provider_registry(deployed):
    identity = _guard(deployed)["model_identity"]
    assert identity("openrouter/openai/gpt-4o-search-preview:nitro") == (
        "openrouter",
        ("openai",),
        "gpt-4o-search-preview",
        ("nitro",),
    )
    assert identity("openai/openai/gpt-4o-search-preview:nitro")[2:] == (
        "gpt-4o-search-preview",
        ("nitro",),
    )
    # A Bedrock id's colons are its version, never a routing variant.
    assert identity("anthropic.claude-sonnet-4-5-20250929-v1:0", "bedrock")[3] == ()


@pytest.mark.parametrize(
    "path,body",
    [
        (
            "/model/anthropic.claude-sonnet-4-5-20250929-v1:0/invoke",
            {"messages": [], "max_tokens": 1},
        ),
        (
            "/model/openai.gpt-oss-120b-1:0/invoke",
            {"model": "invoke/openai.gpt-oss-120b-1:0", "messages": []},
        ),
        (
            "/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A1%3Aimported-model%2Fx/invoke",
            {"prompt": "hi"},
        ),
    ],
)
def test_bedrock_ids_keep_their_colons(deployed, monkeypatch, path, body):
    monkeypatch.setenv(UPSTREAM, "bedrock/invoke/" + path.split("/")[2])
    reason = _guard(deployed)["outbound_refusal"](
        "POST", path, json.dumps(body).encode()
    )
    assert reason is None or "variant" not in reason, reason


# --- C19/C20: the client-tool table is the pinned SDK's ---------------------


def test_generated_table_names_the_sdk_version_uv_lock_pins(deployed):
    """Fails as soon as uv.lock moves the SDK: regenerate the table then."""
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    pinned = {p["name"]: p["version"] for p in lock["package"]}["anthropic"]
    assert _guard(deployed)["ANTHROPIC_SDK_VERSION"] == pinned, (
        "regenerate: uv run --extra judge python -m tools.anthropic_client_tools"
    )


def test_generated_table_matches_the_pinned_sdk():
    """Runs where the pinned SDK is installed (the judge extra)."""
    pytest.importorskip("anthropic")
    assert generator.main(["--check"]) == 0


class _Bash(TypedDict, total=False):
    type: Required[Literal["bash_20250124"]]
    name: Required[Literal["bash"]]
    strict: bool
    allowed_callers: list[Literal["direct", "code_execution_20250825"]]


class _Search(TypedDict, total=False):
    type: Required[Literal["web_search_20250305"]]
    name: Required[Literal["web_search"]]
    allowed_domains: list[str]


class _Custom(TypedDict, total=False):
    type: Literal["custom"]
    name: Required[str]
    input_schema: dict


def _committed():
    guard = _guard(
        (REPO / "src/benchflow/providers/litellm_request_guard.py").read_text()
    )
    return {
        "version": guard["ANTHROPIC_SDK_VERSION"],
        "custom": set(guard["ANTHROPIC_CUSTOM_TOOL"]),
        "tools": {k: set(v) for k, v in guard["ANTHROPIC_CLIENT_TOOLS"].items()},
        "callers": set(guard["ANTHROPIC_CLIENT_CALLERS"]),
    }


@pytest.mark.parametrize(
    "change",
    ["new field on a known tool", "new client tool version", "new caller"],
)
def test_check_fails_when_the_sdk_grows_a_client_tool_or_field(change):
    table = _committed()
    if change == "new field on a known tool":
        table["tools"]["bash_20250124"].add("future_option")
    elif change == "new client tool version":
        table["tools"]["bash_29991231"] = {"type", "name"}
    else:
        table["callers"].add("direct_v2")
    assert generator.main(["--check"], table=table) == 1


def test_generator_classifies_from_the_sdk_types_and_refuses_unknown_families():
    union = _Custom | _Bash | _Search
    table = generator.table_from((union,), "0.0.0")
    assert table["tools"] == {
        "bash_20250124": {"type", "name", "strict", "allowed_callers"}
    }
    assert table["callers"] == {"direct"}
    assert table["custom"] == {"type", "name", "input_schema"}

    class _Future(TypedDict, total=False):
        type: Required[Literal["hologram_20991231"]]

    with pytest.raises(SystemExit, match="unclassified"):
        generator.table_from((_Custom | _Future,), "0.0.0")


def test_committed_table_is_ruff_stable_and_regenerates_byte_for_byte():
    source = (REPO / "src/benchflow/providers/litellm_request_guard.py").read_text()
    assert generator.current_block(source) == generator.render(_committed())


@pytest.mark.parametrize(
    "tool,refused",
    [
        ({"type": "memory_20250818", "name": "memory"}, False),
        ({"type": "bash_20250124", "name": "bash", "strict": True}, False),
        ({"type": "bash_20250124", "name": "bash", "input_examples": [{}]}, False),
        (
            {"type": "bash_20250124", "name": "bash", "allowed_callers": ["direct"]},
            False,
        ),
        (
            {
                "type": "bash_20250124",
                "name": "bash",
                "allowed_callers": ["code_execution_20250825"],
            },
            True,
        ),
        (
            {"type": "memory_20250818", "name": "memory", "allowed_callers": "direct"},
            True,
        ),
        ({"type": "memory_20250818", "name": "memory", "max_uses": 3}, True),
        ({"type": "web_fetch_20260309", "name": "web_fetch"}, True),
    ],
)
def test_both_guards_agree_on_client_tools(deployed, monkeypatch, tool, refused):
    import asyncio

    from fastapi import FastAPI, Request

    guard = _guard(deployed)
    for path in ("/v1/messages", "/v1/messages/count_tokens"):
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "tools": [tool],
        }
        reason = guard["outbound_refusal"]("POST", path, json.dumps(body).encode())
        assert bool(reason) is refused, (path, reason)

    app = FastAPI()

    @app.post("/{rest:path}")
    async def echo(request: Request):
        return {"body": await request.json()}

    app.add_middleware(guard["SingleRouteGuard"])
    monkeypatch.setenv(DISALLOW, "1")

    async def inbound():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://dummy"
        ) as client:
            return await client.post(
                "/v1/messages/count_tokens",
                json={
                    "model": "allowed",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [tool],
                },
            )

    response = asyncio.run(inbound())
    # The inbound guard admits every generated client tool type (memory
    # included); the outbound check then decides its fields.
    type_refused = tool["type"] not in guard["ANTHROPIC_CLIENT_TOOLS"]
    assert (response.status_code == 400) is type_refused, response.text


def test_deployed_guard_needs_no_sdk(deployed):
    """The table travels inside the standalone file both placements copy."""
    assert "import anthropic" not in deployed
    assert generator.BEGIN in deployed and generator.END in deployed
    assert runtime  # both placements ship the same file (c13's fixture)
