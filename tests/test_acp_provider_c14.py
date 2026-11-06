"""What the outbound allowlist admits.

C14: models that search by themselves declare nothing, so the controller's
model decides (catalog). C15: Bedrock ARNs keep their slashes encoded in one
path segment. C16: an Invoke body is the configured model family's own
protocol. C17: Anthropic's versioned client tools run on the caller's side.
Honest traffic for every protocol the check covers must leave unchanged.
Dummy values only; nothing connects anywhere.
"""

import base64
import copy
import json
import re
from typing import Any, cast
from urllib.parse import quote

import httpx
import pytest

from tests import test_acp_provider_c13 as c13
from tests.test_acp_provider_c13 import REFUSED, _hosted

# The c13 fixtures: both deployed guard files over a recording wire.
deployed = c13.deployed
wire = c13.wire

DISALLOW = "BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS"
UPSTREAM = "BENCHFLOW_LITELLM_UPSTREAM_MODEL"
BY_ITSELF = "reaches the web by itself"
_FN = {
    "type": "function",
    "function": {
        "name": "run",
        "description": "agent-executed",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}
_MSG = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
_EXTRA = {"temperature": 0.5, "top_p": 0.9, "stop": ["zz"], "max_tokens": 16}
_VERTEX = {"vertex_project": "dummy-project", "vertex_location": "us-central1"}
_ACCOUNT = "arn:aws:bedrock:us-east-1:123456789012"


async def _send(monkeypatch, wire, route, params, *, restricted):
    """One translated request through the deployed guard; the error text."""
    import litellm

    monkeypatch.setenv(DISALLOW, "1" if restricted else "0")
    monkeypatch.setenv(UPSTREAM, params["model"])
    wire.clear()
    call = {"api_key": "dummy-upstream", **copy.deepcopy(params)}
    try:
        if route == "chat":
            call.setdefault("messages", copy.deepcopy(_MSG))
            await litellm.acompletion(**call)
        elif route == "responses":
            await litellm.aresponses(input="hi", **call)
        else:
            call.setdefault("messages", [{"role": "user", "content": "hi"}])
            create = cast(Any, litellm.anthropic.messages.acreate)
            await create(max_tokens=16, **call)
    except Exception as exc:  # the recorder's stop, or the boundary's refusal
        return str(exc)
    return ""


async def _honest(monkeypatch, wire, route, params):
    """The restricted request leaves exactly as the unrestricted one does."""
    await _send(monkeypatch, wire, route, params, restricted=False)
    unrestricted = [w for w in wire if w[0] == "POST"]
    assert unrestricted, "pinned LiteLLM sent nothing even unrestricted"
    error = await _send(monkeypatch, wire, route, params, restricted=True)
    restricted = [w for w in wire if w[0] == "POST"]
    assert REFUSED not in error, error
    assert restricted == unrestricted, (restricted, unrestricted)
    assert not _hosted(restricted[-1][2])
    return restricted[-1]


# --- C14: models that search or retrieve by themselves -------------------


def _guard(source):
    namespace: dict[str, Any] = {}
    exec(compile(source, "request_guard.py", "exec"), namespace)
    return namespace


_PLAIN_BODY = {"messages": [{"role": "user", "content": "hi"}]}


def test_every_catalogued_model_is_refused_as_configured_and_as_named(
    deployed, monkeypatch
):
    guard = _guard(deployed)
    catalog = guard["HOSTED_RETRIEVAL_MODELS"]
    assert catalog and all(examples for *_, examples in catalog)
    for kind, _pattern, _why, examples in catalog:
        for model in examples:
            # As the controller-resolved model, whatever the request names.
            monkeypatch.setenv(UPSTREAM, model)
            body = json.dumps({**_PLAIN_BODY, "model": "alias"}).encode()
            reason = guard["outbound_refusal"]("POST", "/v1/chat/completions", body)
            assert reason and BY_ITSELF in reason, model
            if kind in {"provider", "router"}:
                continue  # known from the configured route or full id only
            # As the model the translated request names, on every wire shape.
            monkeypatch.setenv(UPSTREAM, "openai/gpt-4.1")
            bare = model.split("/")[-1]
            shapes = [
                ("/v1/chat/completions", {**_PLAIN_BODY, "model": bare}),
                ("/v1/responses", {"input": "hi", "model": bare}),
                ("/v1/messages", {**_PLAIN_BODY, "max_tokens": 1, "model": bare}),
                (f"/model/{quote(bare, safe='')}/converse", {"messages": []}),
            ]
            if ":" not in bare:  # a Gemini model id holds no colon
                shapes.append(
                    (f"/v1beta/models/{bare}:generateContent", {"contents": []})
                )
            for path, payload in shapes:
                reason = guard["outbound_refusal"](
                    "POST", path, json.dumps(payload).encode()
                )
                assert reason and BY_ITSELF in reason, (model, path)


def test_catalog_covers_every_pinned_self_searching_model(deployed):
    """Every pinned LiteLLM chat model of a self-searching family is caught."""
    import litellm

    guard = _guard(deployed)
    family = re.compile(
        r"search-preview|search-api|deep-research|sonar|compound|-online"
    )
    found = []
    for name, info in litellm.model_cost.items():
        if not isinstance(info, dict) or info.get("mode") not in {
            "chat",
            "responses",
            "completion",
            "image_generation",
        }:
            continue
        provider = str(info.get("litellm_provider", ""))
        if provider == "perplexity" or family.search(name.split("/")[-1]):
            found.append(name)
            assert guard["hosted_retrieval_model"](name), name
    assert len(found) >= 20, found


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4.1",
        "gpt-5.5",
        "anthropic/claude-sonnet-4-5",
        "gemini/gemini-2.5-flash",
        "deepseek/deepseek-chat",
        "bedrock/amazon.nova-pro-v1:0",
        "vllm/fake-policy",
        "groq/llama-3.3-70b-versatile",
        "openrouter/openai/gpt-4o",
    ],
)
def test_ordinary_models_are_not_catalogued(deployed, model):
    assert _guard(deployed)["hosted_retrieval_model"](model) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model", ["openai/gpt-5-search-api", "openai/gpt-4o-mini-search-preview"]
)
async def test_reported_search_model_needs_no_tool_and_is_refused(
    monkeypatch, wire, model
):
    """The reported input: pinned LiteLLM sends it with no search field."""
    plain = {"model": model, "api_base": "http://dummy.invalid/v1"}
    await _send(monkeypatch, wire, "chat", plain, restricted=False)
    assert wire and "web_search_options" not in wire[-1][2]
    assert not _hosted(wire[-1][2])
    error = await _send(monkeypatch, wire, "chat", plain, restricted=True)
    assert not wire and REFUSED in error and BY_ITSELF in error


# --- C15/C16: Bedrock ARNs and Invoke per model family -------------------

_BEDROCK_HONEST = [
    # (route, configured model, extra request fields)
    (
        "chat",
        f"bedrock/{_ACCOUNT}:application-inference-profile/abcdefghijkl",
        {"tools": [_FN]},
    ),
    (
        "chat",
        f"bedrock/{_ACCOUNT}:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        {"tools": [_FN]},
    ),
    (
        "chat",
        f"bedrock/converse/{_ACCOUNT}:application-inference-profile/abcdefghijkl",
        {},
    ),
    (
        "chat",
        f"bedrock/invoke/{_ACCOUNT}:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        {"tools": [_FN]},
    ),
    ("messages", f"bedrock/{_ACCOUNT}:application-inference-profile/abcdefghijkl", {}),
    (
        "messages",
        f"bedrock/{_ACCOUNT}:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        {},
    ),
    (
        "chat",
        "bedrock/invoke/anthropic.claude-sonnet-4-5-20250929-v1:0",
        {"tools": [_FN]},
    ),
    ("chat", "bedrock/invoke/amazon.nova-pro-v1:0", {"tools": [_FN]}),
    ("chat", "bedrock/invoke/us.amazon.nova-pro-v1:0", {}),
    ("chat", "bedrock/invoke/amazon.titan-text-express-v1", {}),
    ("chat", "bedrock/invoke/meta.llama3-1-70b-instruct-v1:0", {}),
    ("chat", "bedrock/invoke/mistral.mistral-large-2407-v1:0", {}),
    ("chat", "bedrock/invoke/cohere.command-r-plus-v1:0", {"tools": [_FN], "seed": 1}),
    ("chat", "bedrock/invoke/ai21.jamba-1-5-large-v1:0", {}),
    ("chat", "bedrock/invoke/openai.gpt-oss-120b-1:0", {}),
    (
        "chat",
        "bedrock/invoke/moonshot.kimi-k2-thinking",
        {"tools": [_FN], "tool_choice": "auto"},
    ),
    ("chat", f"bedrock/invoke/llama/{_ACCOUNT}:imported-model/abc123", {}),
    ("chat", f"bedrock/invoke/deepseek_r1/{_ACCOUNT}:imported-model/abc123", {}),
    ("chat", f"bedrock/invoke/qwen2/{_ACCOUNT}:imported-model/abc123", {}),
    ("chat", f"bedrock/invoke/qwen3/{_ACCOUNT}:imported-model/abc123", {}),
    ("chat", "bedrock/amazon.nova-pro-v1:0", {"tools": [_FN]}),
    ("chat", "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0", {"tools": [_FN]}),
]


@pytest.fixture
def bedrock_env(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "dummy")
    monkeypatch.setenv("AWS_REGION", "us-east-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("route,model,extra", _BEDROCK_HONEST)
async def test_honest_bedrock_requests_leave_unchanged(
    monkeypatch, wire, bedrock_env, route, model, extra
):
    params = {"model": model, **extra}
    if route == "chat":
        params = {**_EXTRA, **params}
    _method, path, _body = await _honest(monkeypatch, wire, route, params)
    assert path.endswith(("/converse", "/invoke")), path


@pytest.mark.asyncio
async def test_arn_path_matching_reads_the_encoded_segment(
    monkeypatch, wire, bedrock_env
):
    """The decoded ARN has a slash: the encoded path is what is matched."""
    model = f"bedrock/{_ACCOUNT}:application-inference-profile/abcdefghijkl"
    await _honest(monkeypatch, wire, "chat", {"model": model})
    assert "application-inference-profile/abcdefghijkl/converse" in wire[-1][1]


@pytest.mark.parametrize(
    "configured,body,refused",
    [
        (
            "bedrock/invoke/amazon.nova-pro-v1:0",
            {
                "messages": [],
                "toolConfig": {"tools": [{"systemTool": {"name": "nova_grounding"}}]},
            },
            True,
        ),
        (
            "bedrock/invoke/amazon.nova-pro-v1:0",
            {
                "messages": [],
                "inferenceConfig": {},
                "toolConfig": {
                    "tools": [
                        {"toolSpec": {"name": "run", "inputSchema": {"json": {}}}}
                    ]
                },
            },
            False,
        ),
        (
            "bedrock/invoke/amazon.nova-pro-v1:0",
            {
                "messages": [],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            },
            True,
        ),
        (
            "bedrock/invoke/cohere.command-r-plus-v1:0",
            {"prompt": "hi", "connectors": [{"id": "web-search"}]},
            True,
        ),
        (
            "bedrock/invoke/cohere.command-r-plus-v1:0",
            {"prompt": "hi", "documents": [{"url": "http://dummy.invalid"}]},
            True,
        ),
        (
            "bedrock/invoke/cohere.command-r-plus-v1:0",
            {"prompt": "hi", "tools": [_FN]},
            False,
        ),
        (
            "bedrock/invoke/meta.llama3-1-70b-instruct-v1:0",
            {"prompt": "hi", "tools": [_FN]},
            True,
        ),
        (
            "bedrock/invoke/anthropic.claude-sonnet-4-5-20250929-v1:0",
            {"messages": [], "inferenceConfig": {}},
            True,
        ),
        (
            "bedrock/invoke/anthropic.claude-sonnet-4-5-20250929-v1:0",
            {
                "messages": [],
                "max_tokens": 1,
                "anthropic_version": "bedrock-2023-05-31",
            },
            False,
        ),
        (
            "bedrock/invoke/openai.gpt-oss-120b-1:0",
            {"model": "m", "messages": [], "web_search_options": {}},
            True,
        ),
        (
            "bedrock/invoke/twelvelabs.pegasus-1-2-v1:0",
            {
                "inputPrompt": "hi",
                "mediaSource": {"s3Location": {"uri": "s3://dummy/x"}},
            },
            True,
        ),
        ("bedrock/invoke/stability.sd3-5-large-v1:0", {"prompt": "hi"}, True),
    ],
)
def test_invoke_body_is_checked_as_the_configured_family(
    deployed, monkeypatch, configured, body, refused
):
    monkeypatch.setenv(UPSTREAM, configured)
    model_id = configured.removeprefix("bedrock/invoke/")
    content = json.dumps(body).encode()
    guard = _guard(deployed)
    for operation in ("invoke", "invoke-with-response-stream"):
        reason = guard["outbound_refusal"](
            "POST", f"/model/{model_id}/{operation}", content
        )
        assert bool(reason) is refused, reason
    # The count-tokens wrapper carries the same body, base64 encoded.
    wrapped = {"input": {"invokeModel": {"body": base64.b64encode(content).decode()}}}
    reason = guard["outbound_refusal"](
        "POST", f"/model/{model_id}/count-tokens", json.dumps(wrapped).encode()
    )
    assert bool(reason) is refused, reason


@pytest.mark.parametrize(
    "operation", ["converse", "converse-stream", "invoke", "count-tokens"]
)
def test_encoded_arns_reach_their_validator(deployed, monkeypatch, operation):
    arn = (
        "arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3A"
        "application-inference-profile%2Fabcdefghijkl"
    )
    monkeypatch.setenv(
        UPSTREAM, f"bedrock/{_ACCOUNT}:inference-profile/us.anthropic.claude-x"
    )
    body: dict[str, Any] = {"messages": [], "max_tokens": 1}
    if operation.startswith("converse"):
        body = {"messages": [], "inferenceConfig": {"maxTokens": 1}}
    if operation == "count-tokens":
        body = {"input": {"converse": {"messages": []}}}
    path = f"/model/{arn}/{operation}"
    assert (
        _guard(deployed)["outbound_refusal"]("POST", path, json.dumps(body).encode())
        is None
    )


# --- C17: Anthropic client tools -----------------------------------------

# Every documented option of the pinned SDK's client tools,
# and the memory tool, which the caller also runs.
_OPTIONS = {
    "strict": True,
    "defer_loading": False,
    "input_examples": [{"command": "ls"}],
    "allowed_callers": ["direct"],
    "cache_control": {"type": "ephemeral"},
}
_CLIENT_TOOLS = [
    {"type": "bash_20241022", "name": "bash"},
    {"type": "bash_20250124", "name": "bash", **_OPTIONS},
    {"type": "text_editor_20241022", "name": "str_replace_editor"},
    {"type": "text_editor_20250124", "name": "str_replace_editor", "strict": True},
    {"type": "text_editor_20250429", "name": "str_replace_based_edit_tool"},
    {
        "type": "text_editor_20250728",
        "name": "str_replace_based_edit_tool",
        "max_characters": 1000,
        **_OPTIONS,
    },
    {
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": 8,
        "display_height_px": 8,
        "display_number": 1,
    },
    {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": 8,
        "display_height_px": 8,
    },
    {
        "type": "computer_20251124",
        "name": "computer",
        "display_width_px": 8,
        "display_height_px": 8,
        "enable_zoom": True,
        **_OPTIONS,
    },
    {"type": "memory_20250818", "name": "memory"},
    {"type": "memory_20250818", "name": "memory", **_OPTIONS},
]
_SERVER_TOOLS = [
    {"type": "web_search_20250305", "name": "web_search"},
    {"type": "web_search_20260209", "name": "web_search"},
    {"type": "web_fetch_20250910", "name": "web_fetch"},
    {"type": "web_fetch_20260309", "name": "web_fetch"},
    {"type": "code_execution_20250825", "name": "code_execution"},
    {"type": "code_execution_20260120", "name": "code_execution"},
    {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
    {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
    {"type": "advisor_20260301", "name": "advisor"},
    {"type": "mcp_toolset", "mcp_server_name": "d"},
]
_ANTHROPIC_UPSTREAMS = {
    "anthropic": {
        "model": "anthropic/claude-sonnet-4-5",
        "api_base": "http://dummy.invalid",
    },
    "azure_ai": {
        "model": "azure_ai/claude-sonnet-4-5",
        "api_base": "http://dummy.invalid/anthropic",
    },
    "bedrock_invoke": {"model": "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0"},
    "bedrock_profile": {
        "model": f"bedrock/{_ACCOUNT}:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    },
    "bedrock_converse": {
        "model": f"bedrock/{_ACCOUNT}:application-inference-profile/abcdefghijkl"
    },
    "vertex_claude": {"model": "vertex_ai/claude-sonnet-4-5", **_VERTEX},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream", sorted(_ANTHROPIC_UPSTREAMS))
@pytest.mark.parametrize("tool", _CLIENT_TOOLS, ids=lambda t: t["type"])
async def test_anthropic_client_tools_leave_unchanged(
    monkeypatch, wire, bedrock_env, upstream, tool
):
    params = {
        **_ANTHROPIC_UPSTREAMS[upstream],
        "tools": [tool, {"name": "Read", "input_schema": {"type": "object"}}],
    }
    # _honest is BenchFlow's part: the restricted request leaves exactly as the
    # unrestricted one does. What the translator makes of the tool is
    # LiteLLM's: Messages keeps its type, and Converse carries it as a plain
    # toolSpec or, as LiteLLM 1.103 does with the memory tool, leaves it out.
    _method, _path, sent = await _honest(monkeypatch, wire, "messages", params)
    converse_tools = sent.get("toolConfig", {}).get("tools", [])
    assert (
        tool["type"] in json.dumps(sent)
        or tool["name"] in json.dumps(sent)
        or (upstream == "bedrock_converse" and converse_tools)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream", sorted(_ANTHROPIC_UPSTREAMS))
@pytest.mark.parametrize("tool", _SERVER_TOOLS, ids=lambda t: t["type"])
async def test_anthropic_server_tools_still_never_leave(
    monkeypatch, wire, bedrock_env, upstream, tool
):
    params = {**_ANTHROPIC_UPSTREAMS[upstream], "tools": [tool]}
    await _send(monkeypatch, wire, "messages", params, restricted=False)
    carried = any(_hosted(b) for _, _, b in wire)
    error = await _send(monkeypatch, wire, "messages", params, restricted=True)
    assert not [b for _, _, b in wire if _hosted(b)], wire
    if carried:
        assert REFUSED in error, error


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "bash_20250124", "name": "bash", "allowed_domains": ["x"]},
        {
            "type": "computer_20250124",
            "name": "computer",
            "url": "http://dummy.invalid",
        },
        {
            "type": "text_editor_20250728",
            "name": "e",
            "server_url": "http://dummy.invalid",
        },
        {"type": "bash_29991231", "name": "bash"},
    ],
)
def test_client_tools_admit_only_their_own_fields_and_versions(deployed, tool):
    body = {"messages": [], "max_tokens": 1, "tools": [tool]}
    guard = _guard(deployed)
    assert guard["outbound_refusal"]("POST", "/v1/messages", json.dumps(body).encode())
    converse = {"messages": [], "additionalModelRequestFields": {"tools": [tool]}}
    assert guard["outbound_refusal"](
        "POST", "/model/m/converse", json.dumps(converse).encode()
    )


# --- the real gateway: pinned LiteLLM proxy, real HTTP, dummy upstream ----


@pytest.mark.asyncio
async def test_pinned_gateway_refuses_a_configured_search_model():
    from benchflow.providers import litellm_runtime as runtime
    from benchflow.providers.litellm_config import resolve_litellm_route
    from tests.fixtures.mock_token_logprobs_server import start_server

    upstream = start_server()
    env = {
        "BENCHFLOW_PROVIDER_BASE_URL": upstream.base_url + "/v1",
        "OPENAI_API_KEY": "dummy-upstream",
        DISALLOW: "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    proc = None
    try:
        route = resolve_litellm_route("openai/gpt-5-search-api", env)
        proc = await runtime._start_host_litellm(
            route=route,
            master_key="dummy-gateway",
            agent_env=env,
            environment="local",
            session_id="c14",
            agent_name="codex-acp",
        )
        async with httpx.AsyncClient(timeout=60) as client:
            for path in ("/v1/chat/completions", "/v1/responses"):
                payload: dict[str, Any] = {"model": route.model_alias}
                if path.endswith("responses"):
                    payload["input"] = "Find today's news."
                else:
                    payload["messages"] = [
                        {"role": "user", "content": "Find today's news."}
                    ]
                response = await client.post(
                    proc.base_url + path,
                    headers={"Authorization": "Bearer dummy-gateway"},
                    json=payload,
                )
                assert response.status_code == 400, response.text
                assert BY_ITSELF in response.text
        assert not upstream.requests
    finally:
        if proc:
            await proc.stop()
        upstream.shutdown()
