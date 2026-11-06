"""Translated requests regained hosted tools at 1b2f74c9.

The input boundary accepted local-looking envelopes (a function tool carrying
googleSearch, legacy ``functions``, a Messages tool named web_search) that the
pinned LiteLLM translators turn into provider-hosted capabilities. These tests
drive the real translators through the deployed boundary file and read what
would leave for the provider, for every provider family and entry route.
"""

import base64
import copy
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import httpx
import pytest
import pytest_asyncio

from benchflow.providers import litellm_runtime as runtime
from tests.test_litellm_hardening import _FakeSandbox

DISALLOW = "BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS"
REFUSED = "web tools are disabled for this run"

# Test-side detector, deliberately a denylist: it recognizes what the class
# looks like on the wire, so the allowlist under test is checked independently.
_HOSTED_KEYS = {
    "googleSearch",
    "google_search",
    "googleSearchRetrieval",
    "google_search_retrieval",
    "urlContext",
    "url_context",
    "codeExecution",
    "code_execution",
    "retrieval",
    "enterpriseWebSearch",
    "enterprise_web_search",
    "googleMaps",
    "google_maps",
    "fileSearch",
    "file_search",
    "computerUse",
    "computer_use",
    "web_search_options",
    "mcp_servers",
    "container",
    "systemTool",
}
# Anthropic's versioned bash/text_editor/computer/memory tools run on the
# caller's side and are admitted; OpenAI's
# computer_use_preview is not.
_HOSTED_TYPE = re.compile(
    r"web_search.*|web_fetch.*|code_execution.*|code_interpreter|mcp.*|file_search"
    r"|computer_use_preview|url_context|image_generation|tool_search_tool.*"
    r"|advisor_\d+"
)


# A function's JSON schema is the agent's data, not a capability.
_SCHEMA_FIELDS = {"parameters", "input_schema", "inputSchema", "parametersJsonSchema"}


def _hosted(value):
    if isinstance(value, dict):
        kind = value.get("type")
        if isinstance(kind, str) and _HOSTED_TYPE.fullmatch(kind):
            return True
        return any(
            k in _HOSTED_KEYS or (k not in _SCHEMA_FIELDS and _hosted(v))
            for k, v in value.items()
        )
    if isinstance(value, list):
        return any(_hosted(v) for v in value)
    return False


# Every hosted capability, as an agent can spell it on each entry route.
_GEMINI_KEYS = {
    "googleSearch": {},
    "google_search": {},
    "googleSearchRetrieval": {},
    "urlContext": {},
    "url_context": {},
    "codeExecution": {},
    "code_execution": {},
    "enterpriseWebSearch": {},
    "googleMaps": {},
    "computerUse": {"environment": "ENVIRONMENT_BROWSER"},
    "fileSearch": {"fileSearchStoreNames": ["dummy-store"]},
    "retrieval": {"vertexAiSearch": {"datastore": "dummy-store"}},
}
_TYPED = [
    {"type": "web_search"},
    {"type": "web_search_preview"},
    {"type": "web_search_20250305", "name": "web_search"},
    {"type": "web_fetch_20250910", "name": "web_fetch"},
    {"type": "url_context"},
    {"type": "code_execution_20250825", "name": "code_execution"},
    {"type": "code_interpreter", "container": {"type": "auto"}},
    {"type": "file_search", "vector_store_ids": ["vs_dummy"]},
    {"type": "mcp", "server_label": "d", "server_url": "http://dummy.invalid/mcp"},
    {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": 8,
        "display_height_px": 8,
    },
    {
        "type": "computer_use_preview",
        "display_width": 8,
        "display_height": 8,
        "environment": "browser",
    },
]
CHAT = [
    *({"tools": [{"type": "function", k: v}]} for k, v in _GEMINI_KEYS.items()),
    {
        "tools": [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "object"}},
                "googleSearch": {},
            }
        ]
    },
    *({"functions": [{k: v}]} for k, v in _GEMINI_KEYS.items()),
    {"functions": [{"name": "f", "parameters": {}, "googleSearch": {}}]},
    *({"tools": [tool]} for tool in _TYPED),
    {"web_search_options": {}},
]
RESPONSES = [
    *({"tools": [{"type": "function", k: v}]} for k, v in _GEMINI_KEYS.items()),
    {
        "tools": [
            {"type": "function", "name": "f", "parameters": {}, "googleSearch": {}}
        ]
    },
    *({"tools": [tool]} for tool in _TYPED),
]
_SCHEMA = {"type": "object"}
MESSAGES = [
    {"tools": [{"type": "custom", "name": "web_search", "input_schema": _SCHEMA}]},
    {"tools": [{"name": "web_search", "input_schema": _SCHEMA}]},
    {"tools": [{"name": "web_fetch", "input_schema": _SCHEMA}]},
    *(
        {"tools": [{"name": "f", "input_schema": _SCHEMA, k: v}]}
        for k, v in _GEMINI_KEYS.items()
    ),
    *({"tools": [tool]} for tool in _TYPED),
    {"tools": [{"type": "bash_20250124", "name": "bash"}]},
    {
        "tools": [
            {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
        ]
    },
    {"mcp_servers": [{"type": "url", "url": "http://dummy.invalid/mcp", "name": "d"}]},
]
_VERTEX = {"vertex_project": "dummy-project", "vertex_location": "us-central1"}
PROVIDERS = {
    "openai": {"model": "openai/gpt-4.1", "api_base": "http://dummy.invalid/v1"},
    "azure": {
        "model": "azure/gpt-4.1",
        "api_base": "http://dummy.invalid",
        "api_version": "2025-04-01-preview",
    },
    "deepseek": {"model": "deepseek/deepseek-chat", "api_base": "http://dummy.invalid"},
    "anthropic": {
        "model": "anthropic/claude-sonnet-4-5",
        "api_base": "http://dummy.invalid",
    },
    "azure_ai": {
        "model": "azure_ai/claude-sonnet-4-5",
        "api_base": "http://dummy.invalid/anthropic",
    },
    "bedrock": {"model": "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0"},
    "bedrock_nova": {"model": "bedrock/amazon.nova-pro-v1:0"},
    "gemini": {"model": "gemini/gemini-2.5-flash", "api_base": "http://dummy.invalid"},
    "vertex": {"model": "vertex_ai/gemini-2.5-flash", **_VERTEX},
    "vertex_claude": {"model": "vertex_ai/claude-sonnet-4-5", **_VERTEX},
}
_HONEST_FN = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "agent-executed",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}
HONEST = {
    "chat": [
        {"tools": [_HONEST_FN], "tool_choice": "auto"},
        {
            "tools": [_HONEST_FN],
            "tool_choice": {"type": "function", "function": {"name": "web_search"}},
        },
        {"functions": [{"name": "run", "parameters": {"type": "object"}}]},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
            "tools": [_HONEST_FN],
            "temperature": 0.2,
            "max_tokens": 32,
        },
    ],
    "responses": [
        {
            "tools": [
                {"type": "function", "name": "shell", "parameters": {"type": "object"}}
            ]
        },
        {
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "parameters": {},
                    "strict": False,
                },
                {"type": "custom", "name": "apply_patch", "description": "patch"},
            ],
            "reasoning": {"effort": "low"},
        },
    ],
    "messages": [
        {"tools": [{"name": "Bash", "input_schema": _SCHEMA}]},
        {
            # Claude Code's spelling. The exact name web_search is promoted to
            # hosted search by the pinned Messages adapter (MESSAGES above).
            "tools": [
                {
                    "name": "WebSearch",
                    "description": "agent-executed",
                    "input_schema": _SCHEMA,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "system": [
                {"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}
            ],
            "tool_choice": {"type": "auto"},
        },
        {"tools": [{"type": "custom", "name": "Read", "input_schema": _SCHEMA}]},
    ],
}


@pytest_asyncio.fixture(params=["host", "sandbox"])
async def deployed(request, tmp_path):
    # The exact standalone file both proxy placements import at start-up.
    if request.param == "host":
        runtime._write_runtime_files(tmp_path, config={})
        return (tmp_path / "benchflow_litellm_request_guard.py").read_text()
    sandbox = _FakeSandbox()
    paths = await runtime._upload_runtime_files_to_sandbox(
        sandbox, runtime_dir="/tmp/dummy-runtime", config={}
    )
    return sandbox.uploaded[paths["request_guard"]]


class _Stop(Exception):
    pass


@pytest.fixture
def wire(monkeypatch, deployed):
    """Install the deployed guard over a recorder that stands for the network.

    Anything the recorder sees would have left for the provider. It never
    connects anywhere.
    """
    import aiohttp
    import litellm
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
    from litellm.proxy import proxy_server

    sent = []

    def record(request):
        try:
            body = json.loads(request.read())
        except ValueError:
            body = None
        sent.append((request.method, request.url.path, body))
        if request.method == "GET":
            # A translator's own lookup (a chat template): answer, keep going.
            return httpx.Response(404, request=request)
        raise _Stop

    async def arecord(self, request, **kwargs):
        return record(request)

    monkeypatch.setattr(
        httpx.Client, "send", lambda self, request, **kw: record(request)
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", arecord)
    monkeypatch.setattr(
        aiohttp.ClientSession, "_request", aiohttp.ClientSession._request
    )
    # install() is what sitecustomize runs; keep the vendor app untouched.
    monkeypatch.setattr(proxy_server.app, "add_middleware", lambda *a, **k: None)
    # The gateway config's settings (litellm_config.litellm_proxy_config).
    monkeypatch.setattr(
        litellm, "use_chat_completions_url_for_anthropic_messages", True
    )
    monkeypatch.setattr(litellm, "drop_params", True)
    monkeypatch.setattr(
        VertexBase, "_ensure_access_token", lambda *a, **k: ("dummy", "dummy-project")
    )

    async def atoken(*args, **kwargs):
        return ("dummy", "dummy-project")

    monkeypatch.setattr(VertexBase, "_ensure_access_token_async", atoken)
    for key, value in {
        "AWS_BEARER_TOKEN_BEDROCK": "dummy",
        "AWS_REGION": "us-east-1",
        "AWS_EC2_METADATA_DISABLED": "true",
        "NO_GCE_CHECK": "True",
        "GCE_METADATA_HOST": "127.0.0.1:9",
        "GCE_METADATA_IP": "127.0.0.1:9",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }.items():
        monkeypatch.setenv(key, value)
    namespace = {}
    exec(compile(deployed, "request_guard.py", "exec"), namespace)
    namespace["install"]()
    return sent


async def _translate(route, provider, body):
    import litellm

    # Translators mutate their input; every call gets its own copy.
    params: dict[str, Any] = {
        **PROVIDERS[provider],
        "api_key": "dummy-upstream",
        **copy.deepcopy(body),
    }
    try:
        if route == "chat":
            params.setdefault("messages", [{"role": "user", "content": "hi"}])
            await litellm.acompletion(**params)
        elif route == "responses":
            await litellm.aresponses(input="hi", **params)
        else:
            params.setdefault("messages", [{"role": "user", "content": "hi"}])
            create = cast(Any, litellm.anthropic.messages.acreate)
            await create(max_tokens=16, **params)
    except Exception as exc:  # the recorder's stop, or the boundary's refusal
        return str(exc)
    return ""


async def _both_arms(monkeypatch, wire, route, provider, body):
    monkeypatch.setenv(DISALLOW, "0")
    await _translate(route, provider, body)
    carried = any(_hosted(b) for _, _, b in wire)
    wire.clear()
    monkeypatch.setenv(DISALLOW, "1")
    error = await _translate(route, provider, body)
    return carried, error


_MATRIX = [
    *(("chat", i) for i in range(len(CHAT))),
    *(("responses", i) for i in range(len(RESPONSES))),
    *(("messages", i) for i in range(len(MESSAGES))),
]
_BODIES = {"chat": CHAT, "responses": RESPONSES, "messages": MESSAGES}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(PROVIDERS))
@pytest.mark.parametrize("route,index", _MATRIX)
async def test_translated_hosted_capability_never_leaves(
    monkeypatch, wire, provider, route, index
):
    """Whatever the translator makes of it, no hosted capability reaches the wire."""
    body = _BODIES[route][index]
    carried, error = await _both_arms(monkeypatch, wire, route, provider, body)
    assert not [b for _, _, b in wire if _hosted(b)], (route, provider, body, wire)
    if carried:
        assert REFUSED in error, (route, provider, body, error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route,provider,body",
    [
        # The three reported openings, on the reported provider.
        ("chat", "gemini", {"tools": [{"type": "function", "googleSearch": {}}]}),
        ("chat", "gemini", {"functions": [{"googleSearch": {}}]}),
        ("chat", "gemini", {"functions": [{"urlContext": {}}]}),
        ("chat", "vertex", {"functions": [{"googleSearch": {}}]}),
    ],
)
async def test_reported_openings_are_real_translations_and_refused(
    monkeypatch, wire, route, provider, body
):
    carried, error = await _both_arms(monkeypatch, wire, route, provider, body)
    assert carried, "pinned LiteLLM no longer emits this capability"
    assert not wire, wire
    assert REFUSED in error


_MESSAGES_WEB_SEARCH = [
    (
        "messages",
        "gemini",
        {"tools": [{"type": "custom", "name": "web_search", "input_schema": _SCHEMA}]},
    ),
    (
        "messages",
        "gemini",
        {"tools": [{"name": "web_search", "input_schema": _SCHEMA}]},
    ),
    (
        "messages",
        "vertex",
        {"tools": [{"name": "web_search", "input_schema": _SCHEMA}]},
    ),
    (
        "messages",
        "openai",
        {"tools": [{"name": "web_search", "input_schema": _SCHEMA}]},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("route,provider,body", _MESSAGES_WEB_SEARCH)
async def test_reported_messages_openings_are_closed_upstream(
    monkeypatch, wire, route, provider, body
):
    """LiteLLM 1.91.5 turned a Messages ``web_search`` tool into a hosted
    search capability (refused here); 1.103 sends it on as a plain function
    tool, so nothing hosted is left to refuse and nothing hosted leaves."""
    carried, _ = await _both_arms(monkeypatch, wire, route, provider, body)
    assert not carried
    assert not [b for _, _, b in wire if _hosted(b)], wire


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(PROVIDERS))
@pytest.mark.parametrize(
    "route,index",
    [(route, i) for route, cases in HONEST.items() for i in range(len(cases))],
)
async def test_restricted_function_calling_still_leaves(
    monkeypatch, wire, provider, route, index
):
    body = HONEST[route][index]
    monkeypatch.setenv(DISALLOW, "0")
    await _translate(route, provider, body)
    if not wire:
        pytest.skip("pinned LiteLLM cannot send this pair even unrestricted")
    unrestricted = wire[-1]
    wire.clear()
    monkeypatch.setenv(DISALLOW, "1")
    error = await _translate(route, provider, body)
    assert REFUSED not in error, error
    # The same request leaves, unchanged, as it does without restrictions.
    assert wire and wire[-1] == unrestricted, (route, provider, error)
    assert not _hosted(unrestricted[2])


def _guard(deployed):
    namespace = {}
    exec(compile(deployed, "request_guard.py", "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "path,body",
    [
        ("/v1/chat/completions", {"messages": [], "frontier_tool": {}}),
        (
            "/v1/chat/completions",
            {
                "messages": [],
                "tools": [{"type": "function", "function": {"name": "f"}, "x": 1}],
            },
        ),
        (
            "/v1/chat/completions",
            {
                "messages": [],
                "tools": [
                    {"type": "function", "function": {"name": "f", "googleSearch": {}}}
                ],
            },
        ),
        (
            "/v1/chat/completions",
            {"messages": [], "tool_choice": {"type": "web_search_preview"}},
        ),
        ("/v1/responses", {"input": "x", "tool_choice": {"type": "file_search"}}),
        ("/v1/responses", {"input": [{"type": "item_reference", "id": "dummy"}]}),
        (
            "/v1/responses",
            {
                "input": [
                    {
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": [{"type": "web_search"}],
                    }
                ]
            },
        ),
        ("/v1/responses", {"input": "x", "previous_response_id": "resp_dummy"}),
        (
            "/v1/responses",
            {
                "input": "x",
                "tools": [
                    {"type": "namespace", "name": "n", "tools": [{"type": "mcp"}]}
                ],
            },
        ),
        ("/v1/responses", {"input": "x", "tools": [{"type": "tool_search"}]}),
        ("/v1/messages", {"messages": [], "container": "dummy"}),
        (
            "/v1/messages",
            {
                "messages": [],
                "tools": [{"name": "f", "input_schema": {}, "type": "custom", "x": 1}],
            },
        ),
        ("/v1/messages", {"messages": [], "tool_choice": {"type": "web_search"}}),
        ("/model/dummy/invoke", {"messages": [], "inferenceConfig": {}}),
        (
            "/model/dummy/converse",
            {
                "messages": [],
                "toolConfig": {"tools": [{"systemTool": {"name": "nova_grounding"}}]},
            },
        ),
        (
            "/model/dummy/converse",
            {
                "messages": [],
                "additionalModelRequestFields": {
                    "tools": [{"type": "web_search_20250305"}]
                },
            },
        ),
        (
            "/v1beta/models/m:generateContent",
            {"contents": [], "toolConfig": {"retrievalConfig": {}}},
        ),
        (
            "/v1beta/models/m:generateContent",
            {"contents": [], "cachedContent": "cachedContents/foreign"},
        ),
        (
            "/v1beta/models/m:countTokens",
            {"generateContentRequest": {"tools": [{"googleSearch": {}}]}},
        ),
        (
            "/v1beta/models/m:generateContent",
            {
                "contents": [],
                "tools": [
                    {"functionDeclarations": [{"name": "f", "googleSearch": {}}]}
                ],
            },
        ),
        ("/v1beta/cachedContents", {"contents": [], "tools": [{"urlContext": {}}]}),
        ("/v1/embeddings", {"input": "x"}),
        ("/v1/chat/completions", [{"messages": []}]),
        ("/v1/chat/completions", b"not json"),
    ],
)
def test_outbound_allowlist_refuses_unknown_wire_shapes(deployed, path, body):
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    assert _guard(deployed)["outbound_refusal"]("POST", path, content)


def _invoke(body):
    return base64.b64encode(json.dumps(body).encode()).decode()


_SPEC = {"toolSpec": {"name": "run", "inputSchema": {"json": {"type": "object"}}}}
_BEDROCK_COUNT = "/model/anthropic.claude-sonnet-4-5-20250929-v1:0/count-tokens"


@pytest.mark.parametrize(
    "path,body,refused",
    [
        (
            _BEDROCK_COUNT,
            {
                "input": {
                    "converse": {
                        "messages": [],
                        "toolConfig": {
                            "tools": [_SPEC, {"cachePoint": {"type": "default"}}]
                        },
                    }
                }
            },
            False,
        ),
        (
            _BEDROCK_COUNT,
            {
                "input": {
                    "converse": {
                        "messages": [],
                        "toolConfig": {
                            "tools": [{"systemTool": {"name": "nova_grounding"}}]
                        },
                    }
                }
            },
            True,
        ),
        (
            _BEDROCK_COUNT,
            {
                "input": {
                    "invokeModel": {
                        "body": _invoke(
                            {
                                "messages": [],
                                "max_tokens": 1,
                                "anthropic_version": "bedrock-2023-05-31",
                                "tools": [{"name": "run", "input_schema": {}}],
                            }
                        )
                    }
                }
            },
            False,
        ),
        (
            _BEDROCK_COUNT,
            {
                "input": {
                    "invokeModel": {
                        "body": _invoke(
                            {
                                "messages": [],
                                "tools": [
                                    {
                                        "type": "web_search_20250305",
                                        "name": "web_search",
                                    }
                                ],
                            }
                        )
                    }
                }
            },
            True,
        ),
        (
            _BEDROCK_COUNT,
            {
                "input": {
                    "invokeModel": {
                        "body": _invoke({"messages": [], "mcp_servers": []})
                    }
                }
            },
            True,
        ),
        (_BEDROCK_COUNT, {"input": {"invokeModel": {"body": "not base64 json"}}}, True),
        (_BEDROCK_COUNT, {"input": {"converse": {}, "invokeModel": {}}}, True),
        (
            "/v1beta/cachedContents",
            {
                "model": "models/m",
                "contents": [],
                "displayName": "k",
                "tools": [{"functionDeclarations": [{"name": "run"}]}],
            },
            False,
        ),
        (
            "/v1/projects/p/locations/l/publishers/anthropic/models/claude:count-tokens:rawPredict",
            {
                "model": "claude",
                "messages": [],
                "tools": [{"name": "run", "input_schema": {}}],
            },
            False,
        ),
        (
            "/v1/projects/p/locations/l/publishers/anthropic/models/claude:count-tokens:rawPredict",
            {
                "model": "claude",
                "messages": [],
                "tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}],
            },
            True,
        ),
    ],
)
def test_wrapped_and_cache_requests_obey_the_same_allowlist(
    deployed, path, body, refused
):
    reason = _guard(deployed)["outbound_refusal"](
        "POST", path, json.dumps(body).encode()
    )
    assert bool(reason) is refused, reason


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/images/dummy-marker.png"),
        ("GET", "/v1/files/file-dummy/content"),
        ("PUT", "/v1/chat/completions"),
        ("DELETE", "/v1beta/cachedContents/dummy"),
    ],
)
def test_proxy_side_fetches_are_not_inference(deployed, method, path):
    # A translator downloading a URL (C11's consumer) is refused too.
    assert _guard(deployed)["outbound_refusal"](method, path, b"")


def test_own_gemini_cache_is_the_only_cache_reference(deployed):
    guard = _guard(deployed)
    body = json.dumps({"contents": [], "cachedContent": "cachedContents/own"}).encode()
    path = "/v1beta/models/gemini-2.5-flash:generateContent"
    assert guard["outbound_refusal"]("POST", path, body)
    request = httpx.Request("POST", "http://dummy.invalid/v1beta/cachedContents")
    guard["_record_cache"](
        request,
        httpx.Response(200, json={"name": "cachedContents/own"}, request=request),
    )
    assert guard["outbound_refusal"]("POST", path, body) is None
    assert guard["outbound_refusal"]("GET", "/v1beta/cachedContents", b"") is None


@pytest.mark.asyncio
async def test_aiohttp_outside_checked_client_is_refused(monkeypatch, wire):
    import aiohttp

    monkeypatch.setenv(DISALLOW, "1")
    async with aiohttp.ClientSession() as session:
        with pytest.raises(aiohttp.ClientError, match=REFUSED):
            await session.post("http://dummy.invalid/v1/chat/completions", json={})


@pytest.mark.asyncio
async def test_unrestricted_runs_keep_hosted_tools(monkeypatch, wire):
    monkeypatch.setenv(DISALLOW, "0")
    await _translate(
        "chat", "gemini", {"tools": [{"type": "function", "googleSearch": {}}]}
    )
    assert wire and wire[-1][2]["tools"] == [{"googleSearch": {}}]


# --- the real gateway: pinned LiteLLM proxy, real HTTP, dummy loopback upstream


class _GeminiUpstream(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), _GeminiHandler)
        self.requests = []
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class _GeminiHandler(BaseHTTPRequestHandler):
    server: _GeminiUpstream

    def log_message(self, format, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append({"path": self.path, "body": body})
        if ":countTokens" in self.path:
            payload = {"totalTokens": 1}
        else:
            payload = {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "ok"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
            }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _gateway_requests(model, local):
    """One request per entry route, all carrying ``local`` or its spelling."""
    fn = {"name": "run", "parameters": {"type": "object"}}
    if local:
        return [
            ("/v1/chat/completions", {"tools": [{"type": "function", "function": fn}]}),
            ("/v1/chat/completions", {"functions": [fn]}),
            ("/v1/responses", {"tools": [{"type": "function", **fn}]}),
            ("/messages", {"tools": [{"name": "run", "input_schema": _SCHEMA}]}),
            ("/v1/messages", {"tools": [{"name": "run", "input_schema": _SCHEMA}]}),
        ]
    ws = {"type": "custom", "name": "web_search", "input_schema": _SCHEMA}
    return [
        ("/v1/chat/completions", {"tools": [{"type": "function", "googleSearch": {}}]}),
        ("/v1/chat/completions", {"tools": [{"type": "function", "urlContext": {}}]}),
        ("/v1/chat/completions", {"functions": [{"googleSearch": {}}]}),
        ("/v1/chat/completions", {"functions": [{"urlContext": {}}]}),
        ("/v1/chat/completions", {"functions": [{"codeExecution": {}}]}),
        ("/v1/responses", {"tools": [{"type": "function", "googleSearch": {}}]}),
        ("/v1/responses", {"tools": [{"type": "function", "urlContext": {}}]}),
        ("/messages", {"tools": [ws]}),
        ("/v1/messages", {"tools": [ws]}),
        ("/messages", {"tools": [{"name": "web_search", "input_schema": _SCHEMA}]}),
        ("/v1/messages", {"tools": [{"name": "web_search", "input_schema": _SCHEMA}]}),
    ]


def _reported(family):
    """The reported openings, which pass the input boundary on this route."""
    # LiteLLM 1.103 sends a Messages ``web_search`` tool on as a plain function
    # tool (test_reported_messages_openings_are_closed_upstream): only the
    # chat spellings still become a hosted capability to refuse.
    reported: list[tuple[str, dict[str, Any]]] = [
        ("/v1/chat/completions", {"tools": [{"type": "function", "googleSearch": {}}]}),
    ]
    if family == "gemini":
        reported += [
            ("/v1/chat/completions", {"functions": [{"googleSearch": {}}]}),
            ("/v1/chat/completions", {"functions": [{"urlContext": {}}]}),
        ]
    return reported


async def _gateway_post(client, proc, model, path, body):
    payload = {"model": model, **body}
    if path.endswith("/responses"):
        payload["input"] = "dummy prompt"
    else:
        payload["messages"] = [{"role": "user", "content": "dummy prompt"}]
        if path.endswith("/messages"):
            payload["max_tokens"] = 16
    return await client.post(
        proc.base_url + path,
        headers={"Authorization": "Bearer dummy-gateway"},
        json=payload,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["gemini", "openai"])
async def test_pinned_gateway_refuses_translated_hosted_tools(family):
    """Pinned LiteLLM, real HTTP and a dummy loopback provider, no model call."""
    from benchflow.providers.litellm_config import resolve_litellm_route
    from tests.fixtures.mock_token_logprobs_server import start_server

    upstream = _GeminiUpstream() if family == "gemini" else start_server()
    env = {
        "BENCHFLOW_PROVIDER_BASE_URL": upstream.base_url
        + ("" if family == "gemini" else "/v1"),
        "BENCHFLOW_PROVIDER_API_KEY": "dummy-upstream",
        "GEMINI_API_KEY": "dummy-upstream",
        "OPENAI_API_KEY": "dummy-upstream",
        DISALLOW: "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    model = "gemini/gemini-2.5-flash" if family == "gemini" else "openai/gpt-4.1"
    route = resolve_litellm_route(model, env)
    proc = None
    try:
        proc = await runtime._start_host_litellm(
            route=route,
            master_key="dummy-gateway",
            agent_env=env,
            environment="local",
            session_id="c13",
            agent_name="claude-agent-acp",
        )
        async with httpx.AsyncClient(timeout=60) as client:
            for path, body in _gateway_requests(model, local=False):
                response = await _gateway_post(
                    client, proc, route.model_alias, path, body
                )
                # A translator may also drop a spelling; then nothing hosted
                # is left to refuse and the plain request may succeed.
                assert not [r for r in upstream.requests if _hosted(r["body"])]
                if (path, body) in _reported(family) and path == "/messages":
                    # The input boundary admits it; the pinned LiteLLM
                    # (1.103.3) serves no /messages route, so nothing is sent.
                    assert response.status_code == 404 or REFUSED in response.text
                elif (path, body) in _reported(family):
                    assert response.status_code == 400, (path, response.text)
                    assert REFUSED in response.text, (path, response.text)
            for path, body in _gateway_requests(model, local=True):
                if path == "/messages":
                    continue
                before = len(upstream.requests)
                response = await _gateway_post(
                    client, proc, route.model_alias, path, body
                )
                assert response.status_code == 200, (path, body, response.text)
                assert len(upstream.requests) == before + 1, path
                assert "run" in json.dumps(upstream.requests[-1]["body"])
    finally:
        if proc:
            await proc.stop()
        upstream.shutdown()
