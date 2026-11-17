"""The effort asked for is the effort sent upstream, on every route.

Yuwen Lu's run, ``bench run -a claude -m openrouter/anthropic/claude-opus-5.5
-e max``: Claude Code sent ``output_config.effort: max``, and every call the
gateway sent upstream carried ``reasoning.effort: high``. LiteLLM 1.103.3's
Messages-to-Responses adapter passes the effort through
``normalize_reasoning_effort_value``, which lowers a level its model map cannot
confirm, to ``high`` for a model the map does not know. BenchFlow's own patch
also lowered ``xhigh``/``max`` to ``high`` on Bedrock's Claude 4.x ids, for a
LiteLLM 1.88.0rc1 limit the pinned LiteLLM no longer has.

On Bedrock, LiteLLM also capped the effort at its model map's ceiling (Opus
4.6's ``xhigh`` went out as ``max``), and its Converse mapping gave Opus 4.5 a
thinking budget and no effort at all.

Each case runs the proxy's own route (``resolve_litellm_route``), BenchFlow's
sitecustomize patch module and LiteLLM's Router against a local stub that
records the body it receives. No model is called.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import pytest

LEVELS = ("low", "medium", "high", "xhigh", "max")
_TOOLS = [
    {
        "name": "Bash",
        "description": "Run a command",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
        },
    }
]
_ANTHROPIC_REPLY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 2},
}
_CONVERSE_REPLY = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
    "metrics": {"latencyMs": 1},
}
_RESPONSES_REPLY = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "status": "completed",
    "model": "anthropic/claude-opus-5.5",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


class _Upstream(BaseHTTPRequestHandler):
    """Records each body; answers as the Responses API, Anthropic's Messages
    API, Bedrock's Converse, or Bedrock's InvokeModel (which returns a Messages
    body)."""

    seen: ClassVar[list[dict[str, Any]]] = []

    def log_message(self, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append({"path": self.path, "body": body})
        if self.path.endswith("/responses"):
            reply = _RESPONSES_REPLY
        elif self.path.endswith("/converse"):
            reply = _CONVERSE_REPLY
        else:
            reply = _ANTHROPIC_REPLY
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _sent_effort(route: str, body: dict[str, Any]) -> Any:
    if route == "openrouter":
        return (body.get("reasoning") or {}).get("effort")
    return (body.get("output_config") or {}).get("effort")


def _route_params(route: str, model: str, level: str, upstream: str, monkeypatch):
    from benchflow.providers.litellm_config import resolve_litellm_route

    if route == "openrouter":
        params = dict(
            resolve_litellm_route(model, {"OPENROUTER_API_KEY": "fake"}).litellm_params
        )
        params.update(api_base=f"{upstream}/api/v1", api_key="fake-not-a-key")
    elif route == "anthropic":
        params = dict(
            resolve_litellm_route(model, {"ANTHROPIC_API_KEY": "fake"}).litellm_params
        )
        params.update(api_base=upstream, api_key="fake-not-a-key")
    else:
        # The run's effort, as the run hands it to the Bedrock route
        # (BENCHFLOW_BEDROCK_THINKING_EFFORT in the route's and the proxy's env).
        env = {
            "AWS_BEARER_TOKEN_BEDROCK": "fake-not-a-key",
            "AWS_REGION": "us-east-1",
            "BENCHFLOW_BEDROCK_THINKING_EFFORT": level,
        }
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
        params = dict(resolve_litellm_route(model, env).litellm_params)
        params["aws_bedrock_runtime_endpoint"] = upstream
    return params


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    ("route", "model"),
    [
        ("openrouter", "openrouter/anthropic/claude-opus-5.5"),
        ("anthropic", "anthropic/claude-opus-5-5"),
        ("bedrock", "aws-bedrock/us.anthropic.claude-opus-5-5"),
        ("bedrock", "aws-bedrock/us.anthropic.claude-opus-4-8"),
    ],
)
def test_the_effort_claude_code_asks_for_is_the_effort_sent_upstream(
    upstream, monkeypatch, route, model, level
):
    import litellm

    import benchflow.providers.litellm_bedrock_patch as patch

    patch.register_claude_effort_capabilities([model])
    params = _route_params(route, model, level, upstream, monkeypatch)
    router = litellm.Router(model_list=[{"model_name": "m", "litellm_params": params}])
    _Upstream.seen.clear()
    # Claude Code's /v1/messages shape at --effort <level>.
    asyncio.run(
        router.aanthropic_messages(
            model="m",
            messages=[{"role": "user", "content": "Write hello.txt."}],
            max_tokens=1024,
            thinking={"type": "adaptive"},
            output_config={"effort": level},
            tools=_TOOLS,
        )
    )
    assert _Upstream.seen, "nothing reached the upstream"
    sent = _Upstream.seen[-1]
    assert _sent_effort(route, sent["body"]) == level, (sent["path"], sent["body"])


def test_the_patch_keeps_every_level_and_the_preflight_checks_it():
    """The adapters' normalizer returns what it is given, and the startup
    preflight refuses a proxy where it is LiteLLM's own again."""
    from litellm.llms.anthropic.experimental_pass_through import utils

    import benchflow.providers.litellm_bedrock_patch as patch
    from benchflow.providers.litellm_bedrock_preflight import (
        BEDROCK_PATCH_PREFLIGHT_SOURCE,
    )

    assert patch  # importing it applies the proxy's patches
    for level in LEVELS:
        assert (
            utils.normalize_reasoning_effort_value(level, "anthropic/x", "openai")
            == level
        )
    assert utils.normalize_reasoning_effort_value.__benchflow_bedrock_patch__
    assert "effort pass-through inactive" in BEDROCK_PATCH_PREFLIGHT_SOURCE


def test_the_current_claude_models_are_registered_on_every_route():
    """Opus 5.5, Sonnet 5.5, Haiku 5.5 and Fable 5.2 (Anthropic's model
    reference, 2026-11-03), each taking low to max, as every route names
    them."""
    import benchflow.providers.litellm_bedrock_patch as patch

    assert set(patch.CLAUDE_EFFORT_LEVELS) == {
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "claude-fable-5-2",
    }
    assert set(patch.CLAUDE_EFFORT_LEVELS.values()) == {LEVELS}
    # Registered beside every proxy route's own models (its sitecustomize).
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    patch.register_claude_effort_capabilities([])
    for model in patch.CLAUDE_EFFORT_LEVELS:
        assert AnthropicModelInfo._get_model_capability(
            model, "supports_max_reasoning_effort"
        )
    models = patch.claude_route_models()
    for shape in (
        "anthropic/claude-opus-5-5",
        "bedrock/us.anthropic.claude-sonnet-5-5",
        "vertex_ai/claude-haiku-5-5",
        "openrouter/anthropic/claude-fable-5.2",
    ):
        assert shape in models
    # OpenRouter's id is kept whole: its route is priced from OpenRouter's
    # bill, so no bare map entry is made for it.
    assert patch._effort_capability_keys("openrouter/anthropic/claude-opus-5.5") == (
        "openrouter",
        ["openrouter/anthropic/claude-opus-5.5"],
    )


# Anthropic's model reference (2026-11-13): Opus 4.6 takes low, medium, high
# and max (no xhigh), with adaptive thinking; Opus 4.5 low, medium and high,
# with a thinking budget.
_BEDROCK_4X_LEVELS = {
    "aws-bedrock/us.anthropic.claude-opus-4-6-v1": ("low", "medium", "high", "max"),
    "aws-bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0": ("low", "medium", "high"),
}
_OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    }
]


def _bedrock_effort(call: dict[str, Any]) -> Any:
    body = call["body"]
    if "/converse" in call["path"]:
        body = body.get("additionalModelRequestFields") or {}
    return (body.get("output_config") or {}).get("effort")


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("shape", ["converse", "invoke"])
@pytest.mark.parametrize("model", list(_BEDROCK_4X_LEVELS))
def test_bedrock_opus_4_6_and_4_5_get_the_level_asked_or_fail(
    upstream, monkeypatch, model, shape, level
):
    """LiteLLM capped a Bedrock Claude effort at its map's ceiling (Opus 4.6's
    xhigh went out as max), and its Converse mapping gave Opus 4.5 a thinking
    budget and no effort. A level the model takes is sent as asked; any other
    is never sent lower: the call fails, or the upstream gets it and refuses it.
    Converse is a chat agent's call (OpenHands at LLM_REASONING_EFFORT=<level>),
    InvokeModel Claude Code's."""
    import litellm

    import benchflow.providers.litellm_bedrock_patch as patch

    patch.register_claude_effort_capabilities([model])
    params = _route_params("bedrock", model, level, upstream, monkeypatch)
    router = litellm.Router(model_list=[{"model_name": "m", "litellm_params": params}])
    _Upstream.seen.clear()
    messages = [{"role": "user", "content": "Write hello.txt."}]
    failed = False
    try:
        if shape == "converse":
            asyncio.run(
                router.acompletion(
                    model="m",
                    messages=messages,
                    max_tokens=8192,
                    reasoning_effort=level,
                    tools=_OPENAI_TOOLS,
                )
            )
        else:
            # Claude Code sends adaptive thinking to every model.
            thinking = {"type": "adaptive"}
            asyncio.run(
                router.aanthropic_messages(
                    model="m",
                    messages=messages,
                    max_tokens=8192,
                    thinking=thinking,
                    output_config={"effort": level},
                    tools=_TOOLS,
                )
            )
    except Exception:
        failed = True
    sent = [_bedrock_effort(call) for call in _Upstream.seen]
    if level in _BEDROCK_4X_LEVELS[model]:
        assert not failed and sent == [level], _Upstream.seen
    else:
        assert sent in ([], [level]), _Upstream.seen
        assert failed or sent == [level], _Upstream.seen


def test_the_bedrock_effort_ceiling_is_off_and_the_preflight_checks_it():
    import importlib

    import benchflow.providers.litellm_bedrock_patch as patch
    from benchflow.providers.litellm_bedrock_preflight import (
        BEDROCK_PATCH_PREFLIGHT_SOURCE,
    )

    for name in patch._BEDROCK_EFFORT_CEILING_MODULES:
        ceiling = importlib.import_module(
            name
        ).normalize_bedrock_opus_output_config_effort
        assert ceiling.__benchflow_bedrock_patch__, name
        output_config = {"effort": "xhigh"}
        ceiling("us.anthropic.claude-opus-4-6-v1", output_config)
        assert output_config == {"effort": "xhigh"}
    assert "bedrock effort ceiling active" in BEDROCK_PATCH_PREFLIGHT_SOURCE
    assert "effort level gate inactive" in BEDROCK_PATCH_PREFLIGHT_SOURCE


def test_claude_4x_ids_missing_from_the_map_get_their_documented_levels(monkeypatch):
    """A Bedrock id LiteLLM's map lacks is registered with what its model takes:
    effort kept, and xhigh (Opus 4.6) or xhigh and max (Opus 4.5) still refused."""
    import litellm
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    import benchflow.providers.litellm_bedrock_patch as patch

    opus_46 = "apac.anthropic.claude-opus-4-6-v9"
    opus_45 = "apac.anthropic.claude-opus-4-5-20991231-v9:0"
    sonnet_45 = "apac.anthropic.claude-sonnet-4-5-20991231-v9:0"
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    patch.register_claude_effort_capabilities(
        [f"bedrock/{opus_46}", f"bedrock/{opus_45}", f"bedrock/{sonnet_45}"]
    )
    assert AnthropicConfig._model_supports_effort_param(opus_46, "bedrock")
    assert AnthropicConfig._supports_effort_level(opus_46, "max", "bedrock")
    assert not AnthropicConfig._supports_effort_level(opus_46, "xhigh", "bedrock")
    assert AnthropicConfig._model_supports_effort_param(opus_45, "bedrock")
    for level in ("xhigh", "max"):
        assert not AnthropicConfig._supports_effort_level(opus_45, level, "bedrock")
    assert patch._effort_capability_keys(f"bedrock/{sonnet_45}") == ("", [])
