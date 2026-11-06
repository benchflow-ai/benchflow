"""Guards the report that a picture Claude Code read reached a bridged model as
base64 text, and that a model that reads text only refuses an image part.

LiteLLM 1.91.5's ``/v1/messages`` bridge made a ``tool_result`` holding one
image a tool message whose content is the image's data URI, so Claude Code's
Read of a PNG cost about one token per 1.44 characters and the model never
saw it; one Azure DeepSeek V4.1-Flash run overflowed 1,048,576 tokens after
three Reads. The proxy's callback module gave such a message the image part
the bridge gave an image beside text. LiteLLM 1.103.3 gives every model that
image part itself (BerriAI/litellm#34462). A model that reads text only
answers a request with an image part 400 ("Model only supports text input"
on Azure), as it does every later request, so for such a model the callback
module now gives the tool message the data URI the bridge sent before.

The callback module is loaded as the proxy loads it, from a runtime folder,
and LiteLLM's own translation is driven with synthetic requests.
"""

from __future__ import annotations

import base64
import copy
import importlib.util
from types import ModuleType
from typing import Any

import litellm
import pytest
from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)

from benchflow.providers.litellm_logging import callback_module_source

ROUTE_ENV = "BENCHFLOW_LITELLM_UPSTREAM_MODEL"
# A 1x1 PNG.
PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42"
    "mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_URI = f"data:image/png;base64,{PNG}"
PNG_PART = {"type": "image_url", "image_url": {"url": PNG_URI}}
JPEG = base64.b64encode(
    b"\xff\xd8\xff\xe0" + bytes(range(256)) * 3 + b"\xff\xd9"
).decode()


@pytest.fixture
def proxy_callback(tmp_path, monkeypatch) -> ModuleType:
    """Import callback_module_source() from a runtime folder, as the proxy does."""
    monkeypatch.delenv(ROUTE_ENV, raising=False)
    path = tmp_path / "benchflow_litellm_callback.py"
    path.write_text(callback_module_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("benchflow_litellm_callback", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _route(monkeypatch, route: str) -> None:
    """The run's route, with no entry of its own in litellm.model_cost (which
    may be a map fetched at import) unless the test registers one."""
    monkeypatch.setenv(ROUTE_ENV, route)
    monkeypatch.delitem(litellm.model_cost, route, raising=False)


def _image(data: str = PNG, media: str = "image/png") -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media, "data": data},
    }


def _result(tool_id: str, content: Any) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_id, "content": content}


def _conversation(*results: dict[str, Any]) -> list[dict[str, Any]]:
    """A user turn, the assistant's tool calls, then their results (Claude Code's order)."""
    calls = [
        {
            "type": "tool_use",
            "id": result["tool_use_id"],
            "name": "Read",
            "input": {"file_path": f"/root/{result['tool_use_id']}"},
        }
        for result in results
    ]
    return [
        {"role": "user", "content": [{"type": "text", "text": "Read the files."}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Read them.", "signature": "s"},
                *calls,
            ],
        },
        {"role": "user", "content": list(results)},
    ]


def _translate(messages: list[Any], model: str | None, **kwargs: Any) -> list[Any]:
    return LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
        messages=copy.deepcopy(messages), model=model, **kwargs
    )


def _stock_translate(messages: list[Any], model: str | None) -> list[Any]:
    """LiteLLM's own translation, whatever wraps it now."""
    function = LiteLLMAnthropicMessagesAdapter.__dict__[
        "translate_anthropic_messages_to_openai"
    ]
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    return function(
        LiteLLMAnthropicMessagesAdapter(), messages=copy.deepcopy(messages), model=model
    )


def _tools(built: list[Any]) -> dict[str, Any]:
    return {m["tool_call_id"]: m["content"] for m in built if m.get("role") == "tool"}


@pytest.mark.usefixtures("proxy_callback")
def test_a_lone_tool_result_image_goes_out_as_an_image_part(monkeypatch):
    """LiteLLM's own translation since BerriAI/litellm#34462: the callback
    module leaves it as it is."""
    _route(monkeypatch, "azure/DeepSeek-V4.1-Flash")
    messages = _conversation(_result("a", [_image()]), _result("b", "plain output"))

    built = _translate(messages, "DeepSeek-V4.1-Flash")
    stock = _stock_translate(messages, "DeepSeek-V4.1-Flash")

    assert _tools(stock)["a"] == [PNG_PART]
    assert _tools(built) == {"a": [PNG_PART], "b": "plain output"}
    assert built == stock


@pytest.mark.usefixtures("proxy_callback")
def test_the_part_is_the_one_the_bridge_gives_an_image_beside_text(monkeypatch):
    _route(monkeypatch, "openai/gpt-4o")
    messages = _conversation(
        _result("lone", [_image(JPEG, "image/jpeg")]),
        _result(
            "pair", [{"type": "text", "text": "caption"}, _image(JPEG, "image/jpeg")]
        ),
    )

    tools = _tools(_translate(messages, "gpt-4o"))

    assert tools["lone"] == [tools["pair"][1]]
    assert tools["pair"] == _tools(_stock_translate(messages, "gpt-4o"))["pair"]


@pytest.mark.usefixtures("proxy_callback")
def test_the_handler_sends_the_part_upstream(monkeypatch):
    """The bridged request as the proxy builds it for litellm.acompletion."""
    _route(monkeypatch, "azure/DeepSeek-V4.1-Flash")
    kwargs, _ = (
        LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
            max_tokens=1024,
            messages=_conversation(_result("a", [_image()])),
            model="DeepSeek-V4.1-Flash",
            stream=True,
            extra_kwargs={"custom_llm_provider": "azure"},
        )
    )

    assert _tools(kwargs["messages"]) == {"a": [PNG_PART]}


@pytest.mark.usefixtures("proxy_callback")
def test_the_context_management_polyfill_gets_the_part_too(monkeypatch):
    """Its summary call and token count translate with no model (the route's)
    and keep mid-conversation system turns."""
    _route(monkeypatch, "azure/DeepSeek-V4.1-Flash")
    messages = _conversation(_result("a", [_image()]))

    built = _translate(messages, None, preserve_midturn_system=True)

    assert _tools(built)["a"] == [PNG_PART]


@pytest.mark.usefixtures("proxy_callback")
@pytest.mark.parametrize(
    ("route", "model"),
    [
        ("hosted_vllm/Qwen/Qwen3-Coder-Next", "Qwen/Qwen3-Coder-Next"),
        # LiteLLM 1.103.3's pinned map: deepseek-v4-flash reads images (1.91.5's
        # map said it read text only).
        ("deepseek/deepseek-v4-flash", "deepseek-v4-flash"),
    ],
    ids=["unknown-model", "pinned-map"],
)
def test_a_model_that_reads_images_or_that_no_source_knows_gets_the_part(
    monkeypatch, route, model
):
    _route(monkeypatch, route)
    messages = _conversation(_result("a", [_image()]))

    assert _tools(_translate(messages, model))["a"] == [PNG_PART]


@pytest.mark.usefixtures("proxy_callback")
def test_the_deployment_model_info_decides_first(monkeypatch):
    """config.yaml's model_info (LiteLLM registers it under the route) wins
    over the map's text-only deepseek-v4-pro."""
    _route(monkeypatch, "azure/DeepSeek-V4-Pro")
    monkeypatch.setitem(
        litellm.model_cost, "azure/DeepSeek-V4-Pro", {"supports_vision": True}
    )
    messages = _conversation(_result("a", [_image()]))

    assert _tools(_translate(messages, "DeepSeek-V4-Pro"))["a"] == [PNG_PART]


@pytest.mark.usefixtures("proxy_callback")
@pytest.mark.parametrize(
    ("route", "model"),
    [
        # LiteLLM's pinned map: deepseek-v4-pro reads text only.
        ("azure/DeepSeek-V4-Pro", "DeepSeek-V4-Pro"),
        ("deepseek/deepseek-v4-pro", "deepseek-v4-pro"),
        # The polyfill's calls on that route.
        ("azure/DeepSeek-V4-Pro", None),
        # BenchFlow's registry: GLM-5.3 reads text only.
        ("openai/glm-5.3", "glm-5.3"),
    ],
)
def test_a_text_only_model_gets_the_image_as_its_data_uri(monkeypatch, route, model):
    _route(monkeypatch, route)
    messages = _conversation(_result("a", [_image()]), _result("b", "plain output"))

    built = _translate(messages, model)
    stock = _stock_translate(messages, model)

    assert _tools(stock)["a"] == [PNG_PART]
    assert _tools(built) == {"a": PNG_URI, "b": "plain output"}
    assert [m for m in built if m.get("tool_call_id") != "a"] == [
        m for m in stock if m.get("tool_call_id") != "a"
    ]


@pytest.mark.usefixtures("proxy_callback")
def test_a_deployment_stated_text_only_gets_the_data_uri(monkeypatch):
    _route(monkeypatch, "openai/gpt-4o")
    monkeypatch.setitem(litellm.model_cost, "openai/gpt-4o", {"supports_vision": False})
    messages = _conversation(_result("a", [_image()]))

    assert _tools(_translate(messages, "gpt-4o"))["a"] == PNG_URI


@pytest.mark.usefixtures("proxy_callback")
@pytest.mark.parametrize(
    ("route", "model"),
    [
        ("azure/DeepSeek-V4.1-Flash", "DeepSeek-V4.1-Flash"),
        ("azure/DeepSeek-V4-Pro", "DeepSeek-V4-Pro"),
    ],
    ids=["reads-images", "text-only"],
)
@pytest.mark.parametrize(
    "result",
    [
        _result("a", [{"type": "text", "text": PNG_URI}]),
        _result("a", PNG_URI),
        _result("a", [PNG_URI]),
        _result(
            "a", [{"type": "text", "text": "two"}, {"type": "text", "text": "items"}]
        ),
        _result("a", [{"type": "text", "text": "caption"}, _image()]),
        {"type": "tool_result", "tool_use_id": "a"},
    ],
    ids=[
        "text-like-a-data-uri",
        "string",
        "bare-string-item",
        "texts",
        "text-and-image",
        "empty",
    ],
)
def test_every_other_tool_result_is_the_bridges_own(monkeypatch, route, model, result):
    _route(monkeypatch, route)
    messages = _conversation(result)

    assert _translate(messages, model) == _stock_translate(messages, model)


@pytest.mark.usefixtures("proxy_callback")
def test_the_request_claude_code_sent_is_not_changed(monkeypatch):
    _route(monkeypatch, "azure/DeepSeek-V4-Pro")
    messages = _conversation(_result("a", [_image()]))
    sent = copy.deepcopy(messages)

    built = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
        messages=messages, model="DeepSeek-V4-Pro"
    )

    assert _tools(built)["a"] == PNG_URI
    assert messages == sent


def test_the_text_only_models_are_the_registrys(proxy_callback):
    from benchflow.agents.providers import PROVIDERS

    listed = {
        str(meta["id"]).lower()
        for provider in PROVIDERS.values()
        for meta in provider.models
        if "input" in meta and "image" not in meta["input"]
    }
    assert listed == proxy_callback.REGISTRY_TEXT_ONLY_MODELS
