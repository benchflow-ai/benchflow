"""Guards the report that Claude Code compacted far too early on bridged models.

LiteLLM 1.91.5's ``/v1/messages`` bridge to chat completions opened an empty
text block before it read the upstream answer, so an answer that starts with
its thinking (DeepSeek on Azure, Mistral) reached Claude Code as an empty text
block, then the thinking. Claude Code counted the thinking twice and compacted
at real prompts of 20k-50k, until the proxy's callback module dropped that
block. LiteLLM 1.103.3 opens the first block with the type of the first
upstream chunk that carries something (BerriAI/litellm#34433), so the callback
module now leaves the stream as LiteLLM makes it, and these tests check that
LiteLLM's own stream opens with the answer's first block.

The callback module is loaded as the proxy loads it, from a runtime folder,
and LiteLLM's own stream wrapper is driven with synthetic upstream chunks.
"""

from __future__ import annotations

import importlib.util
import json
from typing import Any

import pytest
from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

from benchflow.providers.litellm_logging import callback_module_source


@pytest.fixture(autouse=True)
def _proxy_callback(tmp_path):
    """Import callback_module_source() from a runtime folder, as the proxy does."""
    path = tmp_path / "benchflow_litellm_callback.py"
    path.write_text(callback_module_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("benchflow_litellm_callback", path)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def _chunk(finish: str | None = None, usage: Usage | None = None, **delta: Any):
    chunk = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(**delta), finish_reason=finish)],
        model="m",
    )
    if usage is not None:
        chunk.usage = usage
    return chunk


def _tool_call(arguments: str, *, call_id: str | None = None, name: str | None = None):
    return [
        ChatCompletionDeltaToolCall(
            id=call_id,
            type="function",
            index=0,
            function=Function(name=name, arguments=arguments),
        )
    ]


def _usage() -> Usage:
    return Usage(prompt_tokens=1200, completion_tokens=80, total_tokens=1280)


def _thinking_text_tool() -> list[Any]:
    return [
        _chunk(role="assistant", content=""),
        _chunk(reasoning_content="The file holds "),
        _chunk(reasoning_content="the answer."),
        _chunk(content="Reading it."),
        _chunk(tool_calls=_tool_call("", call_id="call_1", name="Read")),
        _chunk(tool_calls=_tool_call('{"file_path": "/root/a.txt"}')),
        _chunk(finish="tool_calls"),
        _chunk(usage=_usage()),
    ]


async def _upstream(chunks: list[Any], error: Exception | None = None):
    for chunk in chunks:
        yield chunk
    if error is not None:
        raise error


async def _events(chunks: list[Any], **kwargs: Any) -> list[Any]:
    return [
        event
        async for event in AnthropicStreamWrapper(
            completion_stream=_upstream(chunks), model="m", **kwargs
        )
    ]


def _starts(events: list[Any]) -> list[tuple[int, str]]:
    return [
        (event["index"], event["content_block"]["type"])
        for event in events
        if event["type"] == "content_block_start"
    ]


def _closed_in_order(events: list[Any]) -> bool:
    """Each block stops before the next one starts, at the index it started."""
    open_index = None
    for event in events:
        if event["type"] == "content_block_start":
            if open_index is not None:
                return False
            open_index = event["index"]
        elif event["type"] == "content_block_stop":
            if event["index"] != open_index:
                return False
            open_index = None
    return open_index is None


@pytest.mark.asyncio
async def test_a_thinking_first_answer_reaches_claude_code_thinking_first():
    events = await _events(_thinking_text_tool())

    assert _starts(events) == [(0, "thinking"), (1, "text"), (2, "tool_use")]
    assert _closed_in_order(events)
    assert [event["type"] for event in events[-2:]] == ["message_delta", "message_stop"]


@pytest.mark.asyncio
async def test_the_bytes_claude_code_reads_open_with_the_thinking_block():
    """The proxy's callback module no longer wraps LiteLLM's stream: Claude
    Code reads LiteLLM's own events, and the first block they open is the
    thinking."""
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    translate = LiteLLMAnthropicMessagesAdapter.translate_anthropic_messages_to_openai
    mark = translate.__globals__["BRIDGE_PATCH_MARK"]
    assert getattr(translate, mark, False)
    for name in ("__anext__", "__next__"):
        assert not getattr(AnthropicStreamWrapper.__dict__[name], mark, False)

    wrapper = AnthropicStreamWrapper(
        completion_stream=_upstream(_thinking_text_tool()), model="m"
    )
    sent = b"".join([part async for part in wrapper.async_anthropic_sse_wrapper()])
    events = [
        json.loads(line.removeprefix(b"data: "))
        for line in sent.splitlines()
        if line.startswith(b"data: ")
    ]

    assert _starts(events) == [(0, "thinking"), (1, "text"), (2, "tool_use")]


def test_a_tool_call_first_answer_opens_with_the_tool_call_when_read_sync():
    chunks = [
        _chunk(tool_calls=_tool_call("", call_id="call_7", name="Bash")),
        _chunk(tool_calls=_tool_call('{"command": "ls"}')),
        _chunk(finish="tool_calls"),
        _chunk(usage=_usage()),
    ]
    events = list(
        AnthropicStreamWrapper(completion_stream=iter(list(chunks)), model="m")
    )

    assert _starts(events) == [(0, "tool_use")]
    assert _closed_in_order(events)


@pytest.mark.asyncio
async def test_after_a_compaction_block_the_thinking_block_comes_next():
    """With the context-management polyfill's compaction block first, the
    thinking block is block 1: no empty text block comes between them."""
    chunks = [
        _chunk(reasoning_content="Summarised; go on."),
        _chunk(content="Done."),
        _chunk(finish="stop"),
        _chunk(usage=_usage()),
    ]
    compaction = {"type": "compaction", "content": "summary of the earlier turns"}
    events = await _events(chunks, compaction_block=compaction)

    assert _starts(events) == [(0, "compaction"), (1, "thinking"), (2, "text")]
    assert _closed_in_order(events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunks", "starts"),
    [
        (
            [
                _chunk(role="assistant", content=""),
                _chunk(content="Plain answer, "),
                _chunk(content="no thinking."),
                _chunk(finish="stop"),
                _chunk(usage=_usage()),
            ],
            [(0, "text")],
        ),
        (
            [
                _chunk(content="Text first, "),
                _chunk(reasoning_content="then thinking."),
                _chunk(finish="stop"),
                _chunk(usage=_usage()),
            ],
            [(0, "text"), (1, "thinking")],
        ),
        ([_chunk(finish="stop"), _chunk(usage=_usage())], [(0, "text")]),
        ([_chunk(role="assistant", content=""), _chunk(finish="stop")], [(0, "text")]),
        ([], []),
    ],
    ids=["text-only", "text-then-thinking", "empty", "empty-no-usage", "no-chunks"],
)
async def test_every_other_answer_opens_with_its_own_first_block(chunks, starts):
    """An answer that starts with text opens a text block, and an empty answer
    one empty text block, as LiteLLM 1.91.5 sent them too."""
    events = await _events(chunks)

    assert _starts(events) == starts
    assert _closed_in_order(events)


@pytest.mark.asyncio
async def test_an_upstream_error_comes_after_the_events_it_interrupted():
    """The blank first chunk opens no block (BerriAI/litellm#34433), so the
    error comes right after message_start."""
    error = RuntimeError("upstream connection reset")
    wrapper = AnthropicStreamWrapper(
        completion_stream=_upstream([_chunk(role="assistant", content="")], error),
        model="m",
    )
    seen: list[Any] = []
    with pytest.raises(RuntimeError) as raised:
        async for event in wrapper:
            seen.append(event)

    assert raised.value is error
    assert [event["type"] for event in seen] == ["message_start"]
