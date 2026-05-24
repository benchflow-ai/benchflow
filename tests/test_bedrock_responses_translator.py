"""Regression tests for OpenAI Responses -> Bedrock Converse translation.

Covers the multi-item ``input`` shape the real OpenAI Responses API emits, where
``function_call``, ``function_call_output``, and ``reasoning`` are top-level
items without a ``role`` field. See issue #364.
"""

from __future__ import annotations

import json

import pytest

from benchflow.providers.bedrock_runtime import (
    openai_responses_request_to_bedrock_converse,
)


class TestTopLevelToolItems:
    def test_top_level_function_call_and_output_do_not_crash(self):
        body = {
            "model": "openai.gpt-oss-20b-1:0",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "lookup",
                    "arguments": '{"q":"x"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": "ok",
                },
            ],
        }

        payload = openai_responses_request_to_bedrock_converse(body)

        assert payload["modelId"] == body["model"]
        # user "hi", assistant tool call, user tool result -> 3 messages
        assert [m["role"] for m in payload["messages"]] == [
            "user",
            "assistant",
            "user",
        ]
        assert payload["messages"][0]["content"] == [{"text": "hi"}]
        tool_use = payload["messages"][1]["content"][0]["toolUse"]
        assert tool_use["toolUseId"] == "c1"
        assert tool_use["name"] == "lookup"
        assert tool_use["input"] == {"q": "x"}
        tool_result = payload["messages"][2]["content"][0]["toolResult"]
        assert tool_result["toolUseId"] == "c1"
        assert tool_result["status"] == "success"
        assert tool_result["content"] == [{"text": "ok"}]

    def test_function_call_output_with_non_string_output_is_json_encoded(self):
        body = {
            "model": "openai.gpt-oss-20b-1:0",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": {"temperature": "70F"},
                },
            ],
        }

        payload = openai_responses_request_to_bedrock_converse(body)

        result_text = payload["messages"][0]["content"][0]["toolResult"]["content"][0][
            "text"
        ]
        assert json.loads(result_text) == {"temperature": "70F"}

    def test_function_call_with_empty_arguments(self):
        body = {
            "model": "openai.gpt-oss-20b-1:0",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "noop",
                    "arguments": "",
                },
            ],
        }

        payload = openai_responses_request_to_bedrock_converse(body)

        assert (
            payload["messages"][0]["content"][0]["toolUse"]["input"] == {}
        )

    def test_reasoning_items_are_skipped(self):
        body = {
            "model": "openai.gpt-oss-20b-1:0",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "thinking"}],
                },
            ],
        }

        payload = openai_responses_request_to_bedrock_converse(body)

        # reasoning item dropped; only the user message survives
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == [{"text": "hi"}]

    def test_adjacent_tool_items_coalesce_into_one_message(self):
        """Two consecutive function_calls should land in one assistant turn."""
        body = {
            "model": "openai.gpt-oss-20b-1:0",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "a",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "c2",
                    "name": "b",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": "r1",
                },
                {
                    "type": "function_call_output",
                    "call_id": "c2",
                    "output": "r2",
                },
            ],
        }

        payload = openai_responses_request_to_bedrock_converse(body)

        assert [m["role"] for m in payload["messages"]] == ["assistant", "user"]
        assert len(payload["messages"][0]["content"]) == 2
        assert len(payload["messages"][1]["content"]) == 2
        ids = [b["toolUse"]["toolUseId"] for b in payload["messages"][0]["content"]]
        assert ids == ["c1", "c2"]
        result_ids = [
            b["toolResult"]["toolUseId"] for b in payload["messages"][1]["content"]
        ]
        assert result_ids == ["c1", "c2"]

    def test_text_input_only_string_form_still_works(self):
        body = {"model": "openai.gpt-oss-20b-1:0", "input": "hello world"}

        payload = openai_responses_request_to_bedrock_converse(body)

        assert payload["messages"] == [
            {"role": "user", "content": [{"text": "hello world"}]}
        ]

    def test_unknown_item_type_raises_value_error(self):
        body = {
            "model": "openai.gpt-oss-20b-1:0",
            "input": [{"type": "image_generation_call", "id": "ig_1"}],
        }

        with pytest.raises(ValueError, match="Unsupported OpenAI Responses input"):
            openai_responses_request_to_bedrock_converse(body)

    def test_message_item_without_role_raises_value_error(self):
        body = {
            "model": "openai.gpt-oss-20b-1:0",
            "input": [
                {"type": "message", "content": [{"type": "input_text", "text": "hi"}]}
            ],
        }

        with pytest.raises(ValueError, match="missing string role"):
            openai_responses_request_to_bedrock_converse(body)

    def test_repro_from_issue_364(self):
        """Exact body from issue #364 must translate without raising."""
        body = {
            "model": "openai.gpt-oss-20b-1:0",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "lookup",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": "ok",
                },
            ],
        }

        # Must not raise KeyError('role') anymore.
        payload = openai_responses_request_to_bedrock_converse(body)
        assert "messages" in payload
