"""Tests for Bedrock runtime and translation helpers."""

from __future__ import annotations

import json
import sys

import pytest

from benchflow.providers.bedrock_runtime import (
    anthropic_request_to_bedrock_converse,
    bedrock_response_to_anthropic,
    bedrock_response_to_openai_response,
    bedrock_stream_event_to_anthropic_sse,
    bedrock_stream_event_to_openai_response_sse,
    build_bedrock_client,
    openai_responses_request_to_bedrock_converse,
    resolve_bedrock_region,
    validate_bedrock_runtime_env,
)


class TestBedrockRuntimeEnv:
    def test_resolve_bedrock_region_prefers_aws_region(self):
        assert (
            resolve_bedrock_region(
                {"AWS_REGION": "us-east-1", "AWS_DEFAULT_REGION": "us-west-2"}
            )
            == "us-east-1"
        )

    def test_validate_bedrock_runtime_env_normalizes_region_aliases(self):
        normalized = validate_bedrock_runtime_env(
            {
                "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
                "AWS_DEFAULT_REGION": "us-east-1",
            }
        )
        assert normalized["AWS_REGION"] == "us-east-1"
        assert normalized["AWS_DEFAULT_REGION"] == "us-east-1"

    def test_validate_bedrock_runtime_env_requires_bearer_token(self):
        with pytest.raises(ValueError, match="AWS_BEARER_TOKEN_BEDROCK required"):
            validate_bedrock_runtime_env({"AWS_REGION": "us-east-1"})

    def test_build_bedrock_client_lazy_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "boto3", None)
        with pytest.raises(RuntimeError, match="uv sync --extra bedrock"):
            build_bedrock_client(
                {
                    "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
                    "AWS_REGION": "us-east-1",
                }
            )

    def test_build_bedrock_client_with_injected_boto3_module(self):
        calls = {}

        class FakeBoto3:
            def client(self, **kwargs):
                calls.update(kwargs)
                return {"ok": True}

        client = build_bedrock_client(
            {
                "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
                "AWS_REGION": "us-east-1",
            },
            boto3_module=FakeBoto3(),
        )
        assert client == {"ok": True}
        assert calls == {
            "service_name": "bedrock-runtime",
            "region_name": "us-east-1",
        }


class TestAnthropicTranslation:
    def test_anthropic_request_to_bedrock_converse(self):
        body = {
            "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
            "system": "Be concise.",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup_weather",
                            "input": {"city": "Boston"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "Sunny",
                        }
                    ],
                },
            ],
            "max_tokens": 256,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop_sequences": ["DONE"],
            "tools": [
                {
                    "name": "lookup_weather",
                    "description": "Get current weather",
                    "input_schema": {"type": "object"},
                }
            ],
            "tool_choice": {"type": "tool", "name": "lookup_weather"},
        }

        payload = anthropic_request_to_bedrock_converse(body)

        assert payload["modelId"] == body["model"]
        assert payload["system"] == [{"text": "Be concise."}]
        assert payload["messages"][0]["content"] == [{"text": "Hello"}]
        assert payload["messages"][1]["content"][0]["toolUse"]["name"] == "lookup_weather"
        assert (
            payload["messages"][2]["content"][0]["toolResult"]["toolUseId"]
            == "toolu_1"
        )
        assert payload["inferenceConfig"] == {
            "maxTokens": 256,
            "temperature": 0.2,
            "topP": 0.9,
            "stopSequences": ["DONE"],
        }
        assert payload["toolConfig"]["toolChoice"] == {"tool": {"name": "lookup_weather"}}

    def test_bedrock_response_to_anthropic(self):
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Hi there"},
                        {
                            "toolUse": {
                                "toolUseId": "toolu_1",
                                "name": "lookup_weather",
                                "input": {"city": "Boston"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        normalized = bedrock_response_to_anthropic(
            response,
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
        )
        assert normalized["content"][0] == {"type": "text", "text": "Hi there"}
        assert normalized["content"][1]["type"] == "tool_use"
        assert normalized["usage"] == {"input_tokens": 10, "output_tokens": 5}
        assert normalized["stop_reason"] == "end_turn"


class TestOpenAIResponsesTranslation:
    def test_openai_responses_request_to_bedrock_converse(self):
        body = {
            "model": "openai.gpt-oss-20b-1:0",
            "instructions": "Be concise.",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "lookup_weather",
                            "arguments": "{\"city\":\"Boston\"}",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": {"temperature": "70F"},
                        }
                    ],
                },
            ],
            "max_output_tokens": 128,
            "temperature": 0.1,
            "top_p": 0.95,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "auto",
        }

        payload = openai_responses_request_to_bedrock_converse(body)

        assert payload["modelId"] == body["model"]
        assert payload["system"] == [{"text": "Be concise."}]
        assert payload["messages"][0]["content"] == [{"text": "Hello"}]
        assert payload["messages"][1]["content"][0]["toolUse"]["toolUseId"] == "call_1"
        assert payload["messages"][2]["content"][0]["toolResult"]["toolUseId"] == "call_1"
        assert payload["inferenceConfig"]["maxTokens"] == 128
        assert payload["toolConfig"]["toolChoice"] == {"auto": {}}

    def test_bedrock_response_to_openai_response(self):
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Hello"},
                        {
                            "toolUse": {
                                "toolUseId": "call_1",
                                "name": "lookup_weather",
                                "input": {"city": "Boston"},
                            }
                        },
                    ],
                }
            },
            "usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13},
        }
        normalized = bedrock_response_to_openai_response(
            response,
            model="openai.gpt-oss-20b-1:0",
            response_id="resp_123",
            created_at=42,
        )
        assert normalized["id"] == "resp_123"
        assert normalized["model"] == "openai.gpt-oss-20b-1:0"
        assert normalized["output_text"] == "Hello"
        assert normalized["output"][0]["type"] == "message"
        assert normalized["output"][1]["type"] == "function_call"
        assert normalized["usage"] == {
            "input_tokens": 9,
            "output_tokens": 4,
            "total_tokens": 13,
        }

    def test_bedrock_response_to_openai_response_ignores_reasoning_blocks(self):
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {"text": "internal reasoning"}
                            }
                        },
                        {"text": "visible output"},
                    ],
                }
            },
            "usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13},
        }
        normalized = bedrock_response_to_openai_response(
            response,
            model="openai.gpt-oss-20b-1:0",
        )
        assert normalized["output_text"] == "visible output"
        assert len(normalized["output"]) == 1


class TestStreamTranslation:
    def test_bedrock_stream_event_to_anthropic_sse_text_delta(self):
        frames = bedrock_stream_event_to_anthropic_sse(
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": "Hi"},
                }
            },
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
        )
        assert len(frames) == 1
        assert '"type":"content_block_delta"' in frames[0]
        assert '"text":"Hi"' in frames[0]

    def test_bedrock_stream_event_to_anthropic_sse_stop(self):
        frames = bedrock_stream_event_to_anthropic_sse(
            {"messageStop": {"stopReason": "end_turn"}},
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
        )
        assert len(frames) == 2
        assert '"type":"message_delta"' in frames[0]
        assert '"type":"message_stop"' in frames[1]

    def test_bedrock_stream_event_to_openai_response_sse_text_delta(self):
        frames = bedrock_stream_event_to_openai_response_sse(
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": "Hi"},
                }
            },
            model="openai.gpt-oss-20b-1:0",
        )
        assert len(frames) == 1
        assert '"type":"response.output_text.delta"' in frames[0]
        assert '"delta":"Hi"' in frames[0]

    def test_bedrock_stream_event_to_openai_response_sse_completed(self):
        frames = bedrock_stream_event_to_openai_response_sse(
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 10,
                        "outputTokens": 5,
                        "totalTokens": 15,
                    }
                }
            },
            model="openai.gpt-oss-20b-1:0",
            response_id="resp_123",
        )
        assert len(frames) == 1
        assert '"type":"response.completed"' in frames[0]
        payload = frames[0].split("data: ", 1)[1].strip()
        data = json.loads(payload)
        assert data["response"]["id"] == "resp_123"
        assert data["response"]["usage"]["total_tokens"] == 15

    def test_bedrock_stream_event_to_openai_response_sse_ignores_reasoning_delta(self):
        frames = bedrock_stream_event_to_openai_response_sse(
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"text": "hidden"}},
                }
            },
            model="openai.gpt-oss-20b-1:0",
        )
        assert frames == []
