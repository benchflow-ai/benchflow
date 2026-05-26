"""Tests for Bedrock runtime and translation helpers."""

from __future__ import annotations

import base64
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
        """Guards v0.5-integration@e55219d against Bedrock read-timeout proxy exits."""
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
        assert calls["service_name"] == "bedrock-runtime"
        assert calls["region_name"] == "us-east-1"
        assert calls["config"].connect_timeout == 10
        assert calls["config"].read_timeout == 300


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
        assert (
            payload["messages"][1]["content"][0]["toolUse"]["name"] == "lookup_weather"
        )
        assert (
            payload["messages"][2]["content"][0]["toolResult"]["toolUseId"] == "toolu_1"
        )
        assert payload["inferenceConfig"] == {
            "maxTokens": 256,
            "temperature": 0.2,
            "topP": 0.9,
            "stopSequences": ["DONE"],
        }
        assert payload["toolConfig"]["toolChoice"] == {
            "tool": {"name": "lookup_weather"}
        }

    def test_opus47_request_omits_deprecated_sampling_params(self):
        """Guards v0.5-integration@e55219d against Bedrock Opus4.7 sampling-param 400s."""
        body = {
            "model": "us.anthropic.claude-opus-4-7",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 256,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop_sequences": ["DONE"],
        }

        payload = anthropic_request_to_bedrock_converse(body)

        assert payload["inferenceConfig"] == {
            "maxTokens": 256,
            "stopSequences": ["DONE"],
        }

    def test_tool_result_image_blocks_translate_to_bedrock_images(self):
        """Guards v0.5-integration@e55219d against OpenHands image tool-result proxy crashes."""
        image_bytes = b"fake jpeg bytes"
        body = {
            "model": "us.anthropic.claude-opus-4-7",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [
                                {"type": "text", "text": "Screenshot attached."},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64.b64encode(image_bytes).decode(),
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
            "max_tokens": 256,
        }

        payload = anthropic_request_to_bedrock_converse(body)

        content = payload["messages"][0]["content"][0]["toolResult"]["content"]
        assert content[0] == {"text": "Screenshot attached."}
        assert content[1] == {
            "image": {"format": "jpeg", "source": {"bytes": image_bytes}}
        }

    def test_user_image_blocks_translate_to_bedrock_images(self):
        """Guards v0.5-integration@e55219d against Anthropic image-message proxy crashes."""
        image_bytes = b"fake png bytes"
        body = {
            "model": "us.anthropic.claude-opus-4-7",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(image_bytes).decode(),
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 256,
        }

        payload = anthropic_request_to_bedrock_converse(body)

        assert payload["messages"][0]["content"][1] == {
            "image": {"format": "png", "source": {"bytes": image_bytes}}
        }

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
                            "arguments": '{"city":"Boston"}',
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
        assert (
            payload["messages"][2]["content"][0]["toolResult"]["toolUseId"] == "call_1"
        )
        assert payload["inferenceConfig"]["maxTokens"] == 128
        assert payload["toolConfig"]["toolChoice"] == {"auto": {}}

    def test_opus47_responses_request_omits_deprecated_sampling_params(self):
        """Guards v0.5-integration@e55219d against Bedrock Opus4.7 sampling-param 400s."""
        body = {
            "model": "global.anthropic.claude-opus-4-7",
            "input": "Hello",
            "max_output_tokens": 128,
            "temperature": 0.1,
            "top_p": 0.95,
        }

        payload = openai_responses_request_to_bedrock_converse(body)

        assert payload["inferenceConfig"] == {"maxTokens": 128}

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

    def test_bedrock_stream_event_to_openai_response_sse_tool_use_stop(self):
        block_types: dict[int, str] = {}

        start_frames = bedrock_stream_event_to_openai_response_sse(
            {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "start": {
                        "toolUse": {
                            "toolUseId": "call_123",
                            "name": "lookup_weather",
                        }
                    },
                }
            },
            model="openai.gpt-oss-20b-1:0",
            block_types=block_types,
        )
        delta_frames = bedrock_stream_event_to_openai_response_sse(
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": '{"city":"Boston"}'}},
                }
            },
            model="openai.gpt-oss-20b-1:0",
            block_types=block_types,
        )
        stop_frames = bedrock_stream_event_to_openai_response_sse(
            {"contentBlockStop": {"contentBlockIndex": 1}},
            model="openai.gpt-oss-20b-1:0",
            block_types=block_types,
        )

        assert len(start_frames) == 1
        assert '"type":"function_call"' in start_frames[0]
        assert len(delta_frames) == 1
        assert '"type":"response.function_call_arguments.delta"' in delta_frames[0]
        assert len(stop_frames) == 2
        assert '"type":"response.function_call_arguments.done"' in stop_frames[0]
        assert '"type":"response.output_item.done"' in stop_frames[1]
        assert block_types == {}

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
