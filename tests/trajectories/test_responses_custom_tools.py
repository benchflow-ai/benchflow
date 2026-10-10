"""Guards FrontierPhysics PR #516's tool-only Azure reviewer export failures."""

import json

from benchflow.trajectories.export_prime_sft import normalize_prime_sft_exchange


def _exchange(output, history=()):
    return {
        "request": {
            "body": {
                "instructions": "Inspect the provided artifacts.",
                "input": [
                    {
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": [
                            {
                                "type": "namespace",
                                "name": "functions",
                                "tools": [
                                    {
                                        "type": "custom",
                                        "name": "exec",
                                        "description": "Execute code.",
                                        "format": {"type": "text"},
                                    },
                                    {
                                        "type": "function",
                                        "name": "wait",
                                        "description": "Wait.",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {"id": {"type": "integer"}},
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                    {"role": "user", "content": "Inspect the file."},
                    *history,
                ],
            }
        },
        "response": {
            "status_code": 200,
            "body": {
                "output": output,
                "usage": {"input_tokens": 20, "output_tokens": 5},
            },
        },
    }


def test_tool_only_custom_response_preserves_action_and_declared_tool():
    """PR #516: a real custom call is an assistant action, not no_assistant."""
    code = 'const value = "not JSON: \\n";\ntext(value);'
    exchange = _exchange(
        [
            {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "exec",
                "namespace": "functions",
                "input": code,
            }
        ]
    )
    normalized, skipped = normalize_prime_sft_exchange(exchange)
    assert skipped is None
    assert normalized is not None
    action = normalized.messages[-1]
    assert action["role"] == "assistant" and action["content"] == ""
    tool = action["tool_calls"][0]
    assert tool["id"] == "call_1" and tool["function"]["name"] == "functions.exec"
    assert json.loads(tool["function"]["arguments"]) == {"input": code}
    definitions = {
        row["function"]["name"]: row["function"] for row in normalized.tool_defs
    }
    assert set(definitions) == {"functions.exec", "functions.wait"}
    assert definitions["functions.exec"]["parameters"]["required"] == ["input"]
    assert [row["content"] for row in normalized.messages if row["role"] == "user"] == [
        "Inspect the file."
    ]


def test_custom_history_output_and_function_namespace_are_preserved():
    """PR #516: custom history is not rewritten as an empty user instruction."""
    history = [
        {
            "type": "custom_tool_call",
            "call_id": "call_prev",
            "name": "exec",
            "namespace": "functions",
            "input": "text(42)",
        },
        {"type": "custom_tool_call_output", "call_id": "call_prev", "output": "42"},
    ]
    exchange = _exchange(
        [
            {
                "type": "function_call",
                "call_id": "call_new",
                "name": "wait",
                "namespace": "functions",
                "arguments": '{"id":7}',
            }
        ],
        history,
    )
    normalized, skipped = normalize_prime_sft_exchange(exchange)
    assert skipped is None and normalized is not None
    assert [row["role"] for row in normalized.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert normalized.messages[3] == {
        "role": "tool",
        "tool_call_id": "call_prev",
        "content": "42",
    }
    assert (
        normalized.messages[-1]["tool_calls"][0]["function"]["name"] == "functions.wait"
    )


def test_custom_response_does_not_invent_missing_tool_definitions():
    """PR #516: healthy conversion still fails closed without actual definitions."""
    exchange = _exchange(
        [
            {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "exec",
                "namespace": "functions",
                "input": "text(42)",
            }
        ]
    )
    exchange["request"]["body"]["input"].pop(0)
    normalized, skipped = normalize_prime_sft_exchange(exchange)
    assert normalized is None and skipped == "missing_tool_defs"


def test_unqualified_history_uses_only_unique_declared_namespace():
    """PR #516: Codex input history omits namespaces present on response calls."""
    history = [
        {
            "type": "custom_tool_call",
            "call_id": "previous",
            "name": "exec",
            "input": "text(1)",
        },
        {"type": "custom_tool_call_output", "call_id": "previous", "output": "1"},
    ]
    exchange = _exchange(
        [
            {
                "type": "custom_tool_call",
                "call_id": "next",
                "name": "exec",
                "namespace": "functions",
                "input": "text(2)",
            }
        ],
        history,
    )
    normalized, skipped = normalize_prime_sft_exchange(exchange)
    assert skipped is None and normalized is not None
    assert (
        normalized.messages[2]["tool_calls"][0]["function"]["name"] == "functions.exec"
    )
    exchange["request"]["body"]["input"][0]["tools"].append(
        {
            "type": "namespace",
            "name": "unrelated",
            "tools": [{"type": "custom", "name": "exec"}],
        }
    )
    normalized, skipped = normalize_prime_sft_exchange(exchange)
    assert normalized is None and skipped.startswith("invalid_prime_sft_row:")
