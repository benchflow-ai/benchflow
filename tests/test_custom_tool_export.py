"""Codex's custom tool calls in the training export.

Codex offers a catalogued model ``apply_patch`` as a Responses ``custom`` tool
(freeform input, constrained by a grammar), so a scored Codex trial's
exchanges carry ``custom_tool_call`` and ``custom_tool_call_output`` items.
The export refused them (``unsupported_tool_protocol``), so the trial's
``results.jsonl`` had an ``export_error`` and no exchange.

The export now writes a custom tool as a function tool with one string
parameter, ``input``, and keeps the original definition under ``custom``; a
call becomes a function call with ``{"input": <the exact string>}`` and its
output a tool message with the call's id. A call whose tool has a ``custom``
definition inverts to the original exactly. What it cannot represent it
still refuses.

The fixture has the shape of a DeepSeek Flash GPQA trial's two exchanges
through the model proxy (text replaced).
"""

from __future__ import annotations

import copy
import json

import pytest

from benchflow.trajectories.export_prime_sft import (
    export_prime_sft_jsonl,
    normalize_prime_sft_exchange,
    validate_prime_sft_row,
)
from benchflow.trajectories.export_trl_sft import export_trl_sft_jsonl
from benchflow.trajectories.results import build_rollout_results_record

GRAMMAR = {
    "type": "grammar",
    "syntax": "lark",
    "definition": 'start: begin_patch hunk+ end_patch\nbegin_patch: "*** Begin Patch" LF\n'
    'end_patch: "*** End Patch" LF?\nhunk: "*** Add File: " /(.+)/ LF add_line+\n'
    'add_line: "+" /(.*)/ LF -> line\n%import common.LF\n',
}
PATCH = "*** Begin Patch\n*** Add File: /app/answer.txt\n+D\n*** End Patch\n"
RESULT = "Exit code: 0\nWall time: 0 seconds\nOutput:\nSuccess. Updated the following files:\nA /app/answer.txt\n"
CALL_ID = "call_00_sanitized0001"

TOOLS = [
    {
        "type": "function",
        "name": "exec_command",
        "description": "Runs a command.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
    {
        "type": "custom",
        "name": "apply_patch",
        "description": "Use the `apply_patch` tool to edit files.",
        "format": GRAMMAR,
    },
    {
        "type": "namespace",
        "name": "collaboration",
        "description": "Agents.",
        "tools": [
            {
                "type": "function",
                "name": "spawn_agent",
                "description": "Spawn an agent.",
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": {"task": {"type": "string"}},
                },
            }
        ],
    },
    {"type": "web_search", "external_web_access": True},
]


def _msg(role, text, kind="input_text", **extra):
    return {
        "type": "message",
        "role": role,
        "content": [{"type": kind, "text": text}],
        **extra,
    }


def _history():
    return [
        _msg("developer", "<permissions>sandboxed</permissions>"),
        _msg("user", "<environment_context><cwd>/app</cwd></environment_context>"),
        _msg(
            "user", "Which option? Write the single selected letter to /app/answer.txt."
        ),
    ]


def _call():
    return {
        "type": "custom_tool_call",
        "status": "completed",
        "call_id": CALL_ID,
        "name": "apply_patch",
        "input": PATCH,
    }


def _output():
    return {
        "type": "custom_tool_call_output",
        "id": "ctco_sanitized",
        "call_id": CALL_ID,
        "output": RESULT,
    }


def _exchange(input_items, output_items, tools=None):
    body = {
        "model": "deepseek-flash",
        "instructions": "You are Codex, a coding agent.",
        "input": input_items,
        # The proxy's capture keeps the Responses input under both names.
        "messages": copy.deepcopy(input_items),
        "tools": copy.deepcopy(TOOLS if tools is None else tools),
        "stream": True,
    }
    return {
        "request": {"body": body},
        "response": {
            "status_code": 200,
            "body": {
                "object": "response",
                "status": "completed",
                "output": output_items,
            },
        },
        "metadata": {"call_purpose": "agent", "call_type": "aresponses"},
    }


def first_exchange():
    return _exchange(
        _history(),
        [
            {"type": "reasoning", "summary": []},
            _msg(
                "assistant",
                "Lifetime broadening decides it.",
                "output_text",
                phase="commentary",
            ),
            _call(),
        ],
    )


def second_exchange():
    return _exchange(
        [
            *_history(),
            {"type": "reasoning", "summary": []},
            _msg(
                "assistant",
                "Lifetime broadening decides it.",
                "output_text",
                phase="commentary",
            ),
            _call(),
            _output(),
        ],
        [
            {"type": "reasoning", "summary": []},
            _msg(
                "assistant",
                "Wrote D to /app/answer.txt.",
                "output_text",
                phase="final_answer",
            ),
        ],
    )


def _def(tool_defs, name):
    return next(t for t in tool_defs if t["function"]["name"] == name)


def test_a_custom_tool_definition_becomes_a_marked_function():
    normalized, reason = normalize_prime_sft_exchange(first_exchange())
    assert reason is None
    apply_patch = _def(normalized.tool_defs, "apply_patch")
    assert apply_patch == {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Use the `apply_patch` tool to edit files.",
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            },
        },
        "custom": {"format": GRAMMAR},
    }
    assert "custom" not in _def(normalized.tool_defs, "exec_command")
    assert _def(normalized.tool_defs, "collaboration.spawn_agent")["function"][
        "parameters"
    ] == {"type": "object", "properties": {"task": {"type": "string"}}}


def test_the_call_keeps_its_exact_input_and_id():
    normalized, _ = normalize_prime_sft_exchange(first_exchange())
    last = normalized.messages[-1]
    assert last["role"] == "assistant"
    assert last["content"] == "Lifetime broadening decides it."
    (call,) = last["tool_calls"]
    assert call["id"] == CALL_ID
    assert call["type"] == "function"
    assert call["function"]["name"] == "apply_patch"
    assert json.loads(call["function"]["arguments"]) == {"input": PATCH}


def test_the_output_links_to_its_call():
    normalized, reason = normalize_prime_sft_exchange(second_exchange())
    assert reason is None
    messages = normalized.messages
    calls = [m for m in messages if m.get("tool_calls")]
    tools = [m for m in messages if m["role"] == "tool"]
    assert [c["tool_calls"][0]["id"] for c in calls] == [CALL_ID]
    assert tools == [{"role": "tool", "tool_call_id": CALL_ID, "content": RESULT}]
    assert messages.index(tools[0]) == messages.index(calls[0]) + 1
    assert messages[-1] == {
        "role": "assistant",
        "content": "Wrote D to /app/answer.txt.",
    }
    validate_prime_sft_row({"messages": messages, "tool_defs": normalized.tool_defs}, 1)


def test_the_export_inverts_to_the_original_items():
    """A trainer that wants freeform calls gets the original items back."""
    normalized, _ = normalize_prime_sft_exchange(second_exchange())
    custom = {
        t["function"]["name"]: t["custom"]
        for t in normalized.tool_defs
        if "custom" in t
    }
    inverted = []
    for message in normalized.messages:
        for call in message.get("tool_calls") or []:
            name = call["function"]["name"]
            if name in custom:
                inverted.append(
                    {
                        "type": "custom_tool_call",
                        "call_id": call["id"],
                        "name": name,
                        "input": json.loads(call["function"]["arguments"])["input"],
                    }
                )
        if message["role"] == "tool":
            inverted.append(
                {
                    "type": "custom_tool_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
    original = [
        {
            k: v
            for k, v in item.items()
            if k in ("type", "call_id", "name", "input", "output")
        }
        for item in (_call(), _output())
    ]
    assert inverted == original
    assert custom == {"apply_patch": {"format": GRAMMAR}}


def test_a_namespaced_function_call_takes_its_qualified_name():
    exchange = first_exchange()
    exchange["response"]["body"]["output"][-1] = {
        "type": "function_call",
        "call_id": "call_ns",
        "namespace": "collaboration",
        "name": "spawn_agent",
        "arguments": '{"task": "check"}',
    }
    normalized, reason = normalize_prime_sft_exchange(exchange)
    assert reason is None
    (call,) = normalized.messages[-1]["tool_calls"]
    assert call["function"]["name"] == "collaboration.spawn_agent"


def _rollout(tmp_path, records):
    trial = tmp_path / "gpqa-000__sanitized"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps({"task_name": "gpqa-000", "rewards": {"reward": 1.0}})
    )
    (trial / "trajectory/llm_trajectory.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    return trial


def test_results_jsonl_keeps_the_exchanges(tmp_path):
    trial = _rollout(tmp_path, [first_exchange(), second_exchange()])
    row = build_rollout_results_record(
        trial,
        task_name="gpqa-000",
        rollout_name=trial.name,
        agent="codex-acp",
        agent_name="codex-acp",
        model="deepseek/deepseek-flash",
        n_tool_calls=1,
        prompts=["Which option?"],
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        error=None,
        verifier_error=None,
    )
    assert row["error"] is None
    assert row["info"].get("training_ready_reason") != "export_error"
    assert len(row["trajectory"]) == 2
    completions = [step["completion"][-1] for step in row["trajectory"]]
    assert completions[0]["tool_calls"][0]["function"]["name"] == "apply_patch"
    prompt = row["trajectory"][1]["prompt"]
    assert {"role": "tool", "tool_call_id": CALL_ID, "content": RESULT} in prompt
    assert _def(row["tool_defs"], "apply_patch")["custom"] == {"format": GRAMMAR}


@pytest.mark.parametrize("export", [export_prime_sft_jsonl, export_trl_sft_jsonl])
@pytest.mark.parametrize("row_mode", ["exchange", "rollout"])
def test_directory_exports_write_the_custom_calls(tmp_path, export, row_mode):
    _rollout(tmp_path, [first_exchange(), second_exchange()])
    out = tmp_path / "out.jsonl"
    stats = export(tmp_path, out, row_mode=row_mode)
    assert stats.rows_written >= 1
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    calls = [
        call
        for row in rows
        for message in row.get("messages")
        or row.get("prompt", []) + row.get("completion", [])
        for call in message.get("tool_calls") or []
    ]
    # Prime rows carry JSON-encoded arguments; TRL rows carry the object.
    arguments = [
        json.loads(a) if isinstance(a, str) else a
        for a in (call["function"]["arguments"] for call in calls)
    ]
    assert arguments and all(a == {"input": PATCH} for a in arguments)


# ── The bridged capture ─────────────────────────────────────────────────────
# DeepSeek's route reaches the provider through LiteLLM's Responses-to-chat
# bridge: the proxy's capture keeps Codex's Responses items under
# ``messages``, the chat request sent upstream under ``input`` and ``tools``
# (``apply_patch`` a function of one required string, ``content``; namespace
# members ``namespace__name``), and the chat completion as the response.

BRIDGED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Runs a command.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Use the `apply_patch` tool to edit files.\n```lark\n"
            + GRAMMAR["definition"]
            + "```",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The apply_patch content following the "
                        "specified format",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "collaboration__spawn_agent",
            "description": "Spawn an agent.",
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string"}},
            },
        },
    },
]


def _chat(role, content, **extra):
    return {"role": role, "content": content, **extra}


def _bridged(responses_items, chat_items, message):
    return {
        "request": {
            "body": {
                "model": "deepseek-flash",
                "messages": responses_items,
                "input": chat_items,
                "tools": copy.deepcopy(BRIDGED_TOOLS),
                "reasoning_effort": "high",
                "stream": True,
                "upstream": {"thinking": {"type": "enabled"}},
            }
        },
        "response": {
            "status_code": 200,
            "body": {
                "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            },
        },
        "metadata": {"call_purpose": "agent", "call_type": "aresponses"},
    }


def _wire_call():
    return {
        "id": CALL_ID,
        "type": "function",
        "function": {
            "name": "apply_patch",
            "arguments": json.dumps({"content": PATCH}),
        },
    }


def bridged_exchanges():
    chat = [
        _chat("system", "You are Codex, a coding agent."),
        _chat(
            "user", "Which option? Write the single selected letter to /app/answer.txt."
        ),
    ]
    first = _bridged(
        _history(),
        chat,
        {
            "role": "assistant",
            "content": "Lifetime broadening decides it.",
            "reasoning_content": "Compare the linewidths.",
            "tool_calls": [_wire_call()],
            "function_call": None,
        },
    )
    second = _bridged(
        [
            *_history(),
            _call(),
            _msg(
                "assistant",
                "Lifetime broadening decides it.",
                "output_text",
                phase="commentary",
            ),
            _output(),
        ],
        [
            *chat,
            _chat(
                "assistant",
                "Lifetime broadening decides it.",
                tool_calls=[_wire_call()],
            ),
            _chat("tool", RESULT, tool_call_id=CALL_ID),
        ],
        {
            "role": "assistant",
            "content": "Wrote D to /app/answer.txt.",
            "tool_calls": [],
        },
    )
    return [first, second]


def test_a_bridged_custom_call_takes_the_wire_function_s_parameter():
    """On the bridge the model was offered apply_patch as a function of one
    required string, ``content``: the custom call and its output are written
    as that function's call and result, as the model saw them."""
    first, second = bridged_exchanges()
    normalized, reason = normalize_prime_sft_exchange(first)
    assert reason is None
    (call,) = normalized.messages[-1]["tool_calls"]
    assert json.loads(call["function"]["arguments"]) == {"content": PATCH}
    normalized, reason = normalize_prime_sft_exchange(second)
    assert reason is None
    calls = [c for m in normalized.messages for c in m.get("tool_calls") or []]
    assert [(c["id"], json.loads(c["function"]["arguments"])) for c in calls] == [
        (CALL_ID, {"content": PATCH})
    ]
    assert {"role": "tool", "tool_call_id": CALL_ID, "content": RESULT} in (
        normalized.messages
    )
    assert "custom" not in _def(normalized.tool_defs, "apply_patch")


def test_a_bridged_namespace_call_takes_the_wire_name():
    first, _ = bridged_exchanges()
    first["request"]["body"]["messages"].extend(
        [
            {
                "type": "function_call",
                "call_id": "call_ns",
                "namespace": "collaboration",
                "name": "spawn_agent",
                "arguments": '{"task": "check"}',
            },
            {"type": "function_call_output", "call_id": "call_ns", "output": "ok"},
        ]
    )
    normalized, reason = normalize_prime_sft_exchange(first)
    assert reason is None
    names = [
        c["function"]["name"]
        for m in normalized.messages
        for c in m.get("tool_calls") or []
    ]
    assert names[0] == "collaboration__spawn_agent"


def test_a_bridged_trial_s_results_jsonl_keeps_the_exchanges(tmp_path):
    trial = _rollout(tmp_path, bridged_exchanges())
    row = build_rollout_results_record(
        trial,
        task_name="gpqa-000",
        rollout_name=trial.name,
        agent="codex-acp",
        agent_name="codex-acp",
        model="deepseek/deepseek-flash",
        n_tool_calls=1,
        prompts=["Which option?"],
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        error=None,
        verifier_error=None,
    )
    assert row["error"] is None
    assert row["info"].get("training_ready_reason") != "export_error"
    # The steps results.jsonl selects (the existing selection) keep the
    # custom call and its output as the wire function call and its result.
    (step,) = [s for s in row["trajectory"] if s["extras"]["exchange_index"] == 1]
    calls = [c for m in step["prompt"] for c in m.get("tool_calls") or []]
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [
        {"content": PATCH}
    ]
    assert {"role": "tool", "tool_call_id": CALL_ID, "content": RESULT} in (
        step["prompt"]
    )


# ── What stays refused ──────────────────────────────────────────────────────


def _declared_as_other_function():
    exchange = first_exchange()
    tools = exchange["request"]["body"]["tools"]
    tools[1] = {
        "type": "function",
        "name": "apply_patch",
        "parameters": {"type": "object"},
    }
    return exchange


def _undeclared():
    exchange = first_exchange()
    del exchange["request"]["body"]["tools"][1]
    return exchange


def _orphan_output():
    exchange = second_exchange()
    for key in ("input", "messages"):
        items = exchange["request"]["body"][key]
        items[:] = [i for i in items if i.get("type") != "custom_tool_call"]
    return exchange


def _non_string_input():
    exchange = first_exchange()
    exchange["response"]["body"]["output"][-1]["input"] = {"patch": PATCH}
    return exchange


def _image_output():
    exchange = second_exchange()
    for key in ("input", "messages"):
        exchange["request"]["body"][key][-1]["output"] = [
            {"type": "input_text", "text": "see"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]
    return exchange


def _custom_in_namespace():
    exchange = first_exchange()
    tools = exchange["request"]["body"]["tools"]
    tools[2]["tools"].append(copy.deepcopy(TOOLS[1]) | {"name": "inner"})
    return exchange


def _declared_twice():
    exchange = first_exchange()
    exchange["request"]["body"]["tools"].append(
        {"type": "function", "name": "apply_patch", "parameters": {"type": "object"}}
    )
    return exchange


def _custom_in_additional_tools():
    exchange = first_exchange()
    for key in ("input", "messages"):
        exchange["request"]["body"][key].insert(
            0, {"type": "additional_tools", "tools": [copy.deepcopy(TOOLS[1])]}
        )
    return exchange


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (_declared_as_other_function, "unsupported_tool_protocol: custom_tool_call"),
        (_undeclared, "unsupported_tool_protocol: custom_tool_call"),
        (_orphan_output, "unsupported_tool_protocol: custom_tool_call_output"),
        (_non_string_input, "unsupported_tool_protocol: custom_tool_call"),
        (_image_output, "unsupported_tool_protocol: custom_tool_call_output"),
        (_custom_in_namespace, "unsupported_tool_protocol: custom"),
        (_declared_twice, "unsupported_tool_protocol: custom"),
        (_custom_in_additional_tools, "unsupported_tool_protocol: custom"),
    ],
)
def test_what_the_export_cannot_represent_is_still_refused(make, reason):
    exchange = make()
    original = copy.deepcopy(exchange)
    normalized, got = normalize_prime_sft_exchange(exchange)
    assert normalized is None
    assert got == reason
    assert PATCH not in got
    assert exchange == original
