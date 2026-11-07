"""Guards CONFIRM-trajectories-3 against false consumption at 65a3ab1a."""

import copy
import json

import pytest

from benchflow.trajectories.results import (
    _llm_steps_from_trajectory,
    _training_success_exchange_indices,
)
from tests.trajectories.test_native_retry_selection import (
    EXPORTERS,
    SHAPES,
    exchanges_for,
)


def _legacy_exchanges(history_format):
    exchanges = exchanges_for("legacy")
    for index, command in [(0, "ls"), (2, "pwd")]:
        call = exchanges[index]["response"]["body"]["choices"][0]["message"][
            "function_call"
        ]
        call.pop("id")
        call["arguments"] = json.dumps({"command": command, "options": {"a": 1}})
    body = copy.deepcopy(exchanges[1]["request"]["body"])
    exchanges[1]["request"]["body"] = body
    call = body["messages"][1]["tool_calls"][0]
    call["id"] = "runtime_used"
    # Equivalent JSON with a different key order and whitespace.
    call["function"]["arguments"] = '{"options":{"a":1},"command":"ls"}'
    body["messages"][2]["tool_call_id"] = "runtime_used"
    if history_format == "function":
        body["messages"][1] = {
            "role": "assistant",
            "function_call": copy.deepcopy(call["function"]),
        }
        body["messages"][2] = {
            "role": "function",
            "name": "bash",
            "content": "README",
        }
    elif history_format == "responses":
        body["input"] = [
            body["messages"][0],
            {
                "type": "function_call",
                "call_id": "runtime_used",
                **call["function"],
            },
            {
                "type": "function_call_output",
                "call_id": "runtime_used",
                "output": "README",
            },
        ]
        del body["messages"]
        exchanges[1]["response"]["body"] = {
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": "Done"}],
        }
    elif history_format == "anthropic":
        body["messages"][1:] = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "runtime_used",
                        "name": "bash",
                        "input": {"options": {"a": 1}, "command": "ls"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "runtime_used",
                        "content": "README",
                    }
                ],
            },
        ]
    return exchanges


def _mention_exchanges(shape, carrier):
    exchanges = exchanges_for(shape)
    mentioned = "unused_1"
    messages = [{"role": "user", "content": "Unrelated task"}]
    body = {"messages": messages}
    if carrier in {"user", "assistant", "system", "developer"}:
        messages.append({"role": carrier, "content": f"Mention {mentioned} only"})
    elif carrier == "metadata":
        body["metadata"] = {"note": mentioned}
    elif carrier == "schema":
        body["tools"] = [{"type": "function", "function": {"name": mentioned}}]
    elif carrier == "instructions":
        body["instructions"] = mentioned
    elif carrier == "input-text":
        body = {"input": f"Mention {mentioned} only"}
    elif carrier == "tool-content":
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "other", "function": {"name": "bash", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "other", "content": mentioned},
            ]
        )
    elif carrier == "argument":
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "other",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"x": mentioned}),
                        },
                    }
                ],
            }
        )
    elif carrier == "call-only":
        messages.extend(
            copy.deepcopy(exchanges_for("chat")[1]["request"]["body"]["messages"][1:2])
        )
        messages[1]["tool_calls"][0]["id"] = mentioned
    elif carrier == "result-only":
        messages.append(
            {"role": "tool", "tool_call_id": mentioned, "content": "README"}
        )
    elif carrier == "substring-id":
        messages.extend(
            copy.deepcopy(exchanges_for("chat")[1]["request"]["body"]["messages"][1:])
        )
        messages[1]["tool_calls"][0]["id"] = "prefix_" + mentioned + "_suffix"
        messages[2]["tool_call_id"] = "prefix_" + mentioned + "_suffix"
    exchanges.append({"request": {"body": body}, "response": {"status_code": 500}})
    return exchanges


@pytest.mark.parametrize(
    "history_format", ["chat", "function", "responses", "anthropic"]
)
def test_idless_legacy_call_matches_executed_history(history_format):
    """Guards 65a3ab1a: runtime IDs cannot hide a consumed legacy response."""
    assert _training_success_exchange_indices(_legacy_exchanges(history_format)) == {
        0,
        1,
    }


CARRIERS = [
    "user",
    "assistant",
    "system",
    "developer",
    "metadata",
    "schema",
    "instructions",
    "input-text",
    "tool-content",
    "argument",
    "call-only",
    "result-only",
    "substring-id",
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("carrier", CARRIERS)
def test_request_text_and_unpaired_ids_do_not_consume_response(shape, carrier):
    """Guards 65a3ab1a: only a structural call/result pair is consumption."""
    assert _training_success_exchange_indices(_mention_exchanges(shape, carrier)) == {
        0,
        1,
    }


@pytest.mark.parametrize("history_format", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize(
    "mutation", ["arguments", "name", "prompt", "orphan", "unexecuted", "old-history"]
)
def test_legacy_matching_requires_same_call_and_extending_history(
    history_format, mutation
):
    """Guards 65a3ab1a: ID-less fallback fails closed on a different history."""
    exchanges = _legacy_exchanges(history_format)
    body = exchanges[1]["request"]["body"]
    history = body.get("messages", body.get("input"))
    if mutation == "arguments":
        if history_format == "chat":
            history[1]["tool_calls"][0]["function"]["arguments"] = '{"command":"pwd"}'
        elif history_format == "responses":
            history[1]["arguments"] = '{"command":"pwd"}'
        else:
            history[1]["content"][0]["input"] = {"command": "pwd"}
    elif mutation == "name":
        if history_format == "chat":
            history[1]["tool_calls"][0]["function"]["name"] = "other"
        elif history_format == "responses":
            history[1]["name"] = "other"
        else:
            history[1]["content"][0]["name"] = "other"
    elif mutation == "prompt":
        history[0]["content"] = "Different task"
    elif mutation == "orphan":
        del history[1]
    elif mutation == "unexecuted":
        del history[2]
    else:
        # Identical calls in the original prompt do not consume this response.
        exchanges[0]["request"]["body"] = copy.deepcopy(body)
        exchanges[2]["request"]["body"] = copy.deepcopy(body)
    exchanges.append({"request": {"body": {"messages": []}}})
    assert _training_success_exchange_indices(exchanges) == {1}


@pytest.mark.parametrize("shape", ["chat", "anthropic", "responses-function"])
def test_every_parallel_call_requires_its_result(shape):
    """Guards 65a3ab1a: a partially executed response is not fully consumed."""
    exchanges = exchanges_for(shape, parallel=True)
    history = exchanges[1]["request"]["body"]["messages"]
    if shape == "anthropic":
        history[2]["content"].pop()
    else:
        history.pop()
    exchanges.append({"request": {"body": {"messages": []}}})
    assert _training_success_exchange_indices(exchanges) == {1}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "history_format",
    ["chat", "function", "anthropic", "responses-function", "responses-tool"],
)
def test_explicit_ids_match_only_structural_history_across_protocols(
    shape, history_format
):
    """Guards 65a3ab1a across native requests as well as response forms."""
    exchanges = exchanges_for(shape)
    body = exchanges[1]["request"]["body"]
    if history_format == "anthropic":
        body["messages"] = copy.deepcopy(
            exchanges_for("anthropic")[1]["request"]["body"]["messages"]
        )
    else:
        body["messages"] = copy.deepcopy(
            exchanges_for("chat")[1]["request"]["body"]["messages"]
        )
    if history_format == "function":
        body["messages"][1] = {
            "role": "assistant",
            "function_call": {"id": "used_1", "name": "bash", "arguments": "{}"},
        }
        body["messages"][2] = {"role": "function", "name": "bash", "content": "README"}
    elif history_format.startswith("responses-"):
        kind = history_format.split("-", 1)[1] + "_call"
        body["input"] = [
            body["messages"][0],
            {"type": kind, "call_id": "used_1", "name": "bash", "arguments": "{}"},
            {"type": kind + "_output", "call_id": "used_1", "output": "README"},
        ]
        del body["messages"]
    assert _training_success_exchange_indices(exchanges) == {0, 1}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "mutation",
    [
        "missing-response-id",
        "wrong-id",
        "wrong-name",
        "wrong-arguments",
        "call-role",
        "result-role",
        "wrong-result-id",
        "result-first",
    ],
)
def test_ids_and_signatures_cannot_be_forged_by_other_history_fields(shape, mutation):
    """Guards 65a3ab1a: generated IDs and partial matches carry no evidence."""
    exchanges = exchanges_for(shape)
    history = copy.deepcopy(exchanges_for("chat")[1]["request"]["body"]["messages"])
    exchanges[1]["request"]["body"]["messages"] = history
    call = history[1]["tool_calls"][0]
    if mutation == "missing-response-id":
        response = exchanges[0]["response"]["body"]
        if shape == "anthropic":
            response["content"][1].pop("id")
        elif shape.startswith("responses-"):
            response["output"][0].pop("call_id")
        elif shape == "message":
            response["message"]["tool_calls"][0].pop("id")
        elif shape == "legacy":
            response["choices"][0]["message"]["function_call"].pop("id")
            # Legacy may match name/arguments, but a fake synthetic-ID pair
            # with a different call must still be rejected.
            call["function"]["arguments"] = '{"command":"other"}'
        else:
            response["choices"][0]["message"]["tool_calls"][0].pop("id")
        call["id"] = history[2]["tool_call_id"] = "call_000000"
    elif mutation == "wrong-id":
        call["id"] = history[2]["tool_call_id"] = "runtime_other"
    elif mutation == "wrong-name":
        call["function"]["name"] = "other"
    elif mutation == "wrong-arguments":
        call["function"]["arguments"] = '{"command":"other"}'
    elif mutation == "call-role":
        history[1]["role"] = "user"
    elif mutation == "result-role":
        history[2]["role"] = "user"
    elif mutation == "wrong-result-id":
        history[2]["tool_call_id"] = "other"
    else:
        history[1], history[2] = history[2], history[1]
    exchanges.append({"request": {"body": {"messages": []}}})
    assert _training_success_exchange_indices(exchanges) == {1}


@pytest.mark.parametrize("history_format", ["chat", "anthropic", "responses"])
def test_idless_legacy_response_does_not_supply_missing_modern_history_ids(
    history_format,
):
    """Guards 65a3ab1a: only named legacy results may omit a protocol ID."""
    exchanges = _legacy_exchanges(history_format)
    body = exchanges[1]["request"]["body"]
    if history_format == "chat":
        body["messages"][1]["tool_calls"][0].pop("id")
        body["messages"][2]["tool_call_id"] = ""
    elif history_format == "anthropic":
        body["messages"][1]["content"][0].pop("id")
        body["messages"][2]["content"][0]["tool_use_id"] = ""
    else:
        body["input"][1].pop("call_id")
        body["input"][2]["call_id"] = ""
    exchanges.append({"request": {"body": {"messages": []}}})
    assert _training_success_exchange_indices(exchanges) == {1}


@pytest.mark.parametrize("shape", SHAPES)
def test_protocol_ids_survive_compacted_prompt_history(shape):
    """Guards 65a3ab1a: exact call/result IDs remain evidence after compaction."""
    exchanges = exchanges_for(shape)
    exchanges[1]["request"]["body"]["messages"][0] = {
        "role": "user",
        "content": "Compacted conversation",
    }
    assert _training_success_exchange_indices(exchanges) == {0, 1}


@pytest.mark.parametrize("shape", ["chat", "anthropic", "responses-function"])
@pytest.mark.parametrize(
    "replay_format", ["reverse-calls", "reverse-results", "chunked-result"]
)
def test_parallel_consumption_preserves_call_identity_and_result_chunks(
    shape, replay_format
):
    """Guards 65a3ab1a while preserving supported parallel result histories."""
    exchanges = exchanges_for(shape, parallel=True)
    body = exchanges[1]["request"]["body"]
    history = body["messages"]
    if shape == "anthropic":
        calls = history[1]["content"][1:]
        results = history[2]["content"]
        if replay_format == "reverse-calls":
            history[1]["content"][1:] = list(reversed(calls))
        elif replay_format == "reverse-results":
            results.reverse()
        else:
            results.insert(1, copy.deepcopy(results[0]))
    elif replay_format == "reverse-calls":
        history[1]["tool_calls"].reverse()
    elif replay_format == "reverse-results":
        history[2:] = list(reversed(history[2:]))
    else:
        history.insert(3, copy.deepcopy(history[2]))
    assert _training_success_exchange_indices(exchanges) == {0, 1}


@pytest.mark.parametrize("exporter", EXPORTERS)
@pytest.mark.parametrize("row_mode", ["exchange", "rollout"])
@pytest.mark.parametrize(
    "case",
    [
        "legacy-chat",
        "legacy-function",
        "legacy-responses",
        "legacy-anthropic",
        "mention-anthropic",
        "mention-chat",
        "mention-responses-function",
    ],
)
def test_exporters_and_canonical_results_follow_structural_consumption(
    tmp_path, exporter, row_mode, case
):
    """Guards 65a3ab1a through both exporters and canonical results steps."""
    kind, shape = case.split("-", 1)
    exchanges = (
        _legacy_exchanges(shape)
        if kind == "legacy"
        else _mention_exchanges(shape, "user")
    )
    rollout = tmp_path / "task"
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps({"task_name": "dummy", "rewards": {"reward": 1}})
    )
    (rollout / "trajectory/llm_trajectory.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in exchanges)
    )
    rows, _ = exporter(tmp_path, row_mode=row_mode)
    assert [row["exchange_index"] for row in rows] == (
        [0, 1] if row_mode == "exchange" else [1]
    )
    steps, _, error = _llm_steps_from_trajectory(
        rollout, reward=1, is_truncated=False, trajectory_id_prefix="dummy"
    )
    assert error is None
    assert [step["extras"]["exchange_index"] for step in steps] == [0, 1]
