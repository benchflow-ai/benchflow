"""Guards CONFIRM-trajectories-2 against the native retry gap at fe70610b."""

import copy
import json

import pytest

from benchflow.trajectories.export_prime_sft import (
    convert_benchflow_rollouts_to_prime_sft_rows,
)
from benchflow.trajectories.export_trl_sft import (
    convert_benchflow_rollouts_to_trl_sft_rows,
)
from benchflow.trajectories.results import (
    _llm_steps_from_trajectory,
    _training_success_exchange_indices,
)

SHAPES = [
    "anthropic",
    "chat",
    "message",
    "legacy",
    "responses-function",
    "responses-tool",
]
EXPORTERS = [
    convert_benchflow_rollouts_to_prime_sft_rows,
    convert_benchflow_rollouts_to_trl_sft_rows,
]


def response_body(shape, ids, name="bash"):
    calls = [
        {"id": ident, "type": "function", "function": {"name": name, "arguments": "{}"}}
        for ident in ids
    ]
    message = {"role": "assistant", "content": "Calling tool", "tool_calls": calls}
    if shape == "anthropic":
        return {
            "content": [
                {"type": "text", "text": "Calling tool"},
                *[
                    {"type": "tool_use", "id": ident, "name": name, "input": {}}
                    for ident in ids
                ],
            ],
            "stop_reason": "tool_use",
        }
    if shape.startswith("responses-"):
        return {
            "status": "completed",
            "output": [
                {
                    "type": "function_call"
                    if shape.endswith("function")
                    else "tool_call",
                    "call_id": ident,
                    "name": name,
                    "arguments": "{}",
                }
                for ident in ids
            ],
        }
    if shape == "legacy":
        message = {
            "role": "assistant",
            "content": "Calling tool",
            "function_call": {"id": ids[0], "name": name, "arguments": "{}"},
        }
    if shape == "message":
        return {"message": message}
    return {
        "choices": [
            {
                "message": message,
                "finish_reason": "function_call" if shape == "legacy" else "tool_calls",
            }
        ]
    }


def exchanges_for(shape, parallel=False, name="bash"):
    ids = ["used_1", "used_2"] if parallel and shape != "legacy" else ["used_1"]
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    messages = [{"role": "user", "content": "List files"}]
    if shape == "anthropic":
        tools = [{"name": name, "input_schema": {"type": "object", "properties": {}}}]
    first = {
        "request": {"body": {"messages": messages, "tools": tools}},
        "response": {"status_code": 200, "body": response_body(shape, ids, name)},
    }
    if shape == "anthropic":
        history = [
            {"role": "assistant", "content": first["response"]["body"]["content"]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": ident, "content": "README"}
                    for ident in ids
                ],
            },
        ]
        done = {
            "content": [{"type": "text", "text": "Done"}],
            "stop_reason": "end_turn",
        }
    else:
        history = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": ident,
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                    for ident in ids
                ],
            },
            *[
                {"role": "tool", "tool_call_id": ident, "content": "README"}
                for ident in ids
            ],
        ]
        done = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Done"},
                    "finish_reason": "stop",
                }
            ]
        }
    followup = {
        "request": {"body": {"messages": [*messages, *history], "tools": tools}},
        "response": {"status_code": 200, "body": done},
    }
    late = copy.deepcopy(first)
    late["response"]["body"] = response_body(
        shape, [ident.replace("used", "unused") for ident in ids], name
    )
    return [first, followup, late]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("parallel", [False, True], ids=["single", "parallel"])
def test_shared_selector_prefers_consumed_response(shape, parallel):
    """Guards fe70610b across every response form accepted by normalization."""
    assert _training_success_exchange_indices(exchanges_for(shape, parallel)) == {0, 1}


@pytest.mark.parametrize("exporter", EXPORTERS)
@pytest.mark.parametrize("row_mode", ["exchange", "rollout"])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("parallel", [False, True], ids=["single", "parallel"])
def test_exporters_and_results_select_the_same_consumed_retry(
    tmp_path, exporter, row_mode, shape, parallel
):
    """Guards fe70610b: both directory modes and results.jsonl agree."""
    rollout = tmp_path / "task"
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps({"task_name": "dummy", "rewards": {"reward": 1}})
    )
    (rollout / "trajectory/llm_trajectory.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in exchanges_for(shape, parallel))
    )
    rows, _ = exporter(tmp_path, row_mode=row_mode)
    assert [row["exchange_index"] for row in rows] == (
        [0, 1] if row_mode == "exchange" else [1]
    )
    assert "unused" not in json.dumps(rows)
    steps, _, error = _llm_steps_from_trajectory(
        rollout, reward=1, is_truncated=False, trajectory_id_prefix="dummy"
    )
    assert error is None
    assert [step["extras"]["exchange_index"] for step in steps] == [0, 1]
    assert "unused" not in json.dumps(steps)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name", ["bash", "finish"])
def test_unconsumed_tool_policy_uses_normalized_names(shape, name):
    """Guards fe70610b: finish calls and terminal captures keep their policy."""
    first = exchanges_for(shape, name=name)[0]
    unrelated = {
        "request": {"body": {"messages": [{"role": "user", "content": "New task"}]}},
        "response": {
            "status_code": 200,
            "body": {
                "content": [{"type": "text", "text": "Done"}],
                "stop_reason": "end_turn",
            },
        },
    }
    assert _training_success_exchange_indices([first]) == {0}
    assert _training_success_exchange_indices([first, unrelated]) == (
        {0, 1} if name == "finish" else {1}
    )
