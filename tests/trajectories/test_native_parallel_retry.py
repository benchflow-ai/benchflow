"""Guards CONFIRM-trajectories-4 and the split-turn regression at 3e69addf."""

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
    _training_success_exchange_indices,
    build_rollout_results_record,
)

KINDS = ["function_call", "tool_call"]
CONSUMERS = [
    "selector",
    "prime-exchange",
    "prime-rollout",
    "trl-exchange",
    "trl-rollout",
    "canonical",
]
LAYOUTS = [
    "native",
    "reverse",
    "chunks",
    "text-reasoning",
    "id-alias",
    "compacted",
    "mixed",
    "group-first",
    "group-last",
    "split-chat",
]


def _calls(kind, prefix, count=3):
    return [
        {
            "type": kind,
            "call_id": f"{prefix}_{i}",
            "name": "bash",
            "arguments": json.dumps({"command": command, "options": {"a": 1}}),
        }
        for i, command in enumerate(["ls", "pwd", "whoami"][:count])
    ]


def _chat_calls(calls):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call["call_id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for call in calls
        ],
    }


def _exchanges(kind, replay_kind=None, field="input", layout="native", count=3):
    replay_kind = replay_kind or kind
    prompt = [{"role": "user", "content": "List files and describe the environment"}]
    tools = [
        {
            "type": "function",
            "name": "bash",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    original = {
        "request": {"body": {field: prompt, "tools": tools}},
        "response": {
            "status_code": 200,
            "body": {"status": "completed", "output": _calls(kind, "used", count)},
        },
    }
    calls = _calls(replay_kind, "used", count)
    # Whitespace and JSON key order do not change a call's signature.
    for call in calls:
        arguments = json.loads(call["arguments"])
        call["arguments"] = json.dumps(
            dict(reversed(list(arguments.items()))), separators=(",", ":")
        )
    results = [
        {
            "type": replay_kind + "_output",
            "call_id": call["call_id"],
            "output": f"result-{i}",
        }
        for i, call in enumerate(calls)
    ]
    if layout == "reverse":
        calls.reverse()
        results.reverse()
    elif layout == "chunks":
        results.insert(1, {**results[0], "output": "more-output"})
    elif layout == "text-reasoning":
        calls = [
            {"type": "message", "role": "assistant", "content": "Checking"},
            calls[0],
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "assistant", "content": "Still checking"},
            *calls[1:],
            {"type": "message", "role": "assistant", "content": "Waiting for tools"},
        ]
    elif layout == "id-alias":
        for item in calls + results:
            item["id"] = item.pop("call_id")
    elif layout == "mixed":
        calls[1]["type"] = KINDS[1 - KINDS.index(replay_kind)]
        results[1]["type"] = calls[1]["type"] + "_output"
    elif layout == "group-first":
        calls = [_chat_calls(calls[:2]), *calls[2:]]
    elif layout == "group-last":
        calls = [calls[0], _chat_calls(calls[1:])]
    elif layout == "split-chat":
        calls = [_chat_calls([call]) for call in calls]
    replay_prompt = copy.deepcopy(prompt)
    if layout == "compacted":
        replay_prompt[0]["content"] = "Compacted history"
    followup = {
        "request": {
            "body": {field: [*replay_prompt, *calls, *results], "tools": tools}
        },
        "response": {
            "status_code": 200,
            "body": {
                "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": "Done"}],
            },
        },
    }
    late = copy.deepcopy(original)
    late["response"]["body"]["output"] = _calls(kind, "unused", count)
    return [original, followup, late]


def _write_rollout(tmp_path, exchanges):
    rollout = tmp_path / "task"
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps({"task_name": "dummy", "rewards": {"reward": 1}})
    )
    (rollout / "trajectory/llm_trajectory.jsonl").write_text(
        "".join(json.dumps(exchange) + "\n" for exchange in exchanges)
    )
    return rollout


def _assert_consumed(tmp_path, exchanges, consumer):
    if consumer == "selector":
        assert _training_success_exchange_indices(exchanges) == {0, 1}
        return
    rollout = _write_rollout(tmp_path, exchanges)
    if consumer == "canonical":
        record = build_rollout_results_record(
            rollout,
            task_name="dummy",
            rollout_name="trial",
            agent="dummy",
            agent_name="dummy",
            model=None,
            n_tool_calls=3,
            prompts=["List files and describe the environment"],
            trajectory=[],
            partial_trajectory=False,
            rewards={"reward": 1},
            error=None,
            verifier_error=None,
        )
        assert record["info"]["training_ready"] is True
        assert [step["extras"]["exchange_index"] for step in record["trajectory"]] == [
            0,
            1,
        ]
        rows = [record]
    else:
        name, mode = consumer.split("-")
        exporter = (
            convert_benchflow_rollouts_to_prime_sft_rows
            if name == "prime"
            else convert_benchflow_rollouts_to_trl_sft_rows
        )
        rows, _ = exporter(tmp_path, row_mode=mode)
        assert [row["exchange_index"] for row in rows] == (
            [0, 1] if mode == "exchange" else [1]
        )
    serialized = json.dumps(rows)
    assert "unused" not in serialized
    for i in range(len(exchanges[0]["response"]["body"]["output"])):
        assert f"used_{i}" in serialized
        assert f"result-{i}" in serialized


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("replay_kind", KINDS)
@pytest.mark.parametrize("field", ["input", "messages"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_complete_parallel_turn_selects_consumed_native_response(
    tmp_path, consumer, kind, replay_kind, field, layout
):
    """Guards 3e69addf: native replay partitions cannot change retry selection."""
    _assert_consumed(tmp_path, _exchanges(kind, replay_kind, field, layout), consumer)


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("count", [1, 2])
def test_native_single_and_reviewed_two_call_controls(tmp_path, consumer, kind, count):
    """Guards 3e69addf: retain single-call behavior and the exact review case."""
    _assert_consumed(tmp_path, _exchanges(kind, count=count), consumer)


MUTATIONS = [
    "missing-call",
    "missing-result",
    "wrong-id",
    "wrong-name",
    "wrong-arguments",
    "wrong-result-id",
    "missing-id",
    "duplicate-id",
    "duplicate-call",
    "call-role",
    "result-role",
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("field", ["input", "messages"])
@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("mutation", MUTATIONS)
def test_every_call_in_native_turn_needs_exact_identity_and_result(
    kind, field, position, mutation
):
    """Guards 3e69addf: an ID substring, subset or malformed replay cannot win."""
    exchanges = _exchanges(kind, field=field)
    history = exchanges[1]["request"]["body"][field]
    call = history[1 + position]
    result = history[4 + position]
    if mutation == "missing-call":
        del history[1 + position]
    elif mutation == "missing-result":
        del history[4 + position]
    elif mutation == "wrong-id":
        call["call_id"] = result["call_id"] = "prefix_" + call["call_id"] + "_suffix"
    elif mutation == "wrong-name":
        call["name"] = "other"
    elif mutation == "wrong-arguments":
        call["arguments"] = '{"command":"other"}'
    elif mutation == "wrong-result-id":
        result["call_id"] = "other"
    elif mutation == "missing-id":
        call.pop("call_id")
        result.pop("call_id")
    elif mutation == "duplicate-id":
        call["call_id"] = result["call_id"] = f"used_{(position + 1) % 3}"
    elif mutation == "duplicate-call":
        history.insert(1 + position, copy.deepcopy(call))
    elif mutation == "call-role":
        history[1 + position] = {**_chat_calls([call]), "role": "user"}
    else:
        history[4 + position] = {
            "role": "user",
            "tool_call_id": result["call_id"],
            "content": result["output"],
        }
    # A later capture closes the terminal-response exception for the unused retry.
    exchanges.append({"request": {"body": {field: []}}})
    assert _training_success_exchange_indices(exchanges) == {1}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("field", ["input", "messages"])
@pytest.mark.parametrize("boundary", ["user", "system", "developer", "tool", "unknown"])
def test_parallel_turn_cannot_cross_a_nonassistant_boundary(kind, field, boundary):
    """Guards 3e69addf: only one complete assistant turn proves consumption."""
    exchanges = _exchanges(kind, field=field)
    history = exchanges[1]["request"]["body"][field]
    separator = {"role": boundary, "content": "boundary"}
    if boundary == "tool":
        separator = history.pop(4)
    elif boundary == "unknown":
        separator = {"type": "unknown_item", "content": "boundary"}
    history.insert(2, separator)
    exchanges.append({"request": {"body": {field: []}}})
    assert _training_success_exchange_indices(exchanges) == {1}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("field", ["input", "messages"])
@pytest.mark.parametrize("position", [1, 4])
def test_native_turn_cannot_match_just_a_prefix_or_suffix(kind, field, position):
    """Guards 3e69addf: an extra call is part of the same indivisible turn."""
    exchanges = _exchanges(kind, field=field)
    history = exchanges[1]["request"]["body"][field]
    history.insert(position, _calls(kind, "other", 1)[0])
    history.append({"type": kind + "_output", "call_id": "other_0", "output": "extra"})
    exchanges.append({"request": {"body": {field: []}}})
    assert _training_success_exchange_indices(exchanges) == {1}
