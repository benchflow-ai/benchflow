"""Guard consumed-call identity against candidate ac3bedaa1's retry selection."""

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


def _call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _assistant(calls):
    return {"role": "assistant", "content": "", "tool_calls": calls}


def _result(call_id):
    return {"role": "tool", "tool_call_id": call_id, "content": "done"}


def _exchange(messages, tool_name, reply):
    return {
        "request": {
            "body": {
                "model": "test-model",
                "messages": messages,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        },
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "message": reply,
                        "finish_reason": "tool_calls"
                        if reply.get("tool_calls")
                        else "stop",
                    }
                ]
            },
        },
    }


def _independent_subagents():
    parent = [{"role": "user", "content": "Check this derivation independently."}]
    child = [{"role": "user", "content": "Check this derivation."}]
    spawns = _assistant(
        [_call(f"spawn{i}", "Agent", {"prompt": child[0]["content"]}) for i in (1, 2)]
    )
    calls = [
        _assistant([_call(f"child{i}", "bash", {"command": "python check.py"})])
        for i in (1, 2)
    ]
    done = {"role": "assistant", "content": "The derivation is correct."}
    return [
        _exchange(parent, "Agent", spawns),
        *(_exchange(child, "bash", call) for call in calls),
        *(
            _exchange([*child, call, _result(f"child{i}")], "bash", done)
            for i, call in enumerate(calls, 1)
        ),
        _exchange(
            [*parent, spawns, _result("spawn1"), _result("spawn2")], "Agent", done
        ),
    ]


def _rollout(tmp_path, records):
    rollout = tmp_path / "physics__trial"
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps({"task_name": "physics", "rewards": {"reward": 1}})
    )
    (rollout / "trajectory/llm_trajectory.jsonl").write_text(
        "\n".join(map(json.dumps, records))
    )
    (rollout / "trajectory/acp_trajectory.jsonl").write_text(
        "\n".join(
            json.dumps(
                {"tool_call_id": f"child{i}", "parent_tool_call_id": f"spawn{i}"}
            )
            for i in (1, 2)
        )
    )
    return rollout


@pytest.mark.parametrize(
    "convert",
    [
        convert_benchflow_rollouts_to_prime_sft_rows,
        convert_benchflow_rollouts_to_trl_sft_rows,
    ],
)
def test_independent_consumed_subagent_calls_survive_conversion(tmp_path, convert):
    """Guards ac3bedaa1: identical prompts do not identify one retry conversation."""
    rollout = _rollout(tmp_path, _independent_subagents())
    capture = rollout / "trajectory/llm_trajectory.jsonl"
    original = capture.read_bytes()
    rows, stats = convert(tmp_path, row_mode="exchange", subagent_rows=True)
    assert sorted(row["exchange_index"] for row in rows) == list(range(6))
    owners = {row["exchange_index"]: row.get("parent_tool_call_id") for row in rows}
    assert owners == {
        0: None,
        1: "spawn1",
        2: "spawn2",
        3: "spawn1",
        4: "spawn2",
        5: None,
    }
    assert stats.subagents.subagent_rows_written == 4
    assert stats.subagents.subagent_exchanges_excluded == 0
    assert capture.read_bytes() == original


def test_results_steps_retain_each_consumed_subagent(tmp_path):
    """Guards ac3bedaa1: results and directory exports select the same consumed calls."""
    rollout = _rollout(tmp_path, _independent_subagents())
    steps, _, error = _llm_steps_from_trajectory(
        rollout, reward=1.0, is_truncated=False, trajectory_id_prefix="physics"
    )
    assert error is None
    assert [step["extras"]["exchange_index"] for step in steps] == list(range(6))
    assert steps[1]["extras"]["parent_tool_call_id"] == "spawn1"
    assert steps[2]["extras"]["parent_tool_call_id"] == "spawn2"


def test_duplicate_capture_of_one_consumed_call_is_selected_once():
    """Guards ac3bedaa1: keeping consumed branches must not duplicate one capture."""
    records = _independent_subagents()
    records.insert(2, copy.deepcopy(records[1]))
    assert _training_success_exchange_indices(records) == {0, 2, 3, 4, 5, 6}


def test_unconsumed_retry_does_not_replace_either_consumed_branch():
    """Guards ac3bedaa1: a late retry still loses to responses actually consumed."""
    records = _independent_subagents()
    retry = copy.deepcopy(records[1])
    retry["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["id"] = "unused"
    records.append(retry)
    assert _training_success_exchange_indices(records) == set(range(6))


@pytest.mark.parametrize(
    "convert",
    [
        convert_benchflow_rollouts_to_prime_sft_rows,
        convert_benchflow_rollouts_to_trl_sft_rows,
    ],
)
def test_default_conversion_still_excludes_subagent_calls(tmp_path, convert):
    """Guards ac3bedaa1: preserving identity does not admit child rows by default."""
    _rollout(tmp_path, _independent_subagents())
    rows, stats = convert(tmp_path, row_mode="exchange")
    assert [row["exchange_index"] for row in rows] == [0, 5]
    assert stats.subagents.subagent_exchanges_excluded == 4
