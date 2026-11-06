"""Guards FF-sw-trajectories SFT findings at 363d815b."""

import json

import pytest

from benchflow.trajectories.export_prime_sft import (
    convert_benchflow_rollouts_to_prime_sft_rows,
    normalize_prime_sft_exchange,
)
from benchflow.trajectories.export_trl_sft import (
    convert_benchflow_rollouts_to_trl_sft_rows,
)

EXPORTERS = [
    convert_benchflow_rollouts_to_prime_sft_rows,
    convert_benchflow_rollouts_to_trl_sft_rows,
]


def write_rollout(tmp_path, exchanges):
    rollout = tmp_path / "task"
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps({"task_name": "dummy", "rewards": {"reward": 1}})
    )
    (rollout / "trajectory" / "llm_trajectory.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in exchanges)
    )


@pytest.mark.parametrize("exporter", EXPORTERS)
@pytest.mark.parametrize("results_shape", ["text", "blocks", "parallel"])
def test_anthropic_history_keeps_instruction_and_tool_ids(
    tmp_path, exporter, results_shape
):
    """Guards the fix across Prime and TRL windows."""
    calls = ["one", "two"] if results_shape == "parallel" else ["one"]
    schema = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
    exchange = {
        "request": {
            "path": "/v1/messages",
            "body": {
                "system": "Be helpful",
                "messages": [
                    {"role": "user", "content": "List files"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": ident,
                                "name": "bash",
                                "input": {"command": "ls"},
                            }
                            for ident in calls
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": ident,
                                "content": "README.md"
                                if results_shape == "text"
                                else [{"type": "text", "text": "README.md"}],
                            }
                            for ident in calls
                        ],
                    },
                ],
                "tools": [{"name": "bash", "input_schema": schema}],
            },
        },
        "response": {
            "status_code": 200,
            "body": {"content": [{"type": "text", "text": "Done"}]},
        },
    }
    normalized, reason = normalize_prime_sft_exchange(exchange)
    assert reason is None
    assert normalized is not None
    assert normalized.tool_defs[0]["function"]["parameters"] == schema
    assert [
        m["tool_call_id"] for m in normalized.messages if m["role"] == "tool"
    ] == calls
    assert normalized.messages[0] == {"role": "system", "content": "Be helpful"}
    write_rollout(tmp_path, [exchange])
    rows, _ = exporter(tmp_path, row_mode="exchange")
    assert len(rows) == 1
    messages = rows[0].get("messages") or rows[0]["prompt"] + rows[0]["completion"]
    assert next(m for m in messages if m["role"] == "user")["content"] == "List files"
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == calls


def test_orphan_native_tool_result_is_not_relabelled_as_user_text():
    """Guards FF-sw-trajectories against silently accepting cropped tool history."""
    exchange = {
        "request": {
            "body": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "missing",
                                "content": "dummy result",
                            }
                        ],
                    }
                ]
            }
        },
        "response": {
            "status_code": 200,
            "body": {"content": [{"type": "text", "text": "Done"}]},
        },
    }
    normalized, reason = normalize_prime_sft_exchange(exchange)
    assert normalized is None
    assert "unknown tool_call_id" in reason
