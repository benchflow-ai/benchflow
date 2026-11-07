"""TRL rows never train a tool call whose input was not a JSON object.

TRL trains on a row's completion. In a 30-call Claude Code run on MiMo v2.6
flash through the proxy, 4 responses carried a tool call whose input the
provider's adapter could not read, wrapped as ``{"__unparsedToolInput":
...}``, and 2 of them became completions of the per-call TRL rows: TRL would
have taught the model to emit the wrapper. Such a call, or one whose arguments
are not JSON (which the exporter wraps as ``_malformed_json_arguments``), is
left out of the completions and counted; it stays in later rows' prompts,
where it is what happened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchflow.trajectories.export_trl_sft import (
    convert_benchflow_rollouts_to_trl_sft_rows,
)

TOOL = {
    "type": "function",
    "function": {"name": "Write", "parameters": {"type": "object", "properties": {}}},
}


def _call(call_id: str, arguments: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "Write", "arguments": arguments},
    }


def _job(tmp_path: Path) -> Path:
    """Three calls of one conversation, each request carrying the turns before it."""
    rollout = tmp_path / "jobs" / "run" / "essay__abc123"
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "task_name": "essay",
                "agent": "claude-agent-acp",
                "rewards": {"reward": 1.0},
            }
        )
    )
    wrapped = json.dumps(
        {"__unparsedToolInput": json.dumps({"raw": '{"file_path": "/app/essay.txt"'})}
    )
    calls = [
        _call("call_1", json.dumps({"file_path": "/app/essay.txt", "content": "text"})),
        _call("call_2", wrapped),
        _call("call_3", '{"file_path": "/app/essay.txt", "content": '),
    ]
    history: list[dict[str, Any]] = [{"role": "user", "content": "write the essay"}]
    lines = []
    for call in calls:
        reply = {"role": "assistant", "content": "", "tool_calls": [call]}
        lines.append(
            {
                "request": {
                    "body": {"model": "m", "messages": list(history), "tools": [TOOL]}
                },
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"finish_reason": "tool_calls", "message": reply}]
                    },
                },
                "duration_ms": 1,
            }
        )
        history += [
            reply,
            {"role": "tool", "tool_call_id": call["id"], "content": "ok"},
        ]
    (rollout / "trajectory" / "llm_trajectory.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    return tmp_path / "jobs"


def test_per_call_rows_leave_out_unparsed_tool_input(tmp_path):
    rows, stats = convert_benchflow_rollouts_to_trl_sft_rows(
        _job(tmp_path), row_mode="exchange"
    )
    assert [row["exchange_index"] for row in rows] == [0]
    assert rows[0]["completion"][0]["tool_calls"][0]["function"]["arguments"] == {
        "file_path": "/app/essay.txt",
        "content": "text",
    }
    assert stats.skipped_unparsed_tool_calls == 2
    assert stats.as_dict()["skipped_unparsed_tool_calls"] == 2


def test_a_rollout_row_ending_on_unparsed_tool_input_is_left_out(tmp_path):
    rows, stats = convert_benchflow_rollouts_to_trl_sft_rows(
        _job(tmp_path), row_mode="rollout"
    )
    assert rows == []
    assert stats.skipped_unparsed_tool_calls == 1
