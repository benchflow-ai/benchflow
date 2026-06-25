"""Prime-RL SFT conversion from BenchFlow LLM trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.trajectories.export_prime_sft import (
    convert_benchflow_rollouts_to_prime_sft_rows,
    export_prime_sft_jsonl,
    validate_prime_sft_jsonl,
)


def _write_rollout(
    rollout_dir: Path,
    *,
    reward: float = 1.0,
    exchanges: list[dict] | None = None,
) -> None:
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "demo-task",
                "agent": "openhands",
                "rewards": {"reward": reward},
                "agent_result": {"total_tokens": 123, "n_tool_calls": 1},
            }
        )
    )
    traj = rollout_dir / "trajectory"
    traj.mkdir()
    records = (
        exchanges
        if exchanges is not None
        else [_exchange(final=False), _exchange(final=True)]
    )
    (traj / "llm_trajectory.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )


def _exchange(*, final: bool) -> dict:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an agent."}]},
        {"role": "user", "content": "List files."},
    ]
    if final:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": {"command": "ls"},
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "README.md"},
            ]
        )
    return {
        "request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {
                "model": "test-model",
                "messages": messages,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "description": "Run shell commands.",
                            "parameters": {
                                "type": "object",
                                "properties": {"command": {"type": "string"}},
                                "required": ["command"],
                            },
                        },
                    }
                ],
            },
        },
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Done.",
                            "provider_specific_fields": {"refusal": None},
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        },
        "duration_ms": 10,
    }


def test_convert_rollout_mode_uses_final_successful_exchange(tmp_path: Path) -> None:
    """Guards this PR's Prime-RL converter against dropping tool context."""
    _write_rollout(tmp_path / "job" / "rollout-1")

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(tmp_path / "job")

    assert stats.rows_written == 1
    row = rows[0]
    assert row["task_name"] == "demo-task"
    assert row["source"] == "benchflow-llm-trajectory"
    assert row["reward"] == 1.0
    assert row["tool_defs"][0]["function"]["name"] == "bash"
    assert row["messages"][0] == {"role": "system", "content": "You are an agent."}
    assert any(message.get("role") == "tool" for message in row["messages"])
    assert row["messages"][-1] == {"role": "assistant", "content": "Done."}


def test_exchange_mode_writes_one_row_per_successful_exchange(tmp_path: Path) -> None:
    """Guards this PR's --row-mode exchange behavior."""
    _write_rollout(tmp_path / "job" / "rollout-1")

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(
        tmp_path / "job",
        row_mode="exchange",
    )

    assert stats.rows_written == 2
    assert [row["exchange_index"] for row in rows] == [0, 1]


def test_convert_filters_by_min_reward(tmp_path: Path) -> None:
    """Guards this PR's reward-filtering gate."""
    _write_rollout(tmp_path / "job" / "bad", reward=0.4)

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(
        tmp_path / "job",
        min_reward=1.0,
    )

    assert rows == []
    assert stats.skipped_reward == 1


def test_export_and_validate_prime_sft_jsonl(tmp_path: Path) -> None:
    """Guards this PR's end-to-end JSONL write and validation path."""
    _write_rollout(tmp_path / "job" / "rollout-1")
    out = tmp_path / "train.jsonl"
    manifest = tmp_path / "manifest.json"

    stats = export_prime_sft_jsonl(
        tmp_path / "job",
        out,
        expected_rows=1,
        manifest=manifest,
    )

    assert stats.rows_written == 1
    assert validate_prime_sft_jsonl(out, expected_rows=1) == {
        "ok": True,
        "rows": 1,
        "rows_with_tool_calls": 1,
    }
    assert json.loads(manifest.read_text())["rows_written"] == 1


def test_validate_rejects_banned_message_keys(tmp_path: Path) -> None:
    """Guards this PR's leakage-key validation before Prime-RL ingestion."""
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "secret",
                        "reasoning_content": "do not train",
                    },
                ]
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="banned keys"):
        validate_prime_sft_jsonl(path)


def test_validate_accepts_prime_rollout_prompt_completion_rows(tmp_path: Path) -> None:
    """Guards PR #828: results.jsonl rows are valid Prime-RL SFT inputs too."""
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "do it"}],
                "completion": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "finish",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                ],
                "tool_defs": [
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
        + "\n"
    )

    assert validate_prime_sft_jsonl(path, expected_rows=1) == {
        "ok": True,
        "rows": 1,
        "rows_with_tool_calls": 1,
    }


def test_convert_openhands_responses_shape_preserves_tool_calls(tmp_path: Path) -> None:
    """Guards PR #828: OpenHands mixed chat/request Responses output is structured."""
    rollout = tmp_path / "job" / "openhands__abc123"
    rollout.mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "task_name": "openhands-task",
                "rewards": {"reward": 1.0},
                "agent_result": {"total_tokens": 10},
            }
        )
    )
    traj = rollout / "trajectory"
    traj.mkdir()
    (traj / "llm_trajectory.jsonl").write_text(
        json.dumps(
            {
                "request": {
                    "body": {
                        "model": "gpt-5.4-mini",
                        "messages": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": "Create the file.",
                                    }
                                ],
                            },
                            {
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "file_editor",
                                "arguments": '{"command":"create"}',
                            },
                            {
                                "type": "function_call_output",
                                "call_id": "call_1",
                                "output": "File created.",
                            },
                        ],
                        "tools": [
                            {
                                "type": "function",
                                "name": "file_editor",
                                "description": "Edit files.",
                                "parameters": {"type": "object", "properties": {}},
                            }
                        ],
                    }
                },
                "response": {
                    "status_code": 200,
                    "body": {
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "Done.",
                                    }
                                ],
                            }
                        ],
                        "usage": {"total_tokens": 10},
                    },
                },
            }
        )
        + "\n"
    )

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(tmp_path / "job")

    assert stats.rows_written == 1
    row = rows[0]
    assert row["tool_defs"][0]["function"]["name"] == "file_editor"
    assert row["messages"][1]["role"] == "assistant"
    assert row["messages"][1]["tool_calls"][0]["function"]["name"] == "file_editor"
    assert row["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "File created.",
    }
    assert row["messages"][-1] == {"role": "assistant", "content": "Done."}
