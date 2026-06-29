"""Prime-RL SFT conversion from BenchFlow LLM trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.trajectories.export_prime_sft import (
    PrimeSftTrajectoryJsonlError,
    convert_benchflow_rollouts_to_prime_sft_rows,
    export_prime_sft_jsonl,
    load_llm_trajectory_jsonl,
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


def _anthropic_exchange(*, status_code: int = 200) -> dict:
    """An Anthropic /v1/messages exchange whose response carries content blocks
    (text + tool_use), the shape the chat-response fallback used to flatten."""
    return {
        "request": {
            "method": "POST",
            "path": "/v1/messages",
            "body": {
                "model": "claude-test",
                "messages": [{"role": "user", "content": "List files."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "description": "Run shell commands.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        },
        "response": {
            "status_code": status_code,
            "body": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me list them."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "bash",
                        "input": {"command": "ls"},
                    },
                ],
            },
        },
        "duration_ms": 10,
    }


def test_anthropic_tool_use_content_preserved_as_tool_calls(tmp_path: Path) -> None:
    """Guards #828 Codex P2: Anthropic tool_use blocks must export as tool_calls,
    not be flattened to plain text (which corrupts tool-using training data)."""
    _write_rollout(tmp_path / "job" / "rollout-1", exchanges=[_anthropic_exchange()])

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(tmp_path / "job")

    assert stats.rows_written == 1
    assistant = rows[0]["messages"][-1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Let me list them."
    assert assistant["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "command": "ls"
    }
    assert stats.rows_with_tool_calls == 1


def test_skipped_provider_error_counts_rollouts_not_exchanges(tmp_path: Path) -> None:
    """Guards #828 greptile P1: an all-failed rollout counts as ONE rollout skip,
    with the exchange count surfaced separately."""
    failed = [
        _anthropic_exchange(status_code=500),
        _anthropic_exchange(status_code=500),
        _anthropic_exchange(status_code=429),
    ]
    _write_rollout(tmp_path / "job" / "rollout-1", exchanges=failed)

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(tmp_path / "job")

    assert rows == []
    assert stats.skipped_provider_error == 1
    assert stats.skipped_exchanges_provider_error == 3


def test_consecutive_leading_system_messages_not_dropped(tmp_path: Path) -> None:
    """Guards #828 greptile P2: a row that starts with two system messages must
    remap the second to 'user' instead of being silently dropped as invalid."""
    exchange = _exchange(final=True)
    exchange["request"]["body"]["messages"] = [
        {"role": "system", "content": "Primary system prompt."},
        {"role": "system", "content": "Second system prompt."},
        {"role": "user", "content": "List files."},
    ]
    _write_rollout(tmp_path / "job" / "rollout-1", exchanges=[exchange])

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(tmp_path / "job")

    assert stats.rows_written == 1
    assert stats.skipped_invalid == 0
    roles = [m["role"] for m in rows[0]["messages"]]
    assert roles[0] == "system"
    assert roles[1] == "user"  # second leading system remapped


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


def test_export_accepts_existing_prime_sft_jsonl_input(tmp_path: Path) -> None:
    """Guards blog repro: ``bench train convert <job>/results.jsonl`` works."""
    source = tmp_path / "results.jsonl"
    out = tmp_path / "prime-sft.jsonl"
    manifest = tmp_path / "manifest.json"
    source.write_text(
        "\n".join(
            [
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
                        "reward": 1.0,
                    }
                ),
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "try"},
                            {"role": "assistant", "content": "done"},
                        ],
                        "reward": 0.0,
                    }
                ),
            ]
        )
        + "\n"
    )

    stats = export_prime_sft_jsonl(
        source,
        out,
        min_reward=0.5,
        expected_rows=1,
        manifest=manifest,
    )

    assert stats.rollouts_seen == 2
    assert stats.rows_written == 1
    assert stats.skipped_reward == 1
    row = json.loads(out.read_text())
    assert "messages" in row
    assert "prompt" not in row
    assert "completion" not in row
    assert validate_prime_sft_jsonl(out, expected_rows=1) == {
        "ok": True,
        "rows": 1,
        "rows_with_tool_calls": 1,
    }
    assert json.loads(manifest.read_text())["sources"] == [str(source)]


def test_export_repairs_legacy_results_tool_call_ids(tmp_path: Path) -> None:
    """Guards historical BenchFlow results rows whose assistant/tool ids drifted."""
    source = tmp_path / "results.jsonl"
    out = tmp_path / "prime-sft.jsonl"
    manifest = tmp_path / "manifest.json"
    source.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "Inspect the app."}],
                "completion": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "fc_legacy_1",
                                "type": "function",
                                "function": {
                                    "name": "terminal",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_runtime_1",
                        "content": "first chunk",
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_runtime_1",
                        "content": "second chunk",
                    },
                    {"role": "assistant", "content": "Done."},
                ],
                "tool_defs": [
                    {
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "reward": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stats = export_prime_sft_jsonl(source, out, expected_rows=1, manifest=manifest)

    row = json.loads(out.read_text())
    assert stats.tool_call_ids_rewritten == 1
    assert stats.tool_messages_merged == 1
    assert "trajectory" not in row
    assert row["messages"][1]["tool_calls"][0]["id"] == "call_runtime_1"
    assert row["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call_runtime_1",
        "content": "first chunk\nsecond chunk",
    }
    assert validate_prime_sft_jsonl(out, expected_rows=1) == {
        "ok": True,
        "rows": 1,
        "rows_with_tool_calls": 1,
    }
    manifest_payload = json.loads(manifest.read_text())
    assert manifest_payload["tool_call_ids_rewritten"] == 1
    assert manifest_payload["tool_messages_merged"] == 1


def test_validate_rejects_tool_calls_without_tool_defs(tmp_path: Path) -> None:
    """Guards PR #828 review: tool-using rows must carry tool_defs/tools."""
    path = tmp_path / "missing-tools.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "finish"},
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
                    },
                ],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="tool_calls require non-empty tool_defs"):
        validate_prime_sft_jsonl(path)


def test_validate_rejects_malformed_tool_call_arguments(tmp_path: Path) -> None:
    """Guards PR #848: bad tool-call JSON fails before training."""
    path = tmp_path / "bad-arguments.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_name": "gdoc-rebrand-juniper-cloud-to-umbra-software-69",
                "messages": [
                    {"role": "user", "content": "Replace the old brand."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_588",
                                "type": "function",
                                "function": {
                                    "name": "terminal",
                                    "arguments": '{"security_risk":"MEDIUM","summary":"Replace old brand"',
                                },
                            }
                        ],
                    },
                ],
                "tool_defs": [
                    {
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match=r"function\.arguments is not valid JSON"):
        validate_prime_sft_jsonl(path, expected_rows=1)


def test_validate_rejects_non_object_tool_call_arguments(tmp_path: Path) -> None:
    """Guards PR #848: tool-call args must parse to a JSON object."""
    path = tmp_path / "list-arguments.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Call the tool."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "finish", "arguments": "[]"},
                            }
                        ],
                    },
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

    with pytest.raises(ValueError, match=r"function\.arguments must be a JSON object"):
        validate_prime_sft_jsonl(path, expected_rows=1)


def test_validate_rejects_unknown_tool_name_and_orphan_tool_output(
    tmp_path: Path,
) -> None:
    """Guards PR #848: tool calls must declare tools and linked outputs."""
    unknown_tool = tmp_path / "unknown-tool.jsonl"
    unknown_tool.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Call the tool."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "missing", "arguments": "{}"},
                            }
                        ],
                    },
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
    orphan_tool = tmp_path / "orphan-tool.jsonl"
    orphan_tool.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Call the tool."},
                    {"role": "assistant", "content": "No tool call."},
                    {"role": "tool", "tool_call_id": "call_missing", "content": "ok"},
                ]
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="not found in tool_defs/tools"):
        validate_prime_sft_jsonl(unknown_tool, expected_rows=1)
    with pytest.raises(ValueError, match="unknown tool_call_id"):
        validate_prime_sft_jsonl(orphan_tool, expected_rows=1)


def test_strict_llm_trajectory_loader_rejects_malformed_jsonl(
    tmp_path: Path,
) -> None:
    """Guards PR #828 review: malformed LLM JSONL must not be silently skipped."""
    path = tmp_path / "llm_trajectory.jsonl"
    path.write_text(json.dumps(_exchange(final=True)) + "\n" + '{"request":\n')

    with pytest.raises(PrimeSftTrajectoryJsonlError, match="line 2: invalid JSON"):
        load_llm_trajectory_jsonl(path, strict=True)


def test_convert_rejects_malformed_llm_jsonl(tmp_path: Path) -> None:
    """Guards PR #828 review: bench train convert must fail closed on corruption."""
    rollout = tmp_path / "job" / "rollout-1"
    _write_rollout(rollout)
    trajectory_path = rollout / "trajectory" / "llm_trajectory.jsonl"
    trajectory_path.write_text(
        trajectory_path.read_text() + '{"request":\n',
        encoding="utf-8",
    )

    with pytest.raises(PrimeSftTrajectoryJsonlError, match="line 3: invalid JSON"):
        convert_benchflow_rollouts_to_prime_sft_rows(tmp_path / "job")


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


def test_convert_succeeds_when_trajectory_written_via_to_jsonl_with_secrets(
    tmp_path: Path,
) -> None:
    """PR #849 end-to-end regression: a trajectory whose content carries a secret
    next to a backslash escape must be written as valid JSON by
    ``Trajectory.to_jsonl`` and convert cleanly to a prime-sft row. Before the
    redactor was moved ahead of serialization, the post-``json.dumps`` regex split
    the escape, producing an unparseable ``llm_trajectory.jsonl`` and a hard
    ``Invalid \\escape`` failure.
    """
    from benchflow.trajectories.types import (
        LLMExchange,
        LLMRequest,
        LLMResponse,
        Trajectory,
    )

    secret = "sk-abc1234567defghijklmnop987654"
    code = (
        'token = form.get("token")\n'
        "    if not token:\n"
        f'        raise HTTPException(400, "invalid")  # {secret}\n'
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "agent"}]},
        {"role": "user", "content": "Edit the revoke handler."},
    ]
    traj = Trajectory(
        session_id="s",
        exchanges=[
            LLMExchange(
                request=LLMRequest(
                    body={"model": "m", "messages": messages, "tools": []}
                ),
                response=LLMResponse(
                    status_code=200,
                    body={
                        "choices": [{"message": {"role": "assistant", "content": code}}]
                    },
                ),
            )
        ],
    )

    rollout = tmp_path / "job" / "rollout-1"
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps({"task_name": "t", "agent": "openhands", "rewards": {"reward": 1.0}})
    )
    traj_path = rollout / "trajectory" / "llm_trajectory.jsonl"
    traj_path.write_text(traj.to_jsonl(redact_keys=True) + "\n", encoding="utf-8")

    # the written trajectory parses (the fix) ...
    for line in traj_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)
    # ... and converts to one valid prime-sft row with the secret redacted.
    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(
        tmp_path / "job", min_reward=1.0
    )
    assert stats.rows_written == 1
    assert stats.skipped_invalid == 0
    assert secret not in json.dumps(rows[0])
