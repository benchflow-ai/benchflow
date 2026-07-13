"""CLI coverage for native TRL SFT conversion and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.trajectories import trl_sft_tokenization

runner = CliRunner()


class _FakeTrlTokenizer:
    chat_template = "fake"

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        **kwargs: Any,
    ):
        del kwargs
        input_ids = [1] * len(tools or [])
        assistant_masks = [0] * len(tools or [])
        for message in messages:
            role = message.get("role")
            if role == "assistant":
                input_ids.append(3)
                assistant_masks.append(0)
            size = max(1, len(str(message.get("content") or "")))
            input_ids.extend([2] * size)
            assistant_masks.extend([1 if role == "assistant" else 0] * size)
        if add_generation_prompt:
            input_ids.append(3)
            assistant_masks.append(0)
        if return_dict:
            result = {"input_ids": input_ids}
            if return_assistant_tokens_mask:
                result["assistant_masks"] = assistant_masks
            return result
        return input_ids


def _tool(name: str = "finish") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_call(call_id: str = "call_1", name: str = "finish") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def _write_rollout(rollout_dir: Path) -> None:
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "opencode",
                "rewards": {"reward": 1.0},
                "n_tool_calls": 1,
            }
        )
    )
    trajectory = rollout_dir / "trajectory"
    trajectory.mkdir()
    primary = {
        "request": {
            "body": {
                "model": "m",
                "messages": [{"role": "user", "content": "do it"}],
                "tools": [_tool()],
            }
        },
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [_tool_call()],
                        }
                    }
                ]
            },
        },
        "duration_ms": 1,
    }
    title = {
        "request": {
            "body": {
                "model": "m",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a title generator. Output only a title.",
                    },
                    {"role": "user", "content": "Generate a title."},
                ],
            }
        },
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Repository inspection",
                        }
                    }
                ]
            },
        },
        "duration_ms": 1,
    }
    (trajectory / "llm_trajectory.jsonl").write_text(
        "\n".join((json.dumps(primary), json.dumps(title))) + "\n"
    )


def _patch_tokenizer(monkeypatch) -> None:
    monkeypatch.setattr(
        trl_sft_tokenization,
        "load_tokenizer",
        lambda tokenizer_id, revision: _FakeTrlTokenizer(),
    )
    monkeypatch.setattr(
        trl_sft_tokenization,
        "training_chat_template",
        lambda tokenizer: None,
    )


def test_trl_sft_cli_excludes_opencode_helpers(tmp_path: Path) -> None:
    """Guards PR #925: TRL export keeps agent calls and drops OpenCode helpers."""
    jobs = tmp_path / "jobs"
    rollout = jobs / "run" / "task-a__abc123"
    _write_rollout(rollout)
    out = tmp_path / "trl-sft.jsonl"
    manifest = tmp_path / "manifest.json"

    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(jobs),
            "--format",
            "trl-sft",
            "--row-mode",
            "exchange",
            "--out",
            str(out),
            "--manifest",
            str(manifest),
            "--expected-rows",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    row = json.loads(out.read_text())
    assert row["prompt"] == [{"role": "user", "content": "do it"}]
    assert row["completion"][0]["tool_calls"][0]["function"] == {
        "name": "finish",
        "arguments": {},
    }
    assert row["tools"][0]["function"]["name"] == "finish"
    assert row["source_rollout_dir"] == str(rollout)
    assert json.loads(manifest.read_text())["skipped_helper_calls"] == 1

    result = runner.invoke(
        app,
        [
            "train",
            "validate",
            str(out),
            "--format",
            "trl-sft",
            "--expected-rows",
            "1",
            "--require-tool-calls",
        ],
    )
    assert result.exit_code == 0, result.output


def test_trl_sft_cli_accepts_results_jsonl_trajectory_steps(tmp_path: Path) -> None:
    """Guards PR #925: canonical results.jsonl can feed TRL conversion."""
    source = tmp_path / "results.jsonl"
    source.write_text(
        json.dumps(
            {
                "info": {
                    "source": "benchflow",
                    "training_ready": True,
                    "task_id": "task-a",
                    "agent": "opencode",
                    "model": "m",
                    "rollout_dir": "/tmp/task-a__abc123",
                },
                "reward": 1.0,
                "tool_defs": [_tool()],
                "trajectory": [
                    {
                        "prompt": [
                            {"role": "system", "content": "You are OpenCode."},
                            {"role": "user", "content": "do it"},
                        ],
                        "completion": [
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [_tool_call()],
                            }
                        ],
                        "extras": {
                            "exchange_index": 0,
                            "call_purpose": "agent",
                        },
                    },
                    {
                        "prompt": [
                            {
                                "role": "system",
                                "content": "You are a title generator.",
                            }
                        ],
                        "completion": [{"role": "assistant", "content": "Task title"}],
                        "extras": {
                            "exchange_index": 1,
                            "call_purpose": "title",
                        },
                    },
                ],
            }
        )
        + "\n"
    )
    out = tmp_path / "trl-sft.jsonl"

    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(source),
            "--format",
            "trl-sft",
            "--row-mode",
            "exchange",
            "--out",
            str(out),
            "--expected-rows",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    row = json.loads(out.read_text())
    assert row["prompt"][0]["role"] == "system"
    assert row["tools"] == [_tool()]
    assert row["source_format"] == "benchflow-results-jsonl"


def test_trl_sft_cli_accepts_existing_native_jsonl(tmp_path: Path) -> None:
    """Guards PR #925: TRL conversion can normalize an existing native file."""
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "do it"}],
                "completion": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call()],
                    }
                ],
                "tools": [_tool()],
                "reward": 1.0,
            }
        )
        + "\n"
    )
    out = tmp_path / "normalized.jsonl"

    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(source),
            "--format",
            "trl-sft",
            "--out",
            str(out),
            "--expected-rows",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    row = json.loads(out.read_text())
    assert row["completion"][0]["tool_calls"][0]["function"]["arguments"] == {}
    assert row["source_format"] == "trl-sft"


def test_trl_sft_message_window_keeps_complete_recent_tool_turns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Guards PR #925: token windowing preserves prefix and causal tool groups."""
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "task"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call("call_old", "read")],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_old",
                        "content": "xxxxxxxxxxxx",
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call("call_recent", "read")],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_recent",
                        "content": "recent",
                    },
                ],
                "completion": [{"role": "assistant", "content": "done"}],
                "tools": [_tool("read")],
            }
        )
        + "\n"
    )
    out = tmp_path / "windowed.jsonl"
    manifest = tmp_path / "manifest.json"
    _patch_tokenizer(monkeypatch)

    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(source),
            "--format",
            "trl-sft",
            "--out",
            str(out),
            "--context-policy",
            "message-window",
            "--tokenizer",
            "fake-tokenizer",
            "--max-length",
            "24",
            "--manifest",
            str(manifest),
        ],
    )

    assert result.exit_code == 0, result.output
    row = json.loads(out.read_text())
    assert [message["role"] for message in row["prompt"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert row["prompt"][-1]["tool_call_id"] == "call_recent"
    assert row["context_window"]["messages_dropped"] == 2
    assert row["context_window"]["final_tokens"] == 21
    stats = json.loads(manifest.read_text())
    assert stats["rows_compacted"] == 1
    assert stats["messages_dropped"] == 2


def test_trl_sft_message_window_rejects_unusable_recent_tool_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Guards PR #925: windowing never drops the only recent causal tool group."""
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "task"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call("call_recent", "read")],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_recent",
                        "content": "x" * 40,
                    },
                ],
                "completion": [{"role": "assistant", "content": "done"}],
                "tools": [_tool("read")],
            }
        )
        + "\n"
    )
    _patch_tokenizer(monkeypatch)

    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(source),
            "--format",
            "trl-sft",
            "--out",
            str(tmp_path / "out.jsonl"),
            "--context-policy",
            "message-window",
            "--tokenizer",
            "fake-tokenizer",
            "--max-length",
            "20",
        ],
    )

    assert result.exit_code == 1
    assert "most recent assistant/tool context group does not fit" in result.output


def test_trl_sft_validate_checks_masks_and_length(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Guards PR #925: TRL validation proves target assistant tokens survive."""
    source = tmp_path / "trl-sft.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "do it"}],
                "completion": [{"role": "assistant", "content": "done"}],
                "tools": [],
            }
        )
        + "\n"
    )
    _patch_tokenizer(monkeypatch)

    result = runner.invoke(
        app,
        [
            "train",
            "validate",
            str(source),
            "--format",
            "trl-sft",
            "--tokenizer",
            "fake-tokenizer",
            "--max-length",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        json.loads(result.output)["tokenization"]["min_trainable_assistant_tokens"] == 4
    )

    result = runner.invoke(
        app,
        [
            "train",
            "validate",
            str(source),
            "--format",
            "trl-sft",
            "--tokenizer",
            "fake-tokenizer",
            "--max-length",
            "8",
        ],
    )
    assert result.exit_code == 1
    assert "tokenized length 10 exceeds max_length 8" in result.output
