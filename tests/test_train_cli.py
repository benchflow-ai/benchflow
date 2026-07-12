"""CLI coverage for ``bench train``."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.trajectories import export_trl_sft

runner = CliRunner()


class _FakeTailTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any):
        if kwargs.get("tokenize") is False:
            return "".join(
                str(message.get("content") or "")
                + ("T" * (4 * len(message.get("tool_calls") or [])))
                for message in messages
            )
        length = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                length += len(content)
            prefix_ws = message.get("prefix_ws")
            if isinstance(prefix_ws, str):
                length += len(prefix_ws)
            suffix_ws = message.get("suffix_ws")
            if isinstance(suffix_ws, str):
                length += len(suffix_ws)
            length += 4 * len(message.get("tool_calls") or [])
        return [1] * length + [self.eos_token_id]

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        del kwargs
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        del kwargs
        return "".join(
            "<eos>" if token_id == self.eos_token_id else chr(token_id)
            for token_id in token_ids
        )


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
        input_ids: list[int] = []
        assistant_masks: list[int] = []
        input_ids.extend([1] * len(tools or []))
        assistant_masks.extend([0] * len(tools or []))
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


def _write_rollout(rollout_dir: Path) -> None:
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "result.json").write_text(
        json.dumps(
            {"task_name": "task-a", "rewards": {"reward": 1.0}, "n_tool_calls": 1}
        )
    )
    traj = rollout_dir / "trajectory"
    traj.mkdir()
    (traj / "llm_trajectory.jsonl").write_text(
        json.dumps(
            {
                "request": {
                    "body": {
                        "model": "m",
                        "messages": [{"role": "user", "content": "do it"}],
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "finish",
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
                                "message": {
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
                            }
                        ]
                    },
                },
                "duration_ms": 1,
            }
        )
        + "\n"
    )


def test_train_convert_and_validate_cli(tmp_path: Path) -> None:
    """Guards this PR's public ``bench train`` conversion workflow."""
    jobs = tmp_path / "jobs"
    _write_rollout(jobs / "run" / "task-a__abc123")
    out = tmp_path / "train.jsonl"
    manifest = tmp_path / "manifest.json"

    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(jobs),
            "--out",
            str(out),
            "--manifest",
            str(manifest),
            "--expected-rows",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Converted 1 row" in result.output
    assert out.exists()
    assert json.loads(manifest.read_text())["rows_written"] == 1

    result = runner.invoke(
        app,
        ["train", "validate", str(out), "--expected-rows", "1"],
    )

    assert result.exit_code == 0, result.output
    assert '"rows": 1' in result.output


def test_train_convert_and_validate_trl_sft_cli_excludes_opencode_helpers(
    tmp_path: Path,
) -> None:
    """Guards PR #925: TRL export keeps agent calls and drops OpenCode helpers."""
    jobs = tmp_path / "jobs"
    rollout = jobs / "run" / "task-a__abc123"
    _write_rollout(rollout)
    result_json = json.loads((rollout / "result.json").read_text())
    result_json["agent"] = "opencode"
    (rollout / "result.json").write_text(json.dumps(result_json))
    trajectory_path = rollout / "trajectory" / "llm_trajectory.jsonl"
    primary = json.loads(trajectory_path.read_text())
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
    trajectory_path.write_text(
        "\n".join((json.dumps(primary), json.dumps(title))) + "\n"
    )
    out = tmp_path / "trl-sft.jsonl"
    manifest = tmp_path / "trl-sft-manifest.json"

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
    assert row["completion"][0]["role"] == "assistant"
    assert row["completion"][0]["tool_calls"][0]["function"] == {
        "name": "finish",
        "arguments": {},
    }
    assert row["tools"][0]["function"]["name"] == "finish"
    assert "tool_defs" not in row
    assert row["call_purpose"] == "agent"
    assert row["exchange_index"] == 0
    assert row["source_rollout_dir"] == str(rollout)
    stats = json.loads(manifest.read_text())
    assert stats["rows_written"] == 1
    assert stats["skipped_helper_calls"] == 1

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
    assert '"rows": 1' in result.output


def test_train_convert_accepts_results_jsonl_cli(tmp_path: Path) -> None:
    """Guards the public repro command that converts an existing results.jsonl."""
    source = tmp_path / "results.jsonl"
    source.write_text(
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
        )
        + "\n"
    )
    out = tmp_path / "prime-sft.jsonl"

    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(source),
            "--out",
            str(out),
            "--expected-rows",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Converted 1 row" in result.output
    result = runner.invoke(app, ["train", "validate", str(out), "--expected-rows", "1"])
    assert result.exit_code == 0, result.output


def test_train_convert_trl_sft_accepts_results_jsonl_trajectory_steps(
    tmp_path: Path,
) -> None:
    """Guards PR #925: canonical results.jsonl can feed TRL conversion."""
    source = tmp_path / "results.jsonl"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "finish",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
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
                "tool_defs": tools,
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
                            },
                            {"role": "user", "content": "Generate a title."},
                        ],
                        "completion": [
                            {
                                "role": "assistant",
                                "content": "Task title",
                            }
                        ],
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
    manifest = tmp_path / "manifest.json"

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
            "--manifest",
            str(manifest),
            "--expected-rows",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    row = json.loads(out.read_text())
    assert row["prompt"][0] == {
        "role": "system",
        "content": "You are OpenCode.",
    }
    assert row["completion"][0]["tool_calls"][0]["function"]["arguments"] == {}
    assert row["tools"] == tools
    assert row["task_id"] == "task-a"
    assert row["source_format"] == "benchflow-results-jsonl"
    assert row["source_index"] == 0
    stats = json.loads(manifest.read_text())
    assert stats["skipped_helper_calls"] == 1


def test_train_convert_trl_sft_accepts_existing_native_jsonl(tmp_path: Path) -> None:
    """Guards PR #925: TRL conversion can normalize an existing native file."""
    source = tmp_path / "source-trl-sft.jsonl"
    source.write_text(
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
                                    "arguments": {},
                                },
                            }
                        ],
                    }
                ],
                "tools": [
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
        )
        + "\n"
    )
    out = tmp_path / "normalized-trl-sft.jsonl"

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
    assert row["prompt"] == [{"role": "user", "content": "do it"}]
    assert row["completion"][0]["tool_calls"][0]["function"]["arguments"] == {}
    assert row["source_format"] == "trl-sft"
    assert row["source_index"] == 0


def test_train_validate_trl_sft_checks_tokenizer_masks(
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
    monkeypatch.setattr(
        export_trl_sft,
        "_load_tokenizer",
        lambda tokenizer_id, revision: _FakeTrlTokenizer(),
    )
    monkeypatch.setattr(
        export_trl_sft,
        "_training_chat_template",
        lambda tokenizer: None,
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
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["tokenization"] == {
        "tokenizer": "fake-tokenizer",
        "tokenizer_revision": None,
        "max_length": 20,
        "min_tokens": 10,
        "median_tokens": 10,
        "p95_tokens": 10,
        "max_tokens": 10,
        "min_trainable_assistant_tokens": 4,
    }


def test_train_validate_trl_sft_fails_before_training_on_overlength_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Guards PR #925: TRL validation rejects rows the trainer would truncate."""
    source = tmp_path / "trl-sft.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "long prompt"}],
                "completion": [{"role": "assistant", "content": "done"}],
                "tools": [],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        export_trl_sft,
        "_load_tokenizer",
        lambda tokenizer_id, revision: _FakeTrlTokenizer(),
    )
    monkeypatch.setattr(
        export_trl_sft,
        "_training_chat_template",
        lambda tokenizer: None,
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
    assert "tokenized length 16 exceeds max_length 8" in result.output


def test_train_convert_no_redact_preserves_tool_argument_tokens(tmp_path: Path) -> None:
    """Private SFT conversion should not mutate valid tool-call argument strings."""
    source = tmp_path / "results.jsonl"
    raw_arguments = json.dumps(
        {
            "command": "headers={'Authorization': f'Bearer {token}'}",
            "summary": "use runtime token",
        }
    )
    source.write_text(
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
                                    "name": "terminal",
                                    "arguments": raw_arguments,
                                },
                            }
                        ],
                    }
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
        + "\n"
    )
    out = tmp_path / "prime-sft.jsonl"

    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(source),
            "--out",
            str(out),
            "--expected-rows",
            "1",
            "--no-redact",
        ],
    )

    assert result.exit_code == 0, result.output
    row = json.loads(out.read_text())
    assert row["messages"][1]["tool_calls"][0]["function"]["arguments"] == raw_arguments


def test_train_convert_sanitizes_malformed_tool_call_arguments(
    tmp_path: Path,
) -> None:
    source = tmp_path / "results.jsonl"
    source.write_text(
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
                                    "name": "terminal",
                                    "arguments": '{"command":"python - <<\'PY\'\nunterminated',
                                },
                            }
                        ],
                    }
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
    out = tmp_path / "prime-sft.jsonl"

    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(source),
            "--out",
            str(out),
            "--expected-rows",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    row = json.loads(out.read_text())
    arguments = row["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {
        "_malformed_json_arguments": '{"command":"python - <<\'PY\'\nunterminated'
    }
    result = runner.invoke(app, ["train", "validate", str(out), "--expected-rows", "1"])
    assert result.exit_code == 0, result.output


def test_train_convert_repairs_legacy_results_tool_call_ids(tmp_path: Path) -> None:
    source = tmp_path / "results.jsonl"
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
                        "content": "ok",
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
                "reward": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "prime-sft.jsonl"
    manifest = tmp_path / "manifest.json"

    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(source),
            "--out",
            str(out),
            "--expected-rows",
            "1",
            "--manifest",
            str(manifest),
        ],
    )

    assert result.exit_code == 0, result.output
    row = json.loads(out.read_text())
    assert "trajectory" not in row
    assert row["messages"][1]["tool_calls"][0]["id"] == "call_runtime_1"
    assert json.loads(manifest.read_text())["tool_call_ids_rewritten"] == 1
    result = runner.invoke(app, ["train", "validate", str(out), "--expected-rows", "1"])
    assert result.exit_code == 0, result.output


def test_train_convert_accepts_canonical_selection(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    selected = jobs / "run" / "task-a__good"
    ignored = jobs / "run" / "task-a__ignored"
    _write_rollout(selected)
    _write_rollout(ignored)
    selection = tmp_path / "canonical-selection.json"
    selection.write_text(
        json.dumps(
            {
                "job_dir": str(jobs / "run"),
                "selected": [{"task_id": "task-a", "rollout_dir": str(selected)}],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "train.jsonl"

    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(jobs),
            "--out",
            str(out),
            "--canonical-selection",
            str(selection),
            "--expected-rows",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert sum(1 for _ in out.open()) == 1


def test_train_validate_source_health_requirements(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    rollout = jobs / "run" / "task-a__abc123"
    _write_rollout(rollout)
    out = tmp_path / "train.jsonl"
    result = runner.invoke(
        app,
        ["train", "convert", str(jobs), "--out", str(out), "--expected-rows", "1"],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "train",
            "validate",
            str(out),
            "--source-jobs",
            str(jobs),
            "--expected-rows",
            "1",
            "--require-llm-trajectory",
            "--require-tool-calls",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["source_health"]["total_rows"] == 1
    assert payload["source_health"]["rows_with_tool_calls"] == 1


def test_train_convert_rejects_malformed_llm_jsonl(tmp_path: Path) -> None:
    """Guards PR #828 review: CLI conversion fails closed on corrupted LLM traces."""
    jobs = tmp_path / "jobs"
    rollout = jobs / "run" / "task-a__abc123"
    _write_rollout(rollout)
    trajectory_path = rollout / "trajectory" / "llm_trajectory.jsonl"
    trajectory_path.write_text(
        trajectory_path.read_text() + '{"request":\n',
        encoding="utf-8",
    )
    out = tmp_path / "train.jsonl"

    result = runner.invoke(
        app,
        ["train", "convert", str(jobs), "--out", str(out), "--expected-rows", "1"],
    )

    assert result.exit_code == 1
    assert "invalid JSON" in result.output
    assert not out.exists()


def test_train_run_sft_prime_rl_records_manifest(tmp_path: Path, monkeypatch) -> None:
    """Guards the Prime-RL SFT wrapper command against losing launch provenance."""
    import benchflow.training.backends.prime_rl as prime_rl

    captured: dict[str, Any] = {}

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            captured["argv"] = argv
            captured["cwd"] = kwargs["cwd"]
            self.stdout = io.StringIO("trainer ok\n")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    config = tmp_path / "sft.toml"
    config.write_text("max_steps = 1\n", encoding="utf-8")
    work_dir = tmp_path / "train-run"
    output_dir = tmp_path / "prime-output"
    prime_rl_dir = tmp_path / "prime-rl-checkout"
    prime_rl_dir.mkdir()

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--backend",
            "prime-rl",
            "--config",
            str(config),
            "--data",
            "benchflow/env0-prime-sft",
            "--output-dir",
            str(output_dir),
            "--work-dir",
            str(work_dir),
            "--prime-rl-dir",
            str(prime_rl_dir),
            "--dry-run",
            "--uv-no-sync",
            "--override",
            "model.name=Qwen/Qwen3.5-9B",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Prime-RL SFT completed" in result.output
    assert captured["argv"] == [
        "uv",
        "run",
        "--no-sync",
        "sft",
        "@",
        str(config),
        "--data.name",
        "benchflow/env0-prime-sft",
        "--output-dir",
        str(output_dir),
        "--dry-run",
        "--model.name",
        "Qwen/Qwen3.5-9B",
    ]
    assert captured["cwd"] == str(prime_rl_dir.resolve())
    manifest = json.loads((work_dir / "train-run.json").read_text())
    assert manifest["overall_status"] == "succeeded"
    assert manifest["run_type"] == "sft"
    assert manifest["backend"] == "prime-rl"
    assert manifest["commands"][0]["argv"] == captured["argv"]
    assert manifest["commands"][0]["cwd"] == str(prime_rl_dir.resolve())
    assert manifest["components"] == [
        {
            "checkpoints": [],
            "command_id": "prime-rl-sft",
            "extra": {},
            "logs": ["prime-rl/stdout.log", "prime-rl/stderr.log"],
            "metrics": [],
            "name": "trainer",
            "role": "primary",
            "status": "succeeded",
        }
    ]
    assert (work_dir / "command.txt").read_text().startswith("uv run --no-sync sft @")
    assert (work_dir / "prime-rl" / "stdout.log").read_text() == "trainer ok\n"


def test_train_run_sft_prime_rl_packages_local_jsonl_data(
    tmp_path: Path, monkeypatch
) -> None:
    """Prime-RL load_dataset accepts local dataset dirs, not raw JSONL paths."""
    import benchflow.training.backends.prime_rl as prime_rl

    captured: dict[str, Any] = {}

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            captured["argv"] = argv
            captured["cwd"] = kwargs["cwd"]
            self.stdout = io.StringIO("trainer ok\n")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    config = tmp_path / "sft.toml"
    config.write_text("max_steps = 1\n", encoding="utf-8")
    source = tmp_path / "prime-sft.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "do it"},
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
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "ok",
                    },
                    {"role": "assistant", "content": "done"},
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
        + "\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "train-run"

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--data",
            str(source),
            "--work-dir",
            str(work_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    dataset_dirs = sorted(work_dir.glob("prime-rl-dataset-*"))
    assert len(dataset_dirs) == 1
    dataset_dir = dataset_dirs[0]
    assert not (work_dir / "prime-rl-dataset").exists()
    train_jsonl = dataset_dir / "train.jsonl"
    assert train_jsonl.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert captured["argv"] == [
        "uv",
        "run",
        "sft",
        "@",
        str(config),
        "--data.name",
        str(dataset_dir.resolve()),
        "--output-dir",
        str(work_dir / "prime-rl-output"),
    ]
    manifest = json.loads((work_dir / "train-run.json").read_text())
    assert manifest["extra"]["prime_rl_sft_dataset"] == {
        "dataset_dir": str(dataset_dir.resolve()),
        "kind": "local_jsonl_packaged",
        "resolved_data": str(dataset_dir.resolve()),
        "source_data": str(source),
        "train_jsonl": str(train_jsonl),
        "tool_defs_mode": "preserve",
        "tool_defs_removed_rows": None,
        "chat_template_kwargs": None,
        "chat_template_kwargs_rows": None,
        "message_tail_truncation": "off",
        "message_tail_truncated_rows": None,
        "message_tail_max_area": None,
        "message_tail_max_tokens_before": None,
        "message_tail_max_tokens_after": None,
        "custom_trainer_pretokenized_rows": None,
        "custom_trainer_pretokenized_trainable_tokens": None,
        "validation": {"ok": True, "rows": 1, "rows_with_tool_calls": 1},
    }


def test_train_run_sft_prime_rl_rejects_missing_local_jsonl_data(
    tmp_path: Path, monkeypatch
) -> None:
    import benchflow.training.backends.prime_rl as prime_rl

    config = tmp_path / "sft.toml"
    config.write_text("max_steps = 1\n", encoding="utf-8")
    missing = tmp_path / "missing.jsonl"

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--data",
            str(missing),
            "--work-dir",
            str(tmp_path / "train-run"),
        ],
    )

    assert result.exit_code == 1
    assert "--data JSONL file not found" in result.output


def test_train_run_sft_prime_rl_target_examples_derives_exposure(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards sample-exposure parity for small BenchFlow SFT trajectory sets."""
    import benchflow.training.backends.prime_rl as prime_rl

    captured: dict[str, Any] = {}

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            captured["argv"] = argv
            captured["cwd"] = kwargs["cwd"]
            self.stdout = io.StringIO("trainer ok\n")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    config = tmp_path / "sft.toml"
    config.write_text(
        "\n".join(
            [
                "max_steps = 300",
                "[data]",
                "batch_size = 8",
                'pack_function = "cat"',
                "[scheduler]",
                "decay_steps = 300",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "train-run"
    output_dir = tmp_path / "prime-output"

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
            "--work-dir",
            str(work_dir),
            "--target-examples",
            "300",
            "--pack-function",
            "stack",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["argv"] == [
        "uv",
        "run",
        "sft",
        "@",
        str(config),
        "--output-dir",
        str(output_dir),
        "--max_steps",
        "38",
        "--scheduler.decay_steps",
        "38",
        "--data.pack_function",
        "stack",
    ]
    manifest = json.loads((work_dir / "train-run.json").read_text())
    assert manifest["extra"]["prime_rl_sft_exposure_plan"] == {
        "data_batch_size": 8,
        "derived_max_steps": 38,
        "effective_train_examples": 304,
        "generated_overrides": [
            "max_steps=38",
            "scheduler.decay_steps=38",
            "data.pack_function=stack",
        ],
        "loss_mask": None,
        "model_attn": None,
        "pack_function": "stack",
        "renderer_mode": None,
        "sync_ckpt_to_max_steps": False,
        "sync_scheduler_to_max_steps": True,
        "target_examples": 300,
        "target_micro_steps": None,
        "unapplied_micro_steps": None,
    }


def test_train_run_sft_prime_rl_target_examples_respects_batch_override(
    tmp_path: Path, monkeypatch
) -> None:
    import benchflow.training.backends.prime_rl as prime_rl

    captured: dict[str, Any] = {}

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            del kwargs
            captured["argv"] = argv
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    config = tmp_path / "sft.toml"
    config.write_text("max_steps = 300\n[data]\nbatch_size = 1\n", encoding="utf-8")
    work_dir = tmp_path / "train-run"

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(work_dir),
            "--target-examples",
            "300",
            "--no-sync-scheduler-to-max-steps",
            "--override",
            "data.batch_size=8",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "--data.batch_size" in captured["argv"]
    assert captured["argv"][-2:] == ["--max_steps", "38"]
    manifest = json.loads((work_dir / "train-run.json").read_text())
    assert (
        manifest["extra"]["prime_rl_sft_exposure_plan"]["sync_scheduler_to_max_steps"]
        is False
    )


def test_train_run_sft_prime_rl_target_micro_steps_drops_partial_accumulation(
    tmp_path: Path, monkeypatch
) -> None:
    """Custom-trainer max_steps counted micro-batches, not optimizer updates."""
    import benchflow.training.backends.prime_rl as prime_rl

    captured: dict[str, Any] = {}

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            del kwargs
            captured["argv"] = argv
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    config = tmp_path / "sft.toml"
    config.write_text(
        "max_steps = 300\n[data]\nbatch_size = 8\n[scheduler]\ndecay_steps = 300\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "train-run"

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(work_dir),
            "--target-micro-steps",
            "300",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["argv"][-4:] == [
        "--max_steps",
        "37",
        "--scheduler.decay_steps",
        "37",
    ]
    manifest = json.loads((work_dir / "train-run.json").read_text())
    assert manifest["extra"]["prime_rl_sft_exposure_plan"] == {
        "data_batch_size": 8,
        "derived_max_steps": 37,
        "effective_train_examples": 296,
        "generated_overrides": [
            "max_steps=37",
            "scheduler.decay_steps=37",
        ],
        "loss_mask": None,
        "model_attn": None,
        "pack_function": None,
        "renderer_mode": None,
        "sync_ckpt_to_max_steps": False,
        "sync_scheduler_to_max_steps": True,
        "target_examples": None,
        "target_micro_steps": 300,
        "unapplied_micro_steps": 4,
    }


def test_train_run_sft_prime_rl_records_reproduction_semantics(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards explicit Prime-RL semantics for custom-trainer reproduction runs."""
    import benchflow.training.backends.prime_rl as prime_rl

    captured: dict[str, Any] = {}

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            captured["argv"] = argv
            captured["env"] = kwargs.get("env")
            self.stdout = io.StringIO("trainer ok\n")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    config = tmp_path / "sft.toml"
    config.write_text(
        "\n".join(
            [
                "max_steps = 300",
                "[model]",
                'name = "Qwen/Qwen3.5-9B"',
                'attn = "flash_attention_2"',
                "[data]",
                "batch_size = 8",
                'pack_function = "cat"',
                "[scheduler]",
                "decay_steps = 300",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "train-run"

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(work_dir),
            "--target-examples",
            "300",
            "--pack-function",
            "stack",
            "--loss-mask",
            "all",
            "--model-attn",
            "sdpa",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["argv"][-16:] == [
        "--max_steps",
        "38",
        "--scheduler.decay_steps",
        "38",
        "--data.pack_function",
        "stack",
        "--data.loss_mask.system",
        "true",
        "--data.loss_mask.user",
        "true",
        "--data.loss_mask.assistant",
        "true",
        "--data.loss_mask.tool",
        "true",
        "--model.attn",
        "sdpa",
    ]
    manifest = json.loads((work_dir / "train-run.json").read_text())
    assert manifest["extra"]["prime_rl_sft_exposure_plan"] == {
        "data_batch_size": 8,
        "derived_max_steps": 38,
        "effective_train_examples": 304,
        "generated_overrides": [
            "max_steps=38",
            "scheduler.decay_steps=38",
            "data.pack_function=stack",
            "data.loss_mask.system=true",
            "data.loss_mask.user=true",
            "data.loss_mask.assistant=true",
            "data.loss_mask.tool=true",
            "model.attn=sdpa",
        ],
        "loss_mask": "all",
        "model_attn": "sdpa",
        "pack_function": "stack",
        "renderer_mode": None,
        "sync_ckpt_to_max_steps": False,
        "sync_scheduler_to_max_steps": True,
        "target_examples": 300,
        "target_micro_steps": None,
        "unapplied_micro_steps": None,
    }


def test_train_run_sft_prime_rl_sample_mean_requires_stack(
    tmp_path: Path, monkeypatch
) -> None:
    """Sample-mean loss is only valid when Prime-RL preserves row boundaries."""
    import benchflow.training.backends.prime_rl as prime_rl

    config = tmp_path / "sft.toml"
    config.write_text(
        "\n".join(
            [
                "max_steps = 1",
                "[data]",
                "batch_size = 8",
                'pack_function = "cat"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "train-run"

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(work_dir),
            "--loss-normalization",
            "sample_mean",
        ],
    )

    assert result.exit_code == 1
    assert "sample_mean requires data.pack_function=stack" in result.output


def test_train_run_sft_prime_rl_mobile300_requires_sample_mean(
    tmp_path: Path, monkeypatch
) -> None:
    """The Mobile300 profile refuses native token-weighted Prime-RL loss."""
    import benchflow.training.backends.prime_rl as prime_rl

    config = tmp_path / "sft.toml"
    config.write_text(
        "\n".join(
            [
                "max_steps = 300",
                "[model]",
                'name = "Qwen/Qwen3.5-9B"',
                "[data]",
                "batch_size = 8",
                'pack_function = "stack"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "train-run"

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(work_dir),
            "--compat-profile",
            "env0-mobile300-pr828",
            "--loss-normalization",
            "token_mean",
        ],
    )

    assert result.exit_code == 1
    assert "requires --loss-normalization sample_mean" in result.output


def test_train_run_sft_prime_rl_mobile300_compat_profile(
    tmp_path: Path, monkeypatch
) -> None:
    """The Mobile300 profile expands to the validated Prime-RL wrapper settings."""
    import benchflow.training.backends.prime_rl as prime_rl

    captured: dict[str, Any] = {}

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            captured["argv"] = argv
            captured["env"] = kwargs.get("env")
            self.stdout = io.StringIO("trainer ok\n")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    config = tmp_path / "sft.toml"
    config.write_text(
        "\n".join(
            [
                "max_steps = 300",
                "[model]",
                'name = "Qwen/Qwen3.5-9B"',
                'attn = "flash_attention_2"',
                "[renderer]",
                'name = "qwen3.5"',
                "[data]",
                "batch_size = 8",
                'pack_function = "cat"',
                "[scheduler]",
                "decay_steps = 300",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
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
                    {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
                ],
                "tool_defs": [{"name": "finish", "parameters": {"type": "object"}}],
                "reward": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "train-run"

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        prime_rl,
        "_load_tail_truncation_tokenizer",
        lambda model_name: _FakeTailTokenizer(),
    )

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(work_dir),
            "--data",
            str(data_dir),
            "--compat-profile",
            "env0-mobile300-pr828",
        ],
    )

    assert result.exit_code == 0, result.output
    argv = captured["argv"]
    assert captured["env"]["BENCHFLOW_PRIME_RL_SAMPLE_MEAN_LOSS"] == "1"
    assert captured["env"]["BENCHFLOW_PRIME_RL_PRETOKENIZED_SFT_DATA"] == "1"
    assert captured["env"]["BENCHFLOW_PRIME_RL_STUB_FLASH_ATTN"] == "1"
    assert "--tokenizer.chat_template" not in argv
    assert argv[-22:] == [
        "--max_steps",
        "37",
        "--scheduler.decay_steps",
        "37",
        "--ckpt.interval",
        "37",
        "--ckpt.keep_interval",
        "37",
        "--data.pack_function",
        "stack",
        "--data.loss_mask.system",
        "true",
        "--data.loss_mask.user",
        "true",
        "--data.loss_mask.assistant",
        "true",
        "--data.loss_mask.tool",
        "true",
        "--model.attn",
        "sdpa",
        "--renderer",
        "None",
    ]
    data_idx = argv.index("--data.name")
    staged_dir = Path(argv[data_idx + 1])
    staged_row = json.loads((staged_dir / "train.jsonl").read_text())
    assert "tool_defs" not in staged_row
    assert "tools" not in staged_row
    assert "chat_template_kwargs" not in staged_row
    assert "messages" not in staged_row
    assert staged_row["benchflow_custom_trainer_pretokenized"] == {
        "original_token_count": 12,
        "staged_token_count": 12,
        "trainable_token_count": 11,
    }
    assert staged_row["benchflow_input_ids"] == [
        ord(char) for char in "finishTTTTok"[:-1]
    ]
    assert staged_row["benchflow_target_ids"] == [
        ord(char) for char in "finishTTTTok"[1:]
    ]
    assert staged_row["benchflow_loss_mask"] == [True] * 11
    assert staged_row["benchflow_position_ids"] == list(range(11))

    manifest = json.loads((work_dir / "train-run.json").read_text())
    command_text = (work_dir / "command.txt").read_text()
    assert "BENCHFLOW_PRIME_RL_SAMPLE_MEAN_LOSS=1" in command_text
    assert "BENCHFLOW_PRIME_RL_PRETOKENIZED_SFT_DATA=1" in command_text
    assert "BENCHFLOW_PRIME_RL_STUB_FLASH_ATTN=1" in command_text
    assert "PYTHONPATH=" in command_text
    shim = manifest["extra"]["prime_rl_sft_shim"]
    shim_dir = Path(shim["shim_dir"])
    assert captured["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(shim_dir)
    assert shim["env"]["BENCHFLOW_PRIME_RL_STUB_FLASH_ATTN"] == "1"
    assert "model.attn=sdpa" in shim["guards"]
    sitecustomize_text = Path(shim["sitecustomize"]).read_text(encoding="utf-8")
    assert sitecustomize_text.startswith(
        '"""BenchFlow Prime-RL SFT compatibility shim.'
    )
    assert "if not torch.any(valid_sample_mask):" in sitecustomize_text
    assert "if not torch.all(valid_sample_mask):" not in sitecustomize_text
    assert "_flash_attn_varlen_forward" in sitecustomize_text
    assert "flash-attn import stub enabled for SDPA" in sitecustomize_text
    assert "def benchflow_setup_dataloader" in sitecustomize_text
    assert shim["name"] == "prime_rl_sft_compatibility"
    assert (
        "pretokenized rows bypass Prime-RL stack/cat packing and train one original "
        "row per micro-batch"
    ) in shim["guards"]
    assert manifest["extra"]["prime_rl_sft_compat_profile"]["name"] == (
        "env0-mobile300-pr828"
    )
    assert manifest["extra"]["prime_rl_sft_compat_profile"]["resolved_settings"] == {
        "target_examples": None,
        "target_micro_steps": 300,
        "sync_scheduler_to_max_steps": True,
        "sync_ckpt_to_max_steps": True,
        "pack_function": "stack",
        "loss_mask": "all",
        "loss_normalization": "sample_mean",
        "model_attn": "sdpa",
        "renderer_mode": "none",
        "tool_defs_mode": "omit",
        "chat_template_kwargs": {},
        "message_tail_truncation": "custom-trainer-pretokenized",
    }
    assert (
        manifest["extra"]["prime_rl_sft_exposure_plan"]["effective_train_examples"]
        == 296
    )
    assert manifest["extra"]["prime_rl_sft_exposure_plan"]["unapplied_micro_steps"] == 4
    assert manifest["extra"]["prime_rl_sft_exposure_plan"]["generated_overrides"][
        :4
    ] == [
        "max_steps=37",
        "scheduler.decay_steps=37",
        "ckpt.interval=37",
        "ckpt.keep_interval=37",
    ]
    assert manifest["extra"]["prime_rl_sft_dataset"]["tool_defs_mode"] == "omit"
    assert manifest["extra"]["prime_rl_sft_dataset"]["tool_defs_removed_rows"] == 1
    assert manifest["extra"]["prime_rl_sft_dataset"]["chat_template_kwargs"] is None
    assert (
        manifest["extra"]["prime_rl_sft_dataset"]["chat_template_kwargs_rows"] is None
    )
    assert manifest["extra"]["prime_rl_sft_dataset"]["message_tail_truncation"] == (
        "custom-trainer-pretokenized"
    )
    assert manifest["extra"]["prime_rl_sft_dataset"]["message_tail_truncated_rows"] == 0
    assert manifest["extra"]["prime_rl_sft_dataset"]["message_tail_max_area"] == 128
    assert (
        manifest["extra"]["prime_rl_sft_dataset"]["custom_trainer_pretokenized_rows"]
        == 1
    )
    assert (
        manifest["extra"]["prime_rl_sft_dataset"][
            "custom_trainer_pretokenized_trainable_tokens"
        ]
        == 11
    )


def test_prime_rl_sft_flash_attn_stub_imports_and_fails_closed(tmp_path: Path) -> None:
    """The SDPA shim satisfies import-time ring_flash_attn symbols only."""
    import benchflow.training.backends.prime_rl as prime_rl

    fake_transformers_utils = tmp_path / "fake" / "transformers" / "utils"
    fake_transformers_utils.mkdir(parents=True)
    (fake_transformers_utils.parent / "__init__.py").write_text("", encoding="utf-8")
    (fake_transformers_utils / "__init__.py").write_text("", encoding="utf-8")
    (fake_transformers_utils / "import_utils.py").write_text(
        "PACKAGE_DISTRIBUTION_MAPPING = {}\n",
        encoding="utf-8",
    )

    plan = prime_rl._write_prime_rl_sft_compat_shim(
        tmp_path,
        sample_mean=False,
        pretokenized_data=False,
        stub_flash_attn=True,
    )
    env = os.environ.copy()
    env.update(plan.env)
    env["PYTHONPATH"] = os.pathsep.join(
        (plan.env["PYTHONPATH"], str(tmp_path / "fake"))
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "from flash_attn.flash_attn_interface import _flash_attn_forward",
                    (
                        "from transformers.utils.import_utils import "
                        "PACKAGE_DISTRIBUTION_MAPPING"
                    ),
                    "print(PACKAGE_DISTRIBUTION_MAPPING['flash_attn'])",
                    "try:",
                    "    _flash_attn_forward()",
                    "except RuntimeError as exc:",
                    "    print(str(exc))",
                    "else:",
                    "    raise SystemExit('expected RuntimeError')",
                ]
            ),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "flash-attn import stub enabled for SDPA" in completed.stderr
    assert "['flash-attn']" in completed.stdout
    assert "model.attn=sdpa" in completed.stdout
    assert "called unexpectedly" in completed.stdout


def test_prime_rl_pretokenized_shim_uses_rowwise_dataloader(tmp_path: Path) -> None:
    """Pretokenized rows bypass Prime-RL packers so exposure matches the old trainer."""
    import benchflow.training.backends.prime_rl as prime_rl

    fake_sft = tmp_path / "fake" / "prime_rl" / "trainer" / "sft"
    fake_sft.mkdir(parents=True)
    for package_dir in [
        fake_sft.parent.parent,
        fake_sft.parent,
        fake_sft,
    ]:
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (fake_sft / "data.py").write_text(
        "\n".join(
            [
                "class SFTDataset:",
                "    def _process(self, example):",
                "        return {'native': True, 'example': example}",
                "",
                "class StatefulDataLoader:",
                "    def __init__(self, dataset, batch_size, collate_fn):",
                "        self.dataset = dataset",
                "        self.batch_size = batch_size",
                "        self.collate_fn = collate_fn",
                "",
                "def setup_dataloader(dataset, config):",
                "    return ('native', dataset, config)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plan = prime_rl._write_prime_rl_sft_compat_shim(
        tmp_path,
        sample_mean=False,
        pretokenized_data=True,
        stub_flash_attn=False,
    )
    env = os.environ.copy()
    env.update(plan.env)
    env["PYTHONPATH"] = os.pathsep.join(
        (plan.env["PYTHONPATH"], str(tmp_path / "fake"))
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import json",
                    "import prime_rl.trainer.sft.data as data",
                    "row = {",
                    "    'benchflow_custom_trainer_pretokenized': True,",
                    "    'benchflow_input_ids': [1, 2, 3],",
                    "    'benchflow_target_ids': [2, 3, 4],",
                    "    'benchflow_loss_mask': [True, False, True],",
                    "    'benchflow_position_ids': [0, 1, 2],",
                    "}",
                    "processed = data.SFTDataset()._process(row)",
                    "loader = data.setup_dataloader('dataset', 'config')",
                    "native = data.SFTDataset()._process({'messages': []})",
                    "print(json.dumps({",
                    "    'processed': processed,",
                    "    'loader_class': type(loader).__name__,",
                    "    'loader_batch_size': loader.batch_size,",
                    "    'loader_collate_callable': callable(loader.collate_fn),",
                    "    'native': native,",
                    "}, sort_keys=True))",
                ]
            ),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "BenchFlow Prime-RL pretokenized SFT data shim enabled" in completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {
        "loader_batch_size": 1,
        "loader_class": "StatefulDataLoader",
        "loader_collate_callable": True,
        "native": {"example": {"messages": []}, "native": True},
        "processed": {
            "input_ids": [1, 2, 3],
            "loss_mask": [True, False, True],
            "position_ids": [0, 1, 2],
            "target_ids": [2, 3, 4],
        },
    }


def test_prime_rl_custom_trainer_pretokenized_row_masks_last_assistant_only() -> None:
    """The compatibility row matches the old shifted-label custom trainer path."""
    import benchflow.training.backends.prime_rl as prime_rl

    assert (
        prime_rl._normalize_message_tail_truncation("custom-trainer-token-suffix")
        == "custom-trainer-pretokenized"
    )

    row, before, after, trainable_tokens = prime_rl._custom_trainer_pretokenized_row(
        _FakeTailTokenizer(),
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "OK"},
            ],
            "label_last_assistant_only": True,
            "reward": 1.0,
        },
        max_length=64,
    )

    assert before == 7
    assert after == 7
    assert trainable_tokens == 2
    assert row["benchflow_input_ids"] == [ord(char) for char in "helloO"]
    assert row["benchflow_target_ids"] == [ord(char) for char in "elloOK"]
    assert row["benchflow_loss_mask"] == [False, False, False, False, True, True]
    assert row["benchflow_position_ids"] == list(range(6))
    assert row["benchflow_custom_trainer_pretokenized"] == {
        "original_token_count": 7,
        "staged_token_count": 7,
        "trainable_token_count": 2,
    }


def test_train_run_sft_prime_rl_custom_trainer_compatibility_mode(
    tmp_path: Path, monkeypatch
) -> None:
    """Prime-RL can be launched against a custom-trainer-compatible data copy."""
    import benchflow.training.backends.prime_rl as prime_rl

    captured: dict[str, Any] = {}

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            del kwargs
            captured["argv"] = argv
            self.stdout = io.StringIO("trainer ok\n")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    config = tmp_path / "sft.toml"
    config.write_text(
        "\n".join(
            [
                "max_steps = 300",
                "[renderer]",
                'name = "qwen3.5"',
                "[data]",
                "batch_size = 8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_jsonl = tmp_path / "prime-sft.jsonl"
    source_jsonl.write_text(
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
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "ok",
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
                "reward": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "train-run"

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(work_dir),
            "--data",
            str(source_jsonl),
            "--target-examples",
            "300",
            "--loss-mask",
            "all",
            "--renderer-mode",
            "none",
            "--tool-defs-mode",
            "omit",
            "--chat-template-kwarg",
            "enable_thinking=false",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "--renderer" in captured["argv"]
    renderer_idx = captured["argv"].index("--renderer")
    assert captured["argv"][renderer_idx + 1] == "None"
    assert "--data.name" in captured["argv"]
    data_idx = captured["argv"].index("--data.name")
    staged_dir = Path(captured["argv"][data_idx + 1])
    staged_row = json.loads((staged_dir / "train.jsonl").read_text())
    assert "tool_defs" not in staged_row
    assert "tools" not in staged_row
    assert staged_row["chat_template_kwargs"] == {"enable_thinking": False}
    assert "tool_defs" in json.loads(source_jsonl.read_text())

    manifest = json.loads((work_dir / "train-run.json").read_text())
    assert manifest["extra"]["prime_rl_sft_exposure_plan"]["generated_overrides"] == [
        "max_steps=38",
        "scheduler.decay_steps=38",
        "data.loss_mask.system=true",
        "data.loss_mask.user=true",
        "data.loss_mask.assistant=true",
        "data.loss_mask.tool=true",
        "renderer=None",
    ]
    assert manifest["extra"]["prime_rl_sft_exposure_plan"]["renderer_mode"] == "none"
    assert manifest["extra"]["prime_rl_sft_dataset"]["tool_defs_mode"] == "omit"
    assert manifest["extra"]["prime_rl_sft_dataset"]["tool_defs_removed_rows"] == 1
    assert manifest["extra"]["prime_rl_sft_dataset"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert manifest["extra"]["prime_rl_sft_dataset"]["chat_template_kwargs_rows"] == 1
    assert manifest["extra"]["prime_rl_sft_dataset"]["validation"] == {
        "ok": True,
        "rows": 1,
        "rows_with_tool_calls": 1,
    }


def test_train_run_sft_prime_rl_message_tail_truncation_keeps_user_and_suffix(
    tmp_path: Path, monkeypatch
) -> None:
    """Overlength rows are staged before Prime-RL can truncate from the head."""
    import benchflow.training.backends.prime_rl as prime_rl

    captured: dict[str, Any] = {}

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            del kwargs
            captured["argv"] = argv
            self.stdout = io.StringIO("trainer ok\n")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    config = tmp_path / "sft.toml"
    config.write_text(
        "\n".join(
            [
                "max_steps = 1",
                "[model]",
                'name = "Qwen/Qwen3.5-9B"',
                "[data]",
                "batch_size = 1",
                "seq_len = 10",
                "micro_batch_size = 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_jsonl = tmp_path / "prime-sft.jsonl"
    source_jsonl.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "uu"},
                    {"role": "assistant", "content": "aaaaaa"},
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
                    {"role": "tool", "tool_call_id": "call_1", "content": "bbbbbb"},
                    {"role": "assistant", "content": "cccccc"},
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
        )
        + "\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "train-run"

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        prime_rl,
        "_load_tail_truncation_tokenizer",
        lambda model_name: _FakeTailTokenizer(),
    )

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(work_dir),
            "--data",
            str(source_jsonl),
            "--message-tail-truncation",
            "keep-first-user",
        ],
    )

    assert result.exit_code == 0, result.output
    data_idx = captured["argv"].index("--data.name")
    staged_dir = Path(captured["argv"][data_idx + 1])
    staged_row = json.loads((staged_dir / "train.jsonl").read_text())
    assert [message["content"] for message in staged_row["messages"]] == [
        "uu",
        "cccccc",
    ]

    dataset_plan = json.loads((work_dir / "train-run.json").read_text())["extra"][
        "prime_rl_sft_dataset"
    ]
    assert dataset_plan["message_tail_truncation"] == "keep-first-user"
    assert dataset_plan["message_tail_truncated_rows"] == 1
    assert dataset_plan["message_tail_max_area"] == 10
    assert dataset_plan["message_tail_max_tokens_before"] == 24
    assert dataset_plan["message_tail_max_tokens_after"] == 8


def test_train_run_sft_prime_rl_rejects_tool_defs_omit_for_remote_data(
    tmp_path: Path, monkeypatch
) -> None:
    import benchflow.training.backends.prime_rl as prime_rl

    config = tmp_path / "sft.toml"
    config.write_text("max_steps = 1\n", encoding="utf-8")

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(tmp_path / "train-run"),
            "--data",
            "benchflow/remote-dataset",
            "--tool-defs-mode",
            "omit",
        ],
    )

    assert result.exit_code == 1
    assert "--tool-defs-mode omit requires --data to be a local JSONL" in result.output


def test_train_run_sft_prime_rl_rejects_chat_template_kwargs_for_remote_data(
    tmp_path: Path, monkeypatch
) -> None:
    import benchflow.training.backends.prime_rl as prime_rl

    config = tmp_path / "sft.toml"
    config.write_text("max_steps = 1\n", encoding="utf-8")

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(tmp_path / "train-run"),
            "--data",
            "benchflow/remote-dataset",
            "--chat-template-kwarg",
            "enable_thinking=false",
        ],
    )

    assert result.exit_code == 1
    assert "--chat-template-kwarg requires --data to be a local JSONL" in result.output


def test_train_run_sft_prime_rl_rejects_message_tail_truncation_for_remote_data(
    tmp_path: Path, monkeypatch
) -> None:
    import benchflow.training.backends.prime_rl as prime_rl

    config = tmp_path / "sft.toml"
    config.write_text(
        "\n".join(["[model]", 'name = "Qwen/Qwen3.5-9B"']) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        prime_rl,
        "_load_tail_truncation_tokenizer",
        lambda model_name: _FakeTailTokenizer(),
    )

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(tmp_path / "train-run"),
            "--data",
            "benchflow/remote-dataset",
            "--message-tail-truncation",
            "keep-first-user",
        ],
    )

    assert result.exit_code == 1
    assert "--message-tail-truncation requires --data to be a local JSONL" in (
        result.output
    )


def test_train_run_sft_prime_rl_rejects_qwen35_stack_flash_attn(
    tmp_path: Path, monkeypatch
) -> None:
    """Fail closed on the Prime-RL Qwen3.5 stack/flash-attention path seen on H100."""
    import benchflow.training.backends.prime_rl as prime_rl

    config = tmp_path / "sft.toml"
    config.write_text(
        "\n".join(
            [
                "[model]",
                'name = "Qwen/Qwen3.5-9B"',
                'attn = "flash_attention_2"',
                "[data]",
                "batch_size = 8",
                'pack_function = "cat"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(tmp_path / "train-run"),
            "--pack-function",
            "stack",
        ],
    )

    assert result.exit_code == 1
    assert (
        "stack packing with Qwen/Qwen3.5-* and flash attention is blocked"
        in result.output
    )
    assert "--model-attn sdpa" in result.output


def test_train_run_sft_prime_rl_loss_mask_rejects_override_conflict(
    tmp_path: Path, monkeypatch
) -> None:
    import benchflow.training.backends.prime_rl as prime_rl

    config = tmp_path / "sft.toml"
    config.write_text("max_steps = 1\n", encoding="utf-8")

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(tmp_path / "train-run"),
            "--loss-mask",
            "all",
            "--override",
            "data.loss_mask.user=false",
        ],
    )

    assert result.exit_code == 1
    assert "--loss-mask cannot be combined" in result.output


def test_train_run_sft_prime_rl_target_examples_rejects_manual_max_steps(
    tmp_path: Path, monkeypatch
) -> None:
    import benchflow.training.backends.prime_rl as prime_rl

    config = tmp_path / "sft.toml"
    config.write_text("max_steps = 300\n[data]\nbatch_size = 8\n", encoding="utf-8")
    work_dir = tmp_path / "train-run"

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(work_dir),
            "--target-examples",
            "300",
            "--override",
            "max_steps=300",
        ],
    )

    assert result.exit_code == 1
    assert "--target-examples cannot be combined" in result.output


def test_train_run_sft_prime_rl_publish_flags_update_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    import benchflow.publish.huggingface as hf_publish
    import benchflow.training.backends.prime_rl as prime_rl

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            del argv, kwargs
            self.stdout = io.StringIO("trainer ok\n")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    class FakePublishResult:
        def __init__(self, url: str) -> None:
            self.url = url
            self.commit_url = f"{url}/commit/abc"

    published: list[tuple[str, str, str]] = []
    dataset_manifest_seen: dict[str, Any] = {}

    def fake_publish_folder_to_hf(
        folder, *, repo_id, repo_type, path_in_repo, **kwargs
    ):
        del kwargs
        if repo_type == "dataset":
            dataset_manifest_seen.update(
                json.loads((Path(folder) / "train-run.json").read_text())
            )
        published.append((repo_id, repo_type, path_in_repo))
        return FakePublishResult(
            f"https://huggingface.co/{repo_id}/tree/main/{path_in_repo}"
        )

    config = tmp_path / "sft.toml"
    config.write_text("max_steps = 1\n", encoding="utf-8")
    work_dir = tmp_path / "train-run"
    output_dir = tmp_path / "prime-output"
    output_dir.mkdir()

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(hf_publish, "publish_folder_to_hf", fake_publish_folder_to_hf)

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
            "--work-dir",
            str(work_dir),
            "--publish-model",
            "benchflow/model",
            "--model-tag",
            "env0-test",
            "--model-card",
            "auto",
            "--publish-artifacts",
            "benchflow/artifacts",
            "--hf-prefix",
            "experiments/run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert published == [
        ("benchflow/model", "model", "env0-test"),
        ("benchflow/artifacts", "dataset", "experiments/run"),
    ]
    assert dataset_manifest_seen["extra"]["published"][0]["type"] == "model"
    manifest = json.loads((work_dir / "train-run.json").read_text())
    assert len(manifest["extra"]["published"]) == 2


def test_train_run_sft_prime_rl_publish_failure_updates_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    import benchflow.publish.huggingface as hf_publish
    import benchflow.training.backends.prime_rl as prime_rl

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            del argv, kwargs
            self.stdout = io.StringIO("trainer ok\n")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    def fail_publish(*args, **kwargs):
        del args, kwargs
        raise ValueError("publish denied")

    config = tmp_path / "sft.toml"
    config.write_text("max_steps = 1\n", encoding="utf-8")
    work_dir = tmp_path / "train-run"
    output_dir = tmp_path / "prime-output"
    output_dir.mkdir()

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(hf_publish, "publish_folder_to_hf", fail_publish)

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
            "--work-dir",
            str(work_dir),
            "--publish-model",
            "benchflow/model",
        ],
    )

    assert result.exit_code == 1
    assert "publish denied" in result.output
    manifest = json.loads((work_dir / "train-run.json").read_text())
    assert manifest["overall_status"] == "failed"
    assert manifest["extra"]["publish_error"] == "publish denied"


def test_train_run_sft_prime_rl_failure_updates_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards the Prime-RL SFT wrapper against reporting failed launches as done."""
    import benchflow.training.backends.prime_rl as prime_rl

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            del argv, kwargs
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("boom\n")

        def wait(self) -> int:
            return 7

    config = tmp_path / "sft.toml"
    config.write_text("max_steps = 1\n", encoding="utf-8")
    work_dir = tmp_path / "train-run"

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(work_dir),
        ],
    )

    assert result.exit_code == 7
    assert "Prime-RL SFT failed with exit code 7" in result.output
    manifest = json.loads((work_dir / "train-run.json").read_text())
    assert manifest["overall_status"] == "failed"
    assert manifest["components"][0]["status"] == "failed"
    assert manifest["components"][0]["extra"] == {"returncode": 7}
    assert (work_dir / "prime-rl" / "stderr.log").read_text() == "boom\n"


def test_train_run_sft_resolves_config_relative_to_prime_rl_dir(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards the Prime-RL wrapper's native-checkout config path handling."""
    import benchflow.training.backends.prime_rl as prime_rl

    captured: dict[str, Any] = {}

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            del kwargs
            captured["argv"] = argv
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    prime_rl_dir = tmp_path / "prime-rl"
    config = prime_rl_dir / "examples" / "reverse_text" / "sft.toml"
    config.parent.mkdir(parents=True)
    config.write_text("max_steps = 1\n", encoding="utf-8")
    work_dir = tmp_path / "train-run"

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            "examples/reverse_text/sft.toml",
            "--prime-rl-dir",
            str(prime_rl_dir),
            "--work-dir",
            str(work_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["argv"][4] == str(config.resolve())
    manifest = json.loads((work_dir / "train-run.json").read_text())
    assert manifest["config"] == str(config.resolve())
