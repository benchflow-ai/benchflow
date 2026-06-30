"""CLI coverage for ``bench train``."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from benchflow.cli.main import app

runner = CliRunner()


class _FakeTailTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any):
        del kwargs
        length = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                length += len(content)
            length += 4 * len(message.get("tool_calls") or [])
        return [1] * length + [self.eos_token_id]


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
        "sync_scheduler_to_max_steps": True,
        "target_examples": 300,
        "target_micro_steps": None,
        "unapplied_micro_steps": None,
    }


def test_train_run_sft_prime_rl_mobile300_compat_profile(
    tmp_path: Path, monkeypatch
) -> None:
    """The Mobile300 profile expands to the validated Prime-RL wrapper settings."""
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
    assert argv[-18:] == [
        "--max_steps",
        "37",
        "--scheduler.decay_steps",
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
    assert staged_row["chat_template_kwargs"] == {"enable_thinking": False}

    manifest = json.loads((work_dir / "train-run.json").read_text())
    assert manifest["extra"]["prime_rl_sft_compat_profile"]["name"] == (
        "env0-mobile300-pr828"
    )
    assert manifest["extra"]["prime_rl_sft_compat_profile"]["resolved_settings"] == {
        "target_examples": None,
        "target_micro_steps": 300,
        "sync_scheduler_to_max_steps": True,
        "pack_function": "stack",
        "loss_mask": "all",
        "model_attn": "sdpa",
        "renderer_mode": "none",
        "tool_defs_mode": "omit",
        "chat_template_kwargs": {"enable_thinking": False},
        "message_tail_truncation": "keep-first-user",
    }
    assert (
        manifest["extra"]["prime_rl_sft_exposure_plan"]["effective_train_examples"]
        == 296
    )
    assert manifest["extra"]["prime_rl_sft_exposure_plan"]["unapplied_micro_steps"] == 4
    assert manifest["extra"]["prime_rl_sft_dataset"]["tool_defs_mode"] == "omit"
    assert manifest["extra"]["prime_rl_sft_dataset"]["tool_defs_removed_rows"] == 1
    assert manifest["extra"]["prime_rl_sft_dataset"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert manifest["extra"]["prime_rl_sft_dataset"]["chat_template_kwargs_rows"] == 1
    assert manifest["extra"]["prime_rl_sft_dataset"]["message_tail_truncation"] == (
        "keep-first-user"
    )
    assert manifest["extra"]["prime_rl_sft_dataset"]["message_tail_truncated_rows"] == 0
    assert manifest["extra"]["prime_rl_sft_dataset"]["message_tail_max_area"] == 128


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
