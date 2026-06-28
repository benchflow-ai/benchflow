"""CLI coverage for ``bench train``."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from benchflow.cli.main import app

runner = CliRunner()


def _write_rollout(rollout_dir: Path) -> None:
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "result.json").write_text(
        json.dumps({"task_name": "task-a", "rewards": {"reward": 1.0}})
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
