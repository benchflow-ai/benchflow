"""CLI coverage for ``bench train``."""

from __future__ import annotations

import json
from pathlib import Path

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
