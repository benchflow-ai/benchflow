"""Daytona auto-stop and auto-delete intervals can be shortened per process.

Guards the RL cookbooks' cleanup (cookbook/rl-core): a training or evaluation
run that is killed before it closes its sandboxes must not leave them running
for the day-long default.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from benchflow.sandbox.setup import _create_sandbox_environment, _daytona_minutes
from benchflow.task import Task


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "instruction.md").write_text("Do it.")
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[task]\nname = "benchflow/ttl"\n\n[environment]\n'
    )
    return task


def _captured(tmp_path: Path) -> dict:
    task_dir = _task(tmp_path)
    captured: dict = {}

    class _FakeSandbox:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    rollout_paths = type("Paths", (), {"rollout_dir": tmp_path / "rollout"})()
    with (
        patch("benchflow.sandbox.daytona.DaytonaSandbox", _FakeSandbox),
        patch("benchflow.sandbox.daytona._load_daytona_sdk", lambda: None),
        patch("benchflow.sandbox._sdk_ops.apply", lambda: None),
    ):
        _create_sandbox_environment(
            "daytona", Task(task_dir), task_dir, "r-1", rollout_paths
        )
    return captured


def test_default_intervals_are_a_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BENCHFLOW_DAYTONA_AUTO_STOP_MINS", raising=False)
    monkeypatch.delenv("BENCHFLOW_DAYTONA_AUTO_DELETE_MINS", raising=False)
    captured = _captured(tmp_path)
    assert captured["auto_stop_interval_mins"] == 1440
    assert captured["auto_delete_interval_mins"] == 1440


def test_env_shortens_the_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCHFLOW_DAYTONA_AUTO_STOP_MINS", "30")
    monkeypatch.setenv("BENCHFLOW_DAYTONA_AUTO_DELETE_MINS", "15")
    captured = _captured(tmp_path)
    assert captured["auto_stop_interval_mins"] == 30
    assert captured["auto_delete_interval_mins"] == 15


@pytest.mark.parametrize("raw", ["soon", "0", "-5"])
def test_bad_values_fail_loudly(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCHFLOW_DAYTONA_AUTO_STOP_MINS", raw)
    with pytest.raises(ValueError, match="BENCHFLOW_DAYTONA_AUTO_STOP_MINS"):
        _daytona_minutes("BENCHFLOW_DAYTONA_AUTO_STOP_MINS")
