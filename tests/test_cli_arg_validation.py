"""Regression tests for CLI/config input validation.

These lock in the v0.6.0 stress-test fixes: invalid arguments must fail fast
with a clean message (no deadlock, no raw traceback) before any rollout starts.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pydantic
import pytest
from typer.testing import CliRunner

from benchflow._utils.config import normalize_reasoning_effort
from benchflow.cli.main import app
from benchflow.evaluation import Evaluation
from benchflow.task.config import AgentConfig


def _task_dir(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('schema_version = "1.1"\n')
    (task / "instruction.md").write_text("solve\n")
    return task


async def _fake_run_pass(self):
    # If a validation guard fails to fire, the run is reached and "succeeds";
    # the exit-code assertion then catches the regression.
    return SimpleNamespace(passed=1, total=1, score=1.0, errored=0, verifier_errored=0)


def _invoke(tmp_path: Path, *extra: str):
    task = _task_dir(tmp_path)
    with patch.object(Evaluation, "run", new=_fake_run_pass):
        return CliRunner().invoke(
            app,
            ["eval", "create", "--tasks-dir", str(task), "--agent", "oracle", *extra],
        )


def test_concurrency_zero_rejected_not_deadlocked(tmp_path: Path):
    # Semaphore(0) would deadlock; the CLI must reject it up front instead.
    result = _invoke(tmp_path, "--sandbox", "docker", "--concurrency", "0")
    assert result.exit_code == 1
    assert "--concurrency must be >= 1" in result.stdout
    assert "Traceback (most recent call last)" not in result.stdout


def test_build_concurrency_zero_rejected(tmp_path: Path):
    result = _invoke(
        tmp_path,
        "--sandbox",
        "docker",
        "--concurrency",
        "4",
        "--build-concurrency",
        "0",
    )
    assert result.exit_code == 1
    assert "--build-concurrency must be >= 1" in result.stdout


def test_skill_mode_bogus_clean_error(tmp_path: Path):
    result = _invoke(tmp_path, "--sandbox", "docker", "--skill-mode", "bogus")
    assert result.exit_code == 1
    assert "Invalid --skill-mode" in result.stdout
    assert "Traceback (most recent call last)" not in result.stdout


def test_sandbox_bogus_clean_error(tmp_path: Path):
    # Unknown sandbox values must be rejected at planning, not surface a raw
    # per-task traceback once the rollout starts.
    result = _invoke(tmp_path, "--sandbox", "nope")
    assert result.exit_code == 1
    assert "Invalid --sandbox" in result.stdout
    assert "Traceback (most recent call last)" not in result.stdout


def test_reasoning_effort_bogus_clean_error(tmp_path: Path):
    result = _invoke(tmp_path, "--sandbox", "docker", "--reasoning-effort", "banana")
    assert result.exit_code == 1
    assert "reasoning_effort must be one of" in result.stdout
    assert "Traceback (most recent call last)" not in result.stdout


def test_tasks_dir_missing_clean_error(tmp_path: Path):
    with patch.object(Evaluation, "run", new=_fake_run_pass):
        result = CliRunner().invoke(
            app,
            [
                "eval",
                "create",
                "--tasks-dir",
                str(tmp_path / "does-not-exist"),
                "--agent",
                "oracle",
                "--sandbox",
                "docker",
            ],
        )
    assert result.exit_code == 1
    assert "--tasks-dir not found" in result.stdout
    assert "Traceback (most recent call last)" not in result.stdout


def test_agent_without_default_model_clean_error(tmp_path: Path):
    # codex has no default model; omitting --model must report cleanly, not crash.
    result = _invoke(tmp_path, "--agent", "codex", "--sandbox", "docker")
    assert result.exit_code == 1
    assert "no default model" in result.stdout
    assert "Traceback (most recent call last)" not in result.stdout


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("modal") is not None,
    reason="modal extra installed; missing-extra preflight does not fire",
)
def test_sandbox_modal_without_extra_fails_fast(tmp_path: Path):
    result = _invoke(tmp_path, "--sandbox", "modal")
    assert result.exit_code == 1
    assert "sandbox-modal" in result.stdout
    assert "Traceback (most recent call last)" not in result.stdout


@pytest.mark.parametrize("value", ["banana", "fastest", "9", "lowish"])
def test_normalize_reasoning_effort_rejects_unknown(value: str):
    with pytest.raises(ValueError, match="reasoning_effort must be one of"):
        normalize_reasoning_effort(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("max", "max"),
        ("MAX", "max"),
        ("xhigh", "xhigh"),
        ("minimal", "minimal"),
        ("", None),
    ],
)
def test_normalize_reasoning_effort_accepts_known(value: str, expected):
    assert normalize_reasoning_effort(value) == expected


@pytest.mark.parametrize("bad", [-5, 0, -0.1])
def test_agent_timeout_sec_rejects_nonpositive(bad):
    with pytest.raises(pydantic.ValidationError):
        AgentConfig(timeout_sec=bad)


@pytest.mark.parametrize("ok", [None, 1, 900.0])
def test_agent_timeout_sec_accepts_positive_or_none(ok):
    assert AgentConfig(timeout_sec=ok).timeout_sec == ok
