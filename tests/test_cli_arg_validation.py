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
from benchflow.eval_plan import EvalCreateRequest, EvalPlanError, build_eval_plan
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


def test_loop_strategy_bad_spec_clean_error(tmp_path: Path):
    result = _invoke(tmp_path, "--sandbox", "docker", "--loop-strategy", "bogus")
    assert result.exit_code == 1
    assert "Invalid --loop-strategy" in result.stdout
    assert "Traceback (most recent call last)" not in result.stdout


def test_loop_strategy_k_out_of_range_rejected(tmp_path: Path):
    result = _invoke(
        tmp_path, "--sandbox", "docker", "--loop-strategy", "verify-retry:k=99"
    )
    assert result.exit_code == 1
    assert "k must be between 1 and 10" in result.stdout


def test_loop_strategy_conflicts_with_self_gen(tmp_path: Path):
    result = _invoke(
        tmp_path,
        "--sandbox",
        "docker",
        "--skill-mode",
        "self-gen",
        "--loop-strategy",
        "verify-retry",
    )
    assert result.exit_code == 1
    assert "not supported with --skill-mode self-gen" in result.stdout


def test_loop_strategy_conflicts_with_multiple_prompts(tmp_path: Path):
    result = _invoke(
        tmp_path,
        "--sandbox",
        "docker",
        "--prompt",
        "first",
        "--prompt",
        "second",
        "--loop-strategy",
        "verify-retry",
    )
    assert result.exit_code == 1
    assert "conflicts with multiple" in result.stdout


def test_loop_strategy_accepted_and_plumbed(tmp_path: Path):
    result = _invoke(
        tmp_path,
        "--sandbox",
        "docker",
        "--loop-strategy",
        "verify-retry:k=3,feedback=names",
    )
    assert result.exit_code == 0

    from benchflow.loop_strategies import LoopStrategySpec

    plumb_dir = tmp_path / "plumb"
    plumb_dir.mkdir()
    plan = build_eval_plan(
        EvalCreateRequest(
            tasks_dir=_task_dir(plumb_dir),
            loop_strategy="verify-retry:k=3,feedback=names",
        )
    )
    expected = LoopStrategySpec("verify-retry", {"k": 3, "feedback": "names"})
    assert plan.eval_loop_strategy == expected
    assert plan.make_eval_config().loop_strategy == expected


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


def test_config_file_with_no_default_model_agent_not_rejected(tmp_path: Path):
    # Regression: --config supplies the model from YAML, so build_eval_plan must
    # NOT pre-reject a no-default-model agent (e.g. codex) when --model is omitted.
    plan = build_eval_plan(
        EvalCreateRequest(config_file=tmp_path / "cfg.yaml", agent="codex")
    )
    assert plan is not None


def test_no_source_no_default_model_agent_not_rejected():
    # Regression: with no source the CLI reports "provide a source"; build_eval_plan
    # must not pre-empt that with a "no default model" error for codex.
    plan = build_eval_plan(EvalCreateRequest(agent="codex"))
    assert plan is not None


def test_sandbox_modal_missing_extra_rejected_even_when_installed(
    tmp_path, monkeypatch
):
    # Covers the missing-extra preflight in CI, where the modal extra IS installed
    # (so test_sandbox_modal_without_extra_fails_fast skips). Force `import modal`
    # to fail and assert the actionable EvalPlanError fires.
    import builtins

    task = _task_dir(tmp_path)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "modal":
            raise ModuleNotFoundError("No module named 'modal'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(EvalPlanError, match="sandbox-modal"):
        build_eval_plan(
            EvalCreateRequest(tasks_dir=task, environment="modal", agent="oracle")
        )


def test_source_env_skips_sandbox_preflight(monkeypatch):
    # Regression: --sandbox is ignored by hosted source-env runs, so a missing
    # modal extra (and an unknown sandbox value) must NOT block them — only
    # --tasks-dir/--source-repo/--config use the local sandbox.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "modal":
            raise ModuleNotFoundError("No module named 'modal'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    plan = build_eval_plan(EvalCreateRequest(source_env="org/env", environment="modal"))
    assert plan is not None


@pytest.mark.parametrize("ok", [None, 1, 900.0])
def test_agent_timeout_sec_accepts_positive_or_none(ok):
    assert AgentConfig(timeout_sec=ok).timeout_sec == ok
