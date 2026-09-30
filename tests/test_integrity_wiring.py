"""The integrity option reaches every run path, and the rollout hook is safe.

- ``--integrity`` / YAML / EvaluationConfig / RolloutConfig / TaskRuntimeConfig
  and the sharded worker all carry the mode.
- Strict mode is BenchFlow's separate verifier: it is refused with the launch
  gate's own reasons where that sandbox cannot run, and it only moves the
  verifier.
- The rollout hook writes a verdict without touching rewards, skips reviewer
  rollouts, and fails closed (a Rejected verdict, no exception) when the
  audit itself breaks.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow.cli import main as cli_main
from benchflow.integrity import read_verdict, strict_launch_issues
from benchflow.integrity.trial import force_separate_verifier
from benchflow.models import RolloutResult
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout.task_runtime import TaskRuntimeConfig
from benchflow.task import Task
from benchflow.task.verifier_sandbox import separate_verifier_requested

REPO = Path(__file__).resolve().parents[1]
TASK = REPO / "tests" / "integration" / "deterministic" / "task"


def test_rollout_config_round_trips_integrity() -> None:
    config = RolloutConfig(task_path=TASK, agent="oracle", integrity="STRICT")
    assert config.integrity == "strict"
    assert RolloutConfig.from_dict(config.to_dict()).integrity == "strict"
    assert RolloutConfig(task_path=TASK, agent="oracle").integrity == "off"
    with pytest.raises(ValueError, match="integrity must be one of"):
        RolloutConfig(task_path=TASK, agent="oracle", integrity="paranoid")


def test_task_runtime_passes_integrity_to_its_rollout() -> None:
    config = TaskRuntimeConfig(task_path=TASK, integrity="audit")
    assert config.to_rollout_config().integrity == "audit"


def test_evaluation_yaml_and_cli_flag_reach_the_config(
    tmp_path: Path, monkeypatch
) -> None:
    seen: dict[str, Any] = {}

    async def fake_run(self):
        seen["config"] = self._config
        raise SystemExit(0)

    monkeypatch.setattr(bf.Evaluation, "run", fake_run)
    evaluation = bf.Evaluation(
        tasks_dir=TASK.parent,
        jobs_dir=tmp_path / "jobs",
        config=bf.EvaluationConfig(agent="oracle", integrity="strict"),
    )
    path = evaluation.to_yaml(tmp_path / "job.yaml")
    CliRunner().invoke(cli_main.app, ["eval", "run", "--config", str(path)])
    assert seen["config"].integrity == "strict"

    seen.clear()
    CliRunner().invoke(
        cli_main.app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(TASK.parent),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(tmp_path / "jobs2"),
            "--integrity",
            "audit",
        ],
    )
    assert seen["config"].integrity == "audit"

    # --integrity on top of --config wins over the file, and is not dropped:
    # a safety flag that goes missing fails in the unsafe direction (the
    # operator believes the trials are audited and they are not).
    seen.clear()
    off = bf.Evaluation(
        tasks_dir=TASK.parent,
        jobs_dir=tmp_path / "jobs3",
        config=bf.EvaluationConfig(agent="oracle"),
    ).to_yaml(tmp_path / "off.yaml")
    CliRunner().invoke(
        cli_main.app, ["eval", "run", "--config", str(off), "--integrity", "strict"]
    )
    assert seen["config"].integrity == "strict"

    # ... and without the flag the file's own value still stands.
    seen.clear()
    CliRunner().invoke(cli_main.app, ["eval", "run", "--config", str(path)])
    assert seen["config"].integrity == "strict"


def test_a_bad_integrity_value_is_one_red_line_not_a_traceback(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli_main.app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(TASK.parent),
            "--agent",
            "oracle",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--integrity",
            "bogus",
        ],
    )
    assert result.exit_code == 1
    assert "integrity must be one of off, audit, strict" in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "jobs").exists()  # refused before a job folder


def test_sharded_workers_carry_integrity() -> None:
    from benchflow.eval_sharding import EvalShard, _config_payload
    from benchflow.eval_worker import _evaluation_config

    config = bf.EvaluationConfig(agent="oracle", integrity="audit")
    shard = EvalShard(index=0, task_names=("task",), concurrency=1)
    payload = _config_payload(config, shard=shard)
    assert payload["integrity"] == "audit"
    assert _evaluation_config(payload).integrity == "audit"


def test_strict_is_the_separate_verifier_and_is_gated_by_its_launch_gate() -> None:
    config = Task(TASK).config
    assert strict_launch_issues(config, sandbox="docker", task_dir=TASK) == []
    issues = strict_launch_issues(config, sandbox="modal", task_dir=TASK)
    assert issues and "separate verifier" in issues[0]
    judge = config.model_copy(
        update={"verifier": config.verifier.model_copy(update={"type": "llm-judge"})}
    )
    issues = strict_launch_issues(judge, sandbox="docker", task_dir=TASK)
    assert any("test-script" in issue for issue in issues)

    forced = force_separate_verifier(config)
    assert separate_verifier_requested(forced)
    assert not separate_verifier_requested(config)  # the task's own config is untouched
    assert forced.agent == config.agent and forced.sandbox == config.sandbox


def test_strict_refuses_before_any_sandbox(tmp_path: Path) -> None:
    rollout = Rollout(
        RolloutConfig(
            task_path=TASK,
            agent="oracle",
            environment="modal",
            integrity="strict",
            jobs_dir=tmp_path / "jobs",
        )
    )
    with pytest.raises(ValueError, match="integrity=strict"):
        asyncio.run(rollout.setup())
    assert rollout._env is None


# --- the rollout hook --------------------------------------------------------------


def _finished_rollout(tmp_path: Path, *, integrity: str, purpose: str = "task"):
    trial = tmp_path / "job" / "hello__abc"
    (trial / "verifier").mkdir(parents=True)
    (trial / "verifier" / "reward.txt").write_text("1.0")
    config = RolloutConfig(
        task_path=TASK, agent="claude-agent-acp", integrity=integrity, purpose=purpose
    )
    rollout = SimpleNamespace(
        _config=config,
        _rollout_dir=trial,
        _rollout_name="hello__abc",
        _task=Task(TASK),
        _agent_cwd="/app",
        _effective_skills_dir=None,
        _effective_skills_sandbox_dir=None,
    )
    trajectory = [
        {
            "type": "tool_call",
            "tool_call_id": "t1",
            "kind": "execute",
            "raw_input": {"command": "cat /solution/solve.sh"},
            "status": "completed",
        }
    ]
    rewards = {"reward": 1.0}
    result = RolloutResult(
        task_name="task",
        rollout_name="hello__abc",
        rewards=rewards,
        trajectory=trajectory,
        agent="claude-agent-acp",
        trajectory_source="acp",
        rollout_dir=trial,
    )
    return rollout, result, trial


def test_rollout_hook_writes_a_verdict_and_never_changes_the_reward(
    tmp_path: Path,
) -> None:
    rollout, result, trial = _finished_rollout(tmp_path, integrity="audit")
    Rollout._write_integrity(rollout, result)
    assert result.rewards == {"reward": 1.0}
    assert result.integrity is not None and result.integrity["exploited"] is True
    verdict = read_verdict(trial)
    assert verdict is not None and verdict.verdict == "AgentViolation"
    assert (
        json.loads((trial / "integrity" / "claim_verdict.json").read_text())[
            "reward_effect"
        ]
        == "none"
    )


def test_rollout_hook_is_off_by_default_and_skips_reviewers(tmp_path: Path) -> None:
    rollout, result, trial = _finished_rollout(tmp_path / "off", integrity="off")
    Rollout._write_integrity(rollout, result)
    assert not (trial / "integrity").exists() and result.integrity is None
    rollout, result, trial = _finished_rollout(
        tmp_path / "rev", integrity="audit", purpose="reviewer"
    )
    Rollout._write_integrity(rollout, result)
    assert not (trial / "integrity").exists()


def test_an_audit_that_cannot_record_its_own_failure_is_dropped(
    tmp_path: Path, monkeypatch
) -> None:
    """The fallback's own write can fail too, and must not fail the trial.

    The disk that makes the audit fail (full, read-only, wrong permissions) is
    the same disk the Rejected fallback is written to, so both are inside the
    guard. An escape here would turn a scored trial into an errored one: its
    result.json already holds a complete reward, but the job records the trial
    as ``Unexpected: ...``.
    """

    import benchflow.integrity.trial as trial_module

    def boom(evidence):
        raise RuntimeError("evidence store unreadable")

    def boom_too(trial_dir, *, mode, error, source="rollout"):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(trial_module, "emit_integrity", boom)
    monkeypatch.setattr(trial_module, "emit_integrity_error", boom_too)
    rollout, result, trial = _finished_rollout(tmp_path, integrity="audit")
    Rollout._write_integrity(rollout, result)  # must not raise into scoring
    assert result.integrity is None and read_verdict(trial) is None
    assert result.rewards == {"reward": 1.0}


def test_a_bad_mode_set_after_construction_does_not_raise(tmp_path: Path) -> None:
    """normalize_integrity_mode is inside the guard, not before it."""

    rollout, result, _ = _finished_rollout(tmp_path, integrity="audit")
    rollout._config.integrity = "bogus"  # bypasses __post_init__ validation
    Rollout._write_integrity(rollout, result)
    assert result.integrity is None and result.rewards == {"reward": 1.0}


def test_a_broken_audit_fails_closed_without_raising(
    tmp_path: Path, monkeypatch
) -> None:
    import benchflow.integrity.trial as trial_module

    def boom(evidence):
        raise RuntimeError("evidence store unreadable")

    monkeypatch.setattr(trial_module, "emit_integrity", boom)
    rollout, result, trial = _finished_rollout(tmp_path, integrity="audit")
    Rollout._write_integrity(rollout, result)  # must not raise into scoring
    verdict = read_verdict(trial)
    assert verdict is not None
    assert verdict.verdict == "Rejected" and not verdict.exploited
    assert "integrity audit failed" in verdict.reason
    assert result.rewards == {"reward": 1.0}
