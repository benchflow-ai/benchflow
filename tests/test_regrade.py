"""``bench eval regrade``: re-run a changed verifier on frozen trial workspaces.

A trial is regradable only when its final workspace was frozen before
verifier hardening (``evidence/``, written for rubric review, verifier
recovery, or ``freeze_workspace=True``). Everything else is reported as not
regradable, never guessed.
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.rollout import _review
from benchflow.sandbox.protocol import ExecResult


class LocalTransport:
    async def exec(
        self, cmd: str, *, user: str = "root", timeout_sec: int = 30
    ) -> ExecResult:
        argv = shlex.split(cmd)
        if argv[0] == "python3":
            argv[0] = sys.executable
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        assert process.returncode is not None
        return ExecResult(process.returncode, stdout.decode(), stderr.decode())

    async def download_file(self, src: str, dst: Path) -> None:
        shutil.copyfile(src, dst)


def _freeze_rollout(tmp_path: Path, *, freeze: bool):
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("42\n")
    rollout_dir = tmp_path / "trial"
    rollout_dir.mkdir()

    async def idle(*_args):
        return None

    return SimpleNamespace(
        _branch_child_active=False,
        _review_plan=None,
        _config=SimpleNamespace(
            purpose="task", sandbox_user=None, freeze_workspace=freeze
        ),
        _env=LocalTransport(),
        _agent_env={},
        _planes=SimpleNamespace(quiesce_agent=idle),
        _task=None,  # no recovery contract
        _agent_cwd=str(workspace),
        disconnect=idle,
        _require_rollout_dir=lambda: rollout_dir,
    ), rollout_dir


@pytest.mark.asyncio
async def test_freeze_workspace_captures_evidence_without_a_reviewer(tmp_path):
    rollout, rollout_dir = _freeze_rollout(tmp_path, freeze=True)
    rollout._task = SimpleNamespace(
        config=SimpleNamespace(
            artifacts=[], verifier=SimpleNamespace(submission_files=[])
        ),
    )
    await _review.capture_terminal_workspace(rollout)
    assert getattr(rollout, "_export_error", None) is None
    frozen = rollout_dir / "evidence" / "workspace" / "answer.txt"
    assert frozen.read_text() == "42\n"


@pytest.mark.asyncio
async def test_without_freeze_or_reviewer_nothing_is_captured(tmp_path, monkeypatch):
    rollout, rollout_dir = _freeze_rollout(tmp_path, freeze=False)
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.recovery_ineligible_reason",
        lambda _rollout: "no contract",
    )
    await _review.capture_terminal_workspace(rollout)
    assert not (rollout_dir / "evidence").exists()


def test_freeze_workspace_is_an_evaluation_and_rollout_option():
    from benchflow import EvaluationConfig
    from benchflow.rollout import RolloutConfig

    assert RolloutConfig.__dataclass_fields__["freeze_workspace"].default is False
    config = EvaluationConfig(freeze_workspace=True)
    assert config.freeze_workspace is True


# --- regrade orchestration --------------------------------------------------

import json  # noqa: E402

from benchflow.eval_regrade import (  # noqa: E402
    REGRADE_FILE,
    SUMMARY_FILE,
    aregrade,
    find_trials,
    verdict_change,
)
from benchflow.review.evidence import capture_workspace  # noqa: E402


def _task(tasks: Path, name: str, verifier: str) -> Path:
    task = tasks / name
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n[verifier]\ntimeout_sec = 60\n')
    (task / "instruction.md").write_text("Write the answer to answer.txt.\n")
    (task / "tests" / "test.sh").write_text(verifier)
    return task


async def _trial(job: Path, name: str, task: str, reward, *, frozen: bool, tmp: Path):
    trial = job / name
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(
        json.dumps(
            {"task_path": task, "environment": "daytona", "task_digest": "sha256:old"}
        )
    )
    rewards = None if reward is None else {"reward": reward}
    (trial / "result.json").write_text(
        json.dumps({"task_name": task, "rewards": rewards, "verifier_error": None})
    )
    if frozen:
        workspace = tmp / f"ws-{name}"
        workspace.mkdir()
        (workspace / "answer.txt").write_text("done\n")
        await capture_workspace(LocalTransport(), str(workspace), trial / "evidence")
    return trial


def _runner(new_rewards: dict[str, dict | None], calls: list):
    async def run(trial, task_dir, attempt, *, sandbox=None):
        calls.append((trial.name, task_dir.name, sandbox))
        assert (trial / "evidence" / "workspace" / "answer.txt").is_file()
        (attempt / "verifier").mkdir()
        (attempt / "verifier" / "reward.txt").write_text("1\n")
        return new_rewards[trial.name], None

    return run


@pytest.fixture
async def job(tmp_path):
    tasks = tmp_path / "tasks-fixed"
    _task(tasks, "t1", "#!/bin/bash\ngrep -qi done /app/answer.txt\n")
    _task(tasks, "t2", "#!/bin/bash\nexit 0\n")
    job = tmp_path / "jobs" / "run1"
    await _trial(job, "t1__a", "t1", 0.0, frozen=True, tmp=tmp_path)
    await _trial(job, "t2__b", "t2", 1.0, frozen=True, tmp=tmp_path)
    await _trial(job, "t1__c", "t1", 0.0, frozen=False, tmp=tmp_path)
    return job, tasks


@pytest.mark.asyncio
async def test_regrade_writes_new_scores_beside_untouched_originals(job):
    job, tasks = job
    before = {t.name: (t / "result.json").read_bytes() for t in find_trials(job)}
    calls: list = []

    summary = await aregrade(
        job,
        tasks_dir=tasks,
        reason="verifier fix",
        runner=_runner({"t1__a": {"reward": 1.0}, "t2__b": {"reward": 1.0}}, calls),
    )

    assert sorted(calls) == [("t1__a", "t1", None), ("t2__b", "t2", None)]
    assert summary.counts() == {
        "trials": 3,
        "regraded": 2,
        "changed": 1,
        "fail_to_pass": 1,
        "pass_to_fail": 0,
        "failed": 0,
        "not_regradable": 1,
    }
    [changed] = summary.changed
    assert (changed.trial, changed.change) == ("t1__a", "fail->pass")
    [skipped] = summary.not_regradable
    assert skipped.trial == "t1__c" and "no frozen workspace" in skipped.reason
    assert {
        t.name: (t / "result.json").read_bytes() for t in find_trials(job)
    } == before

    record = json.loads((job / "t1__a" / REGRADE_FILE).read_text())
    assert record["original"]["reward"] == 0.0
    [block] = record["regrades"]
    assert block["id"] == summary.regrade_id == record["latest"]
    assert block["original_reward"] == 0.0 and block["new_reward"] == 1.0
    assert block["reason"] == "verifier fix"
    assert block["verifier_digest"].startswith("sha256:")
    assert block["task_changed"] is True
    assert block["status"] == "complete" and block["change"] == "fail->pass"
    attempt = job / "t1__a" / "regrade" / summary.regrade_id
    assert (attempt / "verifier-files" / "test.sh").is_file()
    assert json.loads((attempt / REGRADE_FILE).read_text()) == block
    assert not (job / "t1__c" / REGRADE_FILE).exists()

    written = json.loads((job / SUMMARY_FILE).read_text())
    assert written["counts"] == summary.counts()
    assert [row["trial"] for row in written["changed"]] == ["t1__a"]


@pytest.mark.asyncio
async def test_a_second_regrade_appends_and_keeps_the_run_original(job):
    job, tasks = job
    first = _runner({"t1__a": {"reward": 1.0}, "t2__b": {"reward": 1.0}}, [])
    second = _runner({"t1__a": {"reward": 0.5}, "t2__b": {"reward": 0.0}}, [])
    await aregrade(job, tasks_dir=tasks, runner=first)
    summary = await aregrade(job, tasks_dir=tasks, runner=second)

    record = json.loads((job / "t1__a" / REGRADE_FILE).read_text())
    assert record["original"]["reward"] == 0.0
    assert [b["new_reward"] for b in record["regrades"]] == [1.0, 0.5]
    assert {t.trial: t.change for t in summary.regraded} == {
        "t1__a": "reward 0 -> 0.5",
        "t2__b": "pass->fail",
    }


@pytest.mark.asyncio
async def test_missing_task_and_tampered_workspace_are_not_regradable(job):
    job, tasks = job
    shutil.rmtree(tasks / "t2")
    (job / "t1__a" / "evidence" / "workspace" / "answer.txt").write_text("edited\n")

    summary = await aregrade(job, tasks_dir=tasks, runner=_runner({}, []))

    reasons = {t.trial: t.reason for t in summary.not_regradable}
    assert "does not match its manifest" in reasons["t1__a"]
    assert "task 't2' not found" in reasons["t2__b"]
    assert summary.regraded == []


@pytest.mark.asyncio
async def test_without_tasks_dir_an_unrecorded_task_is_not_guessed(job):
    job, _ = job
    summary = await aregrade(job / "t1__a", runner=_runner({}, []))
    [row] = summary.trials
    assert row.status == "not_regradable" and "--tasks-dir" in row.reason


@pytest.mark.asyncio
async def test_a_sandbox_failure_is_recorded_not_scored(job):
    job, tasks = job

    async def broken(trial, task_dir, attempt, *, sandbox=None):
        raise RuntimeError("sandbox create failed")

    summary = await aregrade(job / "t1__a", tasks_dir=tasks, runner=broken)
    [row] = summary.trials
    assert row.status == "failed" and "sandbox create failed" in row.reason
    [block] = json.loads((job / "t1__a" / REGRADE_FILE).read_text())["regrades"]
    assert block["status"] == "failed" and block["new_reward"] is None
    assert block["change"] is None
    assert json.loads((job / "t1__a" / "result.json").read_text())["rewards"] == {
        "reward": 0.0
    }


def test_verdict_change():
    assert verdict_change(0.0, 1.0) == "fail->pass"
    assert verdict_change(1.0, 0.0) == "pass->fail"
    assert verdict_change(0.2, 0.4) == "reward 0.2 -> 0.4"
    assert verdict_change(1.0, 1.0) == "same"
    assert verdict_change(None, 1.0) == "scored"
    assert verdict_change(1.0, None) == "unscored"


def test_the_cli_reports_changed_verdicts(job, monkeypatch):
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    job, tasks = job
    monkeypatch.setattr(
        "benchflow.eval_regrade.run_verifier_on_frozen_trial",
        _runner({"t1__a": {"reward": 1.0}, "t2__b": {"reward": 1.0}}, []),
    )
    result = CliRunner().invoke(
        app,
        ["eval", "regrade", str(job), "--tasks-dir", str(tasks), "--reason", "fix"],
    )
    assert result.exit_code == 0, result.output
    assert "fail->pass" in result.output
    assert "t1__c" in result.output and "not regradable" in result.output
    assert json.loads((job / SUMMARY_FILE).read_text())["reason"] == "fix"


def test_regrade_is_public():
    import benchflow as bf

    assert bf.regrade is not None and "regrade" in bf.__all__
    assert "aregrade" in bf.__all__ and "RegradeSummary" in bf.__all__


def test_move_script_swaps_directory_contents_in_place(tmp_path):
    """On Daytona, replacing /app by rmtree during a regrade left every
    later shell with 'getcwd: cannot access parent directories'."""
    import os
    import subprocess

    from benchflow.eval_regrade import _MOVE_SCRIPT

    dest, src = tmp_path / "app", tmp_path / "restored"
    dest.mkdir()
    (dest / "stale.txt").write_text("fresh-image file\n")
    (src / "sub").mkdir(parents=True)
    (src / "answer.json").write_text("{}\n")
    inode = dest.stat().st_ino
    subprocess.run(
        [sys.executable, "-c", _MOVE_SCRIPT, str(dest), str(src)], check=True
    )
    assert dest.stat().st_ino == inode
    assert sorted(os.listdir(dest)) == ["answer.json", "sub"]
    assert not src.exists()
    file_src = tmp_path / "one.txt"
    file_src.write_text("x\n")
    subprocess.run(
        [
            sys.executable,
            "-c",
            _MOVE_SCRIPT,
            str(tmp_path / "out" / "one.txt"),
            str(file_src),
        ],
        check=True,
    )
    assert (tmp_path / "out" / "one.txt").read_text() == "x\n"
