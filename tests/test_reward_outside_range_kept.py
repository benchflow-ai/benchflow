"""A reward outside 0-1 is recorded as test.sh wrote it, with a note.

The EN-278 lead's rule: under integrity ``audit`` and ``off`` the reward is
kept exactly as written (AlgoTune's speedup 1.0079, sldbench's -1.0), the
trial record and the job summary carry an "outside 0-1" note, only the
reward field is judged, and nothing is rewritten to 0. ``strict`` is
unchanged: the reward is refused and the trial scores 0 with the note.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.rollout import _ensure_canonical_rewards
from benchflow.task.verifier import Verifier


def _verifier(tmp_path: Path) -> Verifier:
    from benchflow.task import TaskConfig

    task = MagicMock()
    task.config = TaskConfig.model_validate({"verifier": {}})
    paths = MagicMock()
    paths.reward_text_path = tmp_path / "reward.txt"
    paths.reward_json_path = tmp_path / "reward.json"
    return Verifier(task=task, rollout_paths=paths, sandbox=MagicMock())


@pytest.mark.parametrize("written", ["1.0079", "-1.0", "-1"])
def test_reward_txt_outside_0_1_is_recorded_as_written(tmp_path, written):
    (tmp_path / "reward.txt").write_text(written + "\n")
    rewards = _verifier(tmp_path)._parse_reward_text()
    assert rewards == {"reward": float(written)}
    assert _ensure_canonical_rewards(rewards) == {"reward": float(written)}


@pytest.mark.parametrize("written", [1.0079, -1.0, -1])
def test_reward_json_outside_0_1_is_recorded_as_written(tmp_path, written):
    (tmp_path / "reward.json").write_text(json.dumps({"reward": written}))
    rewards = _verifier(tmp_path)._parse_reward_json_with_text_compat()
    assert rewards == {"reward": written}
    assert _ensure_canonical_rewards(rewards) == {"reward": written}


@pytest.mark.parametrize(
    "payload",
    [
        # researchcodebench: a count beside the reward is a detail.
        {"reward": 1.0079, "code_lines": 5},
        {"reward": -1.0, "code_lines": 5, "metrics": {"speedup": 2.5}},
    ],
)
def test_only_the_reward_field_is_judged(tmp_path, payload):
    (tmp_path / "reward.json").write_text(json.dumps(payload))
    rewards = _verifier(tmp_path)._parse_reward_json_with_text_compat()
    assert rewards == payload
    assert _ensure_canonical_rewards(rewards) == payload


def test_a_count_beside_an_in_range_reward_scores_on_the_reward(tmp_path):
    (tmp_path / "reward.json").write_text(json.dumps({"reward": 1.0, "code_lines": 5}))
    rewards = _verifier(tmp_path)._parse_reward_json_with_text_compat()
    assert rewards == {"reward": 1.0, "code_lines": 5}


class TestOutcome:
    @pytest.fixture
    def harness(self, tmp_path):
        from benchflow.sdk import SDK

        task = MagicMock()
        task.config.verifier.timeout_sec = 5
        task.config.verifier.env = None
        task.config.verifier.reward_range = None
        tp = MagicMock()
        tp.verifier_dir = tmp_path / "verifier"
        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
        return SDK(), env, task, tp

    async def _verify(self, harness, rewards: dict):
        sdk, env, task, tp = harness
        verifier = MagicMock()
        verifier.verify = AsyncMock(return_value=MagicMock(rewards=rewards))
        with (
            patch("benchflow.task.Verifier", return_value=verifier),
            patch(
                "benchflow.sandbox.lockdown.harden_before_verify",
                new_callable=AsyncMock,
            ),
        ):
            return await sdk._verify(env, task, tp, {})

    @pytest.mark.parametrize("written", [1.0079, -1.0])
    @pytest.mark.asyncio
    async def test_recorded_as_written_with_the_note(
        self, harness, tmp_path, caplog, written
    ):
        from benchflow.diagnostics import RolloutDiagnostics
        from benchflow.rollout._results import _build_rollout_result

        rewards, verifier_error, diagnostic = await self._verify(
            harness, {"reward": written}
        )
        assert rewards == {"reward": written}
        assert verifier_error is None
        note = f"reward outside 0-1: reward = {written!r}, recorded as test.sh wrote it"
        assert note in caplog.text
        assert "scored 0" not in caplog.text
        assert diagnostic is not None
        assert diagnostic.to_dict()["recorded_as_written"] is True

        diagnostics = RolloutDiagnostics()
        diagnostics.set(diagnostic)
        rollout_dir = tmp_path / "trial"
        rollout_dir.mkdir()
        _build_rollout_result(
            rollout_dir,
            task_name="t",
            rollout_name="t__1",
            agent="oracle",
            agent_name="oracle",
            model=None,
            n_tool_calls=0,
            prompts=[],
            error=None,
            verifier_error=verifier_error,
            trajectory=[],
            partial_trajectory=False,
            rewards=rewards,
            started_at=datetime.now(),
            timing={},
            diagnostics=diagnostics,
        )
        record = json.loads((rollout_dir / "result.json").read_text())
        assert record["rewards"] == {"reward": written}
        assert record["verifier_error"] is None
        info = record["reward_out_of_range_info"]
        assert (info["key"], info["value"], info["recorded_as_written"]) == (
            "reward",
            written,
            True,
        )

    @pytest.mark.asyncio
    async def test_a_number_the_reward_is_made_from_is_unscored_not_0(
        self, harness, caplog
    ):
        rewards, verifier_error, diagnostic = await self._verify(
            harness, {"speedup": 2.5}
        )
        assert rewards is None
        assert verifier_error is not None
        assert verifier_error.startswith("reward out of range: ")
        assert "scored 0" not in caplog.text
        assert diagnostic is not None

    @pytest.mark.usefixtures("strict_reward_rule")
    @pytest.mark.asyncio
    async def test_strict_is_unchanged(self, harness, caplog):
        rewards, verifier_error, diagnostic = await self._verify(
            harness, {"reward": 1.0079}
        )
        assert rewards == {"reward": 0.0}
        assert verifier_error is None
        assert "Verifier scored 0: reward out of range:" in caplog.text
        assert diagnostic is not None
        assert diagnostic.to_dict()["recorded_as_written"] is False


@pytest.mark.parametrize(
    ("integrity", "kept"), [("off", True), ("audit", True), ("strict", False)]
)
@pytest.mark.asyncio
async def test_rollout_verify_applies_the_trials_integrity_rule(
    monkeypatch, integrity, kept
):
    from benchflow.rewards.validation import keeps_out_of_range_reward
    from benchflow.rollout import Rollout

    seen: list[bool] = []

    async def score(self):
        seen.append(keeps_out_of_range_reward())
        return {"reward": 1.0}

    rollout = object.__new__(Rollout)
    rollout._config = SimpleNamespace(integrity=integrity)
    rollout._muse_credentials_staged = False
    monkeypatch.setattr(Rollout, "_score_finished_solver", score)
    monkeypatch.setattr(
        Rollout, "_stop_sandbox_proxies_before_verify", AsyncMock(), raising=False
    )
    monkeypatch.setattr(
        Rollout, "_snapshot_session_logs_before_verify", AsyncMock(), raising=False
    )
    monkeypatch.setattr("benchflow.rollout.require_safe_branch_world", lambda _r: None)
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.mark_solver_complete", lambda _r: None
    )
    assert await Rollout.verify(rollout) == {"reward": 1.0}
    assert seen == [kept]
    assert keeps_out_of_range_reward() is True


def test_the_job_summary_shows_the_note(tmp_path, caplog):
    from benchflow.evaluation import Evaluation
    from tests.test_resume_settings_match import _fake_runner, _first_job

    job_dir = _first_job(tmp_path, job_name="j")
    (trial,) = job_dir.glob("task-0__*")
    saved = json.loads((trial / "result.json").read_text())
    saved["rewards"] = {"reward": 1.0079}
    saved["reward_out_of_range_info"] = {
        "key": "reward",
        "value": 1.0079,
        "reward_range": [0.0, 1.0],
        "declared": False,
        "file": None,
        "recorded_as_written": True,
    }
    (trial / "result.json").write_text(json.dumps(saved))
    job = Evaluation.resume(job_dir)
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job))
    with caplog.at_level(logging.INFO):
        result = asyncio.run(job.run())
    assert result.total == 2
    summary = json.loads((job_dir / "summary.json").read_text())
    outside = summary["rewards_outside_range"]
    assert outside["count"] == 1
    (row,) = outside["trials"]
    assert (row["task"], row["reward"]) == ("task-0", 1.0079)
    assert row["note"] == (
        "task-0: reward outside 0-1: reward = 1.0079, recorded as test.sh wrote it"
    )
    assert "1 tasks had a reward outside 0-1" in " ".join(caplog.messages)


@pytest.mark.parametrize(
    ("integrity", "kept"), [("off", True), ("audit", True), ("strict", False)]
)
@pytest.mark.asyncio
async def test_soft_verify_applies_the_trials_integrity_rule(
    monkeypatch, integrity, kept
):
    """The in-turn check of a loop strategy: strict still refuses 1.0079."""
    from benchflow.rewards.validation import keeps_out_of_range_reward
    from benchflow.rollout import Rollout

    seen: list[bool] = []

    async def soft(self):
        seen.append(keeps_out_of_range_reward())
        return {"reward": 1.0}, None, None

    rollout = object.__new__(Rollout)
    rollout._config = SimpleNamespace(integrity=integrity)
    rollout._env = None
    monkeypatch.setattr(Rollout, "_soft_verify", soft)
    assert await rollout.soft_verify() == ({"reward": 1.0}, None, None)
    assert seen == [kept]
    assert keeps_out_of_range_reward() is True


async def _written_by_test_sh(tmp_path: Path, rewards: dict):
    """The rollout's verify of a test.sh that wrote ``rewards``."""
    from benchflow.sdk import SDK

    task = MagicMock()
    task.config.verifier.timeout_sec = 5
    task.config.verifier.env = None
    task.config.verifier.reward_range = None
    tp = MagicMock()
    tp.verifier_dir = tmp_path / "verifier"
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
    verifier = MagicMock()
    verifier.verify = AsyncMock(return_value=MagicMock(rewards=rewards))
    with (
        patch("benchflow.task.Verifier", return_value=verifier),
        patch(
            "benchflow.sandbox.lockdown.harden_before_verify", new_callable=AsyncMock
        ),
    ):
        return await SDK()._verify(env, task, tp, {})


@pytest.mark.parametrize(
    ("integrity", "reward", "noted"),
    [("strict", 0.0, False), ("audit", 1.0079, True), (None, 1.0079, True)],
)
@pytest.mark.asyncio
async def test_bench_regrade_applies_the_trials_recorded_integrity_rule(
    tmp_path, monkeypatch, integrity, reward, noted
):
    """A strict trial's regrade gives strict's 0; an audit (or off) trial's
    keeps 1.0079 with the note in its regrade record."""
    from benchflow import eval_regrade
    from tests.test_regrade import REGRADE_FILE, _task, _trial

    tasks = tmp_path / "tasks"
    _task(tasks, "t1", "#!/bin/bash\necho 1.0079 > /logs/verifier/reward.txt\n")
    job = tmp_path / "jobs" / "run1"
    trial = await _trial(job, "t1__a", "t1", 1.0, frozen=True, tmp=tmp_path)
    if integrity is not None:
        config = json.loads((trial / "config.json").read_text())
        (trial / "config.json").write_text(
            json.dumps({**config, "integrity": integrity})
        )

    async def frozen_trial_verify(trial, task_dir, attempt, *, sandbox=None):
        rewards, error, _ = await _written_by_test_sh(tmp_path, {"reward": 1.0079})
        return rewards, error

    # bench regrade's own runner is this rule around the frozen-trial verify.
    regrade_verify = eval_regrade.run_verifier_on_frozen_trial
    assert regrade_verify.__code__.co_name == "under_rule"
    runner = eval_regrade._under_recorded_reward_rule(frozen_trial_verify)
    summary = await eval_regrade.aregrade(job, tasks_dir=tasks, runner=runner)
    if not (trial / REGRADE_FILE).exists():
        # 0.9.x regrades no trial the separate verifier scored, and strict
        # implies it: refused before any sandbox, the strict score stands.
        [row] = summary.not_regradable
        assert integrity == "strict" and "separate verifier" in row.reason
        return
    [block] = json.loads((trial / REGRADE_FILE).read_text())["regrades"]
    assert block["status"] == "complete"
    assert block["new_rewards"] == {"reward": reward}
    info = block.get("reward_out_of_range_info")
    if noted:
        assert (info["value"], info["recorded_as_written"]) == (1.0079, True)
    else:
        assert info is None
