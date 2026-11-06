"""Guards FF-SW-OTHER-C1 at 6d9a8d2e: only returned evaluations own scores."""

import json
import os

import pytest

from benchflow.evaluation import EvaluationResult
from benchflow.models import RolloutResult
from benchflow.skill_eval import SkillEvaluator


def _evaluator(tmp_path):
    skill_dir = tmp_path / "dummy"
    (skill_dir / "evals").mkdir(parents=True)
    (skill_dir / "evals" / "evals.json").write_text(
        json.dumps({"cases": [{"id": "case-001", "question": "published question"}]})
    )
    return SkillEvaluator(skill_dir)


def _write_evidence(rollout_dir, reward, marker):
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "result.json").write_text(
        json.dumps({"rewards": {"reward": reward}, "n_tool_calls": 99})
    )
    (rollout_dir / "prompts.json").write_text(json.dumps({"prompt": marker}))
    (rollout_dir / "trajectory").mkdir(exist_ok=True)
    (rollout_dir / "trajectory" / "acp_trajectory.jsonl").write_text(
        json.dumps({"marker": marker}) + "\n"
    )
    (rollout_dir / "verifier").mkdir(exist_ok=True)
    (rollout_dir / "verifier" / "judge_result.json").write_text(
        json.dumps({"items": [{"criterion": marker, "score": reward}]})
    )


async def _collect(evaluator, tmp_path, backend, with_skill, model="model-a"):
    return await evaluator._run_job(
        tasks_dir=tmp_path / "tasks",
        agent="codex-acp",
        model=model,
        environment=backend,
        jobs_dir=str(tmp_path / "jobs"),
        concurrency=1,
        with_skill=with_skill,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona", "modal"])
@pytest.mark.parametrize("with_skill", [True, False])
@pytest.mark.parametrize(
    "plant",
    [
        "current/case-001__current/artifacts/case-001",
        "current/case-001__current/artifacts/case-001__forged",
        "current/case-001__current/artifacts/nested/case-001",
        "current/case-001__current/agent/case-001",
        "current/case-001__retry",
        "stale/case-001__current",
        "case-001",
        "case-001__linked",
    ],
)
async def test_only_returned_result_can_supply_a_score(
    tmp_path, monkeypatch, backend, with_skill, plant
):
    """Guards FF-SW-OTHER-C1: names, nesting and future mtimes confer no authority."""
    evaluator = _evaluator(tmp_path)
    job_dir = tmp_path / "jobs" / "current"
    rollout_name = "case-001__current"

    async def fake_run(self):
        # Even the genuine on-disk score is not the returned score. Traces
        # belong to this exact job/rollout, rather than the newest candidate.
        _write_evidence(job_dir / rollout_name, 1.0, "current trace")
        planted = tmp_path / "jobs" / plant
        if plant == "case-001__linked":
            unrelated = tmp_path / "unrelated"
            _write_evidence(unrelated, 1.0, "planted trace")
            planted.symlink_to(unrelated, target_is_directory=True)
        else:
            _write_evidence(planted, 1.0, "planted trace")
        os.utime(planted, (4070908800, 4070908800))
        return EvaluationResult(
            job_name="current",
            config=self._config,
            job_dir=job_dir,
            results={
                "case-001": RolloutResult(
                    "case-001", rollout_name, rewards={"reward": 0.0}, n_tool_calls=4
                )
            },
        )

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_run)
    (result,) = await _collect(evaluator, tmp_path, backend, with_skill)
    assert result.reward == 0.0
    assert result.error is None
    assert result.n_tool_calls == 4
    assert result.prompt == "current trace"
    assert result.trajectory == [{"marker": "current trace"}]
    assert result.rubric_results == [{"criterion": "current trace", "score": 1.0}]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona", "modal"])
@pytest.mark.parametrize("trial", range(3))
async def test_models_and_modes_never_reuse_another_evaluations_results(
    tmp_path, monkeypatch, backend, trial
):
    """Guards FF-SW-OTHER-C1 through collection and aggregation, with nop/oracle."""
    evaluator = _evaluator(tmp_path)
    sequence = 0

    async def fake_run(self):
        nonlocal sequence
        sequence += 1
        job_dir = self._jobs_dir / f"job-{trial}-{sequence}"
        rollout_name = "case-001__current"
        marker = f"{self._config.model}/{self._config.skill_mode}"
        reward = float(
            self._config.model == "oracle" and self._config.skill_mode == "with-skill"
        )
        _write_evidence(job_dir / rollout_name, reward, marker)
        # The first model leaves a candidate that wins every later mtime scan.
        planted = job_dir / rollout_name / "artifacts" / "case-001"
        _write_evidence(planted, 1.0, "first model forgery")
        os.utime(planted, (4070908800 + sequence, 4070908800 + sequence))
        return EvaluationResult(
            job_name=job_dir.name,
            config=self._config,
            job_dir=job_dir,
            results={
                "case-001": RolloutResult(
                    "case-001", rollout_name, rewards={"reward": reward}
                )
            },
        )

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_run)
    rows = []
    for model in ("oracle", "nop"):
        for mode in (True, False):
            (row,) = await _collect(evaluator, tmp_path, backend, mode, model)
            rows.append(row)
            assert row.reward == float(model == "oracle" and mode)
            assert row.prompt == f"{model}/{'with-skill' if mode else 'no-skill'}"
    lifts = evaluator._compute_lifts(rows, ["codex-acp"] * 2, ["oracle", "nop"], False)
    assert [lift.with_skill_score for lift in lifts] == [1.0, 0.0]
    assert [lift.baseline_score for lift in lifts] == [0.0, 0.0]
    assert [lift.lift for lift in lifts] == [1.0, 0.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["result", "job_dir", "rollout_name", "files"])
async def test_missing_authoritative_data_never_falls_back_to_discovery(
    tmp_path, monkeypatch, missing
):
    """Guards FF-SW-OTHER-C1 for early failure and older artifact-free results."""
    evaluator = _evaluator(tmp_path)

    async def fake_run(self):
        _write_evidence(self._jobs_dir / "case-001", 1.0, "unrelated")
        return EvaluationResult(
            job_name="current",
            config=self._config,
            job_dir=None if missing == "job_dir" else self._jobs_dir / "current",
            results={}
            if missing == "result"
            else {
                "case-001": RolloutResult(
                    "case-001",
                    "" if missing == "rollout_name" else "case-001__current",
                    rewards={"reward": 0.25},
                    n_tool_calls=7,
                )
            },
        )

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_run)
    (row,) = await _collect(evaluator, tmp_path, "docker", True)
    assert row.reward == (None if missing == "result" else 0.25)
    assert row.n_tool_calls == (0 if missing == "result" else 7)
    assert bool(row.error) == (missing == "result")
    assert row.prompt == "published question"
    assert row.trajectory is None
    assert row.rubric_results is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error_field", ["error", "verifier_error"])
async def test_errors_and_unscored_results_come_from_the_evaluation(
    tmp_path, monkeypatch, error_field
):
    """Guards FF-SW-OTHER-C1: planted success cannot replace a returned failure."""
    evaluator = _evaluator(tmp_path)

    async def fake_run(self):
        _write_evidence(self._jobs_dir / "case-001", 1.0, "forged success")
        return EvaluationResult(
            job_name="current",
            config=self._config,
            results={
                "case-001": RolloutResult(
                    "case-001", **{error_field: "trusted failure"}, n_tool_calls=3
                )
            },
        )

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_run)
    (row,) = await _collect(evaluator, tmp_path, "docker", False)
    assert row.reward is None
    assert row.error == "trusted failure"
    assert row.n_tool_calls == 3
