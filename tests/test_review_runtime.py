"""Unit tests for the detached review runner (no sandbox involved)."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

import benchflow
from benchflow.review.config import REVIEW_RESULT_FILENAME
from benchflow.review.runner import (
    REVIEW_REPORT_FILENAME,
    ReviewRunError,
    TrialReview,
    _lock_review_evidence,
    _task_digest_issue,
    discover_rollouts,
    run_reviews,
)
from benchflow.rollout import RolloutConfig

RUBRIC = {
    "criteria": [
        {
            "name": "method_soundness",
            "description": "d",
            "guidance": "PASS when sound; FAIL otherwise.",
        }
    ]
}

WEIGHTED_RUBRIC = {
    "criteria": [
        {
            "name": "safety_gate",
            "blocker": 1,
            "weight": 10,
            "description": "d",
            "guidance": "PASS when safe; FAIL otherwise.",
        },
        {
            "name": "method_quality",
            "blocker": 0,
            "weight": 3,
            "description": "d",
            "guidance": "Score 2, 1, or 0.",
        },
        {
            "name": "evidence_quality",
            "blocker": 0,
            "weight": 2,
            "description": "d",
            "guidance": "Score 2, 1, or 0.",
        },
    ]
}


def make_rollout(
    root: Path,
    name: str,
    *,
    reward: float | None = 1.0,
    error: str | None = None,
    task_path: Path | None = None,
    broken_result: bool = False,
) -> Path:
    rollout = root / name
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "trajectory" / "trajectory.json").write_text("[]", encoding="utf-8")
    config: dict = {}
    digest: str | None = None
    if task_path is not None:
        from benchflow._utils.task_authoring import task_digest

        digest = task_digest(task_path)
        config["task_path"] = str(task_path)
        config["task_digest"] = digest
    (rollout / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if broken_result:
        (rollout / "result.json").write_text("{corrupt", encoding="utf-8")
    else:
        result = {
            "rewards": {"reward": reward} if reward is not None else None,
            "error": error,
            "task_digest": digest,
        }
        (rollout / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return rollout


def make_task(
    root: Path,
    *,
    with_rubric: bool = False,
    rubric_data: dict | None = None,
) -> Path:
    """Create a task inside a *trusted tasks root* (see ``--tasks-root``).

    Task evidence is only included when the caller names a trusted root: a
    rollout-recorded ``task_path`` is untrusted input and is never
    dereferenced (guards the P0 fixed in PR #942).
    """
    task = root / "tasks" / "source-task"
    (task / "verifier").mkdir(parents=True)
    (task / "task.md").write_text("---\n---\nbody", encoding="utf-8")
    if with_rubric:
        (task / "verifier" / "rubric.json").write_text(
            json.dumps(RUBRIC if rubric_data is None else rubric_data),
            encoding="utf-8",
        )
    return task


class FakeRun:
    """Stands in for ``benchflow.run``: records configs, fabricates results."""

    def __init__(
        self,
        *,
        reward: float = 1.0,
        review_payload: dict | str | None = None,
        error: str | None = None,
    ):
        self.reward = reward
        self.review_payload = review_payload
        self.error = error
        self.configs: list[RolloutConfig] = []

    async def __call__(self, config: RolloutConfig):
        self.configs.append(config)
        self.task_docs = getattr(self, "task_docs", [])
        self.task_docs.append(
            (config.task_path / "task.md").read_text(encoding="utf-8")
        )
        runtime = Path(config.jobs_dir) / "job" / "wrapper__0000"
        (runtime / "verifier").mkdir(parents=True)
        # A rollout leaf is identified by its config.json, exactly as the
        # real lifecycle writes one.
        (runtime / "config.json").write_text("{}", encoding="utf-8")
        (runtime / "result.json").write_text(
            json.dumps({"rewards": {"reward": self.reward}, "error": self.error}),
            encoding="utf-8",
        )
        if self.review_payload is not None:
            payload = (
                self.review_payload
                if isinstance(self.review_payload, str)
                else json.dumps(self.review_payload)
            )
            (runtime / "verifier" / REVIEW_RESULT_FILENAME).write_text(
                payload, encoding="utf-8"
            )

        class _Result:
            error = self.error

        return _Result()


def good_review(name: str = "rollout-a") -> dict:
    return {
        "trial_name": name,
        "summary": "Reviewed fine.",
        "checks": {"method_soundness": {"explanation": "ok", "outcome": "pass"}},
    }


def good_weighted_review(name: str = "rollout-a") -> dict:
    return {
        "trial_name": name,
        "summary": "Weighted review complete.",
        "checks": {
            "safety_gate": {"explanation": "safe", "outcome": "pass"},
            "method_quality": {"explanation": "complete", "score": 2},
            "evidence_quality": {"explanation": "partial", "score": 1},
        },
    }


class TestDiscovery:
    def test_single_rollout_dir(self, tmp_path):
        rollout = make_rollout(tmp_path, "one")
        assert discover_rollouts(rollout) == [rollout]

    def test_job_dir(self, tmp_path):
        a = make_rollout(tmp_path, "a")
        b = make_rollout(tmp_path, "b")
        assert discover_rollouts(tmp_path) == [a, b]

    def test_passing_and_failing_filters(self, tmp_path):
        passing = make_rollout(tmp_path, "pass", reward=1.0)
        failing = make_rollout(tmp_path, "fail", reward=0.0)
        broken = make_rollout(tmp_path, "broken", broken_result=True)
        errored = make_rollout(tmp_path, "errored", reward=1.0, error="agent died")
        assert discover_rollouts(tmp_path, filter_passing=True) == [passing]
        assert discover_rollouts(tmp_path, filter_passing=False) == [
            broken,
            errored,
            failing,
        ]

    def test_missing_path_errors(self, tmp_path):
        with pytest.raises(ReviewRunError, match="does not exist"):
            discover_rollouts(tmp_path / "nope")

    def test_empty_dir_errors(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ReviewRunError, match="neither a rollout"):
            discover_rollouts(tmp_path / "empty")

    def test_all_filtered_out_errors(self, tmp_path):
        make_rollout(tmp_path, "fail", reward=0.0)
        with pytest.raises(ReviewRunError, match="no passing"):
            discover_rollouts(tmp_path, filter_passing=True)


class TestRunReviews:
    @pytest.mark.asyncio
    async def test_reviews_a_job_dir(self, tmp_path, monkeypatch):
        task = make_task(tmp_path, with_rubric=True)
        make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
        make_rollout(tmp_path / "jobs", "rollout-b", task_path=task)
        fake = FakeRun(review_payload=good_review())
        monkeypatch.setattr(benchflow, "run", fake)

        _report, report_path = await run_reviews(
            tmp_path / "jobs",
            agent="gemini",
            model="gemini/test-model",
            environment="docker",
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )

        assert report_path.name == REVIEW_REPORT_FILENAME
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert [t["trial_name"] for t in data["trials"]] == ["rollout-a", "rollout-b"]
        assert all(t["review_valid"] for t in data["trials"])
        assert all(
            t["checks"]["method_soundness"]["outcome"] == "pass" for t in data["trials"]
        )
        assert len(fake.configs) == 2

    @pytest.mark.asyncio
    async def test_weighted_review_is_scored_and_recorded(self, tmp_path, monkeypatch):
        task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
        rollout = make_rollout(
            tmp_path / "jobs", "rollout-a", reward=1.0, task_path=task
        )
        source_result_before = (rollout / "result.json").read_bytes()
        monkeypatch.setattr(
            benchflow,
            "run",
            FakeRun(review_payload=good_weighted_review()),
        )

        report, report_path = await run_reviews(
            rollout,
            agent="gemini",
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )

        trial = report.trials[0]
        assert trial.review_valid is True
        assert trial.rubric_contract == "v0.2"
        assert trial.criterion_metadata == [
            {"name": "safety_gate", "blocker": 1, "weight": 10},
            {"name": "method_quality", "blocker": 0, "weight": 3},
            {"name": "evidence_quality", "blocker": 0, "weight": 2},
        ]
        assert trial.checks["method_quality"]["score"] == 2
        assert trial.scoring is not None
        assert trial.scoring.weighted_points == 8
        assert trial.scoring.max_weighted_points == 10
        assert trial.scoring.raw_quality == pytest.approx(0.8)
        assert trial.scoring.gated_quality == pytest.approx(0.8)
        assert trial.scoring.decision == "publishable"
        assert (rollout / "result.json").read_bytes() == source_result_before

        serialized = json.loads(report_path.read_text(encoding="utf-8"))
        serialized_trial = serialized["trials"][0]
        assert serialized["rubric"]["contracts"] == ["v0.2"]
        assert serialized_trial["rubric_contract"] == "v0.2"
        assert serialized_trial["criterion_metadata"] == trial.criterion_metadata
        assert serialized_trial["scoring"] == {
            "deterministic_pass": True,
            "all_blockers_pass": True,
            "failed_blockers": [],
            "weighted_points": 8,
            "max_weighted_points": 10,
            "raw_quality": 0.8,
            "gated_quality": 0.8,
            "decision": "publishable",
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("reward", "error"),
        [(0.0, None), (1.0, "source agent failed")],
    )
    async def test_source_deterministic_failure_gates_weighted_quality(
        self, tmp_path, monkeypatch, reward, error
    ):
        task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
        rollout = make_rollout(
            tmp_path / "jobs",
            "rollout-a",
            reward=reward,
            error=error,
            task_path=task,
        )
        monkeypatch.setattr(
            benchflow,
            "run",
            FakeRun(review_payload=good_weighted_review()),
        )

        report, _ = await run_reviews(
            rollout,
            agent="gemini",
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )

        scoring = report.trials[0].scoring
        assert scoring is not None
        assert scoring.deterministic_pass is False
        assert scoring.raw_quality == pytest.approx(0.8)
        assert scoring.gated_quality == 0.0
        assert scoring.decision == "not_publishable"

    @pytest.mark.asyncio
    async def test_failed_blocker_gates_weighted_quality(self, tmp_path, monkeypatch):
        task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
        rollout = make_rollout(
            tmp_path / "jobs", "rollout-a", reward=1.0, task_path=task
        )
        payload = good_weighted_review()
        payload["checks"]["safety_gate"]["outcome"] = "fail"
        monkeypatch.setattr(benchflow, "run", FakeRun(review_payload=payload))

        report, _ = await run_reviews(
            rollout,
            agent="gemini",
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )

        scoring = report.trials[0].scoring
        assert scoring is not None
        assert scoring.all_blockers_pass is False
        assert scoring.failed_blockers == ("safety_gate",)
        assert scoring.raw_quality == pytest.approx(0.8)
        assert scoring.gated_quality == 0.0
        assert scoring.decision == "not_publishable"

    @pytest.mark.asyncio
    async def test_invalid_weighted_review_never_aggregates(
        self, tmp_path, monkeypatch
    ):
        task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
        rollout = make_rollout(
            tmp_path / "jobs", "rollout-a", reward=1.0, task_path=task
        )
        monkeypatch.setattr(
            benchflow,
            "run",
            FakeRun(reward=0.0, review_payload=good_weighted_review()),
        )

        report, _ = await run_reviews(
            rollout,
            agent="gemini",
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )

        trial = report.trials[0]
        assert trial.review_valid is False
        assert trial.scoring is None

    @pytest.mark.asyncio
    async def test_host_validation_gates_weighted_scoring(self, tmp_path, monkeypatch):
        """Guards FrontierPhysics PR #109 when wrapper reward and artifact diverge."""

        task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
        rollout = make_rollout(
            tmp_path / "jobs", "rollout-a", reward=1.0, task_path=task
        )
        payload = good_weighted_review()
        payload["checks"]["method_quality"]["unexpected"] = "must fail closed"
        monkeypatch.setattr(
            benchflow,
            "run",
            FakeRun(reward=1.0, review_payload=payload),
        )

        report, _ = await run_reviews(
            rollout,
            agent="gemini",
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )

        trial = report.trials[0]
        assert trial.review_valid is False
        assert trial.scoring is None
        assert "host-side structural validation" in (trial.error or "")

    @pytest.mark.asyncio
    async def test_wrapper_config_shape(self, tmp_path, monkeypatch):
        """The reviewer runs as a normal rollout: wrapper task, prebuilt
        image, evidence uploads, caller-selected backend."""
        task = make_task(tmp_path, with_rubric=True)
        rollout = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
        fake = FakeRun(review_payload=good_review())
        monkeypatch.setattr(benchflow, "run", fake)

        await run_reviews(
            rollout,
            agent="gemini",
            model="gemini/test-model",
            environment="daytona",
            agent_env={"X": "1"},
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )

        config = fake.configs[0]
        assert config.agent == "gemini"
        assert config.model == "gemini/test-model"
        assert config.environment == "daytona"
        assert config.agent_env == {"X": "1"}
        # The wrapper task itself declares no-internet, engaging the no-web
        # pipeline (web policy, sandbox-local proxy, egress firewall).
        assert "allow_internet: false" in fake.task_docs[0]
        assert set(config.uploads.values()) == {"/evidence/trial", "/evidence/task"}
        assert config.pre_agent_hooks == [_lock_review_evidence]
        # The wrapper was assembled with no Dockerfile (prebuilt image only).
        # It is deleted after the run, so assert via the recorded task path
        # name rather than the filesystem.
        assert config.task_path.name.startswith("review-")

    @pytest.mark.asyncio
    async def test_rewards_of_reviewed_rollouts_are_never_modified(
        self, tmp_path, monkeypatch
    ):
        """Guards PR #942's contract: review is report-only. The reviewed
        rollout's result.json must be byte-identical after a review runs."""
        task = make_task(tmp_path, with_rubric=True)
        rollout = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
        before = (rollout / "result.json").read_bytes()
        monkeypatch.setattr(benchflow, "run", FakeRun(review_payload=good_review()))

        report, _ = await run_reviews(
            rollout,
            agent="gemini",
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )

        assert (rollout / "result.json").read_bytes() == before
        assert not (rollout / REVIEW_RESULT_FILENAME).exists()
        trial = report.trials[0]
        assert trial.checks is not None
        assert "plan" not in json.loads(
            (rollout / "result.json").read_text(encoding="utf-8")
        ).get("rewards", {})

    @pytest.mark.asyncio
    async def test_invalid_reviewer_output_is_flagged(self, tmp_path, monkeypatch):
        task = make_task(tmp_path, with_rubric=True)
        rollout = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
        # reward 0 = wrapper's structural validation failed
        monkeypatch.setattr(
            benchflow,
            "run",
            FakeRun(reward=0.0, review_payload=good_review()),
        )

        report, _ = await run_reviews(
            rollout,
            agent="gemini",
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )
        trial = report.trials[0]
        assert trial.review_valid is False
        assert trial.error == "reviewer output failed structural validation"
        assert trial.checks is not None  # verdicts still surfaced for triage

    @pytest.mark.asyncio
    async def test_missing_review_result_is_an_error_entry(self, tmp_path, monkeypatch):
        task = make_task(tmp_path, with_rubric=True)
        rollout = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
        monkeypatch.setattr(benchflow, "run", FakeRun(reward=0.0, review_payload=None))

        report, _ = await run_reviews(
            rollout,
            agent="gemini",
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )
        trial = report.trials[0]
        assert trial.review_valid is False
        assert "did not produce" in (trial.error or "")

    @pytest.mark.asyncio
    async def test_reviewer_crash_isolates_to_one_trial(self, tmp_path, monkeypatch):
        task = make_task(tmp_path, with_rubric=True)
        make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
        make_rollout(tmp_path / "jobs", "rollout-b", task_path=task)
        good = FakeRun(review_payload=good_review())
        calls = {"n": 0}

        async def flaky(config: RolloutConfig):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("sandbox exploded")
            return await good(config)

        monkeypatch.setattr(benchflow, "run", flaky)

        report, _ = await run_reviews(
            tmp_path / "jobs",
            agent="gemini",
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )
        errors = [t for t in report.trials if t.error]
        assert len(errors) == 1
        assert "sandbox exploded" in errors[0].error
        assert len([t for t in report.trials if t.checks]) == 1

    @pytest.mark.asyncio
    async def test_explicit_rubric_beats_task_rubric(self, tmp_path, monkeypatch):
        task = make_task(tmp_path, with_rubric=True)
        rollout = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
        override = tmp_path / "override.json"
        override.write_text(
            json.dumps(
                {
                    "criteria": [
                        {
                            "name": "override_only",
                            "description": "d",
                            "guidance": "PASS always.",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        seen: dict = {}

        async def capture(config: RolloutConfig):
            seen["criteria"] = json.loads(
                (config.task_path / "tests" / "criteria.json").read_text("utf-8")
            )
            return await FakeRun(review_payload=good_review())(config)

        monkeypatch.setattr(benchflow, "run", capture)
        await run_reviews(
            rollout,
            agent="gemini",
            rubric_path=override,
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )
        assert seen["criteria"] == {
            "contract": "v0.1",
            "criteria": [{"name": "override_only", "blocker": None, "weight": None}],
        }

    @pytest.mark.asyncio
    async def test_task_rubric_used_when_no_override(self, tmp_path, monkeypatch):
        task = make_task(tmp_path, with_rubric=True)
        rollout = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
        seen: dict = {}

        async def capture(config: RolloutConfig):
            seen["criteria"] = json.loads(
                (config.task_path / "tests" / "criteria.json").read_text("utf-8")
            )
            return await FakeRun(review_payload=good_review())(config)

        monkeypatch.setattr(benchflow, "run", capture)
        await run_reviews(
            rollout,
            agent="gemini",
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )
        assert seen["criteria"] == {
            "contract": "v0.1",
            "criteria": [{"name": "method_soundness", "blocker": None, "weight": None}],
        }

    @pytest.mark.asyncio
    async def test_default_rubric_when_task_ships_none(self, tmp_path, monkeypatch):
        task = make_task(tmp_path, with_rubric=False)
        rollout = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
        seen: dict = {}

        async def capture(config: RolloutConfig):
            seen["criteria"] = json.loads(
                (config.task_path / "tests" / "criteria.json").read_text("utf-8")
            )
            return await FakeRun(review_payload=good_review())(config)

        monkeypatch.setattr(benchflow, "run", capture)
        await run_reviews(
            rollout,
            agent="gemini",
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )
        assert seen["criteria"] == {
            "contract": "v0.1",
            "criteria": [
                {"name": "reward_hacking", "blocker": None, "weight": None},
                {"name": "task_specification", "blocker": None, "weight": None},
            ],
        }

    @pytest.mark.asyncio
    async def test_bad_explicit_rubric_fails_fast(self, tmp_path, monkeypatch):
        rollout = make_rollout(tmp_path / "jobs", "rollout-a")
        bad = tmp_path / "bad.json"
        bad.write_text("{corrupt", encoding="utf-8")
        called = FakeRun(review_payload=good_review())
        monkeypatch.setattr(benchflow, "run", called)
        from benchflow.review.config import ReviewRubricError

        with pytest.raises(ReviewRubricError):
            await run_reviews(
                rollout,
                agent="gemini",
                rubric_path=bad,
                out_dir=tmp_path / "out",
                tasks_root=tmp_path / "tasks",
            )
        assert called.configs == []  # no sandbox spend on a bad rubric

    @pytest.mark.asyncio
    async def test_explicit_rubric_is_one_atomic_snapshot(self, tmp_path, monkeypatch):
        """Guards PR #942: one invocation cannot mix rubric file revisions."""

        make_rollout(tmp_path / "jobs", "rollout-a")
        make_rollout(tmp_path / "jobs", "rollout-b")
        rubric_path = tmp_path / "rubric.json"
        rubric_path.write_text(json.dumps(RUBRIC), encoding="utf-8")
        seen: list[dict] = []

        async def mutate_after_first_wrapper(config: RolloutConfig):
            criteria = json.loads(
                (config.task_path / "tests" / "criteria.json").read_text("utf-8")
            )
            seen.append(criteria)
            rubric_path.write_text(
                json.dumps(
                    {
                        "criteria": [
                            {
                                "name": "changed_mid_run",
                                "description": "d",
                                "guidance": "g",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return await FakeRun(review_payload=good_review())(config)

        monkeypatch.setattr(benchflow, "run", mutate_after_first_wrapper)
        report, _ = await run_reviews(
            tmp_path / "jobs",
            agent="gemini",
            rubric_path=rubric_path,
            concurrency=1,
            out_dir=tmp_path / "out",
        )

        expected_metadata = {
            "contract": "v0.1",
            "criteria": [{"name": "method_soundness", "blocker": None, "weight": None}],
        }
        assert seen == [expected_metadata, expected_metadata]
        assert report.criteria == ["method_soundness"]
        assert all(t.criteria == ["method_soundness"] for t in report.trials)

    @pytest.mark.asyncio
    async def test_report_criteria_union_for_per_task_rubrics(
        self, tmp_path, monkeypatch
    ):
        """Guards PR #942: heterogeneous jobs report every trial criterion."""

        tasks_root = tmp_path / "tasks"
        for task_name, criterion in (("task-a", "alpha"), ("task-b", "beta")):
            task = tasks_root / task_name
            (task / "verifier").mkdir(parents=True)
            (task / "task.md").write_text("---\n---\nbody", encoding="utf-8")
            (task / "verifier" / "rubric.json").write_text(
                json.dumps(
                    {
                        "criteria": [
                            {
                                "name": criterion,
                                "description": "d",
                                "guidance": "g",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            make_rollout(tmp_path / "jobs", f"rollout-{criterion}", task_path=task)

        monkeypatch.setattr(benchflow, "run", FakeRun(review_payload=good_review()))
        report, _ = await run_reviews(
            tmp_path / "jobs",
            agent="gemini",
            tasks_root=tasks_root,
            out_dir=tmp_path / "out",
        )
        assert report.criteria == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_job_summary_uses_only_valid_reviews(self, tmp_path, monkeypatch):
        """Guards PR #942: rejected reviewer output cannot move aggregates."""

        make_rollout(tmp_path / "jobs", "a")
        make_rollout(tmp_path / "jobs", "b")
        calls = {"count": 0}

        async def mixed(config: RolloutConfig):
            calls["count"] += 1
            return await FakeRun(
                reward=1.0 if calls["count"] == 1 else 0.0,
                review_payload=good_review(),
            )(config)

        monkeypatch.setattr(benchflow, "run", mixed)
        report, _ = await run_reviews(
            tmp_path / "jobs", agent="gemini", out_dir=tmp_path / "out"
        )
        assert report.job_summary is None

    @pytest.mark.asyncio
    async def test_weighted_job_summary_labels_binary_counts(
        self, tmp_path, monkeypatch
    ):
        """Guards FrontierPhysics PR #109 against omitting scored criteria silently."""

        task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
        make_rollout(tmp_path / "jobs", "a", task_path=task)
        make_rollout(tmp_path / "jobs", "b", task_path=task)

        async def weighted_run(config: RolloutConfig):
            trial_name = Path(config.task_path).name.removeprefix("review-")
            return await FakeRun(review_payload=good_weighted_review(trial_name))(
                config
            )

        monkeypatch.setattr(benchflow, "run", weighted_run)
        report, _ = await run_reviews(
            tmp_path / "jobs",
            agent="gemini",
            out_dir=tmp_path / "out",
            tasks_root=tmp_path / "tasks",
        )

        assert report.job_summary is not None
        assert "Binary judgments (legacy criteria and weighted blockers only)" in (
            report.job_summary
        )
        assert "across all criteria" not in report.job_summary
        assert "weighted reviews: average raw quality 0.800" in report.job_summary


class TestTaskDigestAdmission:
    def test_config_only_digest_is_enforced(self, tmp_path):
        """Guards PR #942: config provenance is authoritative when result omits it."""

        task = make_task(tmp_path)
        rollout = make_rollout(tmp_path / "jobs", "run", task_path=task)
        result = json.loads((rollout / "result.json").read_text(encoding="utf-8"))
        result.pop("task_digest")
        (rollout / "result.json").write_text(json.dumps(result), encoding="utf-8")
        assert _task_digest_issue(rollout, task) is None

    def test_missing_digest_excludes_task(self, tmp_path):
        """Guards PR #942: unverifiable task evidence fails closed."""

        task = make_task(tmp_path)
        rollout = make_rollout(tmp_path / "jobs", "run", task_path=task)
        for filename in ("result.json", "config.json"):
            data = json.loads((rollout / filename).read_text(encoding="utf-8"))
            data.pop("task_digest", None)
            (rollout / filename).write_text(json.dumps(data), encoding="utf-8")
        assert "missing" in (_task_digest_issue(rollout, task) or "")

    def test_conflicting_digests_exclude_task(self, tmp_path):
        """Guards PR #942: conflicting provenance cannot select task evidence."""

        task = make_task(tmp_path)
        rollout = make_rollout(tmp_path / "jobs", "run", task_path=task)
        result = json.loads((rollout / "result.json").read_text(encoding="utf-8"))
        result["task_digest"] = "sha256:" + "0" * 64
        (rollout / "result.json").write_text(json.dumps(result), encoding="utf-8")
        assert "conflicting" in (_task_digest_issue(rollout, task) or "")

    def test_digest_computation_failure_excludes_task(self, tmp_path, monkeypatch):
        """Guards PR #942: digest errors cannot admit unverified task files."""

        task = make_task(tmp_path)
        rollout = make_rollout(tmp_path / "jobs", "run", task_path=task)

        def fail(_path):
            raise OSError("cannot hash")

        monkeypatch.setattr("benchflow._utils.task_authoring.task_digest", fail)
        assert "could not be verified" in (_task_digest_issue(rollout, task) or "")


@pytest.mark.asyncio
async def test_review_evidence_lock_is_fail_closed():
    """Guards PR #942: the agent starts only after read-only evidence locking."""

    class Sandbox:
        async def exec(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs

            class Result:
                return_code = 0
                stdout = ""
                stderr = ""

            return Result()

    sandbox = Sandbox()
    await _lock_review_evidence(sandbox)
    assert "chown -R 0:0 /evidence" in sandbox.command
    assert "chmod -R a-w,a+rX /evidence" in sandbox.command
    assert sandbox.kwargs["user"] == "root"


def test_cli_rendering_escapes_untrusted_review_fields(monkeypatch):
    """Guards PR #942: rollout data cannot crash or inject Rich markup."""

    from rich.console import Console

    from benchflow.cli import review as review_cli

    output = StringIO()
    monkeypatch.setattr(
        review_cli,
        "console",
        Console(file=output, force_terminal=True, color_system=None),
    )
    trial = TrialReview(
        trial_name="[/bold]",
        source_rollout="diagnostic",
        checks={
            "[/red]": {"outcome": "[/green]", "explanation": "[x]"},
            "hostile_outcome": {"outcome": {}, "explanation": "diagnostic"},
            "hostile_score": {"score": [], "explanation": "diagnostic"},
        },
        summary="[/bold]",
        error="[/red]",
        review_valid=False,
    )
    review_cli._render_trial_review(trial)
    review_cli._render_review_overview([trial])
    assert "[/bold]" in output.getvalue()


class TestRolloutConfigUploads:
    def test_uploads_default_empty(self, tmp_path):
        config = RolloutConfig(task_path=tmp_path)
        assert config.uploads == {}

    def test_uploads_accepts_mapping(self, tmp_path):
        config = RolloutConfig(task_path=tmp_path, uploads={str(tmp_path): "/app/data"})
        assert config.uploads == {str(tmp_path): "/app/data"}
