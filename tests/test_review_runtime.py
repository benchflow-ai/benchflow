"""Unit tests for the detached review runner (no sandbox involved)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchflow
from benchflow.review.config import REVIEW_RESULT_FILENAME
from benchflow.review.runner import (
    REVIEW_REPORT_FILENAME,
    ReviewRunError,
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
    if task_path is not None:
        config["task_path"] = str(task_path)
    (rollout / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if broken_result:
        (rollout / "result.json").write_text("{corrupt", encoding="utf-8")
    else:
        result = {
            "rewards": {"reward": reward} if reward is not None else None,
            "error": error,
        }
        (rollout / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return rollout


def make_task(root: Path, *, with_rubric: bool = False) -> Path:
    task = root / "source-task"
    (task / "verifier").mkdir(parents=True)
    (task / "task.md").write_text("---\n---\nbody", encoding="utf-8")
    if with_rubric:
        (task / "verifier" / "rubric.json").write_text(
            json.dumps(RUBRIC), encoding="utf-8"
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

        report, _ = await run_reviews(rollout, agent="gemini", out_dir=tmp_path / "out")

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

        report, _ = await run_reviews(rollout, agent="gemini", out_dir=tmp_path / "out")
        trial = report.trials[0]
        assert trial.review_valid is False
        assert trial.error == "reviewer output failed structural validation"
        assert trial.checks is not None  # verdicts still surfaced for triage

    @pytest.mark.asyncio
    async def test_missing_review_result_is_an_error_entry(self, tmp_path, monkeypatch):
        task = make_task(tmp_path, with_rubric=True)
        rollout = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
        monkeypatch.setattr(benchflow, "run", FakeRun(reward=0.0, review_payload=None))

        report, _ = await run_reviews(rollout, agent="gemini", out_dir=tmp_path / "out")
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
            tmp_path / "jobs", agent="gemini", out_dir=tmp_path / "out"
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
            rollout, agent="gemini", rubric_path=override, out_dir=tmp_path / "out"
        )
        assert seen["criteria"] == ["override_only"]

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
        await run_reviews(rollout, agent="gemini", out_dir=tmp_path / "out")
        assert seen["criteria"] == ["method_soundness"]

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
        await run_reviews(rollout, agent="gemini", out_dir=tmp_path / "out")
        assert seen["criteria"] == ["reward_hacking", "task_specification"]

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
                rollout, agent="gemini", rubric_path=bad, out_dir=tmp_path / "out"
            )
        assert called.configs == []  # no sandbox spend on a bad rubric


class TestRolloutConfigUploads:
    def test_uploads_default_empty(self, tmp_path):
        config = RolloutConfig(task_path=tmp_path)
        assert config.uploads == {}

    def test_uploads_accepts_mapping(self, tmp_path):
        config = RolloutConfig(task_path=tmp_path, uploads={str(tmp_path): "/app/data"})
        assert config.uploads == {str(tmp_path): "/app/data"}
