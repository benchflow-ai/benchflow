"""Tests for the rubric-review plane: rubric.json loading, verdict parsing,
and score aggregation (``benchflow.review``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.review import (
    ReviewParams,
    ReviewRubricError,
    aggregate,
    extract_verdicts_object,
    find_review_rubric,
    is_review_rubric_file,
    load_review_rubric,
    parse_reviewer_message,
)
from benchflow.review.config import ReviewCriterion, ReviewerSpec, ReviewRubric
from benchflow.review.scoring import (
    STATUS_SCORED,
    STATUS_UNSCORED,
    CriterionVerdict,
)


def _write_rubric(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps(payload))
    return path


def _minimal(**overrides) -> dict:
    payload = {
        "schema_version": "1.0",
        "criteria": [
            {"id": "correct", "criterion": "The result file contains the right value."}
        ],
    }
    payload.update(overrides)
    return payload


# Loading and validation


class TestLoadReviewRubric:
    def test_minimal_rubric_loads_with_defaults(self, tmp_path: Path) -> None:
        rubric = load_review_rubric(_write_rubric(tmp_path, _minimal()))
        assert len(rubric.criteria) == 1
        criterion = rubric.criteria[0]
        assert criterion.choices == ("no", "yes")
        assert criterion.weight == 1.0
        assert not criterion.required
        assert rubric.pass_threshold == 0.7
        assert rubric.reviewer.mode == "batched"

    def test_missing_schema_version_is_rejected_as_llm_judge_rubric(
        self, tmp_path: Path
    ) -> None:
        # A Harvey-LAB llm-judge rubric.json has no schema_version; the review
        # loader must refuse it with a message that names the discriminator.
        path = _write_rubric(
            tmp_path, {"title": "t", "criteria": [{"id": "c-1", "title": "x"}]}
        )
        with pytest.raises(ReviewRubricError, match="schema_version"):
            load_review_rubric(path)

    def test_unknown_top_level_key_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ReviewRubricError, match="unknown key"):
            load_review_rubric(_write_rubric(tmp_path, _minimal(extra=1)))

    def test_unknown_criterion_key_rejected(self, tmp_path: Path) -> None:
        payload = _minimal()
        payload["criteria"][0]["weigth"] = 2  # the classic typo
        with pytest.raises(ReviewRubricError, match="weigth"):
            load_review_rubric(_write_rubric(tmp_path, payload))

    def test_duplicate_ids_rejected(self, tmp_path: Path) -> None:
        payload = _minimal()
        payload["criteria"].append(dict(payload["criteria"][0]))
        with pytest.raises(ReviewRubricError, match="duplicate id"):
            load_review_rubric(_write_rubric(tmp_path, payload))

    def test_bad_id_pattern_rejected(self, tmp_path: Path) -> None:
        payload = _minimal()
        payload["criteria"][0]["id"] = "Not Valid"
        with pytest.raises(ReviewRubricError, match="must match"):
            load_review_rubric(_write_rubric(tmp_path, payload))

    def test_required_forbids_weight(self, tmp_path: Path) -> None:
        payload = _minimal()
        payload["criteria"][0].update(required=True, weight=2)
        with pytest.raises(ReviewRubricError, match="required"):
            load_review_rubric(_write_rubric(tmp_path, payload))

    def test_nonfinite_weight_rejected(self, tmp_path: Path) -> None:
        # json.loads accepts Infinity/NaN as a nonstandard extension; the
        # loader must reject them rather than corrupt the weighted mean.
        path = tmp_path / "rubric.json"
        path.write_text(
            '{"schema_version": "1.0", "criteria": '
            '[{"id": "a", "criterion": "text", "weight": Infinity}]}'
        )
        with pytest.raises(ReviewRubricError, match="non-finite"):
            load_review_rubric(path)

    def test_choices_need_two_unique_entries(self, tmp_path: Path) -> None:
        payload = _minimal()
        payload["criteria"][0]["choices"] = ["yes"]
        with pytest.raises(ReviewRubricError, match="choices"):
            load_review_rubric(_write_rubric(tmp_path, payload))
        payload["criteria"][0]["choices"] = ["yes", "yes"]
        with pytest.raises(ReviewRubricError, match="duplicates"):
            load_review_rubric(_write_rubric(tmp_path, payload))

    def test_rubric_must_be_scorable(self, tmp_path: Path) -> None:
        # Only metrics and penalties: nothing can earn credit.
        payload = _minimal()
        payload["criteria"][0]["weight"] = 0
        payload["criteria"].append(
            {"id": "penalty", "criterion": "Undesirable thing.", "weight": -2}
        )
        with pytest.raises(ReviewRubricError, match="cannot produce a score"):
            load_review_rubric(_write_rubric(tmp_path, payload))

    def test_unregistered_reviewer_agent_rejected(self, tmp_path: Path) -> None:
        payload = _minimal(reviewer={"agent": "definitely-not-an-agent"})
        with pytest.raises(ReviewRubricError, match=r"reviewer\.agent"):
            load_review_rubric(_write_rubric(tmp_path, payload))

    def test_pass_threshold_bounds(self, tmp_path: Path) -> None:
        with pytest.raises(ReviewRubricError, match="pass_threshold"):
            load_review_rubric(_write_rubric(tmp_path, _minimal(pass_threshold=1.5)))


class TestDiscovery:
    def test_is_review_rubric_requires_schema_version(self, tmp_path: Path) -> None:
        llm_judge = tmp_path / "rubric.json"
        llm_judge.write_text(json.dumps({"criteria": [{"id": "c", "title": "x"}]}))
        assert not is_review_rubric_file(llm_judge)
        assert find_review_rubric(tmp_path) is None

        llm_judge.write_text(json.dumps(_minimal()))
        assert is_review_rubric_file(llm_judge)
        assert find_review_rubric(tmp_path) == llm_judge

    def test_unreadable_or_missing_is_not_claimed(self, tmp_path: Path) -> None:
        assert find_review_rubric(tmp_path) is None
        broken = tmp_path / "rubric.json"
        broken.write_text("{not json")
        assert not is_review_rubric_file(broken)


# Verdict parsing


def _criterion(cid: str = "correct", **overrides) -> ReviewCriterion:
    defaults: dict = {"id": cid, "criterion": "text"}
    defaults.update(overrides)
    return ReviewCriterion(**defaults)


def _verdict_reply(cid: str = "correct", verdict: str = "yes") -> str:
    return json.dumps(
        {
            "verdicts": [
                {
                    "id": cid,
                    "reasoning": "checked /app/result.txt",
                    "evidence": ["/app/result.txt"],
                    "verdict": verdict,
                }
            ]
        }
    )


class TestParseReviewerMessage:
    def test_parses_verdict_embedded_in_prose(self) -> None:
        reply = "I inspected the workspace.\n\n" + _verdict_reply()
        verdicts, error = parse_reviewer_message(reply, [_criterion()])
        assert error is None
        assert verdicts[0].score == 1.0

    def test_skips_echoed_format_example(self) -> None:
        # The prompt's format example contains "..." and fails to parse; the
        # scanner must pass over it and find the real object.
        reply = (
            'Example: {"verdicts": [{"id": "<criterion id>", ...}]}\n'
            + _verdict_reply()
        )
        verdicts, error = parse_reviewer_message(reply, [_criterion()])
        assert error is None
        assert verdicts[0].verdict == "yes"

    def test_no_verdicts_object_is_structural_error(self) -> None:
        verdicts, error = parse_reviewer_message("no json here", [_criterion()])
        assert verdicts == []
        assert error is not None and "verdicts" in error

    def test_id_mismatch_is_structural_error(self) -> None:
        _, error = parse_reviewer_message(
            _verdict_reply(cid="other"), [_criterion()]
        )
        assert error is not None and "missing" in error

    def test_off_menu_verdict_is_unscored_not_zero(self) -> None:
        verdicts, error = parse_reviewer_message(
            _verdict_reply(verdict="maybe"), [_criterion()]
        )
        assert error is None
        assert verdicts[0].score is None
        assert verdicts[0].unscored_reason is not None

    def test_missing_evidence_is_unscored(self) -> None:
        reply = json.dumps(
            {
                "verdicts": [
                    {
                        "id": "correct",
                        "reasoning": "looks fine",
                        "evidence": [],
                        "verdict": "yes",
                    }
                ]
            }
        )
        verdicts, error = parse_reviewer_message(reply, [_criterion()])
        assert error is None
        assert verdicts[0].score is None
        assert "evidence" in (verdicts[0].unscored_reason or "")

    def test_extract_skips_non_verdict_objects(self) -> None:
        text = '{"foo": 1} then {"verdicts": []}'
        assert extract_verdicts_object(text) == {"verdicts": []}


# Aggregation


def _rubric(*criteria: ReviewCriterion, pass_threshold: float = 0.7) -> ReviewRubric:
    return ReviewRubric(
        criteria=tuple(criteria),
        reviewer=ReviewerSpec(),
        pass_threshold=pass_threshold,
    )


def _scored_verdict(cid: str, score: float) -> CriterionVerdict:
    return CriterionVerdict(
        criterion_id=cid,
        verdict="x",
        reasoning="r",
        evidence=("e",),
        score=score,
    )


class TestAggregate:
    def test_weighted_mean_with_clamp(self) -> None:
        rubric = _rubric(
            _criterion("a", weight=2.0),
            _criterion("b", weight=1.0),
        )
        outcome = aggregate(
            rubric, [_scored_verdict("a", 1.0), _scored_verdict("b", 0.0)]
        )
        assert outcome.status == STATUS_SCORED
        assert outcome.review == pytest.approx(2 / 3)
        assert outcome.passed is False

    def test_negative_weight_erodes_but_never_enters_denominator(self) -> None:
        # HealthBench math: denominator sums positive weights only.
        rubric = _rubric(
            _criterion("earn", weight=2.0),
            _criterion("penalty", weight=-2.0),
        )
        clean = aggregate(
            rubric, [_scored_verdict("earn", 1.0), _scored_verdict("penalty", 0.0)]
        )
        assert clean.review == pytest.approx(1.0)
        dirty = aggregate(
            rubric, [_scored_verdict("earn", 1.0), _scored_verdict("penalty", 1.0)]
        )
        assert dirty.review == pytest.approx(0.0)  # (2 - 2) / 2, clamped at 0

    def test_failed_gate_zeroes_review(self) -> None:
        rubric = _rubric(
            _criterion("gate", required=True, weight=0.0),
            _criterion("quality", weight=1.0),
        )
        outcome = aggregate(
            rubric, [_scored_verdict("gate", 0.0), _scored_verdict("quality", 1.0)]
        )
        assert outcome.review == 0.0
        assert outcome.failed_gates == ["gate"]

    def test_gates_only_rubric_scores_binary(self) -> None:
        rubric = _rubric(_criterion("gate", required=True, weight=0.0))
        outcome = aggregate(rubric, [_scored_verdict("gate", 1.0)])
        assert outcome.review == 1.0

    def test_metric_never_moves_score_and_never_blocks(self) -> None:
        rubric = _rubric(
            _criterion("earn", weight=1.0),
            _criterion("diag", weight=0.0),
        )
        unscored_metric = CriterionVerdict(
            criterion_id="diag", verdict=None, unscored_reason="parse"
        )
        outcome = aggregate(
            rubric, [_scored_verdict("earn", 1.0), unscored_metric]
        )
        assert outcome.status == STATUS_SCORED
        assert outcome.review == 1.0
        assert "review/diag" not in outcome.reward_updates(rubric)

    def test_unscored_criterion_blocks_review_instead_of_scoring_zero(self) -> None:
        rubric = _rubric(_criterion("a", weight=1.0), _criterion("b", weight=1.0))
        outcome = aggregate(
            rubric,
            [
                _scored_verdict("a", 1.0),
                CriterionVerdict(
                    criterion_id="b", verdict=None, unscored_reason="off-menu"
                ),
            ],
        )
        assert outcome.status == STATUS_UNSCORED
        assert outcome.review is None
        assert outcome.reward_updates(rubric) == {}

    def test_reward_updates_shape(self) -> None:
        rubric = _rubric(_criterion("a", weight=1.0), pass_threshold=0.5)
        outcome = aggregate(rubric, [_scored_verdict("a", 1.0)])
        updates = outcome.reward_updates(rubric)
        assert updates == {"review": 1.0, "review_passed": 1.0, "review/a": 1.0}
        assert "reward" not in updates  # the execution verifier owns it


# Config threading


class TestReviewParamsThreading:
    def test_from_legacy_passthrough(self, tmp_path: Path) -> None:
        from benchflow.rollout import RolloutConfig

        params = ReviewParams(enabled=True, agent="codex-acp", model="gpt-5.2")
        config = RolloutConfig.from_legacy(task_path=tmp_path, review=params)
        assert config.review is params

    def test_eval_plan_builds_review_params(self) -> None:
        from benchflow.eval_plan import EvalCreateRequest, _build_review_params

        assert _build_review_params(EvalCreateRequest()) is None
        params = _build_review_params(
            EvalCreateRequest(review=True, reviewer_agent="codex-acp")
        )
        assert params is not None
        assert params.enabled is True
        assert params.agent == "codex-acp"

    def test_eval_plan_rejects_unknown_reviewer_agent(self) -> None:
        from benchflow.eval_plan import (
            EvalCreateRequest,
            EvalPlanError,
            _build_review_params,
        )

        with pytest.raises(EvalPlanError, match="reviewer-agent"):
            _build_review_params(EvalCreateRequest(reviewer_agent="nope-agent"))


# tasks check integration


class TestCheckTaskReviewRubric:
    def test_invalid_review_rubric_surfaces_issue(self, tmp_path: Path) -> None:
        from benchflow._utils.task_authoring import check_task

        task = tmp_path / "demo"
        (task / "verifier").mkdir(parents=True)
        (task / "environment").mkdir()
        (task / "oracle").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
        (task / "verifier" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
        (task / "task.md").write_text(
            "---\nschema_version: '1.0'\n---\n\nDo the thing.\n"
        )
        bad = {"schema_version": "1.0", "criteria": []}
        (task / "verifier" / "rubric.json").write_text(json.dumps(bad))

        issues = check_task(task)
        assert any("rubric.json invalid" in issue for issue in issues)

        good = _minimal()
        (task / "verifier" / "rubric.json").write_text(json.dumps(good))
        issues = check_task(task)
        assert not any("rubric.json" in issue for issue in issues)
