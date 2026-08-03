"""Tests for the v1.2 rubric-review schema, parsing, scoring, and threading."""

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
from benchflow.review.prompts import render_review_prompt
from benchflow.review.scoring import (
    STATUS_SCORED,
    STATUS_UNSCORED,
    CriterionVerdict,
)


def _write_rubric(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps(payload))
    return path


def _criterion_payload(**overrides) -> dict:
    payload = {
        "id": "correct",
        "criterion": "The plan derives the requested result from supplied evidence.",
        "criterion_type": "data-handling",
    }
    payload.update(overrides)
    return payload


def _minimal(**overrides) -> dict:
    payload = {
        "schema_version": "1.2",
        "criteria": [_criterion_payload()],
    }
    payload.update(overrides)
    return payload


class TestLoadReviewRubric:
    def test_minimal_v12_rubric_loads(self, tmp_path: Path) -> None:
        """Guards PR #942 remediation: the supplied v1.2 shape is canonical."""

        with pytest.warns(UserWarning, match="no gating"):
            rubric = load_review_rubric(_write_rubric(tmp_path, _minimal()))
        criterion = rubric.criteria[0]
        assert criterion.criterion_type == "data-handling"
        assert criterion.weight == 1.0
        assert not criterion.gating
        assert rubric.reviewer.mode == "per_criterion"
        assert rubric.reviewer.timeout_sec == 1800

    def test_reviewer_v12_fields(self, tmp_path: Path) -> None:
        payload = _minimal(
            reviewer={
                "harness": "gemini",
                "model": "gemini-2.5-flash",
                "timeout_sec": 300,
                "mode": "batched",
            }
        )
        with pytest.warns(UserWarning):
            rubric = load_review_rubric(_write_rubric(tmp_path, payload))
        assert rubric.reviewer.harness == "gemini"
        assert rubric.reviewer.timeout_sec == 300
        assert rubric.reviewer.mode == "batched"

    def test_old_pr_schema_is_rejected(self, tmp_path: Path) -> None:
        """Guards PR #942 remediation against reintroducing choices/required."""

        payload = _minimal(reviewer={"agent": "gemini", "timeout": 300})
        payload["criteria"][0].update(choices=["no", "yes"], required=True)
        with pytest.raises(ReviewRubricError, match="unknown key"):
            load_review_rubric(_write_rubric(tmp_path, payload))

    def test_v10_contract_is_rejected(self, tmp_path: Path) -> None:
        """Guards PR #942 against relabeling the breaking v1.2 shape as v1.0."""

        payload = _minimal(schema_version="1.0")
        with pytest.raises(ReviewRubricError, match=r"must be '1\.2'"):
            load_review_rubric(_write_rubric(tmp_path, payload))

    def test_criterion_type_is_required_and_closed(self, tmp_path: Path) -> None:
        payload = _minimal()
        payload["criteria"][0].pop("criterion_type")
        with pytest.raises(ReviewRubricError, match="criterion_type"):
            load_review_rubric(_write_rubric(tmp_path, payload))
        payload["criteria"][0]["criterion_type"] = "other"
        with pytest.raises(ReviewRubricError, match="criterion_type"):
            load_review_rubric(_write_rubric(tmp_path, payload))

    def test_gating_forbids_weight(self, tmp_path: Path) -> None:
        payload = _minimal()
        payload["criteria"][0].update(gating=True, weight=2)
        with pytest.raises(ReviewRubricError, match="gating=true forbids weight"):
            load_review_rubric(_write_rubric(tmp_path, payload))

    def test_positive_non_gating_weight_is_required(self, tmp_path: Path) -> None:
        payload = _minimal()
        payload["criteria"][0]["weight"] = 0
        with pytest.raises(ReviewRubricError, match="positive non-gating"):
            load_review_rubric(_write_rubric(tmp_path, payload))

    def test_nonfinite_weight_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "rubric.json"
        path.write_text(
            '{"schema_version":"1.2","criteria":['
            '{"id":"a","criterion":"A criterion long enough to validate.",'
            '"criterion_type":"data-handling","weight":Infinity}]}'
        )
        with pytest.raises(ReviewRubricError, match="non-finite"):
            load_review_rubric(path)

    def test_numeric_answer_leak_rejected_but_tolerance_allowed(
        self, tmp_path: Path
    ) -> None:
        """Guards PR #942 remediation: criteria cannot expose verifier answers."""

        (tmp_path / "test_outputs.py").write_text("EXPECTED = 3\nTOLERANCE = 15\n")
        payload = _minimal()
        payload["criteria"][0]["criterion"] = (
            "The final answer contains exactly 3 classified citations."
        )
        with pytest.raises(ReviewRubricError, match="expected answers"):
            load_review_rubric(_write_rubric(tmp_path, payload))

        payload["criteria"][0]["criterion"] = (
            "The plan checks numerical agreement to within 15 percent accuracy."
        )
        with pytest.warns(UserWarning):
            load_review_rubric(_write_rubric(tmp_path, payload))

    @pytest.mark.parametrize("field", ["schema_version", "criteria"])
    def test_required_top_level_fields(self, tmp_path: Path, field: str) -> None:
        payload = _minimal()
        payload.pop(field)
        with pytest.raises(ReviewRubricError):
            load_review_rubric(_write_rubric(tmp_path, payload))

    def test_duplicate_ids_rejected(self, tmp_path: Path) -> None:
        payload = _minimal()
        payload["criteria"].append(dict(payload["criteria"][0]))
        with pytest.raises(ReviewRubricError, match="duplicate id"):
            load_review_rubric(_write_rubric(tmp_path, payload))

    def test_unregistered_harness_rejected(self, tmp_path: Path) -> None:
        payload = _minimal(reviewer={"harness": "definitely-not-an-agent"})
        with pytest.raises(ReviewRubricError, match=r"reviewer\.harness"):
            load_review_rubric(_write_rubric(tmp_path, payload))


class TestDiscovery:
    def test_is_review_rubric_requires_schema_version(self, tmp_path: Path) -> None:
        rubric = tmp_path / "rubric.json"
        rubric.write_text(json.dumps({"criteria": [{"id": "c"}]}))
        assert not is_review_rubric_file(rubric)
        assert find_review_rubric(tmp_path) is None
        rubric.write_text(json.dumps(_minimal()))
        assert is_review_rubric_file(rubric)
        assert find_review_rubric(tmp_path) == rubric


def _criterion(identifier: str = "correct", **overrides) -> ReviewCriterion:
    defaults = {
        "id": identifier,
        "criterion": "A criterion long enough for the test fixture.",
        "criterion_type": "data-handling",
    }
    defaults.update(overrides)
    return ReviewCriterion(**defaults)


def test_review_prompt_requires_read_or_search_evidence() -> None:
    """Guards PR #942 against prompting evidence that runtime rejects."""

    prompt = render_review_prompt(
        [_criterion()],
        task_prompt="Inspect the supplied evidence.",
        trajectory_files=["acp_trajectory.jsonl"],
        first_batch=True,
    )
    assert "only when it names content inspected" in prompt
    assert "``read`` or ``search`` tool event" in prompt
    assert "Shell execution" in prompt


def _verdict_reply(identifier: str = "correct", met: object = True) -> str:
    return json.dumps(
        {
            "verdicts": [
                {
                    "id": identifier,
                    "explanation": "I inspected /review/workspace/root/result.json.",
                    "evidence": ["/review/workspace/root/result.json"],
                    "criterion_met": met,
                }
            ]
        }
    )


class TestParseReviewerMessage:
    def test_binary_contract_parses(self) -> None:
        verdicts, error = parse_reviewer_message(
            _verdict_reply(),
            [_criterion()],
        )
        assert error is None
        assert verdicts[0].criterion_met is True
        assert verdicts[0].score == 1.0

    def test_criterion_met_must_be_boolean(self) -> None:
        _, error = parse_reviewer_message(
            _verdict_reply(met="yes"),
            [_criterion()],
        )
        assert error is not None and "JSON boolean" in error

    def test_last_complete_object_wins(self) -> None:
        """Guards PR #942 remediation for multi-message ACP reviewer turns."""

        partial = json.dumps(
            {
                "verdicts": [
                    {
                        "id": "wrong",
                        "explanation": "partial",
                        "evidence": ["x"],
                        "criterion_met": False,
                    }
                ]
            }
        )
        reply = partial + "\n" + _verdict_reply()
        verdicts, error = parse_reviewer_message(reply, [_criterion()])
        assert error is None
        assert verdicts[0].criterion_id == "correct"
        assert extract_verdicts_object(reply) == json.loads(_verdict_reply())

    def test_fabricated_evidence_is_rejected_against_trace(self) -> None:
        """Guards PR #942 remediation: evidence must correspond to tool activity."""

        _, error = parse_reviewer_message(
            _verdict_reply(),
            [_criterion()],
            evidence_trace='{"title":"read other.txt"}',
        )
        assert error is not None and "not backed" in error
        verdicts, error = parse_reviewer_message(
            _verdict_reply(),
            [_criterion()],
            evidence_trace='{"title":"read result.json"}',
        )
        assert error is None
        assert verdicts[0].score == 1.0

    def test_exact_provider_tool_call_citation_ignores_argument_order(self) -> None:
        """Guards PR #942 against ACP search titles that omit exact arguments."""

        payload = json.loads(_verdict_reply())
        payload["verdicts"][0]["evidence"] = [
            "grep_search(dir_path='/review/trajectory', "
            "include_pattern='*.jsonl', pattern='largest', total_max_matches=50)"
        ]
        evidence_trace = json.dumps(
            [
                {
                    "type": "provider_tool_call",
                    "name": "grep_search",
                    "arguments": {
                        "total_max_matches": 50,
                        "pattern": "largest",
                        "include_pattern": "*.jsonl",
                        "dir_path": "/review/trajectory",
                    },
                }
            ]
        )
        verdicts, error = parse_reviewer_message(
            json.dumps(payload), [_criterion()], evidence_trace=evidence_trace
        )
        assert error is None
        assert verdicts[0].score == 1.0

        payload["verdicts"][0]["evidence"] = [
            "grep_search(dir_path='/review/trajectory', pattern='invented')"
        ]
        _, error = parse_reviewer_message(
            json.dumps(payload), [_criterion()], evidence_trace=evidence_trace
        )
        assert error is not None and "not backed" in error

    def test_empty_evidence_is_structural_error(self) -> None:
        payload = json.loads(_verdict_reply())
        payload["verdicts"][0]["evidence"] = []
        _, error = parse_reviewer_message(json.dumps(payload), [_criterion()])
        assert error is not None and "evidence" in error


def _rubric(*criteria: ReviewCriterion, pass_threshold: float = 0.7) -> ReviewRubric:
    return ReviewRubric(
        criteria=tuple(criteria),
        reviewer=ReviewerSpec(harness="gemini"),
        pass_threshold=pass_threshold,
    )


def _scored(identifier: str, score: float) -> CriterionVerdict:
    return CriterionVerdict(
        criterion_id=identifier,
        criterion_met=bool(score),
        explanation="checked",
        evidence=("result.json",),
        score=score,
    )


class TestAggregate:
    def test_signed_weighted_mean(self) -> None:
        rubric = _rubric(
            _criterion("earn", weight=2.0),
            _criterion("penalty", weight=-1.0),
        )
        outcome = aggregate(rubric, [_scored("earn", 1.0), _scored("penalty", 1.0)])
        assert outcome.status == STATUS_SCORED
        assert outcome.plan == pytest.approx(0.5)

    def test_failed_gate_zeroes_plan(self) -> None:
        rubric = _rubric(
            _criterion("gate", gating=True, weight=0.0),
            _criterion("quality", weight=1.0),
        )
        outcome = aggregate(rubric, [_scored("gate", 0.0), _scored("quality", 1.0)])
        assert outcome.plan == 0.0
        assert outcome.failed_gates == ["gate"]

    def test_unscored_non_metric_nulls_plan(self) -> None:
        rubric = _rubric(_criterion("quality"))
        outcome = aggregate(
            rubric,
            [
                CriterionVerdict(
                    criterion_id="quality",
                    criterion_met=None,
                    unscored_reason="reviewer failed",
                )
            ],
        )
        assert outcome.status == STATUS_UNSCORED
        assert outcome.plan is None
        assert outcome.reward_updates(rubric) == {}

    def test_plan_reward_shape(self) -> None:
        rubric = _rubric(_criterion("quality"), pass_threshold=0.5)
        outcome = aggregate(rubric, [_scored("quality", 1.0)])
        assert outcome.reward_updates(rubric) == {
            "plan": 1.0,
            "plan_passed": 1.0,
            "plan/quality": 1.0,
        }


class TestReviewParamsThreading:
    def test_mapping_roundtrip(self) -> None:
        params = ReviewParams(
            enabled=True,
            harness="codex-acp",
            model="gpt-5.2",
            timeout_sec=600,
            mode="batched",
            reasoning_effort="xhigh",
        )
        assert ReviewParams.from_mapping(params.to_mapping()) == params

    def test_rollout_config_coerces_mapping(self, tmp_path: Path) -> None:
        from benchflow.rollout import RolloutConfig

        config = RolloutConfig(
            task_path=tmp_path,
            review={"enabled": True, "harness": "gemini"},
        )
        assert isinstance(config.review, ReviewParams)
        assert config.review.harness == "gemini"

    def test_eval_plan_builds_v12_review_params(self) -> None:
        from benchflow.eval_plan import EvalCreateRequest, _build_review_params

        params = _build_review_params(
            EvalCreateRequest(
                review=True,
                reviewer_harness="codex-acp",
                reviewer_timeout_sec=300,
                reviewer_mode="batched",
                reviewer_reasoning_effort="xhigh",
            )
        )
        assert params is not None
        assert params.harness == "codex-acp"
        assert params.timeout_sec == 300
        assert params.mode == "batched"
        assert params.reasoning_effort == "xhigh"

    def test_shard_payload_preserves_review(self) -> None:
        """Guards PR #942 remediation against dropping review in worker shards."""

        from benchflow.eval_sharding import EvalShard, _config_payload
        from benchflow.evaluation import EvaluationConfig

        config = EvaluationConfig(
            review=ReviewParams(enabled=True, harness="gemini", mode="batched")
        )
        payload = _config_payload(
            config,
            shard=EvalShard(index=0, task_names=("demo",), concurrency=1),
        )
        assert payload["review"] == config.review.to_mapping()


class TestCheckTaskReviewRubric:
    def test_task_check_uses_v12_and_answer_leak_lint(self, tmp_path: Path) -> None:
        """Guards PR #942 remediation through the public task checker."""

        from benchflow._utils.task_authoring import check_task

        task = tmp_path / "demo"
        (task / "verifier").mkdir(parents=True)
        (task / "environment").mkdir()
        (task / "oracle").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        (task / "verifier" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
        (task / "verifier" / "test_outputs.py").write_text("EXPECTED = 42\n")
        (task / "task.md").write_text(
            "---\nschema_version: '1.3'\n---\n\n## prompt\n\nDo the thing.\n"
        )
        rubric = _minimal(
            reviewer={"harness": "gemini"},
            criteria=[
                _criterion_payload(gating=True),
                _criterion_payload(
                    id="quality",
                    criterion="The plan validates its output against the requested schema.",
                ),
            ],
        )
        (task / "verifier" / "rubric.json").write_text(json.dumps(rubric))
        issues = check_task(task)
        assert not any("rubric.json" in issue for issue in issues)

        rubric["criteria"][1]["criterion"] = (
            "The plan writes exactly 42 entries into the final output artifact."
        )
        (task / "verifier" / "rubric.json").write_text(json.dumps(rubric))
        issues = check_task(task)
        assert any("expected answers" in issue for issue in issues)
