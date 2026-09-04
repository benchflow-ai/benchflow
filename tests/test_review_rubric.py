"""Unit tests for the rubric contract: schema, prompts, and wrapper assembly."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow.review.config import (
    DEFAULT_RUBRIC_PATH,
    LEGACY_REVIEW_RUBRIC_CONTRACT,
    REVIEW_RESULT_FILENAME,
    REVIEW_RUBRIC_CONTRACT,
    WEIGHTED_REVIEW_RUBRIC_CONTRACT,
    ReviewRubricError,
    Rubric,
    build_criteria_guidance,
    build_review_response_model,
    find_task_rubric,
    load_rubric,
)
from benchflow.review.prompts import (
    TASK_MOUNT,
    TRIAL_MOUNT,
    render_review_instruction,
)
from benchflow.review.scoring import (
    PublicationDecision,
    score_weighted_review,
)
from benchflow.review.wrapper import assemble_review_task, copy_evidence

RUBRIC = {
    "criteria": [
        {
            "name": "method_soundness",
            "description": "Internal note for rubric authors.",
            "guidance": "PASS when the recorded method is sound; FAIL otherwise.",
        },
        {
            "name": "output_contract",
            "description": "Another internal note.",
            "guidance": "PASS when required outputs exist; FAIL when missing.",
        },
    ]
}

WEIGHTED_RUBRIC = {
    "criteria": [
        {
            "name": "safety_gate",
            "blocker": 1,
            "weight": 10,
            "description": "The result must satisfy the safety gate.",
            "guidance": "PASS when the safety invariant holds; FAIL otherwise.",
        },
        {
            "name": "method_quality",
            "blocker": 0,
            "weight": 3,
            "description": "Assess the quality of the method.",
            "guidance": "Score 2 for complete, 1 for partial, or 0 for absent.",
        },
        {
            "name": "evidence_quality",
            "blocker": 0,
            "weight": 2,
            "description": "Assess the quality of the evidence.",
            "guidance": "Score 2 for complete, 1 for partial, or 0 for absent.",
        },
    ]
}


def write_rubric(tmp_path: Path, data: dict, name: str = "rubric.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestLoadRubric:
    def test_loads_valid_rubric(self, tmp_path):
        rubric = load_rubric(write_rubric(tmp_path, RUBRIC))
        assert [c.name for c in rubric.criteria] == [
            "method_soundness",
            "output_contract",
        ]
        assert rubric.criteria[0].guidance.startswith("PASS when")
        assert rubric.contract == LEGACY_REVIEW_RUBRIC_CONTRACT
        assert REVIEW_RUBRIC_CONTRACT == LEGACY_REVIEW_RUBRIC_CONTRACT

    def test_loads_frontierphysics_weighted_rubric_contract(self, tmp_path):
        """Guards FrontierPhysics PR #109 at head commit 4bdbd2e against
        rejecting or silently downgrading its weighted rubric.json format.
        """

        rubric = load_rubric(write_rubric(tmp_path, WEIGHTED_RUBRIC))
        assert rubric.contract == WEIGHTED_REVIEW_RUBRIC_CONTRACT
        assert rubric.is_weighted
        assert rubric.metadata() == {
            "contract": "v0.2",
            "criteria": [
                {"name": "safety_gate", "blocker": 1, "weight": 10},
                {"name": "method_quality", "blocker": 0, "weight": 3},
                {"name": "evidence_quality", "blocker": 0, "weight": 2},
            ],
        }

    @pytest.mark.parametrize("field", ["blocker", "weight"])
    def test_rejects_partially_weighted_criterion(self, tmp_path, field):
        criterion = dict(WEIGHTED_RUBRIC["criteria"][1])
        criterion.pop(field)
        with pytest.raises(ReviewRubricError, match="both blocker and weight"):
            load_rubric(write_rubric(tmp_path, {"criteria": [criterion]}))

    def test_rejects_mixed_legacy_and_weighted_contracts(self, tmp_path):
        mixed = {"criteria": [RUBRIC["criteria"][0], WEIGHTED_RUBRIC["criteria"][1]]}
        with pytest.raises(ReviewRubricError, match="cannot be mixed"):
            load_rubric(write_rubric(tmp_path, mixed))

    def test_rejects_explicit_null_scoring_fields(self, tmp_path):
        criterion = {
            **RUBRIC["criteria"][0],
            "blocker": None,
            "weight": None,
        }
        with pytest.raises(ReviewRubricError, match="null is not allowed"):
            load_rubric(write_rubric(tmp_path, {"criteria": [criterion]}))

    def test_rejects_weighted_rubric_with_only_blockers(self, tmp_path):
        only_blockers = {
            "criteria": [
                {**WEIGHTED_RUBRIC["criteria"][0], "name": "first_gate"},
                {**WEIGHTED_RUBRIC["criteria"][0], "name": "second_gate"},
            ]
        }
        with pytest.raises(ReviewRubricError, match="blocker: 0"):
            load_rubric(write_rubric(tmp_path, only_blockers))

    @pytest.mark.parametrize(
        "blocker",
        [True, False, -1, 2, 0.0, 1.0, "0", "1", None],
    )
    def test_blocker_requires_strict_integer_zero_or_one(self, tmp_path, blocker):
        criterion = {**WEIGHTED_RUBRIC["criteria"][1], "blocker": blocker}
        with pytest.raises(ReviewRubricError, match="not a valid rubric"):
            load_rubric(write_rubric(tmp_path, {"criteria": [criterion]}))

    @pytest.mark.parametrize(
        "weight",
        [True, False, 0, 11, 1.0, 10.0, "1", "10", None],
    )
    def test_weight_requires_strict_integer_one_through_ten(self, tmp_path, weight):
        criterion = {**WEIGHTED_RUBRIC["criteria"][1], "weight": weight}
        with pytest.raises(ReviewRubricError, match="not a valid rubric"):
            load_rubric(write_rubric(tmp_path, {"criteria": [criterion]}))

    def test_default_rubric_ships_and_loads(self):
        rubric = load_rubric(None)
        assert DEFAULT_RUBRIC_PATH.is_file()
        assert [c.name for c in rubric.criteria] == [
            "reward_hacking",
            "task_specification",
        ]

    def test_rejects_non_json_suffix(self, tmp_path):
        path = tmp_path / "rubric.yaml"
        path.write_text("criteria: []", encoding="utf-8")
        with pytest.raises(ReviewRubricError, match="JSON"):
            load_rubric(path)

    def test_rejects_invalid_json(self, tmp_path):
        path = tmp_path / "rubric.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ReviewRubricError, match="not valid JSON"):
            load_rubric(path)

    def test_rejects_missing_fields(self, tmp_path):
        bad = {"criteria": [{"name": "x", "guidance": "no description"}]}
        with pytest.raises(ReviewRubricError, match="not a valid rubric"):
            load_rubric(write_rubric(tmp_path, bad))

    def test_rejects_non_identifier_name(self, tmp_path):
        bad = {"criteria": [{"name": "bad-name", "description": "d", "guidance": "g"}]}
        with pytest.raises(ReviewRubricError, match="identifier"):
            load_rubric(write_rubric(tmp_path, bad))

    def test_rejects_unknown_keys(self, tmp_path):
        """Unknown fields fail closed so typos cannot silently change review
        behavior."""
        data = {
            "criteria": [
                {
                    "name": "x",
                    "description": "d",
                    "guidance": "g",
                    "extra": "nope",
                }
            ],
        }
        with pytest.raises(ReviewRubricError, match="not a valid rubric"):
            load_rubric(write_rubric(tmp_path, data))

    def test_rejects_unknown_top_level_keys(self, tmp_path):
        """Guards PR #942: top-level rubric typos must fail closed."""

        data = {**RUBRIC, "criterai": []}
        with pytest.raises(ReviewRubricError, match="not a valid rubric"):
            load_rubric(write_rubric(tmp_path, data))

    def test_rejects_schema_reserved_name(self, tmp_path):
        """Guards PR #942: model-prefixed names cannot erase schema fields."""

        bad = {
            "criteria": [{"name": "model_future", "description": "d", "guidance": "g"}]
        }
        with pytest.raises(ReviewRubricError, match="reserved"):
            load_rubric(write_rubric(tmp_path, bad))

    def test_rejects_empty_criteria(self, tmp_path):
        """An empty rubric would let the wrapper award reward 1 to a review
        containing zero judgments."""
        with pytest.raises(ReviewRubricError, match="not a valid rubric"):
            load_rubric(write_rubric(tmp_path, {"criteria": []}))

    def test_rejects_duplicate_names(self, tmp_path):
        """Duplicate names would silently collapse into one structured-output
        field."""
        entry = {"name": "x", "description": "d", "guidance": "g"}
        with pytest.raises(ReviewRubricError, match="unique"):
            load_rubric(write_rubric(tmp_path, {"criteria": [entry, dict(entry)]}))


class TestFindTaskRubric:
    @pytest.mark.parametrize("tests_dir", ["verifier", "tests"])
    def test_finds_shipped_rubric(self, tmp_path, tests_dir):
        (tmp_path / tests_dir).mkdir()
        target = write_rubric(tmp_path / tests_dir, RUBRIC)
        assert find_task_rubric(tmp_path) == target

    def test_returns_none_without_rubric(self, tmp_path):
        (tmp_path / "verifier").mkdir()
        assert find_task_rubric(tmp_path) is None

    def test_native_verifier_rubric_precedes_legacy_tests(self, tmp_path):
        """Guards PR #942: native verifier/rubric.json wins in dual layouts."""

        (tmp_path / "verifier").mkdir()
        (tmp_path / "tests").mkdir()
        native = write_rubric(tmp_path / "verifier", RUBRIC)
        write_rubric(tmp_path / "tests", RUBRIC)
        assert find_task_rubric(tmp_path) == native


class TestGuidanceAndSchema:
    def test_guidance_line_format(self):
        rubric = Rubric.model_validate(RUBRIC)
        guidance = build_criteria_guidance(rubric)
        assert guidance.splitlines() == [
            "- method_soundness: PASS when the recorded method is sound; FAIL otherwise.",
            "- output_contract: PASS when required outputs exist; FAIL when missing.",
        ]

    def test_description_never_reaches_the_prompt(self):
        rubric = Rubric.model_validate(RUBRIC)
        instruction = render_review_instruction(
            rubric, output_schema={"type": "object"}
        )
        assert "Internal note for rubric authors." not in instruction
        assert "Internal note" not in build_criteria_guidance(rubric)

    def test_response_model_shape(self):
        rubric = Rubric.model_validate(RUBRIC)
        schema = build_review_response_model(rubric).model_json_schema()
        assert set(schema["properties"]) == {"trial_name", "summary", "checks"}
        checks_ref = schema["properties"]["checks"]["$ref"].rsplit("/", 1)[-1]
        checks = schema["$defs"][checks_ref]
        assert set(checks["properties"]) == {"method_soundness", "output_contract"}
        outcome = schema["$defs"]["ReviewOutcomeValue"]["enum"]
        assert sorted(outcome) == ["fail", "not_applicable", "pass"]

    def test_response_model_validates_outcomes(self):
        rubric = Rubric.model_validate({"criteria": RUBRIC["criteria"][:1]})
        model = build_review_response_model(rubric)
        good = {
            "trial_name": "t",
            "summary": "s",
            "checks": {"method_soundness": {"explanation": "e", "outcome": "pass"}},
        }
        assert model.model_validate(good)
        bad = json.loads(json.dumps(good))
        bad["checks"]["method_soundness"]["outcome"] = "maybe"
        with pytest.raises(ValueError):
            model.model_validate(bad)

    def test_legacy_response_model_keeps_permissive_extra_field_behavior(self):
        model = build_review_response_model(
            Rubric.model_validate({"criteria": RUBRIC["criteria"][:1]})
        )
        result = {
            "trial_name": "t",
            "summary": "s",
            "legacy_top_level_extra": True,
            "checks": {
                "method_soundness": {
                    "explanation": "e",
                    "outcome": "pass",
                    "legacy_check_extra": True,
                }
            },
        }
        assert model.model_validate(result)

    @pytest.mark.parametrize("field", ["summary", "explanation"])
    def test_response_model_rejects_whitespace_only_text(self, field):
        model = build_review_response_model(
            Rubric.model_validate({"criteria": RUBRIC["criteria"][:1]})
        )
        result = {
            "trial_name": "t",
            "summary": "summary",
            "checks": {"method_soundness": {"explanation": "e", "outcome": "pass"}},
        }
        if field == "summary":
            result["summary"] = " \t "
        else:
            result["checks"]["method_soundness"]["explanation"] = " \t "
        with pytest.raises(ValueError, match="non-whitespace"):
            model.model_validate(result)

    def test_weighted_guidance_labels_blockers_scores_and_weights(self):
        rubric = Rubric.model_validate(WEIGHTED_RUBRIC)
        guidance = build_criteria_guidance(rubric)
        assert "safety_gate [BLOCKER; answer pass or fail; weight 10" in guidance
        assert "method_quality [SCORED; weight 3; answer 0, 1, or 2]" in guidance
        assert "evidence_quality [SCORED; weight 2; answer 0, 1, or 2]" in guidance

    def test_weighted_response_model_uses_distinct_check_shapes(self):
        model = build_review_response_model(Rubric.model_validate(WEIGHTED_RUBRIC))
        valid = {
            "trial_name": "t",
            "summary": "s",
            "checks": {
                "safety_gate": {"explanation": "safe", "outcome": "pass"},
                "method_quality": {"explanation": "complete", "score": 2},
                "evidence_quality": {"explanation": "partial", "score": 1},
            },
        }
        assert model.model_validate(valid)

        invalid_cases = []
        blocker_not_applicable = json.loads(json.dumps(valid))
        blocker_not_applicable["checks"]["safety_gate"]["outcome"] = "not_applicable"
        invalid_cases.append(blocker_not_applicable)
        scored_as_outcome = json.loads(json.dumps(valid))
        scored_as_outcome["checks"]["method_quality"] = {
            "explanation": "wrong shape",
            "outcome": "pass",
        }
        invalid_cases.append(scored_as_outcome)
        blocker_as_score = json.loads(json.dumps(valid))
        blocker_as_score["checks"]["safety_gate"] = {
            "explanation": "wrong shape",
            "score": 2,
        }
        invalid_cases.append(blocker_as_score)
        for invalid in invalid_cases:
            with pytest.raises(ValueError):
                model.model_validate(invalid)

    @pytest.mark.parametrize("score", [True, False, -1, 3, 1.0, "1", None])
    def test_weighted_response_model_requires_strict_score(self, score):
        rubric = Rubric.model_validate({"criteria": [WEIGHTED_RUBRIC["criteria"][1]]})
        model = build_review_response_model(rubric)
        result = {
            "trial_name": "t",
            "summary": "s",
            "checks": {
                "method_quality": {"explanation": "e", "score": score},
            },
        }
        with pytest.raises(ValueError):
            model.model_validate(result)


class TestWeightedScoring:
    def test_blocker_weights_are_excluded_and_point_eight_is_publishable(self):
        rubric = Rubric.model_validate(WEIGHTED_RUBRIC)
        scoring = score_weighted_review(
            rubric,
            {
                "safety_gate": {"outcome": "pass", "explanation": "safe"},
                "method_quality": {"score": 2, "explanation": "complete"},
                "evidence_quality": {"score": 1, "explanation": "partial"},
            },
            deterministic_pass=True,
        )
        assert scoring.weighted_points == 8
        assert scoring.max_weighted_points == 10
        assert scoring.raw_quality == pytest.approx(0.8)
        assert scoring.gated_quality == pytest.approx(0.8)
        assert scoring.decision is PublicationDecision.PUBLISHABLE

    def test_point_six_five_is_presentable_with_revisions(self):
        rubric_data = {
            "criteria": [
                {**WEIGHTED_RUBRIC["criteria"][1], "weight": 3},
                {**WEIGHTED_RUBRIC["criteria"][2], "weight": 7},
            ]
        }
        scoring = score_weighted_review(
            Rubric.model_validate(rubric_data),
            {
                "method_quality": {"score": 2},
                "evidence_quality": {"score": 1},
            },
            deterministic_pass=True,
        )
        assert scoring.raw_quality == pytest.approx(0.65)
        assert scoring.decision is PublicationDecision.PRESENTABLE_WITH_REVISIONS

    @pytest.mark.parametrize(
        ("deterministic_pass", "blocker_outcome", "failed_blockers"),
        [
            (False, "pass", ()),
            (True, "fail", ("safety_gate",)),
            (False, "fail", ("safety_gate",)),
        ],
    )
    def test_failed_gate_zeroes_gated_quality_but_preserves_raw_quality(
        self, deterministic_pass, blocker_outcome, failed_blockers
    ):
        scoring = score_weighted_review(
            Rubric.model_validate(WEIGHTED_RUBRIC),
            {
                "safety_gate": {"outcome": blocker_outcome},
                "method_quality": {"score": 2},
                "evidence_quality": {"score": 1},
            },
            deterministic_pass=deterministic_pass,
        )
        assert scoring.raw_quality == pytest.approx(0.8)
        assert scoring.gated_quality == 0.0
        assert scoring.failed_blockers == failed_blockers
        assert scoring.decision is PublicationDecision.NOT_PUBLISHABLE

    def test_below_point_six_five_is_not_publishable(self):
        rubric = Rubric.model_validate({"criteria": [WEIGHTED_RUBRIC["criteria"][1]]})
        scoring = score_weighted_review(
            rubric,
            {"method_quality": {"score": 1}},
            deterministic_pass=True,
        )
        assert scoring.raw_quality == pytest.approx(0.5)
        assert scoring.decision is PublicationDecision.NOT_PUBLISHABLE

    @pytest.mark.parametrize("score", [True, False, -1, 3, 1.0, "1", None])
    def test_aggregation_rejects_non_contract_scores(self, score):
        rubric = Rubric.model_validate({"criteria": [WEIGHTED_RUBRIC["criteria"][1]]})
        with pytest.raises(ValueError, match="integer score"):
            score_weighted_review(
                rubric,
                {"method_quality": {"score": score}},
                deterministic_pass=True,
            )

    def test_aggregation_rejects_legacy_rubric(self):
        with pytest.raises(ValueError, match=r"v0\.2"):
            score_weighted_review(
                Rubric.model_validate(RUBRIC),
                {"method_soundness": {"outcome": "pass"}},
                deterministic_pass=True,
            )

    @pytest.mark.parametrize("outcome", [[], {}])
    def test_aggregation_rejects_unhashable_blocker_outcomes(self, outcome):
        with pytest.raises(ValueError, match="pass or fail"):
            score_weighted_review(
                Rubric.model_validate(WEIGHTED_RUBRIC),
                {
                    "safety_gate": {"outcome": outcome},
                    "method_quality": {"score": 2},
                    "evidence_quality": {"score": 1},
                },
                deterministic_pass=True,
            )

    def test_aggregation_requires_boolean_deterministic_gate(self):
        with pytest.raises(ValueError, match="must be a boolean"):
            score_weighted_review(
                Rubric.model_validate(WEIGHTED_RUBRIC),
                {
                    "safety_gate": {"outcome": "pass"},
                    "method_quality": {"score": 2},
                    "evidence_quality": {"score": 1},
                },
                deterministic_pass=1,
            )


class TestPromptRendering:
    def test_instruction_contains_contract(self):
        rubric = Rubric.model_validate(RUBRIC)
        instruction = render_review_instruction(
            rubric, output_schema={"marker": "schema-sentinel"}
        )
        assert TRIAL_MOUNT in instruction
        assert TASK_MOUNT in instruction
        assert "- method_soundness:" in instruction
        assert "schema-sentinel" in instruction
        assert REVIEW_RESULT_FILENAME in instruction

    def test_instruction_without_task_dir(self):
        rubric = Rubric.model_validate(RUBRIC)
        instruction = render_review_instruction(rubric, task_path=None)
        assert TASK_MOUNT not in instruction
        assert "task definition is not available" in instruction.lower()

    def test_custom_template_missing_placeholders_renders(self):
        rubric = Rubric.model_validate(RUBRIC)
        instruction = render_review_instruction(rubric, template="Just review it.")
        assert instruction.startswith("Just review it.")

    def test_custom_template_typo_is_rejected(self):
        """Guards PR #942: misspelled placeholders cannot erase guidance."""

        rubric = Rubric.model_validate(RUBRIC)
        with pytest.raises(KeyError, match="criteria_guidence"):
            render_review_instruction(rubric, template="{criteria_guidence}")

    def test_trial_name_is_rendered_as_json(self):
        """Guards PR #942: hostile legal names remain unambiguous in prompts."""

        rubric = Rubric.model_validate(RUBRIC)
        instruction = render_review_instruction(rubric, trial_name='a"b\nline')
        assert 'exactly "a\\"b\\nline"' in instruction

    def test_weighted_instruction_explains_distinct_judgment_contracts(self):
        instruction = render_review_instruction(
            Rubric.model_validate(WEIGHTED_RUBRIC),
            output_schema={"marker": "weighted-schema"},
        )
        assert "BLOCKER" in instruction
        assert "pass" in instruction and "fail" in instruction
        assert "SCORED" in instruction
        assert "0, 1, or 2" in instruction
        assert "not_applicable" not in instruction
        assert "weighted-schema" in instruction


class TestWrapperAssembly:
    def make_rollout(self, tmp_path: Path) -> Path:
        rollout = tmp_path / "rollout"
        (rollout / "trajectory").mkdir(parents=True)
        (rollout / ".git").mkdir()
        (rollout / ".git" / "HEAD").write_text("ref", encoding="utf-8")
        (rollout / "result.json").write_text(
            json.dumps({"rewards": {"reward": 0.0}}), encoding="utf-8"
        )
        (rollout / REVIEW_RESULT_FILENAME).write_text("{}", encoding="utf-8")
        return rollout

    def test_assembles_wrapper(self, tmp_path):
        rubric = Rubric.model_validate(RUBRIC)
        rollout = self.make_rollout(tmp_path)
        task_dir = tmp_path / "task"
        (task_dir / "verifier").mkdir(parents=True)
        (task_dir / "task.md").write_text("body", encoding="utf-8")

        dest, uploads = assemble_review_task(
            rollout, task_dir, rubric, tmp_path / "wrapper"
        )

        task_md = (dest / "task.md").read_text(encoding="utf-8")
        # Pinned by digest, not a mutable tag (PR #942 re-review).
        assert "docker_image: python@sha256:" in task_md
        assert "test-script" in task_md
        assert not list(dest.rglob("Dockerfile"))
        assert (dest / "tests" / "test.sh").is_file()
        assert (dest / "tests" / "validate.py").is_file()
        metadata = json.loads(
            (dest / "tests" / "criteria.json").read_text(encoding="utf-8")
        )
        assert metadata == rubric.metadata()
        assert uploads == {
            str(dest / "evidence" / "trial"): TRIAL_MOUNT,
            str(dest / "evidence" / "task"): TASK_MOUNT,
        }

    def test_weighted_wrapper_carries_contract_metadata_not_guidance(self, tmp_path):
        rubric = Rubric.model_validate(WEIGHTED_RUBRIC)
        rollout = self.make_rollout(tmp_path)
        dest, _ = assemble_review_task(
            rollout, None, rubric, tmp_path / "weighted-wrapper"
        )
        metadata_text = (dest / "tests" / "criteria.json").read_text(encoding="utf-8")
        assert json.loads(metadata_text) == rubric.metadata()
        assert "Assess the quality" not in metadata_text
        assert "Score 2" not in metadata_text

    def test_wrapper_never_contains_the_rubric_file(self, tmp_path):
        """The rubric is decomposed host-side; the file itself never ships.

        Only the evidence copy of the reviewed task may carry one (it is part
        of that task's own files)."""
        rubric = Rubric.model_validate(RUBRIC)
        rollout = self.make_rollout(tmp_path)
        dest, _ = assemble_review_task(rollout, None, rubric, tmp_path / "wrapper")
        assert list(dest.rglob("rubric.json")) == []

    def test_evidence_excludes_vcs_and_prior_reviews(self, tmp_path):
        rubric = Rubric.model_validate(RUBRIC)
        rollout = self.make_rollout(tmp_path)
        dest, uploads = assemble_review_task(
            rollout, None, rubric, tmp_path / "wrapper"
        )
        trial_copy = dest / "evidence" / "trial"
        assert (trial_copy / "result.json").is_file()
        assert not (trial_copy / ".git").exists()
        assert not (trial_copy / REVIEW_RESULT_FILENAME).exists()
        assert uploads == {str(trial_copy): TRIAL_MOUNT}

    def test_evidence_uses_canonical_trajectory_not_provider_history(self, tmp_path):
        """Guards PR #942 against redundant provider history exhausting context."""

        rollout = tmp_path / "rollout"
        trajectory = rollout / "trajectory"
        trajectory.mkdir(parents=True)
        (trajectory / "acp_trajectory.jsonl").write_text(
            '{"type":"tool_call"}\n', encoding="utf-8"
        )
        (trajectory / "llm_trajectory.jsonl").write_text(
            '{"messages":["expanded provider history"]}\n', encoding="utf-8"
        )

        destination = tmp_path / "copy"
        copy_evidence(rollout, destination)

        assert (destination / "trajectory" / "acp_trajectory.jsonl").is_file()
        assert not (destination / "trajectory" / "llm_trajectory.jsonl").exists()

    def test_permission_normalization_fails_closed(self, tmp_path, monkeypatch):
        """Guards PR #942: chmod failure cannot admit writable evidence."""

        rubric = Rubric.model_validate(RUBRIC)
        rollout = self.make_rollout(tmp_path)
        original_chmod = Path.chmod

        def fail_on_copy(path: Path, mode: int, *args, **kwargs):
            if "wrapper" in path.parts:
                raise OSError("chmod refused")
            return original_chmod(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "chmod", fail_on_copy)
        with pytest.raises(OSError, match="chmod refused"):
            assemble_review_task(rollout, None, rubric, tmp_path / "wrapper")


class TestWrapperValidator:
    """Run the shipped in-sandbox validator exactly as the wrapper does."""

    def run_validator(
        self,
        tmp_path: Path,
        result: dict | str,
        *,
        rubric_data: dict | None = None,
    ) -> tuple[int, str]:
        rubric = Rubric.model_validate(RUBRIC if rubric_data is None else rubric_data)
        rollout = tmp_path / "r"
        rollout.mkdir()
        (rollout / "result.json").write_text("{}", encoding="utf-8")
        dest, _ = assemble_review_task(rollout, None, rubric, tmp_path / "w")
        self._trial_name = rollout.name
        result_path = tmp_path / REVIEW_RESULT_FILENAME
        payload = result if isinstance(result, str) else json.dumps(result)
        result_path.write_text(payload, encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                str(dest / "tests" / "validate.py"),
                str(result_path),
                str(dest / "tests" / "criteria.json"),
                str(dest / "tests" / "trial_name.txt"),
            ],
            capture_output=True,
            text=True,
        )
        return proc.returncode, proc.stdout

    def good_result(self) -> dict:
        return {
            "trial_name": "r",  # matches the rollout dir name used above
            "summary": "Did things.",
            "checks": {
                "method_soundness": {"explanation": "ok", "outcome": "pass"},
                "output_contract": {"explanation": "missing", "outcome": "fail"},
            },
        }

    def good_weighted_result(self) -> dict:
        return {
            "trial_name": "r",
            "summary": "Weighted review complete.",
            "checks": {
                "safety_gate": {"explanation": "safe", "outcome": "pass"},
                "method_quality": {"explanation": "complete", "score": 2},
                "evidence_quality": {"explanation": "partial", "score": 1},
            },
        }

    def test_valid_result_passes(self, tmp_path):
        code, out = self.run_validator(tmp_path, self.good_result())
        assert code == 0, out

    def test_not_applicable_is_valid(self, tmp_path):
        result = self.good_result()
        result["checks"]["method_soundness"]["outcome"] = "not_applicable"
        code, _ = self.run_validator(tmp_path, result)
        assert code == 0

    def test_legacy_validator_preserves_extra_field_compatibility(self, tmp_path):
        result = self.good_result()
        result["legacy_top_level_extra"] = True
        result["checks"]["method_soundness"]["legacy_check_extra"] = True
        code, out = self.run_validator(tmp_path, result)
        assert code == 0, out

    def test_unhashable_legacy_outcome_fails_cleanly(self, tmp_path):
        result = self.good_result()
        result["checks"]["method_soundness"]["outcome"] = []
        code, out = self.run_validator(tmp_path, result)
        assert code == 1
        assert "outcome must be one of" in out

    def test_valid_weighted_result_passes(self, tmp_path):
        code, out = self.run_validator(
            tmp_path,
            self.good_weighted_result(),
            rubric_data=WEIGHTED_RUBRIC,
        )
        assert code == 0, out

    @pytest.mark.parametrize("score", [0, 1, 2])
    def test_each_weighted_score_is_valid(self, tmp_path, score):
        result = self.good_weighted_result()
        result["checks"]["method_quality"]["score"] = score
        code, out = self.run_validator(tmp_path, result, rubric_data=WEIGHTED_RUBRIC)
        assert code == 0, out

    @pytest.mark.parametrize("score", [True, False, -1, 3, 1.0, "1", None])
    def test_invalid_weighted_scores_fail(self, tmp_path, score):
        result = self.good_weighted_result()
        result["checks"]["method_quality"]["score"] = score
        code, out = self.run_validator(tmp_path, result, rubric_data=WEIGHTED_RUBRIC)
        assert code == 1, out

    def test_weighted_blocker_disallows_not_applicable(self, tmp_path):
        result = self.good_weighted_result()
        result["checks"]["safety_gate"]["outcome"] = "not_applicable"
        code, out = self.run_validator(tmp_path, result, rubric_data=WEIGHTED_RUBRIC)
        assert code == 1, out

    @pytest.mark.parametrize(
        "criterion,replacement",
        [
            (
                "safety_gate",
                {"explanation": "wrong shape", "score": 2},
            ),
            (
                "method_quality",
                {"explanation": "wrong shape", "outcome": "pass"},
            ),
            (
                "method_quality",
                {"explanation": "extra field", "score": 2, "outcome": "pass"},
            ),
        ],
    )
    def test_weighted_check_shapes_fail_closed(self, tmp_path, criterion, replacement):
        result = self.good_weighted_result()
        result["checks"][criterion] = replacement
        code, out = self.run_validator(tmp_path, result, rubric_data=WEIGHTED_RUBRIC)
        assert code == 1, out

    def test_trial_name_round_trips_without_whitespace_loss(self, tmp_path):
        """Guards PR #942: validator identity matching must be byte-exact."""

        rubric = Rubric.model_validate(RUBRIC)
        rollout = tmp_path / " run "
        rollout.mkdir()
        (rollout / "result.json").write_text("{}", encoding="utf-8")
        dest, _ = assemble_review_task(rollout, None, rubric, tmp_path / "w-space")
        result = self.good_result()
        result["trial_name"] = " run "
        result_path = tmp_path / "space-review.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                str(dest / "tests" / "validate.py"),
                str(result_path),
                str(dest / "tests" / "criteria.json"),
                str(dest / "tests" / "trial_name.txt"),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda r: r["checks"].pop("method_soundness"), "missing criterion"),
            (
                lambda r: r["checks"].__setitem__("extra", {"outcome": "pass"}),
                "unexpected key",
            ),
            (
                lambda r: r["checks"]["method_soundness"].__setitem__(
                    "outcome", "maybe"
                ),
                "outcome must be one of",
            ),
            (
                lambda r: r["checks"]["method_soundness"].__setitem__(
                    "explanation", "  "
                ),
                "non-empty string",
            ),
            (lambda r: r.__setitem__("summary", ""), "summary"),
            (lambda r: r.pop("trial_name"), "trial_name"),
            (lambda r: r.__setitem__("trial_name", "some-other-run"), "trial_name"),
        ],
    )
    def test_invalid_results_fail(self, tmp_path, mutate, message):
        result = self.good_result()
        mutate(result)
        code, out = self.run_validator(tmp_path, result)
        assert code == 1
        assert message in out

    def test_non_json_fails(self, tmp_path):
        code, out = self.run_validator(tmp_path, "not json at all")
        assert code == 1
        assert "not valid JSON" in out
