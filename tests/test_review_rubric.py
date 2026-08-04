"""Unit tests for the rubric contract: schema, prompts, and wrapper assembly."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow.review.config import (
    DEFAULT_RUBRIC_PATH,
    REVIEW_RESULT_FILENAME,
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

    def test_instruction_requires_conjunctive_returned_evidence(self):
        """Guards PR #942: related searches cannot be promoted to proof."""

        rubric = Rubric.model_validate(RUBRIC)
        instruction = render_review_instruction(rubric)

        assert "A pass requires every stated condition" in instruction
        assert "command title, search attempt" in instruction
        assert "exact recorded observation" in instruction

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
        names = json.loads((dest / "tests" / "criteria.json").read_text("utf-8"))
        assert names == ["method_soundness", "output_contract"]
        assert uploads == {
            str(dest / "evidence" / "trial"): TRIAL_MOUNT,
            str(dest / "evidence" / "task"): TASK_MOUNT,
        }

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

    def run_validator(self, tmp_path: Path, result: dict | str) -> tuple[int, str]:
        rubric = Rubric.model_validate(RUBRIC)
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

    def test_valid_result_passes(self, tmp_path):
        code, out = self.run_validator(tmp_path, self.good_result())
        assert code == 0, out

    def test_not_applicable_is_valid(self, tmp_path):
        result = self.good_result()
        result["checks"]["method_soundness"]["outcome"] = "not_applicable"
        code, _ = self.run_validator(tmp_path, result)
        assert code == 0

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
