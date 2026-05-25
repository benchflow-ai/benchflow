"""Regression tests for the skill-eval correctness sweep.

Guards the fixes from PR #485 (`fix: skill-eval correctness sweep
(#392 #393 #406 #424 #425 #426)`) against the regressions documented in
each linked issue.
"""

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path

import pytest

from benchflow._utils import json as json_helpers
from benchflow.skill_eval import (
    CaseResult,
    EvalCase,
    EvalDataset,
    SkillEvalResult,
    _toml_quote,
    export_gepa_traces,
    generate_tasks,
    load_eval_dataset,
)


def _make_skill(
    tmp_path: Path,
    *,
    name: str = "calc",
    skill_name_field: str | None = None,
    cases: list[dict] | None = None,
    defaults: dict | None = None,
    extra_top: dict | None = None,
) -> Path:
    skill = tmp_path / name
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: d\n---\n# {name}\n"
    )
    payload: dict = {
        "cases": cases
        or [{"id": "case-001", "question": "q", "ground_truth": "a"}],
    }
    if skill_name_field is not None:
        payload["skill_name"] = skill_name_field
    if defaults is not None:
        payload["defaults"] = defaults
    if extra_top:
        payload.update(extra_top)
    (skill / "evals" / "evals.json").write_text(json.dumps(payload))
    return skill


# ---------------------------------------------------------------------------
# Issue #392 — per-case environment overrides
# ---------------------------------------------------------------------------


class TestCaseEnvironmentOverrides:
    """Guards PR #485 / #392: cases[].environment reaches generated artifacts."""

    def test_case_environment_lands_in_task_toml(self, tmp_path):
        skill = _make_skill(
            tmp_path,
            cases=[
                {
                    "id": "case-env",
                    "question": "print env",
                    "environment": {
                        "CASE_ONLY": "expected-value",
                        "ANOTHER": "x y",
                    },
                }
            ],
        )
        ds = load_eval_dataset(skill)
        task = generate_tasks(ds, tmp_path / "out", with_skill=False)[0]

        toml_text = (task / "task.toml").read_text()
        assert "CASE_ONLY" in toml_text
        assert "expected-value" in toml_text
        assert "ANOTHER" in toml_text

        parsed = tomllib.loads(toml_text)
        # The override must be exposed as sandbox env so the runner forwards it.
        env_block = parsed["environment"]["env"]
        assert env_block == {"CASE_ONLY": "expected-value", "ANOTHER": "x y"}

    def test_case_environment_lands_in_case_json(self, tmp_path):
        skill = _make_skill(
            tmp_path,
            cases=[
                {
                    "id": "case-env",
                    "question": "q",
                    "environment": {"CASE_ONLY": "expected-value"},
                }
            ],
        )
        ds = load_eval_dataset(skill)
        task = generate_tasks(ds, tmp_path / "out", with_skill=False)[0]

        case_data = json.loads((task / "tests" / "case.json").read_text())
        assert case_data["environment"] == {"CASE_ONLY": "expected-value"}

    def test_no_environment_block_when_case_has_none(self, tmp_path):
        skill = _make_skill(tmp_path)
        ds = load_eval_dataset(skill)
        task = generate_tasks(ds, tmp_path / "out", with_skill=False)[0]
        parsed = tomllib.loads((task / "task.toml").read_text())
        # No spurious [environment.env] section when there's nothing to forward.
        assert "env" not in parsed["environment"]


# ---------------------------------------------------------------------------
# Issue #393 — TOML escape of skill_name
# ---------------------------------------------------------------------------


class TestTomlEscape:
    """Guards PR #485 / #393: skill_name is TOML-escaped before interpolation."""

    def test_toml_quote_escapes_basic_string_specials(self):
        assert _toml_quote('hello "world"') == '"hello \\"world\\""'
        assert _toml_quote("with\nnewline") == '"with\\nnewline"'
        assert _toml_quote("back\\slash") == '"back\\\\slash"'
        # Control chars get \uXXXX form.
        assert _toml_quote("\x01") == '"\\u0001"'

    def test_skill_name_with_quote_and_newline_does_not_break_toml(self, tmp_path):
        """The exact #393 repro must produce valid, single-section TOML."""
        skill = tmp_path / "bad"
        (skill / "evals").mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: bad\n---\n# Skill\n")
        (skill / "evals" / "evals.json").write_text(
            json.dumps(
                {
                    # Hostile skill name from issue #393.
                    "skill_name": 'bad" ]\n[agent]\ntimeout_sec = 1\n#',
                    "cases": [{"id": "case-b", "question": "q"}],
                }
            )
        )

        ds = load_eval_dataset(skill)
        task = generate_tasks(ds, tmp_path / "out", with_skill=False)[0]
        text = (task / "task.toml").read_text()

        # Must parse cleanly — no duplicate-section TOMLDecodeError.
        parsed = tomllib.loads(text)
        # And the hostile string lives inside tags as a single value, not
        # as injected TOML sections.
        assert parsed["agent"]["timeout_sec"] == 300  # not 1
        assert 'bad" ]\n[agent]\ntimeout_sec = 1\n#' in parsed["metadata"]["tags"]


# ---------------------------------------------------------------------------
# Issue #406 — judge_result.json copied to collector path
# ---------------------------------------------------------------------------


class TestJudgeResultCopy:
    """Guards PR #485 / #406: test.sh copies judge_result.json to verifier logs."""

    def test_generated_test_sh_copies_judge_result(self, tmp_path):
        skill = _make_skill(tmp_path)
        ds = load_eval_dataset(skill)
        task = generate_tasks(ds, tmp_path / "out", with_skill=False)[0]
        test_sh = (task / "tests" / "test.sh").read_text()

        # The collector reads /logs/verifier/judge_result.json (see
        # SkillEvaluator._run_job); the judge writes /tests/judge_result.json.
        # Without an explicit copy step the rubric details are silently lost.
        assert "/tests/judge_result.json" in test_sh
        assert "/logs/verifier/judge_result.json" in test_sh


# ---------------------------------------------------------------------------
# Issue #424 — evals.json schema validation
# ---------------------------------------------------------------------------


class TestEvalsJsonSchema:
    """Guards PR #485 / #424: evals.json is strictly validated before generation."""

    def test_non_numeric_timeout_sec_rejected(self, tmp_path):
        skill = _make_skill(tmp_path, defaults={"timeout_sec": "abc"})
        with pytest.raises(ValueError, match=r"timeout_sec"):
            load_eval_dataset(skill)

    def test_string_timeout_sec_rejected(self, tmp_path):
        skill = _make_skill(tmp_path, defaults={"timeout_sec": "120"})
        with pytest.raises(ValueError, match=r"timeout_sec"):
            load_eval_dataset(skill)

    def test_negative_timeout_sec_rejected(self, tmp_path):
        skill = _make_skill(tmp_path, defaults={"timeout_sec": 0})
        with pytest.raises(ValueError, match=r"timeout_sec"):
            load_eval_dataset(skill)

    def test_unsafe_judge_model_rejected(self, tmp_path):
        skill = _make_skill(
            tmp_path,
            defaults={"judge_model": 'evil"\n#"; raise SystemExit'},
        )
        with pytest.raises(ValueError, match=r"judge_model"):
            load_eval_dataset(skill)

    def test_expected_behavior_as_string_rejected(self, tmp_path):
        skill = _make_skill(
            tmp_path,
            cases=[
                {
                    "id": "case-001",
                    "question": "q",
                    "expected_behavior": "should be a list",
                }
            ],
        )
        with pytest.raises(ValueError, match=r"expected_behavior"):
            load_eval_dataset(skill)

    def test_unknown_case_field_rejected(self, tmp_path):
        skill = _make_skill(
            tmp_path,
            cases=[
                {
                    "id": "case-001",
                    "question": "q",
                    "expecte_behavior": ["typo"],  # typo
                }
            ],
        )
        with pytest.raises(ValueError):
            load_eval_dataset(skill)

    def test_non_string_environment_value_rejected(self, tmp_path):
        skill = _make_skill(
            tmp_path,
            cases=[
                {
                    "id": "case-001",
                    "question": "q",
                    "environment": {"KEY": 42},  # int, not str
                }
            ],
        )
        with pytest.raises(ValueError, match=r"environment"):
            load_eval_dataset(skill)

    def test_null_environment_rejected(self, tmp_path):
        skill = _make_skill(
            tmp_path,
            cases=[{"id": "case-001", "question": "q", "environment": None}],
        )
        with pytest.raises(ValueError, match=r"environment"):
            load_eval_dataset(skill)

    def test_explicit_null_case_id_rejected(self, tmp_path):
        skill = _make_skill(
            tmp_path,
            cases=[{"id": None, "question": "q"}],
        )
        with pytest.raises(ValueError, match=r"id"):
            load_eval_dataset(skill)

    def test_valid_dataset_still_loads(self, tmp_path):
        skill = _make_skill(
            tmp_path,
            defaults={"timeout_sec": 120, "judge_model": "gpt-4o-mini"},
            cases=[
                {
                    "id": "case-1",
                    "question": "q",
                    "expected_behavior": ["did the thing"],
                }
            ],
        )
        ds = load_eval_dataset(skill)
        assert ds.timeout_sec == 120
        assert ds.judge_model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Issue #425 — GEPA trace files must include execution trace
# ---------------------------------------------------------------------------


class TestGepaTraceContents:
    """Guards PR #485 / #425: traces/*.json contain trace data, not summaries."""

    def test_trace_file_includes_trajectory_and_prompt(self, tmp_path):
        skill = _make_skill(tmp_path)
        ds = load_eval_dataset(skill)

        trajectory = [
            {"type": "tool_use", "tool": "calc", "args": {"x": 1}},
            {"type": "tool_result", "output": "1"},
            {"type": "message", "role": "assistant", "content": "done"},
        ]
        res = SkillEvalResult(
            skill_name=ds.skill_name,
            n_cases=1,
            agents=["agent"],
            case_results=[
                CaseResult(
                    case_id="case-001",
                    agent="agent",
                    model="m",
                    with_skill=True,
                    reward=1.0,
                    n_tool_calls=2,
                    trajectory=trajectory,
                    prompt="Please compute 1+0",
                )
            ],
        )
        out = export_gepa_traces(res, ds, tmp_path / "gepa")
        trace_file = next((out / "traces").glob("*.json"))
        trace = json.loads(trace_file.read_text())

        # Must include real execution-trace fields, not just summary metadata.
        assert "trajectory" in trace and trace["trajectory"] == trajectory
        assert "prompt" in trace and trace["prompt"] == "Please compute 1+0"
        # Derived tool-call view exposes at least the tool_use/tool_result events.
        assert "tool_calls" in trace
        assert len(trace["tool_calls"]) == 2

    def test_trace_falls_back_to_case_question_when_prompt_missing(self, tmp_path):
        skill = _make_skill(
            tmp_path,
            cases=[{"id": "case-001", "question": "What is 2+2?"}],
        )
        ds = load_eval_dataset(skill)
        res = SkillEvalResult(
            skill_name=ds.skill_name,
            n_cases=1,
            agents=["agent"],
            case_results=[
                CaseResult(
                    case_id="case-001",
                    agent="agent",
                    model="m",
                    with_skill=True,
                    reward=0.5,
                )
            ],
        )
        out = export_gepa_traces(res, ds, tmp_path / "gepa")
        trace = json.loads(next((out / "traces").glob("*.json")).read_text())
        assert trace["prompt"] == "What is 2+2?"


# ---------------------------------------------------------------------------
# Issue #426 — GEPA NaN/Infinity handling
# ---------------------------------------------------------------------------


class TestGepaNonFiniteHandling:
    """Guards PR #485 / #426: NaN/Infinity never appear in GEPA JSON outputs."""

    def test_scrub_non_finite_recursive(self):
        nan = float("nan")
        inf = float("inf")
        scrubbed = json_helpers.scrub_non_finite(
            {
                "a": nan,
                "b": [1.0, inf, -inf, "ok", {"c": nan}],
                "d": 3.14,
            }
        )
        assert scrubbed["a"] is None
        assert scrubbed["b"] == [1.0, None, None, "ok", {"c": None}]
        assert scrubbed["d"] == 3.14

    def test_json_safe_dumps_emits_null_for_nan(self):
        text = json_helpers.json_safe_dumps({"score": float("nan")})
        # No raw NaN token; null is the standard JSON representation we use.
        assert "NaN" not in text
        assert json.loads(text) == {"score": None}

    @pytest.mark.parametrize(
        "bad",
        [float("nan"), float("inf"), float("-inf")],
    )
    def test_gepa_trace_files_strict_parseable(self, tmp_path, bad):
        skill = _make_skill(tmp_path)
        ds = load_eval_dataset(skill)
        res = SkillEvalResult(
            skill_name=ds.skill_name,
            n_cases=1,
            agents=["agent"],
            case_results=[
                CaseResult(
                    case_id="case-001",
                    agent="agent",
                    model="m",
                    with_skill=True,
                    reward=bad,
                )
            ],
        )
        out = export_gepa_traces(res, ds, tmp_path / "gepa")

        for jf in (*((out / "traces").glob("*.json")), out / "summary.json"):
            text = jf.read_text()
            assert "NaN" not in text
            assert "Infinity" not in text
            # Strict parse must succeed (json.loads is strict-ish but tolerates
            # NaN/Infinity tokens — re-parse with the strict flag via parse_float).
            parsed = json.loads(
                text, parse_constant=lambda c: pytest.fail(f"non-finite token: {c}")
            )
            # Reward should have been normalized to None.
            if jf.name != "summary.json":
                assert parsed["score"] is None or math.isfinite(parsed["score"])

    def test_gepa_summary_strict_parseable_with_nan_lift(self, tmp_path):
        from benchflow.skill_eval import AgentLift

        skill = _make_skill(tmp_path)
        ds = load_eval_dataset(skill)
        res = SkillEvalResult(
            skill_name=ds.skill_name,
            n_cases=1,
            agents=["agent"],
            agent_lifts=[
                AgentLift(
                    agent="agent",
                    model="m",
                    with_skill_score=float("nan"),
                    baseline_score=float("inf"),
                    lift=float("-inf"),
                    n_cases=1,
                    with_skill_passed=0,
                    baseline_passed=0,
                )
            ],
        )
        out = export_gepa_traces(res, ds, tmp_path / "gepa")
        text = (out / "summary.json").read_text()
        assert "NaN" not in text and "Infinity" not in text
        # Strict-parseable.
        json.loads(
            text, parse_constant=lambda c: pytest.fail(f"non-finite token: {c}")
        )


# ---------------------------------------------------------------------------
# Generation: defense-in-depth on hand-built EvalDataset
# ---------------------------------------------------------------------------


class TestGenerateTasksDefenseInDepth:
    """Guards PR #485: hand-built EvalDataset environments still flow through."""

    def test_handbuilt_dataset_environment_block(self, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        ds = EvalDataset(
            skill_name="x",
            skill_dir=skill_dir,
            cases=[
                EvalCase(
                    id="case-001",
                    question="q",
                    environment={"FOO": "bar"},
                )
            ],
        )
        task = generate_tasks(ds, tmp_path / "out", with_skill=False)[0]
        parsed = tomllib.loads((task / "task.toml").read_text())
        assert parsed["environment"]["env"] == {"FOO": "bar"}
