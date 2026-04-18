"""Tests for benchflow.skill_eval — dataset loading, task generation, comparison."""

import json

import pytest

from benchflow.skill_eval import (
    AgentLift,
    CaseResult,
    SkillEvalResult,
    cleanup_tasks,
    export_gepa_traces,
    generate_tasks,
    load_eval_dataset,
)


@pytest.fixture
def skill_dir(tmp_path):
    """Create a minimal skill directory with evals."""
    skill = tmp_path / "calculator"
    skill.mkdir()

    # SKILL.md
    (skill / "SKILL.md").write_text("""---
name: calculator
description: A calculator skill
version: "1.0"
---

# Calculator

Use calc.py to compute math expressions.
""")

    # scripts
    scripts = skill / "scripts"
    scripts.mkdir()
    (scripts / "calc.py").write_text(
        "import sys; print(sum(int(x) for x in sys.argv[1:]))"
    )

    # evals
    evals = skill / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(
        json.dumps(
            {
                "version": "1",
                "skill_name": "calculator",
                "defaults": {
                    "timeout_sec": 120,
                    "judge_model": "claude-haiku-4-5-20251001",
                },
                "cases": [
                    {
                        "id": "calc-001",
                        "question": "What is 2 + 3 * 4? Use the calculator skill.",
                        "ground_truth": "14",
                        "expected_behavior": [
                            "Agent read the calculator SKILL.md",
                            "Agent executed calc.py with '2 + 3 * 4'",
                            "Agent reported correct result of 14",
                        ],
                        "expected_skill": "calculator",
                        "expected_script": "calc.py",
                    },
                    {
                        "id": "calc-002",
                        "question": "What is sqrt(144)?",
                        "ground_truth": "12",
                        "expected_behavior": [
                            "Agent used the calculator skill",
                            "Agent reported 12",
                        ],
                    },
                ],
            }
        )
    )

    return skill


@pytest.fixture
def minimal_skill_dir(tmp_path):
    """Skill directory with minimal evals (no SKILL.md, no expected_behavior)."""
    skill = tmp_path / "simple"
    skill.mkdir()
    evals = skill / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(
        json.dumps(
            {
                "cases": [
                    {"id": "s-001", "question": "Say hello", "ground_truth": "hello"},
                ],
            }
        )
    )
    return skill


# ---------------------------------------------------------------------------
# load_eval_dataset
# ---------------------------------------------------------------------------


class TestLoadEvalDataset:
    def test_loads_valid_dataset(self, skill_dir):
        ds = load_eval_dataset(skill_dir)
        assert ds.skill_name == "calculator"
        assert len(ds.cases) == 2
        assert ds.judge_model == "gemini-2.0-flash"
        assert ds.timeout_sec == 120

    def test_parses_cases(self, skill_dir):
        ds = load_eval_dataset(skill_dir)
        case = ds.cases[0]
        assert case.id == "calc-001"
        assert "2 + 3 * 4" in case.question
        assert case.ground_truth == "14"
        assert len(case.expected_behavior) == 3
        assert case.expected_skill == "calculator"
        assert case.expected_script == "calc.py"

    def test_minimal_dataset(self, minimal_skill_dir):
        ds = load_eval_dataset(minimal_skill_dir)
        assert ds.skill_name == "simple"
        assert len(ds.cases) == 1
        assert ds.cases[0].expected_behavior == []

    def test_missing_evals_json(self, tmp_path):
        skill = tmp_path / "no-evals"
        skill.mkdir()
        with pytest.raises(FileNotFoundError, match="evals.json"):
            load_eval_dataset(skill)

    def test_empty_cases(self, tmp_path):
        skill = tmp_path / "empty"
        skill.mkdir()
        evals = skill / "evals"
        evals.mkdir()
        (evals / "evals.json").write_text(json.dumps({"cases": []}))
        with pytest.raises(ValueError, match="empty"):
            load_eval_dataset(skill)

    def test_missing_question(self, tmp_path):
        skill = tmp_path / "bad"
        skill.mkdir()
        evals = skill / "evals"
        evals.mkdir()
        (evals / "evals.json").write_text(
            json.dumps(
                {
                    "cases": [{"id": "x", "ground_truth": "y"}],
                }
            )
        )
        with pytest.raises(ValueError, match="question"):
            load_eval_dataset(skill)

    def test_duplicate_ids(self, tmp_path):
        skill = tmp_path / "dup"
        skill.mkdir()
        evals = skill / "evals"
        evals.mkdir()
        (evals / "evals.json").write_text(
            json.dumps(
                {
                    "cases": [
                        {"id": "same", "question": "a"},
                        {"id": "same", "question": "b"},
                    ],
                }
            )
        )
        with pytest.raises(ValueError, match="Duplicate"):
            load_eval_dataset(skill)

    def test_auto_generated_ids(self, tmp_path):
        skill = tmp_path / "noid"
        skill.mkdir()
        evals = skill / "evals"
        evals.mkdir()
        (evals / "evals.json").write_text(
            json.dumps(
                {
                    "cases": [
                        {"question": "first"},
                        {"question": "second"},
                    ],
                }
            )
        )
        ds = load_eval_dataset(skill)
        assert ds.cases[0].id == "case-000"
        assert ds.cases[1].id == "case-001"

    def test_defaults_fallback(self, minimal_skill_dir):
        ds = load_eval_dataset(minimal_skill_dir)
        assert ds.judge_model == "gemini-2.0-flash"
        assert ds.timeout_sec == 300


# ---------------------------------------------------------------------------
# generate_tasks
# ---------------------------------------------------------------------------


class TestGenerateTasks:
    def test_generates_correct_structure(self, skill_dir, tmp_path):
        ds = load_eval_dataset(skill_dir)
        output = tmp_path / "generated"
        task_dirs = generate_tasks(ds, output, with_skill=True)

        assert len(task_dirs) == 2
        for td in task_dirs:
            assert (td / "instruction.md").exists()
            assert (td / "task.toml").exists()
            assert (td / "environment" / "Dockerfile").exists()
            assert (td / "tests" / "test.sh").exists()
            assert (td / "tests" / "judge.py").exists()
            assert (td / "tests" / "case.json").exists()

    def test_with_skill_copies_skill(self, skill_dir, tmp_path):
        ds = load_eval_dataset(skill_dir)
        output = tmp_path / "with"
        task_dirs = generate_tasks(ds, output, with_skill=True)

        skills_dir = task_dirs[0] / "environment" / "skills" / "calculator"
        assert skills_dir.exists()
        assert (skills_dir / "SKILL.md").exists()
        assert (skills_dir / "scripts" / "calc.py").exists()
        # evals should NOT be copied
        assert not (skills_dir / "evals").exists()

    def test_without_skill_no_copy(self, skill_dir, tmp_path):
        ds = load_eval_dataset(skill_dir)
        output = tmp_path / "without"
        task_dirs = generate_tasks(ds, output, with_skill=False)

        skills_dir = task_dirs[0] / "environment" / "skills"
        assert not skills_dir.exists()

    def test_instruction_contains_question(self, skill_dir, tmp_path):
        ds = load_eval_dataset(skill_dir)
        output = tmp_path / "gen"
        task_dirs = generate_tasks(ds, output)

        instr = (task_dirs[0] / "instruction.md").read_text()
        assert "2 + 3 * 4" in instr

    def test_case_json_injected(self, skill_dir, tmp_path):
        ds = load_eval_dataset(skill_dir)
        output = tmp_path / "gen"
        task_dirs = generate_tasks(ds, output)

        case_data = json.loads((task_dirs[0] / "tests" / "case.json").read_text())
        assert case_data["id"] == "calc-001"
        assert case_data["ground_truth"] == "14"
        assert len(case_data["expected_behavior"]) == 3

    def test_test_sh_executable(self, skill_dir, tmp_path):
        ds = load_eval_dataset(skill_dir)
        output = tmp_path / "gen"
        task_dirs = generate_tasks(ds, output)

        test_sh = task_dirs[0] / "tests" / "test.sh"
        assert test_sh.stat().st_mode & 0o111  # executable

    def test_task_toml_has_timeout(self, skill_dir, tmp_path):
        ds = load_eval_dataset(skill_dir)
        output = tmp_path / "gen"
        task_dirs = generate_tasks(ds, output)

        toml_text = (task_dirs[0] / "task.toml").read_text()
        assert "timeout_sec = 120" in toml_text

    def test_dockerfile_with_skill_has_copy(self, skill_dir, tmp_path):
        ds = load_eval_dataset(skill_dir)
        output = tmp_path / "gen"
        task_dirs = generate_tasks(ds, output, with_skill=True)

        dockerfile = (task_dirs[0] / "environment" / "Dockerfile").read_text()
        assert "COPY skills/" in dockerfile

    def test_dockerfile_without_skill_no_copy(self, skill_dir, tmp_path):
        ds = load_eval_dataset(skill_dir)
        output = tmp_path / "gen"
        task_dirs = generate_tasks(ds, output, with_skill=False)

        dockerfile = (task_dirs[0] / "environment" / "Dockerfile").read_text()
        assert "COPY skills/" not in dockerfile


# ---------------------------------------------------------------------------
# cleanup_tasks
# ---------------------------------------------------------------------------


class TestCleanupTasks:
    def test_removes_directories(self, tmp_path):
        dirs = [tmp_path / "a", tmp_path / "b"]
        for d in dirs:
            d.mkdir()
            (d / "file.txt").write_text("test")

        cleanup_tasks(dirs)
        assert not dirs[0].exists()
        assert not dirs[1].exists()

    def test_handles_missing_dirs(self, tmp_path):
        cleanup_tasks([tmp_path / "nonexistent"])  # should not raise


# ---------------------------------------------------------------------------
# SkillEvalResult
# ---------------------------------------------------------------------------


class TestSkillEvalResult:
    def test_summary_table(self):
        result = SkillEvalResult(
            skill_name="test",
            n_cases=3,
            agents=["agent-a"],
            agent_lifts=[
                AgentLift(
                    agent="agent-a",
                    model="model-a",
                    with_skill_score=0.8,
                    baseline_score=0.3,
                    lift=0.5,
                    n_cases=3,
                    with_skill_passed=2,
                    baseline_passed=1,
                ),
            ],
        )
        rows = result.summary_table()
        assert len(rows) == 3  # with-skill, baseline, LIFT
        assert rows[0]["mode"] == "with-skill"
        assert rows[1]["mode"] == "baseline"
        assert rows[2]["mode"] == "LIFT"
        assert rows[2]["avg_reward"] == "+0.50"


# ---------------------------------------------------------------------------
# GEPA export
# ---------------------------------------------------------------------------


class TestGepaExport:
    def test_exports_structure(self, skill_dir, tmp_path):
        ds = load_eval_dataset(skill_dir)
        result = SkillEvalResult(
            skill_name="calculator",
            n_cases=2,
            agents=["agent-a"],
            case_results=[
                CaseResult(
                    case_id="calc-001",
                    agent="agent-a",
                    model="m",
                    with_skill=True,
                    reward=0.9,
                ),
                CaseResult(
                    case_id="calc-001",
                    agent="agent-a",
                    model="m",
                    with_skill=False,
                    reward=0.3,
                ),
            ],
            agent_lifts=[
                AgentLift(
                    agent="agent-a",
                    model="m",
                    with_skill_score=0.9,
                    baseline_score=0.3,
                    lift=0.6,
                    n_cases=2,
                    with_skill_passed=1,
                    baseline_passed=0,
                ),
            ],
        )

        out = tmp_path / "gepa"
        export_gepa_traces(result, ds, out)

        assert (out / "skill.md").exists()
        assert (out / "summary.json").exists()
        assert (out / "traces").is_dir()

        traces = list((out / "traces").glob("*.json"))
        assert len(traces) == 2

        summary = json.loads((out / "summary.json").read_text())
        assert summary["skill_name"] == "calculator"
        assert len(summary["lifts"]) == 1
        assert summary["lifts"][0]["lift"] == 0.6
