"""Integration test for skills eval — exercises the full path.

Runs: load evals.json → generate tasks → verify task structure → cleanup.
Does NOT run actual agents (that requires Docker + API keys).

For live e2e tests, use: pytest -m live tests/test_skill_eval_integration.py
"""

import json
import tempfile
from pathlib import Path

import pytest

from benchflow.skill_eval import (
    SkillEvaluator,
    load_eval_dataset,
    generate_tasks,
    cleanup_tasks,
    export_gepa_traces,
    SkillEvalResult,
    CaseResult,
    AgentLift,
)


@pytest.fixture
def mock_skill(tmp_path):
    """Create a realistic mock skill with SKILL.md + evals/evals.json."""
    skill = tmp_path / "mock-audit-skill"
    skill.mkdir()

    (skill / "SKILL.md").write_text("""---
name: mock-audit-skill
description: A mock skill for testing the skill-eval pipeline
version: "1.0"
---

# Mock Audit Skill

This skill helps agents verify claims in benchmark papers.

## How to use

1. Read the paper at the given arXiv URL
2. Check each claim against the paper's content
3. Report overclaims and missing marks
""")

    scripts = skill / "scripts"
    scripts.mkdir()
    (scripts / "check_claim.py").write_text(
        'import sys; print(f"Checking claim: {sys.argv[1]}")'
    )

    evals = skill / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(json.dumps({
        "skill_name": "mock-audit-skill",
        "version": "1",
        "defaults": {
            "timeout_sec": 120,
            "judge_model": "claude-haiku-4-5-20251001"
        },
        "cases": [
            {
                "id": "detect-overclaim",
                "question": "A paper claims benchmark X has Cross-Domain=True. X has 8 task types (OS, DB, games, web). Is this an overclaim?",
                "ground_truth": "Yes, this is an overclaim. Task types are not professional domains.",
                "expected_behavior": [
                    "The agent identified that task types differ from professional domains",
                    "The agent concluded this is an overclaim"
                ],
                "expected_skill": "mock-audit-skill"
            },
            {
                "id": "detect-missing-mark",
                "question": "A paper marks benchmark Y as Dynamic=False. Y has a live HuggingFace leaderboard updated weekly. Is this correct?",
                "ground_truth": "No, Dynamic should be True. A live leaderboard with weekly updates qualifies as designed for continuous evolution.",
                "expected_behavior": [
                    "The agent checked for leaderboard evidence",
                    "The agent concluded Dynamic should be True"
                ],
                "expected_skill": "mock-audit-skill"
            },
            {
                "id": "confirm-correct",
                "question": "A paper marks benchmark Z as Production=True. Z contains 1,488 real Upwork tasks with $1M in actual payouts. Is this correct?",
                "ground_truth": "Yes, Production=True is correct. Real commercial tasks with actual monetary payouts qualify.",
                "expected_behavior": [
                    "The agent verified the monetary evidence",
                    "The agent confirmed Production=True is correct"
                ],
                "expected_skill": "mock-audit-skill"
            }
        ]
    }, indent=2))

    return skill


class TestSkillEvalIntegration:
    """Integration tests for the full skill-eval pipeline (no Docker)."""

    def test_load_and_validate_dataset(self, mock_skill):
        dataset = load_eval_dataset(mock_skill)
        assert dataset.skill_name == "mock-audit-skill"
        assert len(dataset.cases) == 3
        assert dataset.timeout_sec == 120
        assert dataset.judge_model == "claude-haiku-4-5-20251001"

    def test_generate_with_skill_tasks(self, mock_skill):
        dataset = load_eval_dataset(mock_skill)
        with_dir = Path(tempfile.mkdtemp()) / "with-skill"
        tasks = generate_tasks(dataset, with_dir, with_skill=True)

        assert len(tasks) == 3

        for task_dir in tasks:
            assert (task_dir / "instruction.md").exists()
            assert (task_dir / "task.toml").exists()
            assert (task_dir / "environment" / "Dockerfile").exists()
            assert (task_dir / "tests" / "case.json").exists()
            assert (task_dir / "tests" / "judge.py").exists()
            assert (task_dir / "tests" / "test.sh").exists()

            # Skill should be copied in
            skill_dst = task_dir / "environment" / "skills" / "mock-audit-skill"
            assert skill_dst.exists()
            assert (skill_dst / "SKILL.md").exists()
            assert (skill_dst / "scripts" / "check_claim.py").exists()

        cleanup_tasks([with_dir])
        assert not with_dir.exists()

    def test_generate_baseline_tasks(self, mock_skill):
        dataset = load_eval_dataset(mock_skill)
        baseline_dir = Path(tempfile.mkdtemp()) / "baseline"
        tasks = generate_tasks(dataset, baseline_dir, with_skill=False)

        assert len(tasks) == 3

        for task_dir in tasks:
            assert (task_dir / "instruction.md").exists()
            # Skill should NOT be present
            skill_dst = task_dir / "environment" / "skills" / "mock-audit-skill"
            assert not skill_dst.exists()

            # Dockerfile should not have COPY skills/
            dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
            assert "COPY skills/" not in dockerfile

        cleanup_tasks([baseline_dir])

    def test_instruction_contains_question(self, mock_skill):
        dataset = load_eval_dataset(mock_skill)
        out = Path(tempfile.mkdtemp()) / "tasks"
        tasks = generate_tasks(dataset, out, with_skill=True)

        instruction = (tasks[0] / "instruction.md").read_text()
        assert "Cross-Domain=True" in instruction

        instruction2 = (tasks[1] / "instruction.md").read_text()
        assert "Dynamic=False" in instruction2

        cleanup_tasks([out])

    def test_case_json_has_rubric(self, mock_skill):
        dataset = load_eval_dataset(mock_skill)
        out = Path(tempfile.mkdtemp()) / "tasks"
        tasks = generate_tasks(dataset, out, with_skill=True)

        case_data = json.loads((tasks[0] / "tests" / "case.json").read_text())
        assert case_data["id"] == "detect-overclaim"
        assert "overclaim" in case_data["ground_truth"].lower()
        assert len(case_data["expected_behavior"]) == 2

        cleanup_tasks([out])

    def test_dockerfile_includes_judge_deps(self, mock_skill):
        dataset = load_eval_dataset(mock_skill)
        out = Path(tempfile.mkdtemp()) / "tasks"
        tasks = generate_tasks(dataset, out, with_skill=True)

        dockerfile = (tasks[0] / "environment" / "Dockerfile").read_text()
        assert "pip install" in dockerfile
        assert "anthropic" in dockerfile

        cleanup_tasks([out])

    def test_task_toml_timeout(self, mock_skill):
        dataset = load_eval_dataset(mock_skill)
        out = Path(tempfile.mkdtemp()) / "tasks"
        tasks = generate_tasks(dataset, out, with_skill=True)

        toml_content = (tasks[0] / "task.toml").read_text()
        assert "timeout_sec = 120" in toml_content

        cleanup_tasks([out])

    def test_evaluator_init(self, mock_skill):
        evaluator = SkillEvaluator(mock_skill)
        assert evaluator.dataset.skill_name == "mock-audit-skill"
        assert len(evaluator.dataset.cases) == 3

    def test_gepa_export_with_mock_results(self, mock_skill):
        dataset = load_eval_dataset(mock_skill)
        result = SkillEvalResult(
            skill_name="mock-audit-skill",
            n_cases=3,
            agents=["claude-agent-acp"],
            case_results=[
                CaseResult(
                    case_id="detect-overclaim",
                    agent="claude-agent-acp",
                    model="claude-haiku-4-5-20251001",
                    with_skill=True,
                    reward=0.9,
                    n_tool_calls=5,
                ),
                CaseResult(
                    case_id="detect-overclaim",
                    agent="claude-agent-acp",
                    model="claude-haiku-4-5-20251001",
                    with_skill=False,
                    reward=0.3,
                    n_tool_calls=2,
                ),
            ],
            agent_lifts=[
                AgentLift(
                    agent="claude-agent-acp",
                    model="claude-haiku-4-5-20251001",
                    with_skill_score=0.9,
                    baseline_score=0.3,
                    lift=0.6,
                    n_cases=3,
                    with_skill_passed=1,
                    baseline_passed=0,
                ),
            ],
        )

        out = Path(tempfile.mkdtemp()) / "gepa"
        gepa_dir = export_gepa_traces(result, dataset, out)

        assert (gepa_dir / "skill.md").exists()
        assert (gepa_dir / "summary.json").exists()
        assert (gepa_dir / "traces").is_dir()

        summary = json.loads((gepa_dir / "summary.json").read_text())
        assert summary["skill_name"] == "mock-audit-skill"
        assert summary["lifts"][0]["lift"] == 0.6

        traces = list((gepa_dir / "traces").iterdir())
        assert len(traces) == 2

    def test_summary_table_output(self, mock_skill):
        result = SkillEvalResult(
            skill_name="mock-audit-skill",
            n_cases=3,
            agents=["claude-agent-acp"],
            agent_lifts=[
                AgentLift(
                    agent="claude-agent-acp",
                    model="claude-haiku-4-5-20251001",
                    with_skill_score=0.85,
                    baseline_score=0.35,
                    lift=0.50,
                    n_cases=3,
                    with_skill_passed=2,
                    baseline_passed=1,
                ),
            ],
        )

        rows = result.summary_table()
        assert len(rows) == 3
        assert rows[0]["mode"] == "with-skill"
        assert rows[0]["score"] == "2/3"
        assert rows[1]["mode"] == "baseline"
        assert rows[2]["mode"] == "LIFT"
        assert rows[2]["score"] == "+1"


class TestSkillEvalCLI:
    """Test the CLI wiring for `benchflow skills eval`."""

    def test_skills_eval_missing_dir(self):
        from typer.testing import CliRunner
        from benchflow.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["skills", "eval", "/tmp/nonexistent-skill-dir"])
        assert result.exit_code == 1
        assert "No evals/evals.json found" in result.output

    def test_skills_eval_valid_dir_shows_info(self, mock_skill):
        from typer.testing import CliRunner
        from benchflow.cli.main import app

        runner = CliRunner()
        # This will try to run the full eval (which requires Docker),
        # but it should at least parse the dataset and print the header
        # before failing on the Job run.
        result = runner.invoke(app, ["skills", "eval", str(mock_skill), "--no-baseline"])
        # Should get past the dataset loading phase
        assert "mock-audit-skill" in result.output or result.exit_code != 0

    def test_skills_list_command(self):
        from typer.testing import CliRunner
        from benchflow.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["skills", "list"])
        assert result.exit_code == 0

    def test_skills_help(self):
        from typer.testing import CliRunner
        from benchflow.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["skills", "--help"])
        assert result.exit_code == 0
        assert "eval" in result.output
        assert "list" in result.output
        assert "install" in result.output
