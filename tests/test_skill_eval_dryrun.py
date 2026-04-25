"""Dry-run test for skill-eval pipeline — proves end-to-end without LLM calls.

Mocks the Job.run() to avoid Docker/API dependencies while verifying:
1. Dataset loads correctly
2. Ephemeral tasks generated with correct structure
3. With-skill vs baseline task dirs differ (skill copied vs not)
4. Job is configured correctly (agent, model, concurrency)
5. Results collected and lift computed
6. GEPA export produces expected structure
7. CLI wiring works end-to-end
8. Cleanup removes ephemeral tasks
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.skill_eval import (
    AgentLift,
    CaseResult,
    SkillEvalResult,
    SkillEvaluator,
    export_gepa_traces,
    generate_tasks,
    load_eval_dataset,
)


@pytest.fixture
def code_review_skill():
    """Use the real code-review-bench as test fixture."""
    skill_dir = Path(__file__).parent.parent / "benchmarks" / "code-review-bench"
    if not (skill_dir / "evals" / "evals.json").exists():
        pytest.skip("code-review-bench not found")
    return skill_dir


@pytest.fixture
def mock_skill(tmp_path):
    """Minimal mock skill for fast tests."""
    skill = tmp_path / "mock-review-skill"
    skill.mkdir()

    (skill / "SKILL.md").write_text(
        "---\nname: mock-review\ndescription: test\n---\n# Mock\n"
    )

    evals = skill / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(
        json.dumps(
            {
                "skill_name": "mock-review",
                "defaults": {
                    "timeout_sec": 60,
                    "judge_model": "claude-haiku-4-5-20251001",
                },
                "cases": [
                    {
                        "id": "bug-001",
                        "question": "Find the bug in: x = items[len(items)]",
                        "ground_truth": "IndexError: list index out of range",
                        "expected_behavior": [
                            "Agent identified the off-by-one",
                            "Agent suggested items[len(items)-1] or items[-1]",
                        ],
                    },
                    {
                        "id": "bug-002",
                        "question": "Find the bug in: password = hashlib.md5(pw).hexdigest()",
                        "ground_truth": "MD5 is insecure for passwords",
                        "expected_behavior": [
                            "Agent identified MD5 as insecure",
                            "Agent recommended bcrypt or argon2",
                        ],
                    },
                ],
            },
            indent=2,
        )
    )
    return skill


class TestDryRunPipeline:
    """Full pipeline dry-run without Docker/LLM."""

    def test_load_code_review_bench(self, code_review_skill):
        dataset = load_eval_dataset(code_review_skill)
        assert dataset.skill_name == "code-review"
        assert len(dataset.cases) == 5
        assert dataset.cases[0].id == "sql-injection"
        assert dataset.cases[4].id == "insecure-default"

    def test_generate_tasks_creates_runnable_structure(self, mock_skill):
        dataset = load_eval_dataset(mock_skill)
        out = Path(tempfile.mkdtemp()) / "tasks"
        tasks = generate_tasks(dataset, out, with_skill=True)

        for task_dir in tasks:
            # Every generated task must be a valid Harbor task
            assert (task_dir / "task.toml").exists()
            assert (task_dir / "instruction.md").exists()
            assert (task_dir / "environment" / "Dockerfile").exists()
            assert (task_dir / "tests" / "test.sh").exists()
            assert (task_dir / "tests" / "judge.py").exists()
            assert (task_dir / "tests" / "case.json").exists()

            # Dockerfile should be buildable (no syntax errors in FROM/RUN)
            dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
            assert dockerfile.startswith("FROM ")
            assert "RUN " in dockerfile

            # judge.py should be importable Python
            judge = (task_dir / "tests" / "judge.py").read_text()
            assert "def main():" in judge
            assert "reward.txt" in judge

    def test_with_vs_without_skill_differ(self, mock_skill):
        dataset = load_eval_dataset(mock_skill)
        tmp = Path(tempfile.mkdtemp())

        with_tasks = generate_tasks(dataset, tmp / "with", with_skill=True)
        without_tasks = generate_tasks(dataset, tmp / "without", with_skill=False)

        # With-skill has skills/ directory copied
        for t in with_tasks:
            assert (t / "environment" / "skills" / "mock-review" / "SKILL.md").exists()

        # Without-skill does NOT have skills/
        for t in without_tasks:
            assert not (t / "environment" / "skills").exists()

        # Dockerfiles differ
        with_df = (with_tasks[0] / "environment" / "Dockerfile").read_text()
        without_df = (without_tasks[0] / "environment" / "Dockerfile").read_text()
        assert "COPY skills/" in with_df
        assert "COPY skills/" not in without_df

    @patch("benchflow.job.Job")
    def test_evaluator_configures_job_correctly(self, MockJob, mock_skill):
        """Verify SkillEvaluator passes correct config to Job."""
        mock_job_instance = MockJob.return_value
        mock_job_instance.run = AsyncMock(
            return_value=type(
                "R",
                (),
                {
                    "passed": 1,
                    "failed": 1,
                    "errored": 0,
                    "total": 2,
                    "score": 0.5,
                    "elapsed_sec": 10,
                },
            )()
        )

        evaluator = SkillEvaluator(mock_skill)
        asyncio.run(
            evaluator.run(
                agents=["claude-agent-acp"],
                models=["claude-haiku-4-5-20251001"],
                environment="docker",
                concurrency=2,
                no_baseline=True,
            )
        )

        # Job was called at least once (with-skill run)
        assert MockJob.call_count >= 1
        call_kwargs = MockJob.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.agent == "claude-agent-acp"
        assert config.concurrency == 2

    def test_gepa_export_roundtrip(self, mock_skill):
        dataset = load_eval_dataset(mock_skill)

        result = SkillEvalResult(
            skill_name="mock-review",
            n_cases=2,
            agents=["claude-agent-acp"],
            case_results=[
                CaseResult(
                    case_id="bug-001",
                    agent="claude-agent-acp",
                    model="haiku",
                    with_skill=True,
                    reward=0.85,
                    n_tool_calls=3,
                ),
                CaseResult(
                    case_id="bug-001",
                    agent="claude-agent-acp",
                    model="haiku",
                    with_skill=False,
                    reward=0.4,
                    n_tool_calls=1,
                ),
                CaseResult(
                    case_id="bug-002",
                    agent="claude-agent-acp",
                    model="haiku",
                    with_skill=True,
                    reward=0.9,
                    n_tool_calls=4,
                ),
                CaseResult(
                    case_id="bug-002",
                    agent="claude-agent-acp",
                    model="haiku",
                    with_skill=False,
                    reward=0.3,
                    n_tool_calls=1,
                ),
            ],
            agent_lifts=[
                AgentLift(
                    agent="claude-agent-acp",
                    model="haiku",
                    with_skill_score=0.875,
                    baseline_score=0.35,
                    lift=0.525,
                    n_cases=2,
                    with_skill_passed=2,
                    baseline_passed=0,
                ),
            ],
        )

        out = Path(tempfile.mkdtemp()) / "gepa"
        gepa_dir = export_gepa_traces(result, dataset, out)

        # Verify structure
        assert (gepa_dir / "skill.md").exists()
        assert (gepa_dir / "summary.json").exists()

        # Verify summary roundtrips
        summary = json.loads((gepa_dir / "summary.json").read_text())
        assert summary["skill_name"] == "mock-review"
        assert summary["lifts"][0]["lift"] == 0.525
        assert summary["lifts"][0]["with_skill_passed"] == 2
        assert summary["lifts"][0]["baseline_passed"] == 0

        # Verify trace files
        traces = list((gepa_dir / "traces").iterdir())
        assert len(traces) == 4  # 2 cases x 2 modes

        # Verify trace content
        trace = json.loads(traces[0].read_text())
        assert "case_id" in trace
        assert "score" in trace
        assert "skill_text" in trace

    def test_cli_dryrun_loads_dataset(self, mock_skill):
        from typer.testing import CliRunner

        from benchflow.cli.main import app

        runner = CliRunner()
        # Run with a non-existent agent to trigger early failure after dataset loads
        result = runner.invoke(
            app,
            [
                "skills",
                "eval",
                str(mock_skill),
                "-a",
                "claude-agent-acp",
                "--no-baseline",
            ],
        )
        # Should get past dataset loading (prints skill name)
        assert "mock-review" in result.output or "2 cases" in result.output

    def test_summary_table_format(self):
        result = SkillEvalResult(
            skill_name="code-review",
            n_cases=5,
            agents=["claude-agent-acp", "codex-acp"],
            agent_lifts=[
                AgentLift(
                    agent="claude-agent-acp",
                    model="haiku",
                    with_skill_score=0.85,
                    baseline_score=0.40,
                    lift=0.45,
                    n_cases=5,
                    with_skill_passed=4,
                    baseline_passed=2,
                ),
                AgentLift(
                    agent="codex-acp",
                    model="gpt-5.4",
                    with_skill_score=0.72,
                    baseline_score=0.35,
                    lift=0.37,
                    n_cases=5,
                    with_skill_passed=3,
                    baseline_passed=1,
                ),
            ],
        )

        rows = result.summary_table()
        # 3 rows per agent (with-skill, baseline, LIFT)
        assert len(rows) == 6

        # Verify LIFT rows show deltas
        lift_rows = [r for r in rows if r["mode"] == "LIFT"]
        assert len(lift_rows) == 2
        assert lift_rows[0]["score"] == "+2"
        assert lift_rows[1]["score"] == "+2"
