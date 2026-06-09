"""Tests for benchflow.tasks — task authoring (check and init)."""

import json
import os
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from benchflow._utils.task_authoring import (
    _check_ctrf_path,
    check_task,
    init_task,
    migrate_task_to_task_md,
    normalize_task_md,
)
from benchflow.cli.main import app
from benchflow.task import TaskConfig, TaskDocument
from benchflow.task.paths import TaskPaths

WANTED_FEATURE_DIR = Path("docs/examples/task-standard/benchflow-wanted-features")
WANTED_FEATURE_TASKS = (
    "compat-export-loss-reports",
    "ors-episode-reward-contract",
    "prompt-user-semantics",
    "runtime-capability-gate",
    "verifier-native-entrypoint",
    "verifier-package-reward-contract",
)
WANTED_FEATURE_ACCEPTANCE_LIVE_TASKS = (
    "runtime-capability-gate",
    "verifier-package-reward-contract",
)
FIRST_PARTY_MIXED_TASK_MD_FIXTURES = (
    Path("src/benchflow/demo_task"),
    Path("tests/examples/hello-world-task"),
    Path("tests/conformance/acp_smoke"),
)
REAL_SKILLSBENCH_TASK_MD_EXAMPLES = (
    "3d-scan-calc",
    "citation-check",
    "weighted-gdp-calc",
)
USER_RUNTIME_TASK_MD_EXAMPLES = (
    Path("docs/examples/task-md/user-runtime/private-facts-nudges"),
)


class TestCheckTask:
    """check_task(task_dir) -> list[str]"""

    def _make_valid_task(self, tmp_path: Path) -> Path:
        """Create a minimal valid task directory."""
        task = tmp_path / "my-task"
        task.mkdir()
        (task / "task.toml").write_text(
            "[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 120\n"
        )
        (task / "instruction.md").write_text("# Do something\n")
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        tests = task / "tests"
        tests.mkdir()
        (tests / "test.sh").write_text("#!/bin/bash\nexit 0\n")
        return task

    def _make_publication_task(
        self,
        tmp_path: Path,
        *,
        command: str = "./test.sh",
        outputs: str = "    reward_json: /logs/verifier/reward.json\n",
    ) -> Path:
        """Create a minimal native task with verifier.md and rubric files."""
        task = tmp_path / "publication-task"
        task.mkdir()
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        (task / "oracle").mkdir()
        (task / "oracle" / "solve.md").write_text("reference behavior\n")
        verifier = task / "verifier"
        verifier.mkdir()
        (verifier / "test.sh").write_text("#!/bin/bash\necho 1\n")
        (verifier / "rubrics").mkdir()
        (verifier / "rubrics" / "verifier.md").write_text("Rubric.\n")
        (task / "task.md").write_text(
            """---
schema_version: "1.3"
metadata:
  category: capability
environment:
  docker_image: ubuntu:24.04
---

## prompt

Do the thing.
"""
        )
        (verifier / "verifier.md").write_text(
            f"""---
verifier:
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: {command}
  rubric:
    combine: weighted_sum
    dimensions:
      correctness: {{weight: 1.0, source: deterministic}}
  outputs:
{outputs}---
"""
        )
        return task

    def _write_pytest_ctrf_wrapper(self, tmp_path: Path) -> Path:
        pytest_wrapper = tmp_path / "pytest-with-ctrf"
        pytest_wrapper.write_text(
            f"""#!/bin/bash
set -e
CTRF_JSON=""
ARGS=()
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--ctrf" ]; then
    CTRF_JSON="$2"
    shift 2
  else
    ARGS+=("$1")
    shift
  fi
done
"{sys.executable}" -m pytest "${{ARGS[@]}}"
if [ -n "$CTRF_JSON" ]; then
  mkdir -p "$(dirname "$CTRF_JSON")"
  printf '{{"results":{{}}}}' > "$CTRF_JSON"
fi
"""
        )
        pytest_wrapper.chmod(0o755)
        return pytest_wrapper

    def _add_acceptance_evidence(self, task: Path) -> None:
        evidence_dir = task / "evidence" / "calibration"
        evidence_dir.mkdir(parents=True)
        oracle = evidence_dir / "oracle-run.json"
        gold = evidence_dir / "gold-result.json"
        trajectory = evidence_dir / "gold-trajectory.jsonl"
        verifier_report = evidence_dir / "verifier-stability-report.json"
        calibration_report = evidence_dir / "calibration-report.json"
        review = evidence_dir / "review.json"
        oracle.write_text('{"reward": 1.0, "agent": "oracle"}\n')
        gold.write_text('{"reward": 1.0, "case": "gold"}\n')
        trajectory.write_text('{"type": "agent_message", "text": "done"}\n')
        review.write_text(
            json.dumps(
                {
                    "kind": "acceptance-review",
                    "anti_cheat": "passed",
                    "instruction_alignment": "passed",
                },
                indent=2,
            )
            + "\n"
        )
        calibration_report.write_text(
            json.dumps(
                {
                    "kind": "calibration-report",
                    "cases": [
                        {"name": "empty-workspace", "type": "no-op", "reward": 0.0},
                        {"name": "wrong-boundary", "type": "known-bad", "reward": 0.2},
                        {
                            "name": "partial-runtime-view",
                            "type": "partial",
                            "reward": 0.5,
                        },
                        {"name": "gold", "type": "reference", "reward": 1.0},
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        verifier_report.write_text(
            json.dumps(
                {
                    "kind": "verifier-stability-report",
                    "reruns": 5,
                    "flake_rate": 0.0,
                    "min_reward": 0.99,
                    "runs": [
                        {
                            "name": f"oracle-rerun-{index}",
                            "status": "passed",
                            "reward": 1.0,
                        }
                        for index in range(1, 6)
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        trajectory_sha = sha256(trajectory.read_bytes()).hexdigest()
        oracle_sha = sha256(oracle.read_bytes()).hexdigest()
        gold_sha = sha256(gold.read_bytes()).hexdigest()
        verifier_report_sha = sha256(verifier_report.read_bytes()).hexdigest()
        calibration_report_sha = sha256(calibration_report.read_bytes()).hexdigest()
        review_sha = sha256(review.read_bytes()).hexdigest()
        (task / "task.md").write_text(
            f"""---
schema_version: "1.3"
metadata:
  category: capability
environment:
  docker_image: ubuntu:24.04
benchflow:
  evidence:
    oracle_runs:
      required_reward: 1.0
      artifact: evidence/calibration/oracle-run.json
    verifier:
      reruns: 5
      flake_rate: 0.0
      report: evidence/calibration/verifier-stability-report.json
    review:
      anti_cheat: passed
      instruction_alignment: passed
      reviewer: benchflow
      artifact: evidence/calibration/review.json
    calibration:
      no_op_reward_max: 0.0
      known_bad_reward_max: 0.2
      partial_solution_range: [0.3, 0.8]
      report: evidence/calibration/calibration-report.json
      human_or_reference_examples:
        - name: gold
          expected_reward: 1.0
          artifact: evidence/calibration/gold-result.json
    trajectories:
      - path: evidence/calibration/gold-trajectory.jsonl
        kind: acp
        visibility: evidence_only
        sha256: {trajectory_sha}
    artifacts:
      - path: evidence/calibration/oracle-run.json
        sha256: {oracle_sha}
      - path: evidence/calibration/gold-result.json
        sha256: {gold_sha}
      - path: evidence/calibration/verifier-stability-report.json
        sha256: {verifier_report_sha}
      - path: evidence/calibration/calibration-report.json
        sha256: {calibration_report_sha}
      - path: evidence/calibration/review.json
        sha256: {review_sha}
---

## prompt

Do the thing.
"""
        )

    def _add_acceptance_live_evidence(self, task: Path) -> None:
        self._add_acceptance_live_evidence_with_cases(
            task,
            """
        - name: live-reference-verifier
          type: reference
          reruns: 2
          expect:
            reward_min: 0.99
""",
        )

    def _add_acceptance_live_evidence_with_cases(self, task: Path, cases: str) -> None:
        task_md = task / "task.md"
        task_md.write_text(
            task_md.read_text().replace(
                "    trajectories:\n",
                f"""    acceptance_live:
      report: evidence/acceptance/live-report.json
      workspace:
        source: current-worktree
        target: /repo
      cases:
{cases.rstrip()}
    trajectories:
""",
            )
        )

    def _add_acceptance_live_generated_calibration(self, task: Path) -> None:
        task_md = task / "task.md"
        task_md.write_text(
            task_md.read_text().replace(
                "    trajectories:\n",
                """    acceptance_live:
      report: evidence/acceptance/live-report.json
      workspace:
        source: current-worktree
        target: /repo
      calibration:
        from: calibration.report
        reruns: 1
        flake_rate_max: 0.0
    trajectories:
""",
            )
        )

    def _add_acceptance_live_leaderboard_evidence(self, task: Path) -> None:
        task_md = task / "task.md"
        task_md.write_text(
            task_md.read_text().replace(
                "    trajectories:\n",
                """    acceptance_live:
      report: evidence/acceptance/live-report.json
      workspace:
        source: current-worktree
        target: /repo
      calibration:
        from: calibration.report
        reruns: 1
        flake_rate_max: 0.0
      leaderboard:
        required: true
        max_flake_rate: 0.0
      cases:
        - name: live-oracle-rerun
          type: oracle
          reruns: 1
          expect:
            reward_min: 0.99
            flake_rate_max: 0.0
        - name: live-reference-verifier
          type: reference
          reruns: 1
          expect:
            reward_min: 0.99
            flake_rate_max: 0.0
    trajectories:
""",
            )
        )

    def _add_acceptance_live_leaderboard_without_calibration(self, task: Path) -> None:
        task_md = task / "task.md"
        task_md.write_text(
            task_md.read_text().replace(
                "    trajectories:\n",
                """    acceptance_live:
      report: evidence/acceptance/live-report.json
      workspace:
        source: current-worktree
        target: /repo
      leaderboard:
        required: true
        max_flake_rate: 0.0
      cases:
        - name: live-oracle-rerun
          type: oracle
          reruns: 1
          expect:
            reward_min: 0.99
            flake_rate_max: 0.0
        - name: live-reference-verifier
          type: reference
          reruns: 1
          expect:
            reward_min: 0.99
            flake_rate_max: 0.0
    trajectories:
""",
            )
        )

    def _add_calibration_report_commands(self, task: Path) -> None:
        report = task / "evidence" / "calibration" / "calibration-report.json"
        old_sha = sha256(report.read_bytes()).hexdigest()
        data = json.loads(report.read_text())
        commands = {
            "empty-workspace": "rm -rf src tests",
            "wrong-boundary": "rm -f src/benchflow/task/runtime_capabilities.py",
            "partial-runtime-view": "rm -f src/benchflow/task/acceptance_live.py",
        }
        for case in data["cases"]:
            command = commands.get(case["name"])
            if command is not None:
                case["command"] = command
        report.write_text(json.dumps(data, indent=2) + "\n")
        new_sha = sha256(report.read_bytes()).hexdigest()
        task_md = task / "task.md"
        task_md.write_text(task_md.read_text().replace(old_sha, new_sha))

    def test_valid_task_no_issues(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        assert check_task(task) == []

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "not-a-dir"
        f.write_text("hi")
        issues = check_task(f)
        assert issues == [f"Not a directory: {f}"]

    def test_missing_task_toml(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        (task / "task.toml").unlink()
        issues = check_task(task)
        assert "Missing required file: task.toml" in issues

    def test_missing_instruction_md(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        (task / "instruction.md").unlink()
        issues = check_task(task)
        assert "Missing required file: instruction.md" in issues

    def test_missing_environment_dir(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        shutil.rmtree(task / "environment")
        issues = check_task(task)
        assert "Missing required directory: environment/" in issues

    def test_missing_dockerfile(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        (task / "environment" / "Dockerfile").unlink()
        issues = check_task(task)
        assert "Missing environment/Dockerfile" in issues

    def test_missing_tests_dir(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        shutil.rmtree(task / "tests")
        issues = check_task(task)
        assert (
            "Missing verifier/ directory (or legacy tests/; verifier needs "
            "test.sh or a verifier.md selected strategy)" in issues
        )

    def test_empty_tests_dir(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        for f in (task / "tests").iterdir():
            f.unlink()
        issues = check_task(task)
        assert "tests/ directory is empty" in issues

    def test_legacy_verifier_dir_without_entrypoint_is_rejected(self, tmp_path):
        """Guards PR #1's structural verifier-entrypoint validation."""
        task = self._make_valid_task(tmp_path)
        (task / "tests" / "test.sh").unlink()
        (task / "tests" / "README.md").write_text("not runnable\n")

        issues = check_task(task)

        assert any("tests/ has no runnable verifier entrypoint" in i for i in issues)

    def test_native_verifier_dir_without_entrypoint_is_rejected(self, tmp_path):
        """Guards PR #1's native verifier-entrypoint validation."""
        task = tmp_path / "native-no-entrypoint"
        task.mkdir()
        (task / "task.md").write_text(
            """---
task:
  name: benchflow/native-no-entrypoint
---

## prompt

Do the thing.
"""
        )
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        verifier = task / "verifier"
        verifier.mkdir()
        (verifier / "README.md").write_text("not runnable\n")

        issues = check_task(task)

        assert any("verifier/ has no runnable verifier entrypoint" in i for i in issues)

    def test_empty_instruction_md(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        (task / "instruction.md").write_text("")
        issues = check_task(task)
        assert "instruction.md is empty" in issues

    def test_task_toml_missing_agent_section_is_allowed(self, tmp_path):
        """[agent] is optional — runtime AgentConfig defaults to no timeout.

        Guards #379: bench tasks check and bench eval create must agree on
        the missing-[agent] contract. The runtime accepts it, so check_task
        does too.
        """
        task = self._make_valid_task(tmp_path)
        (task / "task.toml").write_text("[verifier]\ntimeout_sec = 120\n")
        issues = check_task(task)
        assert not any("agent" in i for i in issues)

    def test_task_toml_missing_timeout_sec_is_allowed(self, tmp_path):
        """[agent].timeout_sec is optional — defaults to no wall-clock cap.

        Guards #379: keeps check_task in sync with AgentConfig (timeout_sec
        defaults to None).
        """
        task = self._make_valid_task(tmp_path)
        (task / "task.toml").write_text("[agent]\n")
        issues = check_task(task)
        assert not any("timeout_sec" in i for i in issues)

    def test_task_toml_parse_error(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        (task / "task.toml").write_text("not valid {{{{ toml")
        issues = check_task(task)
        assert any(i.startswith("task.toml parse error:") for i in issues)

    def test_multiple_issues_reported(self, tmp_path):
        """An empty directory should report multiple specific issues."""
        task = tmp_path / "empty-task"
        task.mkdir()
        issues = check_task(task)
        assert "Missing required file: task.toml" in issues
        assert "Missing required file: instruction.md" in issues
        assert "Missing required directory: environment/" in issues

    def test_schema_level_accepts_task_md_authoring_fixture(self, tmp_path):
        """A schema example can be parse-valid without being runnable."""
        task = tmp_path / "schema-fixture"
        task.mkdir()
        (task / "task.md").write_text(
            """---
task:
  name: benchflow/schema-fixture
environment:
  docker_image: ubuntu:24.04
agents:
  roles:
    solver:
      agent: codex
scenes:
  - name: solve
    roles: [solver]
---

## prompt

Show that the authoring document parses.
"""
        )

        assert check_task(task, validation_level="schema") == []
        structural_issues = check_task(task)
        assert "Missing required directory: environment/" in structural_issues

    def test_schema_level_still_checks_prompt(self, tmp_path):
        """Schema validation remains an authoring check, not a no-op."""
        task = tmp_path / "empty-prompt"
        task.mkdir()
        (task / "task.md").write_text(
            """---
task:
  name: benchflow/empty-prompt
---

## prompt

"""
        )

        assert "task.md prompt is empty" in check_task(
            task,
            validation_level="schema",
        )

    def test_tasks_check_cli_schema_level_accepts_schema_example(self, tmp_path):
        """The CLI exposes schema-level checks for task.md format fixtures."""
        from typer.testing import CliRunner

        task = tmp_path / "schema-cli"
        task.mkdir()
        (task / "task.md").write_text(
            """---
task:
  name: benchflow/schema-cli
environment:
  docker_image: ubuntu:24.04
---

## prompt

Parse this task.md without requiring runtime files.
"""
        )

        result = CliRunner().invoke(
            app,
            ["tasks", "check", str(task), "--level", "schema"],
        )

        assert result.exit_code == 0, result.output
        assert "valid (schema)" in result.output

    @pytest.mark.parametrize("task_name", REAL_SKILLSBENCH_TASK_MD_EXAMPLES)
    def test_real_skillsbench_task_md_examples_pass_publication_grade(
        self,
        task_name: str,
    ) -> None:
        """Guards PR #1's real SkillsBench publication-grade native packages."""
        task = Path("docs/examples/task-md/real-skillsbench") / task_name

        assert (task / "task.md").is_file()
        assert not (task / "task.toml").exists()
        assert not (task / "instruction.md").exists()
        assert not (task / "tests").exists()
        assert not (task / "solution").exists()
        assert (task / "verifier" / "test.sh").is_file()
        assert (task / "verifier" / "test_outputs.py").is_file()
        assert (task / "verifier" / "verifier.md").is_file()
        assert (task / "verifier" / "rubrics" / "verifier.md").is_file()
        assert (task / "oracle" / "solve.sh").is_file()
        assert (task / "environment" / "skills").is_dir()
        assert check_task(task, validation_level="publication-grade") == []

    @pytest.mark.parametrize(
        "task",
        FIRST_PARTY_MIXED_TASK_MD_FIXTURES,
        ids=lambda path: path.name,
    )
    def test_first_party_mixed_task_md_fixtures_pass_structural_check(
        self,
        task: Path,
    ) -> None:
        """Guards PR #1's native entrypoint dogfood on first-party fixtures."""
        assert (task / "task.md").is_file()
        assert (task / "task.toml").is_file()
        assert (task / "instruction.md").is_file()
        assert (task / "tests" / "test.sh").is_file()
        assert check_task(task) == []

    @pytest.mark.parametrize(
        "task_name",
        [
            "optimize-quadratic-to-nlogn",
            "regex-email-parser",
            "topo-sort-with-cycle-detection",
        ],
    )
    def test_generated_skill_eval_task_md_examples_pass_publication_grade(
        self,
        task_name: str,
    ) -> None:
        """Guards PR #1's generated skill-eval native verifier package path."""
        task = (
            Path("docs/examples/task-md/generated-skill-eval/models-as-skills")
            / task_name
        )

        assert (task / "task.md").is_file()
        assert (task / "verifier" / "test.sh").is_file()
        assert (task / "verifier" / "judge.py").is_file()
        assert (task / "verifier" / "case.json").is_file()
        assert (task / "verifier" / "verifier.md").is_file()
        assert (task / "verifier" / "rubrics" / "verifier.md").is_file()
        assert (task / "oracle" / "README.md").is_file()
        assert check_task(task, validation_level="publication-grade") == []

    @pytest.mark.parametrize(
        "task",
        USER_RUNTIME_TASK_MD_EXAMPLES,
        ids=lambda path: path.name,
    )
    def test_user_runtime_task_md_examples_pass_publication_grade(
        self,
        task: Path,
    ) -> None:
        """Guards PR #1's simulated-user task.md fixture against drift."""
        assert (task / "task.md").is_file()
        assert not (task / "task.toml").exists()
        assert not (task / "instruction.md").exists()
        assert not (task / "tests").exists()
        assert not (task / "solution").exists()
        assert (task / "verifier" / "test.sh").is_file()
        assert (task / "verifier" / "verifier.md").is_file()
        assert (task / "verifier" / "rubrics" / "verifier.md").is_file()
        assert (task / "oracle" / "solve.sh").is_file()
        assert check_task(task, validation_level="publication-grade") == []
        assert (
            check_task(
                task,
                validation_level="runtime-capability",
                sandbox_type="docker",
            )
            == []
        )

    def test_generated_skill_eval_verifier_runs_with_env_paths(self, tmp_path) -> None:
        """Guards PR #1's generated task.md verifier against hardcoded mount paths."""
        source = Path(
            "docs/examples/task-md/generated-skill-eval/models-as-skills/"
            "regex-email-parser/verifier"
        )
        verifier = tmp_path / "verifier"
        shutil.copytree(source, verifier)

        case_path = verifier / "case.json"
        case = json.loads(case_path.read_text())
        case["expected_behavior"] = []
        case["ground_truth"] = "pass-token"
        case_path.write_text(json.dumps(case))

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "answer.txt").write_text("pass-token\n")
        agent_logs = tmp_path / "logs" / "agent"
        agent_logs.mkdir(parents=True)
        (agent_logs / "agent.txt").write_text("agent wrote pass-token\n")
        verifier_logs = tmp_path / "logs" / "verifier"

        result = subprocess.run(
            ["bash", str(verifier / "test.sh")],
            cwd=tmp_path,
            env={
                **os.environ,
                "BENCHFLOW_VERIFIER_DIR": str(verifier),
                "BENCHFLOW_WORKSPACE": str(workspace),
                "BENCHFLOW_AGENT_LOG_DIR": str(agent_logs),
                "BENCHFLOW_REWARD_TEXT": str(verifier_logs / "reward.txt"),
                "BENCHFLOW_REWARD_JSON": str(verifier_logs / "reward.json"),
                "BENCHFLOW_REWARD_DETAILS_JSON": str(
                    verifier_logs / "judge_result.json"
                ),
            },
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr + result.stdout
        assert (verifier_logs / "reward.txt").read_text().strip() == "1.0"
        assert json.loads((verifier_logs / "reward.json").read_text()) == {
            "reward": 1.0
        }

    def test_user_runtime_private_facts_oracle_and_verifier_run_with_env_paths(
        self,
        tmp_path,
    ) -> None:
        """Guards PR #1's simulated-user task.md fixture env path support."""
        task = Path("docs/examples/task-md/user-runtime/private-facts-nudges").resolve()
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        oracle_result = subprocess.run(
            ["bash", str(task / "oracle" / "solve.sh")],
            cwd=tmp_path,
            env={
                **os.environ,
                "BENCHFLOW_WORKSPACE": str(workspace),
            },
            text=True,
            capture_output=True,
            check=False,
        )

        assert oracle_result.returncode == 0, (
            oracle_result.stderr + oracle_result.stdout
        )
        assert (workspace / "order_id.txt").read_text().strip() == "BF-1042"
        assert (workspace / "recovery.json").is_file()

        verifier_logs = tmp_path / "logs" / "verifier"
        verifier_result = subprocess.run(
            ["bash", str(task / "verifier" / "test.sh")],
            cwd=tmp_path,
            env={
                **os.environ,
                "BENCHFLOW_WORKSPACE": str(workspace),
                "BENCHFLOW_REWARD_TEXT": str(verifier_logs / "reward.txt"),
                "BENCHFLOW_REWARD_JSON": str(verifier_logs / "reward.json"),
                "BENCHFLOW_REWARD_DETAILS_JSON": str(
                    verifier_logs / "reward-details.json"
                ),
            },
            text=True,
            capture_output=True,
            check=False,
        )

        assert verifier_result.returncode == 0, (
            verifier_result.stderr + verifier_result.stdout
        )
        assert (verifier_logs / "reward.txt").read_text().strip() == "1.0"
        assert json.loads((verifier_logs / "reward.json").read_text()) == {
            "reward": 1.0
        }
        details = json.loads((verifier_logs / "reward-details.json").read_text())
        assert details["checks"]["order_id_matches_private_fact"] is True

    def test_real_skillsbench_3d_scan_oracle_and_verifier_run_with_env_paths(
        self,
        tmp_path,
    ) -> None:
        """Guards PR #1's real SkillsBench task.md package against hardcoded mount paths."""
        task = Path("docs/examples/task-md/real-skillsbench/3d-scan-calc").resolve()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        shutil.copy2(task / "environment" / "scan_data.stl", workspace)
        shutil.copy2(task / "environment" / "material_density_table.md", workspace)
        shutil.copytree(task / "environment" / "skills", workspace / "skills")

        common_env = {
            **os.environ,
            "BENCHFLOW_WORKSPACE": str(workspace),
        }
        oracle_result = subprocess.run(
            ["bash", str(task / "oracle" / "solve.sh")],
            cwd=tmp_path,
            env=common_env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert oracle_result.returncode == 0, (
            oracle_result.stderr + oracle_result.stdout
        )
        assert (workspace / "mass_report.json").is_file()

        verifier_logs = tmp_path / "logs" / "verifier"
        pytest_wrapper = self._write_pytest_ctrf_wrapper(tmp_path)

        verifier_result = subprocess.run(
            ["bash", str(task / "verifier" / "test.sh")],
            cwd=tmp_path,
            env={
                **common_env,
                "BENCHFLOW_VERIFIER_DIR": str(task / "verifier"),
                "BENCHFLOW_REWARD_TEXT": str(verifier_logs / "reward.txt"),
                "BENCHFLOW_REWARD_JSON": str(verifier_logs / "reward.json"),
                "BENCHFLOW_REWARD_DETAILS_JSON": str(verifier_logs / "ctrf.json"),
                "BENCHFLOW_SKIP_VERIFIER_DEPS": "1",
                "BENCHFLOW_PYTEST_BIN": str(pytest_wrapper),
            },
            text=True,
            capture_output=True,
            check=False,
        )

        assert verifier_result.returncode == 0, (
            verifier_result.stderr + verifier_result.stdout
        )
        assert (verifier_logs / "reward.txt").read_text().strip() == "1.0"
        assert json.loads((verifier_logs / "reward.json").read_text()) == {
            "reward": 1.0
        }
        assert (verifier_logs / "ctrf.json").is_file()

    def test_real_skillsbench_citation_verifier_runs_with_env_paths(
        self,
        tmp_path,
    ) -> None:
        """Guards PR #1's citation task.md verifier against hardcoded mount paths."""
        task = Path("docs/examples/task-md/real-skillsbench/citation-check").resolve()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        shutil.copy2(task / "environment" / "test.bib", workspace)
        (workspace / "answer.json").write_text(
            json.dumps(
                {
                    "fake_citations": [
                        "Advances in Artificial Intelligence for Natural Language Processing",
                        "Blockchain Applications in Supply Chain Management",
                        "Neural Networks in Deep Learning: A Comprehensive Review",
                    ]
                }
            )
        )

        verifier_logs = tmp_path / "logs" / "verifier"
        pytest_wrapper = self._write_pytest_ctrf_wrapper(tmp_path)

        result = subprocess.run(
            ["bash", str(task / "verifier" / "test.sh")],
            cwd=tmp_path,
            env={
                **os.environ,
                "BENCHFLOW_WORKSPACE": str(workspace),
                "BENCHFLOW_VERIFIER_DIR": str(task / "verifier"),
                "BENCHFLOW_REWARD_TEXT": str(verifier_logs / "reward.txt"),
                "BENCHFLOW_REWARD_JSON": str(verifier_logs / "reward.json"),
                "BENCHFLOW_REWARD_DETAILS_JSON": str(verifier_logs / "ctrf.json"),
                "BENCHFLOW_SKIP_VERIFIER_DEPS": "1",
                "BENCHFLOW_PYTEST_BIN": str(pytest_wrapper),
            },
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr + result.stdout
        assert (verifier_logs / "reward.txt").read_text().strip() == "1.0"
        assert json.loads((verifier_logs / "reward.json").read_text()) == {
            "reward": 1.0
        }
        assert (verifier_logs / "ctrf.json").is_file()

    def test_real_skillsbench_weighted_gdp_oracle_and_verifier_run_with_env_paths(
        self,
        tmp_path,
    ) -> None:
        """Guards PR #1's weighted GDP task.md package against hardcoded mount paths."""
        pytest.importorskip("openpyxl")
        task = Path(
            "docs/examples/task-md/real-skillsbench/weighted-gdp-calc"
        ).resolve()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        shutil.copy2(task / "environment" / "gdp.xlsx", workspace)

        oracle_result = subprocess.run(
            ["bash", str(task / "oracle" / "solve.sh")],
            cwd=tmp_path,
            env={
                **os.environ,
                "BENCHFLOW_WORKSPACE": str(workspace),
                "BENCHFLOW_PYTHON_BIN": sys.executable,
            },
            text=True,
            capture_output=True,
            check=False,
        )

        assert oracle_result.returncode == 0, (
            oracle_result.stderr + oracle_result.stdout
        )
        assert (workspace / "gdp.xlsx").is_file()

        verifier_logs = tmp_path / "logs" / "verifier"
        pytest_wrapper = self._write_pytest_ctrf_wrapper(tmp_path)
        verifier_result = subprocess.run(
            ["bash", str(task / "verifier" / "test.sh")],
            cwd=tmp_path,
            env={
                **os.environ,
                "BENCHFLOW_WORKSPACE": str(workspace),
                "BENCHFLOW_VERIFIER_DIR": str(task / "verifier"),
                "BENCHFLOW_REWARD_TEXT": str(verifier_logs / "reward.txt"),
                "BENCHFLOW_REWARD_JSON": str(verifier_logs / "reward.json"),
                "BENCHFLOW_REWARD_DETAILS_JSON": str(verifier_logs / "ctrf.json"),
                "BENCHFLOW_SKIP_VERIFIER_DEPS": "1",
                "BENCHFLOW_PYTEST_BIN": str(pytest_wrapper),
            },
            text=True,
            capture_output=True,
            check=False,
        )

        assert verifier_result.returncode == 0, (
            verifier_result.stderr + verifier_result.stdout
        )
        assert (verifier_logs / "reward.txt").read_text().strip() == "1.0"
        assert json.loads((verifier_logs / "reward.json").read_text()) == {
            "reward": 1.0
        }
        assert (verifier_logs / "ctrf.json").is_file()
        assert (verifier_logs / "gdp_modified.xlsx").is_file()

    def test_runtime_capability_gate_accepts_supported_dogfood_workdir(self):
        """The runtime-capability dogfood task has no unsupported sandbox gaps."""
        task = Path(
            "docs/examples/task-standard/benchflow-wanted-features/"
            "runtime-capability-gate"
        )

        assert check_task(task) == []
        assert check_task(task, sandbox_type="docker") == []

    def test_runtime_capability_gate_detects_verifier_alias_drift(self, tmp_path):
        """Native verifier/ and legacy tests/ cannot silently diverge."""
        task = self._make_valid_task(tmp_path)
        verifier = task / "verifier"
        verifier.mkdir()
        (verifier / "test.sh").write_text("#!/bin/bash\necho 1\n")
        (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 0\n")

        issues = check_task(task, sandbox_type="docker")

        assert any("verifier|tests" in issue for issue in issues)

    def test_check_task_structural_rejects_task_md_task_toml_instruction_drift(
        self,
        tmp_path,
    ):
        """Native and split definitions cannot disagree in structural checks."""
        task = self._make_valid_task(tmp_path)
        (task / "task.md").write_text(
            """---
agent:
  timeout_sec: 301
verifier:
  timeout_sec: 120
---

## prompt

Different native prompt.
"""
        )

        issues = check_task(task)

        assert any("task.md|task.toml" in issue for issue in issues)
        assert any("task.md|instruction.md" in issue for issue in issues)

    @pytest.mark.parametrize(
        ("alias_file", "missing_file"),
        [
            ("instruction.md", "task.toml"),
            ("task.toml", "instruction.md"),
        ],
    )
    def test_check_task_structural_rejects_partial_legacy_split_alias(
        self,
        tmp_path,
        alias_file,
        missing_file,
    ):
        """Guards PR #1 against partial split aliases beside task.md."""
        task = self._make_publication_task(tmp_path)
        if alias_file == "instruction.md":
            (task / alias_file).write_text("Legacy prompt.\n")
        else:
            (task / alias_file).write_text('[task]\nname = "legacy/name"\n')

        issues = check_task(task)

        assert any("task.md|legacy-split" in issue for issue in issues)
        assert any(f"missing {missing_file}" in issue for issue in issues)

    def test_check_task_structural_rejects_oracle_solution_and_verifier_tests_drift(
        self,
        tmp_path,
    ):
        """Native oracle/verifier aliases cannot diverge from split aliases."""
        task = self._make_valid_task(tmp_path)
        (task / "task.md").write_text(
            """---
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
---

## prompt

# Do something
"""
        )
        solution = task / "solution"
        solution.mkdir()
        (solution / "solve.sh").write_text("#!/bin/bash\necho legacy\n")
        oracle = task / "oracle"
        oracle.mkdir()
        (oracle / "solve.sh").write_text("#!/bin/bash\necho native\n")
        verifier = task / "verifier"
        verifier.mkdir()
        (verifier / "test.sh").write_text("#!/bin/bash\necho native\n")

        issues = check_task(task)

        assert any("oracle|solution" in issue for issue in issues)
        assert any("verifier|tests" in issue for issue in issues)

    def test_runtime_capability_level_requires_sandbox(self):
        """Runtime-capability validation is backend-specific."""
        task = Path(
            "docs/examples/task-standard/benchflow-wanted-features/"
            "runtime-capability-gate"
        )

        issues = check_task(task, validation_level="runtime-capability")

        assert issues == ["runtime-capability validation requires --sandbox <backend>"]

    def test_publication_grade_accepts_wanted_feature_dogfood_set(self):
        """Publication-grade validation is dogfooded on real BenchFlow tasks."""
        root = Path("docs/examples/task-standard/benchflow-wanted-features")
        task_dirs = sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "task.md").exists()
        )

        assert {path.name for path in task_dirs} == {
            "compat-export-loss-reports",
            "ors-episode-reward-contract",
            "prompt-user-semantics",
            "runtime-capability-gate",
            "verifier-native-entrypoint",
            "verifier-package-reward-contract",
        }
        for task in task_dirs:
            assert check_task(task, validation_level="publication-grade") == []

    def test_publication_grade_with_sandbox_accepts_wanted_feature_dogfood_set(self):
        """Registry-ready dogfood tasks must pass the sandbox-aware publish gate."""
        root = Path("docs/examples/task-standard/benchflow-wanted-features")

        for task_name in WANTED_FEATURE_TASKS:
            task = root / task_name
            assert (
                check_task(
                    task,
                    validation_level="publication-grade",
                    sandbox_type="docker",
                )
                == []
            )

    def test_publication_grade_with_sandbox_rejects_unsupported_runtime_policy(
        self,
        tmp_path,
    ):
        """Static publication-grade cannot mask unsupported runtime semantics."""
        task = self._make_publication_task(tmp_path)
        task_md = task / "task.md"
        task_md.write_text(
            task_md.read_text().replace(
                "metadata:\n  category: capability\n",
                """metadata:
  category: capability
benchflow:
  runtime_policy:
    requires_state_branching: true
""",
            )
        )

        assert check_task(task, validation_level="publication-grade") == []

        issues = check_task(
            task,
            validation_level="publication-grade",
            sandbox_type="docker",
        )

        assert any(
            "Unsupported runtime feature: benchflow.runtime_policy" in issue
            for issue in issues
        )

    def test_publication_grade_rejects_legacy_split_layout(self, tmp_path):
        """Publication-grade means native task.md, not Harbor/Pier split files."""
        task = self._make_valid_task(tmp_path)

        issues = check_task(task, validation_level="publication-grade")

        assert any("requires task.md" in issue for issue in issues)
        assert any("requires native oracle/" in issue for issue in issues)
        assert any("requires native verifier/" in issue for issue in issues)

    def test_publication_grade_requires_verifier_document(self, tmp_path):
        """A plain test.sh package can be structural, but not publication-grade."""
        task = init_task(
            "native-without-verifier-doc",
            parent_dir=tmp_path,
            no_pytest=True,
            task_format="task-md",
        )
        (task / "task.md").write_text(
            """---
schema_version: "1.3"
metadata:
  category: capability
environment:
  docker_image: ubuntu:24.04
---

## prompt

Create hello.txt.
"""
        )
        (task / "verifier" / "test.sh").write_text(
            '#!/bin/bash\necho "1.0" > /logs/verifier/reward.txt\n'
        )
        (task / "verifier" / "verifier.md").unlink()
        shutil.rmtree(task / "verifier" / "rubrics")
        (task / "oracle" / "solve.sh").write_text("#!/bin/bash\ntouch hello.txt\n")

        assert check_task(task) == []

        issues = check_task(task, validation_level="publication-grade")

        assert issues == ["publication-grade validation requires verifier/verifier.md"]

    def test_publication_grade_requires_explicit_reward_json(self, tmp_path):
        """Publication-grade verifier packages must declare rich reward output."""
        task = self._make_publication_task(tmp_path, outputs="")

        issues = check_task(task, validation_level="publication-grade")

        assert "publication-grade verifier outputs must declare reward_json" in issues

    def test_publication_grade_rejects_script_strategy_without_packaged_artifact(
        self,
        tmp_path,
    ):
        """Guards PR #1 against bare pytest commands as verifier packages."""
        task = self._make_publication_task(tmp_path, command="pytest")

        issues = check_task(task, validation_level="publication-grade")

        assert any(
            "must reference a packaged verifier artifact" in issue for issue in issues
        )

    @pytest.mark.parametrize(
        "command,missing",
        [
            ("bash missing.sh", "missing.sh"),
            ("python missing.py", "missing.py"),
            ("uv run python missing.py", "missing.py"),
        ],
    )
    def test_publication_grade_checks_interpreter_wrapped_scripts(
        self,
        tmp_path,
        command,
        missing,
    ):
        """Publication-grade follows local script args, not only argv[0]."""
        task = self._make_publication_task(tmp_path, command=command)

        issues = check_task(task, validation_level="publication-grade")

        assert any(
            "references missing script" in issue and missing in issue
            for issue in issues
        )

    def test_acceptance_requires_declared_evidence(self, tmp_path):
        """Acceptance is stricter than publication-grade and needs evidence."""
        task = self._make_publication_task(tmp_path)

        assert check_task(task, validation_level="publication-grade") == []
        issues = check_task(task, validation_level="acceptance")

        assert "acceptance validation requires benchflow.evidence mapping" in issues

    def test_acceptance_accepts_declared_evidence(self, tmp_path):
        """Static acceptance validates oracle/review/calibration evidence shape."""
        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)

        assert check_task(task, validation_level="acceptance") == []

    def test_acceptance_live_requires_declared_live_cases(self, tmp_path):
        """Live proof needs an explicit executable case contract."""
        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)

        issues = check_task(
            task,
            validation_level="acceptance-live",
            sandbox_type="docker",
        )

        assert issues == [
            "acceptance-live validation requires "
            "benchflow.evidence.acceptance_live mapping"
        ]

    def test_acceptance_live_rejects_unsafe_report_path(self, tmp_path):
        """Live report artifacts must stay inside the task package."""
        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        self._add_acceptance_live_evidence(task)
        task_md = task / "task.md"
        task_md.write_text(
            task_md.read_text().replace(
                "      report: evidence/acceptance/live-report.json\n",
                "      report: ../live-report.json\n",
            )
        )

        issues = check_task(
            task,
            validation_level="acceptance-live",
            sandbox_type="docker",
        )

        assert issues == ["acceptance-live report must be a safe relative file path"]

    @pytest.mark.asyncio
    async def test_acceptance_live_without_report_does_not_write_artifact(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Live report persistence is an authored contract, not a silent mutation."""
        from benchflow.task import acceptance_live as live_mod

        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        self._add_acceptance_live_evidence(task)
        task_md = task / "task.md"
        task_md.write_text(
            task_md.read_text().replace(
                "      report: evidence/acceptance/live-report.json\n",
                "",
            )
        )
        spec, issues = live_mod.parse_live_acceptance_spec(
            task,
            evidence=TaskDocument.from_path(task / "task.md").benchflow["evidence"],
        )
        assert issues == []
        assert spec is not None
        assert spec.report_path is None

        async def fake_run_single_case(*_args, **_kwargs):
            return 1.0, None

        monkeypatch.setattr(live_mod, "_stage_current_worktree", lambda _tmp: tmp_path)
        monkeypatch.setattr(live_mod, "_run_single_case", fake_run_single_case)

        run_issues = await live_mod._run_live_acceptance_checks(
            task,
            sandbox_type="docker",
            spec=spec,
        )

        assert run_issues == []
        assert not (task / "evidence" / "acceptance" / "live-report.json").exists()

    @pytest.mark.asyncio
    async def test_acceptance_live_report_output_writes_outside_task(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Guards PR #1's live dogfood report-output override."""
        from benchflow.task import acceptance_live as live_mod

        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        self._add_acceptance_live_evidence(task)
        output = tmp_path / "reports" / "live-report.json"
        spec, issues = live_mod.parse_live_acceptance_spec(
            task,
            evidence=TaskDocument.from_path(task / "task.md").benchflow["evidence"],
            report_output=output,
        )
        assert issues == []
        assert spec is not None
        assert spec.report_path == output

        async def fake_run_single_case(*_args, **_kwargs):
            return 1.0, None

        monkeypatch.setattr(live_mod, "_stage_current_worktree", lambda _tmp: tmp_path)
        monkeypatch.setattr(live_mod, "_run_single_case", fake_run_single_case)

        run_issues = await live_mod._run_live_acceptance_checks(
            task,
            sandbox_type="docker",
            spec=spec,
        )

        assert run_issues == []
        assert output.is_file()
        assert not (task / "evidence" / "acceptance" / "live-report.json").exists()
        expected_digest = sha256(output.read_bytes()).hexdigest()
        assert output.with_suffix(".json.sha256").read_text() == (
            f"{expected_digest}  {output.as_posix()}\n"
        )

    @pytest.mark.asyncio
    async def test_acceptance_live_report_only_skips_declared_report_write(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Report-only dogfood validates without dirtying the task package."""
        from benchflow.task import acceptance_live as live_mod

        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        self._add_acceptance_live_evidence(task)
        spec, issues = live_mod.parse_live_acceptance_spec(
            task,
            evidence=TaskDocument.from_path(task / "task.md").benchflow["evidence"],
        )
        assert issues == []
        assert spec is not None
        assert spec.report_path is not None

        async def fake_run_single_case(*_args, **_kwargs):
            return 1.0, None

        monkeypatch.setattr(live_mod, "_stage_current_worktree", lambda _tmp: tmp_path)
        monkeypatch.setattr(live_mod, "_run_single_case", fake_run_single_case)

        report = task / "evidence" / "acceptance" / "live-report.json"
        run_issues = await live_mod._run_live_acceptance_checks(
            task,
            sandbox_type="docker",
            spec=spec,
            write_report=False,
        )

        assert run_issues == []
        assert not report.exists()
        assert not report.with_suffix(".json.sha256").exists()

    def test_acceptance_live_report_output_satisfies_leaderboard_report_requirement(
        self,
        tmp_path,
    ):
        """Guards PR #1's leaderboard report-output override contract."""
        from benchflow.task import acceptance_live as live_mod

        task = self._make_publication_task(tmp_path)
        (task / "oracle" / "solve.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        (task / "oracle" / "solve.sh").chmod(0o755)
        self._add_acceptance_evidence(task)
        self._add_calibration_report_commands(task)
        self._add_acceptance_live_leaderboard_evidence(task)
        task_md = task / "task.md"
        task_md.write_text(
            task_md.read_text().replace(
                "      report: evidence/acceptance/live-report.json\n",
                "",
            )
        )
        output = tmp_path / "live" / "leaderboard-report.json"

        spec, issues = live_mod.parse_live_acceptance_spec(
            task,
            evidence=TaskDocument.from_path(task / "task.md").benchflow["evidence"],
            report_output=output,
        )

        assert issues == []
        assert spec is not None
        assert spec.report_path == output

    def test_acceptance_live_invokes_declared_runner_after_static_passes(
        self,
        tmp_path,
        monkeypatch,
    ):
        """acceptance-live runs only after publication and static evidence pass."""
        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        self._add_acceptance_live_evidence(task)
        calls = []

        def fake_live_runner(
            task_dir, *, sandbox_type, evidence, report_output=None, write_report=True
        ):
            calls.append(
                (task_dir, sandbox_type, evidence, report_output, write_report)
            )
            return []

        monkeypatch.setattr(
            "benchflow._utils.task_authoring.run_live_acceptance_checks",
            fake_live_runner,
        )

        issues = check_task(
            task,
            validation_level="acceptance-live",
            sandbox_type="docker",
        )

        assert issues == []
        assert len(calls) == 1
        assert calls[0][0] == task
        assert calls[0][1] == "docker"
        assert "acceptance_live" in calls[0][2]
        assert calls[0][3] is None

    def test_acceptance_live_static_failure_short_circuits_runner(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Bad static evidence cannot be hidden by a passing live rerun."""
        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        self._add_acceptance_live_evidence(task)
        (task / "evidence" / "calibration" / "gold-result.json").write_text(
            '{"reward": 0.0}\n'
        )

        def fake_live_runner(*_args, **_kwargs):
            raise AssertionError("live runner should not be invoked")

        monkeypatch.setattr(
            "benchflow._utils.task_authoring.run_live_acceptance_checks",
            fake_live_runner,
        )

        issues = check_task(
            task,
            validation_level="acceptance-live",
            sandbox_type="docker",
        )

        assert any("sha256 mismatch" in issue for issue in issues)

    def test_acceptance_live_generates_cases_from_calibration_report(self, tmp_path):
        """Live calibration can execute the static calibration report contract."""
        from benchflow.task import acceptance_live as live_mod

        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        self._add_calibration_report_commands(task)
        self._add_acceptance_live_generated_calibration(task)

        spec, issues = live_mod.parse_live_acceptance_spec(
            task,
            evidence=TaskDocument.from_path(task / "task.md").benchflow["evidence"],
        )

        assert issues == []
        assert spec is not None
        assert [(case.name, case.case_type, case.source) for case in spec.cases] == [
            ("empty-workspace", "no-op", "calibration-report"),
            ("wrong-boundary", "known-bad", "calibration-report"),
            ("partial-runtime-view", "partial", "calibration-report"),
            ("gold", "reference", "calibration-report"),
        ]
        assert spec.cases[0].command == "rm -rf src tests"
        assert spec.cases[0].expect.reward_max == 0.0
        assert spec.cases[1].expect.reward_max == 0.2
        assert spec.cases[2].expect.reward_range == (0.3, 0.8)
        assert spec.cases[3].expect.reward_equals == 1.0
        assert all(case.expect.flake_rate_max == 0.0 for case in spec.cases)

    def test_acceptance_live_generated_calibration_requires_commands_for_low_cases(
        self,
        tmp_path,
    ):
        """Generated live calibration fails closed without executable perturbations."""
        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        self._add_acceptance_live_generated_calibration(task)

        issues = check_task(
            task,
            validation_level="acceptance-live",
            sandbox_type="docker",
        )

        assert (
            "acceptance-live calibration.from report.cases[0].command must be "
            "declared for generated no-op live calibration cases"
        ) in issues
        assert (
            "acceptance-live calibration.from report.cases[1].command must be "
            "declared for generated known-bad live calibration cases"
        ) in issues
        assert (
            "acceptance-live calibration.from report.cases[2].command must be "
            "declared for generated partial live calibration cases"
        ) in issues

    @pytest.mark.asyncio
    async def test_acceptance_live_generated_calibration_persists_case_summaries(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Generated calibration cases are marked in the live report artifact."""
        from benchflow.task import acceptance_live as live_mod

        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        self._add_calibration_report_commands(task)
        self._add_acceptance_live_generated_calibration(task)
        spec, issues = live_mod.parse_live_acceptance_spec(
            task,
            evidence=TaskDocument.from_path(task / "task.md").benchflow["evidence"],
        )
        assert issues == []
        assert spec is not None
        rewards = {
            "no-op": 0.0,
            "known-bad": 0.2,
            "partial": 0.5,
            "reference": 1.0,
        }

        async def fake_run_single_case(*_args, case, **_kwargs):
            return rewards[case.case_type], None

        monkeypatch.setattr(live_mod, "_stage_current_worktree", lambda _tmp: tmp_path)
        monkeypatch.setattr(live_mod, "_run_single_case", fake_run_single_case)

        run_issues = await live_mod._run_live_acceptance_checks(
            task,
            sandbox_type="docker",
            spec=spec,
        )

        assert run_issues == []
        report = json.loads(
            (task / "evidence" / "acceptance" / "live-report.json").read_text()
        )
        assert report["summary"] == {
            "total_runs": 4,
            "passed_runs": 4,
            "failed_runs": 0,
            "flake_rate": 0.0,
            "min_reward": 0.0,
            "max_reward": 1.0,
        }
        assert {case["source"] for case in report["cases"]} == {"calibration-report"}
        assert {run["source"] for run in report["runs"]} == {"calibration-report"}
        assert [
            (case["case"], case["source"], case["status"])
            for case in report["case_summaries"]
        ] == [
            ("empty-workspace", "calibration-report", "passed"),
            ("wrong-boundary", "calibration-report", "passed"),
            ("partial-runtime-view", "calibration-report", "passed"),
            ("gold", "calibration-report", "passed"),
        ]

    @pytest.mark.asyncio
    async def test_acceptance_live_leaderboard_suitability_accepts_complete_report(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A complete live report can become leaderboard-suitable evidence."""
        from benchflow.task import acceptance_live as live_mod

        task = self._make_publication_task(tmp_path)
        (task / "oracle" / "solve.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        (task / "oracle" / "solve.sh").chmod(0o755)
        self._add_acceptance_evidence(task)
        self._add_calibration_report_commands(task)
        self._add_acceptance_live_leaderboard_evidence(task)
        spec, issues = live_mod.parse_live_acceptance_spec(
            task,
            evidence=TaskDocument.from_path(task / "task.md").benchflow["evidence"],
        )
        assert issues == []
        assert spec is not None
        rewards = {
            "oracle": 1.0,
            "reference": 1.0,
            "no-op": 0.0,
            "known-bad": 0.2,
            "partial": 0.5,
        }

        async def fake_run_single_case(*_args, case, **_kwargs):
            return rewards[case.case_type], None

        monkeypatch.setattr(live_mod, "_stage_current_worktree", lambda _tmp: tmp_path)
        monkeypatch.setattr(live_mod, "_run_single_case", fake_run_single_case)

        run_issues = await live_mod._run_live_acceptance_checks(
            task,
            sandbox_type="docker",
            spec=spec,
        )

        assert run_issues == []
        report = json.loads(
            (task / "evidence" / "acceptance" / "live-report.json").read_text()
        )
        assert report["leaderboard_suitability"] == {
            "status": "suitable",
            "required": True,
            "max_flake_rate": 0.0,
            "required_generated_calibration_types": [
                "known-bad",
                "no-op",
                "partial",
                "reference",
            ],
            "observed_generated_calibration_types": [
                "known-bad",
                "no-op",
                "partial",
                "reference",
            ],
            "checks": {
                "has_live_runs": True,
                "all_runs_passed": True,
                "flake_rate_within_limit": True,
                "has_oracle_proof": True,
                "has_reference_proof": True,
                "has_generated_calibration_coverage": True,
            },
            "issues": [],
        }

    @pytest.mark.asyncio
    async def test_acceptance_live_leaderboard_required_fails_without_calibration(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Leaderboard suitability fails closed when calibration evidence is absent."""
        from benchflow.task import acceptance_live as live_mod

        task = self._make_publication_task(tmp_path)
        (task / "oracle" / "solve.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        (task / "oracle" / "solve.sh").chmod(0o755)
        self._add_acceptance_evidence(task)
        self._add_acceptance_live_leaderboard_without_calibration(task)
        spec, issues = live_mod.parse_live_acceptance_spec(
            task,
            evidence=TaskDocument.from_path(task / "task.md").benchflow["evidence"],
        )
        assert issues == []
        assert spec is not None

        async def fake_run_single_case(*_args, **_kwargs):
            return 1.0, None

        monkeypatch.setattr(live_mod, "_stage_current_worktree", lambda _tmp: tmp_path)
        monkeypatch.setattr(live_mod, "_run_single_case", fake_run_single_case)

        run_issues = await live_mod._run_live_acceptance_checks(
            task,
            sandbox_type="docker",
            spec=spec,
        )

        assert run_issues == [
            "acceptance-live leaderboard suitability: missing generated "
            "calibration live case types: known-bad, no-op, partial, reference"
        ]

    def test_acceptance_live_oracle_case_requires_executable_oracle(self, tmp_path):
        """Oracle live cases use selected oracle/solve.sh, not solve.md notes."""
        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        self._add_acceptance_live_evidence_with_cases(
            task,
            """
        - name: live-oracle-rerun
          type: oracle
          expect:
            reward_min: 0.99
""",
        )

        issues = check_task(
            task,
            validation_level="acceptance-live",
            sandbox_type="docker",
        )

        assert any(
            "type=oracle requires executable oracle/solve.sh" in issue
            for issue in issues
        )

    @pytest.mark.asyncio
    async def test_acceptance_live_oracle_case_runs_oracle_before_verify(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Oracle live cases use the full oracle Rollout lifecycle."""
        from types import SimpleNamespace

        from benchflow.task import acceptance_live as live_mod

        task = self._make_publication_task(tmp_path)
        solve = task / "oracle" / "solve.sh"
        solve.write_text("#!/usr/bin/env bash\nexit 0\n")
        solve.chmod(0o755)
        self._add_acceptance_evidence(task)
        self._add_acceptance_live_evidence_with_cases(
            task,
            """
        - name: live-oracle-rerun
          type: oracle
          expect:
            reward_min: 0.99
""",
        )
        spec, issues = live_mod.parse_live_acceptance_spec(
            task,
            evidence=TaskDocument.from_path(task / "task.md").benchflow["evidence"],
        )
        assert issues == []
        assert spec is not None
        case = spec.cases[0]
        calls = []
        seen_configs = []
        seen_uploads = []

        class FakeRollout:
            def __init__(self, config):
                self.config = config

            async def run(self):
                calls.append("run")
                for hook in self.config.pre_agent_hooks or []:
                    await hook(SimpleNamespace(name="env"))
                return SimpleNamespace(
                    trajectory=[{"type": "oracle", "return_code": 0}],
                    rewards={"reward": 1.0},
                    error=None,
                    verifier_error=None,
                )

        class FakeRolloutFactory:
            @classmethod
            async def create(cls, config):
                calls.append("create")
                seen_configs.append(config)
                return FakeRollout(config)

        async def fake_upload_workspace(*args, **kwargs):
            calls.append("upload_workspace")
            seen_uploads.append((args, kwargs))

        monkeypatch.setattr(live_mod, "Rollout", FakeRolloutFactory)
        monkeypatch.setattr(live_mod, "_upload_workspace", fake_upload_workspace)

        result = await live_mod._run_single_case(
            task,
            sandbox_type="docker",
            workspace=spec.workspace,
            staged_worktree=tmp_path,
            case=case,
            run_index=1,
        )

        assert result.error is None
        assert result.reward == 1.0
        assert calls == ["create", "run", "upload_workspace"]
        assert len(seen_configs) == 1
        config = seen_configs[0]
        assert config.environment == "docker"
        assert config.agent == "oracle"
        assert config.primary_agent == "oracle"
        assert config.primary_model is None
        assert config.scenes[0].roles[0].name == "oracle"
        assert config.pre_agent_hooks
        assert config.rollout_name.startswith(
            "acceptance-live-publication-task-live-oracle-rerun-1-"
        )
        assert seen_uploads == [
            (
                (SimpleNamespace(name="env"),),
                {
                    "workspace": spec.workspace,
                    "staged_worktree": tmp_path,
                },
            )
        ]

    @pytest.mark.asyncio
    async def test_acceptance_live_persists_report_and_hash_sidecar(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Live checks leave durable evidence instead of temp-only results."""
        from benchflow.task import acceptance_live as live_mod

        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        self._add_acceptance_live_evidence(task)
        spec, issues = live_mod.parse_live_acceptance_spec(
            task,
            evidence=TaskDocument.from_path(task / "task.md").benchflow["evidence"],
        )
        assert issues == []
        assert spec is not None

        async def fake_run_single_case(*_args, **_kwargs):
            return 1.0, None

        monkeypatch.setattr(live_mod, "_stage_current_worktree", lambda _tmp: tmp_path)
        monkeypatch.setattr(live_mod, "_run_single_case", fake_run_single_case)

        run_issues = await live_mod._run_live_acceptance_checks(
            task,
            sandbox_type="docker",
            spec=spec,
        )

        assert run_issues == []
        report = task / "evidence" / "acceptance" / "live-report.json"
        sidecar = task / "evidence" / "acceptance" / "live-report.json.sha256"
        assert report.is_file()
        assert sidecar.is_file()
        data = json.loads(report.read_text())
        assert data["kind"] == "acceptance-live-report"
        assert data["schema_version"] == "1.0"
        assert data["benchflow_version"]
        assert data["sandbox"] == "docker"
        assert len(data["spec_sha256"]) == 64
        assert len(data["task"]["task_md_sha256"]) == 64
        assert data["workspace"] == {
            "source": "current-worktree",
            "target": "/repo",
            "staged_tree_sha256": data["workspace"]["staged_tree_sha256"],
        }
        assert len(data["workspace"]["staged_tree_sha256"]) == 64
        assert data["summary"] == {
            "total_runs": 2,
            "passed_runs": 2,
            "failed_runs": 0,
            "flake_rate": 0.0,
            "min_reward": 1.0,
            "max_reward": 1.0,
        }
        assert [run["status"] for run in data["runs"]] == ["passed", "passed"]
        assert all(len(run["sha256"]) == 64 for run in data["runs"])
        expected_digest = sha256(report.read_bytes()).hexdigest()
        assert sidecar.read_text() == (
            f"{expected_digest}  evidence/acceptance/live-report.json\n"
        )

    @pytest.mark.asyncio
    async def test_acceptance_live_flake_threshold_allows_tolerated_failed_rerun(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Repeated live cases can measure flake rate instead of failing per-run."""
        from benchflow.task import acceptance_live as live_mod

        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        self._add_acceptance_live_evidence_with_cases(
            task,
            """
        - name: live-reference-verifier
          type: reference
          reruns: 2
          expect:
            reward_min: 0.99
            flake_rate_max: 0.5
""",
        )
        spec, issues = live_mod.parse_live_acceptance_spec(
            task,
            evidence=TaskDocument.from_path(task / "task.md").benchflow["evidence"],
        )
        assert issues == []
        assert spec is not None
        results = iter([(None, "transient verifier failure"), (1.0, None)])

        async def fake_run_single_case(*_args, **_kwargs):
            return next(results)

        monkeypatch.setattr(live_mod, "_stage_current_worktree", lambda _tmp: tmp_path)
        monkeypatch.setattr(live_mod, "_run_single_case", fake_run_single_case)

        run_issues = await live_mod._run_live_acceptance_checks(
            task,
            sandbox_type="docker",
            spec=spec,
        )

        assert run_issues == []
        report = json.loads(
            (task / "evidence" / "acceptance" / "live-report.json").read_text()
        )
        assert report["summary"]["failed_runs"] == 1
        assert report["case_summaries"] == [
            {
                "case": "live-reference-verifier",
                "type": "reference",
                "source": "declared",
                "total_runs": 2,
                "passed_runs": 1,
                "failed_runs": 1,
                "flake_rate": 0.5,
                "flake_rate_max": 0.5,
                "min_reward": 1.0,
                "max_reward": 1.0,
                "status": "passed",
            }
        ]

    @pytest.mark.asyncio
    async def test_acceptance_live_flake_threshold_rejects_excessive_failures(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A repeated live case fails when observed flake exceeds its threshold."""
        from benchflow.task import acceptance_live as live_mod

        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        self._add_acceptance_live_evidence_with_cases(
            task,
            """
        - name: live-reference-verifier
          type: reference
          reruns: 2
          expect:
            reward_min: 0.99
            flake_rate_max: 0.0
""",
        )
        spec, issues = live_mod.parse_live_acceptance_spec(
            task,
            evidence=TaskDocument.from_path(task / "task.md").benchflow["evidence"],
        )
        assert issues == []
        assert spec is not None
        results = iter([(None, "transient verifier failure"), (1.0, None)])

        async def fake_run_single_case(*_args, **_kwargs):
            return next(results)

        monkeypatch.setattr(live_mod, "_stage_current_worktree", lambda _tmp: tmp_path)
        monkeypatch.setattr(live_mod, "_run_single_case", fake_run_single_case)

        run_issues = await live_mod._run_live_acceptance_checks(
            task,
            sandbox_type="docker",
            spec=spec,
        )

        assert run_issues == [
            "acceptance-live case 'live-reference-verifier' flake_rate "
            "0.5 exceeds flake_rate_max 0"
        ]

    @pytest.mark.asyncio
    async def test_acceptance_live_flake_threshold_reports_dep_install_hint(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Guards PR #1's no-network verifier dependency diagnostic."""
        from benchflow.task import acceptance_live as live_mod

        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        self._add_acceptance_live_evidence_with_cases(
            task,
            """
        - name: live-reference-verifier
          type: reference
          reruns: 1
          expect:
            reward_min: 0.99
            flake_rate_max: 0.0
""",
        )
        spec, issues = live_mod.parse_live_acceptance_spec(
            task,
            evidence=TaskDocument.from_path(task / "task.md").benchflow["evidence"],
        )
        assert issues == []
        assert spec is not None

        async def fake_run_single_case(*_args, **_kwargs):
            return live_mod.LiveAcceptanceRunResult(
                reward=None,
                error=(
                    "verifier crashed: verifier exited with rc=1; no reward file\n"
                    "dependency install failed (see verifier/test-stdout.txt in "
                    "the run artifacts for resolver output)"
                ),
                verifier_error_category="verifier_dep_install",
                diagnostic_code="verifier_dep_install",
                artifact_hint="verifier/test-stdout.txt",
            )

        monkeypatch.setattr(live_mod, "_stage_current_worktree", lambda _tmp: tmp_path)
        monkeypatch.setattr(live_mod, "_run_single_case", fake_run_single_case)

        run_issues = await live_mod._run_live_acceptance_checks(
            task,
            sandbox_type="docker",
            spec=spec,
        )

        assert run_issues == [
            "acceptance-live case 'live-reference-verifier' flake_rate "
            "1 exceeds flake_rate_max 0; first failed run indicates verifier "
            "dependency install failed (see verifier/test-stdout.txt in the "
            "run artifacts)"
        ]
        report = json.loads(
            (task / "evidence" / "acceptance" / "live-report.json").read_text()
        )
        assert report["runs"][0]["verifier_error_category"] == "verifier_dep_install"
        assert report["runs"][0]["diagnostic_code"] == "verifier_dep_install"
        assert report["runs"][0]["artifact_hint"] == "verifier/test-stdout.txt"

    def test_acceptance_checks_declared_evidence_hashes(self, tmp_path):
        """Acceptance evidence hashes must match local declared artifacts."""
        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        trajectory = task / "evidence" / "calibration" / "gold-trajectory.jsonl"
        trajectory.write_text('{"type": "agent_message", "text": "tampered"}\n')

        issues = check_task(task, validation_level="acceptance")

        assert any("sha256 mismatch" in issue for issue in issues)

    def test_acceptance_requires_primary_evidence_pins(self, tmp_path):
        """Primary acceptance evidence must be listed with SHA-256 pins."""
        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        task_md = task / "task.md"
        oracle = task / "evidence" / "calibration" / "oracle-run.json"
        oracle_sha = sha256(oracle.read_bytes()).hexdigest()
        text = task_md.read_text()
        text = text.replace(
            "      - path: evidence/calibration/oracle-run.json\n"
            f"        sha256: {oracle_sha}\n",
            "",
        )
        task_md.write_text(text)

        issues = check_task(task, validation_level="acceptance")

        assert any("oracle_runs.artifact must be listed" in issue for issue in issues)

    def test_acceptance_checks_verifier_stability_report(self, tmp_path):
        """Verifier acceptance needs rerun evidence, not only scalar metadata."""
        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        report = task / "evidence" / "calibration" / "verifier-stability-report.json"
        data = json.loads(report.read_text())
        data["runs"][0]["status"] = "failed"
        data["runs"][0]["reward"] = 0.0
        report.write_text(json.dumps(data) + "\n")

        issues = check_task(task, validation_level="acceptance")

        assert any("acceptance verifier.report" in issue for issue in issues)

    def test_acceptance_checks_oracle_artifact_reward(self, tmp_path):
        """Oracle acceptance artifacts must prove the required reward."""
        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        oracle = task / "evidence" / "calibration" / "oracle-run.json"
        oracle.write_text('{"reward": 0.5, "status": "passed"}\n')

        issues = check_task(task, validation_level="acceptance")

        assert any("oracle_runs.artifact.reward" in issue for issue in issues)

    def test_acceptance_checks_review_artifact_status(self, tmp_path):
        """Review acceptance artifacts must agree with declared review status."""
        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        review = task / "evidence" / "calibration" / "review.json"
        review.write_text(
            json.dumps(
                {
                    "kind": "acceptance-review",
                    "anti_cheat": "failed",
                    "instruction_alignment": "passed",
                }
            )
            + "\n"
        )

        issues = check_task(task, validation_level="acceptance")

        assert any("review.artifact.anti_cheat" in issue for issue in issues)

    def test_acceptance_checks_calibration_artifact_reward(self, tmp_path):
        """Calibration examples must match their local reference artifact."""
        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        gold = task / "evidence" / "calibration" / "gold-result.json"
        gold.write_text('{"expected_reward": 0.25}\n')

        issues = check_task(task, validation_level="acceptance")

        assert any(
            "human_or_reference_examples[0].artifact.expected_reward" in issue
            for issue in issues
        )

    def test_acceptance_checks_calibration_report_cases(self, tmp_path):
        """Calibration reports must prove declared no-op/bad/partial/reference bounds."""
        task = self._make_publication_task(tmp_path)
        self._add_acceptance_evidence(task)
        report = task / "evidence" / "calibration" / "calibration-report.json"
        data = json.loads(report.read_text())
        data["cases"][0]["reward"] = 0.5
        report.write_text(json.dumps(data) + "\n")

        issues = check_task(task, validation_level="acceptance")

        assert any("exceeds no_op_reward_max" in issue for issue in issues)


class TestTaskPathAliases:
    def test_native_dirs_are_preferred_over_legacy_aliases(self, tmp_path):
        """Guards oracle/verifier as native names with legacy aliases intact."""
        task = tmp_path / "task"
        (task / "oracle").mkdir(parents=True)
        (task / "solution").mkdir()
        (task / "verifier").mkdir()
        (task / "tests").mkdir()

        paths = TaskPaths(task)

        assert paths.solution_dir == task / "oracle"
        assert paths.tests_dir == task / "verifier"
        assert paths.uses_native_oracle_dir is True
        assert paths.uses_native_verifier_dir is True

    def test_legacy_dirs_remain_supported(self, tmp_path):
        """Guards compatibility for existing tests/ and solution/ packages."""
        task = tmp_path / "task"
        (task / "solution").mkdir(parents=True)
        (task / "tests").mkdir()

        paths = TaskPaths(task)

        assert paths.solution_dir == task / "solution"
        assert paths.tests_dir == task / "tests"
        assert paths.uses_native_oracle_dir is False
        assert paths.uses_native_verifier_dir is False


class TestInitTask:
    """init_task(name, ...) -> Path"""

    def test_default_creates_task_md_structure(self, tmp_path):
        """Guards commit 67378ddd's task.md default authoring standard."""
        task = init_task("my-task", parent_dir=tmp_path)

        assert task == tmp_path / "my-task"
        assert (task / "task.md").exists()
        assert not (task / "task.toml").exists()
        assert not (task / "instruction.md").exists()
        assert (task / "environment" / "Dockerfile").exists()
        assert (task / "verifier" / "test.sh").exists()
        assert (task / "verifier" / "verifier.md").exists()
        assert (task / "verifier" / "rubrics" / "verifier.md").exists()
        assert (task / "verifier" / "rubrics" / "verifier.toml").exists()
        assert (task / "verifier" / "test_outputs.py").exists()
        assert (task / "oracle" / "solve.sh").exists()
        assert not (task / "tests").exists()
        assert not (task / "solution").exists()

    def test_creates_legacy_structure_when_requested(self, tmp_path):
        task = init_task("my-task", parent_dir=tmp_path, task_format="legacy")
        assert task == tmp_path / "my-task"
        assert (task / "task.toml").exists()
        assert (task / "instruction.md").exists()
        assert (task / "environment" / "Dockerfile").exists()
        assert (task / "tests" / "test.sh").exists()
        assert (task / "tests" / "test_outputs.py").exists()
        assert (task / "solution" / "solve.sh").exists()

    def test_tasks_init_cli_defaults_to_task_md(self, tmp_path):
        """Guards the CLI default for the task.md standard."""
        from typer.testing import CliRunner

        result = CliRunner().invoke(
            app,
            ["tasks", "init", "cli-default", "--dir", str(tmp_path)],
        )

        task = tmp_path / "cli-default"
        assert result.exit_code == 0, result.output
        assert (task / "task.md").exists()
        assert (task / "verifier" / "test.sh").exists()
        assert (task / "oracle" / "solve.sh").exists()
        assert not (task / "task.toml").exists()
        assert not (task / "instruction.md").exists()

    def test_tasks_check_cli_sandbox_accepts_verifier_package(self):
        """`bench tasks check --sandbox` accepts executable verifier strategies."""
        from typer.testing import CliRunner

        task = Path(
            "docs/examples/task-standard/benchflow-wanted-features/"
            "verifier-package-reward-contract"
        )

        result = CliRunner().invoke(
            app,
            ["tasks", "check", str(task), "--sandbox", "docker"],
        )

        assert result.exit_code == 0, result.output
        assert "valid" in result.output

    def test_tasks_check_cli_sandbox_accepts_prompt_user_dogfood(self):
        """`prompt-user-semantics` now exercises supported linear user semantics."""
        from typer.testing import CliRunner

        task = Path(
            "docs/examples/task-standard/benchflow-wanted-features/"
            "prompt-user-semantics"
        )

        result = CliRunner().invoke(
            app,
            ["tasks", "check", str(task), "--sandbox", "docker"],
        )

        assert result.exit_code == 0, result.output
        assert "valid" in result.output

    def test_tasks_check_cli_publication_grade_accepts_ors_dogfood(self):
        """The CLI exposes publication-grade validation as a first-class level."""
        from typer.testing import CliRunner

        task = Path(
            "docs/examples/task-standard/benchflow-wanted-features/"
            "ors-episode-reward-contract"
        )

        result = CliRunner().invoke(
            app,
            ["tasks", "check", str(task), "--level", "publication-grade"],
        )

        assert result.exit_code == 0, result.output
        assert "valid (publication-grade)" in result.output

    @pytest.mark.parametrize("task_name", WANTED_FEATURE_TASKS)
    def test_tasks_check_cli_acceptance_level_accepts_wanted_feature_dogfood(
        self,
        task_name,
    ):
        """Guards PR #1's wanted-feature static acceptance dogfood coverage."""
        from typer.testing import CliRunner

        task = WANTED_FEATURE_DIR / task_name

        result = CliRunner().invoke(
            app,
            ["tasks", "check", str(task), "--level", "acceptance"],
        )

        assert result.exit_code == 0, result.output
        assert "valid (acceptance)" in result.output

    @pytest.mark.parametrize("task_name", WANTED_FEATURE_ACCEPTANCE_LIVE_TASKS)
    def test_tasks_check_cli_acceptance_live_accepts_dogfood_with_live_runner(
        self,
        task_name,
        monkeypatch,
    ):
        """The real dogfood task declares live cases; the runner owns execution."""
        from typer.testing import CliRunner

        task = WANTED_FEATURE_DIR / task_name
        calls = []

        def fake_live_runner(
            task_dir, *, sandbox_type, evidence, report_output=None, write_report=True
        ):
            calls.append(
                (task_dir, sandbox_type, evidence, report_output, write_report)
            )
            return []

        monkeypatch.setattr(
            "benchflow._utils.task_authoring.run_live_acceptance_checks",
            fake_live_runner,
        )

        result = CliRunner().invoke(
            app,
            [
                "tasks",
                "check",
                str(task),
                "--level",
                "acceptance-live",
                "--sandbox",
                "docker",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "valid (acceptance-live)" in result.output
        assert calls
        assert calls[0][0] == task
        assert calls[0][1] == "docker"
        assert "acceptance_live" in calls[0][2]
        assert calls[0][3] is None

    def test_verifier_package_acceptance_live_uses_reference_verifier_case(self):
        """Guards PR #1's second acceptance-live wanted-feature dogfood."""
        from benchflow.task import acceptance_live as live_mod

        task = WANTED_FEATURE_DIR / "verifier-package-reward-contract"
        spec, issues = live_mod.parse_live_acceptance_spec(
            task,
            evidence=TaskDocument.from_path(task / "task.md").benchflow["evidence"],
        )

        assert issues == []
        assert spec is not None
        assert spec.report_path == Path("evidence/acceptance/live-report.json")
        assert spec.workspace.target == "/repo"
        assert [(case.name, case.case_type, case.source) for case in spec.cases] == [
            ("live-reference-verifier", "reference", "declared")
        ]
        assert spec.cases[0].expect.reward_min == 0.99
        assert spec.cases[0].expect.flake_rate_max == 0.0

    @pytest.mark.parametrize("task_name", WANTED_FEATURE_ACCEPTANCE_LIVE_TASKS)
    def test_acceptance_live_reports_have_matching_sha256_sidecars(
        self,
        task_name,
    ):
        """Guards PR #1's checked-in acceptance-live report artifacts."""
        from benchflow.task import acceptance_live as live_mod

        task = WANTED_FEATURE_DIR / task_name
        task_document = TaskDocument.from_path(task / "task.md")
        evidence = task_document.benchflow["evidence"]
        live = evidence["acceptance_live"]
        report_rel = live["report"]
        report = task / report_rel
        sidecar = report.with_suffix(report.suffix + ".sha256")
        spec, issues = live_mod.parse_live_acceptance_spec(task, evidence=evidence)

        assert report.is_file()
        assert sidecar.read_text() == (
            f"{sha256(report.read_bytes()).hexdigest()}  {report_rel}\n"
        )
        assert issues == []
        assert spec is not None

        report_json = json.loads(report.read_text())
        assert report_json["kind"] == "acceptance-live-report"
        assert report_json["spec_sha256"] == live_mod._spec_sha256(spec)
        assert (
            report_json["task"]["task_md_sha256"]
            == sha256((task / "task.md").read_bytes()).hexdigest()
        )
        assert (
            report_json["task"]["verifier_sha256"]
            == sha256(TaskPaths(task).test_path.read_bytes()).hexdigest()
        )
        if report_json["workspace"]["source"] == "current-worktree":
            staged_tree_sha = report_json["workspace"]["staged_tree_sha256"]
            assert isinstance(staged_tree_sha, str)
            assert len(staged_tree_sha) == 64
            int(staged_tree_sha, 16)

    def test_tasks_check_cli_acceptance_live_forwards_report_output(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Guards PR #1's CLI report-output forwarding contract."""
        from typer.testing import CliRunner

        task = Path(
            "docs/examples/task-standard/benchflow-wanted-features/"
            "runtime-capability-gate"
        )
        output = tmp_path / "live-report.json"
        calls = []

        def fake_live_runner(
            task_dir, *, sandbox_type, evidence, report_output=None, write_report=True
        ):
            calls.append(
                (task_dir, sandbox_type, evidence, report_output, write_report)
            )
            return []

        monkeypatch.setattr(
            "benchflow._utils.task_authoring.run_live_acceptance_checks",
            fake_live_runner,
        )

        result = CliRunner().invoke(
            app,
            [
                "tasks",
                "check",
                str(task),
                "--level",
                "acceptance-live",
                "--sandbox",
                "docker",
                "--report-output",
                str(output),
            ],
        )

        assert result.exit_code == 0, result.output
        assert calls
        assert calls[0][0] == task
        assert calls[0][1] == "docker"
        assert "acceptance_live" in calls[0][2]
        assert calls[0][3] == output
        assert calls[0][4] is True

    def test_tasks_check_cli_acceptance_live_forwards_no_report_write(
        self,
        monkeypatch,
    ):
        """`--no-report-write` reaches the live runner as write_report=False."""
        from typer.testing import CliRunner

        task = Path(
            "docs/examples/task-standard/benchflow-wanted-features/"
            "runtime-capability-gate"
        )
        calls = []

        def fake_live_runner(
            task_dir, *, sandbox_type, evidence, report_output=None, write_report=True
        ):
            calls.append((report_output, write_report))
            return []

        monkeypatch.setattr(
            "benchflow._utils.task_authoring.run_live_acceptance_checks",
            fake_live_runner,
        )

        result = CliRunner().invoke(
            app,
            [
                "tasks",
                "check",
                str(task),
                "--level",
                "acceptance-live",
                "--sandbox",
                "docker",
                "--no-report-write",
            ],
        )

        assert result.exit_code == 0, result.output
        assert calls == [(None, False)]

    def test_tasks_check_cli_rejects_unknown_runtime_sandbox(self):
        """A typoed sandbox cannot produce a green runtime-capability check."""
        from typer.testing import CliRunner

        task = Path(
            "docs/examples/task-standard/benchflow-wanted-features/"
            "runtime-capability-gate"
        )

        result = CliRunner().invoke(
            app,
            [
                "tasks",
                "check",
                str(task),
                "--level",
                "runtime-capability",
                "--sandbox",
                "not-a-backend",
            ],
        )

        assert result.exit_code == 1
        assert "unknown sandbox backend" in result.output

    def test_creates_task_md_structure(self, tmp_path):
        """Guards commit 67378ddd's 2026-06-04 task.md scaffold entrypoint."""
        task = init_task(
            "my-task-md",
            parent_dir=tmp_path,
            task_format="task-md",
        )

        assert (task / "task.md").exists()
        assert not (task / "task.toml").exists()
        assert not (task / "instruction.md").exists()
        assert (task / "environment" / "Dockerfile").exists()
        assert (task / "verifier" / "test.sh").exists()

    def test_check_task_accepts_authored_task_md(self, tmp_path):
        """Guards commit 67378ddd's 2026-06-04 task.md check path."""
        task = init_task(
            "authored-task-md",
            parent_dir=tmp_path,
            no_pytest=True,
            no_oracle=True,
            task_format="task-md",
        )
        (task / "task.md").write_text(
            """---
version: "1.0"
metadata:
  category: capability
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
environment:
  cpus: 1
---
## prompt

Create hello.txt.
"""
        )
        (task / "verifier" / "test.sh").write_text(
            '#!/bin/bash\necho "1.0" > /logs/verifier/reward.txt\n'
        )
        (task / "verifier" / "verifier.md").unlink()
        shutil.rmtree(task / "verifier" / "rubrics")

        assert check_task(task) == []

    def test_task_md_scaffold_flags_verifier_package_placeholders(self, tmp_path):
        """Guards PR #1's new-benchmark task.md scaffold contract."""
        task = init_task("native-scaffold", parent_dir=tmp_path, task_format="task-md")

        issues = check_task(task)

        assert issues, "scaffold must not pass check_task before editing"
        joined = "\n".join(issues)
        assert "task.md contains unreplaced placeholder" in joined
        assert "verifier/test.sh contains unreplaced placeholder" in joined
        assert "verifier/test_outputs.py contains unreplaced placeholder" in joined
        assert "verifier/verifier.md contains unreplaced placeholder" in joined
        assert "verifier/rubrics/verifier.md contains unreplaced placeholder" in joined
        assert (
            "verifier/rubrics/verifier.toml contains unreplaced placeholder" in joined
        )
        assert "oracle/solve.sh contains unreplaced placeholder" in joined

    def test_authored_task_md_scaffold_reaches_publication_grade(self, tmp_path):
        """Guards PR #1's new-benchmark task.md authoring path."""
        task = init_task(
            "native-publication",
            parent_dir=tmp_path,
            no_pytest=True,
            task_format="task-md",
        )
        (task / "task.md").write_text(
            """---
schema_version: "1.3"
task:
  name: benchflow/native-publication
metadata:
  category: capability
environment:
  workdir: /app
  cpus: 1
  memory_mb: 2048
verifier:
  timeout_sec: 120
agents:
  roles:
    solver:
      agent: claude-agent-acp
scenes:
  - name: solve
    turns:
      - role: solver
benchflow:
  authoring:
    profiles: [code-change]
---
# native-publication

## prompt

Create `/app/output.txt` containing `done`.
"""
        )
        (task / "verifier" / "test.sh").write_text(
            """#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
echo "1.0" > /logs/verifier/reward.txt
cat > /logs/verifier/reward.json <<'JSON'
{"reward": 1.0, "task_success": 1.0}
JSON
cat > /logs/verifier/reward-details.json <<'JSON'
{"criteria": [{"id": "task_success", "score": 1.0, "reason": "authored scaffold"}]}
JSON
"""
        )
        (task / "oracle" / "solve.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\nprintf 'done\\n' > /app/output.txt\n"
        )
        (task / "verifier" / "verifier.md").write_text(
            """---
document_version: "0.3"
verifier:
  name: native-publication
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_mean
    dimensions:
      task_success: {weight: 1.0, source: deterministic}
  outputs:
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/reward-details.json
    aggregate_policy:
      method: weighted_mean
      metrics:
        task_success: 1.0
---

## verifier intent

Check that the authored scaffold has a deterministic verifier contract.
"""
        )
        (task / "verifier" / "rubrics" / "verifier.md").write_text(
            "# Native Publication Rubric\n\n- `task_success`: output file exists.\n"
        )
        (task / "verifier" / "rubrics" / "verifier.toml").write_text(
            """version = "0.1"

[[criteria]]
id = "task_success"
description = "The agent creates the requested output file."
weight = 1.0

[scoring]
method = "weighted_mean"
"""
        )

        assert check_task(task) == []
        assert check_task(task, validation_level="publication-grade") == []

    def test_scaffold_fails_check_until_placeholders_replaced(self, tmp_path):
        """Freshly scaffolded task must FAIL `bench tasks check` (#360).

        Otherwise authors can `bench tasks init foo && bench tasks check foo`
        and get a green ✓ on a task that auto-passes with reward 1.0 —
        silent false positives in eval sets.
        """
        task = init_task("scaffolded", parent_dir=tmp_path, task_format="legacy")
        issues = check_task(task)
        assert issues, "scaffold must not pass check_task before editing"
        # instruction.md, tests/test.sh and solution/solve.sh all carry the
        # placeholder marker — each must be flagged so the author sees what
        # to replace.
        joined = "\n".join(issues)
        assert "instruction.md" in joined
        assert "tests/test.sh" in joined
        assert "solution/solve.sh" in joined

    def test_scaffold_passes_check_after_placeholders_replaced(self, tmp_path):
        """Once the author replaces every [REPLACE: ...] marker, check passes."""
        task = init_task("editable", parent_dir=tmp_path, task_format="legacy")
        (task / "instruction.md").write_text("# editable\n\nDo the thing.\n")
        (task / "tests" / "test.sh").write_text(
            '#!/bin/bash\necho "1.0" > /logs/verifier/reward.txt\n'
        )
        (task / "tests" / "test_outputs.py").write_text(
            "def test_output_contract():\n    assert True\n"
        )
        (task / "solution" / "solve.sh").write_text(
            "#!/bin/bash\ntouch /app/output.txt\n"
        )
        assert check_task(task) == []

    def test_scaffold_test_sh_defaults_to_failure(self, tmp_path):
        """Scaffolded verifier writes 0.0, not 1.0 (#360).

        Fail-closed default means even if check_task were skipped, the task
        cannot give a passing reward without explicit author edits.
        """
        task = init_task("fail-closed", parent_dir=tmp_path, task_format="legacy")
        test_sh = (task / "tests" / "test.sh").read_text()
        assert 'echo "0.0"' in test_sh
        assert 'echo "1.0"' not in test_sh

    def test_scaffold_solution_does_not_satisfy_verifier(self, tmp_path):
        """The placeholder solution must NOT make the placeholder verifier
        pass — issue #360 calls this out explicitly: solution and test must
        not collide on a default-pass.

        We simulate by reading both files and checking that solve.sh exits
        nonzero (it does not produce side effects) while test.sh writes 0.0.
        """
        task = init_task("collision", parent_dir=tmp_path, task_format="legacy")
        solve = (task / "solution" / "solve.sh").read_text()
        test_sh = (task / "tests" / "test.sh").read_text()
        # Solve.sh must not write a passing reward itself (no echo of 1.0
        # into the reward file).
        assert "reward.txt" not in solve
        assert 'echo "1.0"' not in solve
        # Test.sh must default to 0.0 reward.
        assert 'echo "0.0"' in test_sh
        # And the solve script signals it's a placeholder by exiting nonzero.
        assert "exit 1" in solve

    def test_scaffold_pytest_template_fails(self, tmp_path):
        """The pytest verifier template fails until edited (#360)."""
        task = init_task("py-fail", parent_dir=tmp_path, task_format="legacy")
        text = (task / "tests" / "test_outputs.py").read_text()
        assert "pytest.fail" in text
        assert "assert True" not in text

    def test_no_pytest_skips_test_outputs(self, tmp_path):
        task = init_task(
            "no-pytest",
            parent_dir=tmp_path,
            no_pytest=True,
            task_format="legacy",
        )
        assert not (task / "tests" / "test_outputs.py").exists()
        assert (task / "tests" / "test.sh").exists()

    def test_no_oracle_skips_native_oracle_dir(self, tmp_path):
        task = init_task(
            "no-oracle",
            parent_dir=tmp_path,
            no_oracle=True,
        )
        assert not (task / "oracle").exists()

    def test_no_solution_remains_legacy_compat_alias(self, tmp_path):
        task = init_task(
            "no-sol",
            parent_dir=tmp_path,
            no_solution=True,
            task_format="legacy",
        )
        assert not (task / "solution").exists()

    def test_test_sh_is_executable(self, tmp_path):
        task = init_task("exec-test", parent_dir=tmp_path, task_format="legacy")
        import os

        assert os.access(task / "tests" / "test.sh", os.X_OK)

    def test_solve_sh_is_executable(self, tmp_path):
        task = init_task("exec-sol", parent_dir=tmp_path, task_format="legacy")
        import os

        assert os.access(task / "solution" / "solve.sh", os.X_OK)

    def test_raises_on_existing_dir(self, tmp_path):
        (tmp_path / "dupe").mkdir()
        with pytest.raises(FileExistsError):
            init_task("dupe", parent_dir=tmp_path)

    def test_creates_parent_dirs(self, tmp_path):
        task = init_task("deep", parent_dir=tmp_path / "nested" / "tasks")
        assert task.exists()


class TestMigrateTask:
    """migrate_task_to_task_md(task_dir) -> TaskMigrationResult"""

    def _make_legacy_task(self, tmp_path: Path) -> Path:
        task = tmp_path / "legacy-task"
        task.mkdir()
        (task / "task.toml").write_text(
            """schema_version = "1.3"
artifacts = [{ source = "/logs/artifacts", destination = "artifacts" }]

[metadata]
category = "migration"

[agent]
timeout_sec = 300

[verifier]
timeout_sec = 120

[verifier.hardening]
cleanup_conftests = false

[environment]
network_mode = "no-network"
cpus = 2
memory_mb = 4096

[solution.env]
MODE = "legacy-oracle"
"""
        )
        (task / "instruction.md").write_text("# Legacy prompt\n\nCreate hello.txt.\n")
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        (task / "tests").mkdir()
        (task / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
        return task

    def test_migrate_preserves_config_and_prompt_without_deleting_legacy(
        self, tmp_path
    ):
        """Guards commit 67378ddd's 2026-06-04 task.md migration."""
        task = self._make_legacy_task(tmp_path)
        legacy_config = TaskConfig.model_validate_toml((task / "task.toml").read_text())

        result = migrate_task_to_task_md(task)
        document = TaskDocument.from_path(result.task_md)

        assert result.removed_legacy is False
        assert (task / "task.toml").exists()
        assert (task / "instruction.md").exists()
        assert document.config.model_dump() == legacy_config.model_dump()
        assert document.instruction == "# Legacy prompt\n\nCreate hello.txt."
        assert document.frontmatter["oracle"]["env"] == {"MODE": "legacy-oracle"}
        assert "solution" not in document.frontmatter
        assert check_task(task) == []

    def test_migrate_preserves_foreign_extension_keys_under_compat(self, tmp_path):
        """Guards the task.md standard against lossy Harbor fork imports."""
        task = self._make_legacy_task(tmp_path)
        task_toml = task / "task.toml"
        task_toml.write_text(
            'harbor_ext = "kept"\n'
            + task_toml.read_text()
            + """
[[steps]]
name = "phase-one"
runner = "harbor-step-runner"

[environment.modal]
image = "registry.example.com/task:latest"

[verifier.reward_kit]
metric = "exact_match"
"""
        )

        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            TaskConfig.model_validate_toml(task_toml.read_text())

        result = migrate_task_to_task_md(task)
        document = TaskDocument.from_path(result.task_md)

        compat = document.benchflow["compat"]
        assert compat["source"] == "legacy"
        assert compat["extra_paths"] == [
            "environment.modal.image",
            "harbor_ext",
            "steps[0].runner",
            "verifier.reward_kit.metric",
        ]
        assert compat["extra"] == {
            "harbor_ext": "kept",
            "environment": {
                "modal": {"image": "registry.example.com/task:latest"},
            },
            "steps": [{"runner": "harbor-step-runner"}],
            "verifier": {"reward_kit": {"metric": "exact_match"}},
        }
        assert document.config.environment.cpus == 2
        assert document.config.steps is not None
        assert document.config.steps[0].name == "phase-one"
        assert check_task(task) == []

    def test_migrate_can_remove_legacy_pair_after_verified_roundtrip(self, tmp_path):
        """Guards explicit cleanup when adopting task.md as the canonical file."""
        task = self._make_legacy_task(tmp_path)

        result = migrate_task_to_task_md(task, remove_legacy=True)

        assert result.removed_legacy is True
        assert (task / "task.md").exists()
        assert not (task / "task.toml").exists()
        assert not (task / "instruction.md").exists()
        assert check_task(task) == []

    def test_migrate_refuses_to_overwrite_existing_task_md(self, tmp_path):
        """Guards commit 67378ddd's migration against task.md replacement."""
        task = self._make_legacy_task(tmp_path)
        (task / "task.md").write_text("---\n---\n\nExisting document.\n")

        with pytest.raises(FileExistsError):
            migrate_task_to_task_md(task)

    def test_tasks_migrate_cli_writes_task_md(self, tmp_path):
        """Guards the CLI adoption path for legacy task authors."""
        from typer.testing import CliRunner

        task = self._make_legacy_task(tmp_path)

        result = CliRunner().invoke(app, ["tasks", "migrate", str(task)])

        assert result.exit_code == 0, result.output
        assert (task / "task.md").exists()
        assert "kept task.toml and instruction.md" in result.output


class TestNormalizeTaskMd:
    """bench tasks normalize expands minimal task.md authoring."""

    def _copy_with_minimal_task_md(self, tmp_path: Path, name: str) -> Path:
        source = Path("docs/examples/task-standard/benchflow-wanted-features") / name
        task = tmp_path / name
        shutil.copytree(source, task)
        (task / "task.md").write_text((source / "task.min.md").read_text())
        return task

    def test_normalize_task_md_writes_canonical_contract(self, tmp_path: Path) -> None:
        """Guards commit 00b32e2a's handoff goal for generated task contracts."""
        task = self._copy_with_minimal_task_md(tmp_path, "runtime-capability-gate")

        result = normalize_task_md(task, write=True)

        assert result.output_path == task / "task.md"
        document = TaskDocument.from_path(task / "task.md")
        assert document.config.task is not None
        assert document.config.task.name == "benchflow-wanted/runtime-capability-gate"
        assert document.config.environment.workdir == "/repo"
        assert document.benchflow["authoring"]["profiles"] == [
            "code-change",
            "multi-agent",
            "acceptance-live",
            "leaderboard-local",
        ]
        assert document.benchflow["evidence"]["oracle_runs"]["artifact"] == (
            "evidence/acceptance/oracle-run.json"
        )
        assert check_task(task, validation_level="acceptance") == []

    @pytest.mark.parametrize(
        "name",
        [
            "runtime-capability-gate",
            "compat-export-loss-reports",
            "prompt-user-semantics",
        ],
    )
    def test_minimal_real_dogfood_tasks_normalize_to_publication_grade(
        self,
        tmp_path: Path,
        name: str,
    ) -> None:
        """Guards commit 00b32e2a's handoff goal on real BenchFlow tasks."""
        task = self._copy_with_minimal_task_md(tmp_path, name)

        normalize_task_md(task, write=True)

        assert check_task(task, validation_level="publication-grade") == []

    def test_tasks_normalize_cli_prints_normalized_task_md(
        self,
        tmp_path: Path,
    ) -> None:
        """Guards commit 00b32e2a's handoff goal for the CLI normalizer."""
        from typer.testing import CliRunner

        task = self._copy_with_minimal_task_md(tmp_path, "compat-export-loss-reports")

        result = CliRunner().invoke(app, ["tasks", "normalize", str(task)])

        assert result.exit_code == 0, result.output
        assert result.output.startswith("---\n")
        assert "profile:" not in result.output
        assert "schema_version: '1.3'" in result.output
        assert "benchflow-wanted/compat-export-loss-reports" in result.output

    def test_tasks_normalize_cli_writes_output_path(self, tmp_path: Path) -> None:
        """Guards commit 00b32e2a's handoff goal for inspectable normalized files."""
        from typer.testing import CliRunner

        task = self._copy_with_minimal_task_md(tmp_path, "prompt-user-semantics")
        output = tmp_path / "normalized" / "task.md"

        result = CliRunner().invoke(
            app,
            ["tasks", "normalize", str(task), "--output", str(output)],
        )

        assert result.exit_code == 0, result.output
        assert output.exists()
        assert "Normalized:" in result.output
        assert TaskDocument.from_path(output).config.task is not None


class TestExportTask:
    """bench tasks export exposes split-layout compatibility export."""

    def _make_native_task(self, tmp_path: Path) -> Path:
        task = tmp_path / "native-task"
        task.mkdir()
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        (task / "oracle").mkdir()
        (task / "oracle" / "solve.md").write_text("reference\n")
        (task / "verifier").mkdir()
        (task / "verifier" / "test.sh").write_text("#!/bin/bash\necho 1\n")
        (task / "task.md").write_text(
            """---
schema_version: "1.3"
task:
  name: benchflow/export-cli
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
environment:
  network_mode: no-network
benchflow:
  teams:
    - maintainer
---

## prompt

Export this task.
"""
        )
        return task

    def test_tasks_export_cli_writes_split_layout(self, tmp_path: Path) -> None:
        """The task standard export path is available from the CLI."""
        from typer.testing import CliRunner

        task = self._make_native_task(tmp_path)
        out_dir = tmp_path / "exported"

        result = CliRunner().invoke(
            app,
            ["tasks", "export", str(task), str(out_dir), "--target", "harbor"],
        )

        assert result.exit_code == 0, result.output
        assert "Exported:" in result.output
        assert (out_dir / "task.toml").exists()
        assert (out_dir / "instruction.md").read_text() == "Export this task.\n"
        assert (out_dir / "solution" / "solve.md").exists()
        assert (out_dir / "tests" / "test.sh").exists()
        report = json.loads(
            (out_dir / "compatibility" / "export-report.json").read_text()
        )
        assert report["target"] == "harbor"
        assert report["status"] == "degraded"
        assert "benchflow.teams" in {loss["path"] for loss in report["losses"]}

    def test_tasks_export_cli_report_only_outputs_json(self, tmp_path: Path) -> None:
        """Report-only mode lets authors inspect losses before writing export files."""
        from typer.testing import CliRunner

        task = self._make_native_task(tmp_path)

        result = CliRunner().invoke(
            app,
            ["tasks", "export", str(task), "--target", "pier", "--report-only"],
        )

        assert result.exit_code == 0, result.output
        report = json.loads(result.output)
        assert report["target"] == "pier"
        assert report["selected_entrypoint"] == "task.md"
        assert not (tmp_path / "exported").exists()


class TestCtrfPathLint:
    """_check_ctrf_path detects non-standard CTRF output paths (ENG-153).

    Guards PR #356 — ensures `bench tasks check` catches inconsistent
    CTRF paths before they reach production evals.
    """

    def test_standard_path_no_issues(self, tmp_path):
        sh = tmp_path / "test.sh"
        sh.write_text("pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n")
        assert _check_ctrf_path(sh) == []

    def test_standard_path_equals_form(self, tmp_path):
        sh = tmp_path / "test.sh"
        sh.write_text("pytest --ctrf=/logs/verifier/ctrf.json /tests/test_outputs.py\n")
        assert _check_ctrf_path(sh) == []

    def test_nonstandard_filename_flagged(self, tmp_path):
        sh = tmp_path / "test.sh"
        sh.write_text("pytest --ctrf /logs/verifier/ctrf-report.json /tests\n")
        issues = _check_ctrf_path(sh)
        assert len(issues) == 1
        assert "ctrf-report.json" in issues[0]
        assert "non-standard" in issues[0]

    def test_relative_path_flagged(self, tmp_path):
        sh = tmp_path / "test.sh"
        sh.write_text("pytest --ctrf ctrf.json /tests/test_outputs.py\n")
        issues = _check_ctrf_path(sh)
        assert len(issues) == 1
        assert "ctrf.json" in issues[0]

    def test_variable_path_skipped(self, tmp_path):
        sh = tmp_path / "test.sh"
        sh.write_text('pytest --ctrf "$CTRF_DIR/ctrf.json" /tests\n')
        assert _check_ctrf_path(sh) == []

    def test_commented_ctrf_ignored(self, tmp_path):
        sh = tmp_path / "test.sh"
        sh.write_text("# pytest --ctrf /bad/path.json\npytest /tests\n")
        assert _check_ctrf_path(sh) == []

    def test_no_ctrf_flag_no_issues(self, tmp_path):
        sh = tmp_path / "test.sh"
        sh.write_text("pytest /tests/test_outputs.py\n")
        assert _check_ctrf_path(sh) == []

    def test_check_task_integrates_ctrf_lint(self, tmp_path):
        """check_task() calls _check_ctrf_path as part of validation."""
        task = tmp_path / "bad-ctrf"
        task.mkdir()
        (task / "task.toml").write_text(
            "[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 120\n"
        )
        (task / "instruction.md").write_text("# Task\n")
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        tests = task / "tests"
        tests.mkdir()
        (tests / "test.sh").write_text(
            "pytest --ctrf /logs/verifier/ctrf-report.json /tests\n"
        )
        issues = check_task(task)
        assert any("non-standard CTRF path" in i for i in issues)
