"""Parity tests for CLBench -> BenchFlow pipeline.

Validates that generated task directories have correct structure and that
the evaluation pipeline produces valid rewards.

Modes:
  structural  — all required files present, metadata correct
  eval        — evaluate.py produces correct rewards for synthetic inputs
  live        — run real CLBench task with deterministic responses, compare
                original TaskResult.score vs BenchFlow evaluate.py reward
  e2e         — end-to-end: run same agent responses through BOTH original
                CLBench API AND BenchFlow run_task.py pipeline, compare scores
                (Harvey LAB-standard parity)
  all         — structural + eval + live + e2e

Usage::

    python benchmarks/clbench/parity_test.py --output-dir /tmp/clbench-tasks --mode structural
    python benchmarks/clbench/parity_test.py --output-dir /tmp/clbench-tasks --mode eval
    python benchmarks/clbench/parity_test.py --output-dir /tmp/clbench-tasks --mode live --clbench-dir /path/to/continual-learning-bench
    python benchmarks/clbench/parity_test.py --output-dir /tmp/clbench-tasks --mode e2e --clbench-dir /path/to/continual-learning-bench
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EXPECTED_TASKS = [
    "clbench-exploitable-poker",
    "clbench-database-exploration",
    "clbench-cohort-studies",
]

# r_max values per task — must match benchflow.py _CLBENCH_TASKS
_R_MAX = {
    "clbench-exploitable-poker": 9.4875,
    "clbench-database-exploration": 1.0,
    "clbench-cohort-studies": 0.162202,
}


def _task_names_to_check(output_dir: Path) -> list[str]:
    """Return known CLBench task dirs present in this generated output."""
    return [
        task_name for task_name in EXPECTED_TASKS if (output_dir / task_name).is_dir()
    ]


def _check_structural(task_dir: Path) -> list[str]:
    """Check structural parity for a single task directory. Returns list of errors."""
    errors: list[str] = []
    name = task_dir.name

    # task.toml
    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        errors.append(f"{name}: missing task.toml")
    else:
        content = task_toml.read_text()
        if 'name = "clbench/' not in content:
            errors.append(f"{name}: task.toml missing clbench/ prefix in name")
        if "[task]" not in content:
            errors.append(f"{name}: task.toml missing [task] section")
        if "[metadata]" not in content:
            errors.append(f"{name}: task.toml missing [metadata] section")
        if "[agent]" not in content:
            errors.append(f"{name}: task.toml missing [agent] section")
        if "[verifier]" not in content:
            errors.append(f"{name}: task.toml missing [verifier] section")
        if "[environment]" not in content:
            errors.append(f"{name}: task.toml missing [environment] section")

    # instruction.md
    instruction = task_dir / "instruction.md"
    if not instruction.exists():
        errors.append(f"{name}: missing instruction.md")
    elif instruction.stat().st_size == 0:
        errors.append(f"{name}: instruction.md is empty")

    # environment/Dockerfile
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        errors.append(f"{name}: missing environment/Dockerfile")
    else:
        content = dockerfile.read_text()
        if "FROM python:3.13-slim" not in content:
            errors.append(f"{name}: Dockerfile missing FROM python:3.13-slim")
        if "/logs/verifier" not in content:
            errors.append(
                f"{name}: Dockerfile missing /logs/verifier directory creation"
            )

    # environment/run_task.py
    run_task = task_dir / "environment" / "run_task.py"
    if not run_task.exists():
        errors.append(f"{name}: missing environment/run_task.py")

    # environment/schedule.json
    schedule = task_dir / "environment" / "schedule.json"
    if not schedule.exists():
        errors.append(f"{name}: missing environment/schedule.json")

    # tests/test.sh
    test_sh = task_dir / "tests" / "test.sh"
    if not test_sh.exists():
        errors.append(f"{name}: missing tests/test.sh")
    else:
        mode = test_sh.stat().st_mode
        if not (mode & stat.S_IXUSR):
            errors.append(f"{name}: tests/test.sh is not executable")

    # tests/evaluate.py
    evaluate = task_dir / "tests" / "evaluate.py"
    if not evaluate.exists():
        errors.append(f"{name}: missing tests/evaluate.py")

    return errors


def run_structural_parity(output_dir: Path) -> dict:
    """Run structural parity checks on all generated task directories."""
    results = {"tasks_tested": 0, "passed": 0, "errors": []}
    task_names = _task_names_to_check(output_dir)
    if not task_names:
        results["errors"].append(
            f"No generated CLBench task directories in {output_dir}"
        )
        return results

    for task_name in task_names:
        task_dir = output_dir / task_name
        errors = _check_structural(task_dir)
        results["tasks_tested"] += 1
        if not errors:
            results["passed"] += 1
            log.info("PASS: %s", task_name)
        else:
            for err in errors:
                log.error("FAIL: %s", err)
            results["errors"].extend(errors)

    return results


def _run_evaluate_py(evaluate_py: Path, results_file: Path, reward_file: Path) -> float:
    """Run the actual generated evaluate.py with patched file paths."""
    content = evaluate_py.read_text()
    content = content.replace(
        'RESULTS_FILE = "/opt/results.json"',
        f'RESULTS_FILE = "{results_file}"',
    )
    content = content.replace(
        'REWARD_FILE = "/logs/verifier/reward.txt"',
        f'REWARD_FILE = "{reward_file}"',
    )
    result = subprocess.run(
        [sys.executable, "-c", content],
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        log.error("evaluate.py failed: %s", result.stderr)
        return -1.0

    try:
        return float(reward_file.read_text().strip())
    except (ValueError, FileNotFoundError):
        return -1.0


def run_eval_parity(output_dir: Path) -> dict:
    """Run eval parity: test evaluate.py with synthetic results."""
    results = {"tasks_tested": 0, "passed": 0, "tests": []}

    for task_name in _task_names_to_check(output_dir):
        task_dir = output_dir / task_name
        if not task_dir.exists():
            continue

        evaluate_py = task_dir / "tests" / "evaluate.py"
        if not evaluate_py.exists():
            continue

        r_max = _R_MAX[task_name]
        results["tasks_tested"] += 1
        task_passed = True

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Test 1: Valid results -> normalized reward (score / r_max)
            results_file = tmp / "results_valid.json"
            reward_file = tmp / "reward_valid.txt"
            test_score = 0.75
            results_file.write_text(json.dumps({"score": test_score}))
            reward = _run_evaluate_py(evaluate_py, results_file, reward_file)
            expected = max(0.0, min(1.0, test_score / r_max))
            test_result = {
                "name": f"{task_name}/valid_results",
                "expected_reward": str(round(expected, 6)),
                "actual_reward": str(reward),
                "result": "pass" if abs(reward - expected) < 0.001 else "fail",
            }
            results["tests"].append(test_result)
            if test_result["result"] == "fail":
                task_passed = False
                log.error(
                    "FAIL: %s valid_results: expected %s, got %s",
                    task_name,
                    expected,
                    reward,
                )

            # Test 2: Missing results -> 0.0
            results_file = tmp / "results_missing.json"
            reward_file = tmp / "reward_missing.txt"
            # Don't create results_file — it should be missing
            reward = _run_evaluate_py(evaluate_py, results_file, reward_file)
            test_result = {
                "name": f"{task_name}/missing_results",
                "expected_reward": "0.0",
                "actual_reward": str(reward),
                "result": "pass" if abs(reward) < 0.001 else "fail",
            }
            results["tests"].append(test_result)
            if test_result["result"] == "fail":
                task_passed = False
                log.error(
                    "FAIL: %s missing_results: expected 0.0, got %s", task_name, reward
                )

            # Test 3: Malformed results -> 0.0
            results_file = tmp / "results_malformed.json"
            reward_file = tmp / "reward_malformed.txt"
            results_file.write_text("not json at all {{{")
            reward = _run_evaluate_py(evaluate_py, results_file, reward_file)
            test_result = {
                "name": f"{task_name}/malformed_results",
                "expected_reward": "0.0",
                "actual_reward": str(reward),
                "result": "pass" if abs(reward) < 0.001 else "fail",
            }
            results["tests"].append(test_result)
            if test_result["result"] == "fail":
                task_passed = False
                log.error(
                    "FAIL: %s malformed_results: expected 0.0, got %s",
                    task_name,
                    reward,
                )

            # Test 4: Score above r_max -> clamped to 1.0
            results_file = tmp / "results_clamped.json"
            reward_file = tmp / "reward_clamped.txt"
            results_file.write_text(json.dumps({"score": r_max * 2.0}))
            reward = _run_evaluate_py(evaluate_py, results_file, reward_file)
            test_result = {
                "name": f"{task_name}/clamped_results",
                "expected_reward": "1.0",
                "actual_reward": str(reward),
                "result": "pass" if abs(reward - 1.0) < 0.001 else "fail",
            }
            results["tests"].append(test_result)
            if test_result["result"] == "fail":
                task_passed = False
                log.error(
                    "FAIL: %s clamped_results: expected 1.0, got %s", task_name, reward
                )

        if task_passed:
            results["passed"] += 1
            log.info("PASS: %s eval parity", task_name)

    return results


# ── Live parity ──────────────────────────────────────────────────────
# Run the real CLBench task with deterministic responses, capture the
# original TaskResult.score, then feed that score through BenchFlow's
# generated evaluate.py and verify the normalized reward matches.

_LIVE_POKER_SCRIPT = """
import json, sys
sys.path.insert(0, "{clbench_dir}")
from src.tasks.exploitable_poker.task import Poker
from src.interface import Response
from pydantic import BaseModel, Field
from typing import Optional

class PokerAction(BaseModel):
    thinking: str = Field(default="deterministic fold")
    action: str = Field(default="FOLD")
    amount: Optional[int] = None

task = Poker(num_instances={n}, seed=42)
query = task.reset()
for _ in range(200):
    resp = Response(action=PokerAction())
    step = task.step(resp)
    if step.done:
        break
    if step.next_query:
        query = step.next_query

result = task.evaluate()
output = {{
    "score": result.score,
    "r_max": task.r_max,
    "num_outcomes": len(result.instance_outcomes),
    "outcomes": [
        {{
            "instance_id": o.instance_id,
            "instance_index": o.instance_index,
            "reward": o.reward,
        }}
        for o in result.instance_outcomes
    ],
}}
print(json.dumps(output))
"""

_LIVE_DATABASE_SCRIPT = """
import json, sys
sys.path.insert(0, "{clbench_dir}")
from src.tasks.database_exploration.task import DatabaseExploration, DatabaseAction
from src.interface import Response

task = DatabaseExploration(num_instances={n}, seed=42)
query = task.reset()
for _ in range(500):
    resp = Response(action=DatabaseAction(action="ANSWER", content="unknown"))
    step = task.step(resp)
    if step.done:
        break
    if step.next_query:
        query = step.next_query

result = task.evaluate()
output = {{
    "score": result.score,
    "r_max": task.r_max,
    "num_outcomes": len(result.instance_outcomes),
    "outcomes": [
        {{
            "instance_id": o.instance_id,
            "instance_index": o.instance_index,
            "reward": o.reward,
        }}
        for o in result.instance_outcomes
    ],
}}
print(json.dumps(output))
"""

_LIVE_SCRIPTS = {
    "exploitable_poker": _LIVE_POKER_SCRIPT,
    "database_exploration": _LIVE_DATABASE_SCRIPT,
}


def _run_clbench_live(
    clbench_dir: Path,
    python_bin: str,
    task_name: str,
    num_instances: int,
) -> dict | None:
    """Run a CLBench task with deterministic responses, return score info."""
    template = _LIVE_SCRIPTS.get(task_name)
    if template is None:
        log.info("Skipping live parity for %s (no live script)", task_name)
        return None

    script = template.format(
        clbench_dir=clbench_dir,
        n=num_instances,
    )
    result = subprocess.run(
        [python_bin, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(clbench_dir),
    )
    if result.returncode != 0:
        log.error("CLBench live run failed for %s: %s", task_name, result.stderr[-500:])
        return None
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        log.error("Failed to parse CLBench output: %s", result.stdout[:200])
        return None


def run_live_parity(
    output_dir: Path,
    clbench_dir: Path,
    python_bin: str | None = None,
) -> dict:
    """Run live parity: real CLBench task vs BenchFlow evaluate.py.

    For each supported task:
    1. Run CLBench task with deterministic responses -> original score
    2. Write results.json with that score
    3. Run BenchFlow evaluate.py -> BenchFlow reward
    4. Verify: reward == max(0, min(1, original_score / r_max))
    """
    if python_bin is None:
        venv_python = clbench_dir / ".venv" / "bin" / "python"
        python_bin = str(venv_python) if venv_python.exists() else sys.executable

    results: dict = {"tasks_tested": 0, "passed": 0, "tests": []}

    # Map BenchFlow task names to CLBench task names
    bf_to_cl = {
        "clbench-exploitable-poker": "exploitable_poker",
        "clbench-database-exploration": "database_exploration",
    }

    for bf_name, cl_name in bf_to_cl.items():
        task_dir = output_dir / bf_name
        evaluate_py = task_dir / "tests" / "evaluate.py"
        if not task_dir.exists() or not evaluate_py.exists():
            log.warning("Skipping %s: task dir or evaluate.py missing", bf_name)
            continue

        r_max = _R_MAX[bf_name]
        results["tasks_tested"] += 1

        # Step 1: Run CLBench task with deterministic responses
        log.info("Running CLBench %s with deterministic responses...", cl_name)
        live_result = _run_clbench_live(
            clbench_dir, python_bin, cl_name, num_instances=5
        )
        if live_result is None:
            results["tests"].append(
                {
                    "name": f"{bf_name}/live_run",
                    "result": "fail",
                    "reason": "CLBench task failed to run",
                }
            )
            continue

        original_score = live_result["score"]
        clbench_r_max = live_result["r_max"]
        log.info(
            "CLBench %s: score=%.6f, r_max=%.4f, outcomes=%d",
            cl_name,
            original_score,
            clbench_r_max,
            live_result["num_outcomes"],
        )

        # Verify r_max matches what we have in the adapter
        if abs(clbench_r_max - r_max) > 0.0001:
            results["tests"].append(
                {
                    "name": f"{bf_name}/r_max_match",
                    "result": "fail",
                    "expected": str(r_max),
                    "actual": str(clbench_r_max),
                    "reason": "r_max mismatch between adapter and CLBench",
                }
            )
            continue

        results["tests"].append(
            {
                "name": f"{bf_name}/r_max_match",
                "result": "pass",
                "adapter_r_max": str(r_max),
                "clbench_r_max": str(clbench_r_max),
            }
        )

        # Step 2: Write results.json with original score
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_file = tmp / "results.json"
            reward_file = tmp / "reward.txt"
            results_file.write_text(
                json.dumps(
                    {
                        "score": original_score,
                        "summary": "live parity test",
                        "metrics": {},
                        "instance_outcomes": live_result["outcomes"],
                    }
                )
            )

            # Step 3: Run BenchFlow evaluate.py
            benchflow_reward = _run_evaluate_py(evaluate_py, results_file, reward_file)

            # Step 4: Verify normalization
            expected_reward = max(0.0, min(1.0, original_score / r_max))
            delta = abs(benchflow_reward - expected_reward)

            test_result = {
                "name": f"{bf_name}/live_parity",
                "original_score": str(round(original_score, 6)),
                "r_max": str(r_max),
                "expected_reward": str(round(expected_reward, 6)),
                "actual_reward": str(benchflow_reward),
                "delta": str(round(delta, 6)),
                "result": "pass" if delta < 0.001 else "fail",
            }
            results["tests"].append(test_result)

            if test_result["result"] == "pass":
                results["passed"] += 1
                log.info(
                    "PASS: %s live parity — CLBench score=%.6f -> "
                    "BenchFlow reward=%.6f (expected=%.6f)",
                    bf_name,
                    original_score,
                    benchflow_reward,
                    expected_reward,
                )
            else:
                log.error(
                    "FAIL: %s live parity — CLBench score=%.6f -> "
                    "BenchFlow reward=%.6f (expected=%.6f, delta=%.6f)",
                    bf_name,
                    original_score,
                    benchflow_reward,
                    expected_reward,
                    delta,
                )

    return results


# ── End-to-end parity ────────────────────────────────────────────────
# Harvey LAB-standard: run the SAME agent responses through BOTH the
# original CLBench Python API AND BenchFlow's generated run_task.py,
# then compare resulting scores.  Finally run evaluate.py and verify
# the normalized reward matches.
#
# Agent strategy per task:
#   poker      — check-or-call (produces non-trivial positive scores)
#   database   — always ANSWER 'unknown' (score=0, still validates pipeline)

_E2E_POKER_AGENT = """
import json, sys
sys.path.insert(0, "{clbench_dir}")
from src.tasks.exploitable_poker.task import Poker
from src.interface import Response
from pydantic import BaseModel, Field
from typing import Optional

class PokerAction(BaseModel):
    thinking: str = Field(default="check_or_call")
    action: str = Field(default="CHECK")
    amount: Optional[int] = None

task = Poker(num_instances={n}, seed=42)
query = task.reset()
responses_log = []

for _ in range(2000):
    if not task.waiting_for_action:
        break
    # Try CHECK first; if invalid, try CALL
    for action_str in ["CHECK", "CALL"]:
        action = PokerAction(thinking="check_or_call", action=action_str)
        resp = Response(action=action)
        step = task.step(resp)
        if "Invalid" not in step.observation.content:
            responses_log.append(action.model_dump())
            break
    if step.done:
        break
    if step.next_query:
        query = step.next_query

result = task.evaluate()
output = {{
    "score": result.score,
    "r_max": task.r_max,
    "hands_played": task.hands_played,
    "total_profit": task.system_profit,
    "num_outcomes": len(result.instance_outcomes),
    "outcomes": [
        {{
            "instance_id": o.instance_id,
            "instance_index": o.instance_index,
            "reward": o.reward,
            "success": o.success,
        }}
        for o in result.instance_outcomes
    ],
    "responses": responses_log,
}}
print(json.dumps(output))
"""

_E2E_DATABASE_AGENT = """
import json, sys
sys.path.insert(0, "{clbench_dir}")
from src.tasks.database_exploration.task import DatabaseExploration, DatabaseAction
from src.interface import Response

task = DatabaseExploration(num_instances={n}, seed=42)
query = task.reset()
responses_log = []

for _ in range(500):
    action = DatabaseAction(action="ANSWER", content="unknown")
    resp = Response(action=action)
    responses_log.append(action.model_dump())
    step = task.step(resp)
    if step.done:
        break
    if step.next_query:
        query = step.next_query

result = task.evaluate()
output = {{
    "score": result.score,
    "r_max": task.r_max,
    "num_outcomes": len(result.instance_outcomes),
    "outcomes": [
        {{
            "instance_id": o.instance_id,
            "instance_index": o.instance_index,
            "reward": o.reward,
            "success": o.success,
        }}
        for o in result.instance_outcomes
    ],
    "responses": responses_log,
}}
print(json.dumps(output))
"""

_E2E_AGENTS = {
    "exploitable_poker": _E2E_POKER_AGENT,
    "database_exploration": _E2E_DATABASE_AGENT,
}

# Replay script: feeds pre-recorded responses through CLBench task API
# (equivalent to what run_task.py does inside Docker, but run locally).
_E2E_PIPELINE_SCRIPT = """
import json, sys
from pathlib import Path
sys.path.insert(0, "{clbench_dir}")
from src.registry import get_task_class
from src.interface import Response

responses_file = Path("{responses_file}")
results_file = Path("{results_file}")

with open(responses_file) as f:
    responses = [json.loads(line) for line in f if line.strip()]

task_cls = get_task_class("{task_name}")
task = task_cls(num_instances={n}, seed=42)
query = task.reset()
outcomes = []

for resp_data in responses:
    schema_cls = query.response_schema
    action = schema_cls.model_validate(resp_data)
    response = Response(action=action)
    step_result = task.step(response)
    if step_result.instance_outcome:
        outcomes.append({{
            "instance_id": step_result.instance_outcome.instance_id,
            "instance_index": step_result.instance_outcome.instance_index,
            "reward": step_result.instance_outcome.reward,
            "success": step_result.instance_outcome.success,
        }})
    if step_result.done:
        break
    if step_result.next_query:
        query = step_result.next_query

result = task.evaluate()
output = {{
    "score": result.score,
    "summary": result.summary,
    "metrics": {{}},
    "instance_outcomes": outcomes,
}}
results_file.write_text(json.dumps(output, indent=2, default=str))
print(json.dumps({{"score": result.score, "num_outcomes": len(outcomes)}}))
"""


def _run_e2e_agent(
    clbench_dir: Path,
    python_bin: str,
    task_name: str,
    num_instances: int,
) -> dict | None:
    """Run deterministic agent on original CLBench, return score + responses."""
    template = _E2E_AGENTS.get(task_name)
    if template is None:
        log.info("Skipping e2e for %s (no agent script)", task_name)
        return None

    script = template.format(clbench_dir=clbench_dir, n=num_instances)
    result = subprocess.run(
        [python_bin, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(clbench_dir),
    )
    if result.returncode != 0:
        log.error("E2E agent failed for %s: %s", task_name, result.stderr[-500:])
        return None
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        log.error("Failed to parse agent output: %s", result.stdout[:200])
        return None


def _run_e2e_pipeline(
    clbench_dir: Path,
    python_bin: str,
    task_name: str,
    num_instances: int,
    responses_file: Path,
    results_file: Path,
) -> dict | None:
    """Replay responses through BenchFlow pipeline (run_task.py equivalent)."""
    script = _E2E_PIPELINE_SCRIPT.format(
        clbench_dir=clbench_dir,
        task_name=task_name,
        n=num_instances,
        responses_file=responses_file,
        results_file=results_file,
    )
    result = subprocess.run(
        [python_bin, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(clbench_dir),
    )
    if result.returncode != 0:
        log.error("E2E pipeline failed for %s: %s", task_name, result.stderr[-500:])
        return None
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        log.error("Failed to parse pipeline output: %s", result.stdout[:200])
        return None


def run_e2e_parity(
    output_dir: Path,
    clbench_dir: Path,
    python_bin: str | None = None,
) -> dict:
    """End-to-end parity: same agent responses through original vs BenchFlow.

    Harvey LAB-standard validation:
    1. Run deterministic agent on original CLBench API -> (score, responses)
    2. Replay SAME responses through BenchFlow run_task.py -> pipeline_score
    3. Verify: original_score == pipeline_score (pipeline fidelity)
    4. Run evaluate.py -> reward
    5. Verify: reward == max(0, min(1, score / r_max)) (normalization)
    """
    if python_bin is None:
        venv_python = clbench_dir / ".venv" / "bin" / "python"
        python_bin = str(venv_python) if venv_python.exists() else sys.executable

    results: dict = {"tasks_tested": 0, "passed": 0, "tests": []}

    bf_to_cl = {
        "clbench-exploitable-poker": ("exploitable_poker", 20),
        "clbench-database-exploration": ("database_exploration", 5),
    }

    for bf_name, (cl_name, n_instances) in bf_to_cl.items():
        task_dir = output_dir / bf_name
        evaluate_py = task_dir / "tests" / "evaluate.py"
        if not task_dir.exists() or not evaluate_py.exists():
            log.warning("Skipping %s: task dir or evaluate.py missing", bf_name)
            continue

        r_max = _R_MAX[bf_name]
        results["tasks_tested"] += 1
        task_passed = True

        # Step 1: Run agent on original CLBench
        log.info(
            "E2E %s: running agent on original CLBench (%d instances)...",
            cl_name,
            n_instances,
        )
        agent_result = _run_e2e_agent(clbench_dir, python_bin, cl_name, n_instances)
        if agent_result is None:
            results["tests"].append(
                {
                    "name": f"{bf_name}/e2e_agent_run",
                    "result": "fail",
                    "reason": "Agent failed to run on original CLBench",
                }
            )
            continue

        original_score = agent_result["score"]
        responses_log = agent_result["responses"]
        log.info(
            "E2E %s: original score=%.6f, %d responses captured",
            cl_name,
            original_score,
            len(responses_log),
        )

        # r_max check
        if abs(agent_result["r_max"] - r_max) > 0.0001:
            results["tests"].append(
                {
                    "name": f"{bf_name}/e2e_r_max",
                    "result": "fail",
                    "expected": str(r_max),
                    "actual": str(agent_result["r_max"]),
                }
            )
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Write responses to JSONL
            responses_file = tmp / "agent_responses.jsonl"
            with open(responses_file, "w") as f:
                for resp in responses_log:
                    f.write(json.dumps(resp) + "\n")

            results_file = tmp / "results.json"

            # Step 2: Replay through BenchFlow pipeline
            log.info(
                "E2E %s: replaying %d responses through BenchFlow pipeline...",
                cl_name,
                len(responses_log),
            )
            pipeline_result = _run_e2e_pipeline(
                clbench_dir,
                python_bin,
                cl_name,
                n_instances,
                responses_file,
                results_file,
            )
            if pipeline_result is None:
                results["tests"].append(
                    {
                        "name": f"{bf_name}/e2e_pipeline_run",
                        "result": "fail",
                        "reason": "Pipeline failed to replay responses",
                    }
                )
                continue

            pipeline_score = pipeline_result["score"]

            # Step 3: Compare scores — pipeline fidelity
            score_delta = abs(original_score - pipeline_score)
            fidelity_pass = score_delta < 0.0001
            fidelity_result = {
                "name": f"{bf_name}/e2e_score_fidelity",
                "original_score": str(round(original_score, 6)),
                "pipeline_score": str(round(pipeline_score, 6)),
                "delta": str(round(score_delta, 6)),
                "result": "pass" if fidelity_pass else "fail",
            }
            results["tests"].append(fidelity_result)
            if fidelity_pass:
                log.info(
                    "PASS: %s score fidelity — original=%.6f, pipeline=%.6f",
                    bf_name,
                    original_score,
                    pipeline_score,
                )
            else:
                task_passed = False
                log.error(
                    "FAIL: %s score fidelity — original=%.6f, pipeline=%.6f "
                    "(delta=%.6f)",
                    bf_name,
                    original_score,
                    pipeline_score,
                    score_delta,
                )

            # Step 4: Run evaluate.py on pipeline results
            reward_file = tmp / "reward.txt"
            benchflow_reward = _run_evaluate_py(evaluate_py, results_file, reward_file)
            expected_reward = max(0.0, min(1.0, pipeline_score / r_max))
            reward_delta = abs(benchflow_reward - expected_reward)
            norm_pass = reward_delta < 0.001

            norm_result = {
                "name": f"{bf_name}/e2e_normalization",
                "pipeline_score": str(round(pipeline_score, 6)),
                "r_max": str(r_max),
                "expected_reward": str(round(expected_reward, 6)),
                "actual_reward": str(benchflow_reward),
                "delta": str(round(reward_delta, 6)),
                "result": "pass" if norm_pass else "fail",
            }
            results["tests"].append(norm_result)
            if norm_pass:
                log.info(
                    "PASS: %s normalization — score=%.6f / r_max=%.4f -> "
                    "reward=%.6f (expected=%.6f)",
                    bf_name,
                    pipeline_score,
                    r_max,
                    benchflow_reward,
                    expected_reward,
                )
            else:
                task_passed = False
                log.error(
                    "FAIL: %s normalization — expected=%.6f, got=%.6f",
                    bf_name,
                    expected_reward,
                    benchflow_reward,
                )

        if task_passed:
            results["passed"] += 1

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="CLBench parity tests")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing generated task directories",
    )
    parser.add_argument(
        "--mode",
        choices=["structural", "eval", "live", "e2e", "all"],
        default="all",
        help="Which parity checks to run",
    )
    parser.add_argument(
        "--clbench-dir",
        type=Path,
        default=None,
        help="Path to CLBench repo (required for --mode live, e2e, or all)",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=None,
        help="Python binary for CLBench (default: <clbench-dir>/.venv/bin/python)",
    )
    args = parser.parse_args()

    all_passed = True

    if args.mode in ("structural", "all"):
        print("\n=== Structural Parity ===")
        structural = run_structural_parity(args.output_dir)
        print(f"  Tested: {structural['tasks_tested']}, Passed: {structural['passed']}")
        if structural["errors"]:
            all_passed = False
            for err in structural["errors"]:
                print(f"  ERROR: {err}")

    if args.mode in ("eval", "all"):
        print("\n=== Eval Parity ===")
        eval_results = run_eval_parity(args.output_dir)
        print(
            f"  Tested: {eval_results['tasks_tested']}, Passed: {eval_results['passed']}"
        )
        for test in eval_results["tests"]:
            status = "PASS" if test["result"] == "pass" else "FAIL"
            print(
                f"  {status}: {test['name']} (expected={test['expected_reward']}, actual={test['actual_reward']})"
            )
        if eval_results["passed"] < eval_results["tasks_tested"]:
            all_passed = False

    if args.mode in ("live", "all"):
        if args.clbench_dir is None:
            print("\n=== Live Parity ===")
            print("  SKIPPED: --clbench-dir not provided")
        else:
            print("\n=== Live Parity ===")
            live_results = run_live_parity(
                args.output_dir, args.clbench_dir, args.python_bin
            )
            print(
                f"  Tested: {live_results['tasks_tested']}, "
                f"Passed: {live_results['passed']}"
            )
            for test in live_results["tests"]:
                status = "PASS" if test["result"] == "pass" else "FAIL"
                if "original_score" in test:
                    print(
                        f"  {status}: {test['name']} "
                        f"(CLBench score={test['original_score']}, "
                        f"BenchFlow reward={test['actual_reward']}, "
                        f"expected={test['expected_reward']})"
                    )
                else:
                    print(f"  {status}: {test['name']}")
            if live_results["passed"] < live_results["tasks_tested"]:
                all_passed = False

    if args.mode in ("e2e", "all"):
        if args.clbench_dir is None:
            print("\n=== End-to-End Parity ===")
            print("  SKIPPED: --clbench-dir not provided")
        else:
            print("\n=== End-to-End Parity (Harvey LAB-standard) ===")
            e2e_results = run_e2e_parity(
                args.output_dir, args.clbench_dir, args.python_bin
            )
            print(
                f"  Tested: {e2e_results['tasks_tested']}, "
                f"Passed: {e2e_results['passed']}"
            )
            for test in e2e_results["tests"]:
                status = "PASS" if test["result"] == "pass" else "FAIL"
                if "original_score" in test:
                    print(
                        f"  {status}: {test['name']} "
                        f"(original={test['original_score']}, "
                        f"pipeline={test['pipeline_score']})"
                    )
                elif "pipeline_score" in test:
                    print(
                        f"  {status}: {test['name']} "
                        f"(reward={test['actual_reward']}, "
                        f"expected={test['expected_reward']})"
                    )
                else:
                    print(f"  {status}: {test['name']}")
            if e2e_results["passed"] < e2e_results["tasks_tested"]:
                all_passed = False

    if all_passed:
        print("\nAll parity checks passed!")
    else:
        print("\nSome parity checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
