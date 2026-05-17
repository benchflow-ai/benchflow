"""Parity tests for CLBench -> BenchFlow pipeline.

Validates that generated task directories have correct structure and that
the evaluation pipeline produces valid rewards.

Usage::

    python benchmarks/clbench/parity_test.py --output-dir /tmp/clbench-tasks --mode structural
    python benchmarks/clbench/parity_test.py --output-dir /tmp/clbench-tasks --mode eval
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

    for task_name in EXPECTED_TASKS:
        task_dir = output_dir / task_name
        if not task_dir.exists():
            results["errors"].append(f"{task_name}: directory not found")
            results["tasks_tested"] += 1
            continue

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

    for task_name in EXPECTED_TASKS:
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
        choices=["structural", "eval", "all"],
        default="all",
        help="Which parity checks to run",
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

    if all_passed:
        print("\nAll parity checks passed!")
    else:
        print("\nSome parity checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
