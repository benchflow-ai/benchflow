"""Structural parity test for HILBench → BenchFlow pipeline.

Validates that generated task directories have the correct structure
and required files.

Usage::

    python benchmarks/hilbench/parity_test.py --tasks-dir /tmp/hilbench-tasks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REQUIRED_FILES = [
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "tests/test.sh",
    "tests/verify.py",
    "tests/test_patch.diff",
    "tests/task_metadata.json",
]


def _check_task(task_dir: Path) -> list[str]:
    """Check a single task directory for structural issues. Returns list of errors."""
    errors: list[str] = []
    task_id = task_dir.name

    # Check required files exist
    for rel_path in REQUIRED_FILES:
        fpath = task_dir / rel_path
        if not fpath.exists():
            errors.append(f"{task_id}: missing {rel_path}")

    # Check task.toml has required fields
    task_toml = task_dir / "task.toml"
    if task_toml.exists():
        content = task_toml.read_text()
        for field in ["[task]", "name =", "[metadata]", "[agent]", "[verifier]"]:
            if field not in content:
                errors.append(f"{task_id}: task.toml missing field '{field}'")
        if "hilbench/" not in content:
            errors.append(f"{task_id}: task.toml name does not start with 'hilbench/'")

    # Check instruction.md is non-empty
    instruction = task_dir / "instruction.md"
    if instruction.exists() and instruction.stat().st_size == 0:
        errors.append(f"{task_id}: instruction.md is empty")

    # Check test_patch.diff is non-empty
    patch = task_dir / "tests" / "test_patch.diff"
    if patch.exists() and patch.stat().st_size == 0:
        errors.append(f"{task_id}: test_patch.diff is empty")

    # Check task_metadata.json is valid JSON with required fields
    meta = task_dir / "tests" / "task_metadata.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text())
            for key in ["task_id", "repo_name", "tests_to_pass"]:
                if key not in data:
                    errors.append(f"{task_id}: task_metadata.json missing '{key}'")
            if "tests_to_pass" in data and not data["tests_to_pass"]:
                errors.append(f"{task_id}: tests_to_pass is empty")
        except json.JSONDecodeError as exc:
            errors.append(f"{task_id}: task_metadata.json is invalid JSON: {exc}")

    # Check test.sh is executable
    test_sh = task_dir / "tests" / "test.sh"
    if test_sh.exists():
        import stat

        mode = test_sh.stat().st_mode
        if not (mode & stat.S_IEXEC):
            errors.append(f"{task_id}: test.sh is not executable")

    # Check Dockerfile references ubuntu base
    dockerfile = task_dir / "environment" / "Dockerfile"
    if dockerfile.exists():
        content = dockerfile.read_text()
        if "FROM" not in content:
            errors.append(f"{task_id}: Dockerfile missing FROM directive")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="HILBench structural parity test")
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Specific task directories to check (default: all)",
    )
    args = parser.parse_args()

    tasks_dir: Path = args.tasks_dir
    if not tasks_dir.exists():
        log.error("Tasks directory not found: %s", tasks_dir)
        sys.exit(1)

    # Discover task directories
    if args.task_ids:
        task_dirs = [tasks_dir / tid for tid in args.task_ids]
    else:
        task_dirs = sorted(
            d for d in tasks_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        )

    if not task_dirs:
        log.error("No task directories found in %s", tasks_dir)
        sys.exit(1)

    log.info("Checking %d task directories in %s", len(task_dirs), tasks_dir)

    all_errors: list[str] = []
    passed = 0
    failed = 0

    for task_dir in task_dirs:
        if not task_dir.exists():
            log.warning("Task directory not found: %s", task_dir)
            all_errors.append(f"{task_dir.name}: directory not found")
            failed += 1
            continue

        errors = _check_task(task_dir)
        if errors:
            for err in errors:
                log.warning("  FAIL: %s", err)
            all_errors.extend(errors)
            failed += 1
        else:
            passed += 1

    print("\n=== Structural Parity Results ===")
    print(f"  Passed: {passed}/{len(task_dirs)}")
    print(f"  Failed: {failed}/{len(task_dirs)}")
    if all_errors:
        print(f"  Errors ({len(all_errors)}):")
        for err in all_errors:
            print(f"    - {err}")
        sys.exit(1)
    else:
        print("  All tasks passed structural validation.")


if __name__ == "__main__":
    main()
