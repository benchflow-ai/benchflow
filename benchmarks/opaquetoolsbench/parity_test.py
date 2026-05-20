"""Structural parity test for OpaqueToolsBench BFCL → BenchFlow pipeline.

Validates that generated task directories have the correct structure,
required files, and valid content.

Usage::

    python benchmarks/opaquetoolsbench/parity_test.py \\
        --tasks-dir /tmp/opaquetoolsbench-tasks

    python benchmarks/opaquetoolsbench/parity_test.py \\
        --tasks-dir /tmp/opaquetoolsbench-tasks \\
        --task-ids executable-simple-0,executable-simple-1
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
    "tests/evaluate.py",
    "tests/ground_truth.json",
    "solution/solve.sh",
]


def _validate_task(task_dir: Path) -> list[str]:
    """Validate a single task directory. Returns list of error messages."""
    errors: list[str] = []
    task_id = task_dir.name

    # Check required files
    for rel in REQUIRED_FILES:
        fpath = task_dir / rel
        if not fpath.exists():
            errors.append(f"[{task_id}] Missing file: {rel}")
        elif fpath.stat().st_size == 0:
            errors.append(f"[{task_id}] Empty file: {rel}")

    # Validate task.toml
    toml_path = task_dir / "task.toml"
    if toml_path.exists():
        content = toml_path.read_text()
        if 'name = "opaquetoolsbench/' not in content:
            errors.append(
                f"[{task_id}] task.toml missing opaquetoolsbench/ prefix in name"
            )
        if "[agent]" not in content:
            errors.append(f"[{task_id}] task.toml missing [agent] section")
        if "[verifier]" not in content:
            errors.append(f"[{task_id}] task.toml missing [verifier] section")
        if "timeout_sec" not in content:
            errors.append(f"[{task_id}] task.toml missing timeout_sec")

    # Validate instruction.md
    instr_path = task_dir / "instruction.md"
    if instr_path.exists():
        content = instr_path.read_text()
        if "## Query" not in content:
            errors.append(f"[{task_id}] instruction.md missing ## Query section")
        if "## Available Functions" not in content:
            errors.append(f"[{task_id}] instruction.md missing ## Available Functions")
        if "/app/output/response.json" not in content:
            errors.append(f"[{task_id}] instruction.md missing output path reference")

    # Validate ground_truth.json
    gt_path = task_dir / "tests" / "ground_truth.json"
    if gt_path.exists():
        try:
            gt = json.loads(gt_path.read_text())
            if "ground_truth" not in gt:
                errors.append(
                    f"[{task_id}] ground_truth.json missing 'ground_truth' key"
                )
            elif not gt["ground_truth"]:
                errors.append(
                    f"[{task_id}] ground_truth.json has empty ground_truth list"
                )
        except json.JSONDecodeError as e:
            errors.append(f"[{task_id}] ground_truth.json invalid JSON: {e}")

    # Validate test.sh is executable
    for rel in ("tests/test.sh", "solution/solve.sh"):
        script = task_dir / rel
        if not script.exists():
            continue
        import stat

        mode = script.stat().st_mode
        if not (mode & stat.S_IXUSR):
            errors.append(f"[{task_id}] {rel} is not executable")

    # Validate Dockerfile
    dockerfile = task_dir / "environment" / "Dockerfile"
    if dockerfile.exists():
        content = dockerfile.read_text()
        if "FROM" not in content:
            errors.append(f"[{task_id}] Dockerfile missing FROM directive")
        if "/logs/verifier" not in content:
            errors.append(f"[{task_id}] Dockerfile missing /logs/verifier directory")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpaqueToolsBench structural parity test"
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="Path to generated task directories",
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Specific task IDs to validate (default: all)",
    )
    args = parser.parse_args()

    if not args.tasks_dir.exists():
        log.error("Tasks directory not found: %s", args.tasks_dir)
        sys.exit(1)

    # Discover tasks
    if args.task_ids:
        task_dirs = [args.tasks_dir / tid for tid in args.task_ids]
    else:
        task_dirs = sorted(d for d in args.tasks_dir.iterdir() if d.is_dir())

    if not task_dirs:
        log.error("No task directories found in %s", args.tasks_dir)
        sys.exit(1)

    log.info("Validating %d task directories...", len(task_dirs))

    all_errors: list[str] = []
    passed = 0
    failed = 0

    for task_dir in task_dirs:
        if not task_dir.exists():
            log.error("Task dir not found: %s", task_dir)
            all_errors.append(f"[{task_dir.name}] Directory does not exist")
            failed += 1
            continue

        errors = _validate_task(task_dir)
        if errors:
            failed += 1
            all_errors.extend(errors)
            for e in errors:
                log.error(e)
        else:
            passed += 1

    print("\n=== Structural Parity Results ===")
    print(f"  Passed: {passed}/{len(task_dirs)}")
    print(f"  Failed: {failed}/{len(task_dirs)}")

    if all_errors:
        print(f"\n  Errors ({len(all_errors)}):")
        for e in all_errors:
            print(f"    {e}")
        sys.exit(1)
    else:
        print("  All tasks passed structural validation.")


if __name__ == "__main__":
    main()
