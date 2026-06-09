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
import stat
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from benchflow._utils.task_authoring import check_task  # noqa: E402
from benchflow.task.document import TaskDocument, TaskDocumentParseError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LEGACY_REQUIRED_FILES = [
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "tests/test.sh",
    "tests/verify.py",
    "tests/test_patch.diff",
    "tests/tests_to_pass.json",
    "tests/task_metadata.json",
    "solution/solve.sh",
    "solution/solve.patch",
]

TASK_MD_REQUIRED_FILES = [
    "task.md",
    "environment/Dockerfile",
    "verifier/test.sh",
    "verifier/verify.py",
    "verifier/test_patch.diff",
    "verifier/tests_to_pass.json",
    "verifier/task_metadata.json",
    "verifier/verifier.md",
    "verifier/rubrics/verifier.md",
    "oracle/solve.sh",
    "oracle/solve.patch",
]


def _check_task(task_dir: Path, *, task_format: str = "legacy") -> list[str]:
    """Check a single task directory for structural issues. Returns list of errors."""
    errors: list[str] = []
    task_id = task_dir.name
    verifier_dir = task_dir / ("verifier" if task_format == "task-md" else "tests")

    # Check required files exist
    required_files = (
        TASK_MD_REQUIRED_FILES if task_format == "task-md" else LEGACY_REQUIRED_FILES
    )
    for rel_path in required_files:
        fpath = task_dir / rel_path
        if not fpath.exists():
            errors.append(f"{task_id}: missing {rel_path}")

    validation_level = "publication-grade" if task_format == "task-md" else "structural"
    errors.extend(
        f"{task_id}: bench tasks check: {issue}"
        for issue in check_task(task_dir, validation_level=validation_level)
    )

    if task_format == "task-md":
        if any(
            (task_dir / rel).exists()
            for rel in ("task.toml", "instruction.md", "tests", "solution")
        ):
            errors.append(f"{task_id}: native task.md output keeps split-layout files")
        task_md = task_dir / "task.md"
        if task_md.exists():
            try:
                document = TaskDocument.from_path(task_md)
            except TaskDocumentParseError as exc:
                errors.append(f"{task_id}: task.md parse error: {exc}")
            else:
                task_name = (
                    document.config.task.name
                    if document.config.task is not None
                    else ""
                )
                if not task_name.startswith("hilbench/"):
                    errors.append(
                        f"{task_id}: task.md name does not start with 'hilbench/'"
                    )
                if "/workspace/" not in document.instruction:
                    errors.append(f"{task_id}: task.md prompt missing /workspace/")
        if task_md.exists() and not task_md.read_text().lstrip().startswith("---"):
            errors.append(f"{task_id}: task.md missing YAML frontmatter")

    # Check task.toml has required fields
    task_toml = task_dir / "task.toml"
    if task_format == "legacy" and task_toml.exists():
        content = task_toml.read_text()
        for field in ["[task]", "name =", "[metadata]", "[agent]", "[verifier]"]:
            if field not in content:
                errors.append(f"{task_id}: task.toml missing field '{field}'")
        if "hilbench/" not in content:
            errors.append(f"{task_id}: task.toml name does not start with 'hilbench/'")

    # Check instruction.md is non-empty
    instruction = task_dir / "instruction.md"
    if (
        task_format == "legacy"
        and instruction.exists()
        and instruction.stat().st_size == 0
    ):
        errors.append(f"{task_id}: instruction.md is empty")

    # Check test_patch.diff is non-empty
    patch = verifier_dir / "test_patch.diff"
    if patch.exists() and patch.stat().st_size == 0:
        errors.append(f"{task_id}: test_patch.diff is empty")

    # Check task_metadata.json is valid JSON with required fields
    meta = verifier_dir / "task_metadata.json"
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
    test_sh = verifier_dir / "test.sh"
    if test_sh.exists():
        mode = test_sh.stat().st_mode
        if not (mode & stat.S_IEXEC):
            errors.append(f"{task_id}: test.sh is not executable")

    solve_sh = (
        task_dir / ("oracle" if task_format == "task-md" else "solution") / "solve.sh"
    )
    if solve_sh.exists():
        mode = solve_sh.stat().st_mode
        if not (mode & stat.S_IEXEC):
            errors.append(f"{task_id}: solve.sh is not executable")

    # Check Dockerfile uses predictable hilbench-base tag
    dockerfile = task_dir / "environment" / "Dockerfile"
    if dockerfile.exists():
        content = dockerfile.read_text()
        if "FROM" not in content:
            errors.append(f"{task_id}: Dockerfile missing FROM directive")
        elif f"FROM hilbench-base:{task_id}" not in content:
            errors.append(
                f"{task_id}: Dockerfile FROM does not reference hilbench-base:{task_id}"
            )
        if "ln -s /app /workspace" not in content:
            errors.append(f"{task_id}: Dockerfile does not map /workspace to /app")

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
    parser.add_argument(
        "--task-format",
        choices=("legacy", "task-md"),
        default="legacy",
        help="Generated task layout to validate",
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

        errors = _check_task(task_dir, task_format=args.task_format)
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
