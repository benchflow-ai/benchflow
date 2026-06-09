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
from benchflow.task.paths import TaskPaths  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LEGACY_REQUIRED_FILES = [
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "tests/test.sh",
    "tests/evaluate.py",
    "tests/ground_truth.json",
    "solution/solve.sh",
]

TASK_MD_REQUIRED_FILES = [
    "task.md",
    "environment/Dockerfile",
    "verifier/test.sh",
    "verifier/evaluate.py",
    "verifier/ground_truth.json",
    "verifier/verifier.md",
    "verifier/rubrics/verifier.md",
    "oracle/solve.sh",
]


def _validate_task(task_dir: Path, *, task_format: str = "legacy") -> list[str]:
    """Validate a single task directory. Returns list of error messages."""
    errors: list[str] = []
    task_id = task_dir.name
    paths = TaskPaths(task_dir)

    validation_level = "publication-grade" if task_format == "task-md" else "structural"
    errors.extend(
        f"[{task_id}] bench tasks check: {issue}"
        for issue in check_task(task_dir, validation_level=validation_level)
    )

    # Check required files
    required_files = (
        TASK_MD_REQUIRED_FILES if task_format == "task-md" else LEGACY_REQUIRED_FILES
    )
    for rel in required_files:
        fpath = task_dir / rel
        if not fpath.exists():
            errors.append(f"[{task_id}] Missing file: {rel}")
        elif fpath.stat().st_size == 0:
            errors.append(f"[{task_id}] Empty file: {rel}")

    # Validate task.toml/task.md
    toml_path = task_dir / "task.toml"
    task_md_path = task_dir / "task.md"
    if task_format == "task-md":
        if (task_dir / "instruction.md").exists() or toml_path.exists():
            errors.append(
                f"[{task_id}] native task.md output must not keep split-layout files"
            )
        if task_md_path.exists():
            try:
                document = TaskDocument.from_path(task_md_path)
            except TaskDocumentParseError as e:
                errors.append(f"[{task_id}] task.md parse error: {e}")
            else:
                task_name = (
                    document.config.task.name
                    if document.config.task is not None
                    else ""
                )
                if not task_name.startswith("opaquetoolsbench/"):
                    errors.append(
                        f"[{task_id}] task.md missing opaquetoolsbench/ prefix in name"
                    )
                if "/app/output/response.json" not in document.instruction:
                    errors.append(f"[{task_id}] task.md missing output path reference")
                if "## Query" not in document.instruction:
                    errors.append(
                        f"[{task_id}] task.md prompt missing ## Query section"
                    )
                if "## Available Functions" not in document.instruction:
                    errors.append(
                        f"[{task_id}] task.md prompt missing ## Available Functions"
                    )
            if not task_md_path.read_text().lstrip().startswith("---"):
                errors.append(f"[{task_id}] task.md missing YAML frontmatter")
    elif toml_path.exists():
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
    if task_format == "legacy" and instr_path.exists():
        content = instr_path.read_text()
        if "## Query" not in content:
            errors.append(f"[{task_id}] instruction.md missing ## Query section")
        if "## Available Functions" not in content:
            errors.append(f"[{task_id}] instruction.md missing ## Available Functions")
        if "/app/output/response.json" not in content:
            errors.append(f"[{task_id}] instruction.md missing output path reference")

    # Validate ground_truth.json
    verifier_dir = paths.tests_dir
    oracle_dir = paths.solution_dir
    gt_path = verifier_dir / "ground_truth.json"
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
    for script in (paths.test_path, oracle_dir / "solve.sh"):
        if not script.exists():
            continue

        mode = script.stat().st_mode
        if not (mode & stat.S_IXUSR):
            errors.append(
                f"[{task_id}] {script.relative_to(task_dir)} is not executable"
            )

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
    parser.add_argument(
        "--task-format",
        choices=("legacy", "task-md"),
        default="legacy",
        help="Generated task layout to validate",
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

        errors = _validate_task(task_dir, task_format=args.task_format)
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
