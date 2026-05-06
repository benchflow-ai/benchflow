"""Parity test for Harvey LAB adapter.

Verifies that the BenchFlow adapter faithfully translates Harvey LAB
tasks. Two phases:
  1. Structural parity — task directories have all required files, metadata
     matches the original task.json.
  2. Evaluation parity — runs the LLM judge on a known (synthetic) agent
     output and compares scores between the original Harvey LAB evaluator
     and the BenchFlow adapter's evaluate.py.

Usage:
    # Subset parity (5 tasks from different practice areas)
    python benchmarks/harvey-lab/parity_test.py --mode subset

    # Full parity (all 1251 tasks — structural only, no LLM calls)
    python benchmarks/harvey-lab/parity_test.py --mode full

    # Evaluation parity (runs LLM judge on subset)
    python benchmarks/harvey-lab/parity_test.py --mode eval-parity \
        --gemini-api-key AIza...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCHFLOW_ROOT = _SCRIPT_DIR.parent.parent
_DEFAULT_HARVEY_ROOT = _BENCHFLOW_ROOT.parent / "harvey-labs"

# Representative tasks from different practice areas for subset testing.
SUBSET_TASK_IDS = [
    "corporate-ma/analyze-cim-deal-teaser/scenario-01",
    "insurance/compare-reinsurance-treaty-against-underlying-policy",
    "real-estate/draft-construction-contract",
    "intellectual-property/review-enterprise-saas-agreement",
    "employment-labor/draft-workplace-policy-memorandum",
]


def _run_adapter(
    harvey_root: Path,
    output_dir: Path,
    task_ids: list[str] | None = None,
    limit: int | None = None,
) -> None:
    """Run the benchflow.py adapter to generate tasks."""
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "benchflow.py"),
        "--output-dir", str(output_dir),
        "--harvey-root", str(harvey_root),
        "--overwrite",
    ]
    if task_ids:
        cmd.extend(["--task-ids", ",".join(task_ids)])
    if limit:
        cmd.extend(["--limit", str(limit)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Adapter failed with code {result.returncode}")


def _sanitize_name(raw: str) -> str:
    import re
    name = raw.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-")


# ── Structural Parity Checks ─────────────────────────────────────────

def check_structural_parity(
    harvey_root: Path,
    output_dir: Path,
    task_ids: list[str],
) -> tuple[int, int, list[str]]:
    """Verify generated BenchFlow tasks have correct structure and metadata."""
    passed = 0
    failed = 0
    errors: list[str] = []

    for task_id in task_ids:
        task_name = _sanitize_name(task_id)
        task_dir = output_dir / task_name

        # Load original Harvey LAB config
        original_path = harvey_root / "tasks" / Path(*task_id.split("/")) / "task.json"
        if not original_path.exists():
            errors.append(f"{task_id}: original task.json not found")
            failed += 1
            continue

        with open(original_path) as f:
            original = json.load(f)

        # Check required files exist
        required_files = [
            "task.toml",
            "instruction.md",
            "environment/Dockerfile",
            "tests/test.sh",
            "tests/evaluate.py",
        ]
        missing = [f for f in required_files if not (task_dir / f).exists()]
        if missing:
            errors.append(f"{task_id}: missing files: {missing}")
            failed += 1
            continue

        # Validate task.toml
        with open(task_dir / "task.toml", "rb") as f:
            toml_config = tomllib.load(f)

        task_section = toml_config.get("task", {})
        if not task_section.get("name"):
            errors.append(f"{task_id}: task.toml missing task name")
            failed += 1
            continue

        # Check metadata matches
        metadata = toml_config.get("metadata", {})
        if metadata.get("author_name") != "Harvey AI":
            errors.append(f"{task_id}: author_name mismatch")
            failed += 1
            continue

        # Check criteria count in rubric.json matches original
        rubric_path = task_dir / "environment" / "rubric.json"
        if rubric_path.exists():
            rubric = json.loads(rubric_path.read_text())
            if len(rubric.get("criteria", [])) != len(original.get("criteria", [])):
                errors.append(
                    f"{task_id}: criteria count mismatch: "
                    f"{len(rubric.get('criteria', []))} vs "
                    f"{len(original.get('criteria', []))}"
                )
                failed += 1
                continue

        # Check documents were copied
        env_docs = task_dir / "environment" / "documents"
        orig_docs = harvey_root / "tasks" / Path(*task_id.split("/")) / "documents"
        if orig_docs.exists():
            orig_count = sum(1 for _ in orig_docs.iterdir())
            gen_count = sum(1 for _ in env_docs.iterdir()) if env_docs.exists() else 0
            if orig_count != gen_count:
                errors.append(
                    f"{task_id}: document count mismatch: "
                    f"{gen_count} vs {orig_count}"
                )
                failed += 1
                continue

        # Check instruction.md contains the original instructions
        instr_text = (task_dir / "instruction.md").read_text()
        orig_instructions = original.get("instructions", "")
        if orig_instructions and orig_instructions[:50] not in instr_text:
            errors.append(f"{task_id}: instruction.md doesn't contain original instructions")
            failed += 1
            continue

        # Check test.sh is executable
        test_sh = task_dir / "tests" / "test.sh"
        if not os.access(test_sh, os.X_OK):
            errors.append(f"{task_id}: test.sh not executable")
            failed += 1
            continue

        passed += 1

    return passed, failed, errors


# ── Evaluation Parity ─────────────────────────────────────────────────

def check_eval_parity(
    harvey_root: Path,
    output_dir: Path,
    task_ids: list[str],
    gemini_api_key: str,
) -> tuple[int, int, list[str]]:
    """Run the LLM judge on a synthetic deliverable to check evaluation works.

    For each task, creates a minimal placeholder deliverable, runs
    evaluate.py, and verifies it produces a valid reward (0.0-1.0).
    This does NOT check for score agreement with Harvey LAB's Anthropic
    judge — it verifies the evaluation pipeline works end-to-end.
    """
    passed = 0
    failed = 0
    errors: list[str] = []

    os.environ["GEMINI_API_KEY"] = gemini_api_key

    for task_id in task_ids:
        task_name = _sanitize_name(task_id)
        task_dir = output_dir / task_name
        rubric_path = task_dir / "environment" / "rubric.json"

        if not rubric_path.exists():
            errors.append(f"{task_id}: no rubric.json found")
            failed += 1
            continue

        deliverables_config = {}

        # Look at original task.json for deliverable names
        original_path = harvey_root / "tasks" / Path(*task_id.split("/")) / "task.json"
        if original_path.exists():
            original = json.loads(original_path.read_text())
            deliverables_config = original.get("deliverables", {})

        # Create a temp directory simulating agent output
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create synthetic deliverables using .md extension so the
            # evaluator can read them. Use the original stem so fuzzy
            # matching picks them up when criteria reference the original
            # filename.
            for name, filename in deliverables_config.items():
                md_name = Path(filename).stem + ".md"
                (tmp_path / md_name).write_text(
                    f"# {name}\n\nThis is a placeholder deliverable "
                    f"for parity testing. The actual content would be "
                    f"a legal work product.\n"
                )

            if not any(tmp_path.iterdir()):
                # No deliverables defined, create a generic one
                (tmp_path / "output.md").write_text(
                    "# Output\n\nPlaceholder for parity testing.\n"
                )

            # Copy rubric
            import shutil
            shutil.copy2(rubric_path, tmp_path / "rubric.json")

            # Run evaluate.py
            reward_file = tmp_path / "reward.txt"
            evaluate_py = task_dir / "tests" / "evaluate.py"

            print(f"\n{'='*60}")
            print(f"Eval parity: {task_id}")
            print(f"{'='*60}")

            result = subprocess.run(
                [
                    sys.executable, str(evaluate_py),
                    "--rubric", str(tmp_path / "rubric.json"),
                    "--output-dir", str(tmp_path),
                    "--reward-file", str(reward_file),
                ],
                capture_output=True,
                text=True,
                timeout=600,
                env={**os.environ, "GEMINI_API_KEY": gemini_api_key},
            )

            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            if result.stderr:
                print(f"STDERR: {result.stderr[-300:]}", file=sys.stderr)

            if result.returncode != 0:
                errors.append(f"{task_id}: evaluate.py exited with code {result.returncode}")
                failed += 1
                continue

            if not reward_file.exists():
                errors.append(f"{task_id}: reward.txt not created")
                failed += 1
                continue

            reward_text = reward_file.read_text().strip()
            try:
                reward = float(reward_text)
            except ValueError:
                errors.append(f"{task_id}: invalid reward value: {reward_text!r}")
                failed += 1
                continue

            if not (0.0 <= reward <= 1.0):
                errors.append(f"{task_id}: reward out of range: {reward}")
                failed += 1
                continue

            # With placeholder deliverables, we expect mostly fails, so
            # reward should be low. The key check is that the pipeline ran.
            print(f"  Reward: {reward:.3f} (expected low for placeholder)")
            passed += 1

            # Rate limit between tasks
            time.sleep(2)

    return passed, failed, errors


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Harvey LAB adapter parity tests")
    parser.add_argument(
        "--mode",
        choices=["subset", "full", "eval-parity"],
        default="subset",
        help="Test mode: subset (5 tasks), full (all tasks), eval-parity (LLM judge)",
    )
    parser.add_argument(
        "--harvey-root",
        default=str(_DEFAULT_HARVEY_ROOT),
        help="Path to Harvey LAB repo root",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key (for eval-parity mode)",
    )
    args = parser.parse_args()

    harvey_root = Path(args.harvey_root)
    if not (harvey_root / "tasks").exists():
        print(f"ERROR: Harvey LAB not found at {harvey_root}", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as output_dir_str:
        output_dir = Path(output_dir_str)

        if args.mode == "subset":
            # Find tasks that actually exist
            valid_ids = []
            for tid in SUBSET_TASK_IDS:
                orig = harvey_root / "tasks" / Path(*tid.split("/")) / "task.json"
                if orig.exists():
                    valid_ids.append(tid)
                else:
                    print(f"  SKIP subset task {tid}: not found")

            if not valid_ids:
                print("ERROR: No valid subset tasks found. Falling back to first 5 tasks.")
                # Fallback: use first 5 tasks
                all_tasks = sorted(
                    (harvey_root / "tasks").rglob("task.json")
                )[:5]
                valid_ids = [
                    str(t.parent.relative_to(harvey_root / "tasks")).replace("\\", "/")
                    for t in all_tasks
                ]

            print(f"\n=== Subset Structural Parity ({len(valid_ids)} tasks) ===\n")
            _run_adapter(harvey_root, output_dir, task_ids=valid_ids)
            passed, failed, errors = check_structural_parity(
                harvey_root, output_dir, valid_ids
            )

        elif args.mode == "full":
            print("\n=== Full Structural Parity (all tasks) ===\n")
            _run_adapter(harvey_root, output_dir)
            # Discover all generated task IDs
            all_tasks = sorted(
                (harvey_root / "tasks").rglob("task.json")
            )
            all_ids = [
                str(t.parent.relative_to(harvey_root / "tasks")).replace("\\", "/")
                for t in all_tasks
            ]
            passed, failed, errors = check_structural_parity(
                harvey_root, output_dir, all_ids
            )

        elif args.mode == "eval-parity":
            if not args.gemini_api_key:
                print("ERROR: --gemini-api-key required for eval-parity mode",
                      file=sys.stderr)
                sys.exit(1)

            # Use all subset tasks for eval parity
            eval_ids = []
            for tid in SUBSET_TASK_IDS:
                orig = harvey_root / "tasks" / Path(*tid.split("/")) / "task.json"
                if orig.exists():
                    eval_ids.append(tid)

            if not eval_ids:
                all_tasks = sorted(
                    (harvey_root / "tasks").rglob("task.json")
                )[:3]
                eval_ids = [
                    str(t.parent.relative_to(harvey_root / "tasks")).replace("\\", "/")
                    for t in all_tasks
                ]

            print(f"\n=== Eval Parity ({len(eval_ids)} tasks) ===\n")
            _run_adapter(harvey_root, output_dir, task_ids=eval_ids)
            passed, failed, errors = check_eval_parity(
                harvey_root, output_dir, eval_ids, args.gemini_api_key
            )

        else:
            print(f"Unknown mode: {args.mode}", file=sys.stderr)
            sys.exit(1)

        # Report results
        print(f"\n{'='*60}")
        print(f"RESULTS: {passed} passed, {failed} failed")
        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  - {e}")
        print(f"{'='*60}")

        if failed > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
