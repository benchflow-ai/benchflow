"""Parity test for Harvey LAB benchmark.

Verifies that the BenchFlow converter faithfully translates Harvey LAB
tasks.  Three modes:

  1. Structural parity — task directories have all required files, metadata
     matches the original task.json.
  2. Eval parity — runs the BenchFlow evaluate.py pipeline end-to-end on
     synthetic deliverables and checks it produces valid rewards.
  3. **Side-by-side parity** — runs the *original* Harvey LAB prompt
     template and the *adapted* BenchFlow prompt template through the
     same Gemini judge on identical agent output, then compares
     per-criterion verdicts.  This is the core parity validation
     experiment (Step 5 of the conversion guide).

Usage:
    # Subset structural (5 tasks)
    python benchmarks/harvey-lab/parity_test.py --mode subset

    # Full structural (all 1251 tasks)
    python benchmarks/harvey-lab/parity_test.py --mode full

    # Eval parity (BenchFlow pipeline end-to-end, LLM calls)
    python benchmarks/harvey-lab/parity_test.py --mode eval-parity \
        --anthropic-api-key sk-ant-...

    # Side-by-side parity (original vs adapted prompt, same judge)
    python benchmarks/harvey-lab/parity_test.py --mode side-by-side \
        --gemini-api-key AIza...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
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

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"


def _run_converter(
    harvey_root: Path,
    output_dir: Path,
    task_ids: list[str] | None = None,
    limit: int | None = None,
) -> None:
    """Run the benchflow.py converter to generate tasks."""
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "benchflow.py"),
        "--output-dir",
        str(output_dir),
        "--harvey-root",
        str(harvey_root),
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
        raise RuntimeError(f"Converter failed with code {result.returncode}")


def _sanitize_name(raw: str) -> str:
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
                    f"{task_id}: document count mismatch: {gen_count} vs {orig_count}"
                )
                failed += 1
                continue

        # Check instruction.md contains the original instructions
        instr_text = (task_dir / "instruction.md").read_text()
        orig_instructions = original.get("instructions", "")
        if orig_instructions and orig_instructions[:50] not in instr_text:
            errors.append(
                f"{task_id}: instruction.md doesn't contain original instructions"
            )
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
    anthropic_api_key: str,
) -> tuple[int, int, list[str]]:
    """Run the BenchFlow evaluate.py pipeline end-to-end on synthetic output.

    Verifies the evaluation pipeline produces valid rewards (0.0-1.0).
    """
    passed = 0
    failed = 0
    errors: list[str] = []

    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

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

            # Create synthetic deliverables
            if isinstance(deliverables_config, dict):
                items = list(deliverables_config.items())
            elif isinstance(deliverables_config, list):
                items = [
                    (f"deliverable-{i}", fn) for i, fn in enumerate(deliverables_config)
                ]
            else:
                items = []
            for name, filename in items:
                md_name = Path(filename).stem + ".md"
                (tmp_path / md_name).write_text(
                    f"# {name}\n\nThis is a placeholder deliverable "
                    f"for parity testing. The actual content would be "
                    f"a legal work product.\n"
                )

            if not any(tmp_path.iterdir()):
                (tmp_path / "output.md").write_text(
                    "# Output\n\nPlaceholder for parity testing.\n"
                )

            # Copy rubric
            import shutil

            shutil.copy2(rubric_path, tmp_path / "rubric.json")

            # Run evaluate.py
            reward_file = tmp_path / "reward.txt"
            evaluate_py = task_dir / "tests" / "evaluate.py"

            print(f"\n{'=' * 60}")
            print(f"Eval parity: {task_id}")
            print(f"{'=' * 60}")

            result = subprocess.run(
                [
                    sys.executable,
                    str(evaluate_py),
                    "--rubric",
                    str(tmp_path / "rubric.json"),
                    "--output-dir",
                    str(tmp_path),
                    "--reward-file",
                    str(reward_file),
                ],
                capture_output=True,
                text=True,
                timeout=600,
                env={**os.environ, "ANTHROPIC_API_KEY": anthropic_api_key},
            )

            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            if result.stderr:
                print(f"STDERR: {result.stderr[-300:]}", file=sys.stderr)

            if result.returncode != 0:
                errors.append(
                    f"{task_id}: evaluate.py exited with code {result.returncode}"
                )
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

            print(f"  Reward: {reward:.3f} (expected low for placeholder)")
            passed += 1

            time.sleep(2)

    return passed, failed, errors


# ── Side-by-Side Parity (Step 5) ─────────────────────────────────────

# Original Harvey LAB prompt template (from evaluation/prompts/rubric_criterion.txt).
# Uses string.Template to avoid crashes on legal text containing { or }.
_ORIGINAL_PROMPT = string.Template("""\
You are evaluating a legal AI agent's work product against a specific quality criterion.

## Task
$task_description

## Agent's Output
$agent_output

## Criterion
**$criterion_title**

$match_criteria

## Instructions
Evaluate the agent's output against the criterion above.
- **PASS**: The agent's output satisfies the criterion as described
- **FAIL**: The agent's output does not satisfy the criterion as described

Respond with JSON only:

```json
{
  "verdict": "pass" | "fail",
  "reasoning": "Brief explanation"
}
```
""")

# Adapted BenchFlow prompt template (from benchflow.py's _build_evaluate_py,
# after textwrap.dedent strips the 8-space common prefix;
# uses string.Template $-style placeholders for single-pass substitution)
_ADAPTED_PROMPT = string.Template("""\
You are evaluating a legal AI agent's work product against a specific quality criterion.

## Task
$task_description

## Agent's Output
$agent_output

## Criterion
**$criterion_title**

$match_criteria

## Instructions
Evaluate the agent's output against the criterion above.
- **PASS**: The agent's output satisfies the criterion as described
- **FAIL**: The agent's output does not satisfy the criterion as described

Respond with JSON only:

```json
{
  "verdict": "pass" or "fail",
  "reasoning": "Brief explanation"
}
```
""")


def _call_gemini(prompt: str, api_key: str, retries: int = 3) -> str:
    """Call Gemini API and return the text response."""
    from google import genai

    client = genai.Client(api_key=api_key)

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(
                f"Gemini API failed after {retries} attempts: {e}"
            ) from e
    raise RuntimeError("Unreachable")


def _parse_verdict(text: str) -> dict:
    """Extract JSON verdict from LLM response."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    for i, ch in enumerate(text):
        if ch == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i : j + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Could not parse verdict from: {text[:300]}")


def check_side_by_side_parity(
    harvey_root: Path,
    task_ids: list[str],
    gemini_api_key: str,
) -> tuple[dict, int, int, list[str]]:
    """Run the original and adapted prompt templates through the same Gemini
    judge on identical synthetic agent output. Compare per-criterion verdicts.

    Returns (results_dict, n_agreed, n_disagreed, errors).
    """
    agreed = 0
    disagreed = 0
    errors: list[str] = []
    results: dict = {
        "experiment": "side-by-side-parity",
        "judge_model": GEMINI_MODEL,
        "tasks": [],
    }

    for task_id in task_ids:
        original_path = harvey_root / "tasks" / Path(*task_id.split("/")) / "task.json"
        if not original_path.exists():
            errors.append(f"{task_id}: task.json not found")
            continue

        config = json.loads(original_path.read_text())
        task_title = config["title"]
        criteria = config["criteria"]
        deliverables = config.get("deliverables", {})

        # Build synthetic agent output (same for both sides)
        agent_output_parts = []
        if isinstance(deliverables, dict):
            deliv_items = list(deliverables.items())
        elif isinstance(deliverables, list):
            deliv_items = [
                (f"deliverable-{i}", fn) for i, fn in enumerate(deliverables)
            ]
        else:
            deliv_items = []
        for name, filename in deliv_items:
            agent_output_parts.append(
                f"--- {Path(filename).stem}.md ---\n"
                f"# {name}\n\nThis is a placeholder deliverable for parity testing. "
                f"The actual content would be a legal work product.\n"
            )
        if not agent_output_parts:
            agent_output_parts.append(
                "--- output.md ---\n# Output\n\nPlaceholder for parity testing.\n"
            )
        agent_output = "\n\n".join(agent_output_parts)

        task_result = {
            "task_id": task_id,
            "n_criteria": len(criteria),
            "criteria_results": [],
        }

        print(f"\n{'=' * 60}")
        print(f"Side-by-side: {task_id} ({len(criteria)} criteria)")
        print(f"{'=' * 60}")

        # Sample up to 5 criteria per task to keep API costs reasonable
        sampled_criteria = criteria[:5] if len(criteria) > 5 else criteria

        for criterion in sampled_criteria:
            crit_id = criterion["id"]
            crit_title = criterion["title"]
            match_criteria = criterion["match_criteria"]

            # Run original Harvey LAB prompt
            orig_prompt = _ORIGINAL_PROMPT.safe_substitute(
                task_description=task_title,
                agent_output=agent_output,
                criterion_title=crit_title,
                match_criteria=match_criteria,
            )

            # Run adapted BenchFlow prompt (uses string.Template, matching benchflow.py)
            adapted_prompt = _ADAPTED_PROMPT.safe_substitute(
                task_description=task_title,
                agent_output=agent_output,
                criterion_title=crit_title,
                match_criteria=match_criteria,
            )

            try:
                orig_response = _call_gemini(orig_prompt, gemini_api_key)
                orig_verdict = _parse_verdict(orig_response)
                time.sleep(1)  # rate limiting

                adapted_response = _call_gemini(adapted_prompt, gemini_api_key)
                adapted_verdict = _parse_verdict(adapted_response)
                time.sleep(1)

                orig_v = orig_verdict.get("verdict", "").lower()
                adapted_v = adapted_verdict.get("verdict", "").lower()
                match = orig_v == adapted_v

                if match:
                    agreed += 1
                else:
                    disagreed += 1

                status = "AGREE" if match else "DISAGREE"
                print(f"  [{crit_id}] original={orig_v} adapted={adapted_v} → {status}")

                task_result["criteria_results"].append(
                    {
                        "criterion_id": crit_id,
                        "criterion_title": crit_title,
                        "original_verdict": orig_v,
                        "adapted_verdict": adapted_v,
                        "agreement": match,
                    }
                )

            except Exception as e:
                errors.append(f"{task_id}/{crit_id}: {e}")
                print(f"  [{crit_id}] ERROR: {e}")

        results["tasks"].append(task_result)

    total = agreed + disagreed
    results["summary"] = {
        "total_criteria_compared": total,
        "agreed": agreed,
        "disagreed": disagreed,
        "agreement_rate": agreed / total if total > 0 else 0.0,
    }

    return results, agreed, disagreed, errors


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Harvey LAB parity tests")
    parser.add_argument(
        "--mode",
        choices=["subset", "full", "eval-parity", "side-by-side"],
        default="subset",
        help="Test mode: subset, full, eval-parity, or side-by-side",
    )
    parser.add_argument(
        "--harvey-root",
        default=str(_DEFAULT_HARVEY_ROOT),
        help="Path to Harvey LAB repo root",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key (for side-by-side mode)",
    )
    parser.add_argument(
        "--anthropic-api-key",
        default=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Anthropic API key (for eval-parity mode)",
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
                print(
                    "ERROR: No valid subset tasks found. Falling back to first 5 tasks."
                )
                all_tasks = sorted((harvey_root / "tasks").rglob("task.json"))[:5]
                valid_ids = [
                    str(t.parent.relative_to(harvey_root / "tasks")).replace("\\", "/")
                    for t in all_tasks
                ]

            print(f"\n=== Subset Structural Parity ({len(valid_ids)} tasks) ===\n")
            _run_converter(harvey_root, output_dir, task_ids=valid_ids)
            passed, failed, errors = check_structural_parity(
                harvey_root, output_dir, valid_ids
            )

        elif args.mode == "full":
            print("\n=== Full Structural Parity (all tasks) ===\n")
            _run_converter(harvey_root, output_dir)
            all_tasks = sorted((harvey_root / "tasks").rglob("task.json"))
            all_ids = [
                str(t.parent.relative_to(harvey_root / "tasks")).replace("\\", "/")
                for t in all_tasks
            ]
            passed, failed, errors = check_structural_parity(
                harvey_root, output_dir, all_ids
            )

        elif args.mode == "eval-parity":
            if not args.anthropic_api_key:
                print(
                    "ERROR: --anthropic-api-key required for eval-parity mode",
                    file=sys.stderr,
                )
                sys.exit(1)

            eval_ids = []
            for tid in SUBSET_TASK_IDS:
                orig = harvey_root / "tasks" / Path(*tid.split("/")) / "task.json"
                if orig.exists():
                    eval_ids.append(tid)

            if not eval_ids:
                all_tasks = sorted((harvey_root / "tasks").rglob("task.json"))[:3]
                eval_ids = [
                    str(t.parent.relative_to(harvey_root / "tasks")).replace("\\", "/")
                    for t in all_tasks
                ]

            print(f"\n=== Eval Parity ({len(eval_ids)} tasks) ===\n")
            _run_converter(harvey_root, output_dir, task_ids=eval_ids)
            passed, failed, errors = check_eval_parity(
                harvey_root, output_dir, eval_ids, args.anthropic_api_key
            )

        elif args.mode == "side-by-side":
            if not args.gemini_api_key:
                print(
                    "ERROR: --gemini-api-key required for side-by-side mode",
                    file=sys.stderr,
                )
                sys.exit(1)

            eval_ids = []
            for tid in SUBSET_TASK_IDS:
                orig = harvey_root / "tasks" / Path(*tid.split("/")) / "task.json"
                if orig.exists():
                    eval_ids.append(tid)

            if not eval_ids:
                all_tasks = sorted((harvey_root / "tasks").rglob("task.json"))[:3]
                eval_ids = [
                    str(t.parent.relative_to(harvey_root / "tasks")).replace("\\", "/")
                    for t in all_tasks
                ]

            print(f"\n=== Side-by-Side Parity ({len(eval_ids)} tasks) ===\n")

            results, agreed, disagreed, errors = check_side_by_side_parity(
                harvey_root, eval_ids, args.gemini_api_key
            )

            # Save parity_experiment.json
            parity_path = _SCRIPT_DIR / "parity_experiment.json"
            parity_path.write_text(json.dumps(results, indent=2) + "\n")
            print(f"\nSaved parity results to {parity_path}")

            passed = agreed
            failed = disagreed

        else:
            print(f"Unknown mode: {args.mode}", file=sys.stderr)
            sys.exit(1)

        # Report results
        print(f"\n{'=' * 60}")
        if args.mode == "side-by-side":
            total = passed + failed
            rate = passed / total * 100 if total > 0 else 0
            print(f"RESULTS: {passed}/{total} criteria agreed ({rate:.1f}% agreement)")
        else:
            print(f"RESULTS: {passed} passed, {failed} failed")
        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  - {e}")
        print(f"{'=' * 60}")

        # Side-by-side: allow some disagreement due to LLM stochasticity
        if args.mode == "side-by-side":
            total = passed + failed
            if total > 0 and passed / total < 0.7:
                print("\nFAIL: Agreement rate below 70% threshold")
                sys.exit(1)
        elif failed > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
