"""Standalone parity test for ProgramBench → BenchFlow pipeline.

Validates the task generation, Docker image build, and verification
pipeline end-to-end by:
1. Building the BenchFlow Docker image for a task
2. Querying Gemini to produce a solution (compile.sh + source)
3. Running the BenchFlow verifier inside the container
4. Reporting the reward

Usage::

    GOOGLE_API_KEY=... python benchmarks/programbench/parity_test.py \\
        --tasks-dir benchmarks/programbench/tasks \\
        --task-ids abishekvashok__cmatrix.5c082c6

    python benchmarks/programbench/parity_test.py \\
        --tasks-dir /tmp/programbench-task-md \\
        --task-ids example__tool.abcdef0 \\
        --task-format task-md \\
        --mode structural
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

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

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
LEGACY_REQUIRED_FILES = [
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "tests/test.sh",
    "tests/verify.py",
    "tests/tests.json",
    "solution/solve.sh",
]
TASK_MD_REQUIRED_FILES = [
    "task.md",
    "environment/Dockerfile",
    "verifier/test.sh",
    "verifier/verify.py",
    "verifier/tests.json",
    "verifier/verifier.md",
    "verifier/rubrics/verifier.md",
    "oracle/solve.sh",
]


def _gemini_generate(api_key: str, model: str, prompt: str) -> str:
    """Call Gemini API and return the text response."""
    url = f"{GEMINI_API_URL}/{model}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _build_image(task_dir: Path, tag: str) -> bool:
    """Build the BenchFlow Docker image for a task."""
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        log.error("No Dockerfile at %s", dockerfile)
        return False
    result = subprocess.run(
        ["docker", "build", "-t", tag, str(dockerfile.parent)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        log.error("Docker build failed:\n%s", result.stderr[-2000:])
        return False
    return True


def _read_instruction(task_dir: Path) -> str:
    task_md = task_dir / "task.md"
    if task_md.exists():
        return TaskDocument.from_path(task_md).instruction
    return (task_dir / "instruction.md").read_text()


def _get_task_context(task_dir: Path, tag: str) -> str:
    """Read instruction.md and inspect the container to build agent prompt."""
    instruction = _read_instruction(task_dir)

    # Get README/docs from container
    result = subprocess.run(
        ["docker", "run", "--rm", tag, "cat", "/workspace/README.md"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    readme = result.stdout[:4000] if result.returncode == 0 else "(no README)"

    # Get help output
    result = subprocess.run(
        ["docker", "run", "--rm", tag, "/workspace/executable", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    help_text = (
        (result.stdout or result.stderr)[:2000]
        if result.returncode == 0
        else "(help not available)"
    )

    return f"""{instruction}

## README.md from the original project
{readme}

## --help output
{help_text}

## Your task

Based on the above, write a complete implementation. You MUST output:
1. A `compile.sh` script that builds your code into `./executable`
2. All source code files needed

Output your files in this exact format (no markdown fences around the whole thing):

FILE: compile.sh
```
#!/bin/bash
<your build script>
```

FILE: <filename>
```
<file contents>
```

Keep your implementation simple and focused on reproducing the binary's behavior.
"""


def _extract_files(response: str) -> dict[str, str]:
    """Parse Gemini's response into filename -> content dict."""
    files: dict[str, str] = {}
    current_file = None
    current_lines: list[str] = []
    in_code_block = False

    for line in response.split("\n"):
        if line.startswith("FILE:"):
            if current_file and current_lines:
                files[current_file] = "\n".join(current_lines)
            current_file = line.split("FILE:", 1)[1].strip()
            current_lines = []
            in_code_block = False
        elif line.startswith("```") and current_file:
            in_code_block = not in_code_block
        elif in_code_block and current_file:
            current_lines.append(line)

    if current_file and current_lines:
        files[current_file] = "\n".join(current_lines)

    return files


def _run_parity(
    task_dir: Path,
    task_id: str,
    api_key: str,
    model: str,
    task_format: str,
) -> float:
    """Run one parity test. Returns the reward (0.0–1.0)."""
    tag = f"benchflow-parity:{task_id.replace('/', '_')}"

    # Step 1: Build Docker image
    log.info("[%s] Building Docker image...", task_id)
    if not _build_image(task_dir, tag):
        return -1.0

    # Step 2: Get context and query Gemini
    log.info("[%s] Querying %s...", task_id, model)
    prompt = _get_task_context(task_dir, tag)
    try:
        response = _gemini_generate(api_key, model, prompt)
    except Exception as exc:
        log.error("[%s] Gemini API error: %s", task_id, exc)
        return -1.0

    # Step 3: Extract files from response
    files = _extract_files(response)
    if "compile.sh" not in files:
        log.warning("[%s] No compile.sh in response, trying raw response", task_id)
        files["compile.sh"] = "#!/bin/bash\necho 'No compile.sh generated'"

    log.info(
        "[%s] Got %d files from Gemini: %s", task_id, len(files), list(files.keys())
    )

    # Step 4: Create submission and run verifier
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Write agent files
        for fname, content in files.items():
            fpath = tmp / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)

        verifier_name = "verifier" if task_format == "task-md" else "tests"
        verifier_mount = "/verifier" if task_format == "task-md" else "/tests"
        verifier_dir = task_dir.resolve() / verifier_name

        # Run container with agent files mounted + verifier
        container_name = f"parity-{task_id.replace('/', '_').replace('.', '-')}"
        cmd = [
            "docker",
            "run",
            "--name",
            container_name,
            "--rm",
            # Mount agent submission files
            "-v",
            f"{tmp}:/agent_submission:ro",
            # Mount verifier files
            "-v",
            f"{verifier_dir}:{verifier_mount}:ro",
            "-e",
            f"BENCHFLOW_VERIFIER_DIR={verifier_mount}",
            tag,
            "bash",
            "-c",
            # Copy agent files to workspace, then run verifier
            "cp -r /agent_submission/* /workspace/ && "
            "chmod +x /workspace/compile.sh 2>/dev/null; "
            f"bash {verifier_mount}/test.sh; "
            "cat /logs/verifier/reward.txt 2>/dev/null || echo '-1'",
        ]
        log.info("[%s] Running verifier...", task_id)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )

        log.info("[%s] Verifier stdout:\n%s", task_id, result.stdout[-3000:])
        if result.stderr:
            log.info("[%s] Verifier stderr:\n%s", task_id, result.stderr[-1000:])

        # Extract reward from last line
        lines = result.stdout.strip().split("\n")
        try:
            reward = float(lines[-1])
        except (ValueError, IndexError):
            log.warning("[%s] Could not parse reward from output", task_id)
            reward = 0.0

    return reward


def _validate_tests_json(
    task_dir: Path, *, verifier_dir: Path, task_id: str
) -> list[str]:
    errors: list[str] = []
    tests_json = verifier_dir / "tests.json"
    if not tests_json.exists():
        return [f"[{task_id}] Missing file: {tests_json.relative_to(task_dir)}"]
    try:
        tests_data = json.loads(tests_json.read_text())
    except json.JSONDecodeError as exc:
        return [f"[{task_id}] tests.json invalid JSON: {exc}"]
    branches = tests_data.get("branches")
    if not isinstance(branches, dict) or not branches:
        errors.append(f"[{task_id}] tests.json missing non-empty branches map")
    return errors


def _validate_structural_task(
    task_dir: Path,
    *,
    task_id: str,
    task_format: str,
) -> list[str]:
    errors: list[str] = []
    verifier_dir = task_dir / ("verifier" if task_format == "task-md" else "tests")
    required_files = (
        TASK_MD_REQUIRED_FILES if task_format == "task-md" else LEGACY_REQUIRED_FILES
    )

    validation_level = "publication-grade" if task_format == "task-md" else "structural"
    errors.extend(
        f"[{task_id}] bench tasks check: {issue}"
        for issue in check_task(task_dir, validation_level=validation_level)
    )

    for rel_path in required_files:
        file_path = task_dir / rel_path
        if not file_path.exists():
            errors.append(f"[{task_id}] Missing file: {rel_path}")
        elif file_path.is_file() and file_path.stat().st_size == 0:
            errors.append(f"[{task_id}] Empty file: {rel_path}")

    errors.extend(
        _validate_tests_json(task_dir, verifier_dir=verifier_dir, task_id=task_id)
    )

    if task_format == "task-md":
        for rel_path in ("task.toml", "instruction.md", "tests", "solution"):
            if (task_dir / rel_path).exists():
                errors.append(
                    f"[{task_id}] native task.md output must not keep {rel_path}"
                )
        task_md = task_dir / "task.md"
        if task_md.exists():
            try:
                document = TaskDocument.from_path(task_md)
            except TaskDocumentParseError as exc:
                errors.append(f"[{task_id}] task.md parse error: {exc}")
            else:
                task_name = (
                    document.config.task.name
                    if document.config.task is not None
                    else ""
                )
                if task_name != f"programbench/{task_id}":
                    errors.append(f"[{task_id}] task.md name mismatch: {task_name!r}")
                if "Compiled executable" not in document.instruction:
                    errors.append(
                        f"[{task_id}] task.md prompt missing executable contract"
                    )
    else:
        task_toml = task_dir / "task.toml"
        if task_toml.exists() and 'name = "programbench/' not in task_toml.read_text():
            errors.append(f"[{task_id}] task.toml missing programbench/ prefix")
        instruction = task_dir / "instruction.md"
        if (
            instruction.exists()
            and "Compiled executable" not in instruction.read_text()
        ):
            errors.append(f"[{task_id}] instruction.md missing executable contract")

    return errors


def _run_structural_parity(
    tasks_dir: Path,
    task_ids: list[str],
    *,
    task_format: str,
) -> bool:
    all_errors: list[str] = []
    passed = 0

    for task_id in task_ids:
        task_dir = tasks_dir / task_id
        if not task_dir.exists():
            all_errors.append(f"[{task_id}] Task dir not found: {task_dir}")
            continue
        errors = _validate_structural_task(
            task_dir,
            task_id=task_id,
            task_format=task_format,
        )
        if errors:
            all_errors.extend(errors)
            for error in errors:
                log.error(error)
        else:
            passed += 1

    print("\n=== Structural Parity Results ===")
    print(f"  Passed: {passed}/{len(task_ids)}")
    print(f"  Failed: {len(task_ids) - passed}/{len(task_ids)}")

    if all_errors:
        print(f"\n  Errors ({len(all_errors)}):")
        for error in all_errors:
            print(f"    {error}")
        return False
    print("  All tasks passed structural validation.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="ProgramBench parity test")
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument(
        "--mode",
        choices=("live", "structural"),
        default="live",
        help="Parity mode. structural runs offline package checks only.",
    )
    parser.add_argument(
        "--api-key", default=None, help="Gemini API key (or set GOOGLE_API_KEY)"
    )
    parser.add_argument(
        "--model", default="gemini-2.0-flash-lite", help="Gemini model name"
    )
    parser.add_argument(
        "--task-format",
        choices=("legacy", "task-md"),
        default="legacy",
        help="Generated task layout to validate.",
    )
    args = parser.parse_args()

    import os

    if args.mode == "structural":
        ok = _run_structural_parity(
            args.tasks_dir,
            args.task_ids,
            task_format=args.task_format,
        )
        sys.exit(0 if ok else 1)

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    results: dict[str, float] = {}
    for task_id in args.task_ids:
        task_dir = args.tasks_dir / task_id
        if not task_dir.exists():
            log.error("Task dir not found: %s", task_dir)
            results[task_id] = -1.0
            continue
        reward = _run_parity(task_dir, task_id, api_key, args.model, args.task_format)
        results[task_id] = reward
        log.info("[%s] Reward: %.4f", task_id, reward)

    print("\n=== Parity Test Results ===")
    for tid, reward in results.items():
        status = "ERROR" if reward < 0 else f"{reward:.4f}"
        print(f"  {tid}: {status}")
    valid = [r for r in results.values() if r >= 0]
    if valid:
        print(f"  Average: {sum(valid) / len(valid):.4f}")


if __name__ == "__main__":
    main()
