"""Generate BenchFlow task directories from ProgramBench instances.

ProgramBench evaluates whether LM agents can reverse-engineer black-box
software systems — given a compiled binary and its docs, the agent must
re-implement the program from scratch.  This module generates one
BenchFlow task directory per ProgramBench instance.

Requires a local checkout of the ProgramBench repo (or the installed
``programbench`` package) for task metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import yaml

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
_repo_src_path = str(_REPO_SRC)
if _repo_src_path in sys.path:
    sys.path.remove(_repo_src_path)
sys.path.insert(0, _repo_src_path)

from benchflow.task.document import render_task_md  # noqa: E402
from benchflow.task.output_format import (  # noqa: E402
    TASK_OUTPUT_FORMATS,
    TaskOutputFormat,
    ensure_existing_task_output_format,
    oracle_dir_name,
    validate_task_output_format,
    verifier_dir_name,
)

logger = logging.getLogger(__name__)

DOCKER_ORG = "programbench"

# Language display names

_LANG_DISPLAY: dict[str, str] = {
    "rs": "Rust",
    "go": "Go",
    "c": "C",
    "cpp": "C++",
    "hs": "Haskell",
    "java": "Java",
    "bash": "Bash",
}

# Timeout presets by difficulty

_TIMEOUTS: dict[str, tuple[int, int]] = {
    # (agent_timeout, verifier_timeout)
    "easy": (3600, 1200),
    "medium": (5400, 1800),
    "hard": (7200, 2400),
}
_DEFAULT_TIMEOUT = (5400, 1800)
TASK_FORMATS = TASK_OUTPUT_FORMATS
TaskFormat = TaskOutputFormat


@dataclass
class ProgramBenchTask:
    instance_id: str
    repository: str
    commit: str
    language: str
    difficulty: str
    eval_clean_hashes: list[str]
    tests_json: dict


def _image_name(instance_id: str) -> str:
    return f"{DOCKER_ORG}/{instance_id.replace('__', '_1776_')}"


def load_tasks(tasks_dir: Path) -> list[ProgramBenchTask]:
    """Load all ProgramBench tasks from a tasks directory."""
    tasks: list[ProgramBenchTask] = []
    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir():
            continue
        task_yaml = d / "task.yaml"
        if not task_yaml.exists():
            continue
        cfg = yaml.safe_load(task_yaml.read_text())
        tests_json_path = d / "tests.json"
        tests_json = (
            json.loads(tests_json_path.read_text()) if tests_json_path.exists() else {}
        )
        tasks.append(
            ProgramBenchTask(
                instance_id=d.name,
                repository=cfg.get("repository", ""),
                commit=cfg.get("commit", ""),
                language=cfg.get("language", "unknown"),
                difficulty=cfg.get("difficulty", "medium"),
                eval_clean_hashes=cfg.get("eval_clean_hashes", []),
                tests_json=tests_json,
            )
        )
    return tasks


def _render_task_toml(task: ProgramBenchTask) -> str:
    agent_timeout, verifier_timeout = _TIMEOUTS.get(task.difficulty, _DEFAULT_TIMEOUT)
    lang_display = _LANG_DISPLAY.get(task.language, task.language)
    name = f"programbench/{task.instance_id}"
    return f"""\
version = "1.0"

[task]
name = "{name}"

[metadata]
author_name = "ProgramBench (Meta FAIR)"
difficulty = "{task.difficulty}"
category = "programming"
tags = ["program-reconstruction", "{lang_display.lower()}"]

[agent]
timeout_sec = {agent_timeout}

[verifier]
timeout_sec = {verifier_timeout}

[environment]
cpus = 2
memory_mb = 4096
storage_mb = 20480
allow_internet = false
"""


def _render_task_md(task: ProgramBenchTask) -> str:
    agent_timeout, verifier_timeout = _TIMEOUTS.get(task.difficulty, _DEFAULT_TIMEOUT)
    lang_display = _LANG_DISPLAY.get(task.language, task.language)
    instruction = _render_instruction(task).strip()
    frontmatter: dict[str, Any] = {
        "schema_version": "1.3",
        "task": {
            "name": f"programbench/{task.instance_id}",
        },
        "metadata": {
            "author_name": "ProgramBench (Meta FAIR)",
            "difficulty": task.difficulty,
            "category": "programming",
            "tags": ["program-reconstruction", lang_display.lower()],
        },
        "agent": {
            "timeout_sec": agent_timeout,
        },
        "verifier": {
            "timeout_sec": verifier_timeout,
        },
        "environment": {
            "cpus": 2,
            "memory_mb": 4096,
            "storage_mb": 20480,
            "allow_internet": False,
        },
        "benchflow": {
            "document_version": "0.3",
            "source": {
                "benchmark": "ProgramBench",
                "repository": task.repository,
                "commit": task.commit,
                "language": task.language,
                "instance_id": task.instance_id,
            },
            "verifier": {
                "spec": "verifier/verifier.md",
                "rubric": "verifier/rubrics/verifier.md",
                "entrypoint": "verifier/test.sh",
                "implementation": {
                    "type": "test-script",
                    "outputs": {
                        "reward_json": "/logs/verifier/reward.json",
                        "reward_details": "/logs/verifier/reward-details.json",
                    },
                },
            },
        },
    }
    return render_task_md(frontmatter, instruction)


def _render_instruction(task: ProgramBenchTask) -> str:
    lang_display = _LANG_DISPLAY.get(task.language, task.language)
    return f"""\
# Program Reconstruction: {task.repository}

You are given a compiled executable and its documentation.
Your task is to **re-implement the program from scratch** so that your
implementation reproduces the original program's behavior.

## What you have

| Resource | Location |
|----------|----------|
| Compiled executable | `/workspace/executable` |
| Documentation | `/workspace/docs/` (if present) |

Run the executable with `--help` or various inputs to understand its behavior.

## Rules

1. You **must not** decompile or disassemble the executable.
2. You **may** choose any programming language (the original was written in {lang_display}).
3. You **must** create a file called `compile.sh` in `/workspace/` that
   builds your implementation and produces a binary named `executable` in
   the current working directory.
4. You have access to standard development tools installed in the container.

## Deliverables

1. All source code files in `/workspace/`.
2. A build script `/workspace/compile.sh` that, when run, produces
   `./executable`.

## Evaluation

Your implementation will be tested with behavioral tests that compare its
output against the original executable.  Partial credit is awarded based
on the fraction of tests passed.
"""


def _render_dockerfile(task: ProgramBenchTask) -> str:
    image = _image_name(task.instance_id)
    # Use :task (not :task_cleanroom) so the evaluation container has the
    # same build toolchains and system libraries that ProgramBench's own
    # ``programbench eval`` uses.  Wipe the workspace source and reset it
    # to match the cleanroom state (binary + non-source assets only).
    return f"""\
FROM {image}:task

WORKDIR /workspace

# Reset workspace to cleanroom state: wipe source, keep binary + assets.
# ProgramBench eval does the same: start from :task, wipe workspace,
# extract agent submission, then compile.
RUN rm -rf /workspace/.git /workspace/src /workspace/Cargo.* /workspace/Makefile* \\
           /workspace/CMakeLists.txt /workspace/configure* /workspace/meson* \\
           /workspace/*.c /workspace/*.h /workspace/*.rs /workspace/*.go \\
           /workspace/*.java /workspace/*.hs /workspace/*.cabal /workspace/go.* \\
           /workspace/pom.xml /workspace/build.gradle /workspace/setup.py \\
           /workspace/pyproject.toml /workspace/package.json 2>/dev/null; true
# Re-initialize a clean git repo (matches cleanroom's single "Initial commit")
RUN cd /workspace && git init -q && git add -A && \\
    git -c user.email=clean@local -c user.name=clean commit -q --allow-empty -m 'Initial commit' 2>/dev/null; true

# BenchFlow log directories
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts

# Verification dependencies — the verifier downloads test archives from
# HuggingFace and runs pytest suites against the agent's executable.
RUN apt-get update -qq && \\
    apt-get install -y -qq python3 python3-pip git jq && \\
    rm -rf /var/lib/apt/lists/* 2>/dev/null; true

RUN pip3 install --quiet huggingface_hub pyyaml junitparser || \\
    pip3 install --break-system-packages --quiet huggingface_hub pyyaml junitparser
"""


def _render_test_sh(task: ProgramBenchTask) -> str:
    clean_hashes_json = json.dumps(task.eval_clean_hashes)
    return Template("""\
#!/bin/bash
# Verifier for ProgramBench task: $instance_id
# Compiles the agent's submission, downloads test blobs, runs behavioral tests.
set -euo pipefail

verifier_log="${BENCHFLOW_VERIFIER_LOG:-/logs/verifier/verifier.log}"
mkdir -p "$(dirname "$verifier_log")"
exec > >(tee "$verifier_log") 2>&1

VERIFIER_DIR="${BENCHFLOW_VERIFIER_DIR:-/verifier}"
LEGACY_TESTS_DIR="${BENCHFLOW_LEGACY_TESTS_DIR:-/tests}"
if [ ! -f "$VERIFIER_DIR/verify.py" ] && [ -f "$LEGACY_TESTS_DIR/verify.py" ]; then
    VERIFIER_DIR="$LEGACY_TESTS_DIR"
fi

workspace="${BENCHFLOW_WORKSPACE:-/workspace}"
reward_file="${BENCHFLOW_REWARD_TEXT:-/logs/verifier/reward.txt}"
reward_json="${BENCHFLOW_REWARD_JSON:-/logs/verifier/reward.json}"
details_json="${BENCHFLOW_REWARD_DETAILS_JSON:-/logs/verifier/reward-details.json}"
mkdir -p "$(dirname "$reward_file")" "$(dirname "$reward_json")" "$(dirname "$details_json")"

python3 "$VERIFIER_DIR/verify.py" \\
    --instance-id "$instance_id" \\
    --workspace "$workspace" \\
    --clean-hashes '$clean_hashes' \\
    --tests-json "$VERIFIER_DIR/tests.json" \\
    --reward-file "$reward_file"

python3 - "$reward_file" "$reward_json" "$details_json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

reward_path = Path(sys.argv[1])
reward_json_path = Path(sys.argv[2])
details_json_path = Path(sys.argv[3])
reward = float(reward_path.read_text().strip())
reward_json_path.write_text(
    json.dumps({"reward": reward}, indent=2) + "\\n"
)
details_json_path.write_text(
    json.dumps(
        {
            "reward": reward,
            "partial_credit_reward": reward,
            "source": "programbench-behavioral-tests",
        },
        indent=2,
    )
    + "\\n"
)
PY
""").safe_substitute(instance_id=task.instance_id, clean_hashes=clean_hashes_json)


def _render_verifier_md(task: ProgramBenchTask) -> str:
    frontmatter: dict[str, Any] = {
        "document_version": "0.3",
        "verifier": {
            "name": f"programbench-{task.instance_id}-verifier",
            "default_strategy": "deterministic",
            "strategies": {
                "deterministic": {
                    "type": "script",
                    "command": "./test.sh",
                },
            },
            "rubric": {
                "combine": "weighted_sum",
                "dimensions": {
                    "build_contract": {
                        "weight": 0.20,
                        "source": "deterministic",
                    },
                    "anti_cheat_cleanroom": {
                        "weight": 0.20,
                        "source": "deterministic",
                    },
                    "behavioral_tests": {
                        "weight": 0.60,
                        "source": "deterministic",
                    },
                },
            },
            "outputs": {
                "reward_text": "/logs/verifier/reward.txt",
                "reward_json": "/logs/verifier/reward.json",
                "details_json": "/logs/verifier/reward-details.json",
            },
        },
    }
    rendered_frontmatter = yaml.safe_dump(frontmatter, sort_keys=False)
    return (
        f"---\n{rendered_frontmatter}---\n\n## role:reviewer\n\n"
        "The deterministic verifier removes known cleanroom binary hashes, "
        "runs the submitted build script, downloads ProgramBench test branches "
        "from HuggingFace, and awards partial credit from behavioral tests.\n"
    )


def _render_verifier_rubric(task: ProgramBenchTask) -> str:
    return f"""\
# ProgramBench Rubric

Task: `programbench/{task.instance_id}`

- Build contract: the submission must provide `/workspace/compile.sh` and
  produce `./executable`.
- Anti-cheat cleanroom: files matching the original executable hashes are
  removed before compilation.
- Behavioral tests: the rebuilt executable is scored by the official
  ProgramBench test branch archives.

The bundled verifier reports partial credit as `passed_tests / total_tests`.
"""


# verify.py (copied into every task's verifier package)

VERIFY_PY = '''\
"""ProgramBench verifier for BenchFlow.

Compiles the agent's submission, downloads test branch archives from
HuggingFace, runs each branch's pytest suite, parses JUnit XML results,
and writes a partial-credit reward.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _compile(workspace: Path) -> bool:
    """Run compile.sh and return True on success."""
    compile_sh = workspace / "compile.sh"
    if not compile_sh.exists():
        print("ERROR: No compile.sh found in workspace")
        return False
    compile_sh.chmod(0o755)
    result = subprocess.run(
        ["bash", str(compile_sh)],
        cwd=str(workspace),
        timeout=900,
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"ERROR: compile.sh exited with code {result.returncode}")
        return False
    if not (workspace / "executable").exists():
        print("ERROR: compile.sh did not produce ./executable")
        return False
    return True


def _remove_hashed_files(workspace: Path, hashes: list[str]) -> None:
    """Remove files whose SHA-256 matches any of the given hashes.

    Prevents trivial solutions that just copy the original binary.
    """
    if not hashes:
        return
    hash_set = set(hashes)
    result = subprocess.run(
        ["find", str(workspace), "-type", "f", "-exec", "sha256sum", "{}", "+"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    for line in result.stdout.strip().splitlines():
        parts = line.split("  ", 1)
        if len(parts) == 2 and parts[0] in hash_set:
            path = Path(parts[1])
            print(f"Removing hash-matched file: {path}")
            path.unlink(missing_ok=True)


def _stash_executable(workspace: Path) -> Path:
    """Move the compiled executable to a safe location."""
    stash = Path("/opt/benchflow-stashed-executable")
    exe = workspace / "executable"
    if exe.exists():
        import shutil
        shutil.move(str(exe), str(stash))
        stash.chmod(0o755)
    return stash


def _restore_executable(stash: Path, workspace: Path) -> None:
    """Copy the stashed executable back into workspace."""
    import shutil
    dest = workspace / "executable"
    shutil.copy2(str(stash), str(dest))
    dest.chmod(0o755)


def _download_test_blob(instance_id: str, branch: str) -> Path:
    """Download a test branch archive from HuggingFace."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id="programbench/ProgramBench-Tests",
            filename=f"{instance_id}/tests/{branch}.tar.gz",
            repo_type="dataset",
        )
    )


def _run_test_branch(
    workspace: Path, instance_id: str, branch: str, stash: Path
) -> tuple[int, int]:
    """Run one test branch and return (passed, total)."""
    # Download and extract test archive
    try:
        blob_path = _download_test_blob(instance_id, branch)
    except Exception as exc:
        print(f"  WARNING: failed to download blob for branch {branch}: {exc}")
        return 0, 0

    try:
        subprocess.run(
            ["tar", "xzf", str(blob_path), "-C", str(workspace)],
            check=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"  WARNING: failed to extract archive for branch {branch}: {exc}")
        return 0, 0

    # Restore executable
    _restore_executable(stash, workspace)

    # Patch timeout method (same as ProgramBench eval)
    run_sh = workspace / "eval" / "run.sh"
    if run_sh.exists():
        content = run_sh.read_text()
        content = content.replace("--timeout-method=thread", "--timeout-method=signal")
        run_sh.write_text(content)
        run_sh.chmod(0o755)
        # Run tests
        try:
            subprocess.run(
                ["bash", str(run_sh)],
                cwd=str(workspace),
                timeout=3600,
                capture_output=False,
            )
        except subprocess.TimeoutExpired:
            print(f"  WARNING: test run timed out for branch {branch}")

    # Parse results
    passed = 0
    total = 0
    results_xml = workspace / "eval" / "results.xml"
    if results_xml.exists():
        try:
            raw = results_xml.read_text()
            root = ET.fromstring(raw)
            for tc in root.iter("testcase"):
                total += 1
                has_fail = any(
                    c.tag in ("failure", "error") for c in tc
                )
                if not has_fail:
                    passed += 1
        except ET.ParseError as exc:
            print(f"  WARNING: failed to parse results.xml: {exc}")

    # Clean up test artifacts for next branch
    eval_dir = workspace / "eval"
    if eval_dir.exists():
        import shutil
        shutil.rmtree(eval_dir, ignore_errors=True)

    return passed, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--clean-hashes", default="[]")
    parser.add_argument("--tests-json", required=True, type=Path)
    parser.add_argument("--reward-file", required=True, type=Path)
    args = parser.parse_args()

    reward_file: Path = args.reward_file
    reward_file.parent.mkdir(parents=True, exist_ok=True)

    # Load tests.json from the selected verifier package directory.
    tests_json_path = args.tests_json
    if not tests_json_path.exists():
        print(f"ERROR: {tests_json_path} not found")
        reward_file.write_text("0")
        sys.exit(0)

    tests_data = json.loads(tests_json_path.read_text())
    branches = tests_data.get("branches", {})
    clean_hashes: list[str] = json.loads(args.clean_hashes)

    # Step 1: Remove hash-matched files from submission (anti-cheat).
    # Must happen BEFORE compile — matches ProgramBench's eval order.
    # The hashes identify the original pre-compiled binary; removing them
    # before compile prevents trivial copies while allowing a legitimate
    # rebuild that happens to produce a byte-identical executable.
    print("=== Step 1: Checking for copied binaries ===")
    _remove_hashed_files(args.workspace, clean_hashes)

    # Step 2: Compile
    print("=== Step 2: Compiling submission ===")
    if not _compile(args.workspace):
        reward_file.write_text("0")
        sys.exit(0)

    # Step 3: Stash executable
    stash = _stash_executable(args.workspace)

    # Step 4: Run test branches
    print("=== Step 3: Running behavioral tests ===")
    total_passed = 0
    total_tests = 0
    active_branches = [
        name for name, info in branches.items() if not info.get("ignored")
    ]

    for i, branch in enumerate(active_branches):
        print(f"  Branch {i + 1}/{len(active_branches)}: {branch}")
        passed, total = _run_test_branch(
            args.workspace, args.instance_id, branch, stash
        )
        total_passed += passed
        total_tests += total
        print(f"    {passed}/{total} tests passed")

    # Step 5: Compute reward
    reward = total_passed / total_tests if total_tests > 0 else 0.0
    reward_file.write_text(f"{reward:.6f}")
    print(f"\\n=== Result: {total_passed}/{total_tests} = {reward:.4f} ===")


if __name__ == "__main__":
    main()
'''


def _render_solve_sh(task: ProgramBenchTask) -> str:
    """Generate oracle solution script that checks out the original source."""
    return f"""\
#!/bin/bash
# Oracle solution: check out the original source code from the upstream repo.
# The task asks agents to reconstruct the program from a compiled binary —
# the gold answer is the original source at the specified commit.
set -euo pipefail

cd /workspace

# Clone the original repository and check out the exact commit
git clone https://github.com/{task.repository}.git _oracle_src
cd _oracle_src
git checkout {task.commit}
cd /workspace

# Copy source into workspace (overwriting the cleanroom state)
cp -a _oracle_src/. .
rm -rf _oracle_src .git

# Run the existing compile.sh to produce the executable
if [ -f compile.sh ]; then
    chmod +x compile.sh
    bash compile.sh
fi
"""


def generate_task(
    task: ProgramBenchTask,
    output_dir: Path,
    *,
    overwrite: bool = False,
    task_format: TaskFormat = "task-md",
) -> Path:
    """Generate a single BenchFlow task directory for one ProgramBench instance."""
    task_format = validate_task_output_format(task_format)
    task_dir = output_dir / task.instance_id
    if task_dir.exists():
        if not overwrite:
            ensure_existing_task_output_format(task_dir, task_format)
            logger.debug("Skipping existing task %s", task.instance_id)
            return task_dir
        shutil.rmtree(task_dir)

    task_dir.mkdir(parents=True)

    if task_format == "task-md":
        (task_dir / "task.md").write_text(_render_task_md(task))
    else:
        # task.toml
        (task_dir / "task.toml").write_text(_render_task_toml(task))

        # instruction.md
        (task_dir / "instruction.md").write_text(_render_instruction(task))

    # environment/Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(_render_dockerfile(task))

    # oracle/ for native task.md, solution/ for legacy Harbor/Pier layout.
    sol_dir = task_dir / oracle_dir_name(task_format)
    sol_dir.mkdir()
    solve_sh = sol_dir / "solve.sh"
    solve_sh.write_text(_render_solve_sh(task))
    solve_sh.chmod(0o755)

    # verifier/ for native task.md, tests/ for legacy Harbor/Pier layout.
    tests_dir = task_dir / verifier_dir_name(task_format)
    tests_dir.mkdir()

    test_sh = tests_dir / "test.sh"
    test_sh.write_text(_render_test_sh(task))
    test_sh.chmod(0o755)

    (tests_dir / "verify.py").write_text(VERIFY_PY)
    if task_format == "task-md":
        (tests_dir / "verifier.md").write_text(_render_verifier_md(task))
        rubrics_dir = tests_dir / "rubrics"
        rubrics_dir.mkdir()
        (rubrics_dir / "verifier.md").write_text(_render_verifier_rubric(task))

    # Copy tests.json for the verifier
    if task.tests_json:
        (tests_dir / "tests.json").write_text(json.dumps(task.tests_json, indent=2))

    return task_dir


def generate_all(
    tasks_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    task_ids: list[str] | None = None,
    task_format: TaskFormat = "task-md",
) -> list[Path]:
    """Generate BenchFlow task directories for all ProgramBench tasks."""
    task_format = validate_task_output_format(task_format)
    tasks = load_tasks(tasks_dir)

    if task_ids:
        id_set = set(task_ids)
        tasks = [t for t in tasks if t.instance_id in id_set]

    if limit is not None:
        tasks = tasks[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for task in tasks:
        path = generate_task(
            task,
            output_dir,
            overwrite=overwrite,
            task_format=task_format,
        )
        generated.append(path)
        logger.info("Generated %s", task.instance_id)

    logger.info("Generated %d tasks in %s", len(generated), output_dir)
    return generated


def _resolve_tasks_dir(explicit: Path | None) -> Path:
    """Find the ProgramBench tasks directory."""
    if explicit is not None:
        # Accept either the repo root or the data/tasks dir directly.
        candidate = explicit / "src" / "programbench" / "data" / "tasks"
        if candidate.is_dir():
            return candidate
        if (explicit / "task.yaml").exists() or any(explicit.glob("*/task.yaml")):
            return explicit
        raise FileNotFoundError(f"No ProgramBench tasks found at {explicit}")

    # Try the installed package.
    try:
        from programbench.constants import TASKS_DIR

        if TASKS_DIR.is_dir():
            return TASKS_DIR
    except ImportError:
        pass

    # Try common local paths.
    for path in [
        Path.cwd() / "programbench",
        Path.home() / "programbench",
    ]:
        candidate = path / "src" / "programbench" / "data" / "tasks"
        if candidate.is_dir():
            return candidate

    print(
        "ERROR: Cannot find ProgramBench tasks. Either:\n"
        "  1. Install programbench: pip install programbench\n"
        "  2. Pass --programbench-dir /path/to/programbench",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    """CLI entry point — generate BenchFlow tasks from ProgramBench instances."""
    parser = argparse.ArgumentParser(
        description="Generate BenchFlow tasks from ProgramBench instances.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write generated task directories into.",
    )
    parser.add_argument(
        "--programbench-dir",
        type=Path,
        default=None,
        help=(
            "Path to ProgramBench repo or data/tasks directory. "
            "If omitted, tries the installed programbench package."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Generate at most N tasks."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing task directories.",
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Only generate these instance IDs.",
    )
    parser.add_argument(
        "--task-format",
        choices=TASK_FORMATS,
        default="task-md",
        help="Output layout: legacy task.toml/instruction.md or native task.md.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    tasks_dir = _resolve_tasks_dir(args.programbench_dir)
    generated = generate_all(
        tasks_dir,
        args.output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        task_ids=args.task_ids,
        task_format=args.task_format,
    )
    print(f"Generated {len(generated)} tasks in {args.output_dir}")


if __name__ == "__main__":
    main()
