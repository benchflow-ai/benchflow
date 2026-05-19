"""Generate BenchFlow task directories from HILBench SWE instances.

HILBench (Human-in-the-Loop Benchmark) evaluates AI coding agents on
software engineering tasks that may require clarification from humans.
This module generates BenchFlow task directories for the SWE baseline
subset — 100 tasks across 4 repositories (ansible, protonmail/webclients,
navidrome, flipt-io/flipt).

The dataset is loaded from HuggingFace (ScaleAI/hil-bench).  Each task
includes a problem statement, a test patch, and a list of tests the
agent's solution must pass.  Evaluation applies the test patch and runs
pytest, awarding partial credit based on the fraction of tests passed.

Usage:
    python benchmarks/hilbench/benchflow.py --output-dir /tmp/hilbench-tasks
    python benchmarks/hilbench/benchflow.py --output-dir /tmp/hilbench-tasks --limit 10
    python benchmarks/hilbench/benchflow.py --output-dir /tmp/hilbench-tasks \
        --task-ids public_swe_0,public_swe_1
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

logger = logging.getLogger(__name__)

# ── Timeout presets by repo (larger repos get more time) ─────────────

_REPO_TIMEOUTS: dict[str, tuple[int, int]] = {
    # (agent_timeout, verifier_timeout)
    "ansible/ansible": (3600, 300),
    "protonmail/webclients": (3600, 300),
    "navidrome/navidrome": (3600, 300),
    "flipt-io/flipt": (3600, 300),
}
_DEFAULT_TIMEOUT = (3600, 300)


@dataclass
class HILBenchTask:
    task_id: str
    task_type: str
    repo_name: str
    download_link: str
    problem: str
    test_patch: str
    tests_to_pass: list[str]
    test_files: list[str]
    ground_truth_answer: str
    blocker_registry: list[dict] = field(default_factory=list)
    uid: str = ""


def _sanitize_name(raw: str) -> str:
    """Lowercase, replace non-alphanumeric with hyphens, collapse runs."""
    name = raw.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-")


def load_tasks_from_hf(
    *,
    task_type: str = "swe",
) -> list[HILBenchTask]:
    """Load HILBench tasks from HuggingFace, filtered by task_type."""
    from datasets import load_dataset

    ds = load_dataset("ScaleAI/hil-bench", split="train")
    tasks: list[HILBenchTask] = []
    for row in ds:
        if row["task_type"] != task_type:
            continue
        tasks.append(
            HILBenchTask(
                task_id=row["task_id"],
                task_type=row["task_type"],
                repo_name=row["repo_or_db_name"],
                download_link=row["repo_or_db_download_link"],
                problem=row["problem"],
                test_patch=row["test_patch"],
                tests_to_pass=row["tests_to_pass"],
                test_files=row["test_files"],
                ground_truth_answer=row["ground_truth_answer"],
                blocker_registry=row.get("blocker_registry", []),
                uid=row.get("uid", ""),
            )
        )
    return tasks


def _render_task_toml(task: HILBenchTask) -> str:
    agent_timeout, verifier_timeout = _REPO_TIMEOUTS.get(
        task.repo_name, _DEFAULT_TIMEOUT
    )
    name = f"hilbench/{_sanitize_name(task.task_id)}"
    n_tests = len(task.tests_to_pass)
    difficulty = "easy" if n_tests <= 3 else ("hard" if n_tests > 10 else "medium")
    return f"""\
version = "1.0"

[task]
name = "{name}"

[metadata]
author_name = "Scale AI"
difficulty = "{difficulty}"
category = "swe"
tags = ["hilbench", "swe", "{_sanitize_name(task.repo_name)}"]

[agent]
timeout_sec = {agent_timeout}

[verifier]
timeout_sec = {verifier_timeout}

[environment]
cpus = 2
memory_mb = 4096
storage_mb = 20480
"""


def _render_instruction(task: HILBenchTask) -> str:
    return f"""\
# SWE Task: {task.repo_name}

{task.problem}

## Repository

The repository `{task.repo_name}` is available at `/workspace/` inside the container.

## Deliverables

Modify the source code in `/workspace/` to fix the issue described above.
Your changes will be evaluated by running the project's test suite.
"""


def _render_dockerfile(task: HILBenchTask) -> str:
    """Generate Dockerfile using the pre-built HILBench Docker image as base.

    Each HILBench SWE task ships a Docker image tarball on HuggingFace
    (``ScaleAI/hil-bench-swe-images``) that contains the repository at the
    correct commit plus the SWEAP test harness.  The runner downloads and
    loads the tarball via ``docker load``, then tags it as
    ``hilbench-base:<sanitized_task_id>`` so the Dockerfile can reference
    it directly without needing a build arg.
    """
    base_tag = f"hilbench-base:{_sanitize_name(task.task_id)}"
    return f"""\
# HILBench SWE task environment.
# The runner tags the pre-built HuggingFace image as {base_tag}.
FROM {base_tag}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \\
    apt-get install -y -qq \\
        git \\
        python3 \\
        python3-pip \\
        curl \\
        jq \\
    && rm -rf /var/lib/apt/lists/*

# HILBench SWE images keep the repository under /app and expose /testbed as
# a compatibility symlink. BenchFlow tasks and instructions use /workspace.
RUN if [ -d /app ]; then \\
        rm -rf /workspace && ln -s /app /workspace; \\
    elif [ -d /testbed ]; then \\
        rm -rf /workspace && ln -s /testbed /workspace; \\
    fi

WORKDIR /workspace

# BenchFlow log directories
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts

# Label with the HuggingFace download link for traceability
LABEL hilbench.download_link="{task.download_link}"
LABEL hilbench.uid="{task.uid}"
"""


def _render_test_sh(task: HILBenchTask) -> str:
    return Template("""\
#!/bin/bash
# Verifier for HILBench SWE task: $task_id
set -o pipefail

exec > >(tee /logs/verifier/verifier.log) 2>&1

python3 /tests/verify.py \\
    --task-id "$task_id" \\
    --workspace /workspace \\
    --tests-to-pass-file /tests/tests_to_pass.json \\
    --reward-file /logs/verifier/reward.txt
""").safe_substitute(task_id=task.task_id)


# ── verify.py (copied into every task's tests/) ──────────────────────

VERIFY_PY = '''\
"""HILBench SWE verifier for BenchFlow.

Applies the test patch, runs the specified tests via pytest, and writes
a partial-credit reward based on the fraction of FAIL_TO_PASS tests
that now pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _apply_patch(workspace: Path, patch_file: Path) -> bool:
    """Apply the test patch to the workspace."""
    if not patch_file.exists():
        print(f"ERROR: Test patch not found at {patch_file}")
        return False
    result = subprocess.run(
        ["git", "apply", "--verbose", str(patch_file)],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"WARNING: git apply failed, trying with --3way: {result.stderr[:500]}")
        result = subprocess.run(
            ["git", "apply", "--3way", str(patch_file)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"ERROR: git apply --3way also failed: {result.stderr[:500]}")
            return False
    print("Test patch applied successfully.")
    return True


def _run_tests(workspace: Path, tests: list[str]) -> tuple[int, int]:
    """Run specified tests and return (passed, total)."""
    if not tests:
        return 0, 0

    total = len(tests)
    passed = 0

    for test_id in tests:
        result = subprocess.run(
            ["python3", "-m", "pytest", test_id, "-x", "--tb=short", "-q"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            passed += 1
            print(f"  PASS: {test_id}")
        else:
            print(f"  FAIL: {test_id}")
            if result.stdout:
                print(f"    stdout: {result.stdout[-300:]}")
            if result.stderr:
                print(f"    stderr: {result.stderr[-300:]}")

    return passed, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--tests-to-pass-file", required=True, type=Path)
    parser.add_argument("--reward-file", required=True, type=Path)
    args = parser.parse_args()

    reward_file: Path = args.reward_file
    reward_file.parent.mkdir(parents=True, exist_ok=True)

    tests_to_pass: list[str] = json.loads(args.tests_to_pass_file.read_text())

    # Step 1: Apply test patch
    print("=== Step 1: Applying test patch ===")
    patch_file = Path("/tests/test_patch.diff")
    if not _apply_patch(args.workspace, patch_file):
        reward_file.write_text("0")
        sys.exit(0)

    # Step 2: Run tests
    print(f"=== Step 2: Running {len(tests_to_pass)} tests ===")
    passed, total = _run_tests(args.workspace, tests_to_pass)

    # Step 3: Compute reward
    reward = passed / total if total > 0 else 0.0
    reward_file.write_text(f"{reward:.6f}")
    print(f"\\n=== Result: {passed}/{total} = {reward:.4f} ===")


if __name__ == "__main__":
    main()
'''


def generate_task(
    task: HILBenchTask, output_dir: Path, *, overwrite: bool = False
) -> Path:
    """Generate a single BenchFlow task directory for one HILBench instance."""
    sanitized_id = _sanitize_name(task.task_id)
    task_dir = output_dir / sanitized_id
    if task_dir.exists():
        if not overwrite:
            logger.debug("Skipping existing task %s", task.task_id)
            return task_dir
        shutil.rmtree(task_dir)

    task_dir.mkdir(parents=True)

    # task.toml
    (task_dir / "task.toml").write_text(_render_task_toml(task))

    # instruction.md
    (task_dir / "instruction.md").write_text(_render_instruction(task))

    # environment/Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(_render_dockerfile(task))

    # tests/
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()

    test_sh = tests_dir / "test.sh"
    test_sh.write_text(_render_test_sh(task))
    test_sh.chmod(test_sh.stat().st_mode | stat.S_IEXEC)

    (tests_dir / "verify.py").write_text(VERIFY_PY)

    # Save tests_to_pass as a separate JSON file (avoids shell quoting issues)
    (tests_dir / "tests_to_pass.json").write_text(
        json.dumps(task.tests_to_pass, indent=2)
    )

    # Save the test patch
    (tests_dir / "test_patch.diff").write_text(task.test_patch)

    # Save metadata for reference
    (tests_dir / "task_metadata.json").write_text(
        json.dumps(
            {
                "task_id": task.task_id,
                "repo_name": task.repo_name,
                "download_link": task.download_link,
                "tests_to_pass": task.tests_to_pass,
                "test_files": task.test_files,
                "uid": task.uid,
            },
            indent=2,
        )
    )

    return task_dir


def generate_all(
    output_dir: Path,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    task_ids: list[str] | None = None,
) -> list[Path]:
    """Generate BenchFlow task directories for HILBench SWE tasks."""
    tasks = load_tasks_from_hf(task_type="swe")

    if task_ids:
        id_set = set(task_ids)
        tasks = [t for t in tasks if t.task_id in id_set]

    if limit is not None:
        tasks = tasks[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for task in tasks:
        path = generate_task(task, output_dir, overwrite=overwrite)
        generated.append(path)
        logger.info("Generated %s", task.task_id)

    logger.info("Generated %d tasks in %s", len(generated), output_dir)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BenchFlow tasks from HILBench SWE instances"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write generated task directories",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of tasks generated",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing task directories",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated list of specific task IDs to generate",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    task_id_list = args.task_ids.split(",") if args.task_ids else None
    generated = generate_all(
        args.output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        task_ids=task_id_list,
    )
    print(f"Generated {len(generated)} task directories in {args.output_dir}")


if __name__ == "__main__":
    main()
