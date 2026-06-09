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
import sys
from dataclasses import dataclass, field
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

# Timeout presets by repo (larger repos get more time)

_REPO_TIMEOUTS: dict[str, tuple[int, int]] = {
    # (agent_timeout, verifier_timeout)
    "ansible/ansible": (3600, 300),
    "protonmail/webclients": (3600, 300),
    "navidrome/navidrome": (3600, 300),
    "flipt-io/flipt": (3600, 300),
}
_DEFAULT_TIMEOUT = (3600, 300)
TASK_FORMATS = TASK_OUTPUT_FORMATS
TaskFormat = TaskOutputFormat


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


def _difficulty_for_task(task: HILBenchTask) -> str:
    n_tests = len(task.tests_to_pass)
    return "easy" if n_tests <= 3 else ("hard" if n_tests > 10 else "medium")


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
    difficulty = _difficulty_for_task(task)
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


def _render_task_md(task: HILBenchTask) -> str:
    agent_timeout, verifier_timeout = _REPO_TIMEOUTS.get(
        task.repo_name, _DEFAULT_TIMEOUT
    )
    sanitized_id = _sanitize_name(task.task_id)
    instruction = _render_instruction(task).strip()
    frontmatter: dict[str, Any] = {
        "schema_version": "1.3",
        "task": {
            "name": f"hilbench/{sanitized_id}",
        },
        "metadata": {
            "author_name": "Scale AI",
            "difficulty": _difficulty_for_task(task),
            "category": "swe",
            "tags": ["hilbench", "swe", _sanitize_name(task.repo_name)],
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
        },
        "benchflow": {
            "document_version": "0.3",
            "source": {
                "benchmark": "HILBench",
                "task_id": task.task_id,
                "task_type": task.task_type,
                "repo_name": task.repo_name,
                "download_link": task.download_link,
                "uid": task.uid,
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
    --task-id "$task_id" \\
    --workspace "$workspace" \\
    --test-patch "$VERIFIER_DIR/test_patch.diff" \\
    --tests-to-pass-file "$VERIFIER_DIR/tests_to_pass.json" \\
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
            "source": "hilbench-fail-to-pass-tests",
        },
        indent=2,
    )
    + "\\n"
)
PY
""").safe_substitute(task_id=task.task_id)


def _render_verifier_md(task: HILBenchTask) -> str:
    frontmatter: dict[str, Any] = {
        "document_version": "0.3",
        "verifier": {
            "name": f"hilbench-{_sanitize_name(task.task_id)}-verifier",
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
                    "patch_applies": {
                        "weight": 0.25,
                        "source": "deterministic",
                    },
                    "fail_to_pass_tests": {
                        "weight": 0.75,
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
        "The deterministic verifier applies the HILBench test patch, runs the "
        "declared FAIL_TO_PASS pytest targets, and awards partial credit from "
        "the fraction of tests that pass.\n"
    )


def _render_verifier_rubric(task: HILBenchTask) -> str:
    tests = "\n".join(f"- `{test}`" for test in task.tests_to_pass) or "- none"
    return f"""\
# HILBench SWE Rubric

Task: `hilbench/{_sanitize_name(task.task_id)}`

- Patch applies: the HILBench test patch must apply cleanly to `/workspace`.
- FAIL_TO_PASS tests: reward is the fraction of declared tests that pass.

Tests to pass:

{tests}
"""


def _gold_patch(task: HILBenchTask) -> str:
    patch = task.ground_truth_answer.strip()
    if not patch:
        raise ValueError(
            f"HILBench task {task.task_id} is missing ground_truth_answer; "
            "cannot emit oracle solution"
        )
    return f"{patch}\n"


def _render_solve_sh(task: HILBenchTask) -> str:
    return Template("""\
#!/bin/bash
# Oracle solution for HILBench SWE task: $task_id
set -euo pipefail

workspace="${BENCHFLOW_WORKSPACE:-/workspace}"
oracle_dir="${BENCHFLOW_ORACLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
patch_file="${BENCHFLOW_ORACLE_PATCH:-$oracle_dir/solve.patch}"

cd "$workspace"
if ! git apply --verbose "$patch_file"; then
    git apply --3way "$patch_file"
fi
""").safe_substitute(task_id=task.task_id)


# verify.py (copied into every task's verifier package)

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
    parser.add_argument("--test-patch", required=True, type=Path)
    parser.add_argument("--tests-to-pass-file", required=True, type=Path)
    parser.add_argument("--reward-file", required=True, type=Path)
    args = parser.parse_args()

    reward_file: Path = args.reward_file
    reward_file.parent.mkdir(parents=True, exist_ok=True)

    tests_to_pass: list[str] = json.loads(args.tests_to_pass_file.read_text())

    # Step 1: Apply test patch
    print("=== Step 1: Applying test patch ===")
    if not _apply_patch(args.workspace, args.test_patch):
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
    task: HILBenchTask,
    output_dir: Path,
    *,
    overwrite: bool = False,
    task_format: TaskFormat = "task-md",
) -> Path:
    """Generate a single BenchFlow task directory for one HILBench instance."""
    task_format = validate_task_output_format(task_format)
    sanitized_id = _sanitize_name(task.task_id)
    task_dir = output_dir / sanitized_id
    if task_dir.exists():
        if not overwrite:
            ensure_existing_task_output_format(task_dir, task_format)
            logger.debug("Skipping existing task %s", task.task_id)
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
    oracle_dir = task_dir / oracle_dir_name(task_format)
    oracle_dir.mkdir()
    solve_sh = oracle_dir / "solve.sh"
    solve_sh.write_text(_render_solve_sh(task))
    solve_sh.chmod(0o755)
    (oracle_dir / "solve.patch").write_text(_gold_patch(task))

    # verifier/ for native task.md, tests/ for legacy Harbor/Pier layout.
    tests_dir = task_dir / verifier_dir_name(task_format)
    tests_dir.mkdir()

    test_sh = tests_dir / "test.sh"
    test_sh.write_text(_render_test_sh(task))
    test_sh.chmod(test_sh.stat().st_mode | stat.S_IEXEC)

    (tests_dir / "verify.py").write_text(VERIFY_PY)
    if task_format == "task-md":
        (tests_dir / "verifier.md").write_text(_render_verifier_md(task))
        rubrics_dir = tests_dir / "rubrics"
        rubrics_dir.mkdir()
        (rubrics_dir / "verifier.md").write_text(_render_verifier_rubric(task))

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
    task_format: TaskFormat = "task-md",
) -> list[Path]:
    """Generate BenchFlow task directories for HILBench SWE tasks."""
    task_format = validate_task_output_format(task_format)
    tasks = load_tasks_from_hf(task_type="swe")

    if task_ids:
        id_set = set(task_ids)
        tasks = [t for t in tasks if t.task_id in id_set]

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
    parser.add_argument(
        "--task-format",
        choices=TASK_FORMATS,
        default="task-md",
        help="Output layout: legacy task.toml/instruction.md or native task.md",
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
        task_format=args.task_format,
    )
    print(f"Generated {len(generated)} task directories in {args.output_dir}")


if __name__ == "__main__":
    main()
