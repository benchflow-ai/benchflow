"""Task authoring — init, check, and digest benchmark tasks."""

import hashlib
import logging
import os
import re
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_FILES = ["task.toml", "instruction.md"]
REQUIRED_DIRS = ["environment"]
OPTIONAL_FILES = ["environment/Dockerfile"]
OPTIONAL_DIRS = ["tests", "solution"]

# Placeholder marker written by init_task — must be replaced before the task
# is considered authored. Catching this in check_task prevents a freshly
# scaffolded task from being mistaken for a real benchmark (#360).
_PLACEHOLDER_MARKER = "[REPLACE:"


def check_task(task_dir: Path) -> list[str]:
    """Validate a task directory structure. Returns list of issues (empty = valid)."""
    issues = []
    if not task_dir.is_dir():
        return [f"Not a directory: {task_dir}"]

    for f in REQUIRED_FILES:
        if not (task_dir / f).exists():
            issues.append(f"Missing required file: {f}")

    for d in REQUIRED_DIRS:
        if not (task_dir / d).is_dir():
            issues.append(f"Missing required directory: {d}/")

    # Validate task.toml
    # Note: [agent] and [agent].timeout_sec are optional at runtime
    # (AgentConfig defaults to timeout_sec=None → no wall-clock cap). We
    # only surface parse errors here so `bench tasks check` and
    # `bench eval create` agree on what a "valid" task looks like.
    # See #379.
    toml_path = task_dir / "task.toml"
    if toml_path.exists():
        try:
            with open(toml_path, "rb") as f:
                tomllib.load(f)
        except Exception as e:
            issues.append(f"task.toml parse error: {e}")

    # Check instruction.md is non-empty and has no placeholder markers
    instr = task_dir / "instruction.md"
    if instr.exists():
        if instr.stat().st_size == 0:
            issues.append("instruction.md is empty")
        elif _PLACEHOLDER_MARKER in instr.read_text():
            issues.append(
                f"instruction.md contains unreplaced placeholder "
                f"('{_PLACEHOLDER_MARKER} ...' markers) — replace them with "
                f"real task instructions"
            )

    # Check Dockerfile exists
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        issues.append("Missing environment/Dockerfile")

    # Check tests
    tests_dir = task_dir / "tests"
    if tests_dir.is_dir():
        if not any(tests_dir.iterdir()):
            issues.append("tests/ directory is empty")
    else:
        issues.append(
            "Missing tests/ directory (verifier needs test.sh or evaluate.py)"
        )

    # Check CTRF output path consistency (ENG-153)
    test_sh = task_dir / "tests" / "test.sh"
    if test_sh.exists():
        issues.extend(_check_ctrf_path(test_sh))
        if _PLACEHOLDER_MARKER in test_sh.read_text():
            issues.append(
                "tests/test.sh contains unreplaced placeholder — "
                "write real verifier logic before running the task"
            )

    # Detect placeholder solution that has not been replaced (#360).
    solve_sh = task_dir / "solution" / "solve.sh"
    if solve_sh.exists() and _PLACEHOLDER_MARKER in solve_sh.read_text():
        issues.append(
            "solution/solve.sh contains unreplaced placeholder — "
            "write a real oracle solution before running the task"
        )

    return issues


def task_digest(task_dir: Path) -> str:
    """Content digest pinning a task's files, independent of git.

    sha256 over every regular file under ``task_dir``, sorted by POSIX
    relative path; each file contributes
    ``path_utf8 + b"\\x00" + sha256(file_bytes).digest()``. Symlinks and
    file modes are excluded, so the digest is reproducible from a plain
    checkout. Must byte-match the reference digests in the skillsbench
    dataset registry (``registry.json`` / ``docs/dataset-versioning.md``,
    skillsbench PR #922).
    """
    if not task_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {task_dir}")
    files: list[tuple[str, Path]] = []
    # os.walk never descends into symlinked directories (unlike pre-3.13
    # Path.rglob), keeping "symlinks are excluded" true for whole subtrees.
    for dirpath, _dirnames, filenames in os.walk(task_dir):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.is_symlink() or not path.is_file():
                continue
            files.append((path.relative_to(task_dir).as_posix(), path))
    digest = hashlib.sha256()
    for rel, path in sorted(files):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return f"sha256:{digest.hexdigest()}"


_CTRF_STANDARD_PATH = "/logs/verifier/ctrf.json"


def _check_ctrf_path(test_sh: Path) -> list[str]:
    """Warn when test.sh uses --ctrf with a non-standard output path."""
    try:
        text = test_sh.read_text()
    except OSError:
        return []
    uncommented = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    match = re.search(r"--ctrf[= ]([^\s\\]+)", uncommented)
    if not match:
        return []
    path_arg = match.group(1).strip('"').strip("'")
    if path_arg.startswith("$"):
        return []
    if path_arg != _CTRF_STANDARD_PATH:
        return [
            f"test.sh uses non-standard CTRF path '{path_arg}' "
            f"(expected '{_CTRF_STANDARD_PATH}')"
        ]
    return []


def init_task(
    name: str,
    parent_dir: Path = Path("tasks"),
    no_pytest: bool = False,
    no_solution: bool = False,
) -> Path:
    """Scaffold a new task directory with standard structure."""
    task_dir = parent_dir / name
    if task_dir.exists():
        raise FileExistsError(f"Task directory already exists: {task_dir}")

    task_dir.mkdir(parents=True)

    # task.toml
    (task_dir / "task.toml").write_text("""version = "1.0"

[metadata]
author_name = ""
difficulty = "medium"
category = "capability"
tags = []

[agent]
timeout_sec = 300

[verifier]
timeout_sec = 120

[environment]
cpus = 1
memory_mb = 2048
""")

    # instruction.md — placeholders MUST be replaced (#360). Leaving them in
    # place is a `bench tasks check` failure so a scaffolded task cannot be
    # mistaken for an authored benchmark.
    (task_dir / "instruction.md").write_text(f"""# {name}

[REPLACE: one-sentence summary of what the agent must do.]

## Goal

[REPLACE: describe the goal, constraints, and expected outputs.
Be specific — list files to produce, commands to run, or behaviours to verify.]

## Success criteria

[REPLACE: list the conditions the verifier in tests/test.sh checks for.]
""")

    # environment/
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("""FROM ubuntu:24.04

# Install dependencies
RUN apt-get update -qq && apt-get install -y -qq curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Log directories
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
""")

    # tests/
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    # Verifier defaults to FAILURE (0.0) until the author replaces the
    # placeholder. A scaffold that auto-passes would silently inflate eval
    # results — see #360.
    (tests_dir / "test.sh").write_text("""#!/bin/bash
# Verifier script — write reward to /logs/verifier/reward.txt (float 0.0-1.0).
# Exit 0 after writing it; nonzero exit means verifier infrastructure failure.

# [REPLACE: write real verification logic here. The scaffold defaults to 0.0
# so an unedited task cannot accidentally count as a passing benchmark.]
echo "[REPLACE: write real verifier logic] — defaulting to failure" >&2
echo "0.0" > /logs/verifier/reward.txt
""")
    (tests_dir / "test.sh").chmod(0o755)

    if not no_pytest:
        # Fails by default until the author writes real assertions (#360).
        (
            tests_dir / "test_outputs.py"
        ).write_text("""\"\"\"Pytest-based verifier. Run by Harbor after agent completes.\"\"\"

import pytest


def test_placeholder():
    # [REPLACE: write real verification logic. Until then this test fails so
    # a scaffolded task cannot accidentally count as passing.]
    pytest.fail("[REPLACE: write real verifier assertions in tests/test_outputs.py]")
""")

    # solution/ — placeholder MUST be replaced. The oracle solution should
    # cause the verifier in tests/test.sh to write 1.0; the unedited scaffold
    # deliberately does not, so an init+check round-trip can't be mistaken
    # for a real benchmark (#360).
    if not no_solution:
        sol_dir = task_dir / "solution"
        sol_dir.mkdir()
        (sol_dir / "solve.sh").write_text(f"""#!/bin/bash
# Oracle solution — demonstrates the task is solvable.
# Used by: bench eval create --agent oracle --tasks-dir tasks/{name}

# [REPLACE: implement the oracle solution. It must satisfy the verifier in
# tests/test.sh so that running solve.sh → test.sh produces reward 1.0.]
echo "[REPLACE: implement oracle solution for {name}]" >&2
exit 1
""")
        (sol_dir / "solve.sh").chmod(0o755)

    return task_dir
