"""Task authoring — init and check benchmark tasks."""

import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from benchflow.task.config import TaskConfig
from benchflow.task.document import (
    TaskDocument,
    TaskDocumentParseError,
    render_task_md_from_legacy,
)
from benchflow.task.paths import TaskPaths

logger = logging.getLogger(__name__)

LEGACY_REQUIRED_FILES = ["task.toml", "instruction.md"]
TASK_DOCUMENT_FILE = "task.md"
REQUIRED_DIRS = ["environment"]
OPTIONAL_FILES = ["environment/Dockerfile"]
OPTIONAL_DIRS = ["verifier", "oracle", "tests", "solution"]

# Placeholder marker written by init_task — must be replaced before the task
# is considered authored. Catching this in check_task prevents a freshly
# scaffolded task from being mistaken for a real benchmark (#360).
_PLACEHOLDER_MARKER = "[REPLACE:"


@dataclass(frozen=True)
class TaskMigrationResult:
    task_dir: Path
    task_md: Path
    removed_legacy: bool


def check_task(task_dir: Path) -> list[str]:
    """Validate a task directory structure. Returns list of issues (empty = valid)."""
    issues = []
    if not task_dir.is_dir():
        return [f"Not a directory: {task_dir}"]

    task_md = task_dir / TASK_DOCUMENT_FILE
    has_task_md = task_md.exists()

    if has_task_md:
        issues.extend(_check_task_document(task_md))
    else:
        for f in LEGACY_REQUIRED_FILES:
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
    if toml_path.exists() and not has_task_md:
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

    paths = TaskPaths(task_dir)

    # Check verifier code. Native tasks use verifier/; tests/ remains the
    # legacy alias for existing Harbor-style task packages.
    verifier_dir = paths.tests_dir
    verifier_label = _logical_dir_label(paths, kind="verifier")
    if verifier_dir.is_dir():
        if not any(verifier_dir.iterdir()):
            issues.append(f"{verifier_label}/ directory is empty")
    else:
        issues.append(
            "Missing verifier/ directory (or legacy tests/; verifier needs "
            "test.sh or evaluate.py)"
        )

    # Check CTRF output path consistency (ENG-153)
    test_sh = paths.test_path
    if test_sh.exists():
        issues.extend(_check_ctrf_path(test_sh))
        if _PLACEHOLDER_MARKER in test_sh.read_text():
            issues.append(
                f"{verifier_label}/test.sh contains unreplaced placeholder — "
                "write real verifier logic before running the task"
            )

    # Detect placeholder oracle scripts that have not been replaced (#360).
    for solve_sh in (paths.oracle_dir / "solve.sh", paths.legacy_solution_dir / "solve.sh"):
        if solve_sh.exists() and _PLACEHOLDER_MARKER in solve_sh.read_text():
            label = "oracle" if solve_sh.parent == paths.oracle_dir else "solution"
            issues.append(
                f"{label}/solve.sh contains unreplaced placeholder — "
                "write a real oracle solution before running the task"
            )

    return issues


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


def _logical_dir_label(paths: TaskPaths, *, kind: Literal["verifier", "oracle"]) -> str:
    if kind == "verifier":
        return (
            TaskPaths.NATIVE_VERIFIER_DIRNAME
            if paths.uses_native_verifier_dir
            else TaskPaths.LEGACY_TESTS_DIRNAME
        )
    return (
        TaskPaths.NATIVE_ORACLE_DIRNAME
        if paths.uses_native_oracle_dir
        else TaskPaths.LEGACY_SOLUTION_DIRNAME
    )


def init_task(
    name: str,
    parent_dir: Path = Path("tasks"),
    no_pytest: bool = False,
    no_oracle: bool = False,
    task_format: Literal["legacy", "task-md"] = "task-md",
    *,
    no_solution: bool | None = None,
) -> Path:
    """Scaffold a new task directory with standard structure."""
    if no_solution is not None:
        no_oracle = no_solution
    if task_format not in ("legacy", "task-md"):
        raise ValueError("task_format must be 'legacy' or 'task-md'")

    task_dir = parent_dir / name
    if task_dir.exists():
        raise FileExistsError(f"Task directory already exists: {task_dir}")

    task_dir.mkdir(parents=True)

    if task_format == "task-md":
        _write_task_md(task_dir, name)
    else:
        _write_legacy_task_files(task_dir, name)

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

    verifier_dirname = (
        TaskPaths.LEGACY_TESTS_DIRNAME
        if task_format == "legacy"
        else TaskPaths.NATIVE_VERIFIER_DIRNAME
    )
    oracle_dirname = (
        TaskPaths.LEGACY_SOLUTION_DIRNAME
        if task_format == "legacy"
        else TaskPaths.NATIVE_ORACLE_DIRNAME
    )

    # verifier/
    tests_dir = task_dir / verifier_dirname
    tests_dir.mkdir()
    # Verifier defaults to FAILURE (0.0) until the author replaces the
    # placeholder. A scaffold that auto-passes would silently inflate eval
    # results - see #360.
    (tests_dir / "test.sh").write_text("""#!/bin/bash
# Verifier script - write reward to /logs/verifier/reward.txt (float 0.0-1.0).
# Exit 0 after writing it; nonzero exit means verifier infrastructure failure.

# [REPLACE: write real verification logic here. The scaffold defaults to 0.0
# so an unedited task cannot accidentally count as a passing benchmark.]
echo "[REPLACE: write real verifier logic] - defaulting to failure" >&2
echo "0.0" > /logs/verifier/reward.txt
""")
    (tests_dir / "test.sh").chmod(0o755)

    if not no_pytest:
        # Fails by default until the author writes real assertions (#360).
        (
            tests_dir / "test_outputs.py"
        ).write_text("""\"\"\"Pytest-based verifier. Run by BenchFlow after agent completes.\"\"\"

import pytest


def test_placeholder():
    # [REPLACE: write real verification logic. Until then this test fails so
    # a scaffolded task cannot accidentally count as passing.]
    pytest.fail("[REPLACE: write real verifier assertions in this file]")
""")

    # oracle/ - placeholder MUST be replaced. The oracle solution should
    # cause the verifier in test.sh to write 1.0; the unedited scaffold
    # deliberately does not, so an init+check round-trip can't be mistaken
    # for a real benchmark (#360).
    if not no_oracle:
        sol_dir = task_dir / oracle_dirname
        sol_dir.mkdir()
        (sol_dir / "solve.sh").write_text(f"""#!/bin/bash
# Oracle solution - demonstrates the task is solvable.
# Used by: bench eval create --agent oracle --tasks-dir tasks/{name}

# [REPLACE: implement the oracle solution. It must satisfy the verifier in
# {verifier_dirname}/test.sh so that running solve.sh -> test.sh produces reward 1.0.]
echo "[REPLACE: implement oracle solution for {name}]" >&2
exit 1
""")
        (sol_dir / "solve.sh").chmod(0o755)

    return task_dir


def migrate_task_to_task_md(
    task_dir: Path,
    *,
    overwrite: bool = False,
    remove_legacy: bool = False,
) -> TaskMigrationResult:
    """Convert a legacy task.toml + instruction.md pair into task.md.

    The migration is intentionally non-destructive by default: authors can
    inspect the generated document before deleting the legacy pair. Config
    equivalence is checked before writing so migration cannot silently lose
    supported task configuration.
    """

    task_dir = Path(task_dir)
    task_md = task_dir / TASK_DOCUMENT_FILE
    task_toml = task_dir / "task.toml"
    instruction_md = task_dir / "instruction.md"

    if not task_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {task_dir}")
    missing = [
        path.name for path in (task_toml, instruction_md) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot migrate task without legacy files: " + ", ".join(missing)
        )
    if task_md.exists() and not overwrite:
        raise FileExistsError(
            f"{task_md} already exists; pass overwrite=True to replace it"
        )

    legacy_config = TaskConfig.model_validate_toml(task_toml.read_text())
    rendered = render_task_md_from_legacy(task_dir)
    document = TaskDocument.from_text(rendered, path=task_md)
    if document.config.model_dump() != legacy_config.model_dump():
        raise ValueError(
            "Generated task.md does not preserve task.toml config semantics"
        )
    if document.instruction != instruction_md.read_text().strip():
        raise ValueError(
            "Generated task.md does not preserve instruction.md prompt text"
        )

    task_md.write_text(rendered)
    if remove_legacy:
        task_toml.unlink()
        instruction_md.unlink()

    return TaskMigrationResult(
        task_dir=task_dir,
        task_md=task_md,
        removed_legacy=remove_legacy,
    )


def _check_task_document(task_md: Path) -> list[str]:
    issues: list[str] = []
    try:
        document = TaskDocument.from_path(task_md)
    except TaskDocumentParseError as e:
        return [f"task.md parse error: {e}"]
    except Exception as e:
        return [f"task.md parse error: {e}"]

    text = task_md.read_text()
    if not document.instruction.strip():
        issues.append("task.md prompt is empty")
    if _PLACEHOLDER_MARKER in text:
        issues.append(
            "task.md contains unreplaced placeholder - replace "
            "task prompts, role prompts, and simulated-user notes"
        )
    return issues


def _write_legacy_task_files(task_dir: Path, name: str) -> None:
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

    (task_dir / "instruction.md").write_text(f"""# {name}

[REPLACE: one-sentence summary of what the agent must do.]

## Goal

[REPLACE: describe the goal, constraints, and expected outputs.
Be specific — list files to produce, commands to run, or behaviours to verify.]

## Success criteria

[REPLACE: list the conditions the verifier in tests/test.sh checks for.]
""")


def _write_task_md(task_dir: Path, name: str) -> None:
    (task_dir / TASK_DOCUMENT_FILE).write_text(f"""---
version: "1.0"
metadata:
  author_name: ""
  difficulty: medium
  category: capability
  tags: []
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
environment:
  cpus: 1
  memory_mb: 2048
agents:
  roles:
    solver:
      agent: claude-agent-acp
scenes:
  - name: solve
    turns:
      - role: solver
---
# {name}

## prompt

[REPLACE: describe the goal, constraints, and expected outputs.
Be specific - list files to produce, commands to run, or behaviours to verify.]

## role:solver

[REPLACE: optional solver-specific guidance. Delete this section if the task
does not need per-role prompting.]

## user-persona

[REPLACE: optional NudgeBench-style simulated-user persona. Delete this section
if the task does not need a user.]
""")
