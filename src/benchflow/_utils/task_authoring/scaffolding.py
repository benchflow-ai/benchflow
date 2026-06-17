"""Mutating task-authoring cluster: scaffold templates, init, and migration.

Writes new task directories (Dockerfile / verifier / oracle scaffolds), migrates
legacy ``task.toml`` + ``instruction.md`` pairs into ``task.md``, and normalizes
human-authored ``task.md`` documents into canonical machine form.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from benchflow.task.document import (
    TaskDocument,
    render_normalized_task_md,
    render_task_md_from_legacy,
)
from benchflow.task.imports import import_task_config_toml
from benchflow.task.paths import TaskPaths
from benchflow.task.verifier_document import VERIFIER_DOCUMENT_FILENAME

from ._evidence_paths import TASK_DOCUMENT_FILE


@dataclass(frozen=True)
class TaskMigrationResult:
    task_dir: Path
    task_md: Path
    removed_legacy: bool
    migrated_legacy_dirs: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskNormalizeResult:
    task_dir: Path
    task_md: Path
    normalized_text: str
    output_path: Path | None


@dataclass(frozen=True)
class ScaffoldResult:
    """A freshly scaffolded task plus every file the scaffold wrote.

    ``files`` are relative POSIX paths under ``task_dir``, sorted, derived from
    what actually landed on disk — so a ``Created:`` summary can list the real
    scaffold instead of a hand-maintained subset that drifts from it.
    """

    task_dir: Path
    files: list[str]


def init_task(
    name: str,
    parent_dir: Path = Path("tasks"),
    no_pytest: bool = False,
    no_oracle: bool = False,
    task_format: Literal["legacy", "task-md"] = "task-md",
    *,
    no_solution: bool | None = None,
) -> Path:
    """Scaffold a new task directory; return its path.

    Thin wrapper over :func:`scaffold_task` for callers that only need the
    directory. Use :func:`scaffold_task` when you also need the exact list of
    files written (e.g. to print an accurate ``Created:`` summary).
    """
    return scaffold_task(
        name,
        parent_dir=parent_dir,
        no_pytest=no_pytest,
        no_oracle=no_oracle,
        task_format=task_format,
        no_solution=no_solution,
    ).task_dir


def scaffold_task(
    name: str,
    parent_dir: Path = Path("tasks"),
    no_pytest: bool = False,
    no_oracle: bool = False,
    task_format: Literal["legacy", "task-md"] = "task-md",
    *,
    no_solution: bool | None = None,
) -> ScaffoldResult:
    """Scaffold a new task directory with standard structure.

    Returns the directory together with every file written (see
    :class:`ScaffoldResult`).
    """
    if no_solution is not None:
        no_oracle = no_solution
    if task_format not in ("legacy", "task-md"):
        raise ValueError("task_format must be 'legacy' or 'task-md'")
    # The name is a single directory segment under parent_dir, never a path. Reject
    # separators / '..' / leading dots / whitespace so `init "../escape"` can't write
    # outside parent_dir and names stay safe for HF/registry/shell tooling.
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or name != name.strip()
        or name.startswith(".")
        or any(c.isspace() for c in name)
    ):
        raise ValueError(
            "task name must be a single path segment "
            f"(no '/', '..', leading dot, or spaces); got {name!r}"
        )

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

    if task_format == "task-md":
        _write_task_md_verifier_package(task_dir, name)

    # oracle/ - placeholder MUST be replaced. The oracle solution should
    # cause the verifier in test.sh to write 1.0; the unedited scaffold
    # deliberately does not, so an init+check round-trip can't be mistaken
    # for a real benchmark (#360).
    if not no_oracle:
        sol_dir = task_dir / oracle_dirname
        sol_dir.mkdir()
        (sol_dir / "solve.sh").write_text(f"""#!/bin/bash
# Oracle solution - demonstrates the task is solvable.
# Used by: bench eval run --agent oracle --tasks-dir tasks/{name}

# [REPLACE: implement the oracle solution. It must satisfy the verifier in
# {verifier_dirname}/test.sh so that running solve.sh -> test.sh produces reward 1.0.]
echo "[REPLACE: implement oracle solution for {name}]" >&2
exit 1
""")
        (sol_dir / "solve.sh").chmod(0o755)

    written = sorted(
        path.relative_to(task_dir).as_posix()
        for path in task_dir.rglob("*")
        if path.is_file()
    )
    return ScaffoldResult(task_dir=task_dir, files=written)


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
    missing = [path.name for path in (task_toml, instruction_md) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot migrate task without legacy files: " + ", ".join(missing)
        )
    if task_md.exists() and not overwrite:
        raise FileExistsError(
            f"{task_md} already exists; pass --overwrite (CLI) / overwrite=True "
            "(API) to replace it"
        )

    legacy_config = import_task_config_toml(
        task_toml.read_text(), source="legacy"
    ).config
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
    migrated_legacy_dirs: tuple[str, ...] = ()
    if remove_legacy:
        task_toml.unlink()
        instruction_md.unlink()
        migrated_legacy_dirs = _promote_legacy_task_md_alias_dirs(task_dir)

    return TaskMigrationResult(
        task_dir=task_dir,
        task_md=task_md,
        removed_legacy=remove_legacy,
        migrated_legacy_dirs=migrated_legacy_dirs,
    )


def _promote_legacy_task_md_alias_dirs(task_dir: Path) -> tuple[str, ...]:
    """Adopt native directory names when removing split-format aliases."""

    migrated: list[str] = []
    for legacy_name, native_name in (
        (TaskPaths.LEGACY_TESTS_DIRNAME, TaskPaths.NATIVE_VERIFIER_DIRNAME),
        (TaskPaths.LEGACY_SOLUTION_DIRNAME, TaskPaths.NATIVE_ORACLE_DIRNAME),
    ):
        legacy_dir = task_dir / legacy_name
        native_dir = task_dir / native_name
        if not legacy_dir.is_dir() or native_dir.exists():
            continue
        legacy_dir.rename(native_dir)
        migrated.append(f"{legacy_name}/ -> {native_name}/")
    return tuple(migrated)


def normalize_task_md(
    task_dir: Path,
    *,
    output_path: Path | None = None,
    write: bool = False,
) -> TaskNormalizeResult:
    """Normalize a human-authored ``task.md`` into canonical machine form."""

    task_dir = Path(task_dir)
    task_md = task_dir / TASK_DOCUMENT_FILE
    if not task_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {task_dir}")
    if not task_md.is_file():
        raise FileNotFoundError(f"Missing task.md: {task_md}")
    if output_path is not None and write:
        raise ValueError("Use either output_path or write=True, not both")

    normalized = render_normalized_task_md(task_md.read_text(), path=task_md)
    destination = task_md if write else output_path
    if destination is not None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(normalized)

    return TaskNormalizeResult(
        task_dir=task_dir,
        task_md=task_md,
        normalized_text=normalized,
        output_path=destination,
    )


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


def _write_task_md_verifier_package(task_dir: Path, name: str) -> None:
    verifier_dir = task_dir / TaskPaths.NATIVE_VERIFIER_DIRNAME
    rubrics_dir = verifier_dir / "rubrics"
    rubrics_dir.mkdir()
    (verifier_dir / VERIFIER_DOCUMENT_FILENAME).write_text(f"""---
document_version: "0.3"
verifier:
  name: {name}
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_mean
    dimensions:
      task_success: {{weight: 1.0, source: deterministic}}
  outputs:
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/reward-details.json
    aggregate_policy:
      method: weighted_mean
      metrics:
        task_success: 1.0
---

## verifier intent

[REPLACE: describe what the verifier measures and which task outputs it reads.]
""")
    (rubrics_dir / "verifier.md").write_text(f"""# {name} Verifier Rubric

- `task_success`: [REPLACE: define the exact observable success condition the
  verifier checks.]
""")
    (rubrics_dir / "verifier.toml").write_text("""version = "0.1"

[[criteria]]
id = "task_success"
description = "[REPLACE: define the exact observable success condition the verifier checks.]"
weight = 1.0

[scoring]
method = "weighted_mean"
""")


def _write_task_md(task_dir: Path, name: str) -> None:
    (task_dir / TASK_DOCUMENT_FILE).write_text(f"""---
schema_version: "1.3"
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
---
# {name}

## prompt

[REPLACE: describe the goal, constraints, and expected outputs.
Be specific - list files to produce, commands to run, or behaviours to verify.]
""")
