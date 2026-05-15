"""Task authoring helpers — init and check benchmark tasks."""

import logging
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_FILES = ["task.toml", "instruction.md"]
REQUIRED_DIRS = ["environment"]


def check_task(task_dir: Path) -> list[str]:
    """Validate a task directory structure. Returns list of issues."""

    issues = []
    if not task_dir.is_dir():
        return [f"Not a directory: {task_dir}"]

    for filename in REQUIRED_FILES:
        if not (task_dir / filename).exists():
            issues.append(f"Missing required file: {filename}")

    for dirname in REQUIRED_DIRS:
        if not (task_dir / dirname).is_dir():
            issues.append(f"Missing required directory: {dirname}/")

    toml_path = task_dir / "task.toml"
    if toml_path.exists():
        try:
            with open(toml_path, "rb") as f:
                config = tomllib.load(f)
            if "agent" not in config:
                issues.append("task.toml missing [agent] section")
            elif "timeout_sec" not in config.get("agent", {}):
                issues.append("task.toml [agent] missing timeout_sec")
        except Exception as e:
            issues.append(f"task.toml parse error: {e}")

    instruction = task_dir / "instruction.md"
    if instruction.exists() and instruction.stat().st_size == 0:
        issues.append("instruction.md is empty")

    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        issues.append("Missing environment/Dockerfile")

    tests_dir = task_dir / "tests"
    if tests_dir.is_dir():
        if not any(tests_dir.iterdir()):
            issues.append("tests/ directory is empty")
    else:
        issues.append("Missing tests/ directory (verifier needs test.sh or evaluate.py)")

    return issues


def init_task(
    name: str,
    parent_dir: Path = Path("tasks"),
    no_pytest: bool = False,
    no_solution: bool = False,
) -> Path:
    """Scaffold a new task directory with the standard BenchFlow shape."""

    task_dir = parent_dir / name
    if task_dir.exists():
        raise FileExistsError(f"Task directory already exists: {task_dir}")

    task_dir.mkdir(parents=True)
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

<!-- Write clear, specific instructions for the agent. -->
<!-- Describe the goal, constraints, and expected output. -->

""")

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

    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("""#!/bin/bash
# Verifier script — exit 0 for pass, non-zero for fail.
# Write reward to /logs/verifier/reward.txt (float 0.0-1.0).

echo "1.0" > /logs/verifier/reward.txt
""")
    (tests_dir / "test.sh").chmod(0o755)

    if not no_pytest:
        (tests_dir / "test_outputs.py").write_text(
            '"""Pytest-based verifier. Run after agent completes."""\n\n'
            "def test_placeholder():\n"
            "    # Replace with actual verification logic\n"
            "    assert True\n"
        )

    if not no_solution:
        solution_dir = task_dir / "solution"
        solution_dir.mkdir()
        (solution_dir / "solve.sh").write_text("""#!/bin/bash
# Oracle solution — demonstrates the task is solvable.
# Used by: benchflow run -a oracle -t tasks/{name}

echo "TODO: implement oracle solution"
""")
        (solution_dir / "solve.sh").chmod(0o755)

    return task_dir
