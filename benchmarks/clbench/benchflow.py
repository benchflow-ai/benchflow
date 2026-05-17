"""Generate BenchFlow task directories from Continual Learning Bench tasks.

CLBench evaluates how well AI agents learn from past environment interactions
across sequential task instances.  Each CLBench task becomes ONE BenchFlow
task — the agent runs the full sequential evaluation inside a Docker container
using the CLBench harness.

Requires a local checkout of the CLBench repo for schedule/variant metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

logger = logging.getLogger(__name__)

# ── Task definitions ─────────────────────────────────────────────────

_CLBENCH_TASKS: dict[str, dict] = {
    "exploitable_poker": {
        "display_name": "Exploitable Poker",
        "difficulty": "medium",
        "category": "continual-learning",
        "tags": [
            "continual-learning",
            "sequential",
            "poker",
            "game-theory",
            "opponent-modeling",
        ],
        "num_instances": 120,
        "r_max": 9.4875,
        "response_schema": "PokerAction(thinking: str, action: str, amount: Optional[int])",
        "reward_description": "profit / big_blind per hand",
        "pip_extras": "poker",
        "extra_pip": "",
        "setup_cmd": "",
        "description": (
            "Play heads-up poker against exploitable opponents. "
            "Learn their strategies and adapt to maximize profit over time."
        ),
    },
    "database_exploration": {
        "display_name": "Database Exploration",
        "difficulty": "medium",
        "category": "continual-learning",
        "tags": [
            "continual-learning",
            "sequential",
            "database",
            "sql",
            "schema-learning",
        ],
        "num_instances": 30,
        "r_max": 1.0,
        "response_schema": "DatabaseAction(action: str, content: str)",
        "reward_description": "1 - (regret / max_queries_per_question)",
        "pip_extras": "database_exploration",
        "extra_pip": "",
        "setup_cmd": "cd /opt/clbench && python -m src.cli setup database_exploration",
        "description": (
            "Answer questions about an unknown SQLite database. "
            "Reduce exploratory queries over time as you learn the schema."
        ),
    },
    "cohort_studies": {
        "display_name": "Cohort Studies",
        "difficulty": "hard",
        "category": "continual-learning",
        "tags": [
            "continual-learning",
            "sequential",
            "medical",
            "statistics",
            "survival-analysis",
        ],
        "num_instances": 18,
        "r_max": 0.162202,
        "response_schema": "ToolCallResponse (discriminated union of tool calls)",
        "reward_description": "information gain in bits over flat baseline",
        "pip_extras": "cohort_studies",
        "extra_pip": "",
        "setup_cmd": "",
        "description": (
            "Analyse sequential clinical studies with different variable availability "
            "and coding conventions. Estimate survival for population cohorts by "
            "synthesising evidence across studies."
        ),
    },
}


def _sanitize_name(name: str) -> str:
    """Sanitize a task name: lowercase, hyphens for non-alphanumeric."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@dataclass
class CLBenchTaskInfo:
    task_id: str
    display_name: str
    difficulty: str
    category: str
    tags: list[str]
    num_instances: int
    r_max: float
    response_schema: str
    reward_description: str
    pip_extras: str
    extra_pip: str
    setup_cmd: str
    description: str
    schedule_json: dict = field(default_factory=dict)


def load_tasks(
    clbench_dir: Path,
    task_ids: list[str] | None = None,
) -> list[CLBenchTaskInfo]:
    """Load CLBench task definitions."""
    tasks: list[CLBenchTaskInfo] = []

    for task_id, meta in _CLBENCH_TASKS.items():
        if task_ids and task_id not in task_ids:
            continue

        schedule_json: dict = {}
        schedule_path = (
            clbench_dir / "src" / "tasks" / task_id / "schedules" / "default.json"
        )
        if schedule_path.exists():
            with open(schedule_path) as f:
                schedule_json = json.load(f)
                schedule_json.pop("_canary", None)

        tasks.append(
            CLBenchTaskInfo(
                task_id=task_id,
                display_name=meta["display_name"],
                difficulty=meta["difficulty"],
                category=meta["category"],
                tags=meta["tags"],
                num_instances=meta["num_instances"],
                r_max=meta["r_max"],
                response_schema=meta["response_schema"],
                reward_description=meta["reward_description"],
                pip_extras=meta["pip_extras"],
                extra_pip=meta["extra_pip"],
                setup_cmd=meta["setup_cmd"],
                description=meta["description"],
                schedule_json=schedule_json,
            )
        )

    return tasks


# ── Renderers ────────────────────────────────────────────────────────


def _render_task_toml(task: CLBenchTaskInfo) -> str:
    sanitized = _sanitize_name(task.task_id)
    name = f"clbench/{sanitized}"
    tags_str = ", ".join(f'"{t}"' for t in task.tags)
    return f"""\
version = "1.0"

[task]
name = "{name}"

[metadata]
author_name = "Parth Asawa et al."
author_email = "clbench@continual-learning-bench.com"
difficulty = "{task.difficulty}"
category = "{task.category}"
tags = [{tags_str}]

[agent]
timeout_sec = 3600.0

[verifier]
timeout_sec = 300.0

[environment]
build_timeout_sec = 600
cpus = 2
memory_mb = 4096
storage_mb = 10240
"""


_INSTRUCTION_TEMPLATE = Template("""\
# ${display_name}

## Overview

This is a **continual learning** challenge from the Continual Learning Bench.
You must complete a sequence of related task instances and **learn from feedback**
to improve your performance over time.

${description}

## Task Details

- **Number of instances**: ${num_instances}
- **Response format**: `${response_schema}`
- **Reward metric**: ${reward_description} (0-1 per instance, higher is better)
- **Maximum per-instance reward**: ${r_max}

## How It Works

1. You will receive a prompt for each instance in sequence.
2. After each response, you will receive feedback about your performance.
3. Use this feedback to improve on subsequent instances.
4. Your goal is to maximize cumulative reward across all instances.

## Important

- This is a sequential evaluation — instance order matters.
- You are expected to **learn and adapt** from past interactions.
- The evaluation measures both your raw performance AND your improvement over time.
- The key metric is **gain**: your reward minus a stateless baseline that cannot learn.

## Interaction

The task runs inside the CLBench harness. Respond to each prompt according to
the expected response schema. The harness will provide feedback after each step.
""")


def _render_instruction(task: CLBenchTaskInfo) -> str:
    return _INSTRUCTION_TEMPLATE.safe_substitute(
        display_name=task.display_name,
        description=task.description,
        num_instances=str(task.num_instances),
        response_schema=task.response_schema,
        reward_description=task.reward_description,
        r_max=str(task.r_max),
    )


def _render_dockerfile(task: CLBenchTaskInfo) -> str:
    setup_lines = ""
    if task.setup_cmd:
        setup_lines = f"\n# Task-specific setup\nRUN {task.setup_cmd}\n"

    extra_pip_lines = ""
    if task.extra_pip:
        extra_pip_lines = f"\nRUN pip install --no-cache-dir {task.extra_pip}\n"

    # Build pip install spec with task-specific extras
    pip_spec = "."
    if task.pip_extras:
        pip_spec = f".[{task.pip_extras}]"

    return f"""\
FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system deps
RUN apt-get update -qq && \\
    apt-get install -y -qq git curl jq && \\
    rm -rf /var/lib/apt/lists/*

# Clone CLBench and install with task-specific extras
RUN git clone https://github.com/pgasawa/continual-learning-bench /opt/clbench && \\
    cd /opt/clbench && \\
    pip install --no-cache-dir "{pip_spec}"
{extra_pip_lines}{setup_lines}
# Copy task-specific files
COPY run_task.py /opt/run_task.py
COPY schedule.json /opt/schedule.json

# Create verifier output directory
RUN mkdir -p /logs/verifier

WORKDIR /opt/clbench
"""


_RUN_TASK_PY = """\
#!/usr/bin/env python3
\"\"\"Drive a CLBench task via its Python API.

Reads agent responses from /opt/agent_responses.jsonl (one JSON per line).
Writes results to /opt/results.json when the task completes.
\"\"\"

import json
import sys
from pathlib import Path

RESPONSES_FILE = Path("/opt/agent_responses.jsonl")
RESULTS_FILE = Path("/opt/results.json")


def main() -> None:
    task_name = sys.argv[1] if len(sys.argv) > 1 else ""
    if not task_name:
        print("Usage: run_task.py <task_name>")
        sys.exit(1)

    # Import after clbench is installed
    from src.registry import get_task_class

    task_cls = get_task_class(task_name)
    task = task_cls()

    query = task.reset()
    outcomes = []

    if RESPONSES_FILE.exists():
        with open(RESPONSES_FILE) as f:
            responses = [json.loads(line) for line in f if line.strip()]
    else:
        responses = []

    for resp_data in responses:
        from pydantic import BaseModel

        schema_cls = query.response_schema
        action = schema_cls.model_validate(resp_data)

        from src.interface import Response

        response = Response(action=action)
        step_result = task.step(response)

        if step_result.instance_outcome:
            outcomes.append(
                {
                    "instance_id": step_result.instance_outcome.instance_id,
                    "instance_index": step_result.instance_outcome.instance_index,
                    "reward": step_result.instance_outcome.reward,
                    "success": step_result.instance_outcome.success,
                }
            )

        if step_result.done:
            break

        if step_result.next_query:
            query = step_result.next_query

    result = task.evaluate()
    output = {
        "score": result.score,
        "summary": result.summary,
        "metrics": result.metrics,
        "instance_outcomes": outcomes,
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2, default=str))
    print(f"Task complete. Score: {result.score:.4f}")


if __name__ == "__main__":
    main()
"""


_EVALUATE_PY = """\
#!/usr/bin/env python3
\"\"\"Evaluate CLBench task results and write reward.\"\"\"
import json
import sys

RESULTS_FILE = "/opt/results.json"
REWARD_FILE = "/logs/verifier/reward.txt"


def main():
    try:
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        reward = results.get("score", 0.0)
        reward = max(0.0, min(1.0, float(reward)))
    except Exception:
        reward = 0.0

    with open(REWARD_FILE, "w") as f:
        f.write(str(reward))

    print(f"Reward: {reward}")


if __name__ == "__main__":
    main()
"""


_TEST_SH = """\
#!/bin/bash
set -e
python3 /tests/evaluate.py
"""


def generate_task(
    task: CLBenchTaskInfo,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Generate a single BenchFlow task directory for one CLBench task."""
    sanitized = _sanitize_name(task.task_id)
    task_dir = output_dir / f"clbench-{sanitized}"

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

    # environment/
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(_render_dockerfile(task))
    (env_dir / "run_task.py").write_text(_RUN_TASK_PY)
    (env_dir / "schedule.json").write_text(
        json.dumps(task.schedule_json, indent=2) if task.schedule_json else "{}"
    )

    # tests/
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(_TEST_SH)
    test_sh.chmod(0o755)
    (tests_dir / "evaluate.py").write_text(_EVALUATE_PY)

    return task_dir


def generate_all(
    clbench_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    task_ids: list[str] | None = None,
) -> list[Path]:
    """Generate BenchFlow task directories for all CLBench tasks."""
    tasks = load_tasks(clbench_dir, task_ids=task_ids)

    if limit is not None:
        tasks = tasks[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for task in tasks:
        path = generate_task(task, output_dir, overwrite=overwrite)
        generated.append(path)
        logger.info("Generated %s -> %s", task.task_id, path.name)

    logger.info("Generated %d tasks in %s", len(generated), output_dir)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BenchFlow tasks from Continual Learning Bench"
    )
    parser.add_argument(
        "--clbench-dir",
        type=Path,
        required=True,
        help="Path to cloned CLBench repo",
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
        help="Cap number of tasks to generate",
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
        help="Comma-separated list of task IDs to generate",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    task_ids = args.task_ids.split(",") if args.task_ids else None

    generated = generate_all(
        clbench_dir=args.clbench_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        task_ids=task_ids,
    )

    print(f"\nGenerated {len(generated)} task(s) in {args.output_dir}")
    for p in generated:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
