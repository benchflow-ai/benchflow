"""Generate BenchFlow task directories from ContinualLearningBench tasks.

ContinualLearningBench evaluates how well AI agents learn from past environment interactions
across sequential task instances.  Each ContinualLearningBench task becomes ONE BenchFlow
task — the agent runs the full sequential evaluation inside a Docker container
using the ContinualLearningBench harness.

Requires a local checkout of the ContinualLearningBench repo for schedule/variant metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
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
    validate_task_output_format,
    verifier_dir_name,
)

logger = logging.getLogger(__name__)
TASK_FORMATS = TASK_OUTPUT_FORMATS
TaskFormat = TaskOutputFormat

# Task definitions

_CONTINUALLEARNINGBENCH_TASKS: dict[str, dict] = {
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
        "setup_cmd": "cd /opt/continuallearningbench && python -m src.cli setup database_exploration",
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
class ContinualLearningBenchTaskInfo:
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
    continuallearningbench_dir: Path,
    task_ids: list[str] | None = None,
) -> list[ContinualLearningBenchTaskInfo]:
    """Load ContinualLearningBench task definitions."""
    tasks: list[ContinualLearningBenchTaskInfo] = []

    for task_id, meta in _CONTINUALLEARNINGBENCH_TASKS.items():
        if task_ids and task_id not in task_ids:
            continue

        schedule_json: dict = {}
        schedule_path = (
            continuallearningbench_dir
            / "src"
            / "tasks"
            / task_id
            / "schedules"
            / "default.json"
        )
        if schedule_path.exists():
            with open(schedule_path) as f:
                schedule_json = json.load(f)
                schedule_json.pop("_canary", None)

        tasks.append(
            ContinualLearningBenchTaskInfo(
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


def _render_task_toml(task: ContinualLearningBenchTaskInfo) -> str:
    sanitized = _sanitize_name(task.task_id)
    name = f"continuallearningbench/{sanitized}"
    tags_str = ", ".join(f'"{t}"' for t in task.tags)
    return f"""\
version = "1.0"

[task]
name = "{name}"

[metadata]
author_name = "Parth Asawa et al."
author_email = "continuallearningbench@continual-learning-bench.com"
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


def _render_task_md(task: ContinualLearningBenchTaskInfo) -> str:
    sanitized = _sanitize_name(task.task_id)
    instruction = _render_instruction(task).strip()
    frontmatter: dict[str, Any] = {
        "schema_version": "1.3",
        "task": {
            "name": f"continuallearningbench/{sanitized}",
        },
        "metadata": {
            "author_name": "Parth Asawa et al.",
            "author_email": "continuallearningbench@continual-learning-bench.com",
            "difficulty": task.difficulty,
            "category": task.category,
            "tags": task.tags,
        },
        "agent": {
            "timeout_sec": 3600.0,
        },
        "verifier": {
            "timeout_sec": 300.0,
        },
        "environment": {
            "build_timeout_sec": 600,
            "cpus": 2,
            "memory_mb": 4096,
            "storage_mb": 10240,
        },
        "benchflow": {
            "document_version": "0.3",
            "source": {
                "benchmark": "ContinualLearningBench",
                "task_id": task.task_id,
                "display_name": task.display_name,
                "num_instances": task.num_instances,
                "r_max": task.r_max,
                "response_schema": task.response_schema,
            },
            "oracle": {
                "evidence": "oracle/README.md",
                "static_solution": False,
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


_INSTRUCTION_TEMPLATE = Template("""\
# ${display_name}

## Overview

This is a **continual learning** challenge from ContinualLearningBench.
You must complete a sequence of related task instances and **learn from feedback**
to improve your performance over time.

${description}

## Task Details

- **Number of instances**: ${num_instances}
- **Response format**: `${response_schema}`
- **Reward metric**: ${reward_description} (higher is better)
- **Maximum per-instance reward**: ${r_max}

## How It Works

1. Prepare a sequence of responses for the ContinualLearningBench driver.
2. The driver replays those responses through the original ContinualLearningBench harness.
3. You can inspect the resulting score/summary, revise your responses, and rerun.
4. Your goal is to maximize cumulative reward across all instances.

## Important

- This is a sequential evaluation — instance order matters.
- You are expected to **learn and adapt** from past interactions.
- The evaluation measures both your raw performance AND your improvement over time.
- The key metric is **gain**: your reward minus a stateless baseline that cannot learn.

## Interaction

The task is mediated by the ContinualLearningBench driver at `/opt/run_task.py`.

1. Write one JSON object per planned response to `/opt/agent_responses.jsonl`,
   matching the response schema above.
2. Run `python /opt/run_task.py ${task_id}` from inside the sandbox.
3. Inspect the printed score/summary. You may edit `/opt/agent_responses.jsonl`
   and rerun the driver to improve the result.

The driver replays your responses through the ContinualLearningBench harness and writes
`/opt/results.json` for the verifier. Do not create or edit `/opt/results.json`
manually.
""")


def _render_instruction(task: ContinualLearningBenchTaskInfo) -> str:
    return _INSTRUCTION_TEMPLATE.safe_substitute(
        display_name=task.display_name,
        description=task.description,
        num_instances=str(task.num_instances),
        response_schema=task.response_schema,
        reward_description=task.reward_description,
        r_max=str(task.r_max),
        task_id=task.task_id,
    )


def _render_dockerfile(task: ContinualLearningBenchTaskInfo) -> str:
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

# Clone ContinualLearningBench and install with task-specific extras
RUN git clone https://github.com/pgasawa/continual-learning-bench /opt/continuallearningbench && \\
    cd /opt/continuallearningbench && \\
    pip install --no-cache-dir "{pip_spec}"
{extra_pip_lines}{setup_lines}
# Copy task-specific files
COPY run_task.py /opt/run_task.py
COPY schedule.json /opt/schedule.json

# Create verifier output directory
RUN mkdir -p /logs/verifier

# Agent-writable driver input/output files. The agent should edit
# agent_responses.jsonl and let run_task.py produce results.json.
RUN touch /opt/agent_responses.jsonl /opt/results.json && \\
    chmod 666 /opt/agent_responses.jsonl /opt/results.json

WORKDIR /opt/continuallearningbench
"""


_RUN_TASK_PY = """\
#!/usr/bin/env python3
\"\"\"Drive a ContinualLearningBench task via its Python API.

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

    # Import after continuallearningbench is installed
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


_EVALUATE_PY_TEMPLATE = Template("""\
#!/usr/bin/env python3
\"\"\"Evaluate ContinualLearningBench task results and write reward.\"\"\"
import json
import os

RESULTS_FILE = os.environ.get("BENCHFLOW_RESULTS_JSON", "/opt/results.json")
REWARD_FILE = os.environ.get("BENCHFLOW_REWARD_TEXT", "/logs/verifier/reward.txt")
R_MAX = ${r_max}


def main():
    try:
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        score = float(results.get("score", 0.0))
        reward = max(0.0, min(1.0, score / R_MAX)) if R_MAX > 0 else 0.0
    except Exception:
        reward = 0.0

    with open(REWARD_FILE, "w") as f:
        f.write(str(reward))

    print(f"Reward: {reward}")


if __name__ == "__main__":
    main()
""")


_TEST_SH = """\
#!/bin/bash
set -euo pipefail

verifier_log="${BENCHFLOW_VERIFIER_LOG:-/logs/verifier/verifier.log}"
mkdir -p "$(dirname "$verifier_log")"
exec > >(tee "$verifier_log") 2>&1

VERIFIER_DIR="${BENCHFLOW_VERIFIER_DIR:-/verifier}"
LEGACY_TESTS_DIR="${BENCHFLOW_LEGACY_TESTS_DIR:-/tests}"
if [ ! -f "$VERIFIER_DIR/evaluate.py" ] && [ -f "$LEGACY_TESTS_DIR/evaluate.py" ]; then
    VERIFIER_DIR="$LEGACY_TESTS_DIR"
fi

reward_file="${BENCHFLOW_REWARD_TEXT:-/logs/verifier/reward.txt}"
reward_json="${BENCHFLOW_REWARD_JSON:-/logs/verifier/reward.json}"
details_json="${BENCHFLOW_REWARD_DETAILS_JSON:-/logs/verifier/reward-details.json}"
mkdir -p "$(dirname "$reward_file")" "$(dirname "$reward_json")" "$(dirname "$details_json")"

BENCHFLOW_REWARD_TEXT="$reward_file" python3 "$VERIFIER_DIR/evaluate.py"

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
            "source": "continuallearningbench-results-json",
        },
        indent=2,
    )
    + "\\n"
)
PY
"""


def _render_verifier_md(task: ContinualLearningBenchTaskInfo) -> str:
    frontmatter: dict[str, Any] = {
        "document_version": "0.3",
        "verifier": {
            "name": f"continuallearningbench-{_sanitize_name(task.task_id)}-verifier",
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
                    "episode_score": {
                        "weight": 0.70,
                        "source": "deterministic",
                    },
                    "learning_gain": {
                        "weight": 0.30,
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
        "The deterministic verifier reads `/opt/results.json` produced by the "
        "ContinualLearningBench episode driver and normalizes score by the "
        "task-specific maximum reward.\n"
    )


def _render_verifier_rubric(task: ContinualLearningBenchTaskInfo) -> str:
    return f"""\
# ContinualLearningBench Rubric

Task: `continuallearningbench/{_sanitize_name(task.task_id)}`

- Episode score: `/opt/results.json` must contain the score produced by the
  original ContinualLearningBench harness.
- Learning gain: the task rewards improvement across ordered instances rather
  than a single static answer.
- Normalization: reward is clamped to `score / {task.r_max}`.
"""


def _render_oracle_readme(task: ContinualLearningBenchTaskInfo) -> str:
    return f"""\
# Oracle Evidence

ContinualLearningBench task `{task.task_id}` does not have a static file-level
oracle solution. The benchmark's ground truth is the original sequential
environment and its reward function.

Agents must interact with `/opt/run_task.py`, inspect feedback, and improve
responses across `{task.num_instances}` ordered instances. The verifier reads
`/opt/results.json` generated by that harness and normalizes the reported score
against `r_max = {task.r_max}`.
"""


def _ensure_existing_generated_task_current(
    task_dir: Path,
    task_format: TaskFormat,
) -> None:
    """Reject stale same-format task dirs that old converters can still satisfy."""
    verifier_dir = task_dir / verifier_dir_name(task_format)
    evaluate_py = verifier_dir / "evaluate.py"
    test_sh = verifier_dir / "test.sh"
    missing: list[str] = []

    if not evaluate_py.exists():
        missing.append(f"{evaluate_py.relative_to(task_dir)}")
    else:
        evaluate_text = evaluate_py.read_text()
        for marker in ("BENCHFLOW_RESULTS_JSON", "BENCHFLOW_REWARD_TEXT"):
            if marker not in evaluate_text:
                missing.append(f"{evaluate_py.relative_to(task_dir)}:{marker}")

    if not test_sh.exists():
        missing.append(f"{test_sh.relative_to(task_dir)}")
    else:
        test_text = test_sh.read_text()
        for marker in ("BENCHFLOW_VERIFIER_DIR", "BENCHFLOW_REWARD_JSON"):
            if marker not in test_text:
                missing.append(f"{test_sh.relative_to(task_dir)}:{marker}")

    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"{task_dir} already exists but was generated by an older "
            "ContinualLearningBench converter "
            f"({joined}); pass --overwrite or use a fresh output directory."
        )


def generate_task(
    task: ContinualLearningBenchTaskInfo,
    output_dir: Path,
    *,
    overwrite: bool = False,
    task_format: TaskFormat = "task-md",
) -> Path:
    """Generate a single BenchFlow task directory for one ContinualLearningBench task."""
    task_format = validate_task_output_format(task_format)
    sanitized = _sanitize_name(task.task_id)
    task_dir = output_dir / f"continuallearningbench-{sanitized}"

    if task_dir.exists():
        if not overwrite:
            ensure_existing_task_output_format(task_dir, task_format)
            _ensure_existing_generated_task_current(task_dir, task_format)
            logger.debug("Skipping existing task %s", task.task_id)
            return task_dir
        shutil.rmtree(task_dir)

    task_dir.mkdir(parents=True)

    if task_format == "task-md":
        (task_dir / "task.md").write_text(_render_task_md(task))
        oracle_dir = task_dir / "oracle"
        oracle_dir.mkdir()
        (oracle_dir / "README.md").write_text(_render_oracle_readme(task))
    else:
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

    # verifier/ for native task.md, tests/ for legacy Harbor/Pier layout.
    tests_dir = task_dir / verifier_dir_name(task_format)
    tests_dir.mkdir()
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(_TEST_SH)
    test_sh.chmod(0o755)
    (tests_dir / "evaluate.py").write_text(
        _EVALUATE_PY_TEMPLATE.safe_substitute(r_max=str(task.r_max))
    )
    if task_format == "task-md":
        (tests_dir / "verifier.md").write_text(_render_verifier_md(task))
        rubrics_dir = tests_dir / "rubrics"
        rubrics_dir.mkdir()
        (rubrics_dir / "verifier.md").write_text(_render_verifier_rubric(task))

    return task_dir


def generate_all(
    continuallearningbench_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    task_ids: list[str] | None = None,
    task_format: TaskFormat = "task-md",
) -> list[Path]:
    """Generate BenchFlow task directories for all ContinualLearningBench tasks."""
    task_format = validate_task_output_format(task_format)
    tasks = load_tasks(continuallearningbench_dir, task_ids=task_ids)

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
        logger.info("Generated %s -> %s", task.task_id, path.name)

    logger.info("Generated %d tasks in %s", len(generated), output_dir)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BenchFlow tasks from ContinualLearningBench"
    )
    parser.add_argument(
        "--continuallearningbench-dir",
        type=Path,
        required=True,
        help="Path to cloned ContinualLearningBench repo",
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
    parser.add_argument(
        "--task-format",
        choices=TASK_FORMATS,
        default="task-md",
        help="Output layout: legacy task.toml/instruction.md or native task.md",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    task_ids = args.task_ids.split(",") if args.task_ids else None

    generated = generate_all(
        continuallearningbench_dir=args.continuallearningbench_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        task_ids=task_ids,
        task_format=args.task_format,
    )

    print(f"\nGenerated {len(generated)} task(s) in {args.output_dir}")
    for p in generated:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
