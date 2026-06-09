"""Generate BenchFlow task directories from OpaqueToolsBench BFCL instances.

OpaqueToolsBench studies whether LLM agents can recover the meaning of
opacified tools — tools whose names, descriptions, and parameters have
been deliberately obscured.  The BFCL (Berkeley Function Calling
Leaderboard) environment tests general function calling: given a natural
language query and tool definitions, the agent must produce the correct
function call(s).

This module generates one BenchFlow task directory per BFCL test item
from the ``executable_simple`` and ``executable_multiple_function``
categories shipped in the OpaqueToolsBench repo.

Requires a local checkout of the OpaqueToolsBench repo for the
``tool_configs/`` JSON files.

Usage:
    python benchmarks/opaquetoolsbench/benchflow.py \\
        --opaquetoolsbench-dir ~/OpaqueToolsBench \\
        --output-dir /tmp/opaquetoolsbench-tasks

    python benchmarks/opaquetoolsbench/benchflow.py \\
        --opaquetoolsbench-dir ~/OpaqueToolsBench \\
        --output-dir /tmp/opaquetoolsbench-tasks \\
        --limit 10
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
    oracle_dir_name,
    validate_task_output_format,
    verifier_dir_name,
)

logger = logging.getLogger(__name__)

# Categories we convert

BFCL_CATEGORIES = [
    "executable_simple",
    "executable_multiple_function",
]

# Timeout presets

_AGENT_TIMEOUT = 600  # 10 min — single function-call tasks are fast
_VERIFIER_TIMEOUT = 120  # 2 min — evaluation is lightweight
TASK_FORMATS = TASK_OUTPUT_FORMATS
TaskFormat = TaskOutputFormat


def _sanitize_name(raw: str) -> str:
    """Lowercase, replace non-alphanumeric with hyphens, collapse runs."""
    name = raw.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-")


@dataclass
class BFCLTask:
    """One BFCL test item ready for BenchFlow conversion."""

    test_id: int
    category: str
    question: str
    tools: list[dict]
    ground_truth: list[str]
    name_mapping: dict[str, str]
    execution_result_type: list[str]
    execution_result: list | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def instance_id(self) -> str:
        return f"{_sanitize_name(self.category)}-{self.test_id}"

    @property
    def task_name(self) -> str:
        return f"opaquetoolsbench/{self.instance_id}"

    @property
    def n_tools(self) -> int:
        return len(self.tools)

    @property
    def n_ground_truth(self) -> int:
        return len(self.ground_truth)


def load_tasks(opaquetoolsbench_dir: Path) -> list[BFCLTask]:
    """Load all BFCL base-config tasks from OpaqueToolsBench repo."""
    configs_dir = opaquetoolsbench_dir / "src" / "datasets" / "bfcl" / "tool_configs"
    if not configs_dir.exists():
        raise FileNotFoundError(
            f"BFCL tool_configs not found at {configs_dir}. "
            "Clone OpaqueToolsBench and pass --opaquetoolsbench-dir."
        )

    tasks: list[BFCLTask] = []
    for category in BFCL_CATEGORIES:
        config_file = configs_dir / f"{category}_base_config.json"
        if not config_file.exists():
            logger.warning("Config not found: %s", config_file)
            continue

        config = json.loads(config_file.read_text())
        for test in config.get("tests", []):
            tasks.append(
                BFCLTask(
                    test_id=test["test_id"],
                    category=category,
                    question=test["question"],
                    tools=test.get("tools", []),
                    ground_truth=test.get("ground_truth", []),
                    name_mapping=test.get("name_mapping", {}),
                    execution_result_type=test.get(
                        "execution_result_type", ["exact_match"]
                    ),
                    execution_result=test.get("execution_result"),
                    metadata=test.get("metadata", {}),
                )
            )

    return tasks


def _render_task_toml(task: BFCLTask) -> str:
    tag_list = ", ".join(
        f'"{t}"'
        for t in [
            "function-calling",
            _sanitize_name(task.category),
        ]
    )
    return f"""\
version = "1.0"

[task]
name = "{task.task_name}"

[metadata]
author_name = "OpaqueToolsBench (Hallinan et al.)"
difficulty = "easy"
category = "function-calling"
tags = [{tag_list}]

[agent]
timeout_sec = {_AGENT_TIMEOUT}

[verifier]
timeout_sec = {_VERIFIER_TIMEOUT}

[environment]
cpus = 1
memory_mb = 1024
storage_mb = 2048
allow_internet = false
"""


def _render_instruction(task: BFCLTask) -> str:
    """Generate instruction.md for the agent."""
    lines = [
        "# Function Calling Task",
        "",
        "## Query",
        "",
        task.question,
        "",
        "## Available Functions",
        "",
    ]

    for tool in task.tools:
        lines.append(f"### `{tool['name']}`")
        desc = tool.get("description", "")
        if desc:
            lines.append(f"\n{desc}\n")
        params = tool.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))
        if props:
            lines.append("**Parameters:**\n")
            for pname, pdef in props.items():
                ptype = pdef.get("type", "any")
                pdesc = pdef.get("description", "")
                req = " *(required)*" if pname in required else ""
                lines.append(f"- `{pname}` ({ptype}){req}: {pdesc}")
            lines.append("")

    lines.extend(
        [
            "## Instructions",
            "",
            "Determine the correct function call(s) to answer the query above.",
            "Write your response as a JSON array of function call objects to "
            "`/app/output/response.json`.",
            "",
            "Each function call object must have this format:",
            "```json",
            "{",
            '  "function": "<function_name>",',
            '  "args": {',
            '    "<param1>": <value1>,',
            '    "<param2>": <value2>',
            "  }",
            "}",
            "```",
            "",
            "Example (array with one call):",
            "```json",
            "[",
            '  {"function": "my_func", "args": {"x": 42, "y": "hello"}}',
            "]",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def _render_task_md(task: BFCLTask) -> str:
    tag = _sanitize_name(task.category)
    instruction = _render_instruction(task).strip()
    frontmatter: dict[str, Any] = {
        "schema_version": "1.3",
        "task": {
            "name": task.task_name,
        },
        "metadata": {
            "author_name": "OpaqueToolsBench (Hallinan et al.)",
            "difficulty": "easy",
            "category": "function-calling",
            "tags": ["function-calling", tag],
        },
        "agent": {
            "timeout_sec": _AGENT_TIMEOUT,
        },
        "verifier": {
            "timeout_sec": _VERIFIER_TIMEOUT,
        },
        "environment": {
            "cpus": 1,
            "memory_mb": 1024,
            "storage_mb": 2048,
            "allow_internet": False,
        },
        "benchflow": {
            "document_version": "0.3",
            "source": {
                "benchmark": "OpaqueToolsBench",
                "category": task.category,
                "test_id": task.test_id,
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


def _render_dockerfile() -> str:
    """Generate Dockerfile for the evaluation environment.

    Test files (test.sh, evaluate.py, ground_truth.json) are uploaded
    at runtime by BenchFlow/Harbor — they are NOT copied during the
    Docker build. The build context is ``environment/`` which does not
    include the sibling ``tests/`` directory.
    """
    return """\
# Pinned by digest for reproducibility.
FROM python:3.13-slim@sha256:dc1546eefcbe8caaa1f004f16ab76b204b5e1dbd58ff81b899f21cd40541232f

WORKDIR /app

RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /app/output /verifier /tests
"""


def _render_test_sh(task: BFCLTask) -> str:
    return Template("""\
#!/bin/bash
# Verifier for OpaqueToolsBench BFCL task: $instance_id
set -euo pipefail

verifier_log="${BENCHFLOW_VERIFIER_LOG:-/logs/verifier/verifier.log}"
mkdir -p "$(dirname "$verifier_log")"
exec > >(tee "$verifier_log") 2>&1

VERIFIER_DIR="${BENCHFLOW_VERIFIER_DIR:-/verifier}"
LEGACY_TESTS_DIR="${BENCHFLOW_LEGACY_TESTS_DIR:-/tests}"
if [ ! -f "$VERIFIER_DIR/evaluate.py" ] && [ -f "$LEGACY_TESTS_DIR/evaluate.py" ]; then
    VERIFIER_DIR="$LEGACY_TESTS_DIR"
fi

response_path="${BENCHFLOW_RESPONSE_PATH:-/app/output/response.json}"
reward_file="${BENCHFLOW_REWARD_TEXT:-/logs/verifier/reward.txt}"
reward_json="${BENCHFLOW_REWARD_JSON:-/logs/verifier/reward.json}"
details_json="${BENCHFLOW_REWARD_DETAILS_JSON:-/logs/verifier/reward-details.json}"
mkdir -p "$(dirname "$reward_file")" "$(dirname "$reward_json")" "$(dirname "$details_json")"

python3 "$VERIFIER_DIR/evaluate.py" \\
    --response "$response_path" \\
    --ground-truth "$VERIFIER_DIR/ground_truth.json" \\
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
            "matched_all_ground_truth_calls": reward >= 1.0,
        },
        indent=2,
    )
    + "\\n"
)
PY
""").safe_substitute(instance_id=task.instance_id)


def _render_solution_sh(task: BFCLTask) -> str:
    ground_truth = json.dumps(task.ground_truth)
    return Template("""\
#!/bin/bash
# Oracle solution for OpaqueToolsBench BFCL task: $instance_id
set -euo pipefail

mkdir -p /app/output
python3 - <<'PY'
from __future__ import annotations

import ast
import json
from pathlib import Path

ground_truth = $ground_truth


def parse_python_call(call_str: str) -> dict:
    tree = ast.parse(call_str)
    if not isinstance(tree.body[0], ast.Expr):
        raise ValueError(f"Expected expression: {call_str}")
    node = tree.body[0].value
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"Expected function call: {call_str}")

    args: dict = {}
    for kw in node.keywords:
        if kw.arg:
            args[kw.arg] = ast.literal_eval(kw.value)
    for i, arg in enumerate(node.args):
        args[f"_positional_{i}"] = ast.literal_eval(arg)

    return {"function": node.func.id, "args": args}


calls = [parse_python_call(call) for call in ground_truth]
Path("/app/output/response.json").write_text(json.dumps(calls, indent=2))
PY
""").safe_substitute(
        instance_id=task.instance_id,
        ground_truth=ground_truth,
    )


def _render_verifier_md(task: BFCLTask) -> str:
    frontmatter: dict[str, Any] = {
        "document_version": "0.3",
        "verifier": {
            "name": f"opaquetoolsbench-{task.instance_id}-verifier",
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
                    "function_name": {
                        "weight": 0.35,
                        "source": "deterministic",
                    },
                    "arguments": {
                        "weight": 0.45,
                        "source": "deterministic",
                    },
                    "call_count": {
                        "weight": 0.20,
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
        "The deterministic script compares the submitted function-call JSON "
        "against the OpaqueToolsBench BFCL ground truth for this instance.\n"
    )


def _render_verifier_rubric(task: BFCLTask) -> str:
    return f"""\
# OpaqueToolsBench BFCL Rubric

Task: `{task.task_name}`

- Function name: every required function call must use the expected function.
- Arguments: every ground-truth argument must be present and value-equivalent.
- Call count: all expected calls must be matched without reusing a response.

The bundled verifier awards `1.0` only when all ground-truth calls match, and
`0.0` otherwise.
"""


# evaluate.py (copied into every task's tests/)

EVALUATE_PY = '''\
"""OpaqueToolsBench BFCL verifier for BenchFlow.

Compares the agent's function call response against ground truth using
AST-based matching. Writes a binary reward (1.0 or 0.0) to the reward
file.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


def parse_python_call(call_str: str) -> tuple[str, dict] | None:
    """Parse a Python function call string into (name, kwargs)."""
    try:
        tree = ast.parse(call_str)
        if not isinstance(tree.body[0], ast.Expr):
            return None
        node = tree.body[0].value
        if not isinstance(node, ast.Call):
            return None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        else:
            return None

        kwargs: dict = {}
        for kw in node.keywords:
            if kw.arg:
                try:
                    val = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    val = ast.unparse(kw.value)
                kwargs[kw.arg] = val

        for i, arg in enumerate(node.args):
            try:
                val = ast.literal_eval(arg)
            except (ValueError, SyntaxError):
                val = ast.unparse(arg)
            kwargs[f"_positional_{i}"] = val

        return func_name, kwargs
    except (SyntaxError, AttributeError, TypeError, IndexError):
        return None


def parse_call(data):
    """Parse a function call from JSON dict or Python call string."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return parse_python_call(data)

    if isinstance(data, dict):
        name = data.get("function", "")
        args = data.get("args", {})
        if isinstance(args, dict):
            return name, args
    return None


def values_match(v1, v2, tolerance=0.05):
    """Compare two values with tolerance for floats."""
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        if v2 == 0:
            return v1 == 0
        return abs(v1 - v2) <= abs(v2) * tolerance
    if isinstance(v1, str) and isinstance(v2, str):
        return v1.strip().lower() == v2.strip().lower()
    if isinstance(v1, list) and isinstance(v2, list):
        if len(v1) != len(v2):
            return False
        return all(values_match(a, b, tolerance) for a, b in zip(v1, v2))
    if isinstance(v1, dict) and isinstance(v2, dict):
        if set(v1.keys()) != set(v2.keys()):
            return False
        return all(values_match(v1[k], v2[k], tolerance) for k in v1)
    return v1 == v2


def call_matches(model_call, gt_call_str):
    """Check if a model call matches a ground-truth call string."""
    model_parsed = parse_call(model_call)
    gt_parsed = parse_python_call(gt_call_str)

    if model_parsed is None or gt_parsed is None:
        return False

    m_name, m_args = model_parsed
    g_name, g_args = gt_parsed

    if m_name != g_name:
        return False

    # Check all ground-truth params are present and correct
    for key, g_val in g_args.items():
        if key not in m_args:
            return False
        if not values_match(m_args[key], g_val):
            return False

    return True


def evaluate(response_path: Path, gt_path: Path) -> float:
    """Evaluate agent response against ground truth. Returns 1.0 or 0.0."""
    gt_data = json.loads(gt_path.read_text())
    gt_calls = gt_data.get("ground_truth", [])

    if not response_path.exists():
        print("No response file found at", response_path)
        return 0.0

    try:
        response = json.loads(response_path.read_text())
    except json.JSONDecodeError:
        print("Invalid JSON in response file")
        return 0.0

    if not isinstance(response, list):
        response = [response]

    if len(response) < len(gt_calls):
        print(
            f"Not enough function calls: got {len(response)}, "
            f"expected {len(gt_calls)}"
        )
        return 0.0

    # Each ground truth call must be matched by exactly one model call
    matched = [False] * len(gt_calls)
    used = [False] * len(response)

    for gi, gt_call in enumerate(gt_calls):
        for ri, resp_call in enumerate(response):
            if used[ri]:
                continue
            if call_matches(resp_call, gt_call):
                matched[gi] = True
                used[ri] = True
                break

    if all(matched):
        print(f"All {len(gt_calls)} ground-truth call(s) matched.")
        return 1.0
    else:
        n_matched = sum(matched)
        print(
            f"Matched {n_matched}/{len(gt_calls)} ground-truth call(s)."
        )
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpaqueToolsBench BFCL verifier"
    )
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--reward-file", required=True, type=Path)
    args = parser.parse_args()

    reward_file: Path = args.reward_file
    reward_file.parent.mkdir(parents=True, exist_ok=True)

    reward = evaluate(args.response, args.ground_truth)
    reward_file.write_text(f"{reward:.6f}")
    print(f"Reward: {reward:.4f}")


if __name__ == "__main__":
    main()
'''


def generate_task(
    task: BFCLTask,
    output_dir: Path,
    *,
    overwrite: bool = False,
    task_format: TaskFormat = "task-md",
) -> Path:
    """Generate a single BenchFlow task directory for one BFCL test item."""
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
    (env_dir / "Dockerfile").write_text(_render_dockerfile())

    # verifier/ for native task.md, tests/ for legacy Harbor/Pier layout.
    tests_dir = task_dir / verifier_dir_name(task_format)
    tests_dir.mkdir()

    test_sh = tests_dir / "test.sh"
    test_sh.write_text(_render_test_sh(task))
    test_sh.chmod(0o755)

    (tests_dir / "evaluate.py").write_text(EVALUATE_PY)
    if task_format == "task-md":
        (tests_dir / "verifier.md").write_text(_render_verifier_md(task))
        rubrics_dir = tests_dir / "rubrics"
        rubrics_dir.mkdir()
        (rubrics_dir / "verifier.md").write_text(_render_verifier_rubric(task))

    # Ground truth JSON
    gt_data = {
        "ground_truth": task.ground_truth,
        "execution_result_type": task.execution_result_type,
    }
    if task.execution_result is not None:
        gt_data["execution_result"] = task.execution_result
    (tests_dir / "ground_truth.json").write_text(json.dumps(gt_data, indent=2))

    # oracle/ for native task.md, solution/ for legacy Harbor/Pier layout.
    solution_dir = task_dir / oracle_dir_name(task_format)
    solution_dir.mkdir()
    solve_sh = solution_dir / "solve.sh"
    solve_sh.write_text(_render_solution_sh(task))
    solve_sh.chmod(0o755)

    return task_dir


def generate_all(
    opaquetoolsbench_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    task_ids: list[str] | None = None,
    task_format: TaskFormat = "task-md",
) -> list[Path]:
    """Generate BenchFlow task directories for all BFCL tasks."""
    task_format = validate_task_output_format(task_format)
    tasks = load_tasks(opaquetoolsbench_dir)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BenchFlow tasks from OpaqueToolsBench BFCL"
    )
    parser.add_argument(
        "--opaquetoolsbench-dir",
        type=Path,
        required=True,
        help="Path to OpaqueToolsBench repo checkout",
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

    tid_list = args.task_ids.split(",") if args.task_ids else None
    generated = generate_all(
        args.opaquetoolsbench_dir,
        args.output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        task_ids=tid_list,
        task_format=args.task_format,
    )
    print(f"Generated {len(generated)} tasks in {args.output_dir}")


if __name__ == "__main__":
    main()
