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
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

logger = logging.getLogger(__name__)

# ── Categories we convert ────────────────────────────────────────────

BFCL_CATEGORIES = [
    "executable_simple",
    "executable_multiple_function",
]

# ── Timeout presets ──────────────────────────────────────────────────

_AGENT_TIMEOUT = 600  # 10 min — single function-call tasks are fast
_VERIFIER_TIMEOUT = 120  # 2 min — evaluation is lightweight


def _sanitize_name(raw: str) -> str:
    """Lowercase, replace non-alphanumeric with hyphens, collapse runs."""
    name = raw.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-")


# ── Data classes ─────────────────────────────────────────────────────


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


# ── Loader ───────────────────────────────────────────────────────────


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


# ── Renderers ────────────────────────────────────────────────────────


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
            '{',
            '  "function": "<function_name>",',
            '  "args": {',
            '    "<param1>": <value1>,',
            '    "<param2>": <value2>',
            '  }',
            '}',
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

RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /app/output
"""


def _render_test_sh(task: BFCLTask) -> str:
    return Template("""\
#!/bin/bash
# Verifier for OpaqueToolsBench BFCL task: $instance_id
set -o pipefail

exec > >(tee /logs/verifier/verifier.log) 2>&1

python3 /tests/evaluate.py \\
    --response /app/output/response.json \\
    --ground-truth /tests/ground_truth.json \\
    --reward-file /logs/verifier/reward.txt
""").safe_substitute(instance_id=task.instance_id)


# ── evaluate.py (copied into every task's tests/) ───────────────────

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


# ── Task generation ─────────────────────────────────────────────────


def generate_task(
    task: BFCLTask, output_dir: Path, *, overwrite: bool = False
) -> Path:
    """Generate a single BenchFlow task directory for one BFCL test item."""
    task_dir = output_dir / task.instance_id
    if task_dir.exists():
        if not overwrite:
            logger.debug("Skipping existing task %s", task.instance_id)
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
    (env_dir / "Dockerfile").write_text(_render_dockerfile())

    # tests/
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()

    test_sh = tests_dir / "test.sh"
    test_sh.write_text(_render_test_sh(task))
    test_sh.chmod(0o755)

    (tests_dir / "evaluate.py").write_text(EVALUATE_PY)

    # Ground truth JSON
    gt_data = {
        "ground_truth": task.ground_truth,
        "execution_result_type": task.execution_result_type,
    }
    if task.execution_result is not None:
        gt_data["execution_result"] = task.execution_result
    (tests_dir / "ground_truth.json").write_text(json.dumps(gt_data, indent=2))

    return task_dir


def generate_all(
    opaquetoolsbench_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    task_ids: list[str] | None = None,
) -> list[Path]:
    """Generate BenchFlow task directories for all BFCL tasks."""
    tasks = load_tasks(opaquetoolsbench_dir)

    if task_ids:
        id_set = set(task_ids)
        tasks = [t for t in tasks if t.instance_id in id_set]

    if limit is not None:
        tasks = tasks[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for task in tasks:
        path = generate_task(task, output_dir, overwrite=overwrite)
        generated.append(path)
        logger.info("Generated %s", task.instance_id)

    logger.info("Generated %d tasks in %s", len(generated), output_dir)
    return generated


# ── CLI ──────────────────────────────────────────────────────────────


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
    )
    print(f"Generated {len(generated)} tasks in {args.output_dir}")


if __name__ == "__main__":
    main()
