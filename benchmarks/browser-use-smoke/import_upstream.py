#!/usr/bin/env python3
"""Import selected official Browser Use benchmark tasks into temp task dirs.

The official Browser Use benchmark ships encrypted task suites. This script
mirrors the public loader, decrypts only in memory, and writes selected tasks to
an operator-chosen output directory so BenchFlow can run them without committing
plaintext benchmark tasks to the repo.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from benchflow.adapters.browser_use import (
    load_encrypted_benchmark_tasks,
    official_task_descriptor,
)

_TASK_ID_INVALID = re.compile(r"[^a-z0-9._-]+")


def import_tasks(
    *,
    encrypted_file: Path,
    out_dir: Path,
    benchmark: str = "BU_Bench_V1",
    task_indices: list[int],
    interleave: bool = True,
    overwrite: bool = False,
    judge_model: str = "gemini-2.5-flash",
    judge_env_key: str = "GEMINI_API_KEY",
) -> list[Path]:
    tasks = load_encrypted_benchmark_tasks(
        encrypted_file,
        benchmark=benchmark,
        interleave=interleave,
    )
    selected = _selected_indices(total=len(tasks), task_indices=task_indices)
    written: list[Path] = []
    for index in selected:
        descriptor = official_task_descriptor(
            tasks[index],
            benchmark=benchmark,
            task_index=index,
            judge_model=judge_model,
            judge_env_key=judge_env_key,
        )
        task_dir = out_dir / f"{index:03d}-{_task_slug(descriptor['task_id'])}"
        _write_task_dir(task_dir, descriptor, overwrite=overwrite)
        written.append(task_dir)
    return written


def _selected_indices(*, total: int, task_indices: list[int]) -> list[int]:
    if not task_indices:
        raise ValueError("select at least one task index")
    bad = [index for index in task_indices if index < 0 or index >= total]
    if bad:
        raise ValueError(f"task index out of range for {total} tasks: {bad}")
    return task_indices


def _write_task_dir(
    task_dir: Path,
    descriptor: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if task_dir.exists():
        if not overwrite:
            raise FileExistsError(f"task dir already exists: {task_dir}")
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    (task_dir / "browser-use-task.json").write_text(
        json.dumps(descriptor, indent=2) + "\n"
    )

    environment = task_dir / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "WORKDIR /app\n"
        "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /app\n"
    )

    tests = task_dir / "tests"
    tests.mkdir()
    (tests / "rubric.toml").write_text(_rubric_toml(descriptor))
    (tests / "verifier.md").write_text(_verifier_md(descriptor))
    (tests / "context.md").write_text(_judge_context(descriptor))


def _rubric_toml(descriptor: dict[str, Any]) -> str:
    task = str(descriptor["confirmed_task"]).strip()
    answer = descriptor.get("answer")
    ground_truth = (
        "No explicit ground truth is provided; judge against the task request "
        "and the browser trace artifact."
        if answer is None
        else str(answer)
    )
    return f"""\
[judge]
model = "{descriptor["verifier"]["judge"]["model"]}"
files = ["browser-use-smoke-trace.json"]

[scoring]
aggregation = "weighted_mean"

[[criterion]]
name = "browser_use_task_success"
type = "binary"
description = \"\"\"
Evaluate the Browser Use or Stagehand artifact for the official Browser Use
benchmark task below.

Task:
{task}

Ground truth:
{ground_truth}

Pass only if the artifact's final_result, agent steps, and any screenshots show
that the agent actually completed the requested browser task. Fail if the
artifact is empty, off-task, only describes intentions, reports an error, or
does not answer the user request.
\"\"\"
"""


def _judge_context(descriptor: dict[str, Any]) -> str:
    answer = descriptor.get("answer")
    ground_truth = "not provided" if answer is None else str(answer)
    return (
        "# Browser Use Benchmark Task\n\n"
        f"Benchmark: {descriptor['benchmark']}\n"
        f"Task ID: {descriptor['task_id']}\n"
        f"Category: {descriptor.get('category', 'unknown')}\n\n"
        "## Task\n\n"
        f"{descriptor['confirmed_task']}\n\n"
        "## Ground Truth\n\n"
        f"{ground_truth}\n"
    )


def _verifier_md(descriptor: dict[str, Any]) -> str:
    judge = descriptor["verifier"]["judge"]
    return f"""\
---
verifier:
  default_strategy: browser_use_judge
  strategies:
    browser_use_judge:
      type: llm-judge
      rubric: rubric.toml
      model: {judge["model"]}
      input_dir: {judge["input_dir"]}
      context_file: context.md
  outputs:
    reward_json: /logs/verifier/reward.json
---
"""


def _task_slug(task_id: object) -> str:
    slug = _TASK_ID_INVALID.sub("-", str(task_id).strip().lower()).strip("-._")
    if not slug:
        slug = "task"
    if not slug[0].isalnum():
        slug = f"task-{slug}"
    return slug[:80]


def _parse_indices(raw: str) -> list[int]:
    indices: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(int(part))
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        help="Clone of https://github.com/browser-use/benchmark.",
    )
    parser.add_argument("--encrypted-file", type=Path)
    parser.add_argument("--benchmark", default="BU_Bench_V1")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task-indices", default="0")
    parser.add_argument("--raw-order", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--judge-model", default="gemini-2.5-flash")
    parser.add_argument("--judge-env-key", default="GEMINI_API_KEY")
    args = parser.parse_args()

    encrypted_file = args.encrypted_file
    if encrypted_file is None:
        if args.upstream_repo is None:
            raise SystemExit("provide --encrypted-file or --upstream-repo")
        encrypted_file = args.upstream_repo / f"{args.benchmark}.enc"

    written = import_tasks(
        encrypted_file=encrypted_file,
        out_dir=args.out_dir,
        benchmark=args.benchmark,
        task_indices=_parse_indices(args.task_indices),
        interleave=not args.raw_order,
        overwrite=args.overwrite,
        judge_model=args.judge_model,
        judge_env_key=args.judge_env_key,
    )
    print(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "tasks_dir": str(args.out_dir),
                "task_dirs": [str(path) for path in written],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
