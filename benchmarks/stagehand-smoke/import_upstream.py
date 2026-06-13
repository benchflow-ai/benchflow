#!/usr/bin/env python3
"""Import selected official Stagehand eval tasks into temp task dirs.

The importer reads Stagehand's TypeScript task modules and writes normalized
``stagehand-task.json`` descriptors to an operator-chosen output directory.
It is intentionally conservative: dynamic task shapes fail with structured
unsupported details instead of being silently mis-translated.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from benchflow.adapters.inbound import UnsupportedInboundTaskError
from benchflow.adapters.stagehand import official_task_descriptor_from_source

_TASK_ID_INVALID = re.compile(r"[^a-z0-9._-]+")


def import_tasks(
    *,
    stagehand_repo: Path,
    out_dir: Path,
    tasks: list[str],
    overwrite: bool = False,
    upstream_commit: str | None = None,
) -> tuple[list[Path], list[dict[str, Any]]]:
    commit = upstream_commit or _git_commit(stagehand_repo)
    written: list[Path] = []
    unsupported: list[dict[str, Any]] = []
    for task in tasks:
        source_file = _task_source_file(stagehand_repo=stagehand_repo, task=task)
        source = source_file.read_text()
        try:
            descriptor = official_task_descriptor_from_source(
                source,
                source_file=source_file,
                upstream_commit=commit,
            )
        except UnsupportedInboundTaskError as exc:
            unsupported.append(exc.report.to_dict())
            continue

        task_dir = out_dir / _task_slug(str(descriptor["task_id"]))
        _write_task_dir(task_dir, descriptor, overwrite=overwrite)
        written.append(task_dir)
    return written, unsupported


def discover_tasks(stagehand_repo: Path) -> list[str]:
    tasks_root = stagehand_repo / "packages" / "evals" / "tasks" / "bench"
    if not tasks_root.is_dir():
        raise FileNotFoundError(f"Stagehand bench tasks root not found: {tasks_root}")
    return sorted(
        path.relative_to(tasks_root).with_suffix("").as_posix()
        for path in tasks_root.rglob("*.ts")
    )


def build_support_report(
    *,
    stagehand_repo: Path,
    out_dir: Path,
    requested_tasks: list[str],
    written: list[Path],
    unsupported: list[dict[str, Any]],
    upstream_commit: str | None,
) -> dict[str, Any]:
    supported: list[dict[str, Any]] = []
    for task_dir in written:
        descriptor_path = task_dir / "stagehand-task.json"
        descriptor = json.loads(descriptor_path.read_text())
        supported.append(
            {
                "task_id": descriptor.get("task_id"),
                "task_dir": str(task_dir),
                "source_file": descriptor.get("source_file"),
                "success_check": descriptor.get("success_check"),
                "original_runner": descriptor.get("original_runner"),
            }
        )
    return {
        "schema": "benchflow.stagehand-import-support.v1",
        "benchmark": "stagehand-evals",
        "source_repo": str(stagehand_repo),
        "out_dir": str(out_dir),
        "upstream_commit": upstream_commit,
        "tasks_requested": requested_tasks,
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "supported": supported,
        "unsupported": unsupported,
    }


def _task_source_file(*, stagehand_repo: Path, task: str) -> Path:
    normalized = task.strip().removesuffix(".ts")
    if not normalized:
        raise ValueError("Stagehand task name cannot be empty")
    task_path = stagehand_repo / "packages" / "evals" / "tasks" / "bench"
    source_file = task_path / f"{normalized}.ts"
    if not source_file.is_file():
        raise FileNotFoundError(f"Stagehand task source not found: {source_file}")
    return source_file


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
    (task_dir / "stagehand-task.json").write_text(
        json.dumps(descriptor, indent=2) + "\n"
    )
    environment = task_dir / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "WORKDIR /app\n"
        "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /app\n"
    )


def _git_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _task_slug(task_id: str) -> str:
    slug = _TASK_ID_INVALID.sub("-", task_id.strip().lower()).strip("-._")
    if not slug:
        slug = "task"
    if not slug[0].isalnum():
        slug = f"task-{slug}"
    return slug[:80]


def _parse_tasks(raw: str, *, stagehand_repo: Path) -> list[str]:
    tasks: list[str] = []
    for item in raw.split(","):
        task = item.strip()
        if task == "all":
            tasks.extend(discover_tasks(stagehand_repo))
        elif task:
            tasks.append(task)
    if not tasks:
        raise ValueError("select at least one Stagehand task")
    return list(dict.fromkeys(tasks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stagehand-repo",
        type=Path,
        required=True,
        help="Clone of https://github.com/browserbase/stagehand.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        default="agent/sign_in",
        help=(
            "Comma-separated Stagehand bench task ids, e.g. agent/sign_in; "
            "use 'all' to inventory every official bench task."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--upstream-commit")
    parser.add_argument(
        "--support-report-out",
        type=Path,
        help="Write supported/unsupported task inventory JSON to this path.",
    )
    args = parser.parse_args()
    tasks = _parse_tasks(args.tasks, stagehand_repo=args.stagehand_repo)

    written, unsupported = import_tasks(
        stagehand_repo=args.stagehand_repo,
        out_dir=args.out_dir,
        tasks=tasks,
        overwrite=args.overwrite,
        upstream_commit=args.upstream_commit,
    )
    support_report = build_support_report(
        stagehand_repo=args.stagehand_repo,
        out_dir=args.out_dir,
        requested_tasks=tasks,
        written=written,
        unsupported=unsupported,
        upstream_commit=args.upstream_commit or _git_commit(args.stagehand_repo),
    )
    if args.support_report_out is not None:
        args.support_report_out.parent.mkdir(parents=True, exist_ok=True)
        args.support_report_out.write_text(json.dumps(support_report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "task_dirs": [str(path) for path in written],
                "unsupported": unsupported,
                "support_report": (
                    str(args.support_report_out)
                    if args.support_report_out is not None
                    else support_report
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
