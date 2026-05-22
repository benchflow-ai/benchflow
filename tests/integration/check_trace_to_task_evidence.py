#!/usr/bin/env python3
"""Generate and validate trace-to-task release evidence.

The release suite manifest declares trace-to-task as a release blocker. This
checker makes that lane executable:

- parse each declared trace source,
- generate BenchFlow task directories,
- run task authoring checks on every generated task,
- optionally run oracle Docker evals and validate reward artifacts, and
- write a machine-readable evidence summary.

Guards: ENG-93 trace-to-task e2e evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchflow._utils.task_authoring import check_task
from benchflow.traces.parsers import parse_claude_code_file, parse_opentraces_file
from benchflow.traces.task_gen import generate_tasks_from_traces

try:
    from tests.integration.run_suite import (
        DEFAULT_SUITE,
        SuiteError,
        load_suite,
        select_lanes,
    )
except ModuleNotFoundError:
    from run_suite import DEFAULT_SUITE, SuiteError, load_suite, select_lanes

LANE_ID = "trace-to-task-e2e"
SUMMARY_FILENAME = "trace-evidence.json"
LOG_TAIL_CHARS = 4000


def _source_output_name(source: dict[str, Any]) -> str:
    fmt = source.get("format")
    if fmt == "claude-code":
        return "jsonl-fixture"
    if fmt == "opentraces":
        return "opentraces-fixture"

    path = Path(str(source.get("path", "trace-source"))).stem
    slug = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")
    return slug or "trace-source"


def _load_source(source: dict[str, Any], repo_root: Path) -> list:
    path_value = source.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise SuiteError(f"{LANE_ID} source missing path: {source!r}")

    path = repo_root / path_value
    kind = source.get("kind")
    if kind == "jsonl_trace":
        fmt = source.get("format", "claude-code")
        if fmt == "claude-code":
            return parse_claude_code_file(path)
        if fmt == "opentraces":
            return parse_opentraces_file(path)
        raise SuiteError(f"unsupported jsonl_trace format: {fmt}")
    if kind == "opentraces_jsonl":
        return parse_opentraces_file(path)

    raise SuiteError(f"unsupported {LANE_ID} source kind: {kind}")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _latest_result_path(jobs_dir: Path, before: set[Path]) -> Path | None:
    result_paths = set(jobs_dir.rglob("result.json"))
    new_paths = result_paths - before
    if not new_paths:
        return None
    return max(new_paths, key=lambda path: path.stat().st_mtime)


def _log_tail(value: str) -> str:
    return value[-LOG_TAIL_CHARS:]


def _attach_failure_logs(record: dict[str, Any], completed: subprocess.CompletedProcess[str]) -> None:
    if completed.stdout:
        record["stdout_tail"] = _log_tail(completed.stdout)
    if completed.stderr:
        record["stderr_tail"] = _log_tail(completed.stderr)


def _run_oracle_eval(task_dir: Path, jobs_dir: Path, sandbox: str) -> dict[str, Any]:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    before = set(jobs_dir.rglob("result.json"))
    command = [
        "uv",
        "run",
        "bench",
        "eval",
        "create",
        "--tasks-dir",
        str(task_dir),
        "--agent",
        "oracle",
        "--sandbox",
        sandbox,
        "--jobs-dir",
        str(jobs_dir),
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
    )
    result_path = _latest_result_path(jobs_dir, before)
    record: dict[str, Any] = {
        "task_path": str(task_dir),
        "jobs_dir": str(jobs_dir),
        "command": command,
        "returncode": completed.returncode,
        "result_path": str(result_path) if result_path else None,
        "fresh_result": result_path is not None,
        "status": "fail",
    }
    if completed.returncode != 0:
        record["message"] = "bench eval create returned non-zero"
        _attach_failure_logs(record, completed)
        return record
    if result_path is None:
        record["message"] = "bench eval create did not produce a fresh result.json"
        _attach_failure_logs(record, completed)
        return record

    data = json.loads(result_path.read_text())
    reward = _number((data.get("rewards") or {}).get("reward"))
    record.update(
        {
            "task_name": data.get("task_name"),
            "agent": data.get("agent"),
            "reward": reward,
            "error": data.get("error"),
            "verifier_error": data.get("verifier_error"),
        }
    )
    if data.get("error"):
        record["message"] = f"run has error: {data['error']}"
        return record
    if data.get("verifier_error"):
        record["message"] = f"run has verifier_error: {data['verifier_error']}"
        return record
    if reward != 1.0:
        record["message"] = f"expected reward 1.0, got {reward!r}"
        return record

    record["status"] = "pass"
    record["message"] = "oracle eval reward=1"
    return record


def _generated_trace_task_set(suite: dict[str, Any]) -> dict[str, Any]:
    task_sets = suite["axes"].get("task_sets", {})
    task_set = task_sets.get("generated_trace_tasks")
    if not isinstance(task_set, dict):
        raise SuiteError("axes.task_sets.generated_trace_tasks must be a mapping")
    return task_set


def generate_trace_evidence(
    *,
    suite_path: Path,
    repo_root: Path,
    evidence_dir: Path | None,
    run_eval: bool,
    sandbox: str,
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    lane = select_lanes(suite, [LANE_ID])[0]
    task_set = _generated_trace_task_set(suite)

    if evidence_dir is None:
        configured = task_set.get("evidence_dir")
        if not isinstance(configured, str) or not configured:
            raise SuiteError("generated_trace_tasks missing evidence_dir")
        evidence_dir = repo_root / configured
    elif not evidence_dir.is_absolute():
        evidence_dir = repo_root / evidence_dir

    evidence_dir.mkdir(parents=True, exist_ok=True)
    sources = lane.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise SuiteError(f"{LANE_ID} lane must declare source files")

    summary: dict[str, Any] = {
        "suite": suite["suite"],
        "suite_version": suite["version"],
        "lane_id": LANE_ID,
        "evidence_dir": str(evidence_dir),
        "run_eval": run_eval,
        "sandbox": sandbox,
        "sources": [],
        "eval_runs": [],
        "status": "pass",
    }
    if run_eval:
        summary["oracle_run_id"] = uuid4().hex

    for source in sources:
        if not isinstance(source, dict):
            raise SuiteError(f"{LANE_ID} source must be a mapping")
        output_name = _source_output_name(source)
        output_dir = evidence_dir / output_name
        traces = _load_source(source, repo_root)
        generated = generate_tasks_from_traces(
            traces,
            output_dir,
            author="benchflow-dogfood",
            min_steps=1,
            overwrite=True,
        )
        generated_records = []
        for task_dir in generated:
            issues = check_task(task_dir)
            generated_records.append(
                {
                    "path": str(task_dir),
                    "check_status": "pass" if not issues else "fail",
                    "issues": issues,
                }
            )

        source_status = "pass"
        if not generated or any(record["issues"] for record in generated_records):
            source_status = "fail"

        source_record = {
            "source": source,
            "output_dir": str(output_dir),
            "trace_count": len(traces),
            "generated_count": len(generated),
            "generated_tasks": generated_records,
            "status": source_status,
        }
        summary["sources"].append(source_record)

        if run_eval and generated:
            jobs_dir = (
                evidence_dir / "oracle-runs" / summary["oracle_run_id"] / output_name
            )
            eval_record = _run_oracle_eval(generated[0], jobs_dir, sandbox)
            eval_record["source_output"] = output_name
            summary["eval_runs"].append(eval_record)

    if any(source["status"] != "pass" for source in summary["sources"]):
        summary["status"] = "fail"
    if run_eval and (
        not summary["eval_runs"]
        or any(run["status"] != "pass" for run in summary["eval_runs"])
    ):
        summary["status"] = "fail"

    summary_path = evidence_dir / SUMMARY_FILENAME
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and validate trace-to-task release evidence."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=DEFAULT_SUITE,
        help=f"Suite manifest path (default: {DEFAULT_SUITE})",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="BenchFlow repo root (default: current directory).",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="Override evidence directory from generated_trace_tasks.evidence_dir.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run oracle eval for one generated task per source class.",
    )
    parser.add_argument(
        "--sandbox",
        default="docker",
        help="Sandbox to use for oracle eval evidence (default: docker).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = generate_trace_evidence(
            suite_path=args.suite,
            repo_root=args.repo_root.resolve(),
            evidence_dir=args.evidence_dir,
            run_eval=args.run_eval,
            sandbox=args.sandbox,
        )
    except SuiteError as exc:
        print(f"ERROR: {exc}")
        return 2

    print("Trace-to-task release evidence")
    print("-" * 80)
    for source in summary["sources"]:
        source_path = source["source"].get("path", "<unknown>")
        print(
            f"{source_path:<55} {source['status'].upper():<5} "
            f"generated={source['generated_count']} traces={source['trace_count']}"
        )
        for task in source["generated_tasks"]:
            print(f"  {task['check_status'].upper():<5} {task['path']}")
    for run in summary["eval_runs"]:
        print(
            f"eval {run['source_output']:<40} {run['status'].upper():<5} "
            f"reward={run.get('reward')}"
        )
        print(f"  result: {run.get('result_path')}")
    print("-" * 80)
    print(f"summary: {summary['summary_path']}")

    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
