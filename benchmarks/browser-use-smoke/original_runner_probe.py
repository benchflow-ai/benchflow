#!/usr/bin/env python3
"""Probe the official Browser Use benchmark runner and write scrubbed evidence.

This runner intentionally does not read Browser Use's raw run_data trace files:
those can contain decrypted task text, ground truth, model output, and
screenshots. It only inspects the public summary and per-task result records
that the upstream runner already writes for local verification.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

SCHEMA = "benchflow.browser-use-original-runner-probe.v1"
_SUMMARY_RE = re.compile(r"^Summary:\s*(?P<path>.+?)\s*$", re.MULTILINE)
_TRACE_RE = re.compile(r"^Trace artifacts:\s*(?P<path>.+?)\s*$", re.MULTILINE)
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"dtn_[A-Za-z0-9_-]{12,}"),
    re.compile(r"AQ\.[A-Za-z0-9_.-]{12,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
)


def probe_original_runner(
    *,
    upstream_repo: Path,
    task_indices: list[int],
    benchmark: str = "BU_Bench_V1",
    framework: str = "browser-use",
    browser: str = "local_headless",
    model: str = "gemini-2.5-flash",
    parallel: int = 1,
    task_timeout_sec: int = 1800,
    runner_timeout_sec: int = 2100,
    no_interleave: bool = False,
) -> dict[str, Any]:
    """Run the official framework runner and classify its artifact outcome."""

    if not task_indices:
        raise ValueError("select at least one task index")
    runner = upstream_repo / "run_framework_eval.py"
    if not runner.is_file():
        return _blocked_report(
            upstream_repo=upstream_repo,
            command=[],
            benchmark=benchmark,
            task_indices=task_indices,
            framework=framework,
            browser=browser,
            model=model,
            parallel=parallel,
            task_timeout_sec=task_timeout_sec,
            runner_timeout_sec=runner_timeout_sec,
            elapsed_sec=0.0,
            failure_class="missing-upstream-framework-runner",
            reason=f"missing runner: {runner}",
        )

    command = [
        "uv",
        "run",
        "python",
        "run_framework_eval.py",
        "--benchmark",
        benchmark,
        "--framework",
        framework,
        "--browser",
        browser,
        "--model",
        model,
        "--task-indices",
        ",".join(str(index) for index in task_indices),
        "--parallel",
        str(parallel),
        "--task-timeout",
        str(task_timeout_sec),
    ]
    if no_interleave:
        command.append("--no-interleave")

    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=upstream_repo,
            text=True,
            capture_output=True,
            timeout=runner_timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        return _blocked_report(
            upstream_repo=upstream_repo,
            command=command,
            benchmark=benchmark,
            task_indices=task_indices,
            framework=framework,
            browser=browser,
            model=model,
            parallel=parallel,
            task_timeout_sec=task_timeout_sec,
            runner_timeout_sec=runner_timeout_sec,
            elapsed_sec=elapsed,
            failure_class="original-runner-process-timeout",
            reason=f"runner exceeded {runner_timeout_sec}s wall timeout",
            stdout=exc.stdout,
            stderr=exc.stderr,
        )

    elapsed = time.monotonic() - started
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    summary_path = _printed_path(stdout, _SUMMARY_RE, upstream_repo=upstream_repo)
    trace_dir = _printed_path(stdout, _TRACE_RE, upstream_repo=upstream_repo)
    summary_entry = _latest_summary_entry(summary_path)
    task_results = _summary_task_results(summary_entry)
    task_result_files = _task_result_files(trace_dir)
    raw_trace_files = _raw_trace_files(trace_dir)
    artifacts = {
        "summary_file": str(summary_path) if summary_path else None,
        "summary_file_present": bool(summary_path and summary_path.is_file()),
        "run_data_dir": str(trace_dir) if trace_dir else None,
        "run_data_dir_present": bool(trace_dir and trace_dir.is_dir()),
        "task_result_files": [str(path) for path in task_result_files],
        "task_result_count": len(task_result_files),
        "raw_trace_file_count": len(raw_trace_files),
        "raw_trace_policy": (
            "raw run_data traces may contain decrypted task text, ground truth, "
            "model output, and screenshots; this probe records paths/counts only"
        ),
    }
    failure_class = _classify_outcome(
        returncode=proc.returncode,
        task_results=task_results,
        summary_entry=summary_entry,
        stdout=stdout,
        stderr=stderr,
        artifacts=artifacts,
    )
    status = "completed" if failure_class is None else "blocked"
    all_trace_complete = bool(task_results) and all(
        _positive_int(item.get("steps")) and not item.get("error")
        for item in task_results
    )

    return {
        "schema": SCHEMA,
        "status": status,
        **({"failure_class": failure_class} if failure_class else {}),
        "source": {
            "type": "browser-use-benchmark",
            "upstream_repo": str(upstream_repo),
            "benchmark": benchmark,
            "task_indices": task_indices,
            "interleaved_order": not no_interleave,
        },
        "runner": {
            "framework": framework,
            "browser": browser,
            "model": model,
            "parallel": parallel,
            "task_timeout_sec": task_timeout_sec,
            "runner_timeout_sec": runner_timeout_sec,
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 3),
            "command": _shell_join(command),
        },
        "summary": _scrub_summary(summary_entry),
        "task_results": [_scrub_task_result(item) for item in task_results],
        "artifacts": artifacts,
        "checks": {
            "score_recorded": any(_number(item.get("score")) is not None for item in task_results),
            "trace_complete": all_trace_complete,
            "result_count": len(task_results),
            "expected_result_count": len(task_indices),
            "artifact_shape": {
                "summary": bool(summary_path and summary_path.is_file()),
                "task_results": len(task_result_files),
                "raw_trace_files": len(raw_trace_files),
            },
        },
        "output": {
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
        },
    }


def _blocked_report(
    *,
    upstream_repo: Path,
    command: list[str],
    benchmark: str,
    task_indices: list[int],
    framework: str,
    browser: str,
    model: str,
    parallel: int,
    task_timeout_sec: int,
    runner_timeout_sec: int,
    elapsed_sec: float,
    failure_class: str,
    reason: str,
    stdout: str | bytes | None = None,
    stderr: str | bytes | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "blocked",
        "failure_class": failure_class,
        "reason": reason,
        "source": {
            "type": "browser-use-benchmark",
            "upstream_repo": str(upstream_repo),
            "benchmark": benchmark,
            "task_indices": task_indices,
        },
        "runner": {
            "framework": framework,
            "browser": browser,
            "model": model,
            "parallel": parallel,
            "task_timeout_sec": task_timeout_sec,
            "runner_timeout_sec": runner_timeout_sec,
            "returncode": None,
            "elapsed_sec": round(elapsed_sec, 3),
            "command": _shell_join(command) if command else None,
        },
        "summary": None,
        "task_results": [],
        "artifacts": {
            "summary_file_present": False,
            "run_data_dir_present": False,
            "task_result_count": 0,
            "raw_trace_file_count": 0,
            "raw_trace_policy": (
                "raw run_data traces may contain decrypted task text, ground truth, "
                "model output, and screenshots; this probe records paths/counts only"
            ),
        },
        "checks": {
            "score_recorded": False,
            "trace_complete": False,
            "result_count": 0,
            "expected_result_count": len(task_indices),
        },
        "output": {
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
        },
    }


def _printed_path(text: str, pattern: re.Pattern[str], *, upstream_repo: Path) -> Path | None:
    match = pattern.search(text)
    if not match:
        return None
    path = Path(match.group("path").strip())
    return path if path.is_absolute() else upstream_repo / path


def _latest_summary_entry(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, list) and payload and isinstance(payload[-1], dict):
        return payload[-1]
    if isinstance(payload, dict):
        return payload
    return None


def _summary_task_results(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(summary, dict):
        return []
    raw = summary.get("task_results")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _task_result_files(trace_dir: Path | None) -> list[Path]:
    if trace_dir is None:
        return []
    task_results = trace_dir / "_task_results"
    if not task_results.is_dir():
        return []
    return sorted(task_results.glob("task_*.json"))


def _raw_trace_files(trace_dir: Path | None) -> list[Path]:
    if trace_dir is None or not trace_dir.is_dir():
        return []
    return sorted(
        path
        for path in trace_dir.glob("*.json")
        if path.parent.name != "_task_results"
    )


def _classify_outcome(
    *,
    returncode: int,
    task_results: list[dict[str, Any]],
    summary_entry: dict[str, Any] | None,
    stdout: str,
    stderr: str,
    artifacts: dict[str, Any],
) -> str | None:
    combined = f"{stdout}\n{stderr}".lower()
    errors = " ".join(str(item.get("error") or "") for item in task_results).lower()
    haystack = f"{combined}\n{errors}"
    if returncode != 0 and not summary_entry:
        if "module" in haystack and "not found" in haystack:
            return "runner-dependency-missing"
        if "keyerror" in haystack and "model" in haystack:
            return "unsupported-model"
        if "api key" in haystack or "unauthorized" in haystack:
            return "llm-auth-or-provider-error"
        return "runner-exited-without-summary"
    if summary_entry is None:
        return "runner-exited-without-summary"
    if not task_results:
        return "runner-exited-without-task-results"
    if all(_number(item.get("score")) is not None for item in task_results):
        if any(_positive_int(item.get("steps")) for item in task_results):
            return None
        if "timed out" in haystack:
            if "browser" in haystack or "chromium" in haystack or "local" in haystack:
                return "host-local-browser-startup-timeout"
            return "original-runner-task-timeout"
        if "browser" in haystack and ("timeout" in haystack or "failed" in haystack):
            return "host-local-browser-startup-failure"
        if "api key" in haystack or "unauthorized" in haystack:
            return "llm-auth-or-provider-error"
        if not artifacts.get("raw_trace_file_count"):
            return "runner-produced-no-agent-trace"
    return None


def _scrub_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    allowed = (
        "run_start",
        "benchmark",
        "framework",
        "framework_ref",
        "browser",
        "model",
        "params",
        "task_indices",
        "tasks_completed",
        "tasks_successful",
        "total_steps",
        "total_duration",
        "total_cost",
    )
    return {key: _safe_value(summary.get(key)) for key in allowed if key in summary}


def _scrub_task_result(item: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "task_id",
        "task_index",
        "score",
        "steps",
        "duration",
        "cost",
        "error",
    )
    return {key: _safe_value(item.get(key)) for key in allowed if key in item}


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    return value


def _tail(value: str | bytes | None, *, max_chars: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    return _redact(text[-max_chars:])


def _redact(value: str) -> str:
    out = value
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("<redacted>", out)
    return out


def _shell_join(parts: list[str]) -> str:
    return _redact(" ".join(shlex.quote(part) for part in parts))


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any) -> bool:
    number = _number(value)
    return number is not None and number > 0 and int(number) == number


def _parse_indices(raw: str) -> list[int]:
    indices: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            indices.append(int(part))
    return indices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        required=True,
        help="Clone of https://github.com/browser-use/benchmark.",
    )
    parser.add_argument("--benchmark", default="BU_Bench_V1")
    parser.add_argument("--framework", default="browser-use")
    parser.add_argument("--browser", default="local_headless")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--task-indices", default="0")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--task-timeout-sec", type=int, default=1800)
    parser.add_argument("--runner-timeout-sec", type=int, default=2100)
    parser.add_argument("--no-interleave", action="store_true")
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args(argv)

    report = probe_original_runner(
        upstream_repo=args.upstream_repo,
        task_indices=_parse_indices(args.task_indices),
        benchmark=args.benchmark,
        framework=args.framework,
        browser=args.browser,
        model=args.model,
        parallel=args.parallel,
        task_timeout_sec=args.task_timeout_sec,
        runner_timeout_sec=args.runner_timeout_sec,
        no_interleave=args.no_interleave,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(payload)
    print(payload, end="")
    return 0 if report["status"] in {"completed", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
