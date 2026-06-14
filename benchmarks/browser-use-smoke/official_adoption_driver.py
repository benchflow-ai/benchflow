#!/usr/bin/env python3
"""Run one official Browser Use slice through the BenchFlow adoption loop.

This driver stitches together the official importer, BenchFlow eval run, and
official original-runner probe into one resumable evidence bundle. It keeps
plaintext task material and raw upstream traces under an operator-chosen temp
directory and writes only scrubbed adoption artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from import_upstream import import_tasks
from original_runner_probe import probe_original_runner

from benchflow.environment_adapter_parity import (
    write_environment_adapter_adoption_report,
    write_environment_adapter_loop_state,
    write_environment_adapter_parity_experiment,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def run_official_adoption(
    *,
    upstream_repo: Path,
    work_dir: Path,
    task_indices: list[int],
    benchmark: str = "BU_Bench_V1",
    encrypted_file: Path | None = None,
    raw_order: bool = False,
    overwrite: bool = False,
    agent: str = "browser-use-agent",
    model: str = "gemini-2.5-flash",
    sandbox: str = "docker",
    concurrency: int = 1,
    agent_idle_timeout: int = 900,
    judge_model: str = "gemini-2.5-flash",
    judge_env_key: str = "GEMINI_API_KEY",
    original_framework: str = "browser-use",
    original_browser: str = "local_headless",
    original_model: str = "gemini-2.5-flash",
    original_task_timeout_sec: int = 1800,
    original_runner_timeout_sec: int = 2100,
    parity_out: Path | None = None,
) -> dict[str, Any]:
    """Run the smallest official Browser Use adoption loop and write evidence."""

    if len(task_indices) != 1:
        raise ValueError("official Browser Use adoption driver expects one task index")

    work_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = work_dir / "tasks"
    jobs_dir = work_dir / "jobs"
    output_dir = parity_out.parent if parity_out else work_dir / "browser-use-official"
    output_dir.mkdir(parents=True, exist_ok=True)
    parity_path = parity_out or output_dir / "parity_experiment.json"
    adoption_report_path = parity_path.with_name("adoption_report.json")
    loop_state_path = parity_path.with_name("loop_state.json")
    original_probe_path = parity_path.with_name("original_runner_probe.json")

    selected_encrypted_file = encrypted_file or upstream_repo / f"{benchmark}.enc"
    task_dirs = import_tasks(
        encrypted_file=selected_encrypted_file,
        out_dir=tasks_dir,
        benchmark=benchmark,
        task_indices=task_indices,
        interleave=not raw_order,
        overwrite=overwrite,
        judge_model=judge_model,
        judge_env_key=judge_env_key,
    )
    if len(task_dirs) != 1:
        raise AssertionError(f"expected one imported task dir, got {len(task_dirs)}")

    task_dir = task_dirs[0]
    _run_task_check(task_dir, sandbox=sandbox)
    docker_before = _benchflow_docker_resources()
    bench_eval = _run_benchflow_eval(
        tasks_dir,
        jobs_dir,
        agent=agent,
        model=model,
        sandbox=sandbox,
        concurrency=concurrency,
        agent_idle_timeout=agent_idle_timeout,
    )
    docker_after = _wait_for_benchflow_docker_cleanup(expected=docker_before)
    leaked = _leaked_benchflow_resources(before=docker_before, after=docker_after)
    if leaked["containers"] or leaked["networks"]:
        raise AssertionError(
            "BenchFlow-owned Docker resources THIS run created were not cleaned "
            f"up: leaked={leaked} (before={docker_before} after={docker_after})"
        )

    bench_result_path = _single_result_json(jobs_dir)
    bench_result = json.loads(bench_result_path.read_text())
    artifact_path = (
        bench_result_path.parent / "artifacts" / ("browser-use-smoke-trace.json")
    )
    if not artifact_path.is_file():
        raise AssertionError(f"missing BenchFlow artifact: {artifact_path}")
    artifact = json.loads(artifact_path.read_text())

    original_probe = probe_original_runner(
        upstream_repo=upstream_repo,
        task_indices=task_indices,
        benchmark=benchmark,
        framework=original_framework,
        browser=original_browser,
        model=original_model,
        parallel=1,
        task_timeout_sec=original_task_timeout_sec,
        runner_timeout_sec=original_runner_timeout_sec,
        no_interleave=raw_order,
    )
    original_probe_path.write_text(json.dumps(original_probe, indent=2) + "\n")

    task_id = str(
        bench_result.get("task_name")
        or bench_result.get("task_id")
        or _probe_task_id(original_probe)
        or task_dir.name
    )
    original = _original_from_probe(original_probe, task_id=task_id)
    cleanup = {
        "docker_available": docker_after["available"],
        "docker_containers": len(leaked["containers"]),
        "docker_networks": len(leaked["networks"]),
    }
    summary = _summary(
        original_probe=original_probe,
        original=original,
        bench_result=bench_result,
        artifact=artifact,
        bench_eval=bench_eval,
        cleanup=cleanup,
    )
    parity_experiment = write_environment_adapter_parity_experiment(
        parity_path,
        benchmark="browser-use-official",
        task_id=task_id,
        original=original,
        benchflow=bench_result,
        artifact=artifact,
        summary=summary,
        agent=agent,
        sandbox=sandbox,
        environment_adapter="browser",
        benchmark_adapter="browser-use",
        artifact_manifest=_artifact_manifest(),
    )
    adoption_report = write_environment_adapter_adoption_report(
        adoption_report_path,
        parity_experiment=parity_experiment,
        original=original,
        benchflow=bench_result,
        artifact=artifact,
        summary=summary,
        parity_experiment_path=parity_path,
    )
    loop_state = write_environment_adapter_loop_state(
        loop_state_path,
        parity_experiment=parity_experiment,
        adoption_report=adoption_report,
        commands=_loop_commands(
            upstream_repo=upstream_repo,
            work_dir=work_dir,
            task_indices=task_indices,
            benchmark=benchmark,
            agent=agent,
            model=model,
            sandbox=sandbox,
            parity_out=parity_path,
            original_browser=original_browser,
            original_model=original_model,
        ),
        artifacts={
            "parity_experiment": parity_path,
            "adoption_report": adoption_report_path,
            "loop_state": loop_state_path,
            "original_runner_probe": original_probe_path,
            "benchflow_result": bench_result_path,
            "benchflow_agent_artifact": artifact_path,
        },
        source={
            "type": "browser-use-benchmark",
            "upstream_repo": str(upstream_repo),
            "benchmark": benchmark,
            "selected_tasks": task_indices,
            "task_dirs": [str(path) for path in task_dirs],
            "plaintext_policy": (
                "official encrypted tasks are decrypted only under work_dir"
            ),
        },
        roles=_roles(
            original_probe=original_probe,
            bench_result=bench_result,
            parity_experiment=parity_experiment,
            cleanup=cleanup,
        ),
        queue=_queue(original_probe),
    )

    result = {
        "ok": parity_experiment.get("status") == "parity-confirmed",
        "status": parity_experiment.get("status"),
        "loop_status": loop_state.get("status"),
        "task_id": task_id,
        "original_runner": {
            "status": original_probe.get("status"),
            "failure_class": original_probe.get("failure_class"),
            "trace_complete": _nested(original_probe, "checks", "trace_complete"),
        },
        "benchflow": {
            "reward": _nested(bench_result, "rewards", "reward"),
            "trajectory_steps": _nested(bench_result, "trajectory_summary", "steps"),
            "artifact_steps": len(artifact.get("steps") or []),
            "screenshots_b64": len(artifact.get("screenshots_b64") or []),
        },
        "cleanup": cleanup,
        "artifacts": {
            "parity_experiment": str(parity_path),
            "adoption_report": str(adoption_report_path),
            "loop_state": str(loop_state_path),
            "original_runner_probe": str(original_probe_path),
        },
    }
    return result


def _run_task_check(task_dir: Path, *, sandbox: str) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "bench",
            "tasks",
            "check",
            str(task_dir),
            "--level",
            "runtime-capability",
            "--sandbox",
            sandbox,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"task check failed with rc {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _run_benchflow_eval(
    tasks_dir: Path,
    jobs_dir: Path,
    *,
    agent: str,
    model: str,
    sandbox: str,
    concurrency: int,
    agent_idle_timeout: int,
) -> dict[str, Any]:
    cmd = [
        "uv",
        "run",
        "--extra",
        "judge",
        "bench",
        "eval",
        "create",
        "--tasks-dir",
        str(tasks_dir),
        "--agent",
        agent,
        "--model",
        model,
        "--sandbox",
        sandbox,
        "--jobs-dir",
        str(jobs_dir),
        "--concurrency",
        str(concurrency),
        "--agent-idle-timeout",
        str(agent_idle_timeout),
        "--json",
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_eval_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "BenchFlow official Browser Use eval failed with rc "
            f"{result.returncode}\nCommand: {_shell_join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            "BenchFlow official Browser Use eval did not emit JSON\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"BenchFlow eval JSON was not an object: {payload!r}")
    return payload


def _eval_env() -> dict[str, str]:
    env = os.environ.copy()
    if "GOOGLE_API_KEY" not in env and env.get("GEMINI_API_KEY"):
        env["GOOGLE_API_KEY"] = env["GEMINI_API_KEY"]
    if "GEMINI_API_KEY" not in env and env.get("GOOGLE_API_KEY"):
        env["GEMINI_API_KEY"] = env["GOOGLE_API_KEY"]
    return env


def _single_result_json(jobs_dir: Path) -> Path:
    matches = sorted(jobs_dir.glob("*/*/result.json"))
    if len(matches) != 1:
        tree = "\n".join(str(path) for path in sorted(jobs_dir.rglob("*")))
        raise AssertionError(
            f"expected one BenchFlow result.json, found {len(matches)}\n{tree}"
        )
    return matches[0]


def _original_from_probe(report: dict[str, Any], *, task_id: str) -> dict[str, Any]:
    task_result = _first_task_result(report)
    score = task_result.get("score") if task_result else None
    steps = task_result.get("steps") if task_result else 0
    duration = task_result.get("duration") if task_result else None
    error = task_result.get("error") if task_result else report.get("failure_class")
    return {
        "task_id": _probe_task_id(report) or task_id,
        "framework": f"{_nested(report, 'runner', 'framework')}:"
        f"{_nested(report, 'runner', 'browser')}",
        "score": score,
        "num_steps": steps,
        "duration_sec": duration,
        "final_result": None,
        "screenshots_b64": [],
        "error": error,
        "probe_status": report.get("status"),
        "failure_class": report.get("failure_class"),
    }


def _summary(
    *,
    original_probe: dict[str, Any],
    original: dict[str, Any],
    bench_result: dict[str, Any],
    artifact: dict[str, Any],
    bench_eval: dict[str, Any],
    cleanup: dict[str, Any],
) -> dict[str, Any]:
    trajectory_summary = bench_result.get("trajectory_summary") or {}
    return {
        "ok": original_probe.get("status") == "completed"
        and _nested(original_probe, "checks", "trace_complete") is True,
        "task_id": original.get("task_id") or bench_result.get("task_name"),
        "original": {
            "score": original.get("score"),
            "num_steps": original.get("num_steps"),
            "duration_sec": original.get("duration_sec"),
            "probe_status": original_probe.get("status"),
            "failure_class": original_probe.get("failure_class"),
            "trace_complete": _nested(original_probe, "checks", "trace_complete"),
        },
        "benchflow": {
            "reward": _nested(bench_result, "rewards", "reward"),
            "agent": bench_result.get("agent"),
            "trajectory_steps": trajectory_summary.get("steps"),
            "tool_calls": bench_result.get("n_tool_calls")
            or trajectory_summary.get("tool_call_steps"),
            "artifact_steps": len(artifact.get("steps") or []),
            "screenshots_b64": len(artifact.get("screenshots_b64") or []),
        },
        "cleanup": cleanup,
        "benchflow_eval": bench_eval,
        "original_runner_probe": {
            "schema": original_probe.get("schema"),
            "status": original_probe.get("status"),
            "failure_class": original_probe.get("failure_class"),
            "checks": original_probe.get("checks"),
        },
    }


def _artifact_manifest() -> list[dict[str, Any]]:
    return [
        {
            "id": "browser-trace-steps",
            "source": "artifact",
            "path": "steps",
            "kind": "sequence",
            "min_count": 1,
        },
        {
            "id": "browser-screenshots",
            "source": "artifact",
            "path": "screenshots_b64",
            "kind": "sequence",
            "min_count": 1,
        },
        {
            "id": "browser-final-result",
            "source": "artifact",
            "path": "final_result",
            "kind": "field",
        },
        {
            "id": "browser-trajectory-steps",
            "source": "benchflow",
            "path": "trajectory_summary.steps",
            "kind": "numeric",
            "numeric_min": 3,
        },
    ]


def _roles(
    *,
    original_probe: dict[str, Any],
    bench_result: dict[str, Any],
    parity_experiment: dict[str, Any],
    cleanup: dict[str, Any],
) -> list[dict[str, Any]]:
    original_ready = (
        original_probe.get("status") == "completed"
        and _nested(original_probe, "checks", "trace_complete") is True
    )
    bench_ready = bench_result.get("error") in (None, "")
    parity_ready = parity_experiment.get("status") == "parity-confirmed"
    cleanup_ready = _cleanup_zero(cleanup)
    return [
        {"name": "scout", "status": "passed", "artifact": "source"},
        {"name": "builder", "status": "passed", "artifact": "adapter-diff"},
        {
            "name": "original-runner",
            "status": "passed" if original_ready else "blocked",
            "artifact": "original_runner_probe",
            "notes": {
                "failure_class": original_probe.get("failure_class"),
                "trace_complete": _nested(original_probe, "checks", "trace_complete"),
            },
        },
        {
            "name": "benchflow-runner",
            "status": "passed" if bench_ready else "blocked",
            "artifact": "benchflow-result",
        },
        {
            "name": "verifier",
            "status": "passed" if parity_ready else "pending",
            "artifact": "parity_experiment",
        },
        {
            "name": "auditor",
            "status": "passed" if cleanup_ready else "blocked",
            "artifact": "adoption_report",
        },
        {"name": "reviewer", "status": "pending", "artifact": "review-report"},
        {"name": "queue", "status": "queued", "artifact": "next-chunks"},
    ]


def _queue(original_probe: dict[str, Any]) -> list[dict[str, Any]]:
    failure_class = original_probe.get("failure_class")
    return [
        {
            "id": "official-browser-use-original-runner-parity",
            "status": "blocked" if failure_class else "queued",
            "reason": failure_class,
        },
        {
            "id": "broader-browser-use-task-slice",
            "status": "queued",
            "depends_on": "official-browser-use-original-runner-parity",
        },
    ]


def _loop_commands(
    *,
    upstream_repo: Path,
    work_dir: Path,
    task_indices: list[int],
    benchmark: str,
    agent: str,
    model: str,
    sandbox: str,
    parity_out: Path,
    original_browser: str,
    original_model: str,
) -> list[str]:
    indices = ",".join(str(index) for index in task_indices)
    return [
        _shell_join(
            [
                "uv",
                "run",
                "python",
                "benchmarks/browser-use-smoke/official_adoption_driver.py",
                "--upstream-repo",
                str(upstream_repo),
                "--work-dir",
                str(work_dir),
                "--benchmark",
                benchmark,
                "--task-indices",
                indices,
                "--agent",
                agent,
                "--model",
                model,
                "--sandbox",
                sandbox,
                "--original-browser",
                original_browser,
                "--original-model",
                original_model,
                "--parity-out",
                str(parity_out),
                "--overwrite",
            ]
        ),
        _shell_join(
            [
                "uv",
                "run",
                "bench",
                "agent",
                "verify",
                "browser-use-official",
                "--benchmarks-dir",
                "<evidence-root>",
                "--json",
            ]
        ),
    ]


def _first_task_result(report: dict[str, Any]) -> dict[str, Any] | None:
    task_results = report.get("task_results")
    if isinstance(task_results, list) and task_results:
        first = task_results[0]
        if isinstance(first, dict):
            return first
    return None


def _probe_task_id(report: dict[str, Any]) -> str | None:
    first = _first_task_result(report)
    if first and first.get("task_id"):
        return str(first["task_id"])
    return None


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _cleanup_zero(cleanup: dict[str, Any]) -> bool:
    for key, value in cleanup.items():
        if not str(key).endswith(("containers", "networks")):
            continue
        try:
            if int(value) != 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _benchflow_docker_resources() -> dict[str, Any]:
    if shutil.which("docker") is None:
        return {"available": False, "containers": [], "networks": []}
    return {
        "available": True,
        "containers": _docker_ids("container", "container", "ls", "-aq"),
        "networks": _docker_ids("network", "network", "ls", "-q"),
    }


def _wait_for_benchflow_docker_cleanup(
    *,
    expected: dict[str, Any],
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    snapshot = _benchflow_docker_resources()
    while snapshot != expected and time.monotonic() < deadline:
        time.sleep(0.25)
        snapshot = _benchflow_docker_resources()
    return snapshot


def _leaked_benchflow_resources(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, list[str]]:
    """Resources THIS run created and failed to clean up.

    Scopes the cleanup check to the run: a resource counts as leaked only if it
    is present after the run but was not present before it. Pre-existing or
    concurrently-running benchflow-owned containers/networks (which this run did
    not create) are tolerated, so an unrelated container can no longer turn a
    clean run into a false leak failure.
    """

    before_containers = set(before.get("containers") or [])
    before_networks = set(before.get("networks") or [])
    return {
        "containers": sorted(
            cid
            for cid in (after.get("containers") or [])
            if cid not in before_containers
        ),
        "networks": sorted(
            nid for nid in (after.get("networks") or []) if nid not in before_networks
        ),
    }


def _docker_ids(resource: str, *args: str) -> list[str]:
    result = subprocess.run(
        [
            "docker",
            *args,
            "--filter",
            "label=benchflow.owned=true",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"docker {resource} listing failed")
    return sorted(line for line in result.stdout.splitlines() if line)


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _parse_indices(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--encrypted-file", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--benchmark", default="BU_Bench_V1")
    parser.add_argument("--task-indices", default="0")
    parser.add_argument("--raw-order", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--agent", default="browser-use-agent")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--sandbox", default="docker")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--agent-idle-timeout", type=int, default=900)
    parser.add_argument("--judge-model", default="gemini-2.5-flash")
    parser.add_argument("--judge-env-key", default="GEMINI_API_KEY")
    parser.add_argument("--original-framework", default="browser-use")
    parser.add_argument("--original-browser", default="local_headless")
    parser.add_argument("--original-model", default="gemini-2.5-flash")
    parser.add_argument("--original-task-timeout-sec", type=int, default=1800)
    parser.add_argument("--original-runner-timeout-sec", type=int, default=2100)
    parser.add_argument("--parity-out", type=Path)
    args = parser.parse_args()

    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="benchflow-bu-official-") as tmp:
            result = run_official_adoption(
                upstream_repo=args.upstream_repo,
                work_dir=Path(tmp),
                task_indices=_parse_indices(args.task_indices),
                benchmark=args.benchmark,
                encrypted_file=args.encrypted_file,
                raw_order=args.raw_order,
                overwrite=args.overwrite,
                agent=args.agent,
                model=args.model,
                sandbox=args.sandbox,
                concurrency=args.concurrency,
                agent_idle_timeout=args.agent_idle_timeout,
                judge_model=args.judge_model,
                judge_env_key=args.judge_env_key,
                original_framework=args.original_framework,
                original_browser=args.original_browser,
                original_model=args.original_model,
                original_task_timeout_sec=args.original_task_timeout_sec,
                original_runner_timeout_sec=args.original_runner_timeout_sec,
                parity_out=args.parity_out,
            )
            print(json.dumps(result, indent=2))
            return

    result = run_official_adoption(
        upstream_repo=args.upstream_repo,
        work_dir=args.work_dir,
        task_indices=_parse_indices(args.task_indices),
        benchmark=args.benchmark,
        encrypted_file=args.encrypted_file,
        raw_order=args.raw_order,
        overwrite=args.overwrite,
        agent=args.agent,
        model=args.model,
        sandbox=args.sandbox,
        concurrency=args.concurrency,
        agent_idle_timeout=args.agent_idle_timeout,
        judge_model=args.judge_model,
        judge_env_key=args.judge_env_key,
        original_framework=args.original_framework,
        original_browser=args.original_browser,
        original_model=args.original_model,
        original_task_timeout_sec=args.original_task_timeout_sec,
        original_runner_timeout_sec=args.original_runner_timeout_sec,
        parity_out=args.parity_out,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
