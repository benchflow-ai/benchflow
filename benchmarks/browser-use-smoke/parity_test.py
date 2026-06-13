#!/usr/bin/env python3
"""Run original-vs-BenchFlow parity for the Browser Use smoke fixture."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from original_runner import run_original

from benchflow.environment_adapter_parity import (
    write_environment_adapter_adoption_report,
    write_environment_adapter_loop_state,
    write_environment_adapter_parity_experiment,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TASK_DIR = HERE / "tasks" / "open-local-page"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        default="browser-use-smoke",
        choices=(
            "browser-use-smoke",
            "browser-use-cli",
            "browser-use-agent",
            "stagehand-agent",
        ),
        help="BenchFlow agent adapter to compare against the original runner.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Model to pass through bench eval create for LLM-driven browser "
            "agents such as browser-use-agent or stagehand-agent."
        ),
    )
    parser.add_argument(
        "--parity-out",
        type=Path,
        help="Write bench-agent-verify parity_experiment.json evidence here.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="benchflow-browser-use-smoke-") as tmp:
        work = Path(tmp)
        jobs_dir = work / "jobs"

        docker_before = _benchflow_docker_resources()
        original = run_original(TASK_DIR)
        bench_eval = _run_benchflow(
            TASK_DIR,
            jobs_dir,
            agent=args.agent,
            model=args.model,
        )
        docker_after = _wait_for_benchflow_docker_cleanup(expected=docker_before)
        if docker_before != docker_after:
            raise AssertionError(
                "BenchFlow-owned Docker resources changed after cleanup: "
                f"before={docker_before} after={docker_after}"
            )
        bench_result_path = _single_result_json(jobs_dir)
        bench_result = json.loads(bench_result_path.read_text())

        artifact_path = bench_result_path.parent / "artifacts" / (
            "browser-use-smoke-trace.json"
        )
        if not artifact_path.is_file():
            raise AssertionError(f"missing BenchFlow artifact: {artifact_path}")
        artifact = json.loads(artifact_path.read_text())

        summary = _compare(
            original,
            bench_result,
            artifact,
            docker_after,
            expected_agent=args.agent,
            bench_eval=bench_eval,
        )
        if args.parity_out:
            parity_experiment = write_environment_adapter_parity_experiment(
                args.parity_out,
                benchmark="browser-use-smoke",
                task_id=original["task_id"],
                original=original,
                benchflow=bench_result,
                artifact=artifact,
                summary=summary,
                agent=args.agent,
                sandbox="docker",
                environment_adapter="browser",
                benchmark_adapter="browser-use",
                artifact_manifest=_artifact_manifest(args.agent),
            )
            adoption_report_path = args.parity_out.with_name("adoption_report.json")
            adoption_report = write_environment_adapter_adoption_report(
                adoption_report_path,
                parity_experiment=parity_experiment,
                original=original,
                benchflow=bench_result,
                artifact=artifact,
                summary=summary,
                parity_experiment_path=args.parity_out,
            )
            loop_state_path = args.parity_out.with_name("loop_state.json")
            write_environment_adapter_loop_state(
                loop_state_path,
                parity_experiment=parity_experiment,
                adoption_report=adoption_report,
                commands=_loop_commands(args),
                artifacts={
                    "parity_experiment": args.parity_out,
                    "adoption_report": adoption_report_path,
                },
                source={
                    "type": "checked-in-fixture",
                    "path": str(TASK_DIR.relative_to(REPO_ROOT)),
                    "selected_tasks": [original["task_id"]],
                },
                queue=[
                    {
                        "id": "official-browser-use-original-runner-parity",
                        "status": "queued",
                    }
                ],
            )
            summary["parity_experiment"] = str(args.parity_out)
            summary["adoption_report"] = str(adoption_report_path)
            summary["loop_state"] = str(loop_state_path)
        print(json.dumps(summary, indent=2))


def _artifact_manifest(agent: str) -> list[dict[str, Any]]:
    screenshot_requirement: dict[str, Any]
    if agent == "browser-use-smoke":
        screenshot_requirement = {
            "id": "browser-screenshot-field",
            "source": "artifact",
            "path": "screenshots_b64",
            "kind": "sequence",
            "exists": True,
            "required": False,
        }
    else:
        screenshot_requirement = {
            "id": "browser-screenshots",
            "source": "artifact",
            "path": "screenshots_b64",
            "kind": "sequence",
            "min_count": 1,
        }
    return [
        {
            "id": "browser-trace-steps",
            "source": "artifact",
            "path": "steps",
            "kind": "sequence",
            "min_count": 1,
        },
        {
            "id": "browser-runtime-trace-schema",
            "source": "artifact",
            "path": "schema",
            "kind": "field",
            "equals": "benchflow.browser-runtime-trace.v1",
        },
        screenshot_requirement,
        {
            "id": "browser-final-result",
            "source": "artifact",
            "path": "final_result",
            "kind": "field",
        },
        {
            "id": "browser-environment-readiness",
            "source": "artifact",
            "path": "environment.readiness.status",
            "kind": "field",
            "equals": "ready",
        },
        {
            "id": "browser-environment-content-hash",
            "source": "artifact",
            "path": "environment.readiness.content_sha256",
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


def _loop_commands(args: argparse.Namespace) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "benchmarks/browser-use-smoke/parity_test.py",
    ]
    if args.agent != "browser-use-smoke":
        command.extend(["--agent", str(args.agent)])
    if args.model:
        command.extend(["--model", str(args.model)])
    if args.parity_out:
        command.extend(["--parity-out", str(args.parity_out)])
    prefix = (
        "GEMINI_API_KEY=<redacted> "
        if args.agent in {"browser-use-agent", "stagehand-agent"}
        else ""
    )
    return [
        prefix + _shell_join(command),
        f"uv run bench agent verify {shlex.quote(str(args.agent))} "
        "--benchmarks-dir <evidence-root> --json",
    ]


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _run_benchflow(
    native_task: Path,
    jobs_dir: Path,
    *,
    agent: str,
    model: str | None,
) -> dict[str, Any]:
    cmd = [
        "uv",
        "run",
        "bench",
        "eval",
        "create",
        "--tasks-dir",
        str(native_task),
        "--agent",
        agent,
        "--sandbox",
        "docker",
        "--jobs-dir",
        str(jobs_dir),
        "--json",
    ]
    if model:
        cmd.extend(["--model", model])
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "BenchFlow smoke failed with rc "
            f"{result.returncode}\nCommand: {cmd}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            "BenchFlow smoke did not emit parseable eval JSON\n"
            f"Command: {cmd}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"BenchFlow smoke eval JSON was not an object: {payload!r}")
    return payload


def _single_result_json(jobs_dir: Path) -> Path:
    matches = sorted(jobs_dir.glob("*/*/result.json"))
    if len(matches) != 1:
        tree = "\n".join(str(path) for path in sorted(jobs_dir.rglob("*")))
        raise AssertionError(
            f"expected one BenchFlow result.json, found {len(matches)}\n{tree}"
        )
    return matches[0]


def _compare(
    original: dict[str, Any],
    bench_result: dict[str, Any],
    artifact: dict[str, Any],
    docker_snapshot: dict[str, Any],
    *,
    expected_agent: str,
    bench_eval: dict[str, Any],
) -> dict[str, Any]:
    original_score = float(original["score"])
    bench_reward = float(bench_result["rewards"]["reward"])
    if bench_reward != original_score:
        raise AssertionError(
            f"reward mismatch: original={original_score} benchflow={bench_reward}"
        )
    if original["final_result"] != artifact["final_result"]:
        raise AssertionError(
            "final_result mismatch: "
            f"original={original['final_result']!r} "
            f"benchflow={artifact['final_result']!r}"
        )
    if original["num_steps"] <= 0 or not artifact["steps"]:
        raise AssertionError("trace steps missing from original or BenchFlow artifact")
    if "screenshots_b64" not in original or "screenshots_b64" not in artifact:
        raise AssertionError("screenshot artifact keys missing")
    if bench_result.get("error") is not None:
        raise AssertionError(f"BenchFlow error was not clean: {bench_result['error']}")
    if bench_result["agent"] != expected_agent:
        raise AssertionError(f"unexpected BenchFlow agent: {bench_result['agent']}")
    trajectory_summary = bench_result.get("trajectory_summary") or {}
    tool_calls = int(
        bench_result.get("n_tool_calls")
        or trajectory_summary.get("tool_call_steps")
        or 0
    )
    if tool_calls < 1:
        raise AssertionError(f"BenchFlow agent emitted no tool calls: {bench_result}")
    if int(trajectory_summary.get("steps", 0)) < 3:
        raise AssertionError(f"BenchFlow trajectory is too thin: {bench_result}")
    _assert_eval_report(
        bench_eval,
        expected_agent=expected_agent,
        expected_environment="docker",
        min_trajectory_steps=3,
    )

    return {
        "ok": True,
        "task_id": original["task_id"],
        "original": {
            "score": original_score,
            "num_steps": original["num_steps"],
            "screenshots_b64": len(original["screenshots_b64"]),
        },
        "benchflow": {
            "reward": bench_reward,
            "agent": bench_result["agent"],
            "trajectory_steps": trajectory_summary["steps"],
            "tool_calls": tool_calls,
            "artifact_steps": len(artifact["steps"]),
            "screenshots_b64": len(artifact["screenshots_b64"]),
        },
        "cleanup": {
            "docker_available": docker_snapshot["available"],
            "benchflow_containers": len(docker_snapshot["containers"]),
            "benchflow_networks": len(docker_snapshot["networks"]),
        },
        "benchflow_eval": bench_eval,
    }


def _assert_eval_report(
    payload: dict[str, Any],
    *,
    expected_agent: str,
    expected_environment: str,
    min_trajectory_steps: int,
) -> None:
    if payload.get("status") != "completed" or payload.get("ok") is not True:
        raise AssertionError(f"BenchFlow eval JSON did not complete cleanly: {payload}")
    result = payload.get("result")
    summary = payload.get("summary")
    if not isinstance(result, dict) or not isinstance(summary, dict):
        raise AssertionError(f"BenchFlow eval JSON missing result/summary: {payload}")
    if int(result.get("total") or 0) != 1 or int(summary.get("total") or 0) != 1:
        raise AssertionError(f"BenchFlow eval JSON expected one task: {payload}")
    if int(result.get("errored") or 0) or int(result.get("verifier_errored") or 0):
        raise AssertionError(f"BenchFlow eval result had errors: {payload}")
    if int(summary.get("errored") or 0) or int(summary.get("verifier_errored") or 0):
        raise AssertionError(f"BenchFlow eval summary had errors: {payload}")
    if summary.get("agent") != expected_agent:
        raise AssertionError(f"BenchFlow eval summary agent mismatch: {payload}")
    if summary.get("environment") != expected_environment:
        raise AssertionError(f"BenchFlow eval summary environment mismatch: {payload}")
    if int(summary.get("total_trajectory_steps") or 0) < min_trajectory_steps:
        raise AssertionError(f"BenchFlow eval summary is trace-thin: {payload}")
    if not payload.get("summary_path"):
        raise AssertionError(f"BenchFlow eval JSON missing summary_path: {payload}")


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


if __name__ == "__main__":
    main()
