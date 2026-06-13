#!/usr/bin/env python3
"""Run original-vs-BenchFlow parity for cookbook desktop smoke slices."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
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
TASKS = {
    "osworld": HERE / "tasks" / "osworld-ubuntu-smoke",
}
ARTIFACT_MANIFEST = [
    {
        "id": "desktop-runtime-trace-schema",
        "source": "artifact",
        "path": "schema",
        "kind": "field",
        "equals": "benchflow.desktop-runtime-trace.v1",
    },
    {
        "id": "desktop-trace-steps",
        "source": "artifact",
        "path": "steps",
        "kind": "sequence",
        "min_count": 1,
    },
    {
        "id": "desktop-screenshots",
        "source": "artifact",
        "path": "screenshots_b64",
        "kind": "sequence",
        "min_count": 1,
    },
    {
        "id": "desktop-screenshot-method",
        "source": "artifact",
        "path": "screenshot_method",
        "kind": "field",
        "not_equals": "fallback-png",
    },
    {
        "id": "desktop-trajectory-steps",
        "source": "benchflow",
        "path": "trajectory_summary.steps",
        "kind": "numeric",
        "numeric_min": 5,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        default="computer-use-smoke",
        choices=("computer-use-smoke",),
        help="BenchFlow agent adapter to compare against the original runner.",
    )
    parser.add_argument(
        "--sandbox",
        default="cua",
        choices=("cua",),
        help="BenchFlow sandbox provider for the desktop smoke.",
    )
    parser.add_argument(
        "--task",
        default="osworld",
        choices=tuple(TASKS),
        help="Cookbook smoke slice to compare.",
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        help="External cookbook task dir, for importer-generated /tmp slices.",
    )
    parser.add_argument(
        "--parity-out",
        type=Path,
        help="Write bench-agent-verify parity_experiment.json evidence here.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="benchflow-use-computer-smoke-") as tmp:
        work = Path(tmp)
        jobs_dir = work / "jobs"
        task_dir = args.task_dir or TASKS[args.task]
        if not task_dir.is_dir():
            raise AssertionError(f"missing cookbook smoke task: {task_dir}")

        resources_before = _cua_docker_resources()
        original = run_original(task_dir)
        bench_eval = _run_benchflow(
            task_dir,
            jobs_dir,
            agent=args.agent,
            sandbox=args.sandbox,
        )
        resources_after = _cua_docker_resources()
        if resources_before != resources_after:
            raise AssertionError(
                "Cua Docker resources changed after cleanup: "
                f"before={resources_before} after={resources_after}"
            )

        bench_result_path = _single_result_json(jobs_dir)
        bench_result = json.loads(bench_result_path.read_text())
        artifact_path = bench_result_path.parent / "artifacts" / (
            "computer-use-smoke-trace.json"
        )
        if not artifact_path.is_file():
            raise AssertionError(f"missing BenchFlow artifact: {artifact_path}")
        artifact = json.loads(artifact_path.read_text())

        summary = _compare(
            original,
            bench_result,
            artifact,
            resources_after,
            expected_agent=args.agent,
            expected_environment=args.sandbox,
            bench_eval=bench_eval,
        )
        if args.parity_out:
            parity_experiment = write_environment_adapter_parity_experiment(
                args.parity_out,
                benchmark="use-computer-cookbook-smoke",
                task_id=original["task_id"],
                original=original,
                benchflow=bench_result,
                artifact=artifact,
                summary=summary,
                agent=args.agent,
                sandbox=args.sandbox,
                environment_adapter="desktop",
                benchmark_adapter="use-computer-cookbook",
                artifact_manifest=ARTIFACT_MANIFEST,
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
                commands=_loop_commands(args, task_dir),
                artifacts={
                    "parity_experiment": args.parity_out,
                    "adoption_report": adoption_report_path,
                },
                source={
                    "type": (
                        "external-task-dir"
                        if args.task_dir
                        else "checked-in-cookbook-fixture"
                    ),
                    "path": str(task_dir),
                    "selected_tasks": [original["task_id"]],
                },
                queue=[
                    {
                        "id": "broader-cuagym-import-scan",
                        "status": "queued",
                    },
                    {
                        "id": "windowsagentarena-macosworld-provider-mapping",
                        "status": "queued",
                    },
                ],
            )
            summary["parity_experiment"] = str(args.parity_out)
            summary["adoption_report"] = str(adoption_report_path)
            summary["loop_state"] = str(loop_state_path)
        print(json.dumps(summary, indent=2))


def _run_benchflow(
    native_task: Path,
    jobs_dir: Path,
    *,
    agent: str,
    sandbox: str,
) -> dict[str, Any]:
    env = {
        **os.environ,
        "BENCHFLOW_CUA_LOCAL": os.environ.get("BENCHFLOW_CUA_LOCAL", "1"),
        "BENCHFLOW_CUA_LINUX_KIND": os.environ.get(
            "BENCHFLOW_CUA_LINUX_KIND", "container"
        ),
    }
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
        sandbox,
        "--jobs-dir",
        str(jobs_dir),
        "--json",
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "BenchFlow use-computer cookbook smoke failed with rc "
            f"{result.returncode}\nCommand: {cmd}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            "BenchFlow use-computer cookbook smoke did not emit parseable eval JSON\n"
            f"Command: {cmd}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from exc
    if not isinstance(payload, dict):
        raise AssertionError(
            f"BenchFlow cookbook eval JSON was not an object: {payload!r}"
        )
    return payload


def _loop_commands(args: argparse.Namespace, task_dir: Path) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "benchmarks/use-computer-cookbook-smoke/parity_test.py",
    ]
    if args.agent != "computer-use-smoke":
        command.extend(["--agent", str(args.agent)])
    if args.sandbox != "cua":
        command.extend(["--sandbox", str(args.sandbox)])
    if args.task_dir:
        command.extend(["--task-dir", str(task_dir)])
    elif args.task != "osworld":
        command.extend(["--task", str(args.task)])
    if args.parity_out:
        command.extend(["--parity-out", str(args.parity_out)])
    return [
        "BENCHFLOW_CUA_LOCAL=1 BENCHFLOW_CUA_LINUX_KIND=container "
        + _shell_join(command),
        "uv run bench agent verify use-computer-cookbook-smoke "
        "--benchmarks-dir <evidence-root> --require-adoption-report --json",
    ]


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


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
    resources_snapshot: dict[str, Any],
    *,
    expected_agent: str,
    expected_environment: str,
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
    if not original["screenshots_b64"] or not artifact["screenshots_b64"]:
        raise AssertionError("screenshot artifacts missing")
    if artifact.get("screenshot_method") == "fallback-png":
        raise AssertionError(
            "BenchFlow computer-use smoke used fallback screenshot: "
            f"{artifact.get('screenshot_error')}"
        )
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
    if tool_calls < 3:
        raise AssertionError(f"BenchFlow agent emitted too few tool calls: {tool_calls}")
    if int(trajectory_summary.get("steps", 0)) < 5:
        raise AssertionError(f"BenchFlow trajectory is too thin: {bench_result}")
    _assert_eval_report(
        bench_eval,
        expected_agent=expected_agent,
        expected_environment=expected_environment,
        min_trajectory_steps=5,
    )

    return {
        "ok": True,
        "task_id": original["task_id"],
        "original": {
            "score": original_score,
            "num_steps": original["num_steps"],
            "screenshots_b64": len(original["screenshots_b64"]),
            "dimensions": original["dimensions"],
        },
        "benchflow": {
            "reward": bench_reward,
            "agent": bench_result["agent"],
            "trajectory_steps": trajectory_summary["steps"],
            "tool_calls": tool_calls,
            "artifact_steps": len(artifact["steps"]),
            "screenshots_b64": len(artifact["screenshots_b64"]),
            "screenshot_method": artifact["screenshot_method"],
        },
        "cleanup": {
            "docker_available": resources_snapshot["available"],
            "cua_containers": len(resources_snapshot["containers"]),
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


def _cua_docker_resources() -> dict[str, Any]:
    if shutil.which("docker") is None:
        return {"available": False, "containers": []}
    result = subprocess.run(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=cua.sandbox=true",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError("docker Cua container listing failed")
    return {
        "available": True,
        "containers": sorted(line for line in result.stdout.splitlines() if line),
    }


if __name__ == "__main__":
    main()
