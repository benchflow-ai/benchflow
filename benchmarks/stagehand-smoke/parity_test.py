#!/usr/bin/env python3
"""Run original-vs-BenchFlow parity for selected official Stagehand evals."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from import_upstream import import_tasks

from benchflow.environment_adapter_parity import (
    write_environment_adapter_adoption_report,
    write_environment_adapter_loop_state,
    write_environment_adapter_parity_experiment,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
_ORIGINAL_RESULT_MARKER = "BENCHFLOW_STAGEHAND_ORIGINAL_RESULT="


RunFn = Callable[..., subprocess.CompletedProcess[str]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stagehand-repo",
        type=Path,
        required=True,
        help="Clone of https://github.com/browserbase/stagehand.",
    )
    parser.add_argument("--task", default="agent/sign_in")
    parser.add_argument("--model", default="google/gemini-3.5-flash")
    parser.add_argument("--provider", default="google")
    parser.add_argument("--agent-mode", default="dom")
    parser.add_argument("--benchflow-agent", default="stagehand-agent")
    parser.add_argument("--sandbox", default="docker")
    parser.add_argument("--parity-out", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--upstream-commit")
    parser.add_argument("--keep-work-dir", action="store_true")
    args = parser.parse_args()

    summary = run_parity(
        stagehand_repo=args.stagehand_repo,
        task=args.task,
        model=args.model,
        provider=args.provider,
        agent_mode=args.agent_mode,
        benchflow_agent=args.benchflow_agent,
        sandbox=args.sandbox,
        parity_out=args.parity_out,
        work_dir=args.work_dir,
        upstream_commit=args.upstream_commit,
        keep_work_dir=args.keep_work_dir,
    )
    print(json.dumps(summary, indent=2))


def run_parity(
    *,
    stagehand_repo: Path,
    task: str,
    model: str,
    provider: str,
    agent_mode: str,
    benchflow_agent: str,
    sandbox: str,
    parity_out: Path | None = None,
    work_dir: Path | None = None,
    upstream_commit: str | None = None,
    keep_work_dir: bool = False,
    run_fn: RunFn = subprocess.run,
) -> dict[str, Any]:
    if sandbox != "docker":
        raise ValueError("Stagehand parity currently expects --sandbox docker")
    if work_dir is None:
        temp = tempfile.TemporaryDirectory(prefix="benchflow-stagehand-parity-")
        root = Path(temp.name)
    else:
        temp = None
        root = work_dir
        root.mkdir(parents=True, exist_ok=True)

    try:
        import_dir = root / "imported"
        jobs_dir = root / "jobs"
        original = run_original_stagehand(
            stagehand_repo=stagehand_repo,
            task=task,
            model=model,
            provider=provider,
            agent_mode=agent_mode,
            run_fn=run_fn,
        )
        imported, unsupported = import_tasks(
            stagehand_repo=stagehand_repo,
            out_dir=import_dir,
            tasks=[task],
            overwrite=True,
            upstream_commit=upstream_commit,
        )
        if unsupported:
            raise AssertionError(
                f"selected Stagehand task is unsupported: {unsupported}"
            )
        if len(imported) != 1:
            raise AssertionError(f"expected one imported task dir, got {imported}")
        descriptor = _load_stagehand_descriptor(imported[0])

        docker_before = _benchflow_docker_resources(run_fn=run_fn)
        bench_eval = _run_benchflow(
            imported[0],
            jobs_dir,
            agent=benchflow_agent,
            model=model,
            sandbox=sandbox,
            run_fn=run_fn,
        )
        docker_after = _benchflow_docker_resources(run_fn=run_fn)
        docker_leaked = _leaked_benchflow_resources(
            before=docker_before, after=docker_after
        )
        if docker_leaked["containers"] or docker_leaked["networks"]:
            raise AssertionError(
                "BenchFlow-owned Docker resources THIS run created were not "
                f"cleaned up: leaked={docker_leaked} "
                f"(before={docker_before} after={docker_after})"
            )

        result_path = _single_result_json(jobs_dir)
        bench_result = json.loads(result_path.read_text())
        artifact_path = (
            result_path.parent / "artifacts" / "browser-use-smoke-trace.json"
        )
        # A genuine agent failure (e.g. a live anti-bot 403 that blocks the
        # browser) leaves the BenchFlow eval completed with a reward but no
        # trace artifact. That is an honest parity DIVERGENCE, not an infra
        # error: record it as evidence rather than crashing, matching the
        # browser-use harness. The eval-completed contract is still enforced
        # below by ``_assert_eval_report``.
        artifact = _load_optional_json(artifact_path)
        verifier_artifact = _load_optional_json(
            result_path.parent / "artifacts" / "stagehand-url-verifier.json"
        )
        bench_url = _stagehand_current_url(artifact, verifier_artifact)
        parity_artifact = {
            **artifact,
            "final_result": _semantic_final_result(
                bench_url if bench_url is not None else "",
                descriptor,
            ),
        }
        summary = _compare(
            original,
            bench_result,
            artifact,
            verifier_artifact,
            docker_leaked,
            bench_eval=bench_eval,
            expected_agent=benchflow_agent,
            descriptor=descriptor,
        )

        if parity_out is not None:
            original_for_parity = {
                **original,
                "final_result": _semantic_final_result(
                    str(original["final_url"]),
                    descriptor,
                ),
            }
            parity = write_environment_adapter_parity_experiment(
                parity_out,
                benchmark="stagehand-smoke",
                task_id=task,
                original=original_for_parity,
                benchflow=bench_result,
                artifact=parity_artifact,
                summary=summary,
                agent=benchflow_agent,
                sandbox=sandbox,
                environment_adapter="browser",
                benchmark_adapter="stagehand-evals",
                artifact_manifest=_artifact_manifest(descriptor),
                unsupported=unsupported,
            )
            adoption_path = parity_out.with_name("adoption_report.json")
            adoption_report = write_environment_adapter_adoption_report(
                adoption_path,
                parity_experiment=parity,
                original=original_for_parity,
                benchflow=bench_result,
                artifact=parity_artifact,
                summary=summary,
                parity_experiment_path=parity_out,
                unsupported=unsupported,
            )
            loop_state_path = parity_out.with_name("loop_state.json")
            write_environment_adapter_loop_state(
                loop_state_path,
                parity_experiment=parity,
                adoption_report=adoption_report,
                commands=_loop_commands(
                    stagehand_repo=stagehand_repo,
                    task=task,
                    model=model,
                    provider=provider,
                    agent_mode=agent_mode,
                    benchflow_agent=benchflow_agent,
                    sandbox=sandbox,
                    parity_out=parity_out,
                    keep_work_dir=keep_work_dir,
                ),
                artifacts={
                    "parity_experiment": parity_out,
                    "adoption_report": adoption_path,
                },
                source={
                    "type": "upstream-stagehand-checkout",
                    "path": str(stagehand_repo),
                    "revision": upstream_commit or _git_revision(stagehand_repo),
                    "selected_tasks": [task],
                },
                queue=[
                    {
                        "id": "stagehand-verifier-reward-contract",
                        "status": "queued",
                    },
                    {
                        "id": "stagehand-expected-answer-contract",
                        "status": "queued",
                    },
                ],
            )
            summary["parity_experiment"] = str(parity_out)
            summary["adoption_report"] = str(adoption_path)
            summary["loop_state"] = str(loop_state_path)

        summary["work_dir"] = str(root)
        return summary
    finally:
        if temp is not None and not keep_work_dir:
            temp.cleanup()


def run_original_stagehand(
    *,
    stagehand_repo: Path,
    task: str,
    model: str,
    provider: str,
    agent_mode: str,
    run_fn: RunFn = subprocess.run,
) -> dict[str, Any]:
    runner = (
        stagehand_repo
        / "packages"
        / "evals"
        / "dist"
        / "esm"
        / "framework"
        / "runner.js"
    )
    if not runner.is_file():
        raise FileNotFoundError(
            f"Stagehand built runner not found: {runner}. Run pnpm install/build first."
        )
    tasks_root = _stagehand_tasks_root(stagehand_repo)
    script = _original_runner_script(
        stagehand_repo,
        task,
        model,
        provider,
        agent_mode,
        tasks_root=tasks_root,
    )
    env = _stagehand_original_env(model)
    cmd = ["node", "--input-type=module", "-e", script]
    if _stagehand_tasks_need_tsx(tasks_root):
        cmd[1:1] = ["--import", "tsx"]
    started_at = time.monotonic()
    result = run_fn(
        cmd,
        cwd=stagehand_repo / "packages" / "evals",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    duration_sec = time.monotonic() - started_at
    if result.returncode != 0:
        raise AssertionError(
            "Stagehand original runner failed with rc "
            f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    raw = _extract_marked_json(result.stdout)
    return _normalize_original_result(
        raw,
        task=task,
        fallback_duration_sec=duration_sec,
    )


def _original_runner_script(
    stagehand_repo: Path,
    task: str,
    model: str,
    provider: str,
    agent_mode: str,
    *,
    tasks_root: Path,
) -> str:
    runner = (
        stagehand_repo
        / "packages"
        / "evals"
        / "dist"
        / "esm"
        / "framework"
        / "runner.js"
    )
    return f"""
import {{ discoverTasks, resolveTarget, runEvals }} from {json.dumps(runner.as_uri())};
const registry = await discoverTasks({json.dumps(str(tasks_root))}, false);
const tasks = resolveTarget(registry, {json.dumps(task)});
const result = await runEvals({{
  tasks,
  registry,
  concurrency: 1,
  trials: 1,
  environment: "LOCAL",
  useApi: false,
  modelOverride: {json.dumps(model)},
  provider: {json.dumps(provider)},
  agentMode: {json.dumps(agent_mode)},
  harness: "stagehand",
  verbose: false,
}});
process.stdout.write("\\n{_ORIGINAL_RESULT_MARKER}" + JSON.stringify(result) + "\\n");
"""


def _stagehand_tasks_root(stagehand_repo: Path) -> Path:
    dist_tasks = stagehand_repo / "packages" / "evals" / "dist" / "esm" / "tasks"
    if dist_tasks.is_dir():
        return dist_tasks
    return stagehand_repo / "packages" / "evals" / "tasks"


def _stagehand_tasks_need_tsx(tasks_root: Path) -> bool:
    return "dist" not in tasks_root.parts


def _stagehand_original_env(model: str) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("EVAL_AGENT_MODELS", model)
    gemini = env.get("GEMINI_API_KEY")
    if gemini:
        env.setdefault("GOOGLE_GENERATIVE_AI_API_KEY", gemini)
        env.setdefault("GOOGLE_API_KEY", gemini)
    return env


def _loop_commands(
    *,
    stagehand_repo: Path,
    task: str,
    model: str,
    provider: str,
    agent_mode: str,
    benchflow_agent: str,
    sandbox: str,
    parity_out: Path,
    keep_work_dir: bool,
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "benchmarks/stagehand-smoke/parity_test.py",
        "--stagehand-repo",
        str(stagehand_repo),
        "--task",
        task,
        "--model",
        model,
        "--parity-out",
        str(parity_out),
    ]
    if provider != "google":
        command.extend(["--provider", provider])
    if agent_mode != "dom":
        command.extend(["--agent-mode", agent_mode])
    if benchflow_agent != "stagehand-agent":
        command.extend(["--benchflow-agent", benchflow_agent])
    if sandbox != "docker":
        command.extend(["--sandbox", sandbox])
    if keep_work_dir:
        command.append("--keep-work-dir")
    return [
        "GEMINI_API_KEY=<redacted> " + _shell_join(command),
        "uv run bench agent verify stagehand-smoke "
        f"--benchmarks-dir {shlex.quote(str(parity_out.parent.parent))} "
        "--require-adoption-report "
        f"--loop-report-out {shlex.quote(str(parity_out.with_name('loop-report.json')))} "
        "--json",
    ]


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _extract_marked_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_ORIGINAL_RESULT_MARKER):
            payload = json.loads(line[len(_ORIGINAL_RESULT_MARKER) :])
            if not isinstance(payload, dict):
                raise AssertionError("Stagehand original marker was not a JSON object")
            return payload
    raise AssertionError(f"Stagehand original output missing JSON marker:\n{stdout}")


def _normalize_original_result(
    raw: Mapping[str, Any],
    *,
    task: str,
    fallback_duration_sec: float | None = None,
) -> dict[str, Any]:
    results = raw.get("results")
    if (
        not isinstance(results, Sequence)
        or isinstance(results, (str, bytes))
        or not results
    ):
        raise AssertionError(f"Stagehand original result has no results: {raw!r}")
    first = results[0]
    if not isinstance(first, Mapping):
        raise AssertionError(
            f"Stagehand original result entry is not a mapping: {first!r}"
        )
    output = first.get("output")
    if not isinstance(output, Mapping):
        raise AssertionError(f"Stagehand original output is not a mapping: {first!r}")
    score = _number(first.get("score"))
    if score is None:
        score = 1.0 if output.get("_success") is True else 0.0
    logs = output.get("logs")
    log_count = (
        len(logs)
        if isinstance(logs, Sequence) and not isinstance(logs, (str, bytes))
        else 0
    )
    final_url = _extract_original_final_url(output)
    duration = _metric_seconds(output, "total_ms")
    if duration is None:
        duration = fallback_duration_sec
    return {
        "framework": "stagehand-evals",
        "task_id": task,
        "score": score,
        "success": output.get("_success") is True,
        "final_result": final_url,
        "final_url": final_url,
        "num_steps": max(log_count, 1),
        "logs_count": log_count,
        "screenshots_b64": [],
        "duration_sec": duration,
        "summary": raw.get("summary"),
    }


def _extract_original_final_url(output: Mapping[str, Any]) -> str:
    for key in ("observations", "finalUrl", "currentUrl", "url"):
        value = output.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value
    log_url = _extract_url_from_logs(output.get("logs"))
    if log_url is not None:
        return log_url
    raise AssertionError(f"Stagehand original output has no final URL: {output!r}")


def _extract_url_from_logs(logs: object) -> str | None:
    if not isinstance(logs, Sequence) or isinstance(logs, (str, bytes)):
        return None
    for item in reversed(logs):
        url = _find_http_url(item)
        if url is not None:
            return url
    return None


def _find_http_url(value: object) -> str | None:
    if isinstance(value, str):
        return value if value.startswith("http") else None
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        direct = mapping.get("url")
        if isinstance(direct, str) and direct.startswith("http"):
            return direct
        for child in mapping.values():
            found = _find_http_url(child)
            if found is not None:
                return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found = _find_http_url(child)
            if found is not None:
                return found
    return None


def _metric_seconds(output: Mapping[str, Any], key: str) -> float | None:
    metrics = output.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    metric = metrics.get(key)
    if not isinstance(metric, Mapping):
        return None
    value = _number(metric.get("value"))
    return None if value is None else value / 1000.0


def _run_benchflow(
    native_task: Path,
    jobs_dir: Path,
    *,
    agent: str,
    model: str,
    sandbox: str,
    run_fn: RunFn,
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
        "--model",
        model,
        "--sandbox",
        sandbox,
        "--jobs-dir",
        str(jobs_dir),
        "--concurrency",
        "1",
        "--json",
    ]
    result = run_fn(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "BenchFlow Stagehand run failed with rc "
            f"{result.returncode}\nCommand: {cmd}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise AssertionError("BenchFlow eval JSON was not an object")
    return payload


def _single_result_json(jobs_dir: Path) -> Path:
    matches = sorted(jobs_dir.glob("*/*/result.json"))
    if len(matches) != 1:
        tree = "\n".join(str(path) for path in sorted(jobs_dir.rglob("*")))
        raise AssertionError(
            f"expected one BenchFlow result.json, found {len(matches)}\n{tree}"
        )
    return matches[0]


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _load_stagehand_descriptor(task_dir: Path) -> dict[str, Any]:
    path = task_dir / "stagehand-task.json"
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise AssertionError(f"Stagehand descriptor is not a JSON object: {path}")
    return payload


def _compare(
    original: Mapping[str, Any],
    bench_result: Mapping[str, Any],
    artifact: Mapping[str, Any],
    verifier_artifact: Mapping[str, Any],
    docker_snapshot: Mapping[str, Any],
    *,
    bench_eval: Mapping[str, Any],
    expected_agent: str,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    # The BenchFlow eval must COMPLETE cleanly: that is an infra contract, not a
    # parity question, so it stays a hard error (mirrors the browser-use
    # harness's _assert_eval_report). Everything past this point is an
    # original-vs-BenchFlow comparison: a legitimate agent failure (anti-bot
    # 403, no trace, reward 0.0 from the now-fixed verifier) is an honest
    # DIVERGENCE that we RECORD as evidence instead of crashing.
    _assert_eval_report(bench_eval, expected_agent=expected_agent)

    original_score = float(original["score"])
    raw_bench_reward = _number(_mapping(bench_result.get("rewards")).get("reward"))
    if raw_bench_reward is None:
        # No reward at all means the eval did not actually score the task; that
        # is an infra failure, not a divergence, and must stay a hard error.
        raise AssertionError(f"BenchFlow result has no numeric reward: {bench_result}")
    bench_reward = float(raw_bench_reward)
    original_url = str(original["final_url"])
    # Use the reward from result.json even when no trace exists; the URL may be
    # absent (no trace) without aborting the comparison.
    bench_url = _stagehand_current_url(artifact, verifier_artifact)

    trajectory = _mapping(bench_result.get("trajectory_summary"))
    tool_calls = int(
        bench_result.get("n_tool_calls") or trajectory.get("tool_call_steps") or 0
    )
    steps = _sequence_or_empty(artifact.get("steps"))
    screenshots = _sequence_or_empty(artifact.get("screenshots_b64"))
    trajectory_steps = int(trajectory.get("steps") or 0)

    # Collect divergence notes. Any non-empty note flips the summary to a
    # recorded divergence (parity-recorded -> verdict parity-divergent), with
    # the original reward, the BenchFlow reward, and the reward delta kept as
    # honest evidence rather than raising AssertionError.
    divergences: list[str] = []
    if original_score != bench_reward:
        divergences.append(
            f"reward-mismatch: original={original_score} benchflow={bench_reward}"
        )
    if not _url_satisfies_success_check(original_url, descriptor):
        divergences.append(
            f"original-url-fails-success-check: {original_url!r}"
        )
    if bench_url is None:
        divergences.append("benchflow-no-trace: BenchFlow artifact has no current URL")
    elif not _url_satisfies_success_check(bench_url, descriptor):
        divergences.append(f"benchflow-url-fails-success-check: {bench_url!r}")
    if bench_result.get("agent") != expected_agent:
        divergences.append(
            f"unexpected-benchflow-agent: {bench_result.get('agent')}"
        )
    if bench_result.get("error") is not None:
        divergences.append(f"benchflow-error: {bench_result['error']}")
    if bench_result.get("verifier_error") is not None:
        divergences.append(f"benchflow-verifier-error: {bench_result['verifier_error']}")
    if trajectory_steps < 3:
        divergences.append("benchflow-thin-trajectory")
    if tool_calls < 1:
        divergences.append("benchflow-no-tool-calls")
    if not steps:
        divergences.append("benchflow-no-trace: Stagehand artifact has no steps")
    if not screenshots:
        divergences.append(
            "benchflow-no-trace: Stagehand artifact has no screenshots"
        )

    reward_delta = abs(bench_reward - original_score)
    summary: dict[str, Any] = {
        "ok": not divergences,
        "task_id": original["task_id"],
        "original": {
            "score": original_score,
            "num_steps": original["num_steps"],
            "logs_count": original["logs_count"],
            "screenshots_b64": 0,
            "final_url": original_url,
        },
        "benchflow": {
            "reward": bench_reward,
            "agent": bench_result.get("agent"),
            "trajectory_steps": trajectory_steps,
            "tool_calls": tool_calls,
            "artifact_steps": len(steps),
            "screenshots_b64": len(screenshots),
            "final_url": bench_url,
        },
        "cleanup": {
            "docker_available": docker_snapshot["available"],
            "benchflow_containers": len(docker_snapshot["containers"]),
            "benchflow_networks": len(docker_snapshot["networks"]),
        },
        "benchflow_eval": dict(bench_eval),
    }
    if divergences:
        summary["divergence"] = {
            "status": "divergent",
            "notes": divergences,
            "original_reward": original_score,
            "benchflow_reward": bench_reward,
            "reward_delta": reward_delta,
        }
    return summary


def _semantic_final_result(url: str, descriptor: Mapping[str, Any]) -> str:
    success_check = _mapping(descriptor.get("success_check"))
    check_type = success_check.get("type")
    expected = success_check.get("value")
    if check_type in {"url_exact", "url_contains"} and isinstance(expected, str):
        return f"{check_type}:{expected}"
    return url


def _url_satisfies_success_check(url: str, descriptor: Mapping[str, Any]) -> bool:
    success_check = _mapping(descriptor.get("success_check"))
    check_type = success_check.get("type")
    expected = success_check.get("value")
    if not isinstance(expected, str):
        return bool(url)
    if check_type == "url_exact":
        return url == expected
    if check_type == "url_contains":
        return expected in url
    return bool(url)


def _stagehand_current_url(
    artifact: Mapping[str, Any], verifier_artifact: Mapping[str, Any]
) -> str | None:
    """Return the BenchFlow current URL, or ``None`` when no trace produced one.

    A genuine agent failure (e.g. an anti-bot 403 that blocks the browser)
    leaves no trace and hence no current URL. That is an honest divergence the
    caller RECORDS as evidence, so this returns ``None`` instead of raising.
    """

    value = artifact.get("stagehand_current_url") or verifier_artifact.get(
        "current_url"
    )
    if isinstance(value, str) and value.startswith("http"):
        return value
    return None


def _assert_eval_report(payload: Mapping[str, Any], *, expected_agent: str) -> None:
    if payload.get("status") != "completed" or payload.get("ok") is not True:
        raise AssertionError(f"BenchFlow eval JSON did not complete cleanly: {payload}")
    result = _mapping(payload.get("result"))
    summary = _mapping(payload.get("summary"))
    if int(result.get("total") or 0) != 1 or int(summary.get("total") or 0) != 1:
        raise AssertionError(f"BenchFlow eval JSON expected one task: {payload}")
    if int(result.get("errored") or 0) or int(result.get("verifier_errored") or 0):
        raise AssertionError(f"BenchFlow eval result had errors: {payload}")
    if int(summary.get("errored") or 0) or int(summary.get("verifier_errored") or 0):
        raise AssertionError(f"BenchFlow eval summary had errors: {payload}")
    if summary.get("agent") != expected_agent:
        raise AssertionError(f"BenchFlow eval summary agent mismatch: {payload}")
    if int(summary.get("total_trajectory_steps") or 0) < 3:
        raise AssertionError(f"BenchFlow eval summary is trace-thin: {payload}")
    if not payload.get("summary_path"):
        raise AssertionError(f"BenchFlow eval JSON missing summary_path: {payload}")


def _artifact_manifest(descriptor: Mapping[str, Any]) -> list[dict[str, Any]]:
    success_check = _mapping(descriptor.get("success_check"))
    expected_url = success_check.get("value")
    check_type = success_check.get("type")
    url_requirement: dict[str, Any] = {
        "id": "stagehand-current-url",
        "source": "artifact",
        "path": "stagehand_current_url",
        "kind": "field",
    }
    if isinstance(expected_url, str) and check_type == "url_contains":
        url_requirement["contains"] = expected_url
    elif isinstance(expected_url, str):
        url_requirement["equals"] = expected_url
    else:
        url_requirement["required"] = True
    return [
        {
            "id": "stagehand-trace-steps",
            "source": "artifact",
            "path": "steps",
            "kind": "sequence",
            "min_count": 1,
        },
        {
            "id": "stagehand-screenshots",
            "source": "artifact",
            "path": "screenshots_b64",
            "kind": "sequence",
            "min_count": 1,
        },
        url_requirement,
        {
            "id": "stagehand-trajectory-steps",
            "source": "benchflow",
            "path": "trajectory_summary.steps",
            "kind": "numeric",
            "numeric_min": 3,
        },
    ]


def _benchflow_docker_resources(*, run_fn: RunFn) -> dict[str, Any]:
    if shutil.which("docker") is None:
        return {"available": False, "containers": [], "networks": []}
    return {
        "available": True,
        "containers": _docker_ids(["docker", "container", "ls", "-aq"], run_fn=run_fn),
        "networks": _docker_ids(["docker", "network", "ls", "-q"], run_fn=run_fn),
    }


def _leaked_benchflow_resources(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    """Resources THIS run created and failed to clean up.

    A resource counts as leaked only if it is present after the run but was not
    present before it. Pre-existing or concurrently-running benchflow-owned
    containers/networks are tolerated so an unrelated container cannot turn a
    clean run into a false leak failure.
    """

    before_containers = set(before.get("containers") or [])
    before_networks = set(before.get("networks") or [])
    return {
        "available": after.get("available", False),
        "containers": sorted(
            cid
            for cid in (after.get("containers") or [])
            if cid not in before_containers
        ),
        "networks": sorted(
            nid for nid in (after.get("networks") or []) if nid not in before_networks
        ),
    }


def _docker_ids(cmd: list[str], *, run_fn: RunFn) -> list[str]:
    result = run_fn(
        [*cmd, "--filter", "label=benchflow.owned=true"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"Docker listing failed for {cmd}")
    return sorted(line for line in result.stdout.splitlines() if line)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence_or_empty(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return []


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


if __name__ == "__main__":
    main()
