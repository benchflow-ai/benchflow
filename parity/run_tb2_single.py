"""Run Terminal-Bench 2.0 single-turn — supports resume (skips completed tasks).

Improvements over naive runner:
- Concurrency 4 (not 8) to avoid Docker network exhaustion
- Docker network cleanup after each task
- Skip tasks that already have rewards
"""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

from benchflow.sdk import SDK

CONCURRENCY = 4
JOBS_DIR = "parity/terminal-bench-2.0/single-turn"


def get_done_tasks() -> set[str]:
    """Find tasks that already have a result (completed OR errored on Daytona)."""
    done = set()
    results_dir = Path(JOBS_DIR)
    if not results_dir.exists():
        return done
    for rfile in results_dir.rglob("result.json"):
        try:
            r = json.loads(rfile.read_text())
            finished = r.get("finished_at", "")
            # Skip if completed with rewards
            if r.get("rewards") is not None:
                done.add(r["task_name"])
            # Also skip if errored on Daytona (after the Daytona run started)
            # to avoid infinite retries
            elif r.get("error") and finished > "2026-03-23 04:57":
                done.add(r["task_name"])
        except Exception:
            pass
    return done


def prune_docker():
    """Clean up stopped containers and unused networks."""
    try:
        subprocess.run(
            ["docker", "container", "prune", "-f"],
            capture_output=True, timeout=30,
        )
        subprocess.run(
            ["docker", "network", "prune", "-f"],
            capture_output=True, timeout=30,
        )
    except Exception:
        pass


async def run_task(sdk, task_dir, api_key, jobs_dir):
    try:
        result = await sdk.run(
            task_path=task_dir,
            agent="claude-agent-acp",
            model="claude-haiku-4-5-20251001",
            prompts=[None],  # instruction.md only
            agent_env={"ANTHROPIC_API_KEY": api_key},
            jobs_dir=jobs_dir,
            environment=os.environ.get("BENCHFLOW_ENV", "docker"),
        )
        reward = result.rewards.get("reward") if result.rewards else None
        status = "PASS" if reward == 1 else ("FAIL" if reward is not None else "ERR")
        err = f" ({result.error[:60]})" if result.error else ""
        print(f"  [{status}] {task_dir.name} (tools={result.n_tool_calls}){err}", flush=True)
        return {
            "task": task_dir.name,
            "reward": reward,
            "n_tool_calls": result.n_tool_calls,
            "n_prompts": result.n_prompts,
            "error": result.error,
        }
    except Exception as e:
        print(f"  [ERR] {task_dir.name}: {e}", flush=True)
        return {"task": task_dir.name, "reward": None, "error": str(e)}
    finally:
        # Cleanup after each task to free Docker resources
        prune_docker()


async def main():
    sdk = SDK()
    api_key = os.environ["ANTHROPIC_API_KEY"]
    tasks_dir = Path(".ref/terminal-bench-2")
    all_task_dirs = sorted(
        [d for d in tasks_dir.iterdir() if d.is_dir() and (d / "task.toml").exists()]
    )

    completed = get_done_tasks()
    task_dirs = [d for d in all_task_dirs if d.name not in completed]

    Path(JOBS_DIR).mkdir(parents=True, exist_ok=True)

    # Prune before start
    prune_docker()

    print(f"TB2 single-turn: {len(all_task_dirs)} total, {len(completed)} already done, {len(task_dirs)} to run")
    print(f"Concurrency: {CONCURRENCY}")
    start = time.time()

    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded(td):
        async with sem:
            return await run_task(sdk, td, api_key, JOBS_DIR)

    results = await asyncio.gather(*[bounded(td) for td in task_dirs])

    elapsed = time.time() - start
    solved = sum(1 for r in results if r.get("reward") == 1)
    errors = sum(1 for r in results if r.get("error"))

    print(f"\n=== THIS RUN ===")
    if results:
        print(f"Solved: {solved}/{len(results)} ({100 * solved / len(results):.1f}%)")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed / 60:.1f} min")

    print(f"\n=== CUMULATIVE ===")
    all_results = {}
    for rfile in Path(JOBS_DIR).rglob("result.json"):
        try:
            r = json.loads(rfile.read_text())
            task = r["task_name"]
            if task not in all_results or (r.get("rewards") is not None):
                all_results[task] = r
        except Exception:
            pass

    total = len(all_results)
    total_solved = sum(1 for r in all_results.values() if r.get("rewards") and r["rewards"].get("reward") == 1.0)
    total_failed = sum(1 for r in all_results.values() if r.get("rewards") and r["rewards"].get("reward") == 0.0)
    total_errors = sum(1 for r in all_results.values() if r.get("error") and r.get("rewards") is None)

    print(f"Total tasks attempted: {total}/{len(all_task_dirs)}")
    print(f"Solved: {total_solved}")
    print(f"Failed: {total_failed}")
    print(f"Errors: {total_errors}")
    if total > 0:
        print(f"Score (solved/total): {total_solved}/{len(all_task_dirs)} ({100 * total_solved / len(all_task_dirs):.1f}%)")
        print(f"Score (solved/completed): {total_solved}/{total_solved + total_failed} ({100 * total_solved / max(1, total_solved + total_failed):.1f}%)")

    summary = {
        "benchmark": "terminal-bench-2.0",
        "agent": "claude-agent-acp",
        "model": "claude-haiku-4-5-20251001",
        "mode": "single-turn",
        "total_tasks": len(all_task_dirs),
        "attempted": total,
        "solved": total_solved,
        "failed": total_failed,
        "errors": total_errors,
        "elapsed_sec": elapsed,
        "results": list(all_results.values()),
    }
    (Path(JOBS_DIR) / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved to {JOBS_DIR}/summary.json")


if __name__ == "__main__":
    asyncio.run(main())
