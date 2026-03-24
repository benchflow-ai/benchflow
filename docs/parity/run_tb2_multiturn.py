"""Run Terminal-Bench 2.0 with multi-turn recheck prompt (Haiku 4.5).

Supports resume — skips tasks that already have rewards.
"""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

from benchflow.sdk import SDK

CONCURRENCY = 4
JOBS_DIR = "docs/parity/terminal-bench-2.0/multi-turn-haiku"


def get_done_tasks() -> set[str]:
    """Find tasks that already have a result with rewards."""
    done = set()
    results_dir = Path(JOBS_DIR)
    if not results_dir.exists():
        return done
    for rfile in results_dir.rglob("result.json"):
        try:
            r = json.loads(rfile.read_text())
            if r.get("rewards") is not None:
                done.add(r["task_name"])
        except Exception:
            pass
    return done


def prune_docker():
    """Clean up stopped containers and unused networks."""
    try:
        subprocess.run(["docker", "container", "prune", "-f"], capture_output=True, timeout=30)
        subprocess.run(["docker", "network", "prune", "-f"], capture_output=True, timeout=30)
    except Exception:
        pass


async def run_task(sdk, task_dir, api_key, jobs_dir, env_type, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            result = await sdk.run(
                task_path=task_dir,
                agent="claude-agent-acp",
                model="claude-haiku-4-5-20251001",
                prompts=[
                    None,  # instruction.md
                    "Review your solution. Check for errors, test it, and fix any issues.",
                ],
                agent_env={"ANTHROPIC_API_KEY": api_key},
                jobs_dir=jobs_dir,
                environment=env_type,
            )
            # Retry on install failures (Daytona npm timeouts)
            if result.error and "install failed" in result.error and attempt < max_retries:
                print(f"  [RETRY {attempt}/{max_retries}] {task_dir.name}: install failed, retrying...", flush=True)
                continue

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
            if attempt < max_retries and "install" in str(e).lower():
                print(f"  [RETRY {attempt}/{max_retries}] {task_dir.name}: {e}", flush=True)
                continue
            print(f"  [ERR] {task_dir.name}: {e}", flush=True)
            return {"task": task_dir.name, "reward": None, "error": str(e)}
        finally:
            if env_type == "docker":
                prune_docker()


async def main():
    sdk = SDK()
    api_key = os.environ["ANTHROPIC_API_KEY"]
    env_type = os.environ.get("BENCHFLOW_ENV", "docker")
    tasks_dir = Path(".ref/terminal-bench-2")
    all_task_dirs = sorted(
        [d for d in tasks_dir.iterdir() if d.is_dir() and (d / "task.toml").exists()]
    )

    done = get_done_tasks()
    task_dirs = [d for d in all_task_dirs if d.name not in done]

    Path(JOBS_DIR).mkdir(parents=True, exist_ok=True)
    if env_type == "docker":
        prune_docker()

    print(f"TB2 multi-turn (Haiku): {len(all_task_dirs)} total, {len(done)} done, {len(task_dirs)} to run")
    print(f"Concurrency: {CONCURRENCY}, Environment: {env_type}")
    start = time.time()

    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded(td):
        async with sem:
            return await run_task(sdk, td, api_key, JOBS_DIR, env_type)

    results = await asyncio.gather(*[bounded(td) for td in task_dirs])

    elapsed = time.time() - start
    solved = sum(1 for r in results if r.get("reward") == 1)
    errors = sum(1 for r in results if r.get("error"))

    print(f"\n=== RESULTS ===")
    if results:
        print(f"Solved: {solved}/{len(results)} ({100 * solved / len(results):.1f}%)")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed / 60:.1f} min")

    # Cumulative
    all_results = {}
    for rfile in Path(JOBS_DIR).rglob("result.json"):
        try:
            r = json.loads(rfile.read_text())
            task = r["task_name"]
            if task not in all_results or (r.get("rewards") is not None):
                all_results[task] = r
        except Exception:
            pass

    total_solved = sum(1 for r in all_results.values() if r.get("rewards") and r["rewards"].get("reward") == 1.0)
    total_failed = sum(1 for r in all_results.values() if r.get("rewards") and r["rewards"].get("reward") == 0.0)
    total_errors = sum(1 for r in all_results.values() if r.get("error") and r.get("rewards") is None)

    print(f"\n=== CUMULATIVE ===")
    print(f"Solved: {total_solved}/{len(all_task_dirs)}")
    print(f"Failed: {total_failed}")
    print(f"Errors: {total_errors}")

    summary = {
        "benchmark": "terminal-bench-2.0",
        "agent": "claude-agent-acp",
        "model": "claude-haiku-4-5-20251001",
        "mode": "multi-turn-recheck",
        "total_tasks": len(all_task_dirs),
        "solved": total_solved,
        "failed": total_failed,
        "errors": total_errors,
        "elapsed_sec": elapsed,
        "results": list(all_results.values()),
    }
    (Path(JOBS_DIR) / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved to {JOBS_DIR}/summary.json")


if __name__ == "__main__":
    asyncio.run(main())
