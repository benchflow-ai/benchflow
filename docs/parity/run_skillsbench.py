"""Run SkillsBench (excluding tasks needing external API keys)."""

import asyncio
import json
import os
import time
from pathlib import Path

from benchflow.sdk import SDK

CONCURRENCY = 8
EXCLUDE = {"scheduling-email-assistant", "mhc-layer-impl"}


async def run_task(sdk, task_dir, api_key, jobs_dir):
    try:
        result = await sdk.run(
            task_path=task_dir,
            agent="claude-agent-acp",
            model="claude-haiku-4-5-20251001",
            agent_env={"ANTHROPIC_API_KEY": api_key},
            jobs_dir=jobs_dir,
        )
        reward = result.rewards.get("reward") if result.rewards else None
        status = "PASS" if reward == 1 else "FAIL"
        print(f"  [{status}] {task_dir.name} (tools={result.n_tool_calls})", flush=True)
        return {
            "task": task_dir.name,
            "reward": reward,
            "n_tool_calls": result.n_tool_calls,
            "error": result.error,
        }
    except Exception as e:
        print(f"  [ERR] {task_dir.name}: {e}", flush=True)
        return {"task": task_dir.name, "reward": None, "error": str(e)}


async def main():
    sdk = SDK()
    api_key = os.environ["ANTHROPIC_API_KEY"]
    tasks_dir = Path(".ref/skillsbench/tasks")
    task_dirs = sorted(
        [
            d
            for d in tasks_dir.iterdir()
            if d.is_dir() and (d / "task.toml").exists() and d.name not in EXCLUDE
        ]
    )

    jobs_dir = "docs/parity/skillsbench"
    Path(jobs_dir).mkdir(parents=True, exist_ok=True)

    print(f"SkillsBench: {len(task_dirs)} tasks, concurrency={CONCURRENCY}")
    start = time.time()

    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded(td):
        async with sem:
            return await run_task(sdk, td, api_key, jobs_dir)

    results = await asyncio.gather(*[bounded(td) for td in task_dirs])

    elapsed = time.time() - start
    solved = sum(1 for r in results if r.get("reward") == 1)
    errors = sum(1 for r in results if r.get("error"))

    print(f"\n=== RESULTS ===")
    print(f"Solved: {solved}/{len(results)} ({100 * solved / len(results):.1f}%)")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed / 60:.1f} min")

    summary = {
        "benchmark": "skillsbench",
        "agent": "claude-agent-acp",
        "model": "claude-haiku-4-5-20251001",
        "mode": "single-turn",
        "total": len(results),
        "solved": solved,
        "errors": errors,
        "elapsed_sec": elapsed,
        "results": results,
    }
    (Path(jobs_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved to {jobs_dir}/summary.json")


if __name__ == "__main__":
    asyncio.run(main())
