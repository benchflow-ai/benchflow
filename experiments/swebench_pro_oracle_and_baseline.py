"""SWE-bench Pro: oracle validation + Harbor baseline.

Two experiments on the 4 testable SWE-bench Pro tasks:

  1) Oracle validation — runs gold solution (solve.sh), verifies reward=1.0
     for all 4 tasks. Confirms the --rootdir=/tests fix resolved qutebrowser
     and openlibrary failures.

  2) Harbor baseline — single-round agent evaluation (no progressive
     disclosure). Runs the same 4 tasks with a real agent to establish
     baseline pass rates for comparison with progressive disclosure.

Usage:
    # Oracle only (no API key needed):
    python experiments/swebench_pro_oracle_and_baseline.py --oracle-only

    # Full run (oracle + baseline):
    GEMINI_API_KEY=... python experiments/swebench_pro_oracle_and_baseline.py

    # Customize agent/model/backend:
    GEMINI_API_KEY=... python experiments/swebench_pro_oracle_and_baseline.py \
        --agent gemini --model gemini-3.1-pro-preview --backend daytona

Results → experiments/swebench-pro-results.csv
"""

import argparse
import asyncio
import csv
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "src"))

import benchflow as bf
from benchflow.trial import Scene, TrialConfig

SWEBENCH_PRO_ROOT = Path(__file__).resolve().parents[1] / ".ref" / "swebenchpro"

TASKS = [
    "instance_ansible__ansible-0ea40e09d1b35bcb69ff4d9cecf3d0defa4b36e8-v30a923fb5c164d6cd18280c02422f75e611e8fb2",
    "instance_flipt-io__flipt-02e21636c58e86c51119b63e0fb5ca7b813b07b1",
    "instance_internetarchive__openlibrary-00bec1e7c8f3272c469a58e1377df03f955ed478-v13642507b4fc1f8d234172bf8129942da2c2ca26",
    "instance_qutebrowser__qutebrowser-01d1d1494411380d97cac14614a829d3a69cecaf-v2ef375ac784985212b1805e1d0431dc8f1b3c171",
]

TASK_LABELS = {
    "instance_ansible": "ansible",
    "instance_flipt": "flipt",
    "instance_internet": "openlibrary",
    "instance_qutebrowser": "qutebrowser",
}

RESULTS_FILE = Path(__file__).parent / "swebench-pro-results.csv"
JOBS_DIR = Path("/tmp/swebench-pro-jobs")


def task_label(task_name: str) -> str:
    for prefix, label in TASK_LABELS.items():
        if task_name.startswith(prefix):
            return label
    return task_name[:30]


async def run_oracle(task_path: Path, backend: str) -> dict:
    """Run oracle (gold solution) on a single task. Returns result dict."""
    label = task_label(task_path.name)
    logger.info(f"[oracle] {label}: starting")
    t0 = time.time()

    config = TrialConfig(
        task_path=task_path,
        agent="oracle",
        environment=backend,
        sandbox_user="agent",
        jobs_dir=str(JOBS_DIR / "oracle"),
    )

    try:
        result = await bf.run(config)
        elapsed = time.time() - t0
        reward = None
        if result.rewards:
            reward = result.rewards.get("reward", result.rewards.get("exact_match"))
        logger.info(
            f"[oracle] {label}: reward={reward} "
            f"error={result.error!r} ({elapsed:.0f}s)"
        )
        return {
            "experiment": "oracle",
            "task": label,
            "task_full": task_path.name,
            "reward": reward,
            "error": result.error or "",
            "elapsed_s": round(elapsed, 1),
            "n_tool_calls": result.n_tool_calls,
        }
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"[oracle] {label}: crashed: {e}", exc_info=True)
        return {
            "experiment": "oracle",
            "task": label,
            "task_full": task_path.name,
            "reward": None,
            "error": str(e),
            "elapsed_s": round(elapsed, 1),
            "n_tool_calls": 0,
        }


async def run_baseline(
    task_path: Path, agent: str, model: str, backend: str
) -> dict:
    """Run single-round baseline (no progressive disclosure) on a single task."""
    label = task_label(task_path.name)
    logger.info(f"[baseline] {label}: starting ({agent}/{model})")
    t0 = time.time()

    config = TrialConfig(
        task_path=task_path,
        scenes=[Scene.single(agent=agent, model=model)],
        environment=backend,
        sandbox_user="agent",
        jobs_dir=str(JOBS_DIR / "baseline"),
    )

    try:
        result = await bf.run(config)
        elapsed = time.time() - t0
        reward = None
        if result.rewards:
            reward = result.rewards.get("reward", result.rewards.get("exact_match"))
        logger.info(
            f"[baseline] {label}: reward={reward} "
            f"tools={result.n_tool_calls} error={result.error!r} ({elapsed:.0f}s)"
        )
        return {
            "experiment": "baseline",
            "task": label,
            "task_full": task_path.name,
            "agent": agent,
            "model": model,
            "reward": reward,
            "error": result.error or "",
            "elapsed_s": round(elapsed, 1),
            "n_tool_calls": result.n_tool_calls,
        }
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"[baseline] {label}: crashed: {e}", exc_info=True)
        return {
            "experiment": "baseline",
            "task": label,
            "task_full": task_path.name,
            "agent": agent,
            "model": model,
            "reward": None,
            "error": str(e),
            "elapsed_s": round(elapsed, 1),
            "n_tool_calls": 0,
        }


async def main():
    parser = argparse.ArgumentParser(description="SWE-bench Pro oracle + baseline")
    parser.add_argument("--oracle-only", action="store_true", help="Skip baseline")
    parser.add_argument("--agent", default=os.environ.get("AGENT", "gemini"))
    parser.add_argument("--model", default=os.environ.get("MODEL", "gemini-3.1-pro-preview"))
    parser.add_argument("--backend", default=os.environ.get("BACKEND", "docker"))
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()

    task_paths = [SWEBENCH_PRO_ROOT / t for t in TASKS]
    missing = [p for p in task_paths if not p.exists()]
    if missing:
        logger.error(f"Missing tasks: {[p.name for p in missing]}")
        sys.exit(1)

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    # Phase 1: Oracle validation (all 4 tasks in parallel)
    logger.info("=" * 60)
    logger.info("Phase 1: Oracle validation (gold solution)")
    logger.info("=" * 60)

    sem = asyncio.Semaphore(args.concurrency)

    async def bounded_oracle(tp):
        async with sem:
            return await run_oracle(tp, args.backend)

    oracle_results = await asyncio.gather(
        *[bounded_oracle(tp) for tp in task_paths]
    )
    results.extend(oracle_results)

    oracle_pass = sum(1 for r in oracle_results if r["reward"] == 1.0)
    logger.info(f"Oracle: {oracle_pass}/{len(oracle_results)} passed")

    if oracle_pass < len(oracle_results):
        failed = [r["task"] for r in oracle_results if r["reward"] != 1.0]
        logger.warning(f"Oracle failures: {failed}")

    # Phase 2: Baseline (single-round agent)
    if not args.oracle_only:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Phase 2: Baseline ({args.agent}/{args.model})")
        logger.info("=" * 60)

        async def bounded_baseline(tp):
            async with sem:
                return await run_baseline(tp, args.agent, args.model, args.backend)

        baseline_results = await asyncio.gather(
            *[bounded_baseline(tp) for tp in task_paths]
        )
        results.extend(baseline_results)

        baseline_pass = sum(1 for r in baseline_results if r.get("reward", 0) == 1.0)
        logger.info(f"Baseline: {baseline_pass}/{len(baseline_results)} passed")

    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)

        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Results written to {RESULTS_FILE}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<12} {'Task':<15} {'Reward':<10} {'Tools':<8} {'Time':<8} {'Error'}")
    print("-" * 80)
    for r in results:
        reward_str = str(r.get("reward", "—"))
        tools_str = str(r.get("n_tool_calls", "—"))
        time_str = f"{r.get('elapsed_s', 0):.0f}s"
        error_str = (r.get("error", "") or "")[:40]
        print(f"{r['experiment']:<12} {r['task']:<15} {reward_str:<10} {tools_str:<8} {time_str:<8} {error_str}")


if __name__ == "__main__":
    asyncio.run(main())
