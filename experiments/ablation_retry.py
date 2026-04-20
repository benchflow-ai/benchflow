"""Retry errored/missing trials from the full ablation run (concurrency=64).

Parsed from /tmp/ablation-full-run-64.log. Runs at concurrency=16 to avoid
Daytona SSH contention. Appends results to ablation-retry-results.csv.

Usage:
  source /workspace/scripts/agent-env.sh
  ABLATION_MODEL=gemini-3.1-flash-lite-preview ABLATION_AGENT=gemini \
    uv run python experiments/ablation_retry.py
"""
import asyncio
import csv
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from experiments.reviewer_ablation import (
    TB2_ROOT,
    run_baseline,
    run_reviewer,
)

RETRY_CSV = Path(__file__).parent / "ablation-retry-results.csv"
MODEL = os.environ.get("ABLATION_MODEL", "gemini-3.1-flash-lite-preview")
AGENT = os.environ.get("ABLATION_AGENT", "gemini")

# 77 trials that errored or were missing from the concurrency=64 run.
# 6 baseline, 41 reviewer, 30 reviewer+spec (including 1 fully missing).
RETRY_LIST: list[tuple[str, str]] = [
    ("build-cython-ext", "reviewer"),
    ("build-pmars", "reviewer"),
    ("build-pmars", "reviewer+spec"),
    ("build-pov-ray", "reviewer"),
    ("chess-best-move", "reviewer"),
    ("chess-best-move", "reviewer+spec"),
    ("circuit-fibsqrt", "reviewer"),
    ("code-from-image", "reviewer+spec"),
    ("count-dataset-tokens", "reviewer"),
    ("count-dataset-tokens", "reviewer+spec"),
    ("crack-7z-hash", "baseline"),
    ("crack-7z-hash", "reviewer"),
    ("crack-7z-hash", "reviewer+spec"),
    ("custom-memory-heap-crash", "reviewer"),
    ("db-wal-recovery", "reviewer"),
    ("db-wal-recovery", "reviewer+spec"),
    ("dna-assembly", "reviewer"),
    ("dna-assembly", "reviewer+spec"),
    ("dna-insert", "reviewer"),
    ("extract-elf", "reviewer+spec"),
    ("feal-differential-cryptanalysis", "reviewer+spec"),
    ("feal-linear-cryptanalysis", "reviewer"),
    ("filter-js-from-html", "reviewer"),
    ("fix-code-vulnerability", "reviewer"),
    ("fix-ocaml-gc", "reviewer"),
    ("fix-ocaml-gc", "reviewer+spec"),
    ("git-multibranch", "reviewer"),
    ("git-multibranch", "reviewer+spec"),
    ("hf-model-inference", "reviewer"),
    ("kv-store-grpc", "reviewer+spec"),
    ("large-scale-text-editing", "baseline"),
    ("large-scale-text-editing", "reviewer"),
    ("llm-inference-batching-scheduler", "reviewer"),
    ("log-summary-date-ranges", "reviewer"),
    ("log-summary-date-ranges", "reviewer+spec"),
    ("mailman", "reviewer"),
    ("mailman", "reviewer+spec"),
    ("make-doom-for-mips", "baseline"),
    ("make-doom-for-mips", "reviewer"),
    ("make-doom-for-mips", "reviewer+spec"),
    ("merge-diff-arc-agi-task", "reviewer"),
    ("modernize-scientific-stack", "reviewer+spec"),
    ("mteb-retrieve", "reviewer"),
    ("mteb-retrieve", "reviewer+spec"),
    ("multi-source-data-merger", "reviewer"),
    ("multi-source-data-merger", "reviewer+spec"),
    ("nginx-request-logging", "reviewer"),
    ("nginx-request-logging", "reviewer+spec"),
    ("overfull-hbox", "reviewer"),
    ("password-recovery", "baseline"),
    ("password-recovery", "reviewer"),
    ("path-tracing", "reviewer"),
    ("path-tracing", "reviewer+spec"),
    ("polyglot-c-py", "reviewer"),
    ("prove-plus-comm", "reviewer"),
    ("pytorch-model-recovery", "baseline"),
    ("pytorch-model-recovery", "reviewer"),
    ("pytorch-model-recovery", "reviewer+spec"),
    ("qemu-startup", "baseline"),
    ("qemu-startup", "reviewer+spec"),
    ("query-optimize", "reviewer+spec"),
    ("regex-chess", "reviewer"),
    ("reshard-c4-data", "reviewer"),
    ("rstan-to-pystan", "reviewer+spec"),
    ("sam-cell-seg", "reviewer"),
    ("sqlite-with-gcov", "reviewer+spec"),
    ("torch-pipeline-parallelism", "reviewer"),
    ("torch-tensor-parallelism", "reviewer"),
    ("torch-tensor-parallelism", "reviewer+spec"),
    ("train-fasttext", "reviewer"),
    ("train-fasttext", "reviewer+spec"),
    ("tune-mjcf", "reviewer"),
    ("tune-mjcf", "reviewer+spec"),
    ("vulnerable-secret", "reviewer"),
    ("vulnerable-secret", "reviewer+spec"),
    ("winning-avg-corewars", "reviewer"),
    ("winning-avg-corewars", "reviewer+spec"),
]

COLS = ["benchmark", "task", "condition", "model", "backend", "rounds",
        "reward", "wall_sec", "tool_calls", "error"]

_SEM = asyncio.Semaphore(16)


def _append_csv(row: dict) -> None:
    """Append a single row to the retry CSV (creates header if needed)."""
    write_header = not RETRY_CSV.exists()
    with open(RETRY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


async def _run_one(task_name: str, condition: str) -> dict:
    task_path = TB2_ROOT / task_name
    logger.info(f"RETRY  {task_name} / {condition}")

    if condition == "baseline":
        result = await run_baseline(task_path, task_name)
    else:
        result = await run_reviewer(task_path, task_name, condition)

    row = {
        "benchmark": "tb2",
        "task": task_name,
        "condition": condition,
        "model": MODEL,
        "backend": "daytona",
        **result,
    }
    logger.info(
        f"DONE   {task_name} / {condition} -> "
        f"reward={row['reward']} wall={row['wall_sec']}s tools={row['tool_calls']} err={row.get('error')}"
    )
    _append_csv(row)
    return row


async def _bounded(task_name: str, condition: str) -> dict:
    async with _SEM:
        return await _run_one(task_name, condition)


async def main() -> None:
    Path("/tmp/ablation-jobs").mkdir(parents=True, exist_ok=True)

    # Remove stale retry CSV if it exists
    if RETRY_CSV.exists():
        RETRY_CSV.unlink()

    logger.info(f"=== ABLATION RETRY: {len(RETRY_LIST)} trials, concurrency=16, agent={AGENT}, model={MODEL} ===")

    tasks = [_bounded(tn, cond) for tn, cond in RETRY_LIST]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = sum(1 for r in results if isinstance(r, dict) and not r.get("error"))
    errors = sum(1 for r in results if isinstance(r, dict) and r.get("error"))
    exceptions = sum(1 for r in results if isinstance(r, Exception))
    logger.info(f"=== RETRY COMPLETE: {successes} ok, {errors} errors, {exceptions} exceptions ===")
    logger.info(f"Results: {RETRY_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
