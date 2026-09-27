#!/usr/bin/env python3
"""Run several agent/model combinations on one task and compare them.

A plain script, no asyncio: bf.run_batch runs the configs with bounded
concurrency, prints each result as it finishes, and returns them in input
order. The comparison is written to CSV and JSONL (no pandas needed).

Each combination is --run AGENT[:MODEL]; the oracle needs no credentials, a
real agent needs its usual credentials (see `bench doctor`).

Usage:
  uv run python docs/examples/python-sdk/run-many.py
  uv run python docs/examples/python-sdk/run-many.py --sandbox daytona \\
      --run oracle --run claude-agent-acp:claude-haiku-4-5 --out results/compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import benchflow as bf

HELLO_WORLD = Path(__file__).resolve().parents[3] / "tests/examples/hello-world-task"


def main(args: argparse.Namespace) -> int:
    configs = []
    for spec in args.run or ["oracle"]:
        agent, _, model = spec.partition(":")
        configs.append(
            bf.RolloutConfig(
                task_path=Path(args.task),
                agent=agent,
                model=model or None,
                environment=args.sandbox,
                jobs_dir=args.jobs_dir,
            )
        )

    def progress(done: bf.Completed) -> None:
        r = done.result
        print(f"finished {args.run[done.index] if args.run else 'oracle'}: {r}")

    results = bf.run_batch(configs, concurrency=args.concurrency, on_result=progress)

    print(
        f"\n{results.n_passed}/{len(results)} passed, mean reward {results.mean_reward}"
    )
    for spec, r in zip(args.run or ["oracle"], results, strict=True):
        print(
            f"  {spec:40} reward={r.reward} tokens={r.total_tokens} dir={r.rollout_dir}"
        )
    out = Path(args.out)
    print(
        "wrote",
        results.to_csv(out.with_suffix(".csv")),
        "and",
        results.to_jsonl(out.with_suffix(".jsonl")),
    )
    return 0 if all(r.error is None for r in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sandbox", default="docker", choices=["docker", "daytona"])
    parser.add_argument("--task", default=str(HELLO_WORLD))
    parser.add_argument("--run", action="append", help="AGENT[:MODEL], repeatable")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--jobs-dir", default="jobs/python-sdk-examples")
    parser.add_argument("--out", default="jobs/python-sdk-examples/compare")
    raise SystemExit(main(parser.parse_args()))
