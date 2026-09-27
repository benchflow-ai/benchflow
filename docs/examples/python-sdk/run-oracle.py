#!/usr/bin/env python3
"""Run one task with the oracle agent and read the typed result.

The oracle runs the task's own solution (solution/solve.sh), so this needs no
model credentials: it checks that the sandbox, the task and its verifier work.

Requirements: Docker running (default), or DAYTONA_API_KEY and the
sandbox-daytona extra for --sandbox daytona.

Usage:
  uv run python docs/examples/python-sdk/run-oracle.py
  uv run python docs/examples/python-sdk/run-oracle.py --sandbox daytona --task path/to/task
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import benchflow as bf

HELLO_WORLD = Path(__file__).resolve().parents[3] / "tests/examples/hello-world-task"


async def main(args: argparse.Namespace) -> int:
    result = await bf.run(
        bf.RolloutConfig(
            task_path=Path(args.task),
            agent="oracle",
            environment=args.sandbox,
            jobs_dir=args.jobs_dir,
        )
    )
    print(result)  # RolloutResult(task=..., OK, rewards=..., ...)
    print(f"reward:    {result.reward}")  # rewards["reward"], None if unscored
    print(f"passed:    {result.passed}")  # the scoring outcome is a pass
    print(f"outcome:   {result.score_outcome}")  # passed / failed / errored / ...
    print(f"artifacts: {result.rollout_dir}")  # result.json, trajectory/, verifier/
    if result.error or result.verifier_error:
        print(
            f"error:     {result.error_category}: {result.error or result.verifier_error}"
        )
    return 0 if result.passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sandbox", default="docker", choices=["docker", "daytona"])
    parser.add_argument("--task", default=str(HELLO_WORLD))
    parser.add_argument("--jobs-dir", default="jobs/python-sdk-examples")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
