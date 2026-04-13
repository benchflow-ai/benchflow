#!/usr/bin/env python3
"""Inner runner: executes one (version, benchmark, task, pattern) cell.

Invoked by run_matrix.py once per pinned venv × cell. Reads its arguments
from the environment so the orchestrator can compose simple subprocess
calls and parse a single JSON line off stdout.

Required env:
    RH_TASK_PATH         Absolute path to the task directory (Harbor format).
    RH_PATTERN_ID        Pattern label for logging only ("P1", "P7", ...).
    RH_BENCHMARK         Benchmark label ("skillsbench", ...).
    RH_VERSION_LABEL     Version label ("0.2.0", "0.2.1", "harbor-orig").
    RH_JOBS_DIR          Directory under which trial output goes.
    RH_TRIAL_NAME        Unique trial name.
    RH_ENVIRONMENT       "daytona" or "docker" (default daytona).

Stdout: exactly one JSON line with version, reward, error, verifier_error.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback


async def _run() -> dict:
    import benchflow
    from benchflow import SDK

    task_path = os.environ["RH_TASK_PATH"]
    jobs_dir = os.environ["RH_JOBS_DIR"]
    trial_name = os.environ["RH_TRIAL_NAME"]
    environment = os.environ.get("RH_ENVIRONMENT", "daytona")

    sdk = SDK()

    result = await sdk.run(
        task_path=task_path,
        agent="oracle",
        environment=environment,
        jobs_dir=jobs_dir,
        trial_name=trial_name,
    )

    reward = None
    rewards = getattr(result, "rewards", None)
    if isinstance(rewards, dict):
        reward = rewards.get("reward")

    return {
        "benchflow_version": getattr(benchflow, "__version__", "unknown"),
        "reward": reward,
        "error": getattr(result, "error", None),
        "verifier_error": getattr(result, "verifier_error", None),
    }


def main() -> int:
    try:
        payload = asyncio.run(_run())
    except Exception as exc:
        sys.stderr.write(traceback.format_exc())
        print(
            json.dumps(
                {
                    "benchflow_version": None,
                    "reward": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        )
        return 1

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
