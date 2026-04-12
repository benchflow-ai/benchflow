#!/usr/bin/env python3
"""Inner runner: executes one version of benchflow against the attack task.

Invoked by run_comparison.py once per pinned venv. Prints exactly one JSON
line to stdout (version, reward, error) so the parent process can parse it
without worrying about pip or Docker noise on stderr.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


async def _run() -> dict:
    import benchflow
    from benchflow import SDK

    task_path = HERE / "attack_task"
    sdk = SDK()

    result = await sdk.run(
        task_path=str(task_path),
        agent="oracle",
        jobs_dir=os.environ.get("BENCHJACK_JOBS_DIR", str(HERE / ".jobs")),
        trial_name=os.environ.get("BENCHJACK_TRIAL_NAME", "attack"),
    )

    reward = None
    rewards = getattr(result, "rewards", None)
    if isinstance(rewards, dict):
        reward = rewards.get("reward")

    return {
        "version": getattr(benchflow, "__version__", "unknown"),
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
                    "version": None,
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
