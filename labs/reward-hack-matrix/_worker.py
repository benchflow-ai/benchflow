#!/usr/bin/env python3
"""Long-lived sweep worker — one per benchflow version.

Reads NDJSON trial requests from stdin and emits NDJSON result lines on
stdout. The benchflow SDK is imported **once** at startup, then each trial
runs as an asyncio coroutine under a local ``asyncio.Semaphore``.

This replaces the old subprocess-per-trial design in ``run_matrix.py``
which OOM'd a ~8 GB dev container at ``--concurrency 64`` because each
subprocess re-imported the full benchflow + harbor + daytona SDK (~300–400
MB each × 64 = ~20 GB peak).

Protocol
--------
Input (one JSON object per line on stdin)::

    {"id": "<cell_id>", "task_path": "...", "jobs_dir": "...",
     "trial_name": "...", "environment": "daytona"}

Output (one JSON object per line on stdout)::

    {"id": "<cell_id>", "reward": 1.0, "error": null,
     "verifier_error": null, "benchflow_version": "0.2.0"}

    {"id": "<cell_id>", "reward": null,
     "error": "ExceptionType: message", ...}

A single line ``{"__ready__": true, "benchflow_version": "..."}`` is sent
as soon as the SDK is imported so the orchestrator can wait for worker
startup before fanning out trials.

Concurrency is bounded by the ``--concurrency`` argument — the orchestrator
should set this to ``daytona_cap / num_workers`` (e.g. 32 when running 2
workers against a 64-sandbox Daytona cap).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback


async def _stream_requests(stdin: asyncio.StreamReader):
    while True:
        line = await stdin.readline()
        if not line:
            return
        line = line.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"__error__": f"bad input line: {exc}"})


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


async def _run_trial(sdk, req: dict) -> dict:
    """Execute one trial. Returns a result dict with the original id."""
    import benchflow

    try:
        result = await sdk.run(
            task_path=req["task_path"],
            agent="oracle",
            environment=req.get("environment", "daytona"),
            jobs_dir=req["jobs_dir"],
            trial_name=req["trial_name"],
        )
        reward = None
        rewards = getattr(result, "rewards", None)
        if isinstance(rewards, dict):
            reward = rewards.get("reward")
        return {
            "id": req["id"],
            "benchflow_version": getattr(benchflow, "__version__", "unknown"),
            "reward": reward,
            "error": getattr(result, "error", None),
            "verifier_error": getattr(result, "verifier_error", None),
        }
    except Exception as exc:
        tb = traceback.format_exc()
        return {
            "id": req["id"],
            "benchflow_version": getattr(benchflow, "__version__", "unknown"),
            "reward": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": tb[-1500:],
        }


async def _main_async(concurrency: int) -> int:
    # Import once per worker lifetime — the whole point of this design.
    import benchflow
    from benchflow import SDK

    sdk = SDK()

    _emit(
        {
            "__ready__": True,
            "benchflow_version": getattr(benchflow, "__version__", "unknown"),
            "concurrency": concurrency,
        }
    )

    sem = asyncio.Semaphore(concurrency)

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    in_flight: set[asyncio.Task] = set()
    write_lock = asyncio.Lock()

    async def _handle(req: dict) -> None:
        async with sem:
            result = await _run_trial(sdk, req)
        async with write_lock:
            _emit(result)

    async for req in _stream_requests(reader):
        if "__error__" in req:
            continue
        task = asyncio.create_task(_handle(req))
        in_flight.add(task)
        task.add_done_callback(in_flight.discard)

    # stdin closed — wait for outstanding trials to finish
    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)

    _emit({"__done__": True})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    return asyncio.run(_main_async(args.concurrency))


if __name__ == "__main__":
    sys.exit(main())
