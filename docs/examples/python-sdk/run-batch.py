#!/usr/bin/env python3
"""Run every task under a directory as one job, watch results arrive, export them.

Evaluation discovers the task directories, runs them with the given
concurrency and retries, and writes summary.json. stream() yields each task's
result as it finishes. The job records its config in <job_dir>/evaluation.json,
so an interrupted job can be finished from its directory with --resume.

Usage:
  uv run python docs/examples/python-sdk/run-batch.py --tasks-dir path/to/tasks
  uv run python docs/examples/python-sdk/run-batch.py --tasks-dir path/to/tasks \\
      --sandbox daytona --agent claude-agent-acp --model claude-haiku-4-5 --concurrency 4
  uv run python docs/examples/python-sdk/run-batch.py --resume jobs/python-sdk-batch/<job>
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib

import benchflow as bf


async def main(args: argparse.Namespace) -> int:
    if args.resume:
        evaluation = bf.Evaluation.resume(args.resume)
    else:
        if not args.tasks_dir:
            raise SystemExit("--tasks-dir is required unless --resume is given")
        evaluation = bf.Evaluation(
            tasks_dir=args.tasks_dir,
            jobs_dir=args.jobs_dir,
            config=bf.EvaluationConfig(
                agent=args.agent,
                model=args.model,
                environment=args.sandbox,
                concurrency=args.concurrency,
                retry=bf.RetryConfig(max_retries=1),
            ),
        )

    async with contextlib.aclosing(evaluation.stream()) as stream:
        async for name, result in stream:
            status = "PASS" if result.passed else result.score_outcome.upper()
            print(f"  {status:6} {name:30} reward={result.reward}")

    job = evaluation.result
    assert job is not None
    print(f"job {job.job_name}: {job.passed}/{job.total} passed, {job.errored} errored")
    print(f"pass rate {job.score:.2f}, mean reward {job.mean_reward}")
    print(f"artifacts, summary.json and evaluation.json under {job.job_dir}")
    print("wrote", job.to_csv(job.job_dir / "results.csv"))
    return 0 if job.errored == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks-dir")
    parser.add_argument("--resume", help="finish the job in this job directory")
    parser.add_argument("--sandbox", default="docker", choices=["docker", "daytona"])
    parser.add_argument("--agent", default="oracle")
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--jobs-dir", default="jobs/python-sdk-batch")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
