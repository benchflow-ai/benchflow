#!/usr/bin/env python3
"""Branch an agent run into scored children with one call: bf.branch.

The parent runs the first prompt (the checkpoint: the agent writes
draft.txt), the sandbox is snapshotted, and each child starts from that
snapshot with its own prompt and a fresh agent session, then is scored by the
task's verifier. The parent is restored, finishes its remaining prompts and
is verified too. This is `bench eval branch` from Python; the job folder is
the same (tree.json, per-child observation.json, summary.json).

Children do not remember the parent's conversation (unless
--resume-session), so each child prompt must stand on its own.

Usage:
  uv run python docs/examples/python-sdk/run-branch.py
  uv run python docs/examples/python-sdk/run-branch.py --sandbox daytona --concurrency 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import benchflow as bf

HELLO_WORLD = Path(__file__).resolve().parents[3] / "tests/examples/hello-world-task"


def main(args: argparse.Namespace) -> int:
    result = bf.branch(
        Path(args.task),
        agent=args.agent,
        model=args.model,
        sandbox=args.sandbox,
        # The parent's prompts; the fork comes after the first one.
        prompts=[
            "This is step 1 of a two-step exercise. In the current working "
            "directory, create draft.txt containing exactly one line: Hello world\n"
            "Do not create any other files.",
            "@instruction",  # the task's own instruction
        ],
        checkpoint_after=1,
        children={
            "baseline": None,  # the parent's remaining prompts (@instruction)
            "hint-reuse-draft": (
                "draft.txt in the current working directory already holds the "
                "final text. Rename it to hello.txt without changing it, then stop."
            ),
        },
        concurrency=args.concurrency,  # >1 runs children in their own sandboxes
        jobs_dir=args.jobs_dir,
    )
    print(f"V(checkpoint) = {result.value}  (fork {result.fork_status})")
    for child in result.children:
        print(
            f"  {child.label:18} {child.status:9} reward={child.reward} ({child.reward_source})"
        )
    print(f"parent reward = {result.parent_reward}; parent {result.parent_restore}")
    print(f"job folder: {result.job_dir}")
    if result.error:
        print(f"error: {result.error}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sandbox", default="docker", choices=["docker", "daytona"])
    parser.add_argument("--agent", default="claude-agent-acp")
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--task", default=str(HELLO_WORLD))
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--jobs-dir", default="jobs/python-sdk-branch")
    raise SystemExit(main(parser.parse_args()))
