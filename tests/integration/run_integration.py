"""Standalone runner for integration tests.

Usage::

    GEMINI_API_KEY=... DAYTONA_API_KEY=... python tests/integration/run_integration.py

    # Single agent
    python tests/integration/run_integration.py --agent gemini

    # Dry run (show what would be run)
    python tests/integration/run_integration.py --dry-run

    # Specific tasks only
    python tests/integration/run_integration.py --tasks jax-computing-basics,data-to-d3

Guards: ENG-6 integration test plan (issue #253).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Reuse constants from conftest
SKILLSBENCH_TASKS = [
    "jax-computing-basics",
    "python-scala-translation",
    "jpg-ocr-stat",
    "grid-dispatch-operator",
    "threejs-to-obj",
    "data-to-d3",
    "lake-warming-attribution",
    "weighted-gdp-calc",
    "shock-analysis-supply",
]

ALL_AGENTS = [
    "claude-agent-acp",
    "pi-acp",
    "openclaw",
    "codex-acp",
    "gemini",
    "opencode",
    "harvey-lab-harness",
    "openhands",
]

AGENT_REQUIRED_KEYS: dict[str, list[str]] = {
    "claude-agent-acp": [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ],
    "pi-acp": [],
    "openclaw": [],
    "codex-acp": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "opencode": [],
    "harvey-lab-harness": [],
    "openhands": [],
}

AGENT_MODEL_OVERRIDES: dict[str, str] = {
    "claude-agent-acp": "claude-haiku-4-5-20251001",
    "codex-acp": "gpt-5.4-nano",
}

SUBSCRIPTION_AUTH_FILES: dict[str, str] = {
    "claude-agent-acp": "~/.claude/.credentials.json",
    "codex-acp": "~/.codex/auth.json",
}


def has_creds(agent: str) -> bool:
    required = AGENT_REQUIRED_KEYS.get(agent, [])
    if not required:
        return bool(
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
    if any(os.environ.get(k) for k in required):
        return True
    sub_file = SUBSCRIPTION_AUTH_FILES.get(agent)
    return bool(sub_file and Path(sub_file).expanduser().is_file())


async def run_agent_matrix(
    agents: list[str],
    tasks: list[str],
    model: str,
    environment: str,
    concurrency: int,
    jobs_root: Path,
    dry_run: bool = False,
) -> dict:
    """Run the agent×task matrix and return a summary dict."""
    from benchflow.job import Job, JobConfig, RetryConfig
    from benchflow.task_download import resolve_source

    tasks_dir = resolve_source("benchflow-ai/skillsbench", path="tasks", ref="main")
    all_task_names = {
        d.name for d in tasks_dir.iterdir() if d.is_dir() and (d / "task.toml").exists()
    }
    selected = set(tasks)
    exclude = all_task_names - selected

    summary: dict[str, dict] = {}
    wall_start = time.monotonic()

    for agent in agents:
        if not has_creds(agent):
            logger.warning("Skipping %s — no credentials", agent)
            summary[agent] = {"skipped": True, "reason": "no credentials"}
            continue

        if dry_run:
            logger.info(
                "[DRY RUN] Would run %d tasks with %s/%s on %s (concurrency=%d)",
                len(tasks),
                agent,
                model,
                environment,
                concurrency,
            )
            summary[agent] = {"dry_run": True, "tasks": len(tasks)}
            continue

        agent_jobs_dir = jobs_root / f"integration-{agent}"
        agent_jobs_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running %s with %d tasks...", agent, len(tasks))
        agent_start = time.monotonic()

        agent_model = AGENT_MODEL_OVERRIDES.get(agent, model)
        config = JobConfig(
            agent=agent,
            model=agent_model,
            environment=environment,
            concurrency=concurrency,
            retry=RetryConfig(max_retries=1),
            exclude_tasks=exclude,
        )

        job = Job(
            tasks_dir=tasks_dir,
            jobs_dir=agent_jobs_dir,
            config=config,
        )

        try:
            result = await job.run()
            elapsed = time.monotonic() - agent_start
            summary[agent] = {
                "total": result.total,
                "passed": result.passed,
                "failed": result.failed,
                "errored": result.errored,
                "score": round(result.score, 4),
                "elapsed_sec": round(elapsed, 1),
            }
            logger.info(
                "%s: %d/%d (%.1f%%) in %.0fs",
                agent,
                result.passed,
                result.total,
                result.score * 100,
                elapsed,
            )
        except Exception as e:
            logger.error("Agent %s failed: %s", agent, e)
            summary[agent] = {"error": str(e)}

    wall_elapsed = time.monotonic() - wall_start

    return {
        "agents": summary,
        "model": model,
        "environment": environment,
        "concurrency": concurrency,
        "tasks": tasks,
        "wall_elapsed_sec": round(wall_elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BenchFlow integration test runner")
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Run a single agent (default: all)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names (default: all 9)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-lite-preview",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="daytona",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--jobs-root",
        type=str,
        default="jobs/integration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    args = parser.parse_args()

    # Validate environment
    if not args.dry_run:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            logger.error("GEMINI_API_KEY or GOOGLE_API_KEY required")
            sys.exit(1)
        if not os.environ.get("DAYTONA_API_KEY"):
            logger.error("DAYTONA_API_KEY required for Daytona backend")
            sys.exit(1)

    agents = [args.agent] if args.agent else ALL_AGENTS
    tasks = args.tasks.split(",") if args.tasks else SKILLSBENCH_TASKS
    jobs_root = Path(args.jobs_root)
    jobs_root.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(
        run_agent_matrix(
            agents=agents,
            tasks=tasks,
            model=args.model,
            environment=args.environment,
            concurrency=args.concurrency,
            jobs_root=jobs_root,
            dry_run=args.dry_run,
        )
    )

    # Write summary
    summary_path = jobs_root / "integration-summary.json"
    summary_path.write_text(json.dumps(result, indent=2))
    logger.info("Summary written to %s", summary_path)

    # Print table
    print("\n" + "=" * 70)
    print(f"{'Agent':<25} {'Score':>8} {'Pass':>5} {'Fail':>5} {'Err':>5} {'Time':>8}")
    print("-" * 70)
    for agent_name, data in result["agents"].items():
        if data.get("skipped"):
            print(f"{agent_name:<25} {'SKIP':>8}")
        elif data.get("dry_run"):
            print(f"{agent_name:<25} {'DRY':>8} {data['tasks']:>5}")
        elif data.get("error"):
            print(f"{agent_name:<25} {'ERROR':>8}   {data['error'][:30]}")
        else:
            print(
                f"{agent_name:<25} {data['score']:>7.1%} "
                f"{data['passed']:>5} {data['failed']:>5} "
                f"{data['errored']:>5} {data['elapsed_sec']:>7.0f}s"
            )
    print("=" * 70)
    print(f"Total wall time: {result['wall_elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
