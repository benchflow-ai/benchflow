"""Run ACP conformance smoke against registered agents.

Usage: env -u ANTHROPIC_API_KEY python run_conformance.py [agent-name ...]

If no agent names given, runs all registered agents that have credentials
available in the environment. Results are printed as a table and written
to conformance-results.json.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from benchflow.agents.registry import AGENTS
from benchflow.job import Job, JobConfig

TASK_DIR = Path(__file__).parent
RESULTS_FILE = Path(__file__).parent / "conformance-results.json"

AGENT_MODELS = {
    "claude-agent-acp": "claude-haiku-4-5-20251001",
    "pi-acp": "gemini-3.1-flash-lite-preview",
    "openclaw": "gemini-3.1-flash-lite-preview",
    "codex-acp": "gpt-5.4-nano",
    "gemini": "gemini-3.1-flash-lite-preview",
}

ENV_KEYS = {
    "claude-agent-acp": ["ANTHROPIC_API_KEY"],
    "pi-acp": ["ANTHROPIC_API_KEY"],
    "openclaw": [],
    "codex-acp": ["OPENAI_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
}


SUBSCRIPTION_AUTH_FILES = {
    "claude-agent-acp": "~/.claude/.credentials.json",
    "codex-acp": "~/.codex/auth.json",
}


def has_creds(agent_name: str) -> bool:
    keys = ENV_KEYS.get(agent_name, [])
    if not keys:
        return True
    if any(os.environ.get(k) for k in keys):
        return True
    sub_file = SUBSCRIPTION_AUTH_FILES.get(agent_name)
    if sub_file and Path(sub_file).expanduser().exists():
        return True
    return False


async def run_one(agent_name: str) -> dict:
    model = AGENT_MODELS.get(agent_name, "claude-haiku-4-5-20251001")
    config = JobConfig(
        agent=agent_name,
        model=model,
        environment="daytona",
    )
    job = Job(
        tasks_dir=TASK_DIR,
        jobs_dir=Path(f"/tmp/conformance-jobs/{agent_name}"),
        config=config,
    )
    t0 = time.time()
    try:
        result = await job.run()
        elapsed = time.time() - t0
        return {
            "agent": agent_name,
            "model": model,
            "passed": result.passed,
            "total": result.total,
            "errors": result.errored,
            "elapsed_sec": round(elapsed, 1),
            "status": "PASS" if result.passed > 0 else "FAIL",
        }
    except Exception as e:
        return {
            "agent": agent_name,
            "model": model,
            "passed": 0,
            "total": 1,
            "errors": 1,
            "elapsed_sec": round(time.time() - t0, 1),
            "status": f"ERROR: {e!s:.80}",
        }


async def main() -> None:
    requested = sys.argv[1:] or list(AGENTS.keys())
    results = []
    for name in requested:
        if name not in AGENTS:
            print(f"SKIP {name} — not in registry")
            continue
        if not has_creds(name):
            print(f"SKIP {name} — no credentials in env ({ENV_KEYS[name]})")
            results.append({"agent": name, "status": "SKIP (no creds)"})
            continue
        print(f"\n{'=' * 60}")
        print(f"CONFORMANCE: {name} (model={AGENT_MODELS.get(name, '?')})")
        print(f"{'=' * 60}")
        r = await run_one(name)
        results.append(r)
        print(
            f"  → {r['status']}  (reward={r.get('passed', 0)}/{r.get('total', 1)}, {r.get('elapsed_sec', 0)}s)"
        )

    print(f"\n{'=' * 60}")
    print("CONFORMANCE SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  {r['agent']:25s} {r['status']}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
