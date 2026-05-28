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
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from benchflow.agents.registry import AGENTS
from benchflow.evaluation import Evaluation, EvaluationConfig

TASK_DIR = Path(__file__).parent
RESULTS_FILE = Path(__file__).parent / "conformance-results.json"

AGENT_MODELS = {
    "claude-agent-acp": "claude-haiku-4-5-20251001",
    "pi-acp": "gemini-3.1-flash-lite-preview",
    "openclaw": "gemini-3.1-flash-lite-preview",
    "codex-acp": "gpt-5.4-mini",
    "gemini": "gemini-3.1-flash-lite-preview",
    "mini-swe": "gemini-3.1-flash-lite-preview",
}

ENV_KEYS = {
    "claude-agent-acp": ["ANTHROPIC_API_KEY"],
    "pi-acp": ["ANTHROPIC_API_KEY"],
    "openclaw": [],
    "codex-acp": ["OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "mini-swe": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
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
    return bool(sub_file and Path(sub_file).expanduser().exists())


def openai_model_preflight(model: str) -> str | None:
    """Return an error string if API-key auth cannot access *model*."""
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY")
    if not key:
        return None
    req = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        return f"OpenAI model preflight failed ({exc.code}): {body}"
    except Exception as exc:
        return f"OpenAI model preflight failed: {type(exc).__name__}: {exc}"

    model_ids = {item.get("id") for item in payload.get("data", [])}
    if model not in model_ids:
        return f"OpenAI model {model!r} is not available for this API key"
    return None


async def run_one(agent_name: str) -> dict:
    model = AGENT_MODELS.get(agent_name, "claude-haiku-4-5-20251001")
    config = EvaluationConfig(
        agent=agent_name,
        model=model,
        environment="daytona",
    )
    job = Evaluation(
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
        model = AGENT_MODELS.get(name, "claude-haiku-4-5-20251001")
        if name == "codex-acp" and (reason := openai_model_preflight(model)):
            print(f"ERROR {name} — {reason}")
            results.append(
                {"agent": name, "model": model, "status": f"ERROR: {reason}"}
            )
            continue
        print(f"\n{'=' * 60}")
        print(f"CONFORMANCE: {name} (model={model})")
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
