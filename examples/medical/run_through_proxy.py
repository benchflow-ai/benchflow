"""Run the LangGraph Medical-Assistant agent HOSTED BY BenchFlow.

Starts a loopback LiteLLM proxy via ``ensure_litellm_runtime`` and runs the
supervisor→specialists graph against it, so every node's raw LLM call (router,
answer, guardrail, and the confidence-gated web fallback) is tracked by the proxy
— usage/cost aggregated and the raw-LLM exchanges persisted to
``out/medical-assistant/trajectory/llm_trajectory.jsonl`` in BenchFlow's canonical
format. The agent never sees the raw provider key (proxy isolation invariant).

    uv pip install langgraph langchain-openai     # example-only deps
    set -a; . ./sb-run.env; set +a
    uv run python examples/medical/run_through_proxy.py "side effects of metformin?"
    # MEDICAL_CONFIDENCE_THRESHOLD=1.01 forces the confidence-gated web handoff.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from benchflow.providers import (
    ensure_litellm_runtime,
    extract_usage,
    stop_provider_runtime,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from medical_assistant import run


async def _main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY required (the real upstream key)")
    query = sys.argv[1] if len(sys.argv) > 1 else \
        "What are the main side effects of metformin?"

    print("starting BenchFlow LiteLLM proxy (environment=local)…", flush=True)
    agent_env, runtime = await ensure_litellm_runtime(
        agent="deepagents",
        agent_env={"DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]},
        model="deepseek/deepseek-v4-pro",
        runtime=None,
        environment="local",
        session_id="medical-assistant",
    )
    os.environ.update(agent_env)  # the graph's ChatOpenAI now reads BENCHFLOW_PROVIDER_*
    print("  proxy base :", agent_env.get("BENCHFLOW_PROVIDER_BASE_URL"))
    print("  raw key hidden from agent:", "DEEPSEEK_API_KEY" not in agent_env, flush=True)

    result: dict = {}
    try:
        result = await asyncio.to_thread(run, query)  # the LangGraph agent
    finally:
        await stop_provider_runtime(runtime)
    usage = extract_usage(runtime)

    run_dir = Path("out/medical-assistant")
    proxy_traj = getattr(getattr(runtime, "server", None), "trajectory", None)
    if proxy_traj is not None and proxy_traj.exchanges:
        (run_dir / "trajectory").mkdir(parents=True, exist_ok=True)
        (run_dir / "trajectory" / "llm_trajectory.jsonl").write_text(
            proxy_traj.to_jsonl(redact_keys=True))

    path = " → ".join(t["node"] for t in result.get("trace", []))
    print("\nquery      :", query)
    print("agent path :", path)
    print("route      :", result.get("route"), "| confidence:", result.get("confidence"),
          "| guardrail safe:", result.get("safe"))
    print("answer     :", (result.get("answer") or "").replace("\n", " ")[:280])
    print("proxy usage:", json.dumps(usage))
    if proxy_traj is not None and proxy_traj.exchanges:
        print(f"llm_trajectory: {run_dir}/trajectory/llm_trajectory.jsonl "
              f"({len(proxy_traj.exchanges)} node LLM calls, canonical format)")


if __name__ == "__main__":
    asyncio.run(_main())
