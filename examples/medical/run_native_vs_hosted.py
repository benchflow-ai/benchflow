"""Run the Medical-Assistant bench TWO ways and diff them.

(a) NATIVE  — every node calls deepseek DIRECTLY. No proxy, no BenchFlow
    tracking: you get an answer, but no usage/cost and no llm_trajectory.
(b) HOSTED  — every specialist (supervisor / answer / guardrail) gets its OWN
    LiteLLM proxy, so each agent produces a SEPARATE llm_trajectory.jsonl +
    its own usage/cost. This is the fix for "the multi-agent run only emits a
    single trajectory": one shared proxy = one mixed log; one proxy per agent
    = one log per agent.

  set -a; . ./sb-run.env; set +a
  uv run python examples/medical/run_native_vs_hosted.py "side effects of metformin?"
  # threshold 1.01 (default below) forces the confidence-gated web handoff so
  # the 'answer' specialist fires twice — visible as 2 exchanges in its log.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from benchflow.providers import (
    ensure_litellm_runtime,
    extract_usage,
    stop_provider_runtime,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import medical_assistant as ma

LLM_AGENTS = ("supervisor", "answer", "guardrail")  # the nodes that call an LLM


def _summary(result: dict) -> dict:
    return {
        "path": " → ".join(t["node"] for t in result.get("trace", [])),
        "route": result.get("route"),
        "confidence": result.get("confidence"),
        "safe": result.get("safe"),
        "answer": (result.get("answer") or "").replace("\n", " ")[:200],
    }


async def native_run(query: str) -> dict:
    """Direct provider — no proxy. _AGENT_PROVIDERS empty → _llm() reads env."""
    ma._AGENT_PROVIDERS.clear()
    os.environ["BENCHFLOW_PROVIDER_BASE_URL"] = "https://api.deepseek.com"
    os.environ["BENCHFLOW_PROVIDER_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
    os.environ["BENCHFLOW_PROVIDER_MODEL"] = "deepseek-v4-pro"  # no prefix for direct
    result = await asyncio.to_thread(ma.run, query)
    return _summary(result)


async def hosted_run(query: str) -> tuple[dict, dict]:
    """One proxy per specialist → a separate trajectory + usage per agent."""
    ma._AGENT_PROVIDERS.clear()
    runtimes: dict = {}
    print(f"starting {len(LLM_AGENTS)} per-agent LiteLLM proxies…", flush=True)
    for ag in LLM_AGENTS:
        env, rt = await ensure_litellm_runtime(
            agent="deepagents",
            agent_env={"DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]},
            model="deepseek/deepseek-v4-pro",
            runtime=None,
            environment="local",
            session_id=f"medical-{ag}",
        )
        runtimes[ag] = rt
        ma.set_agent_provider(
            ag,
            env["BENCHFLOW_PROVIDER_BASE_URL"],
            env["BENCHFLOW_PROVIDER_API_KEY"],
            env["BENCHFLOW_PROVIDER_MODEL"],
        )
        print(f"  {ag:<10}: {env['BENCHFLOW_PROVIDER_BASE_URL']}", flush=True)

    run_dir = Path("out/medical-hosted")
    per_agent: dict = {}
    try:
        result = await asyncio.to_thread(ma.run, query)
        await asyncio.sleep(1.5)  # let each proxy's async callback flush
    finally:
        for ag in LLM_AGENTS:
            await stop_provider_runtime(
                runtimes[ag]
            )  # stop FIRST → parses callback log
    for ag in LLM_AGENTS:  # then read usage + trajectory (populated on stop)
        usage = extract_usage(runtimes[ag])
        traj = getattr(getattr(runtimes[ag], "server", None), "trajectory", None)
        n = len(traj.exchanges) if traj is not None else 0
        path = None
        if traj is not None and traj.exchanges:
            d = run_dir / ag / "trajectory"
            d.mkdir(parents=True, exist_ok=True)
            path = d / "llm_trajectory.jsonl"
            path.write_text(traj.to_jsonl(redact_keys=True))
        per_agent[ag] = {
            "calls": n,
            "usage": usage,
            "trajectory": str(path) if path else None,
        }
    return _summary(result), per_agent


async def _main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY required (the real upstream key)")
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What are the main side effects of metformin?"
    )
    os.environ.setdefault("MEDICAL_CONFIDENCE_THRESHOLD", "1.01")  # force web handoff

    print("=" * 72)
    print("NATIVE (direct deepseek — no BenchFlow)")
    print("=" * 72)
    native = await native_run(query)

    print("\n" + "=" * 72)
    print("HOSTED (one BenchFlow proxy PER specialist)")
    print("=" * 72)
    hosted, per_agent = await hosted_run(query)

    print("\n" + "=" * 72)
    print(f"COMPARISON · query: {query!r}")
    print("=" * 72)
    print(f"{'':<14}{'NATIVE':<40}{'HOSTED':<40}")
    for k in ("path", "route", "confidence", "safe"):
        print(f"{k:<14}{native[k]!s:<40}{hosted[k]!s:<40}")
    print(f"{'answer':<14}{native['answer'][:36]:<40}{hosted['answer'][:36]:<40}")
    tot = sum(
        (per_agent[a]["usage"] or {}).get("total_tokens", 0) or 0 for a in LLM_AGENTS
    )
    cost = sum(
        (per_agent[a]["usage"] or {}).get("cost_usd", 0) or 0 for a in LLM_AGENTS
    )
    print(
        f"{'tracking':<14}{'none (no proxy → no usage/traj)':<40}"
        f"{f'{tot} tok · ${cost:.5f} · per-agent':<40}"
    )

    print("\n--- HOSTED: SEPARATE trajectory per agent ---")
    for ag in LLM_AGENTS:
        pa = per_agent[ag]
        u = pa["usage"] or {}
        print(
            f"  {ag:<10} {pa['calls']} call(s) · "
            f"tok={u.get('total_tokens')} cost=${u.get('cost_usd')}"
        )
        print(f"             {pa['trajectory']}")
    print(
        "\nNATIVE produced NO trajectory and NO usage — that observability only "
        "exists when the LLM is routed through BenchFlow."
    )


if __name__ == "__main__":
    asyncio.run(_main())
