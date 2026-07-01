"""MVP: a multi-agent workflow through ONE BenchFlow proxy → an unmixed agent tree.

Runs the LangGraph medical assistant (supervisor → answer → guardrail) through a
SINGLE LiteLLM proxy. Each node tags its call with bf.* metadata (agent id, parent
pointer, run id, session id); the proxy callback records it; we reconstruct the
AGENT TREE from the one shared trajectory and assert the agents are NOT mixed.

Same runner, three backends — pass the proxy environment so behaviour can be
compared identically across local / docker / daytona:

    set -a; . ~/sb-run.env; set +a
    uv run python examples/medical/run_agent_tree.py --env local
    uv run python examples/medical/run_agent_tree.py --env docker
    uv run python examples/medical/run_agent_tree.py --env daytona
"""

from __future__ import annotations

import argparse
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
from benchflow.trajectories import build_agent_tree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import medical_assistant as ma

EXPECTED_PARENTS = {"supervisor": None, "answer": "supervisor", "guardrail": "answer"}


def _render(tree, usage_by_agent) -> list[str]:
    lines: list[str] = []

    def walk(node, depth: int) -> None:
        u = usage_by_agent.get(node.agent_id, {})
        lines.append(
            f"{'  ' * depth}└─ {node.agent_id}  "
            f"({len(node.exchanges)} call(s)"
            + (f", {u.get('tokens')} tok" if u.get("tokens") else "")
            + ")"
        )
        for child in node.children:
            walk(child, depth + 1)

    for root in tree.roots:
        walk(root, 0)
    for orphan in tree.orphans:
        lines.append(f"[orphan] {orphan.agent_id} (parent "
                     f"{orphan.parent_agent_id!r} absent)")
        walk(orphan, 0)
    return lines


def _assert_unmixed(tree) -> list[str]:
    """Return a list of problems; empty list == the tree is correct + unmixed."""
    problems: list[str] = []
    seen = {n.agent_id for n in tree.nodes()}
    if seen != {"supervisor", "answer", "guardrail"}:
        problems.append(f"agents = {sorted(seen)}, expected supervisor/answer/guardrail")
    if [r.agent_id for r in tree.roots] != ["supervisor"]:
        problems.append(f"roots = {[r.agent_id for r in tree.roots]}, expected [supervisor]")
    for node in tree.nodes():
        # NOT MIXED: every exchange under this node was made BY this agent
        for ex in node.exchanges:
            aid = (ex.request.body.get("bf") or {}).get("agent_id")
            if aid != node.agent_id:
                problems.append(f"MIXED: {node.agent_id} node holds a call from {aid!r}")
        # parent link matches the declared graph edge
        if EXPECTED_PARENTS.get(node.agent_id) != node.parent_agent_id:
            problems.append(
                f"{node.agent_id} parent = {node.parent_agent_id!r}, "
                f"expected {EXPECTED_PARENTS.get(node.agent_id)!r}"
            )
    return problems


async def _main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="local", choices=["local", "docker", "daytona"])
    p.add_argument("--query", default="What are the main side effects of metformin?")
    args = p.parse_args()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY required")
    os.environ.setdefault("MEDICAL_CONFIDENCE_THRESHOLD", "1.01")  # force the web handoff

    print(f"== starting ONE LiteLLM proxy (environment={args.env}) ==", flush=True)
    agent_env, runtime = await ensure_litellm_runtime(
        agent="deepagents",
        agent_env={"DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]},
        model="deepseek/deepseek-v4-pro", runtime=None, environment=args.env,
        session_id=f"medical-tree-{args.env}",
    )
    ma._AGENT_PROVIDERS.clear()           # ONE proxy, not one-per-agent
    os.environ.update(agent_env)          # the graph's ChatOpenAI now hits the proxy
    ma.reset_run(session_id=f"medical-{args.env}")
    print("  proxy base:", agent_env.get("BENCHFLOW_PROVIDER_BASE_URL"), flush=True)

    result: dict = {}
    try:
        result = await asyncio.to_thread(ma.run, args.query)
        await asyncio.sleep(1.5)          # let the proxy callback flush
    finally:
        await stop_provider_runtime(runtime)
    usage = extract_usage(runtime)
    traj = getattr(getattr(runtime, "server", None), "trajectory", None)

    tree = build_agent_tree(traj) if traj is not None else build_agent_tree([])
    path = " → ".join(t["node"] for t in result.get("trace", []))
    print("\nagent path :", path)
    print("proxy calls:", len(traj.exchanges) if traj else 0,
          "| usage:", json.dumps(usage))
    print("\nAGENT TREE (one shared proxy, reconstructed from bf.*):")
    for line in _render(tree, {}):
        print("  " + line)

    problems = _assert_unmixed(tree)
    out_dir = Path(f"out/medical-tree-{args.env}")
    (out_dir / "trajectory").mkdir(parents=True, exist_ok=True)
    if traj is not None and traj.exchanges:
        (out_dir / "trajectory" / "llm_trajectory.jsonl").write_text(
            traj.to_jsonl(redact_keys=True))
    (out_dir / "agent_tree.json").write_text(json.dumps({
        "environment": args.env,
        "roots": [r.agent_id for r in tree.roots],
        "agents": {n.agent_id: {"parent": n.parent_agent_id, "calls": len(n.exchanges)}
                   for n in tree.nodes()},
        "unmixed_ok": not problems,
    }, indent=2))
    print(f"\npersisted: {out_dir}/trajectory/llm_trajectory.jsonl + agent_tree.json")

    if problems:
        print("\n❌ TREE CHECK FAILED:")
        for pr in problems:
            print("   -", pr)
        return 1
    print(f"\n✅ [{args.env}] one proxy · {len(tree.roots)} root · agents NOT mixed · "
          "tree = supervisor → answer → guardrail")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
