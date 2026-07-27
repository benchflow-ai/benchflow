"""Run the multi-agent medical workflow INSIDE a BenchFlow sandbox, end to end.

The whole LangGraph workflow (supervisor → answer → guardrail) executes isolated
in a Docker or Daytona sandbox, routing every agent's LLM call through the
BenchFlow LiteLLM proxy. Each node tags its call with bf.* metadata; the proxy
records it; after the run we download the trajectory and reconstruct the AGENT
TREE, asserting the agents are NOT mixed — identical behaviour on both backends.

Topology (a runtime detail; the tracking outcome is identical):
  - docker : agent runs in a container; proxy on the host docker bridge, reached
             from the container.
  - daytona: agent runs in a remote sandbox; proxy runs sandbox-local (127.0.0.1).

    set -a; . ~/sb-run.env; set +a                 # DEEPSEEK_API_KEY
    set -a; . ~/.daytona.env; set +a               # DAYTONA_API_KEY (daytona only)
    uv run python examples/medical/run_in_sandbox.py --env docker
    uv run python examples/medical/run_in_sandbox.py --env daytona
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from benchflow.providers import (
    ensure_litellm_runtime,
    extract_usage,
    stop_provider_runtime,
)
from benchflow.task.config import SandboxConfig
from benchflow.trajectories import build_agent_tree

HERE = Path(__file__).resolve().parent
ENV_DIR = HERE / "sandbox_env"
sys.path.insert(0, str(HERE))
from run_agent_tree import EXPECTED_PARENTS, _render  # noqa: E402 - after sys.path

# Runs inside the sandbox: import the uploaded graph, run it through the proxy
# (BENCHFLOW_PROVIDER_* come in via the exec env), print the path as one line.
_DRIVER = r'''
import json, os, sys
sys.path.insert(0, "/app")
import medical_assistant as ma
os.environ.setdefault("MEDICAL_CONFIDENCE_THRESHOLD", "1.01")
ma._AGENT_PROVIDERS.clear()                      # ONE shared proxy, tagged per agent
ma.reset_run(session_id=os.environ.get("BF_SESSION_ID", "medical-sandbox"))
result = ma.run(os.environ.get("MED_QUERY", "What are the main side effects of metformin?"))
print("TRACE_JSON:" + json.dumps({
    "path": [t["node"] for t in result.get("trace", [])],
    "route": result.get("route"), "confidence": result.get("confidence"),
    "safe": result.get("safe"),
}))
'''


def _make_sandbox(env: str):
    cfg = SandboxConfig(allow_internet=True)
    common = dict(environment_dir=ENV_DIR, environment_name="medicalmvp",
                  session_id=f"med-{env}", rollout_paths=None, task_env_config=cfg)
    if env == "docker":
        from benchflow.sandbox.docker import DockerSandbox
        return DockerSandbox(**common)
    from benchflow.sandbox.daytona import DaytonaSandbox
    return DaytonaSandbox(**common)


def _assert_unmixed(tree) -> list[str]:
    problems: list[str] = []
    seen = {n.agent_id for n in tree.nodes()}
    if seen != {"supervisor", "answer", "guardrail"}:
        problems.append(f"agents = {sorted(seen)}, expected supervisor/answer/guardrail")
    if [r.agent_id for r in tree.roots] != ["supervisor"]:
        problems.append(f"roots = {[r.agent_id for r in tree.roots]}, expected [supervisor]")
    for node in tree.nodes():
        for ex in node.exchanges:
            aid = (ex.request.body.get("bf") or {}).get("agent_id")
            if aid != node.agent_id:
                problems.append(f"MIXED: {node.agent_id} node holds a call from {aid!r}")
        if EXPECTED_PARENTS.get(node.agent_id) != node.parent_agent_id:
            problems.append(f"{node.agent_id} parent={node.parent_agent_id!r}, "
                            f"expected {EXPECTED_PARENTS.get(node.agent_id)!r}")
    return problems


async def _main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="docker", choices=["docker", "daytona"])
    p.add_argument("--query", default="What are the main side effects of metformin?")
    args = p.parse_args()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY required")

    sandbox = _make_sandbox(args.env)
    print(f"== provisioning {args.env} sandbox (image w/ langgraph deps) ==", flush=True)
    await sandbox.start(force_build=False)
    runtime = None
    trace = None
    try:
        # Proxy: sandbox-local for daytona, host-bridge for docker (reached from
        # the agent container). Either way the agent only sees BENCHFLOW_PROVIDER_*.
        sandbox_arg = sandbox if args.env == "daytona" else None
        proxy_environment = "daytona" if args.env == "daytona" else "docker"
        agent_env, runtime = await ensure_litellm_runtime(
            agent="deepagents",
            agent_env={"DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]},
            model="deepseek/deepseek-v4-pro", runtime=None,
            environment=proxy_environment,
            session_id=f"medical-sandbox-{args.env}", sandbox=sandbox_arg,
        )
        print("  proxy base (agent-visible):",
              agent_env.get("BENCHFLOW_PROVIDER_BASE_URL"), flush=True)

        # Upload the graph + a tiny driver into the sandbox.
        await sandbox.upload_file(str(HERE / "medical_assistant.py"),
                                  "/app/medical_assistant.py")
        tmp = Path(tempfile.mkdtemp())
        (tmp / "driver.py").write_text(_DRIVER)
        await sandbox.upload_file(str(tmp / "driver.py"), "/app/driver.py")

        run_env = dict(agent_env)
        run_env.update({
            "MEDICAL_CONFIDENCE_THRESHOLD": "1.01",
            "BF_SESSION_ID": f"medical-{args.env}",
            "MED_QUERY": args.query,
        })
        print("== running the medical graph INSIDE the sandbox ==", flush=True)
        res = await sandbox.exec("python3 /app/driver.py", env=run_env, timeout_sec=300)
        for line in (res.stdout or "").splitlines():
            if line.startswith("TRACE_JSON:"):
                trace = json.loads(line[len("TRACE_JSON:"):])
        print("  driver rc =", res.return_code,
              "| in-sandbox path:", " → ".join(trace["path"]) if trace else "(none)")
        if res.return_code != 0:
            print("  STDERR:", (res.stderr or "")[:900])
        await asyncio.sleep(1.5)  # let the proxy callback flush
    finally:
        if runtime is not None:
            await stop_provider_runtime(runtime)  # downloads callback log -> trajectory

    usage = extract_usage(runtime) if runtime is not None else {}
    traj = getattr(getattr(runtime, "server", None), "trajectory", None)
    tree = build_agent_tree(traj) if traj is not None else build_agent_tree([])
    print("\nproxy calls:", len(traj.exchanges) if traj else 0, "| usage:", json.dumps(usage))
    print(f"AGENT TREE (one shared proxy, in {args.env} sandbox):")
    for line in _render(tree, {}):
        print("  " + line)

    problems = _assert_unmixed(tree)
    out_dir = Path(f"out/medical-sandbox-{args.env}")
    (out_dir / "trajectory").mkdir(parents=True, exist_ok=True)
    if traj is not None and traj.exchanges:
        (out_dir / "trajectory" / "llm_trajectory.jsonl").write_text(
            traj.to_jsonl(redact_keys=True))
    (out_dir / "agent_tree.json").write_text(json.dumps({
        "environment": args.env,
        "ran_in_sandbox": True,
        "roots": [r.agent_id for r in tree.roots],
        "agents": {n.agent_id: {"parent": n.parent_agent_id, "calls": len(n.exchanges)}
                   for n in tree.nodes()},
        "unmixed_ok": not problems,
    }, indent=2))

    await sandbox.stop(delete=True)
    if problems or not trace:
        print("\n❌ FAILED:", "; ".join(problems) or "no in-sandbox trace")
        return 1
    print(f"\n✅ [{args.env}] agent ran IN sandbox · one proxy · agents NOT mixed · "
          "tree = supervisor → answer → guardrail")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
