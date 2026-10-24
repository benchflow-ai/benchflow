#!/usr/bin/env python3
"""Credential-free full Rollout artifact smoke for native science harnesses."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from native_acp_harness_smoke import _MockProvider

from benchflow.sdk import SDK

AGENTS = ("openscience", "deepseek-harness")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"expected JSON objects in {path}")
    return rows


def _validate(rollout_dir: Path, agent: str) -> dict[str, Any]:
    required = [
        rollout_dir / "result.json",
        rollout_dir / "results.jsonl",
        rollout_dir / "timing.json",
        rollout_dir / "prompts.json",
        rollout_dir / "rewards.jsonl",
        rollout_dir / "trajectory/acp_trajectory.jsonl",
        rollout_dir / "trajectory/llm_trajectory.jsonl",
        rollout_dir / "verifier/reward.txt",
        rollout_dir / "verifier/test-stdout.txt",
        rollout_dir / "trainer/adp.jsonl",
        rollout_dir / "trainer/atif.json",
        rollout_dir / "trainer/verifiers.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"{agent} missing artifacts: {missing}")

    result = _json(rollout_dir / "result.json")
    results = _jsonl(rollout_dir / "results.jsonl")
    timing = _json(rollout_dir / "timing.json")
    acp = _jsonl(rollout_dir / "trajectory/acp_trajectory.jsonl")
    llm = _jsonl(rollout_dir / "trajectory/llm_trajectory.jsonl")
    rewards = _jsonl(rollout_dir / "rewards.jsonl")
    trainer_adp = _jsonl(rollout_dir / "trainer/adp.jsonl")
    trainer_atif = _json(rollout_dir / "trainer/atif.json")
    trainer_verifiers = _jsonl(rollout_dir / "trainer/verifiers.jsonl")
    event_types = sorted(
        {str(row.get("type")) for row in acp if isinstance(row.get("type"), str)}
    )
    agent_result = result.get("agent_result") or {}
    reward = (result.get("rewards") or {}).get("reward")

    if result.get("error") is not None or result.get("verifier_error") is not None:
        raise RuntimeError(
            f"{agent} rollout errored: {result.get('error') or result.get('verifier_error')}"
        )
    if reward != 1.0:
        raise RuntimeError(f"{agent} reward mismatch: {reward!r}")
    if not {"user_message", "tool_call", "agent_message"}.issubset(event_types):
        raise RuntimeError(f"{agent} ACP event gap: {event_types}")
    if not llm:
        raise RuntimeError(f"{agent} LLM trajectory is empty")
    if agent_result.get("usage_source") != "provider_response":
        raise RuntimeError(
            f"{agent} usage source: {agent_result.get('usage_source')!r}"
        )
    if int(agent_result.get("total_tokens") or 0) <= 0:
        raise RuntimeError(f"{agent} token usage was not captured")
    if not rewards:
        raise RuntimeError(f"{agent} reward log is empty")
    if len(results) != 1 or results[0].get("reward") != 1.0:
        raise RuntimeError(f"{agent} results export is incomplete")
    if (results[0].get("info") or {}).get("training_ready") is not True:
        raise RuntimeError(f"{agent} results export is not training-ready")
    if (rollout_dir / "verifier/reward.txt").read_text().strip() != "1":
        raise RuntimeError(f"{agent} verifier reward artifact mismatch")
    if not all(
        timing.get(key) is not None
        for key in ("agent_setup", "agent_execution", "verifier", "total")
    ):
        raise RuntimeError(f"{agent} timing artifact is incomplete")
    if not trainer_adp or not trainer_atif or not trainer_verifiers:
        raise RuntimeError(f"{agent} trainer artifacts are incomplete")

    return {
        "agent": agent,
        "rollout_dir": str(rollout_dir),
        "reward": reward,
        "usage_source": agent_result.get("usage_source"),
        "total_tokens": agent_result.get("total_tokens"),
        "acp_event_types": event_types,
        "acp_rows": len(acp),
        "llm_rows": len(llm),
        "training_ready": (results[0].get("info") or {}).get("training_ready"),
        "trainer_files": ["adp.jsonl", "atif.json", "verifiers.jsonl"],
    }


async def _run(agent: str, jobs_dir: Path, provider: _MockProvider) -> dict[str, Any]:
    job_name = f"artifact-smoke-{agent}"
    rollout_name = "demo"
    host_endpoint = f"http://127.0.0.1:{provider.server.server_port}/v1"
    result = await SDK().run(
        task_path=Path("src/benchflow/demo_task"),
        agent=agent,
        prompts=[
            "ARTIFACT_SMOKE: create hello.txt exactly as requested using bash, then finish."
        ],
        model="vllm/deepseek-v4-flash",
        agent_env={
            "OPENAI_API_KEY": "credential-free-artifact-smoke",
            "BENCHFLOW_PROVIDER_API_KEY": "credential-free-artifact-smoke",
            "BENCHFLOW_PROVIDER_BASE_URL": host_endpoint,
            "BENCHFLOW_PROVIDER_PROTOCOL": "openai-completions",
        },
        job_name=job_name,
        rollout_name=rollout_name,
        jobs_dir=jobs_dir,
        environment="docker",
        sandbox_user="agent",
        usage_tracking="required",
    )
    rollout_dir = jobs_dir / job_name / rollout_name
    if result.error:
        raise RuntimeError(f"{agent} SDK result error: {result.error}")
    return _validate(rollout_dir, agent)


async def _main(agents: list[str], jobs_dir: Path) -> list[dict[str, Any]]:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    with _MockProvider() as provider:
        return [await _run(agent, jobs_dir, provider) for agent in agents]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=(*AGENTS, "all"), default="all")
    parser.add_argument("--jobs-dir", type=Path, required=True)
    args = parser.parse_args()
    agents = list(AGENTS) if args.agent == "all" else [args.agent]
    prior_debug = os.environ.get("DEBUG")
    os.environ["DEBUG"] = "false"
    try:
        print(json.dumps(asyncio.run(_main(agents, args.jobs_dir)), sort_keys=True))
    finally:
        if prior_debug is None:
            os.environ.pop("DEBUG", None)
        else:
            os.environ["DEBUG"] = prior_debug


if __name__ == "__main__":
    main()
