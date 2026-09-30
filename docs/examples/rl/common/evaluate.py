"""Evaluate an OpenAI-compatible endpoint on a BenchFlow task set.

    python docs/examples/rl/common/evaluate.py \\
        --tasks-dir tasks/v1/test \\
        --base-url https://router.huggingface.co/v1 --model openai/gpt-oss-20b \\
        --api-key-env HF_TOKEN --sandbox daytona --concurrency 8 \\
        --out results/gpt-oss-20b

The policy gets the harness the cookbooks train with: the task prompt plus the
shared harness message, and the ``run_bash`` and ``submit`` tools of
``benchflow.integrations.trl``. Every episode runs in its own BenchFlow sandbox
(``TaskRuntime``) on Daytona or Docker, and ends the way a training rollout
ends: through the TRL adapter's verification and the attribution rule of
``benchflow.integrations.rewards``. Infrastructure failures (a sandbox that
never started, the model endpoint failing, a verifier crash on an untouched
sandbox) are dropped and counted; every other failure scores 0.

Writes to ``--out``:

- ``summary.json``: the solve rate over kept episodes with a 95% Wilson
  interval, drops by reason, zeros by reason, results per task kind, and
  token totals.
- ``episodes.jsonl``: one line per episode.
- ``jobs/``: the BenchFlow rollout folders, plus ``rollouts.jsonl`` with every
  episode's messages and decision, dropped episodes included, for audit.

The API key is read from the environment variable named by ``--api-key-env``
and only sent to ``--base-url``. It never enters a sandbox.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import MAX_TURNS, harness_config

from benchflow.integrations.rewards import (
    RewardDecision,
    model_endpoint_failure,
    summarize,
)
from benchflow.integrations.trl import (
    BenchFlowRuntimeEnvironment,
    BenchFlowSpec,
    bash_tool_schemas,
    finish_rollout,
)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
CONTEXT_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "prompt is too long",
)


class EndpointError(Exception):
    """The model endpoint failed in a way the policy could not have caused."""


class ContextExhausted(Exception):
    """The endpoint refused the conversation as too long: the policy's budget ran out."""


@dataclass
class Episode:
    task_id: str
    sample: int
    kind: str | None
    level: int | None
    decision: RewardDecision | None = None
    turns: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ended: str = ""
    elapsed_sec: float = 0.0
    messages: list[dict[str, Any]] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        decision = self.decision.as_dict() if self.decision else {}
        return {
            "task_id": self.task_id,
            "sample": self.sample,
            "kind": self.kind,
            "level": self.level,
            **decision,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ended": self.ended,
            "elapsed_sec": round(self.elapsed_sec, 2),
        }


class Endpoint:
    """A minimal OpenAI-compatible chat client with bounded retries."""

    def __init__(self, args: argparse.Namespace, api_key: str) -> None:
        headers = {"Authorization": f"Bearer {api_key}"}
        for header in args.header:
            name, _, value = header.partition(":")
            headers[name.strip()] = value.strip()
        self.client = httpx.Client(
            base_url=args.base_url.rstrip("/"),
            headers=headers,
            timeout=args.request_timeout,
        )
        self.args = args

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.args.model,
            "messages": messages,
            "tools": bash_tool_schemas(),
            "tool_choice": "auto",
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
        }
        if self.args.extra_body:
            body.update(json.loads(self.args.extra_body))
        last = "no attempt"
        for attempt in range(self.args.retries + 1):
            if attempt:
                time.sleep(min(60.0, 2.0**attempt) * (0.5 + random.random()))
            try:
                response = self.client.post("/chat/completions", json=body)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
                continue
            if response.status_code == 200:
                return response.json()
            text = response.text[:500]
            if response.status_code == 400 and any(
                m in text.lower() for m in CONTEXT_MARKERS
            ):
                raise ContextExhausted(text)
            last = f"HTTP {response.status_code}: {text}"
            if response.status_code not in RETRYABLE_STATUS:
                break
        raise EndpointError(last)


def _tool_result(
    env: BenchFlowRuntimeEnvironment, call: dict[str, Any]
) -> tuple[str, bool]:
    """Run one tool call; return its result text and whether it ended the episode."""

    function = call.get("function") or {}
    name = function.get("name")
    try:
        raw = function.get("arguments")
        arguments = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        if name == "run_bash":
            return env.run_bash(str(arguments["command"])), False
        if name == "submit":
            return env.submit(str(arguments["answer"])), True
        raise ValueError(f"Tool {name} not found.")
    except Exception as exc:  # the same feedback TRL gives the policy
        return json.dumps({"error": str(exc)}), False


def run_episode(
    row: dict[str, Any],
    sample: int,
    endpoint: Endpoint,
    args: argparse.Namespace,
    meta: dict,
) -> Episode:
    task_id = row["benchflow_task_id"]
    episode = Episode(task_id, sample, meta.get("kind"), meta.get("level"))
    started = time.monotonic()
    env = BenchFlowRuntimeEnvironment(
        harness_config(environment=args.sandbox, jobs_dir=args.out / "jobs")
    )
    messages = [dict(message) for message in row["prompt"]]
    observation = env.reset(**row)
    if observation:
        messages[-1]["content"] += observation
    drop: RewardDecision | None = None
    try:
        for turn in range(args.max_turns + 1):
            if env.decision is not None and env.decision.dropped:
                episode.ended = "sandbox_start"
                break
            try:
                reply = endpoint.chat(messages)
            except ContextExhausted:
                episode.ended = "context_exhausted"
                break
            usage = reply.get("usage") or {}
            episode.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            episode.completion_tokens += int(usage.get("completion_tokens") or 0)
            message = (reply.get("choices") or [{}])[0].get("message") or {}
            calls = message.get("tool_calls") or []
            assistant = {"role": "assistant", "content": message.get("content") or ""}
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            episode.turns += 1
            if not calls:
                episode.ended = "no_tool_call"
                break
            if turn == args.max_turns:
                episode.ended = "turn_limit"
                break
            done = False
            for call in calls:
                episode.tool_calls += 1
                result, done = _tool_result(env, call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": result,
                    }
                )
                if done:
                    break
            if done:
                episode.ended = "submitted"
                break
    except EndpointError as exc:
        episode.ended = "model_endpoint"
        drop = model_endpoint_failure(exc)
    episode.decision = finish_rollout(env, messages, drop=drop)
    episode.messages = messages
    episode.elapsed_sec = time.monotonic() - started
    return episode


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise(episodes: list[Episode], args: argparse.Namespace) -> dict[str, Any]:
    decisions = [e.decision for e in episodes if e.decision is not None]
    base = summarize(decisions)
    kept = [d.reward for d in decisions if d.reward is not None]
    solved = sum(1 for r in kept if r >= 1.0)
    by_kind: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[Episode]] = defaultdict(list)
    for e in episodes:
        groups[e.kind or "unknown"].append(e)
    for kind, group in sorted(groups.items()):
        kind_kept = [
            e.decision.reward
            for e in group
            if e.decision and e.decision.reward is not None
        ]
        kind_solved = sum(1 for r in kind_kept if r >= 1.0)
        by_kind[kind] = {
            "episodes": len(group),
            "kept": len(kind_kept),
            "solve_rate": kind_solved / len(kind_kept) if kind_kept else None,
            "ci95": wilson(kind_solved, len(kind_kept)),
        }
    return {
        "model": args.model,
        "base_url": args.base_url,
        "tasks_dir": str(args.tasks_dir),
        "sandbox": args.sandbox,
        "episodes": len(episodes),
        "kept": base["kept"],
        "dropped": base["dropped"],
        "drop_reasons": base["drop_reasons"],
        "zero_reasons": base["zero_reasons"],
        "flagged": base["flagged"],
        "solved": solved,
        "solve_rate": solved / len(kept) if kept else None,
        "ci95": wilson(solved, len(kept)),
        "mean_reward": base["mean_reward"],
        "by_kind": by_kind,
        "ended": dict(Counter(e.ended for e in episodes)),
        "tokens": {
            "prompt": sum(e.prompt_tokens for e in episodes),
            "completion": sum(e.completion_tokens for e in episodes),
        },
        "mean_turns": (sum(e.turns for e in episodes) / len(episodes))
        if episodes
        else None,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_turns": args.max_turns,
            "samples_per_task": args.samples,
        },
    }


def _task_meta(tasks_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = tasks_dir / "manifest.jsonl"
    if not manifest.is_file():
        return {}
    rows = [
        json.loads(line) for line in manifest.read_text().splitlines() if line.strip()
    ]
    return {row["task"]: row for row in rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument(
        "--base-url", required=True, help="OpenAI-compatible base URL, ending in /v1"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--api-key-env", default="OPENAI_API_KEY", help="env var holding the key"
    )
    parser.add_argument(
        "--header", action="append", default=[], help="extra 'Name: value' header"
    )
    parser.add_argument("--sandbox", choices=["daytona", "docker"], default="daytona")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1, help="episodes per task")
    parser.add_argument("--limit", type=int, help="evaluate only the first N tasks")
    parser.add_argument(
        "--include", action="append", default=[], help="task id to include"
    )
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--max-tokens", type=int, default=1024, help="per model call")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--extra-body", help="JSON merged into each request body")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--owner", help="Daytona owner label (BENCHFLOW_DAYTONA_OWNER)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(
            f"error: set {args.api_key_env} to the endpoint's API key", file=sys.stderr
        )
        return 2
    if args.owner:
        os.environ["BENCHFLOW_DAYTONA_OWNER"] = args.owner
    args.out.mkdir(parents=True, exist_ok=True)
    spec = BenchFlowSpec(tasks_dir=args.tasks_dir, include_tasks=args.include)
    rows = (
        list(spec.train_dataset_rows)[: args.limit]
        if args.limit
        else list(spec.train_dataset_rows)
    )
    meta = _task_meta(args.tasks_dir)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["tasks"] = len(rows)
    (args.out / "config.json").write_text(json.dumps(config, indent=1) + "\n")

    endpoint = Endpoint(args, api_key)
    episodes: list[Episode] = []
    lock = threading.Lock()
    work = [(row, sample) for row in rows for sample in range(args.samples)]
    print(
        f"evaluating {args.model} on {len(rows)} tasks x {args.samples} sample(s), "
        f"{args.sandbox}, concurrency {args.concurrency}",
        flush=True,
    )
    with (
        (args.out / "episodes.jsonl").open("w") as out,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool,
    ):
        futures = {
            pool.submit(
                run_episode,
                row,
                sample,
                endpoint,
                args,
                meta.get(row["benchflow_task_id"], {}),
            ): row
            for row, sample in work
        }
        for future in concurrent.futures.as_completed(futures):
            episode = future.result()
            with lock:
                episodes.append(episode)
                out.write(json.dumps(episode.row()) + "\n")
                out.flush()
                d = episode.decision
                print(
                    f"[{len(episodes)}/{len(work)}] {episode.task_id} "
                    f"{d.reason if d else '?'} reward={d.reward if d else None} "
                    f"turns={episode.turns} ended={episode.ended}",
                    flush=True,
                )
    summary = summarise(episodes, args)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    ci = summary["ci95"]
    rate = summary["solve_rate"]
    print(
        f"solve rate {rate:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f}) over {summary['kept']} kept "
        f"episodes; dropped {summary['dropped']} {summary['drop_reasons']}"
        if rate is not None and ci is not None
        else f"no kept episodes; dropped {summary['dropped']} {summary['drop_reasons']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
