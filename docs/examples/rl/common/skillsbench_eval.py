"""Held-out evaluation on SkillsBench: several policies ("arms"), one harness.

    python docs/examples/rl/common/skillsbench_eval.py --tasks-dir tasks \\
        --arms arms.json --servers http://localhost:8000/v1,http://localhost:8001/v1 \\
        --samples 2 --concurrency 64 --owner sbh-main --out results/

Every arm is scored with rl-core's shared evaluator (``evaluate.py``): the same
OpenAI-compatible client with retries, tool-call recovery, endings, and the
TRL adapter's verification and attribution rule. The episode loop below is
``evaluate.run_episode`` with these changes, all the same for every arm:

1. Per-task budgets from the task's own ``agent.timeout_sec`` (T): a turn cap
   of round(T / 30) clamped to [--min-turns, --max-turns], and a per-command
   timeout of round(T / 10) clamped to [60, 300] seconds. An episode that runs
   longer than min(2 T, --episode-cap) seconds of wall clock ends
   ("episode_timeout") and is verified as left.
2. The task's skills: the sandbox gets ``environment/skills`` the way BenchFlow
   gives them to an agent (``skill_mode="with-skill"``: copied to /skills and
   linked into the agents' skill-discovery paths), and the harness message
   lists each skill's name and description, as agent harnesses list skills.
3. The harness message names the task's working directory and the command
   timeout, and says that the files left in the sandbox are what gets checked;
   ``submit`` writes its answer to /tmp/answer.txt (outside the workspace).
4. Tool output is cut at --max-output-chars (training used 2,000).
5. Cached prompt tokens and reasoning characters are recorded per episode.
6. Work runs from one queue for all arms, ordered by sample, then task, then
   arm, so a stop at --stop-at leaves the arms with the same tasks done. An
   episode still running at --stop-at is dropped as "cutoff" (not scored).

Writes ``<out>/<arm>/episodes.jsonl`` (one line per finished episode; reruns
skip what is there), ``<out>/<arm>/jobs/`` (rollout folders and
``rollouts.jsonl`` with every message), and ``<out>/config.json``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate as ev  # noqa: E402  rl-core's shared evaluator
from harness import harness_config, shorten_daytona_lifetimes, sweep_daytona  # noqa: E402

from benchflow.integrations.rewards import model_endpoint_failure  # noqa: E402
from benchflow.integrations.trl import (  # noqa: E402
    BenchFlowRuntimeEnvironment,
    BenchFlowSpec,
    finish_rollout,
)

MESSAGE = (
    "\n\nYou are working in a Linux sandbox. Use the run_bash tool to run shell "
    "commands; each call starts in {workspace} and is stopped after {timeout} seconds. "
    "The files you leave in the sandbox are what gets checked. When you are done, "
    "call the submit tool once with a short final answer, or the word done: it ends "
    "the task."
)
SKILLS = (
    "\n\nSkills for this task are installed in {skills_dir}. Each skill is a folder "
    "with a SKILL.md that explains when and how to use it, and may include scripts "
    "and reference files. Read a skill's SKILL.md before you rely on it. Available "
    "skills:\n{listing}"
)
CUTOFF = threading.Event()
LOCK = threading.Lock()


def frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text()
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    return (yaml.safe_load(match.group(1)) or {}) if match else {}


def task_settings(task_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    fm = frontmatter(task_dir / "task.md")
    timeout = float((fm.get("agent") or {}).get("timeout_sec") or 900)
    sandbox = fm.get("sandbox") or {}
    skills = []
    root = task_dir / "environment" / "skills"
    for skill_md in sorted(root.glob("*/SKILL.md")):
        meta = frontmatter(skill_md)
        name = str(meta.get("name") or skill_md.parent.name)
        description = " ".join(str(meta.get("description") or "").split())
        skills.append((skill_md.parent.name, name, description))
    return {
        "agent_timeout_sec": timeout,
        "max_turns": max(args.min_turns, min(args.max_turns, round(timeout / 30))),
        "bash_timeout_sec": max(60, min(300, round(timeout / 10))),
        "wall_budget_sec": min(2 * timeout, args.episode_cap),
        "skills_dir": str(sandbox.get("skills_dir") or "/skills").rstrip("/"),
        "skills": skills,
        "category": (fm.get("metadata") or {}).get("category"),
        "difficulty": (fm.get("metadata") or {}).get("difficulty"),
    }


def harness_message(settings: dict[str, Any], workspace: str) -> str:
    text = MESSAGE.format(workspace=workspace, timeout=settings["bash_timeout_sec"])
    if settings["skills"]:
        listing = "\n".join(
            f"- {folder}: {description}" if description else f"- {folder}"
            for folder, _name, description in settings["skills"]
        )
        text += SKILLS.format(skills_dir=settings["skills_dir"], listing=listing)
    return text


class Arm:
    def __init__(self, spec: dict[str, Any], servers: list[str], args: argparse.Namespace) -> None:
        self.name = spec["name"]
        self.model = spec["model"]
        self.max_tokens = int(spec["max_tokens"])
        self.extra_body = spec.get("extra_body")
        self.out = args.out / self.name
        self.endpoints = []
        for url in servers:
            ns = argparse.Namespace(
                base_url=url,
                header=[],
                request_timeout=args.request_timeout,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                extra_body=json.dumps(self.extra_body) if self.extra_body else None,
                retries=args.retries,
            )
            key = os.environ.get(spec.get("api_key_env") or "", "") or "EMPTY"
            self.endpoints.append(ev.Endpoint(ns, key))


class Gate:
    """At most ``limit`` episodes run at once; the limit can change during the run.

    It is read from ``<out>/concurrency`` (an integer) when that file exists, at most
    every 15 seconds, else ``--concurrency``. Lowering it lets running episodes finish.
    """

    def __init__(self, path: Path, default: int) -> None:
        self.path, self.default = path, default
        self.active = 0
        self.cond = threading.Condition()
        self._limit, self._read = default, 0.0

    def limit(self) -> int:
        now = time.monotonic()
        if now - self._read > 15:
            self._read = now
            try:
                self._limit = max(1, int(self.path.read_text().strip()))
            except (OSError, ValueError):
                self._limit = self.default
        return self._limit

    def acquire(self) -> bool:
        with self.cond:
            while self.active >= self.limit():
                if CUTOFF.is_set():
                    return False
                self.cond.wait(timeout=5)
            if CUTOFF.is_set():
                return False
            self.active += 1
            return True

    def release(self) -> None:
        with self.cond:
            self.active -= 1
            self.cond.notify_all()


class Servers:
    """Send each episode to the server with the fewest episodes in flight."""

    def __init__(self, n: int) -> None:
        self.active = [0] * n
        self.lock = threading.Lock()

    def take(self) -> int:
        with self.lock:
            index = min(range(len(self.active)), key=lambda i: (self.active[i], i))
            self.active[index] += 1
            return index

    def give(self, index: int) -> None:
        with self.lock:
            self.active[index] -= 1


def run_episode(
    row: dict[str, Any],
    sample: int,
    arm: Arm,
    server: int,
    settings: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """evaluate.run_episode with the changes listed in the module docstring."""

    task_id = row["benchflow_task_id"]
    episode = ev.Episode(task_id, sample, settings["category"], None)
    started = time.monotonic()
    endpoint = arm.endpoints[server]
    harness = harness_config(
        environment=args.sandbox,
        jobs_dir=arm.out / "jobs",
        bash_timeout_sec=settings["bash_timeout_sec"],
        max_output_chars=args.max_output_chars,
        submit_path="/tmp/answer.txt",
        reset_message=None,
        skill_mode="with-skill" if args.skills else "no-skill",
    )
    env = BenchFlowRuntimeEnvironment(harness)
    messages = [dict(message) for message in row["prompt"]]
    env.reset(**row)
    runtime = env._runtime
    workspace = runtime.workspace if runtime is not None else "the task's working directory"
    messages[-1]["content"] += harness_message(settings, workspace)
    cached = 0
    reasoning_chars = 0
    drop = None
    try:
        for turn in range(settings["max_turns"] + 1):
            if env.decision is not None and env.decision.dropped:
                episode.ended = "sandbox_start"
                break
            if CUTOFF.is_set():
                episode.ended = "cutoff"
                break
            if time.monotonic() - started > settings["wall_budget_sec"]:
                episode.ended = "episode_timeout"
                break
            try:
                reply = endpoint.chat(messages)
            except ev.ContextExhausted:
                episode.ended = "context_exhausted"
                break
            except ev.InvalidToolCall:
                episode.ended = "invalid_tool_call"
                break
            usage = reply.get("usage") or {}
            episode.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            episode.completion_tokens += int(usage.get("completion_tokens") or 0)
            cached += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            choice = (reply.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            reasoning_chars += len(
                str(message.get("reasoning_content") or message.get("reasoning") or "")
            )
            episode.truncated_calls += choice.get("finish_reason") == "length"
            calls = message.get("tool_calls") or []
            if not calls:
                calls = ev.recover_tool_calls(message)
                episode.recovered_calls += bool(calls)
            assistant = {"role": "assistant", "content": message.get("content") or ""}
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            episode.turns += 1
            if not calls:
                episode.ended = (
                    "cut_at_max_tokens"
                    if choice.get("finish_reason") == "length"
                    else "no_tool_call"
                )
                break
            if turn == settings["max_turns"]:
                episode.ended = "turn_limit"
                break
            done = False
            for call in calls:
                episode.tool_calls += 1
                result, done = ev._tool_result(env, call)
                messages.append(
                    {"role": "tool", "tool_call_id": call.get("id", ""), "content": result}
                )
                if done:
                    break
            if done:
                episode.ended = "submitted"
                break
    except ev.EndpointError as exc:
        episode.ended = "model_endpoint"
        drop = model_endpoint_failure(exc)
    if episode.ended == "cutoff":
        # Stopped by our deadline, not by the task or the policy: no score. The
        # sandbox is closed without verification and the episode is re-queued
        # on resume. (RewardDecision only allows infrastructure drop reasons.)
        env._close()
        episode.elapsed_sec = time.monotonic() - started
        row_out = episode.row()
        row_out.update(reward=None, dropped=True, reason="cutoff")
    else:
        episode.decision = finish_rollout(env, messages, drop=drop)
        episode.elapsed_sec = time.monotonic() - started
        row_out = episode.row()
    row_out.update(
        arm=arm.name,
        server=server,
        cached_prompt_tokens=cached,
        reasoning_chars=reasoning_chars,
        max_turns=settings["max_turns"],
        bash_timeout_sec=settings["bash_timeout_sec"],
        agent_timeout_sec=settings["agent_timeout_sec"],
        difficulty=settings["difficulty"],
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    return row_out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tasks-dir", type=Path, required=True)
    p.add_argument("--arms", type=Path, required=True, help="JSON list of arms")
    p.add_argument("--servers", required=True, help="comma-separated base URLs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sandbox", choices=["daytona", "docker"], default="daytona")
    p.add_argument("--owner", required=True, help="Daytona owner label")
    p.add_argument("--concurrency", type=int, default=64, help="episodes at once (see Gate)")
    p.add_argument("--max-workers", type=int, default=64, help="upper bound on --concurrency")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--include", action="append", default=[])
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--only-arms", default="")
    p.add_argument("--min-turns", type=int, default=20)
    p.add_argument("--max-turns", type=int, default=50)
    p.add_argument("--episode-cap", type=float, default=3600.0)
    p.add_argument("--max-output-chars", type=int, default=8000)
    p.add_argument("--no-skills", dest="skills", action="store_false")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--request-timeout", type=float, default=1800.0)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--stop-at", type=float, default=0.0, help="UNIX time to stop")
    p.add_argument("--cleanup-only", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["BENCHFLOW_DAYTONA_OWNER"] = args.owner
    if args.cleanup_only:
        print(f"swept {args.owner}: {sweep_daytona(args.owner)}", flush=True)
        return 0
    shorten_daytona_lifetimes()
    args.out.mkdir(parents=True, exist_ok=True)
    arm_specs = json.loads(args.arms.read_text())
    if args.only_arms:
        keep = set(args.only_arms.split(","))
        arm_specs = [a for a in arm_specs if a["name"] in keep]
    servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    arms = [Arm(spec, servers, args) for spec in arm_specs]
    spec = BenchFlowSpec(
        tasks_dir=args.tasks_dir, include_tasks=args.include, exclude_tasks=args.exclude
    )
    rows = {row["benchflow_task_id"]: row for row in spec.train_dataset_rows}
    settings = {tid: task_settings(Path(row["benchflow_task_dir"]), args) for tid, row in rows.items()}
    order = sorted(rows)
    random.Random(args.seed).shuffle(order)
    done: set[tuple[str, str, int]] = set()
    for arm in arms:
        arm.out.mkdir(parents=True, exist_ok=True)
        path = arm.out / "episodes.jsonl"
        if path.is_file():
            for line in path.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Re-run what our own stop or a harness error cut short.
                if r.get("ended") not in ("cutoff", "harness_error"):
                    done.add((arm.name, r["task_id"], int(r["sample"])))
    work = [
        (sample, tid, arm)
        for sample in range(args.samples)
        for tid in order
        for arm in arms
        if (arm.name, tid, sample) not in done
    ]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(arms=arm_specs, tasks=len(rows), order=order, settings=settings, queued=len(work))
    (args.out / "config.json").write_text(json.dumps(config, indent=1, default=str) + "\n")
    print(
        f"{len(rows)} tasks x {args.samples} samples x {len(arms)} arms: {len(work)} episodes "
        f"queued ({len(done)} already done), concurrency {args.concurrency}, servers {servers}",
        flush=True,
    )
    balance = Servers(len(servers))
    gate = Gate(args.out / "concurrency", args.concurrency)

    def one(item: tuple[int, str, Arm]) -> dict[str, Any] | None:
        sample, tid, arm = item
        if CUTOFF.is_set() or not gate.acquire():
            return None
        try:
            return _one(sample, tid, arm)
        finally:
            gate.release()

    def _one(sample: int, tid: str, arm: Arm) -> dict[str, Any] | None:
        server = balance.take()
        try:
            out = run_episode(rows[tid], sample, arm, server, settings[tid], args)
        except Exception as exc:  # never lose the queue to one episode
            out = {"task_id": tid, "sample": sample, "arm": arm.name, "ended": "harness_error",
                   "reward": None, "reason": f"{type(exc).__name__}: {exc}"[:500]}
        finally:
            balance.give(server)
        with LOCK:
            with (arm.out / "episodes.jsonl").open("a") as fh:
                fh.write(json.dumps(out, default=str) + "\n")
        print(
            f"{time.strftime('%H:%M:%S')} {arm.name} {tid} s{sample} reward={out.get('reward')} "
            f"turns={out.get('turns')} ended={out.get('ended')} {out.get('elapsed_sec')}s",
            flush=True,
        )
        return out

    def stop(signum: int, _frame: Any) -> None:
        print(f"signal {signum}: stopping", flush=True)
        CUTOFF.set()
        ev.STOP.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.stop_at:
        def watch() -> None:
            while not CUTOFF.is_set():
                if time.time() >= args.stop_at:
                    print("stop-at reached: stopping", flush=True)
                    CUTOFF.set()
                    return
                time.sleep(5)
        threading.Thread(target=watch, daemon=True).start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        list(pool.map(one, work))
    if args.sandbox == "daytona":
        print(f"swept {args.owner}: {sweep_daytona(args.owner)}", flush=True)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
