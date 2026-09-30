"""Multi-turn RL on Fireworks' serverless Training API, with BenchFlow tasks as the environment.

    python fireworks_rl.py screen --tasks-dir tasks/v1/train --limit 32 --out runs/screen
    python fireworks_rl.py train  --tasks-dir tasks/v1/train --tasks-file runs/screen/learnable.txt \\
        --steps 8 --output-model-id bf-rl-qwen38-demo --out runs/train

Fireworks runs the LoRA's forward and backward passes and serves its current
weights for sampling; this loop runs everything else on your machine. Each
episode is the shared harness that ``evaluate.py`` uses: the task prompt plus
``HARNESS_MESSAGE``, the ``run_bash`` and ``submit`` tools, one BenchFlow
sandbox (``BenchFlowRuntimeEnvironment``, Daytona or Docker), and the task's
own verifier. The policy is sampled token by token (``fireworks_chat``), so
every trained token is exactly what the policy sampled, with its logprob.

The reward is the BenchFlow verifier's, under the training rule of
``benchflow.integrations.rewards``: an infrastructure failure (the sandbox never
started, the model endpoint failed, the verifier crashed on an untouched
sandbox) is dropped from its group and counted; every other failure, timeouts
included, scores 0. Groups are GRPO groups: several episodes of one task,
advantages standardized within the group, groups whose kept rewards are all
equal skipped.

``screen`` samples each task a few times with the untrained weights and writes
``learnable.txt``: the tasks whose episodes disagree. With a pass/fail
verifier, a task that always passes or always fails gives GRPO nothing.

``train`` saves a sampler checkpoint each step, rolls out a batch of groups
with it, and takes one ``importance_sampling`` step. At the end it saves a
final checkpoint and promotes it to a private Fireworks model (a LoRA add-on),
which ``fireworks_deploy.py`` serves for ``evaluate.py``.

The Fireworks key is read from ``FIREWORKS_API_KEY`` and only reaches
Fireworks: sampling happens here, never in a sandbox.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))

from evaluate import Episode, _task_meta, _tool_result, wilson  # noqa: E402
from fireworks_chat import (  # noqa: E402
    QwenChat,
    Turn,
    checkpoint_name,
    datum_arrays,
    group_advantages,
    join_turns,
)
from harness import MAX_TURNS, harness_config  # noqa: E402

from benchflow.integrations.rewards import (  # noqa: E402
    RewardDecision,
    model_endpoint_failure,
    summarize,
)
from benchflow.integrations.trl import (  # noqa: E402
    BenchFlowRuntimeEnvironment,
    BenchFlowSpec,
    bash_tool_schemas,
    finish_rollout,
)

SERVERLESS_URL = "https://api.fireworks.ai/training/v1/serverless"
BASE_MODEL = "accounts/fireworks/models/qwen3p8-27b"
TOKENIZER = "Qwen/Qwen3.8-27B"
TOKENIZER_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
# fireworks.ai/pricing, serverless training, Qwen 3.8 27B, USD per 1M tokens,
# checked 2026-09-30: prefill, cached prefill, sample, train.
PRICES = {"prefill": 1.86, "cached_prefill": 0.372, "sample": 5.595, "train": 4.103}


class EndpointError(Exception):
    """Sampling failed in a way the policy could not have caused."""


class ContextExhausted(Exception):
    """The conversation no longer fits: the policy's budget ran out."""


@dataclass
class Rollout:
    """One episode plus the token record training needs."""

    episode: Episode
    turns: list[Turn] = field(default_factory=list)
    truncated_turns: int = 0

    @property
    def reward(self) -> float | None:
        decision = self.episode.decision
        return None if decision is None else decision.reward


class Policy:
    """The LoRA's current weights, sampled token by token through a Fireworks sampler."""

    def __init__(self, chat: QwenChat, sampler: Any, args: argparse.Namespace):
        import tinker

        self.chat = chat
        self.sampler = sampler
        self.args = args
        self.params = tinker.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stop=chat.stop_tokens,
        )

    def act(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[dict[str, Any], Turn]:
        import tinker

        prompt = self.chat.render(messages, tools)
        if len(prompt) + self.args.max_tokens > self.args.max_seq_len:
            raise ContextExhausted(f"{len(prompt)} prompt tokens")
        last = "no attempt"
        for attempt in range(self.args.retries + 1):
            if attempt:
                time.sleep(min(60.0, 2.0**attempt) * (0.5 + random.random()))
            try:
                result = self.sampler.sample(
                    prompt=tinker.ModelInput.from_ints(prompt),
                    num_samples=1,
                    sampling_params=self.params,
                ).result(timeout=self.args.request_timeout)
            except Exception as exc:  # transport, 429/5xx, an expired session
                last = f"{type(exc).__name__}: {str(exc)[:300]}"
                continue
            sequence = result.sequences[0]
            tokens = [int(t) for t in sequence.tokens]
            logprobs = [float(x) for x in (sequence.logprobs or [])]
            if not tokens or len(logprobs) != len(tokens):
                last = f"sampler returned {len(tokens)} tokens and {len(logprobs)} logprobs"
                continue
            return self.chat.parse(tokens), Turn(prompt, tokens, logprobs)
        raise EndpointError(last)


def run_episode(row: dict[str, Any], sample: int, policy: Policy, args: argparse.Namespace, meta: dict) -> Rollout:
    """``evaluate.run_episode`` with the Training API's sampler as the model."""

    episode = Episode(row["benchflow_task_id"], sample, meta.get("kind"), meta.get("level"))
    rollout = Rollout(episode)
    started = time.monotonic()
    env = BenchFlowRuntimeEnvironment(
        harness_config(environment=args.sandbox, jobs_dir=args.out / "jobs", background_start=True)
    )
    tools = bash_tool_schemas()
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
                message, record = policy.act(messages, tools)
            except ContextExhausted:
                episode.ended = "context_exhausted"
                break
            rollout.turns.append(record)
            episode.prompt_tokens += len(record.prompt)
            episode.completion_tokens += len(record.completion)
            if len(record.completion) >= args.max_tokens:
                rollout.truncated_turns += 1
            messages.append(message)
            episode.turns += 1
            calls = message.get("tool_calls") or []
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
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})
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
    return rollout


def run_groups(
    rows: list[dict[str, Any]], policy: Policy, args: argparse.Namespace, meta: dict, log: Any, step: int
) -> list[list[Rollout]]:
    """``group_size`` episodes of each row, ``concurrency`` sandboxes at a time."""

    groups: list[list[Rollout | None]] = [[None] * args.group_size for _ in rows]
    lock = threading.Lock()
    work = [(g, s) for g in range(len(rows)) for s in range(args.group_size)]
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(run_episode, rows[g], s, policy, args, meta.get(rows[g]["benchflow_task_id"], {})): (g, s)
            for g, s in work
        }
        for future in concurrent.futures.as_completed(futures):
            g, s = futures[future]
            rollout = future.result()
            with lock:
                groups[g][s] = rollout
                done += 1
                episode = rollout.episode
                d = episode.decision
                row = {
                    "step": step,
                    **episode.row(),
                    "truncated_turns": rollout.truncated_turns,
                    "sequences": len(join_turns(rollout.turns)),
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(
                    f"  [{done}/{len(work)}] {episode.task_id} {d.reason if d else '?'} "
                    f"reward={d.reward if d else None} turns={episode.turns} ended={episode.ended}",
                    flush=True,
                )
    return [[r for r in group if r is not None] for group in groups]


class Session:
    """One serverless training session: a LoRA run plus its samplers."""

    def __init__(self, args: argparse.Namespace):
        from fireworks.training.sdk import FiretitanServiceClient
        from transformers import AutoTokenizer

        key = os.environ.get("FIREWORKS_API_KEY", "")
        if not key:
            raise SystemExit("error: set FIREWORKS_API_KEY")
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
        self.chat = QwenChat(self.tokenizer)
        self.service = FiretitanServiceClient(api_key=key, base_url=SERVERLESS_URL)
        self.client = self.service.create_lora_training_client(
            base_model=args.base_model, rank=args.lora_rank, seed=args.seed
        )
        self.session_id = self.service.training_session_id
        self.session_name = self.service.training_session_name
        self.run_id = self.client.run_id
        print(f"session {self.session_id} run {self.run_id}", flush=True)

    def policy(self, name: str) -> tuple[Policy, Any, str]:
        """Save the current weights as a sampler checkpoint and sample from it."""

        path = self.client.save_weights_for_sampler(name).result().path
        sampler = self.service.create_sampling_client(model_path=path, tokenizer=self.tokenizer)
        return Policy(self.chat, sampler, self.args), sampler, path

    def train_step(self, datums: list[Any], learning_rate: float) -> dict[str, Any]:
        import tinker

        fb = self.client.forward_backward(datums, "importance_sampling").result()
        self.client.optim_step(
            tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)
        ).result()
        return dict(getattr(fb, "metrics", None) or {})

    def promote(self, name: str, output_model_id: str) -> dict[str, Any]:
        """Promote the named sampler checkpoint to a private Fireworks model."""

        from fireworks.training.sdk import FireworksClient

        fw = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"])
        rows = fw.list_training_session_checkpoints(self.session_name)
        wanted = f"-{name}-"
        candidates = [r for r in rows if r.get("promotable") and wanted in str(r.get("checkpointName", r.get("name")))]
        if not candidates:
            raise RuntimeError(f"no promotable checkpoint named {name!r} in {self.session_name}")
        target = max(candidates, key=lambda r: r.get("createTime", ""))
        return fw.promote_session_checkpoint(
            name=target["name"], output_model_id=output_model_id, base_model=self.args.base_model
        )

    def close(self) -> None:
        close = getattr(self.service, "close", None)
        if callable(close):
            close()


def to_datums(groups: list[list[Rollout]]) -> tuple[list[Any], dict[str, Any]]:
    """Importance-sampling datums for every group with a reward spread."""

    import tinker

    datums: list[Any] = []
    stats = Counter()
    for group in groups:
        advantages = group_advantages([r.reward for r in group])
        if all(a in (None, 0.0) for a in advantages):
            stats["groups_without_signal"] += 1
            continue
        stats["groups_with_signal"] += 1
        for rollout, advantage in zip(group, advantages, strict=True):
            if advantage is None or advantage == 0.0 or not rollout.turns:
                continue
            for sequence in join_turns(rollout.turns):
                arrays = datum_arrays(sequence, advantage)
                datums.append(
                    tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(arrays["input_tokens"]),
                        loss_fn_inputs={
                            "target_tokens": arrays["target_tokens"],
                            "logprobs": arrays["logprobs"],
                            "advantages": arrays["advantages"],
                        },
                    )
                )
                stats["datums"] += 1
                stats["train_tokens"] += len(arrays["input_tokens"])
                stats["sampled_tokens_trained"] += sum(1 for a in arrays["advantages"] if a != 0.0)
    return datums, dict(stats)


def rollout_cost(groups: list[list[Rollout]]) -> dict[str, float]:
    """Token meters of a batch of rollouts; prefill is billed as if never cached."""

    prefill = sum(len(t.prompt) for g in groups for r in g for t in r.turns)
    sample = sum(len(t.completion) for g in groups for r in g for t in r.turns)
    return {"prefill_tokens": prefill, "sample_tokens": sample}


def usd(prefill: int, sample: int, train: int) -> float:
    """An upper bound in USD: prefill at the uncached rate."""

    return (prefill * PRICES["prefill"] + sample * PRICES["sample"] + train * PRICES["train"]) / 1e6


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict]:
    include: list[str] = list(args.include)
    if args.tasks_file:
        include += [t.strip() for t in args.tasks_file.read_text().splitlines() if t.strip()]
    spec = BenchFlowSpec(tasks_dir=args.tasks_dir, include_tasks=include)
    rows = list(spec.train_dataset_rows)
    if args.limit:
        rows = rows[: args.limit]
    return rows, _task_meta(args.tasks_dir)


def summary_of(rollouts: list[Rollout]) -> dict[str, Any]:
    decisions = [r.episode.decision for r in rollouts if r.episode.decision is not None]
    base = summarize(decisions)
    kept = [d.reward for d in decisions if d.reward is not None]
    solved = sum(1 for r in kept if r >= 1.0)
    return {
        "episodes": len(rollouts),
        "kept": base["kept"],
        "dropped": base["dropped"],
        "drop_reasons": base["drop_reasons"],
        "zero_reasons": base["zero_reasons"],
        "mean_reward": base["mean_reward"],
        "solve_rate": solved / len(kept) if kept else None,
        "ci95": wilson(solved, len(kept)),
        "ended": dict(Counter(r.episode.ended for r in rollouts)),
        "truncated_turns": sum(r.truncated_turns for r in rollouts),
        "turns": sum(r.episode.turns for r in rollouts),
    }


def cmd_screen(args: argparse.Namespace) -> int:
    rows, meta = load_rows(args)
    print(f"screening {len(rows)} tasks x {args.group_size} with the untrained weights", flush=True)
    session = Session(args)
    started = time.monotonic()
    try:
        policy, sampler, path = session.policy(checkpoint_name("base", 0))
        try:
            with (args.out / "episodes.jsonl").open("a") as log:
                groups = run_groups(rows, policy, args, meta, log, step=0)
        finally:
            sampler.close()
    finally:
        session.close()
    per_task = {}
    learnable = []
    for row, group in zip(rows, groups, strict=True):
        kept = [r.reward for r in group if r.reward is not None]
        mean = sum(kept) / len(kept) if kept else None
        per_task[row["benchflow_task_id"]] = {"kept": len(kept), "mean_reward": mean, "rewards": [r.reward for r in group]}
        if len(kept) >= 2 and len(set(kept)) > 1:
            learnable.append(row["benchflow_task_id"])
    meters = rollout_cost(groups)
    record = {
        "session": session.session_id,
        "snapshot": path,
        "base_model": args.base_model,
        "tasks": len(rows),
        "group_size": args.group_size,
        "learnable": learnable,
        **summary_of([r for g in groups for r in g]),
        **meters,
        "usd_upper_bound": round(usd(meters["prefill_tokens"], meters["sample_tokens"], 0), 4),
        "elapsed_sec": round(time.monotonic() - started, 1),
        "per_task": per_task,
    }
    (args.out / "screen.json").write_text(json.dumps(record, indent=1) + "\n")
    (args.out / "learnable.txt").write_text("".join(f"{t}\n" for t in learnable))
    print(
        f"screen: {len(learnable)}/{len(rows)} tasks learnable; mean reward {record['mean_reward']}; "
        f"about ${record['usd_upper_bound']} (upper bound)",
        flush=True,
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    rows, meta = load_rows(args)
    if not rows:
        raise SystemExit("error: no tasks to train on")
    rng = random.Random(args.seed)
    order = rows[:]
    rng.shuffle(order)
    batches = [
        [order[(step * args.groups_per_step + i) % len(order)] for i in range(args.groups_per_step)]
        for step in range(args.steps)
    ]
    session = Session(args)
    spent = 0.0
    per_step_cost: list[float] = []
    record: dict[str, Any] = {
        "session": session.session_id,
        "session_name": session.session_name,
        "run": session.run_id,
        "base_model": args.base_model,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"},
        "steps": [],
    }
    try:
        with (args.out / "episodes.jsonl").open("a") as log, (args.out / "metrics.jsonl").open("a") as metrics:
            for step, batch in enumerate(batches):
                estimate = max(per_step_cost) if per_step_cost else args.first_step_usd
                if spent + estimate > args.max_usd:
                    print(f"stopping before step {step}: ${spent:.2f} spent, next step about ${estimate:.2f}", flush=True)
                    record["stopped"] = f"budget: ${spent:.2f} + ${estimate:.2f} > ${args.max_usd}"
                    break
                t0 = time.monotonic()
                policy, sampler, path = session.policy(checkpoint_name("s", step))
                try:
                    groups = run_groups(batch, policy, args, meta, log, step)
                finally:
                    sampler.close()
                t1 = time.monotonic()
                datums, stats = to_datums(groups)
                fb_metrics: dict[str, Any] = {}
                if datums:
                    fb_metrics = session.train_step(datums, args.learning_rate)
                t2 = time.monotonic()
                meters = rollout_cost(groups)
                cost = usd(meters["prefill_tokens"], meters["sample_tokens"], stats.get("train_tokens", 0))
                spent += cost
                per_step_cost.append(cost)
                row = {
                    "step": step,
                    "snapshot": path,
                    "tasks": [r["benchflow_task_id"] for r in batch],
                    **summary_of([r for g in groups for r in g]),
                    **stats,
                    **meters,
                    "trained": bool(datums),
                    "fb_metrics": {k: v for k, v in fb_metrics.items() if isinstance(v, int | float)},
                    "usd_upper_bound": round(cost, 4),
                    "usd_total_upper_bound": round(spent, 4),
                    "rollout_sec": round(t1 - t0, 1),
                    "train_sec": round(t2 - t1, 1),
                }
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                record["steps"].append(row)
                print(
                    f"step {step}: mean reward {row['mean_reward']} over {row['kept']} kept "
                    f"({row['dropped']} dropped), {stats.get('groups_with_signal', 0)} groups with signal, "
                    f"{stats.get('datums', 0)} datums, ${cost:.2f} (total ${spent:.2f})",
                    flush=True,
                )
        final = checkpoint_name("final", len(record["steps"]))
        record["final_snapshot"] = session.client.save_weights_for_sampler(final).result().path
        if args.output_model_id and any(step["trained"] for step in record["steps"]):
            record["promoted"] = session.promote(final, args.output_model_id)
            print(f"promoted {final} to {args.output_model_id}", flush=True)
        elif args.output_model_id:
            print("not promoting: no step changed the weights", flush=True)
    finally:
        record["usd_total_upper_bound"] = round(spent, 4)
        (args.out / "train.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
        session.close()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("screen", cmd_screen), ("train", cmd_train)):
        p = sub.add_parser(name)
        p.set_defaults(func=func)
        p.add_argument("--tasks-dir", type=Path, required=True)
        p.add_argument("--tasks-file", type=Path, help="task ids, one per line (screen's learnable.txt)")
        p.add_argument("--include", action="append", default=[], help="task id to include")
        p.add_argument("--limit", type=int, help="only the first N tasks")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--base-model", default=BASE_MODEL)
        p.add_argument("--tokenizer", default=TOKENIZER)
        p.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
        p.add_argument("--lora-rank", type=int, default=32)
        p.add_argument("--group-size", type=int, default=4, help="episodes per task (GRPO group)")
        p.add_argument("--sandbox", choices=["daytona", "docker"], default="daytona")
        p.add_argument("--concurrency", type=int, default=8, help="sandboxes at a time")
        p.add_argument("--max-turns", type=int, default=MAX_TURNS)
        p.add_argument("--max-tokens", type=int, default=1024, help="per model call, as evaluate.py")
        p.add_argument("--max-seq-len", type=int, default=32768)
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--retries", type=int, default=4)
        p.add_argument("--request-timeout", type=float, default=600.0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--owner", help="Daytona owner label (BENCHFLOW_DAYTONA_OWNER)")
        if name == "train":
            p.add_argument("--steps", type=int, default=8)
            p.add_argument("--groups-per-step", type=int, default=8, help="tasks per step")
            p.add_argument("--learning-rate", type=float, default=4e-5)
            p.add_argument("--max-usd", type=float, default=60.0, help="stop before a step that would pass it")
            p.add_argument("--first-step-usd", type=float, default=6.0, help="step 0's cost guess, for --max-usd")
            p.add_argument("--output-model-id", help="promote the final checkpoint to this model id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.owner:
        os.environ["BENCHFLOW_DAYTONA_OWNER"] = args.owner
    args.out.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
