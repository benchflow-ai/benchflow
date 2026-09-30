"""Held-out evaluation of Tinker policies on BenchFlow tasks, with 95% intervals.

    uv run --no-sync python docs/examples/rl/tinker/tinker_eval.py \\
        --tasks-dir <test tasks> --model Qwen/Qwen3.6-35B-A3B \\
        --policy base --policy tinker://<run>:train:0/sampler_weights/final \\
        --samples 4 --out runs/eval.json

Every policy runs `--samples` episodes of every task through the training
harness (tinker_env.py: same tools, prompt, limits and training rule), sampled
from Tinker at `--temperature`. `base` is the untrained base model. The score
is pass@1 with a two-stage bootstrap interval (tinker_stats.py); the first
policy is the baseline for paired differences. Infrastructure drops are
replaced while the retry budget lasts and are reported, never scored.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tinker  # noqa: E402
import tinker_env as te  # noqa: E402
import tinker_stats  # noqa: E402
from tinker_cookbook.completers import TinkerTokenCompleter  # noqa: E402
from tinker_cookbook.exceptions import AllTrajectoriesFailedError  # noqa: E402
from tinker_cookbook.rl.rollouts import do_group_rollout  # noqa: E402
from tinker_episode import close_all_live  # noqa: E402
from tinker_train import add_env_args, env_config  # noqa: E402

log = logging.getLogger("tinker_eval")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tasks-dir", required=True, type=Path, help="the test split")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        help="`base`, or a tinker:// sampler checkpoint; repeat to compare",
    )
    parser.add_argument(
        "--label", action="append", default=[], help="a name per --policy"
    )
    parser.add_argument(
        "--samples", type=int, default=4, help="episodes per task per policy"
    )
    parser.add_argument(
        "--max-retries", type=int, default=2, help="replacements per task"
    )
    parser.add_argument("--out", required=True, type=Path, help="the result JSON")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    add_env_args(parser)
    args = parser.parse_args(argv)
    if args.label and len(args.label) != len(args.policy):
        parser.error("give one --label per --policy, or none")
    return args


def reasons_of(builder: te.BenchFlowEnvGroupBuilder) -> Counter[str]:
    """Decision reasons and endings of the episodes that ran (spares aside)."""
    counts: Counter[str] = Counter()
    for env in builder.envs:
        decision = env.episode.decision
        if env.episode.started and decision is not None and not decision.dropped:
            counts[f"{decision.reason}/{env.episode.ended}"] += 1
    return counts


async def evaluate_policy(
    ref: str,
    label: str,
    tasks: list[te.TaskSpec],
    config: te.EnvConfig,
    args: argparse.Namespace,
    service: tinker.ServiceClient,
) -> dict[str, Any]:
    if ref == "base":
        client = await service.create_sampling_client_async(base_model=args.model)
    else:
        client = await service.create_sampling_client_async(model_path=ref)
    policy = TinkerTokenCompleter(
        client, max_tokens=config.max_tokens, temperature=args.temperature
    )
    strategy = te.DropInfrastructureFailures(
        max_retries=args.max_retries, min_group_size=1
    )
    drops_before = Counter(te.DROPS.counts)

    async def one(
        task: te.TaskSpec,
    ) -> tuple[str, list[float], Counter[str], list[dict]]:
        builder = te.BenchFlowEnvGroupBuilder(
            task, group_size=args.samples, config=config, job_name=f"eval-{label}"
        )
        try:
            group = await do_group_rollout(builder, policy, strategy=strategy)
            rewards = group.get_total_rewards()
        except AllTrajectoriesFailedError:
            rewards = []
        episodes = [
            {
                "reason": env.episode.decision.reason,
                "ended": env.episode.ended,
                "reward": env.episode.decision.reward,
                "rollout_dir": str(env.episode.rollout_dir)
                if env.episode.rollout_dir
                else None,
            }
            for env in builder.envs
            if env.episode.started and env.episode.decision is not None
        ]
        return task.name, rewards, reasons_of(builder), episodes

    done = await asyncio.gather(*(one(task) for task in tasks))
    results = {name: rewards for name, rewards, _, _ in done}
    exits: Counter[str] = Counter()
    for _, _, counts, _ in done:
        exits.update(counts)
    summary = tinker_stats.summarize(results, seed=args.seed)
    log.info(
        "%s: solve rate %.3f [%.3f, %.3f] over %d tasks, %d episodes",
        label,
        summary["solve_rate"],
        summary["ci95_low"],
        summary["ci95_high"],
        summary["tasks"],
        summary["episodes"],
    )
    return {
        "policy": ref,
        "label": label,
        "summary": summary,
        "rewards": results,
        "reasons": dict(exits),
        "infrastructure_drops": dict(Counter(te.DROPS.counts) - drops_before),
        "episodes": {name: eps for name, _, _, eps in done},
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks = te.load_tasks(args.tasks_dir, include=args.include, exclude=args.exclude)
    config = env_config(args, args.out.parent / "trials")
    labels = args.label or [
        "base" if ref == "base" else ref.rstrip("/").rsplit("/", 1)[-1]
        for ref in args.policy
    ]
    service = tinker.ServiceClient(base_url=args.base_url)
    evaluations = []
    try:
        for ref, label in zip(args.policy, labels, strict=True):
            evaluations.append(
                await evaluate_policy(ref, label, tasks, config, args, service)
            )
    finally:
        left = await close_all_live()
        if left:
            log.warning("closed %d sandboxes left open", left)
    comparisons = [
        {
            "before": evaluations[0]["label"],
            "after": other["label"],
            **tinker_stats.paired_difference(
                evaluations[0]["rewards"], other["rewards"], seed=args.seed
            ),
        }
        for other in evaluations[1:]
    ]
    return {
        "tasks_dir": str(args.tasks_dir),
        "tasks": [t.name for t in tasks],
        "model": args.model,
        "renderer": config.renderer_name,
        "samples": args.samples,
        "temperature": args.temperature,
        "evaluations": evaluations,
        "comparisons": comparisons,
        "infrastructure_drops": dict(te.DROPS.counts),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    te.configure(
        max_sandboxes=args.max_sandboxes,
        drops_path=args.out.with_name(args.out.stem + "-drops.jsonl"),
        groups_path=args.out.with_name(args.out.stem + "-groups.jsonl"),
    )
    doc = asyncio.run(run(args))
    args.out.write_text(json.dumps(doc, indent=1, default=str))
    for ev in doc["evaluations"]:
        s = ev["summary"]
        print(
            f"{ev['label']}: solve rate {s['solve_rate']:.3f} "
            f"(95% CI {s['ci95_low']:.3f}-{s['ci95_high']:.3f}), "
            f"{s['solved']}/{s['episodes']} episodes over {s['tasks']} tasks; reasons {ev['reasons']}"
        )
    for c in doc["comparisons"]:
        print(
            f"{c['after']} - {c['before']}: {c['delta']:+.3f} "
            f"(95% CI {c['low']:+.3f} to {c['high']:+.3f}) over {c['tasks']} tasks"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
