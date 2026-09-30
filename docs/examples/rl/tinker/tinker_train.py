"""Train a LoRA policy on BenchFlow tasks with Tinker. See README.md.

    uv run --no-sync python docs/examples/rl/tinker/tinker_train.py \\
        --tasks-dir <train tasks> --model Qwen/Qwen3.6-35B-A3B --steps 20 \\
        --log-path runs/tinker-train

tinker-cookbook's `rl.train.main` runs the loop: each step samples
`--groups-per-batch` tasks, runs `--group-size` episodes of each on BenchFlow
sandboxes (tinker_env.py), centers each episode's reward on its group's mean,
and takes one LoRA update. Checkpoints go to Tinker (`checkpoints.jsonl` in
the log path lists their tinker:// paths); metrics to `metrics.jsonl`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tinker_env as te  # noqa: E402
from tinker_cookbook import model_info  # noqa: E402
from tinker_cookbook.rl import train as rl_train  # noqa: E402
from tinker_episode import (  # noqa: E402
    BASH_TIMEOUT_SEC,
    HARNESS_MESSAGE,
    MAX_OUTPUT_CHARS,
    MAX_TURNS,
    SUBMIT_PATH,
    EpisodeSettings,
    close_all_live,
)

log = logging.getLogger("tinker_train")

# USD per million tokens (prefill, sample, train), from
# https://tinker-docs.thinkingmachines.ai/tinker/models/ on 2026-09-30.
PRICES = {
    "Qwen/Qwen3.6-35B-A3B": (0.54, 1.335, 1.177),
    "Qwen/Qwen3.5-9B-Base": (0.66, 1.995, 1.463),
    "openai/gpt-oss-20b": (0.18, 0.45, 0.396),
    "openai/gpt-oss-120b": (0.33, 0.84, 0.737),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": (0.39, 0.99, 0.88),
}


def add_env_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by training and evaluation (tinker_eval.py)."""
    g = parser.add_argument_group("model and harness")
    g.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B", help="Tinker base model")
    g.add_argument(
        "--renderer",
        default=None,
        help="tinker-cookbook renderer (default: the model's recommended one)",
    )
    g.add_argument(
        "--max-turns", type=int, default=MAX_TURNS, help="tool-calling turns"
    )
    g.add_argument(
        "--max-tokens", type=int, default=2048, help="sampled tokens per turn"
    )
    g.add_argument("--max-trajectory-tokens", type=int, default=32768)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument(
        "--base-url",
        default=None,
        help="Tinker API server (default: $TINKER_BASE_URL, else Thinking Machines' "
        "hosted service); a self-hosted SkyRL Tinker server works here",
    )
    s = parser.add_argument_group("sandboxes")
    s.add_argument("--sandbox", default="daytona", help="BenchFlow sandbox backend")
    s.add_argument(
        "--max-sandboxes", type=int, default=16, help="live sandboxes at once"
    )
    s.add_argument(
        "--command-timeout",
        type=int,
        default=BASH_TIMEOUT_SEC,
        help="seconds per run_bash",
    )
    s.add_argument(
        "--episode-timeout",
        type=float,
        default=900,
        help="seconds per episode once its sandbox is up",
    )
    h = parser.add_argument_group(
        "harness (defaults: the shared RL harness; change them for other task sets)"
    )
    h.add_argument(
        "--harness-message",
        default=HARNESS_MESSAGE,
        help="text appended to every task prompt (default: the shared harness "
        "message, which assumes the family's /workdir and answer file)",
    )
    h.add_argument(
        "--max-output-chars",
        type=int,
        default=MAX_OUTPUT_CHARS,
        help="run_bash output shown to the model",
    )
    h.add_argument(
        "--submit-path", default=SUBMIT_PATH, help="where submit(answer) writes"
    )
    h.add_argument(
        "--sandbox-user",
        default="agent",
        help="user the policy's commands run as; 'root' for task sets whose "
        "harnesses run as root (the shared family needs 'agent': a root policy "
        "could read the oracle)",
    )
    h.add_argument(
        "--integrity",
        choices=["off", "audit", "strict"],
        default="off",
        help="BenchShield reward-integrity audit (needs a BenchFlow with "
        "benchflow.integrity); an exploit scores 0 and is flagged",
    )


def env_config(args: argparse.Namespace, jobs_dir: Path) -> te.EnvConfig:
    renderer = args.renderer or model_info.get_recommended_renderer_name(args.model)
    return te.EnvConfig(
        model_name=args.model,
        renderer_name=renderer,
        episode=EpisodeSettings(
            sandbox=args.sandbox,
            jobs_dir=str(jobs_dir),
            command_timeout_sec=args.command_timeout,
            max_output_chars=args.max_output_chars,
            submit_path=args.submit_path,
            sandbox_user=None if args.sandbox_user == "root" else args.sandbox_user,
            episode_timeout_sec=args.episode_timeout,
            integrity=args.integrity,
        ),
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        max_trajectory_tokens=args.max_trajectory_tokens,
        prompt_suffix=args.harness_message,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tasks-dir", required=True, type=Path, help="the train split")
    parser.add_argument(
        "--include", action="append", default=[], help="only these tasks"
    )
    parser.add_argument(
        "--exclude", action="append", default=[], help="skip these tasks"
    )
    parser.add_argument("--log-path", required=True, type=Path)
    add_env_args(parser)
    t = parser.add_argument_group("training")
    t.add_argument("--steps", type=int, default=20)
    t.add_argument(
        "--group-size", type=int, default=4, help="episodes per task per step"
    )
    t.add_argument("--groups-per-batch", type=int, default=4, help="tasks per step")
    t.add_argument("--learning-rate", type=float, default=4e-5)
    t.add_argument("--lora-rank", type=int, default=32)
    t.add_argument("--save-every", type=int, default=5)
    t.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="replacements per group for infrastructure drops",
    )
    t.add_argument("--seed", type=int, default=0)
    t.add_argument(
        "--load-checkpoint",
        default=None,
        help="start from these weights (tinker://...)",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> rl_train.Config:
    tasks = te.load_tasks(args.tasks_dir, include=args.include, exclude=args.exclude)
    config = env_config(args, args.log_path / "trials")
    builder = te.BenchFlowDatasetBuilder(
        train_tasks=tasks,
        config=config,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        n_batches=args.steps,
        seed=args.seed,
    )
    return rl_train.Config(
        learning_rate=args.learning_rate,
        dataset_builder=builder,
        model_name=args.model,
        recipe_name="benchflow_tinker_cookbook",
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        log_path=str(args.log_path),
        renderer_name=config.renderer_name,
        lora_rank=args.lora_rank,
        eval_every=0,
        save_every=args.save_every,
        load_checkpoint_path=args.load_checkpoint,
        base_url=args.base_url,
        rollout_error_tolerance=te.DropInfrastructureFailures(
            max_retries=args.max_retries, drop_constant_groups=True
        ),
        num_groups_to_log=2,
        max_steps=args.steps,
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def curve_from_records(log_path: Path) -> list[dict]:
    """Per step: every scored episode and every group, dropped groups included.

    The cookbook's own `env/all/reward/total` covers only the groups it
    trained on; with constant groups dropped that mean is biased, so the
    curve comes from the rollout records and the group log instead.
    """
    episodes = read_jsonl(log_path / "trials" / "rollouts.jsonl")
    groups = read_jsonl(log_path / "groups.jsonl")
    steps: dict[str, dict] = {}
    for row in episodes:
        if row.get("reward") is None or not str(row.get("step", "")).startswith(
            "train-"
        ):
            continue
        step = steps.setdefault(row["step"], {"rewards": [], "groups": {}})
        step["rewards"].append(row["reward"])
    for row in groups:
        step = steps.setdefault(row["where"], {"rewards": [], "groups": {}})
        step["groups"][row["kind"]] = step["groups"].get(row["kind"], 0) + 1
    curve = []
    for name in sorted(steps):
        rewards = steps[name]["rewards"]
        curve.append(
            {
                "step": int(name.split("-")[-1]),
                "episodes": len(rewards),
                "mean_reward": sum(rewards) / len(rewards) if rewards else None,
                "solve_rate": sum(r >= 1.0 for r in rewards) / len(rewards)
                if rewards
                else None,
                "groups": steps[name]["groups"],
            }
        )
    return curve


def summarize(args: argparse.Namespace) -> dict:
    """The reward curve, drops, group counts, a cost bound and the checkpoints."""
    metrics = read_jsonl(args.log_path / "metrics.jsonl")
    prefill = sum(m.get("env/all/total_ob_tokens", 0) for m in metrics)
    sampled = sum(m.get("env/all/total_ac_tokens", 0) for m in metrics)
    price = PRICES.get(args.model)
    # Training tokens are at most prefill + sampled (every sampled prompt is a
    # prefix of a trained sequence), so this bound overstates the cost. Token
    # counts cover the trained groups only (the cookbook's metrics); dropped
    # groups were sampled too, so scale by episodes sampled over trained.
    curve = curve_from_records(args.log_path)
    trained = sum(m.get("env/all/total_episodes", 0) for m in metrics)
    sampled_episodes = sum(point["episodes"] for point in curve)
    scale = sampled_episodes / trained if trained else 1.0
    cost = (
        None
        if price is None or args.base_url
        else scale
        * (prefill * price[0] + sampled * price[1] + (prefill + sampled) * price[2])
        / 1e6
    )
    records = read_jsonl(args.log_path / "trials" / "rollouts.jsonl")
    sandbox_hours = (
        sum(r.get("timings", {}).get("sandbox_sec", 0) for r in records) / 3600
    )
    return {
        "model": args.model,
        "steps": len(curve),
        "sandbox_hours": round(sandbox_hours, 2),
        "curve": curve,
        "groups": dict(te.GROUPS.counts),
        "infrastructure_drops": dict(te.DROPS.counts),
        "sandbox_peak": te.sandbox_slots().peak,
        "tokens_trained_groups": {"prefill": prefill, "sampled": sampled},
        "cost_upper_bound_usd": cost,
        "checkpoints": read_jsonl(args.log_path / "checkpoints.jsonl"),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }


async def train(args: argparse.Namespace) -> None:
    config = build_config(args)
    builder = config.dataset_builder
    parity = te.chat_template_parity(builder.train_tasks[0], builder.config)
    log.info("first prompt vs the model's chat template: %s", parity)
    (args.log_path / "chat_template_parity.txt").write_text(parity + "\n")
    try:
        await rl_train.main(config)
    finally:
        left = await close_all_live()
        if left:
            log.warning("closed %d sandboxes left open by an interrupted run", left)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args.log_path.mkdir(parents=True, exist_ok=True)
    te.configure(
        max_sandboxes=args.max_sandboxes,
        drops_path=args.log_path / "infrastructure_drops.jsonl",
        groups_path=args.log_path / "groups.jsonl",
    )
    try:
        asyncio.run(train(args))
    finally:
        summary = summarize(args)
        (args.log_path / "summary.json").write_text(json.dumps(summary, indent=1))
        log.info(
            "steps %d, groups %s, drops %s, sandbox peak %d, cost upper bound %s USD",
            summary["steps"],
            summary["groups"],
            summary["infrastructure_drops"],
            summary["sandbox_peak"],
            summary["cost_upper_bound_usd"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
