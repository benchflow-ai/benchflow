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

import tinker_cost  # noqa: E402
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
    RunStopped,
    own_daytona_run,
    run_guarded,
    sweep_owner,
)

log = logging.getLogger("tinker_train")


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
        "--stop-after",
        type=int,
        default=None,
        help="end after this many steps with a full checkpoint (the data order is "
        "still that of --steps); rerun with the same --log-path to go on, for "
        "example after costing the first step (tinker_cost.py gate)",
    )
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
    t.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=0,
        help="stop cleanly when the log path's disk has less free space (GiB); "
        "rerun with the same --log-path to resume from the last checkpoint",
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
        max_steps=min(args.steps, args.stop_after or args.steps),
    )


def read_jsonl(path: Path) -> list[dict]:
    """JSON lines; a line cut short by a killed writer is skipped."""
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _step_index(name: object) -> int:
    try:
        return int(str(name).rsplit("-", 1)[-1])
    except ValueError:
        return -1


def prune_for_resume(log_path: Path) -> int:
    """Set aside the records of steps a resumed run will run again.

    The cookbook resumes after its last checkpoint; episodes, groups and drops
    of the steps after it (an interrupted step included) move to
    *.discarded.jsonl, so the curve counts every step once (tinker_cost.py
    still counts them as spent).
    """
    from tinker_cookbook import checkpoint_utils

    last = checkpoint_utils.get_last_checkpoint(str(log_path))
    resume = last.batch if last is not None else 0
    moved = 0
    for relative, key in (
        ("trials/rollouts.jsonl", "step"),
        ("groups.jsonl", "where"),
        ("infrastructure_drops.jsonl", "where"),
    ):
        path = log_path / relative
        rows = read_jsonl(path)
        gone = [r for r in rows if _step_index(r.get(key)) >= resume]
        if not gone:
            continue
        keep = [r for r in rows if _step_index(r.get(key)) < resume]
        with path.with_suffix(".discarded.jsonl").open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in gone)
        path.write_text("".join(json.dumps(r) + "\n" for r in keep))
        moved += len(gone)
    return moved


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
    """The reward curve, drops, group counts, the cost and the checkpoints."""
    curve = curve_from_records(args.log_path)
    # From each episode's token tally at list price (tinker_cost.py): every
    # sampled episode pays prefill and sampling, trained groups pay training.
    steps = tinker_cost.training_steps(args.log_path)
    priced = args.model in tinker_cost.PRICES and not args.base_url
    cost = [
        {
            "step": s["step"],
            "episodes": s["episodes"],
            "trained": s["trained"],
            **{k: s.get(k, 0) for k in tinker_cost.TOKEN_KEYS},
            **(tinker_cost.usd(args.model, s) if priced else {}),
        }
        for s in steps
    ]
    records = read_jsonl(args.log_path / "trials" / "rollouts.jsonl")
    sandbox_hours = (
        sum(r.get("timings", {}).get("sandbox_sec", 0) for r in records) / 3600
    )
    return {
        "model": args.model,
        "steps": len(curve),
        "sandbox_hours": round(sandbox_hours, 2),
        "curve": curve,
        "groups": _count(read_jsonl(args.log_path / "groups.jsonl"), "kind"),
        "infrastructure_drops": _count(
            read_jsonl(args.log_path / "infrastructure_drops.jsonl"), "reason"
        ),
        "sandbox_peak": te.sandbox_slots().peak,
        "cost_by_step": cost,
        "cost_usd": sum(c["usd"] for c in cost) if priced else None,
        "cost_usd_no_cache": sum(c["usd_no_cache"] for c in cost) if priced else None,
        "checkpoints": read_jsonl(args.log_path / "checkpoints.jsonl"),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }


def _count(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    return counts


async def train(args: argparse.Namespace) -> None:
    config = build_config(args)
    builder = config.dataset_builder
    parity = te.chat_template_parity(builder.train_tasks[0], builder.config)
    log.info("first prompt vs the model's chat template: %s", parity)
    (args.log_path / "chat_template_parity.txt").write_text(parity + "\n")
    await run_guarded(
        rl_train.main(config),
        disk_path=args.log_path,
        min_free_gib=args.min_free_disk_gb,
    )


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
    owner = own_daytona_run("tinker-rl-train") if args.sandbox == "daytona" else None
    moved = prune_for_resume(args.log_path)
    if moved:
        log.info("resuming: set aside %d records of steps that run again", moved)
    code = 0
    try:
        asyncio.run(train(args))
    except RunStopped as exc:
        log.error("%s; rerun with the same --log-path to resume", exc)
        code = 3
    finally:
        sweep_owner(owner)
        summary = summarize(args)
        (args.log_path / "summary.json").write_text(json.dumps(summary, indent=1))
        log.info(
            "steps %d, groups %s, drops %s, sandbox peak %d, cost %s USD "
            "(no-cache bound %s) at list price",
            summary["steps"],
            summary["groups"],
            summary["infrastructure_drops"],
            summary["sandbox_peak"],
            summary["cost_usd"],
            summary["cost_usd_no_cache"],
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
