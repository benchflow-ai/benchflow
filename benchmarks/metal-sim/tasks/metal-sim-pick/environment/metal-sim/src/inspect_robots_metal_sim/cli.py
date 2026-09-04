"""``metal-sim-eval``: run a registered simulator task and emit a machine-readable summary.

Used by the BenchFlow verifiers (``bench/tasks/*/verifier/test.sh``) and handy for local checks:

    metal-sim-eval --task metal-sim-reach --policy-file /app/policy.py --log-dir logs --json out.json
    metal-sim-eval --task metal-sim-pick --policy metal_ik
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from inspect_robots.eval import eval as ir_eval
from inspect_robots.registry import resolve


def _kv(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"expected key=value, got {pair!r}")
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def summarize(log: Any) -> dict[str, Any]:
    """Distil an ``EvalLog`` into the reward record the verifiers consume."""
    scenes = []
    for sample in log.samples:
        reduced = dict(sample.reduced or {})
        scenes.append(
            {
                "scene_id": sample.scene_id,
                "status": sample.status,
                "success": float(reduced.get("success_at_end", 0.0)),
                "min_distance": reduced.get("min_distance_to_goal"),
                "steps": reduced.get("episode_length"),
                "error": sample.error,
                "termination": list(sample.termination_reasons or []),
            }
        )
    successes = [s["success"] for s in scenes]
    return {
        "task": log.eval.task,
        "policy": log.eval.policy,
        "embodiment": log.eval.embodiment,
        "status": log.status,
        "n_scenes": len(scenes),
        "success_rate": (sum(successes) / len(successes)) if successes else 0.0,
        "scenes": scenes,
        "duration_s": getattr(log.stats, "duration_s", None),
        "total_steps": getattr(log.stats, "total_steps", None),
        "error": log.error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a policy on a Metal simulator task.")
    parser.add_argument("--task", default="metal-sim-reach", help="registered task name")
    parser.add_argument("-T", dest="task_args", action="append", default=[], metavar="k=v", help="task factory argument")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--policy", default=None, help="registered policy name (default: metal_ik)")
    group.add_argument("--policy-file", default=None, help="Python file with act(observation); uses metal_pyfile")
    parser.add_argument("-P", dest="policy_args", action="append", default=[], metavar="k=v", help="policy argument")
    parser.add_argument("-E", dest="embodiment_args", action="append", default=[], metavar="k=v", help="embodiment argument")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None, help="write the summary JSON here (also printed)")
    parser.add_argument("--no-frames", action="store_true", help="do not store camera frames in the log")
    args = parser.parse_args(argv)

    task = resolve("task", args.task, **_kv(args.task_args))
    if args.policy_file:
        policy = resolve("policy", "metal_pyfile", path=args.policy_file, **_kv(args.policy_args))
    else:
        policy = resolve("policy", args.policy or "metal_ik", **_kv(args.policy_args))
    embodiment = resolve("embodiment", "metal_sim", **_kv(args.embodiment_args))
    try:
        logs = ir_eval(task, policy, embodiment, log_dir=args.log_dir, seed=args.seed, store_frames=not args.no_frames)
    finally:
        embodiment.close()
    summary = summarize(logs[0])
    text = json.dumps(summary, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    return 0 if summary["status"] != "error" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
