"""What a Tinker run costs, from its own token counts at list price. Stdlib only.

Tinker bills per million tokens: prefill (the prompt tokens of each sampling
call; the part served from its prefix cache bills at the cached-prefill price,
a fifth of prefill), sampled tokens, and training tokens (every token of every
datum a training step runs forward and backward). `tinker billing usage`
lags hours, so a run is budgeted from its token counts:

- training (a tinker_train.py log path): the TokenTally of each episode in
  trials/rollouts.jsonl. Every sampled episode pays prefill and sampling; the
  episodes of trained groups (groups.jsonl, not dropped) also pay training.
- a tinker_eval.py result: the same tallies, without training.
- an evaluate.py folder (the shared evaluator through Tinker's OpenAI-compatible
  endpoint): prompt and completion tokens per episode. The endpoint does not
  report the cached share, so the estimate uses the share a training-path run
  measured (`reusable` over `prefill`), and the no-cache bound is reported too.

    python tinker_cost.py LOG_PATH_OR_RESULT [...] --model Qwen/Qwen3.8-27B
    python tinker_cost.py gate --model M --train LOG --steps 20 \\
        --spent CALIBRATION --spent BASE_EVAL --reserve BASE_EVAL --budget 200 --cap 250

`gate` projects a training run from its first step(s): spent + the mean step
cost times the steps + the reserve (evaluations still to run, estimated as the
given ones). It exits 0 to go on and 4 to stop, when the estimate passes
--budget or the no-cache bound passes --cap.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# USD per million tokens (prefill, sample, train), from Tinker's price list
# (https://tinker-docs.thinkingmachines.ai/tinker/models/).
PRICES: dict[str, tuple[float, float, float]] = {
    "Qwen/Qwen3.8-27B": (1.86, 5.595, 4.103),
    "Qwen/Qwen3.6-35B-A3B": (0.54, 1.335, 1.177),
    "Qwen/Qwen3.5-9B-Base": (0.66, 1.995, 1.463),
    "openai/gpt-oss-20b": (0.18, 0.45, 0.396),
    "openai/gpt-oss-120b": (0.33, 0.84, 0.737),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": (0.39, 0.99, 0.88),
}
# Cached prefill bills at a fifth of prefill (Qwen3.8-27B: 0.372 of 1.86).
CACHED_SHARE = 0.2

TOKEN_KEYS = ("prefill", "reusable", "sampled", "train")
STOP = 4  # gate's exit code for "stop"


def usd(model: str, tokens: dict[str, float]) -> dict[str, float]:
    """The estimate (reusable prefill at the cached price) and the no-cache bound."""
    prefill, sample, train = PRICES[model]
    fresh = tokens.get("prefill", 0) - tokens.get("reusable", 0)
    common = tokens.get("sampled", 0) * sample + tokens.get("train", 0) * train
    return {
        "usd": (
            fresh * prefill
            + tokens.get("reusable", 0) * prefill * CACHED_SHARE
            + common
        )
        / 1e6,
        "usd_no_cache": (tokens.get("prefill", 0) * prefill + common) / 1e6,
    }


def _rows(path: Path) -> list[dict[str, Any]]:
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


def _add(total: dict[str, float], tokens: dict[str, Any], keys: Iterable[str]) -> None:
    for key in keys:
        total[key] = total.get(key, 0) + int(tokens.get(key) or 0)


def training_steps(
    log_path: Path | str, *, set_aside: bool = False
) -> list[dict[str, Any]]:
    """Tokens per training step: every sampled episode, training for trained groups.

    With `set_aside`, the records a resumed run set aside (*.discarded.jsonl:
    steps it ran again). They were paid for, but are not part of the run.
    """
    log_path = Path(log_path)
    suffix = ".discarded.jsonl" if set_aside else ".jsonl"
    trained = {
        (g["where"], g["task"])
        for g in _rows(log_path / f"groups{suffix}")
        if not g.get("dropped")
    }
    steps: dict[str, dict[str, Any]] = {}
    for row in _rows(log_path / "trials" / f"rollouts{suffix}"):
        where = str(row.get("step", ""))
        if not where.startswith("train-"):
            continue
        step = steps.setdefault(
            where, {"step": int(where.rsplit("-", 1)[-1]), "episodes": 0, "trained": 0}
        )
        tokens = row.get("tokens") or {}
        step["episodes"] += 1
        _add(step, tokens, ("prefill", "reusable", "sampled"))
        # Dropped episodes (reward None) leave their group: never trained.
        if (where, row.get("task_id")) in trained and row.get("reward") is not None:
            step["trained"] += 1
            _add(step, tokens, ("train",))
    return [steps[k] for k in sorted(steps)]


def reusable_share(log_path_or_result: Path | str) -> float | None:
    """The share of prompt tokens a prefix cache can serve, as a run measured it."""
    totals = cost_of(log_path_or_result, model=None)["tokens"]
    return totals["reusable"] / totals["prefill"] if totals.get("prefill") else None


def cost_of(
    path: Path | str, *, model: str | None, reusable_fraction: float | None = None
) -> dict[str, Any]:
    """Tokens and USD of a training log path, a tinker_eval.py JSON or an evaluate.py folder."""
    path = Path(path)
    tokens: dict[str, float] = dict.fromkeys(TOKEN_KEYS, 0)
    aside: dict[str, float] = dict.fromkeys(TOKEN_KEYS, 0)
    kind = ""
    if (path / "trials" / "rollouts.jsonl").is_file() or (
        path / "groups.jsonl"
    ).is_file():
        kind = "training"
        for step in training_steps(path):
            _add(tokens, step, TOKEN_KEYS)
        # Steps a resumed run ran again were paid for twice.
        for step in training_steps(path, set_aside=True):
            _add(tokens, step, TOKEN_KEYS)
            _add(aside, step, TOKEN_KEYS)
    elif path.is_file():
        kind = "tinker_eval"
        doc = json.loads(path.read_text())
        for evaluation in doc.get("evaluations", []):
            for episodes in evaluation.get("episodes", {}).values():
                for episode in episodes:
                    _add(
                        tokens,
                        episode.get("tokens") or {},
                        ("prefill", "reusable", "sampled"),
                    )
    elif (path / "episodes.jsonl").is_file():
        kind = "evaluate"
        for row in _rows(path / "episodes.jsonl"):
            tokens["prefill"] += int(row.get("prompt_tokens") or 0)
            tokens["sampled"] += int(row.get("completion_tokens") or 0)
        tokens["reusable"] = round(tokens["prefill"] * (reusable_fraction or 0.0))
    else:
        raise FileNotFoundError(
            f"no training log, tinker_eval.py result or evaluate.py folder: {path}"
        )
    out: dict[str, Any] = {"path": str(path), "kind": kind, "tokens": tokens}
    if kind == "training":
        out["set_aside_tokens"] = aside
    if model is not None:
        out.update(usd(model, tokens))
    return out


def gate(args: argparse.Namespace) -> dict[str, Any]:
    steps = training_steps(args.train)
    if not steps:
        raise SystemExit(f"no training step recorded under {args.train}")
    share = args.reusable_fraction
    if share is None:
        share = reusable_share(args.train)
    per_step = [
        {
            "step": s["step"],
            **usd(args.model, s),
            "episodes": s["episodes"],
            "trained": s["trained"],
        }
        for s in steps
    ]
    mean = {
        k: sum(s[k] for s in per_step) / len(per_step) for k in ("usd", "usd_no_cache")
    }
    spent = [cost_of(p, model=args.model, reusable_fraction=share) for p in args.spent]
    # Steps a resumed run ran again: paid for, outside the mean step.
    aside = usd(args.model, cost_of(args.train, model=None)["set_aside_tokens"])
    reserve = [
        cost_of(p, model=args.model, reusable_fraction=share) for p in args.reserve
    ]
    projected = {
        k: sum(c[k] for c in spent)
        + aside[k]
        + mean[k] * args.steps
        + sum(c[k] for c in reserve)
        for k in ("usd", "usd_no_cache")
    }
    reasons = []
    if projected["usd"] > args.budget:
        reasons.append(
            f"the estimate {projected['usd']:.2f} USD passes the budget {args.budget:g}"
        )
    if args.cap is not None and projected["usd_no_cache"] > args.cap:
        reasons.append(
            f"the no-cache bound {projected['usd_no_cache']:.2f} USD passes the cap {args.cap:g}"
        )
    return {
        "model": args.model,
        "steps_planned": args.steps,
        "steps_measured": per_step,
        "mean_step": mean,
        "reusable_share": share,
        "spent": spent,
        "set_aside": aside,
        "reserve": reserve,
        "projected": projected,
        "budget": args.budget,
        "cap": args.cap,
        "go": not reasons,
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "gate":
        parser = argparse.ArgumentParser(prog="tinker_cost.py gate")
        parser.add_argument("--model", required=True)
        parser.add_argument(
            "--train", required=True, type=Path, help="the training log path"
        )
        parser.add_argument(
            "--steps", required=True, type=int, help="steps the run will take"
        )
        parser.add_argument("--spent", action="append", default=[], type=Path)
        parser.add_argument("--reserve", action="append", default=[], type=Path)
        parser.add_argument("--reusable-fraction", type=float, default=None)
        parser.add_argument("--budget", required=True, type=float)
        parser.add_argument("--cap", type=float, default=None)
        parser.add_argument("--out", type=Path, default=None)
        args = parser.parse_args(argv[1:])
        doc = gate(args)
        text = json.dumps(doc, indent=1)
        if args.out is not None:
            args.out.write_text(text + "\n")
        print(text)
        return 0 if doc["go"] else STOP
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reusable-fraction", type=float, default=None)
    args = parser.parse_args(argv)
    for path in args.paths:
        print(
            json.dumps(
                cost_of(
                    path, model=args.model, reusable_fraction=args.reusable_fraction
                )
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
