"""Seeded rollouts: pass@k, variance and reset reproducibility over a job.

A job run with ``bench eval run --seeds 0-4`` has one trial per (task, seed). `seed_report` groups trials by base
task (from the episode record, else the ``<task>--seed-<n>`` folder name) and reports, per task: rewards by seed,
mean, sample standard deviation, the unbiased pass@k for k = 1..n (Chen et al. 2021), and whether trials with the
same seed started from the same initial state; then the means over tasks.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from .sidecar import split_seed_name


def parse_seeds(text: str) -> list[int]:
    """``"0-4"`` -> [0, 1, 2, 3, 4]; ``"0,3,7"`` -> [0, 3, 7]; ``"5"`` -> [5]; ranges and lists combine.
    Seeds are non-negative integers."""
    seeds: list[int] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            a, b = int(lo), int(hi)
            if b < a:
                raise ValueError(f"empty seed range {part!r}")
            seeds += list(range(a, b + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("no seeds given")
    if any(s < 0 for s in seeds):
        raise ValueError("seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"repeated seeds in {text!r}")
    return seeds


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of P(at least one of k samples passes) from n samples with c passes."""
    if k > n:
        raise ValueError("k must be <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _latest_results(job_dir: Path) -> list[tuple[Path, dict]]:
    """One result per task name, as resume and summary select them: scored beats unscored, then newest.
    Retry attempts and results nested inside a trial (e.g. the episode record) are not extra trials."""
    from benchflow._utils.result_paths import iter_task_result_paths

    best: dict[str, tuple[tuple, Path, dict]] = {}
    for path in iter_task_result_paths(Path(job_dir)):
        try:
            r = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(r, dict):
            continue
        task = r.get("task_name") or path.parent.name.split("__")[0]
        rank = (r.get("rewards") is not None, path.stat().st_mtime, str(path))
        if task not in best or rank >= best[task][0]:
            best[task] = (rank, path, r)
    return [(p, r) for _, p, r in sorted(best.values(), key=lambda x: str(x[1]))]


def _trial_rows(job_dir: Path) -> list[dict]:
    rows = []
    for rj, r in _latest_results(job_dir):
        name = r.get("task_name") or rj.parent.name.split("__")[0]
        base, seed = split_seed_name(name)
        ep = rj.parent / "verifier" / "episode" / "result.json"
        epr: dict[str, Any] = {}
        if ep.exists():
            try:
                epr = json.loads(ep.read_text())
            except json.JSONDecodeError:
                epr = {}
        if seed is None and "seed" in epr:
            seed = int(epr["seed"])
        rewards = r.get("rewards") or {}
        reward = rewards.get("reward") if isinstance(rewards, dict) else None
        rows.append(
            {
                "trial": rj.parent.name,
                "task": base,
                "seed": seed,
                "reward": None if reward is None else float(reward),
                "error": r.get("error") or r.get("verifier_error"),
                "outcome": epr.get("outcome"),
                "steps_used": epr.get("steps_used"),
                "return": epr.get("return"),
                "initial_state_sha256": epr.get("initial_state_sha256"),
            }
        )
    return rows


def seed_report(job_dir: str | Path, rows: list[dict] | None = None) -> dict:
    rows = _trial_rows(Path(job_dir)) if rows is None else rows
    tasks: dict[str, dict] = {}
    for row in rows:
        t = tasks.setdefault(row["task"], {"trials": []})
        t["trials"].append(row)
    out_tasks: dict[str, dict] = {}
    for name, t in sorted(tasks.items()):
        scored = [r for r in t["trials"] if r["reward"] is not None and not r["error"]]
        rewards = [r["reward"] for r in scored]
        n = len(rewards)
        c = sum(1 for x in rewards if x >= 1.0)
        by_seed: dict[str, list] = {}
        hashes: dict[str, set] = {}
        for r in t["trials"]:
            key = str(r["seed"])
            by_seed.setdefault(key, []).append(r["reward"])
            if r["initial_state_sha256"]:
                hashes.setdefault(key, set()).add(r["initial_state_sha256"])
        entry: dict[str, Any] = {
            "n": n,
            "errored": len(t["trials"]) - n,
            "passes": c,
            "rewards_by_seed": by_seed,
            "mean": statistics.fmean(rewards) if rewards else None,
            "std": statistics.stdev(rewards) if n > 1 else (0.0 if n == 1 else None),
            "pass_at_k": {
                str(k): round(pass_at_k(n, c, k), 6) for k in range(1, n + 1)
            },
            "mean_return": statistics.fmean(
                [r["return"] for r in scored if r["return"] is not None]
            )
            if any(r["return"] is not None for r in scored)
            else None,
        }
        repeated = {
            s: len(h) == 1 for s, h in hashes.items() if len(by_seed.get(s, [])) > 1
        }
        if repeated:
            entry["reset_reproducible"] = all(repeated.values())
        entry["distinct_initial_states"] = (
            len({h for hs in hashes.values() for h in hs}) or None
        )
        out_tasks[name] = entry
    ks = sorted({int(k) for e in out_tasks.values() for k in e["pass_at_k"]})
    summary = {
        "tasks": len(out_tasks),
        "trials": len(rows),
        "mean_reward": statistics.fmean(
            [e["mean"] for e in out_tasks.values() if e["mean"] is not None]
        )
        if any(e["mean"] is not None for e in out_tasks.values())
        else None,
        "pass_at_k": {
            str(k): round(
                statistics.fmean(
                    [
                        e["pass_at_k"][str(k)]
                        for e in out_tasks.values()
                        if str(k) in e["pass_at_k"]
                    ]
                ),
                6,
            )
            for k in ks
        },
    }
    return {"summary": summary, "tasks": out_tasks}


def format_report(report: dict) -> str:
    lines = []
    ks = list(report["summary"]["pass_at_k"])
    head = f"{'task':<44} {'n':>3} {'pass':>4} {'mean':>6} {'std':>6} " + " ".join(
        f"{'p@' + k:>6}" for k in ks
    )
    lines.append(head)
    for name, e in report["tasks"].items():
        mean = "-" if e["mean"] is None else f"{e['mean']:.3f}"
        std = "-" if e["std"] is None else f"{e['std']:.3f}"
        pk = " ".join(
            f"{e['pass_at_k'][k]:>6.3f}" if k in e["pass_at_k"] else f"{'-':>6}"
            for k in ks
        )
        lines.append(
            f"{name[:44]:<44} {e['n']:>3} {e['passes']:>4} {mean:>6} {std:>6} {pk}"
        )
    s = report["summary"]
    mr = "-" if s["mean_reward"] is None else f"{s['mean_reward']:.3f}"
    lines.append(
        f"{'all tasks':<44} {s['trials']:>3} {'':>4} {mr:>6} {'':>6} "
        + " ".join(f"{v:>6.3f}" for v in s["pass_at_k"].values())
    )
    return "\n".join(lines)
