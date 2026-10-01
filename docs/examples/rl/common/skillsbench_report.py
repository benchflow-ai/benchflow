#!/usr/bin/env python3
"""Report for the SkillsBench held-out run: per arm, paired by task, by category.

  skillsbench_report.py <runs/main> [--excluded excluded.txt] [--json out.json]

Solved = reward >= 1.0 (the verifier's reward). Dropped episodes (sandbox start,
model endpoint, cutoff at the stop time, harness error) are not scored. Excluded
tasks (failed the oracle or do-nothing control) are left out everywhere.

Sets of units (a unit is one task and sample):
- common set: the units every arm finished (scored). Headline per-arm rates and
  the paired differences use it, so every number is on the same units.
- complete prefix: the longest stretch of the seeded queue (sample-major, task
  order from config.json) in which every arm finished every unit. Units still
  running at the stop are the long ones, so the common set can be selected by
  outcome; the prefix cannot. Shown as a sensitivity check.
Paired differences: per task, the mean over that task's units in the set, then
the mean over tasks; 95% interval by bootstrap over tasks (10,000 resamples,
seed 0; units are taken in sorted order, so the intervals are the same on every
run). Per-arm rates: Wilson interval over units, and a bootstrap over tasks
(the two samples of a task are correlated, so Wilson is too narrow then).
If a unit has more than one scored row, the earliest (finished_at) is kept.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ["base-off", "tinker-off", "prime-off", "base-on"]
PAIRS = [("tinker-off", "base-off"), ("prime-off", "base-off"), ("base-on", "base-off"),
         ("base-on", "tinker-off"), ("base-on", "prime-off"), ("tinker-off", "prime-off")]
DROPS = {"sandbox_start", "model_endpoint", "cutoff", "harness_error", "verifier_crash_clean_run"}


def wilson(k: float, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot(values: list[float], n=10000, seed=0):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    m = len(values)
    stats = sorted(sum(values[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return (stats[int(0.025 * n)], stats[int(0.975 * n) - 1])


def fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-"


def kept(r):
    return r.get("reward") is not None and not r.get("dropped") and r.get("ended") not in DROPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--excluded", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    excl_path = a.excluded or (a.root.parent.parent / "excluded.txt")
    excluded = {}
    if excl_path.is_file():
        for line in excl_path.read_text().splitlines():
            if line.strip():
                t, _, why = line.replace("    ", "\t").partition("\t")
                excluded[t.strip()] = why.strip()
    cfg_path = a.root / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    cfg_arms = [x["name"] for x in cfg.get("arms", [])]
    arms = [x for x in ARMS if x in cfg_arms] + [x for x in cfg_arms if x not in ARMS] if cfg_arms else \
        [x for x in ARMS if (a.root / x / "episodes.jsonl").is_file()]
    settings = cfg.get("settings", {})
    order = [t for t in cfg.get("order", []) if t not in excluded]
    samples = int(cfg.get("samples", 1) or 1)
    cat = {t: s.get("category") for t, s in settings.items()}
    eps, dups = {}, Counter()
    for arm in arms:
        rows = []
        path = a.root / arm / "episodes.jsonl"
        for line in (path.read_text().splitlines() if path.is_file() else []):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("task_id") in excluded:
                continue
            rows.append(r)
        eps[arm] = rows
    units = {}
    for arm in arms:
        u = {}
        for r in sorted((r for r in eps[arm] if kept(r)), key=lambda r: r.get("finished_at") or ""):
            key = (r["task_id"], int(r["sample"]))
            if key in u:
                dups[arm] += 1
                continue
            u[key] = r
        units[arm] = u
    common = set.intersection(*(set(u) for u in units.values())) if units else set()
    prefix = []
    for s in range(samples):
        for t in order:
            if all((t, s) in units[arm] for arm in arms):
                prefix.append((t, s))
            else:
                break
        else:
            continue
        break
    out = {"arms": {}, "pairs": {}, "pairs_prefix": {}, "categories": {}, "excluded": excluded,
           "common_units": len(common), "prefix_units": len(prefix), "duplicates": dict(dups)}
    L = ["# SkillsBench held-out: results\n"]
    L.append(f"Excluded tasks ({len(excluded)}): " + "; ".join(f"{t} ({w})" for t, w in sorted(excluded.items())) + "\n")
    L.append(f"Common set: {len(common)} units (task, sample) finished by all {len(arms)} arms, over "
             f"{len({t for t, _ in common})} tasks. Complete prefix of the queue: {len(prefix)} units.\n")
    if dups:
        L.append(f"WARNING: duplicate scored rows (earliest kept): {dict(dups)}\n")

    def rate_table(title, unitset):
        L.append(f"## Per arm, {title}\n")
        L.append("| Arm | Solved / units | Rate | 95% Wilson (units) | 95% bootstrap (tasks) | Mean reward |")
        L.append("|---|---|---|---|---|---|")
        res = {}
        for arm in arms:
            rows = [units[arm][u] for u in sorted(unitset)]
            k = sum(1 for r in rows if r["reward"] >= 1.0)
            per_task = defaultdict(list)
            for (t, _s) in sorted(unitset):
                per_task[t].append(units[arm][(t, _s)]["reward"] >= 1.0)
            tvals = [sum(v) / len(v) for v in per_task.values()]
            mr = sum(r["reward"] for r in rows) / len(rows) if rows else float("nan")
            res[arm] = {"n": len(rows), "solved": k, "rate": k / len(rows) if rows else None,
                        "wilson": wilson(k, len(rows)), "boot_tasks": boot(tvals), "mean_reward": mr}
            L.append(f"| {arm} | {k} / {len(rows)} | {k / len(rows) if rows else float('nan'):.3f} | "
                     f"{fmt_ci(wilson(k, len(rows)))} | {fmt_ci(boot(tvals))} | {mr:.3f} |")
        L.append("")
        return res

    out["arms_common"] = rate_table("common set (headline)", common)
    no_to = {u for u in common if all(units[arm][u].get("ended") != "episode_timeout" for arm in arms)}
    out["common_without_timeouts"] = len(no_to)
    out["arms_no_timeouts"] = rate_table(
        f"common set without the {len(common) - len(no_to)} units where any arm hit the wall-clock guard (sensitivity)", no_to)
    if prefix:
        out["arms_prefix"] = rate_table("complete prefix of the queue (sensitivity)", prefix)
    L.append("## Per arm, all scored units (each arm on its own units)\n")
    L.append("| Arm | Solved / units | Rate [95% Wilson] | Unfinished units (last ending) | Endings of every row |")
    L.append("|---|---|---|---|---|")
    for arm in arms:
        allk = list(units[arm].values())
        ka = sum(1 for r in allk if r["reward"] >= 1.0)
        last = {}
        for r in eps[arm]:
            key = (r.get("task_id"), int(r.get("sample", 0)))
            if key not in units[arm]:
                last[key] = r.get("ended") or r.get("reason") or "?"
        lost = Counter(last.values())
        out["arms"][arm] = {"all_n": len(allk), "all_solved": ka, "all_ci": wilson(ka, len(allk)),
                            "unfinished": dict(lost), "endings": dict(Counter(r.get("ended") for r in eps[arm]))}
        L.append(f"| {arm} | {ka} / {len(allk)} | {ka / len(allk) if allk else float('nan'):.3f} "
                 f"{fmt_ci(wilson(ka, len(allk)))} | {sum(lost.values())} {dict(lost)} | "
                 f"{dict(Counter(r.get('ended') for r in eps[arm]))} |")

    def paired(unitset, pairs=PAIRS):
        res = {}
        for x, y in pairs:
            if x not in units or y not in units:
                continue
            per_task = defaultdict(lambda: [[], []])
            for (t, s) in sorted(unitset):
                rx, ry = units[x][(t, s)], units[y][(t, s)]
                per_task[t][0].append((rx["reward"] >= 1.0) - (ry["reward"] >= 1.0))
                per_task[t][1].append(rx["reward"] - ry["reward"])
            d = [sum(v[0]) / len(v[0]) for v in per_task.values()]
            dr = [sum(v[1]) / len(v[1]) for v in per_task.values()]
            better = sum(1 for v in d if v > 0)
            worse = sum(1 for v in d if v < 0)
            same = len(d) - better - worse
            res[f"{x} - {y}"] = {"tasks": len(d), "diff": sum(d) / len(d) if d else None, "ci": boot(d),
                                 "reward_diff": sum(dr) / len(dr) if dr else None, "reward_ci": boot(dr),
                                 "better": better, "worse": worse, "same": same}
        return res

    def paired_table(title, res):
        L.append(f"\n## Paired by task, {title}\n")
        L.append("| A minus B | Tasks | Solve-rate difference [95% bootstrap] | Mean-reward difference [95%] | Tasks A better / worse / same |")
        L.append("|---|---|---|---|---|")
        for name, v in res.items():
            if v["diff"] is None:
                continue
            L.append(f"| {name} | {v['tasks']} | {v['diff']:+.3f} {fmt_ci(v['ci'])} | "
                     f"{v['reward_diff']:+.3f} {fmt_ci(v['reward_ci'])} | {v['better']} / {v['worse']} / {v['same']} |")

    out["pairs"] = paired(common)
    paired_table("common set (headline)", out["pairs"])
    out["pairs_no_timeouts"] = paired(no_to)
    paired_table("common set without wall-clock-guard units (sensitivity)", out["pairs_no_timeouts"])
    if prefix:
        out["pairs_prefix"] = paired(prefix)
        paired_table("complete prefix (sensitivity)", out["pairs_prefix"])
    # pairwise: each pair on all units both arms finished (more units for pairs of fast arms)
    pw = {}
    for x, y in PAIRS:
        if x in units and y in units:
            pw.update(paired(set(units[x]) & set(units[y]), pairs=[(x, y)]))
    out["pairs_pairwise"] = pw
    paired_table("each pair on all units both arms finished", pw)

    L.append("\n## By category, common set (solved / units; intervals are wide below ~10 tasks)\n")
    cats = sorted({cat.get(t) or "?" for t, _ in common})
    L.append("| Category | Tasks | " + " | ".join(arms) + " | " + " | ".join(f"{x} - {y}" for x, y in PAIRS[:3]) + " |")
    L.append("|---|---|" + "---|" * (len(arms) + 3))
    for c in cats:
        cu = [u for u in common if (cat.get(u[0]) or "?") == c]
        ntask = len({t for t, _ in cu})
        cells = []
        for arm in arms:
            k = sum(1 for u in cu if units[arm][u]["reward"] >= 1.0)
            cells.append(f"{k}/{len(cu)} = {k / len(cu):.2f}")
            out["categories"].setdefault(c, {})[arm] = {"solved": k, "n": len(cu)}
        pr = paired(set(cu), pairs=PAIRS[:3])
        for x, y in PAIRS[:3]:
            v = pr.get(f"{x} - {y}")
            cells.append(f"{v['diff']:+.2f} {fmt_ci(v['ci'])}" if v and v["diff"] is not None else "-")
            out["categories"].setdefault(c, {})[f"{x} - {y}"] = v
        L.append(f"| {c} | {ntask} | " + " | ".join(cells) + " |")

    L.append("\n## Endings and tokens (finished episodes, excluded tasks left out)\n")
    L.append("| Arm | Episodes | Endings | Mean turns | Prompt tokens / ep | Cached share | Completion tokens / ep | Reasoning chars / ep | Calls cut at cap | Recovered calls | Mean minutes | Sandbox-hours |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        rows = [r for r in eps[arm] if r.get("turns") is not None]
        n = len(rows) or 1
        pt = sum(r.get("prompt_tokens", 0) for r in rows)
        ct = sum(r.get("cached_prompt_tokens", 0) for r in rows)
        sh = sum(r.get("elapsed_sec") or 0 for r in rows) / 3600
        out["arms"][arm].update(prompt_per_ep=pt / n, cached_share=ct / pt if pt else None,
                                completion_per_ep=sum(r.get("completion_tokens", 0) for r in rows) / n,
                                turns=sum(r.get("turns", 0) for r in rows) / n,
                                minutes=sum(r.get("elapsed_sec") or 0 for r in rows) / n / 60, sandbox_hours=sh)
        L.append(f"| {arm} | {len(rows)} | {dict(Counter(r.get('ended') for r in rows))} | "
                 f"{sum(r.get('turns', 0) for r in rows) / n:.1f} | {pt / n:,.0f} | "
                 f"{(ct / pt if pt else 0):.2f} | {sum(r.get('completion_tokens', 0) for r in rows) / n:,.0f} | "
                 f"{sum(r.get('reasoning_chars', 0) for r in rows) / n:,.0f} | "
                 f"{sum(r.get('truncated_calls', 0) for r in rows)} | {sum(r.get('recovered_calls', 0) for r in rows)} | "
                 f"{sum(r.get('elapsed_sec') or 0 for r in rows) / n / 60:.1f} | {sh:.1f} |")
    print("\n".join(L))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
