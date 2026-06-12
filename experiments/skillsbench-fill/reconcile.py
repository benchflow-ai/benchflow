#!/usr/bin/env python3
"""Reconcile the SkillsBench max-effort fill target against HF PR5 (read-only).

Target grid = 3 models x {with,without} x 91 tasks x 3 trials = 1638.

An existing PR5 cell is CREDITED toward the target only if it is:
  - healthy: error==None, partial_trajectory==False, reward in [0,1], timing.total>0,
    (tokens>0 for models whose provider surfaces usage: opus, minimax)
  - provably max-effort: the per-model effort env is present in config.json.agent_env
    (opus: BENCHFLOW_BEDROCK_THINKING_EFFORT=max; gemini: LLM_REASONING_EFFORT=high;
     minimax: LLM_REASONING_EFFORT=max)
  - correct skill posture: include_task_skills matches the (with/without) mode

Emits, under --out:
  queue.jsonl            one row per (model,mode,task,trial_slot) still NEEDED
  experiments_ledger.json full 1638-cell grid for the dashboard (credited slots ->
                          published, needed -> queued)
  reconcile_report.json  counts: credited, needed, and rejected-by-reason

Does NOT modify PR5. Trial ids in PR5 are random hex, so credit counts DISTINCT
healthy cells per (model,mode,task) toward 3 (not specific trial indices).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = "benchflow/skillsbench-leaderboard"
PR_REF = "refs/pr/5"
V11 = "submissions/skillsbench/v1.1"
ENV_PATH = os.path.expanduser("~/Downloads/GitHub/bingran-you/.env")

MODELS = {
    "opus-4.8": {
        "slug": "aws-bedrock-us.anthropic.claude-opus-4-8",
        "model": "aws-bedrock/us.anthropic.claude-opus-4-8",
        "effort_key": "BENCHFLOW_BEDROCK_THINKING_EFFORT", "effort_val": "max",
        "needs_tokens": True,
    },
    "gemini-3.5-flash": {
        "slug": "gemini-3.5-flash", "model": "gemini-3.5-flash",
        "effort_key": "LLM_REASONING_EFFORT", "effort_val": "high",
        "needs_tokens": False,  # native gemini path doesn't surface usage
    },
    "minimax-m3": {
        "slug": "minimax-MiniMax-M3", "model": "minimax/MiniMax-M3",
        "effort_key": "LLM_REASONING_EFFORT", "effort_val": "max",
        "needs_tokens": True,
    },
}
SLUG2MODEL = {m["slug"]: k for k, m in MODELS.items()}
MODES = ("with", "without")
N_TRIALS = 3


def hf_token() -> str:
    for line in open(ENV_PATH):
        m = re.match(r'^\s*(?:export\s+)?HUGGING_FACE_TOKEN\s*=\s*["\']?([^"\'\s]+)', line)
        if m:
            return m.group(1)
    raise SystemExit("HUGGING_FACE_TOKEN not found in " + ENV_PATH)


def load_tasks(tasks_dir: str) -> list[str]:
    p = Path(tasks_dir)
    return sorted(d.name for d in p.iterdir() if d.is_dir() and (d / "task.toml").exists())


def group_model_modehint(group: str):
    """(model_key, mode_hint) from a v1.1 group dir name; mode_hint None if not encoded."""
    if group.startswith("openhands-with-skills__"):
        return SLUG2MODEL.get(group.split("__", 1)[1]), "with"
    if group.startswith("openhands-no-skills__"):
        return SLUG2MODEL.get(group.split("__", 1)[1]), "without"
    if group.startswith("openhands__"):
        return SLUG2MODEL.get(group.split("__", 1)[1]), None
    return None, None


def credit_gate(cfg: dict, res: dict, model_key: str, mode: str):
    m = MODELS[model_key]
    if res.get("error") is not None:
        return False, "error"
    if res.get("partial_trajectory"):
        return False, "partial"
    rew = (res.get("rewards") or {}).get("reward")
    try:
        if rew is None or not (0.0 <= float(rew) <= 1.0):
            return False, "reward"
    except (TypeError, ValueError):
        return False, "reward"
    tot = (res.get("timing") or {}).get("total")
    if not tot or tot <= 0:
        return False, "timing"
    if m["needs_tokens"] and not ((res.get("agent_result") or {}).get("total_tokens") or 0) > 0:
        return False, "tokens"
    ae = cfg.get("agent_env", {}) or {}
    if str(ae.get(m["effort_key"], "")).lower() != m["effort_val"]:
        return False, "effort"
    its = bool(cfg.get("include_task_skills"))
    if (mode == "with") != its:
        return False, "skillmode"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=os.path.expanduser("~/Downloads/GitHub/BenchFlow.ai/skillsbench/tasks"))
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)) + "/state")
    args = ap.parse_args()

    os.environ.setdefault("HF_TOKEN", hf_token())
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.environ["HF_TOKEN"])

    tasks = load_tasks(args.tasks_dir)
    print(f"tasks: {len(tasks)} (from {args.tasks_dir})")
    target = len(MODELS) * len(MODES) * len(tasks) * N_TRIALS
    print(f"target grid: {len(MODELS)} models x {len(MODES)} modes x {len(tasks)} tasks x {N_TRIALS} trials = {target}")

    # credited[(model,mode,task)] -> {trialid: {reward,tokens,timing,hf_path}}
    credited: dict = defaultdict(dict)
    rejected = defaultdict(lambda: defaultdict(int))  # model -> reason -> count
    seen_cells = 0

    for grp in api.list_repo_tree(REPO, repo_type="dataset", revision=PR_REF, path_in_repo=V11, recursive=False):
        group = grp.path.split("/")[-1]
        model_key, mode_hint = group_model_modehint(group)
        if model_key is None:
            continue
        result_paths = [
            k.path for k in api.list_repo_tree(REPO, repo_type="dataset", revision=PR_REF, path_in_repo=grp.path, recursive=True)
            if k.path.endswith("/result.json")
        ]
        for rpath in result_paths:
            seen_cells += 1
            cell_dir = rpath.rsplit("/", 1)[0]
            leaf = cell_dir.split("/")[-1]
            if "__" not in leaf:
                continue
            task, tid = leaf.rsplit("__", 1)
            if task not in tasks:
                rejected[model_key]["task_not_in_grid"] += 1
                continue
            try:
                res = json.load(open(hf_hub_download(REPO, rpath, repo_type="dataset", revision=PR_REF)))
                cfg = json.load(open(hf_hub_download(REPO, cell_dir + "/config.json", repo_type="dataset", revision=PR_REF)))
            except Exception:
                rejected[model_key]["unreadable"] += 1
                continue
            mode = mode_hint or ("with" if cfg.get("include_task_skills") else "without")
            ok, reason = credit_gate(cfg, res, model_key, mode)
            if not ok:
                rejected[model_key][reason] += 1
                continue
            credited[(model_key, mode, task)][tid] = {
                "reward": (res.get("rewards") or {}).get("reward"),
                "tokens": (res.get("agent_result") or {}).get("total_tokens"),
                "timing_total_s": (res.get("timing") or {}).get("total"),
                "hf_path": cell_dir,
            }

    # Build the full grid ledger + the queue of needed cells.
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    queue = []
    need_counts = defaultdict(int)
    for model_key, mc in MODELS.items():
        for mode in MODES:
            for task in tasks:
                creds = list(credited[(model_key, mode, task)].values())
                for slot in range(1, N_TRIALS + 1):
                    cid = f"{model_key}__{mode}__{task}__t{slot}"
                    if slot <= len(creds):
                        c = creds[slot - 1]
                        rows.append({
                            "cell_id": cid, "model": model_key, "model_slug": mc["slug"],
                            "effort": mc["effort_val"], "skill_mode": mode, "task": task,
                            "trial_slot": slot, "status": "published", "sandbox": "daytona",
                            "reward": c["reward"], "health": "healthy", "review_verdict": "pass",
                            "task_skills_loading": 1 if mode == "with" else 0,
                            "tokens": {"total": c["tokens"]} if c["tokens"] else None,
                            "timing_total_s": c["timing_total_s"], "hf_path": c["hf_path"],
                            "note": "existing PR5 cell (pending deep re-review)", "updated_at": now,
                        })
                    else:
                        need_counts[(model_key, mode)] += 1
                        row = {
                            "cell_id": cid, "model": model_key, "model_slug": mc["slug"],
                            "effort": mc["effort_val"], "skill_mode": mode, "task": task,
                            "trial_slot": slot, "status": "queued", "sandbox": "daytona",
                            "updated_at": now,
                        }
                        rows.append(row)
                        queue.append({k: row[k] for k in ("cell_id", "model", "skill_mode", "task", "trial_slot")})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queue))
    ledger = {"as_of": now, "target": target, "rows": rows}
    (out / "experiments_ledger.json").write_text(json.dumps(ledger, indent=2))
    report = {
        "as_of": now, "target": target, "tasks": len(tasks),
        "existing_cells_seen": seen_cells,
        "credited_total": sum(len(v) for v in credited.values()),
        "credited_by_model_mode": {f"{mk}/{md}": sum(len(credited[(mk, md, t)]) for t in tasks)
                                   for mk in MODELS for md in MODES},
        "needed_total": len(queue),
        "needed_by_model_mode": {f"{mk}/{md}": need_counts[(mk, md)] for mk in MODELS for md in MODES},
        "rejected_by_model_reason": {mk: dict(rejected[mk]) for mk in MODELS if rejected[mk]},
    }
    (out / "reconcile_report.json").write_text(json.dumps(report, indent=2))

    print("\n=== reconcile report ===")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}/queue.jsonl ({len(queue)} cells), experiments_ledger.json ({len(rows)} rows), reconcile_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
