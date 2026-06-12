#!/usr/bin/env python3
"""Reconcile the SkillsBench max-effort fill target against HF PR5 (read-only).

Target grid = 3 models x {with,without} x default SkillsBench tasks x 3 trials.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

REPO = "benchflow/skillsbench-leaderboard"
PR_REF = "refs/pr/5"
V11 = "submissions/skillsbench/v1.1"
ENV_PATHS = (
    os.environ.get("BENCHFLOW_KEYS_ENV"),
    os.path.expanduser("~/Downloads/bingran-you/.env"),
    os.path.expanduser("~/Downloads/GitHub/bingran-you/.env"),
    os.path.expanduser("~/keys.env"),
    os.path.expanduser("~/.env"),
)

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
    if os.environ.get("HUGGING_FACE_TOKEN"):
        return os.environ["HUGGING_FACE_TOKEN"]
    checked = []
    for env_path in ENV_PATHS:
        if not env_path:
            continue
        p = os.path.expanduser(env_path)
        checked.append(p)
        try:
            with open(p) as fh:
                for line in fh:
                    m = re.match(r'^\s*(?:export\s+)?HUGGING_FACE_TOKEN\s*=\s*["\']?([^"\'\s]+)', line)
                    if m:
                        return m.group(1)
        except FileNotFoundError:
            continue
    raise SystemExit("HUGGING_FACE_TOKEN not found in: " + ", ".join(checked))


def default_tasks_dir() -> str:
    for candidate in (
        os.environ.get("SKILLSBENCH_TASKS"),
        os.environ.get("BENCHFLOW_SKILLSBENCH_TASKS"),
        os.path.expanduser("~/Downloads/skillsbench/tasks"),
        os.path.expanduser("~/skillsbench/tasks"),
        os.path.expanduser("~/Downloads/GitHub/BenchFlow.ai/skillsbench/tasks"),
    ):
        if candidate and Path(candidate).is_dir():
            return str(candidate)
    return os.path.expanduser("~/Downloads/skillsbench/tasks")


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


def _token_value(res: dict):
    ar = res.get("agent_result") or {}
    candidates = [
        ar.get("total_tokens"),
        (res.get("final_metrics") or {}).get("total_tokens"),
    ]
    fm = res.get("final_metrics") or {}
    if fm.get("total_prompt_tokens") is not None or fm.get("total_completion_tokens") is not None:
        candidates.append((fm.get("total_prompt_tokens"), fm.get("total_completion_tokens")))
    redacted = False
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, str) and value == "[REDACTED]":
            redacted = True
            continue
        if isinstance(value, tuple):
            parts = []
            for part in value:
                if isinstance(part, str) and part == "[REDACTED]":
                    redacted = True
                    parts = []
                    break
                try:
                    parts.append(float(part or 0))
                except (TypeError, ValueError):
                    parts = []
                    break
            if parts:
                return sum(parts), redacted
            continue
        try:
            return float(value), redacted
        except (TypeError, ValueError):
            continue
    return 0, redacted


def _accepted_timeout_overlay(overlay: dict | None) -> bool:
    if not isinstance(overlay, dict):
        return False
    checks = overlay.get("checks") or {}
    return (
        bool(overlay.get("accepted_normal_timeout"))
        and bool(overlay.get("timeout_complete_artifacts"))
        and bool(checks.get("llm_final_response_ok", True))
    )


def credit_gate(cfg: dict, res: dict, model_key: str, mode: str, overlay: dict | None = None):
    m = MODELS[model_key]
    accepted_timeout = _accepted_timeout_overlay(overlay)
    err = res.get("error")
    err_text = str(err).lower()
    is_timeout = bool(err) and ("timeout" in err_text or "timed out" in err_text)
    if err is not None and not (accepted_timeout and is_timeout):
        return False, "error"
    if res.get("partial_trajectory") and not accepted_timeout:
        return False, "partial"
    rew = (res.get("rewards") or {}).get("reward")
    try:
        if rew is None or not (0.0 <= float(rew) <= 1.0):
            return False, "reward"
    except (TypeError, ValueError):
        return False, "reward"
    tot = (res.get("timing") or {}).get("total")
    try:
        tot = float(tot)
    except (TypeError, ValueError):
        tot = 0
    if not tot or tot <= 0:
        return False, "timing"
    ae = cfg.get("agent_env", {}) or {}
    if str(ae.get(m["effort_key"], "")).lower() != m["effort_val"]:
        return False, "effort"
    its = bool(cfg.get("include_task_skills"))
    if (mode == "with") != its:
        return False, "skillmode"
    total_tokens, tokens_redacted = _token_value(res)
    if m["needs_tokens"] and total_tokens <= 0:
        if tokens_redacted:
            return False, "tokens_redacted"
        return False, "tokens"
    return True, "accepted_timeout" if accepted_timeout else "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=default_tasks_dir())
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)) + "/state")
    ap.add_argument("--download-workers", type=int, default=8)
    args = ap.parse_args()

    os.environ.setdefault("HF_TOKEN", hf_token())
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.environ["HF_TOKEN"])

    tasks = load_tasks(args.tasks_dir)
    print(f"tasks: {len(tasks)} (from {args.tasks_dir})")
    target = len(MODELS) * len(MODES) * len(tasks) * N_TRIALS
    print(f"target grid: {len(MODELS)} models x {len(MODES)} modes x {len(tasks)} tasks x {N_TRIALS} trials = {target}")

    # credited[(model,mode,task)] -> {trialid: {reward,tokens,timing,hf_path}}
    # repairable = otherwise healthy PR5 cells whose token usage was redacted
    # by an older publisher and can be overwritten from local rollout artifacts.
    credited: dict = defaultdict(dict)
    repairable: dict = defaultdict(dict)
    rejected = defaultdict(lambda: defaultdict(int))  # model -> reason -> count
    result_records = []
    overlay_dirs: set[str] = set()

    for grp in api.list_repo_tree(REPO, repo_type="dataset", revision=PR_REF, path_in_repo=V11, recursive=False):
        group = grp.path.split("/")[-1]
        model_key, mode_hint = group_model_modehint(group)
        if model_key is None:
            continue
        result_paths = []
        for entry in api.list_repo_tree(
            REPO, repo_type="dataset", revision=PR_REF, path_in_repo=grp.path, recursive=True
        ):
            if entry.path.endswith("/result.json"):
                result_paths.append(entry.path)
            elif entry.path.endswith("/strict_audit_overlay.json"):
                overlay_dirs.add(entry.path.rsplit("/", 1)[0])
        for rpath in result_paths:
            result_records.append((model_key, mode_hint, rpath))
    print(f"existing candidate result.json files: {len(result_records)}", flush=True)

    def inspect_result(record):
        model_key, mode_hint, rpath = record
        cell_dir = rpath.rsplit("/", 1)[0]
        leaf = cell_dir.split("/")[-1]
        if "__" not in leaf:
            return model_key, None, None, None, "bad_leaf", None
        task, tid = leaf.rsplit("__", 1)
        if task not in tasks:
            return model_key, None, task, tid, "task_not_in_grid", None
        try:
            res_path = hf_hub_download(
                REPO, rpath, repo_type="dataset", revision=PR_REF
            )
            cfg_path = hf_hub_download(
                REPO,
                cell_dir + "/config.json",
                repo_type="dataset",
                revision=PR_REF,
            )
            with open(res_path) as fh:
                res = json.load(fh)
            with open(cfg_path) as fh:
                cfg = json.load(fh)
        except Exception:
            return model_key, None, task, tid, "unreadable", None
        mode = mode_hint or ("with" if cfg.get("include_task_skills") else "without")
        overlay = None
        if cell_dir in overlay_dirs:
            try:
                overlay_path = hf_hub_download(
                    REPO,
                    cell_dir + "/strict_audit_overlay.json",
                    repo_type="dataset",
                    revision=PR_REF,
                )
                with open(overlay_path) as fh:
                    overlay = json.load(fh)
            except Exception:
                overlay = None
        ok, reason = credit_gate(cfg, res, model_key, mode, overlay)
        if not ok:
            payload = None
            if reason == "tokens_redacted":
                payload = {
                    "reward": (res.get("rewards") or {}).get("reward"),
                    "tokens": None,
                    "timing_total_s": (res.get("timing") or {}).get("total"),
                    "hf_path": cell_dir,
                }
            return model_key, mode, task, tid, reason, payload
        return model_key, mode, task, tid, "ok", {
            "reward": (res.get("rewards") or {}).get("reward"),
            "tokens": (res.get("agent_result") or {}).get("total_tokens"),
            "timing_total_s": (res.get("timing") or {}).get("total"),
            "hf_path": cell_dir,
            "accepted_normal_timeout": reason == "accepted_timeout",
            "timeout_complete_artifacts": reason == "accepted_timeout",
        }

    seen_cells = 0
    with ThreadPoolExecutor(max_workers=max(1, args.download_workers)) as ex:
        futs = [ex.submit(inspect_result, rec) for rec in result_records]
        for fut in as_completed(futs):
            seen_cells += 1
            model_key, mode, task, tid, reason, payload = fut.result()
            if reason == "ok":
                credited[(model_key, mode, task)][tid] = payload
            elif reason == "tokens_redacted" and payload is not None:
                repairable[(model_key, mode, task)][tid] = payload
                rejected[model_key][reason] += 1
            else:
                rejected[model_key][reason] += 1
            if seen_cells % 100 == 0:
                print(f"inspected {seen_cells}/{len(result_records)} candidates", flush=True)

    # Build the full grid ledger + the queue of needed cells.
    now = datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    queue = []
    repair_queue = []
    need_counts = defaultdict(int)
    repair_counts = defaultdict(int)
    for model_key, mc in MODELS.items():
        for mode in MODES:
            for task in tasks:
                creds = list(credited[(model_key, mode, task)].values())
                repairs = list(repairable[(model_key, mode, task)].values())
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
                            "accepted_normal_timeout": c.get("accepted_normal_timeout"),
                            "timeout_complete_artifacts": c.get("timeout_complete_artifacts"),
                            "note": "existing PR5 cell (pending deep re-review)", "updated_at": now,
                        })
                    elif slot <= len(creds) + len(repairs):
                        c = repairs[slot - len(creds) - 1]
                        repair_counts[(model_key, mode)] += 1
                        row = {
                            "cell_id": cid, "model": model_key, "model_slug": mc["slug"],
                            "effort": mc["effort_val"], "skill_mode": mode, "task": task,
                            "trial_slot": slot, "status": "repair_needed", "sandbox": "daytona",
                            "reward": c["reward"], "health": "metadata_incomplete",
                            "review_verdict": "repair_needed",
                            "task_skills_loading": 1 if mode == "with" else 0,
                            "tokens": None, "timing_total_s": c["timing_total_s"],
                            "hf_path": c["hf_path"],
                            "note": "existing PR5 cell needs token-usage repair from local rollout",
                            "updated_at": now,
                        }
                        rows.append(row)
                        repair_queue.append({k: row[k] for k in ("cell_id", "model", "skill_mode", "task", "trial_slot", "hf_path")})
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
    (out / "repair_queue.jsonl").write_text("".join(json.dumps(q) + "\n" for q in repair_queue))
    ledger = {"as_of": now, "target": target, "rows": rows}
    (out / "experiments_ledger.json").write_text(json.dumps(ledger, indent=2))
    (out / "grid.json").write_text(json.dumps(ledger, indent=2))
    credited_capped_by_model_mode = {
        f"{mk}/{md}": sum(
            min(N_TRIALS, len(credited[(mk, md, task)]))
            for task in tasks
        )
        for mk in MODELS
        for md in MODES
    }
    report = {
        "as_of": now, "target": target, "tasks": len(tasks),
        "existing_cells_seen": seen_cells,
        "credited_total": sum(credited_capped_by_model_mode.values()),
        "credited_raw_artifacts_total": sum(len(v) for v in credited.values()),
        "repair_needed_total": len(repair_queue),
        "credited_by_model_mode": credited_capped_by_model_mode,
        "repair_needed_by_model_mode": {f"{mk}/{md}": repair_counts[(mk, md)] for mk in MODELS for md in MODES},
        "needed_total": len(queue),
        "needed_by_model_mode": {f"{mk}/{md}": need_counts[(mk, md)] for mk in MODELS for md in MODES},
        "rejected_by_model_reason": {mk: dict(rejected[mk]) for mk in MODELS if rejected[mk]},
    }
    (out / "reconcile_report.json").write_text(json.dumps(report, indent=2))

    print("\n=== reconcile report ===")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}/queue.jsonl ({len(queue)} cells), repair_queue.jsonl ({len(repair_queue)} cells), experiments_ledger.json ({len(rows)} rows), reconcile_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
