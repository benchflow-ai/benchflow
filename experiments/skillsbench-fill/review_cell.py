#!/usr/bin/env python3
"""Mechanical bulk reviewer for the SkillsBench max-effort fill (runs on the VM).

Automates the benchflow-experiment-review checklist per completed cell and writes
review/<cell>.json with a verdict + evidence. Subagents then deep-audit every
'fail' + a random sample for judgment (plausibility / reward-hacking).

For each state/<cell>.json with status==completed and no review yet:
  health      error is None, partial_trajectory False, reward in [0,1], timing.total>0
  tokens      total_tokens>0 (required for opus/minimax; advisory for gemini)
  trajectory  acp_trajectory.jsonl >100B and llm_trajectory.jsonl present
  effort      opus: thinking/reasoning refs in llm_trajectory (max thinking fired)
              gemini: agent_env LLM_REASONING_EFFORT==high ; minimax: ==max
  skills      extract_harness_skills task_skills_loading == (1 if with else 0)
verdict: pass iff all true; fail if a model-side criterion fails; quarantine if
infra/artifacts missing (run_failed or unreadable).

Usage: review_cell.py [--limit N] [--cell <id>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
STATE, REVIEW = ROOT / "state", ROOT / "review"
SB_TASKS = os.path.expanduser("~/skillsbench/tasks")
EXTRACT = os.path.expanduser(
    "~/Experiment/benchflow/.claude/skills/benchflow-experiment-review/scripts/extract_harness_skills.py"
)
RECONCILE_FILES = {"experiments_ledger.json", "queue.jsonl", "reconcile_report.json", "grid.json"}
THINK_RE = re.compile(rb"thinking|reasoning_content|reasoningContent|redacted_thinking|signature", re.I)


def _rollout(st: dict) -> Path | None:
    rd = st.get("rollout_dir")
    if rd and Path(rd).is_dir():
        return Path(rd)
    root = st.get("run_root")
    if root:
        hits = [d for d in glob.glob(f"{root}/**/{st['task']}__*", recursive=True) if Path(d).is_dir()]
        if hits:
            return Path(sorted(hits)[0])
    return None


def review(st: dict) -> dict:
    cell, model, mode, task = st["cell_id"], st["model"], st["skill_mode"], st["task"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = {"cell_id": cell, "updated_at": now, "checklist": {}, "notes": ""}
    if st.get("status") != "completed":
        return {**out, "verdict": "quarantine", "health": "unhealthy",
                "notes": f"status={st.get('status')} ({st.get('error')})"}
    roll = _rollout(st)
    if roll is None:
        return {**out, "verdict": "quarantine", "health": "unhealthy", "notes": "no rollout dir"}
    try:
        res = json.load(open(roll / "result.json"))
        cfg = json.load(open(roll / "config.json"))
    except Exception as e:
        return {**out, "verdict": "quarantine", "health": "unhealthy", "notes": f"unreadable: {e}"}

    ar = res.get("agent_result") or {}
    ae = cfg.get("agent_env", {}) or {}
    acp, llm = roll / "trajectory/acp_trajectory.jsonl", roll / "trajectory/llm_trajectory.jsonl"
    rew = (res.get("rewards") or {}).get("reward")
    tot = (res.get("timing") or {}).get("total")
    tokens = ar.get("total_tokens")

    err = res.get("error")
    # A genuine agent timeout with a complete trajectory + score is a HEALTHY
    # normal_timeout (publishable per the review skill), NOT an infra failure.
    is_timeout = bool(err) and "timed out" in str(err).lower()
    out["outcome"] = "normal_timeout" if is_timeout else "completed"
    c = out["checklist"]
    c["error_ok"] = (err is None) or is_timeout
    c["not_partial"] = (not res.get("partial_trajectory")) or is_timeout
    try:
        c["reward_valid"] = rew is not None and 0.0 <= float(rew) <= 1.0
    except (TypeError, ValueError):
        c["reward_valid"] = False
    c["timing_ok"] = bool(tot and tot > 0)
    c["acp_present"] = acp.is_file() and acp.stat().st_size > 100
    c["llm_present"] = llm.is_file() and llm.stat().st_size > 50
    c["tokens_ok"] = bool(tokens and tokens > 0) if model != "gemini-3.5-flash" else True

    # effort provenance
    if model == "opus-4.8":
        think = 0
        if llm.is_file():
            with open(llm, "rb") as fh:
                think = len(THINK_RE.findall(fh.read()))
        c["effort_ok"] = think > 0 or str(ae.get("BENCHFLOW_BEDROCK_THINKING_EFFORT", "")).lower() == "max"
        out["thinking_refs"] = think
    elif model == "gemini-3.5-flash":
        c["effort_ok"] = str(ae.get("LLM_REASONING_EFFORT", "")).lower() == "high"
    else:  # minimax-m3
        c["effort_ok"] = str(ae.get("LLM_REASONING_EFFORT", "")).lower() == "max"

    # skill posture via extract_harness_skills
    tsl, tsl_status = None, "unknown"
    if llm.is_file():
        try:
            p = subprocess.run(["python3", EXTRACT, str(llm), "--task-path", f"{SB_TASKS}/{task}"],
                               capture_output=True, text=True, timeout=120)
            j = json.loads(p.stdout)
            tsl, tsl_status = j.get("task_skills_loading"), j.get("task_skills_loading_status")
        except Exception as e:
            tsl_status = f"extract_error:{e}"
    out["task_skills_loading"], out["task_skills_loading_status"] = tsl, tsl_status
    c["skill_posture_ok"] = (tsl == (1 if mode == "with" else 0))

    out["reward"] = rew
    out["tokens"] = {"total": tokens} if tokens else None
    out["timing_total_s"] = tot
    out["usage_source"] = ar.get("usage_source")
    healthy = all(c.values())
    # infra/incomplete (missing trajectory, provider error, no reward) => quarantine+requeue,
    # NOT a model fail. A model fail has a real trajectory+reward but flunks effort/skill/leakage.
    infra = (err is not None and not is_timeout) or not c["acp_present"] or not c["llm_present"] or not c["reward_valid"]
    if healthy:
        out["health"], out["verdict"] = "healthy", "pass"
    elif infra:
        out["health"], out["verdict"] = "unhealthy", "quarantine"
        out["notes"] = "infra/requeue: " + ",".join(k for k, v in c.items() if not v)
    else:
        out["health"], out["verdict"] = "unhealthy", "fail"
        out["notes"] = "model-side: " + ",".join(k for k, v in c.items() if not v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cell", default="")
    a = ap.parse_args()
    REVIEW.mkdir(exist_ok=True)
    done = {Path(f).stem for f in glob.glob(str(REVIEW / "*.json"))}
    reviewed = passed = failed = quar = 0
    for sf in sorted(glob.glob(str(STATE / "*.json"))):
        if os.path.basename(sf) in RECONCILE_FILES:
            continue
        try:
            st = json.load(open(sf))
        except Exception:
            continue
        cell = st.get("cell_id")
        if not cell or (a.cell and cell != a.cell) or cell in done:
            continue
        if st.get("status") != "completed":
            continue
        rv = review(st)
        json.dump(rv, open(REVIEW / f"{cell}.json", "w"), indent=2)
        reviewed += 1
        passed += rv["verdict"] == "pass"
        failed += rv["verdict"] == "fail"
        quar += rv["verdict"] == "quarantine"
        print(f"[{rv['verdict']:10s}] {cell} reward={rv.get('reward')} tsl={rv.get('task_skills_loading')} {rv.get('notes','')}")
        if a.limit and reviewed >= a.limit:
            break
    print(f"\nreviewed {reviewed}: pass={passed} fail={failed} quarantine={quar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
