#!/usr/bin/env python3
"""Per-cell evidence for the subagent /benchflow-experiment-review audit.
Given cell_ids, emit the exact facts each subagent needs to apply the checklist:
result fields (error/partial/reward/timing), acp+llm trajectory presence, and the
task_skills_loading posture (tsl) from the skill's own extract_harness_skills.py.
Output: JSON list. The subagent JUDGES from this (pass iff all checklist items hold)."""
import json, sys, os, subprocess, re, glob

EX = os.path.expanduser("~/bf-pr607/.claude/skills/benchflow-experiment-review/scripts/extract_harness_skills.py")
SB = os.path.expanduser("~/skillsbench/tasks")


def extract_reward(r, rd):
    """Canonical reward: result.json rewards.reward, then verifier/reward.txt,
    then the terminal record in rewards.jsonl. Uses `is not None` throughout so a
    legitimate reward of 0.0 (falsy) is never mistaken for a missing value."""
    rw = r.get("rewards")
    if isinstance(rw, dict) and rw.get("reward") is not None:
        return rw.get("reward")
    rt = os.path.join(rd, "verifier", "reward.txt")
    if os.path.exists(rt):
        try:
            s = open(rt).read().strip()
            if s:
                return float(s)
        except Exception:
            pass
    rl = os.path.join(rd, "rewards.jsonl")
    if os.path.exists(rl):
        try:
            val = None
            for line in open(rl):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("tag") == "reward" and rec.get("type") == "terminal" and rec.get("value") is not None:
                    val = rec.get("value")  # keep the last terminal reward
            if val is not None:
                return val
        except Exception:
            pass
    return None


def extract_tokens(r):
    """Canonical total tokens: result.json agent_result.total_tokens, falling back
    to final_metrics prompt+completion. `is not None` so a true 0 isn't dropped."""
    ar = r.get("agent_result")
    if isinstance(ar, dict) and ar.get("total_tokens") is not None:
        return ar.get("total_tokens")
    fm = r.get("final_metrics")
    if isinstance(fm, dict):
        p, c = fm.get("total_prompt_tokens"), fm.get("total_completion_tokens")
        if p is not None or c is not None:
            return (p or 0) + (c or 0)
    return None


def rollout_of(cell):
    sf = os.path.expanduser(f"~/sb-fill/state/{cell}.json")
    if os.path.exists(sf):
        try:
            rd = (json.load(open(sf)) or {}).get("rollout_dir", "")
            if rd and os.path.isdir(rd):
                return rd
        except Exception:
            pass
    # fallback: newest complete rollout under jobs/<cell>/
    cands = [d for d in glob.glob(os.path.expanduser(f"~/sb-fill/jobs/{cell}/**/"), recursive=True)
             if os.path.exists(os.path.join(d, "result.json"))]
    return sorted(cands)[-1].rstrip("/") if cands else ""


def evidence(cell):
    m = re.match(r"(.+?)__(with|without)__(.+)__t(\d+)$", cell)
    if not m:
        return {"cell": cell, "artifacts": "unparseable_cell"}
    model, mode, task, _ = m.groups()
    ev = {"cell": cell, "model": model, "mode": mode, "task": task}
    rd = rollout_of(cell)
    ev["rollout_dir"] = rd
    if not rd:
        ev["artifacts"] = "missing"
        return ev
    rj = os.path.join(rd, "result.json")
    if not os.path.exists(rj):
        ev["artifacts"] = "no_result"
        return ev
    try:
        r = json.load(open(rj))
        ev["err"] = r.get("error")
        ev["partial"] = r.get("partial_trajectory")
        ev["reward"] = extract_reward(r, rd)
        ev["timing_total"] = (r.get("timing") or {}).get("total")
        ev["tokens"] = extract_tokens(r)
    except Exception as e:
        ev["artifacts"] = f"result_unreadable:{e}"
        return ev
    acp = os.path.join(rd, "trajectory", "acp_trajectory.jsonl")
    llm = os.path.join(rd, "trajectory", "llm_trajectory.jsonl")
    ev["acp_present"] = os.path.exists(acp) and os.path.getsize(acp) > 0
    ev["llm_present"] = os.path.exists(llm) and os.path.getsize(llm) > 0
    if ev["llm_present"] and os.path.exists(EX):
        try:
            o = subprocess.run(["python3", EX, llm, "--task-path", f"{SB}/{task}"],
                               capture_output=True, text=True, timeout=90)
            try:
                tj = json.loads(o.stdout.strip())
            except Exception:
                mm = re.search(r'"task_skills_loading"\s*:\s*(\d+)', o.stdout)
                tj = {"task_skills_loading": int(mm.group(1))} if mm else {}
            ev["tsl"] = tj.get("task_skills_loading")
            ev["tsl_status"] = tj.get("task_skills_loading_status")
        except Exception as e:
            ev["tsl_error"] = str(e)
    return ev


print(json.dumps([evidence(c) for c in sys.argv[1:]]))
