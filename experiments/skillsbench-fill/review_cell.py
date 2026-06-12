#!/usr/bin/env python3
"""Mechanical bulk reviewer for the SkillsBench max-effort fill (runs on the VM).

Automates the benchflow-experiment-review checklist per completed cell and writes
review/<cell>.json with a verdict + evidence. Subagents then deep-audit every
'fail' + a random sample for judgment (plausibility / reward-hacking).

For each state/<cell>.json with status==completed and no review yet:
  health      completed or normal_timeout with complete artifacts, reward in [0,1],
              timing.total>0
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
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
STATE, REVIEW = ROOT / "state", ROOT / "review"
ACCEPTED_TIMEOUTS = ROOT / "accepted_timeouts"
SB_TASKS = os.path.expanduser(
    os.environ.get("BENCHFLOW_SKILLSBENCH_TASKS")
    or os.environ.get("SKILLSBENCH_TASKS")
    or "~/skillsbench/tasks"
)


def default_extract_script() -> str:
    for candidate in (
        os.environ.get("BENCHFLOW_EXPERIMENT_REVIEW_EXTRACT"),
        "~/Experiment/benchflow/.claude/skills/benchflow-experiment-review/scripts/extract_harness_skills.py",
        str(ROOT.parents[1] / ".claude/skills/benchflow-experiment-review/scripts/extract_harness_skills.py"),
    ):
        if candidate and Path(os.path.expanduser(candidate)).exists():
            return os.path.expanduser(candidate)
    return os.path.expanduser(
        os.environ.get("BENCHFLOW_EXPERIMENT_REVIEW_EXTRACT")
        or "~/Experiment/benchflow/.claude/skills/benchflow-experiment-review/scripts/extract_harness_skills.py"
    )


EXTRACT = default_extract_script()
RECONCILE_FILES = {
    "experiments_ledger.json",
    "grid.json",
    "queue.jsonl",
    "raw_pool_availability.json",
    "reconcile_report.json",
    "repair_queue.jsonl",
    "token_repair_details.json",
    "token_repair_queue.jsonl",
}
THINK_RE = re.compile(rb"thinking|reasoning_content|reasoningContent|redacted_thinking|signature", re.I)


def _walk_json(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _opus_adaptive_max(llm: Path) -> tuple[bool, bool, int]:
    """Return (has_adaptive_thinking, has_output_config_max, thinking_refs)."""
    adaptive = False
    max_effort = False
    thinking_refs = 0
    if not llm.is_file():
        return adaptive, max_effort, thinking_refs
    with open(llm, "rb") as fh:
        raw = fh.read()
    thinking_refs = len(THINK_RE.findall(raw))
    for line in raw.splitlines():
        try:
            data = json.loads(line)
        except Exception:
            continue
        for obj in _walk_json(data):
            thinking = obj.get("thinking")
            if (
                isinstance(thinking, dict)
                and str(thinking.get("type", "")).lower() == "adaptive"
            ) or (isinstance(thinking, str) and "adaptive" in thinking.lower()):
                adaptive = True
            output_config = obj.get("output_config")
            if isinstance(output_config, dict) and str(output_config.get("effort", "")).lower() == "max":
                max_effort = True
        if adaptive and max_effort:
            break
    return adaptive, max_effort, thinking_refs


def _rollout(st: dict) -> Path | None:
    candidates = []
    rd = st.get("rollout_dir")
    if rd and Path(rd).is_dir():
        candidates.append(Path(rd))
    root = st.get("run_root")
    if root:
        candidates.extend(Path(d) for d in glob.glob(f"{root}/**/{st['task']}__*", recursive=True) if Path(d).is_dir())
    if not candidates:
        return None

    def score(path: Path) -> tuple:
        result = path / "result.json"
        llm = path / "trajectory/llm_trajectory.jsonl"
        acp = path / "trajectory/acp_trajectory.jsonl"
        cfg = path / "config.json"
        reward_valid = False
        try:
            with open(result) as fh:
                res = json.load(fh)
            rew = (res.get("rewards") or {}).get("reward")
            reward_valid = rew is not None and 0.0 <= float(rew) <= 1.0
        except Exception:
            pass
        complete = result.exists() and cfg.exists() and llm.exists() and acp.exists()
        llm_size = llm.stat().st_size if llm.exists() else 0
        acp_size = acp.stat().st_size if acp.exists() else 0
        try:
            mtime = result.stat().st_mtime
        except OSError:
            mtime = path.stat().st_mtime
        return (reward_valid, complete, llm_size > 50, acp_size > 100, mtime)

    # A cell directory may contain failed bootstrap/provider attempts plus a
    # later complete attempt. Prefer a scored rollout with verifier reward over
    # the raw state rollout_dir or lexical order.
    return max(dict.fromkeys(candidates), key=score)


def _accepted_timeout_overlay(cell_id: str) -> bool:
    """Return True only when a separate strict-audit overlay accepts a timeout.

    Raw ``result.json`` files stay immutable. A salvage pass can write
    ``accepted_timeouts/<cell>.json`` with ``accepted_normal_timeout=true`` after
    human/subagent review confirms the partial timeout has complete artifacts and
    no pending tool calls.
    """
    path = ACCEPTED_TIMEOUTS / f"{cell_id}.json"
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception:
        return False
    return bool(data.get("accepted_normal_timeout"))


def review(st: dict) -> dict:
    cell, model, mode, task = st["cell_id"], st["model"], st["skill_mode"], st["task"]
    now = datetime.now(UTC).isoformat(timespec="seconds")
    out = {"cell_id": cell, "updated_at": now, "checklist": {}, "notes": ""}
    if st.get("status") != "completed":
        return {**out, "verdict": "quarantine", "health": "unhealthy",
                "notes": f"status={st.get('status')} ({st.get('error')})"}
    roll = _rollout(st)
    if roll is None:
        return {**out, "verdict": "quarantine", "health": "unhealthy", "notes": "no rollout dir"}
    out["rollout_dir"] = str(roll)
    if "__" in roll.name:
        out["trial_id"] = roll.name.rsplit("__", 1)[1]
    try:
        with open(roll / "result.json") as fh:
            res = json.load(fh)
        with open(roll / "config.json") as fh:
            cfg = json.load(fh)
    except Exception as e:
        return {**out, "verdict": "quarantine", "health": "unhealthy", "notes": f"unreadable: {e}"}

    ar = res.get("agent_result") or {}
    ae = cfg.get("agent_env", {}) or {}
    acp, llm = roll / "trajectory/acp_trajectory.jsonl", roll / "trajectory/llm_trajectory.jsonl"
    rew = (res.get("rewards") or {}).get("reward")
    tot = (res.get("timing") or {}).get("total")
    tokens = ar.get("total_tokens")

    err = res.get("error")
    # A timeout may be a healthy normal_timeout only when the trajectory is not
    # partial and verifier metadata is complete. Do not paper over partial
    # trajectories just because the error string is a timeout.
    is_timeout = bool(err) and "timed out" in str(err).lower()
    out["outcome"] = "normal_timeout" if is_timeout else "completed"
    c = out["checklist"]
    c["error_ok"] = (err is None) or is_timeout
    summary = res.get("trajectory_summary") or {}
    out["partial_trajectory"] = bool(res.get("partial_trajectory"))
    out["trajectory_summary_partial"] = bool(
        summary.get("partial_trajectory", res.get("partial_trajectory"))
    )
    out["trajectory_source"] = res.get("trajectory_source") or summary.get(
        "trajectory_source"
    )
    try:
        c["reward_valid"] = rew is not None and 0.0 <= float(rew) <= 1.0
    except (TypeError, ValueError):
        c["reward_valid"] = False
    c["timing_ok"] = bool(tot and tot > 0)
    c["acp_present"] = acp.is_file() and acp.stat().st_size > 100
    c["llm_present"] = llm.is_file() and llm.stat().st_size > 50
    c["tokens_ok"] = bool(tokens and tokens > 0) if model != "gemini-3.5-flash" else True
    c["sandbox_daytona"] = str(cfg.get("environment") or st.get("sandbox") or "").lower() == "daytona"
    timeout_complete = (
        is_timeout
        and str(res.get("error_category") or "").lower() == "timeout"
        and not res.get("verifier_error")
        and c["reward_valid"]
        and c["timing_ok"]
        and c["acp_present"]
        and c["llm_present"]
        and c["tokens_ok"]
        and (ar.get("n_tool_calls") or 0) > 0
        and (summary.get("steps") or 0) > 0
        and (summary.get("tool_call_steps") or 0) > 0
        and str(res.get("trajectory_source") or summary.get("trajectory_source") or "") != "scraped"
    )
    out["timeout_complete_artifacts"] = timeout_complete
    accepted_normal_timeout = timeout_complete and _accepted_timeout_overlay(cell)
    out["accepted_normal_timeout"] = accepted_normal_timeout
    c["not_partial"] = (not res.get("partial_trajectory")) or accepted_normal_timeout
    c["trajectory_summary_ok"] = (
        not summary.get("partial_trajectory", res.get("partial_trajectory"))
    ) or accepted_normal_timeout

    # effort provenance
    if model == "opus-4.8":
        adaptive, max_effort, think = _opus_adaptive_max(llm)
        c["effort_ok"] = adaptive and max_effort
        out["thinking_refs"] = think
        out["opus_adaptive_thinking"] = adaptive
        out["opus_output_config_effort_max"] = max_effort
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
    if healthy:
        out["health"], out["verdict"] = "healthy", "pass"
    else:
        # The fill target needs publishable data, so any failed mechanical
        # health check is treated as rerunnable experiment-fidelity debt.
        out["health"], out["verdict"] = "unhealthy", "quarantine"
        out["notes"] = "rerun: " + ",".join(k for k, v in c.items() if not v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cell", default="")
    a = ap.parse_args()
    REVIEW.mkdir(exist_ok=True)
    existing_reviews = {}
    for f in glob.glob(str(REVIEW / "*.json")):
        try:
            with open(f) as fh:
                existing_reviews[Path(f).stem] = json.load(fh)
        except Exception:
            existing_reviews[Path(f).stem] = {"verdict": "quarantine"}
    reviewed = passed = failed = quar = 0
    for sf in sorted(glob.glob(str(STATE / "*.json"))):
        if os.path.basename(sf) in RECONCILE_FILES:
            continue
        try:
            with open(sf) as fh:
                st = json.load(fh)
        except Exception:
            continue
        if not isinstance(st, dict):
            continue
        cell = st.get("cell_id")
        if not cell or (a.cell and cell != a.cell):
            continue
        if st.get("status") != "completed":
            continue
        prior = existing_reviews.get(cell)
        if prior and prior.get("verdict") != "quarantine":
            continue
        rv = review(st)
        with open(REVIEW / f"{cell}.json", "w") as fh:
            json.dump(rv, fh, indent=2)
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
