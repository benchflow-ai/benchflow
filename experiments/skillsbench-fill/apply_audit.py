#!/usr/bin/env python3
"""Apply the audit Workflow's AGENT verdicts as final (runs on the VM).

The agent's per-cell conclusion overrides the mechanical (script) verdict. The audit
Workflow's "flag" verdict is NOT uniform, so we compose three orthogonal primitives
instead of a flat flag->delete:

  delete_from_pr5 : remove these cell dirs from refs/pr/5 (CommitOperationDelete).
  set_verdict     : {cell: "pass"|"fail"|"quarantine"} written into review/<cell>.json
                    (verdict + health + note + optional task_skills_loading), overriding
                    the mechanical verdict. pass=>publishable (cron republishes),
                    fail/quarantine=>excluded.
  requeue         : rm state/<cell>.json + published/<cell>.json so the runner re-runs it
                    (use ONLY for transient/infra failures; do NOT requeue cells whose
                    root cause -- open network egress, broken skill bundling -- is unfixed,
                    or they just reproduce the problem).

Then rebuild the ledger; the cron republishes the corrected set.

Usage: apply_audit.py --verdicts agent_verdicts.json [--dry-run]
  verdicts json:
    {"delete_from_pr5": ["cell", ...],
     "set_verdict": {"cell": {"verdict": "pass"|"fail"|"quarantine",
                              "note": "...", "task_skills_loading": 1}, ...},
     "requeue": ["cell", ...]}
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
REPO = "benchflow/skillsbench-leaderboard"
PR_REF = "refs/pr/5"


def hf_token() -> str:
    if os.environ.get("HUGGING_FACE_TOKEN"):
        return os.environ["HUGGING_FACE_TOKEN"]
    for p in (os.path.expanduser("~/keys.env"), os.path.expanduser("~/.env")):
        try:
            for line in open(p):
                m = re.match(r'^\s*(?:export\s+)?HUGGING_FACE_TOKEN\s*=\s*["\']?([^"\'\s]+)', line)
                if m:
                    return m.group(1)
        except FileNotFoundError:
            continue
    raise SystemExit("no HUGGING_FACE_TOKEN")


def _published_path(cell: str):
    p = ROOT / "published" / f"{cell}.json"
    if p.exists():
        try:
            return json.load(open(p)).get("hf_path")
        except Exception:
            return None
    return None


def _set_verdict(cell: str, spec: dict, now: str):
    rv = ROOT / "review" / f"{cell}.json"
    d = {}
    if rv.exists():
        try:
            d = json.load(open(rv))
        except Exception:
            d = {}
    verdict = spec["verdict"]
    health = "healthy" if verdict == "pass" else "unhealthy"
    d.update(cell_id=cell, verdict=verdict, health=health, review_verdict=verdict,
             agent_final=True, note=spec.get("note", ""), updated_at=now)
    if "task_skills_loading" in spec:
        d["task_skills_loading"] = spec["task_skills_loading"]
        d["task_skills_loading_status"] = "agent_override"
    json.dump(d, open(rv, "w"), indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    v = json.load(open(a.verdicts))
    delete_from_pr5 = list(dict.fromkeys(v.get("delete_from_pr5", [])))
    set_verdict = v.get("set_verdict", {})
    requeue = list(dict.fromkeys(v.get("requeue", [])))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # 1) override verdicts
    counts = {"pass": 0, "fail": 0, "quarantine": 0}
    for cell, spec in set_verdict.items():
        counts[spec["verdict"]] = counts.get(spec["verdict"], 0) + 1
        if not a.dry_run:
            _set_verdict(cell, spec, now)

    # 2) resolve PR5 deletes
    del_paths = []
    for cell in delete_from_pr5:
        hp = _published_path(cell)
        if hp:
            del_paths.append((cell, hp))
        else:
            print(f"  [warn] {cell} requested for PR5 delete but no published record (skip)")

    # 3) requeue: fully reset the cell (state + published + review) so the runner re-runs
    # it AND review_cell.py re-reviews the fresh result (it skips cells with a review file).
    for cell in requeue:
        if not a.dry_run:
            (ROOT / "state" / f"{cell}.json").unlink(missing_ok=True)
            (ROOT / "published" / f"{cell}.json").unlink(missing_ok=True)
            (ROOT / "review" / f"{cell}.json").unlink(missing_ok=True)

    print(f"set_verdict: {counts}  | delete_from_pr5: {len(del_paths)} resolved / {len(delete_from_pr5)} req"
          f"  | requeue: {len(requeue)}")
    for c, p in del_paths:
        print(f"  DEL {p}   ({c})")
    if a.dry_run:
        print("DRY RUN — no review writes, no PR5 deletes, no requeue")
        return 0

    # remove published records for deleted cells too (so dashboard stops linking them)
    for cell, _ in del_paths:
        (ROOT / "published" / f"{cell}.json").unlink(missing_ok=True)

    if del_paths:
        os.environ.setdefault("HF_TOKEN", hf_token())
        from huggingface_hub import CommitOperationDelete, HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        ops = [CommitOperationDelete(path_in_repo=p, is_folder=True) for _, p in del_paths]
        CH = 200
        for i in range(0, len(ops), CH):
            api.create_commit(repo_id=REPO, repo_type="dataset", revision=PR_REF, operations=ops[i:i + CH],
                              commit_message=f"audit: remove {len(ops)} agent-flagged contaminated cells")
            print(f"  deleted batch {i // CH + 1}: {len(ops[i:i+CH])} cells from {PR_REF}")
    os.system(f"cd {ROOT} && python3 build_ledger.py --root . | tail -1")
    print("[done] agent verdicts applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
