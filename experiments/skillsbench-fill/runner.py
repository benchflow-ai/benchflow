#!/usr/bin/env python3
"""Rolling pooled driver for the SkillsBench max-effort fill (runs on the VM).

Keeps --concurrency run_cell.sh processes in flight, drawn from state/queue.jsonl.
- Known-heavy tasks start on docker (overflow Daytona's 10GB cap).
- A daytona cell that fails with ENOSPC/infra error is retried on docker (once).
- Restart-safe: skips cells already completed/review_pass/published.

Usage:
  runner.py --concurrency 18 --only-tasks citation-check,3d-scan-calc       # canary
  runner.py --concurrency 50                                                # pooled scale (VM-safe)
  runner.py --concurrency 18 --only-models opus-4.8 --only-tasks citation-check
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
JOBS = ROOT / "jobs"
STATE = ROOT / "state"
QUEUE = STATE / "queue.jsonl"

# Cap concurrent LOCAL docker sandboxes (heavy tasks). Many simultaneous docker
# setups (apt/pip) + GCP search-domain DNS amplification saturate systemd-resolved
# and wedge the whole VM network. Bound it. Set in main() from --docker-concurrency.
_DOCKER_SEM: threading.Semaphore | None = None

# Tasks that overflow Daytona's hard 10GB image cap (storage_mb>10240 or heavy ML image).
HEAVY = {
    "latex-formula-extraction", "fix-druid-loophole-cve", "organize-messy-files",
    "syzkaller-ppdev-syzlang", "video-tutorial-indexer", "video-filler-word-remover",
    "debug-trl-grpo", "data-to-d3", "fix-visual-stability", "multilingual-video-dubbing",
    "react-performance-debugging", "taxonomy-tree-merge",
}
DONE_STATES = {"completed", "review_pass", "published"}


def cell_state(cell: str) -> dict:
    f = STATE / f"{cell}.json"
    if f.exists():
        try:
            return json.load(open(f)) or {}
        except Exception:
            return {}
    return {}


def should_skip(st: dict) -> bool:
    """Skip cells already done, or actively running under another runner/orphan
    (fresh < 45 min). Stale 'running' (crashed) falls through and is re-run."""
    status = st.get("status")
    if status in DONE_STATES:
        return True
    if status == "running":
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(st.get("started_at", ""))).total_seconds()
        except Exception:
            age = 1e9
        return age < 2700
    return False


def run_cell(model, mode, task, slot, sandbox):
    cell = f"{model}__{mode}__{task}__t{slot}"
    cmd = ["bash", str(ROOT / "run_cell.sh"), model, mode, task, str(slot), sandbox, str(JOBS), str(STATE)]
    if sandbox == "docker" and _DOCKER_SEM is not None:
        # hold the slot for the whole docker run so only N docker sandboxes exist at once
        with _DOCKER_SEM:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        return cell, json.load(open(STATE / f"{cell}.json"))
    except Exception:
        return cell, {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=18)
    ap.add_argument("--docker-concurrency", type=int, default=4,
                    help="max concurrent LOCAL docker sandboxes (heavy tasks); guards VM DNS/network")
    ap.add_argument("--skip-heavy", action="store_true",
                    help="skip HEAVY-set tasks entirely (their docker builds fail on this VM); run only light->daytona cells")
    ap.add_argument("--only-tasks", default="")
    ap.add_argument("--only-models", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-retries", type=int, default=1)
    a = ap.parse_args()
    JOBS.mkdir(exist_ok=True)
    STATE.mkdir(exist_ok=True)
    global _DOCKER_SEM
    _DOCKER_SEM = threading.Semaphore(max(1, a.docker_concurrency))

    tasks_f = {t for t in a.only_tasks.split(",") if t}
    models_f = {m for m in a.only_models.split(",") if m}
    work = []
    for line in open(QUEUE):
        q = json.loads(line)
        if tasks_f and q["task"] not in tasks_f:
            continue
        if models_f and q["model"] not in models_f:
            continue
        if a.skip_heavy and q["task"] in HEAVY:
            continue
        if should_skip(cell_state(q["cell_id"])):
            continue
        sandbox = "docker" if q["task"] in HEAVY else "daytona"
        work.append((q["model"], q["skill_mode"], q["task"], q["trial_slot"], sandbox))
    if a.limit:
        work = work[:a.limit]
    print(f"work: {len(work)} cells | concurrency={a.concurrency} | docker-concurrency={a.docker_concurrency} | heavy->docker", flush=True)

    retries: dict[str, int] = {}
    done = ok = 0
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        pending = {ex.submit(run_cell, *w): w for w in work}
        while pending:
            finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in finished:
                model, mode, task, slot, sandbox = pending.pop(fut)
                cell, st = fut.result()
                done += 1
                status = st.get("status")
                if status == "completed":
                    ok += 1
                elif status == "run_failed":
                    n = retries.get(cell, 0)
                    if n < a.max_retries:
                        retries[cell] = n + 1
                        nb = "docker" if (sandbox == "daytona" and st.get("enospc")) else sandbox
                        print(f"  retry {cell} on {nb} (was {sandbox}, enospc={st.get('enospc')})", flush=True)
                        w2 = (model, mode, task, slot, nb)
                        pending[ex.submit(run_cell, *w2)] = w2
                print(f"[{done}] {cell} -> {status} (completed so far: {ok})", flush=True)
    print(f"DONE: {ok} completed of {len(work)} initial cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
