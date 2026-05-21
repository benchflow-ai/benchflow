#!/usr/bin/env python3
"""Generate ``dashboard/data.json`` — the feed for the BenchFlow v0.5 dashboard.

Sections, two live and the rest authored / derived:

  live      — test results (parsed from ``junit.xml``)
            — the jobs tree (scanned from ``jobs/``: groups → runs → tasks →
              artifacts, mirroring the folder layout)
            — the work timeline (derived from ``git log main..v0.5-integration``)
  authored  — the concept map (the architecture)
            — the roadmap (the v0.5 milestones + Linear issues)
            — the agent advisories (the 4×-review punch list), each cross-linked
              to the capability and the job group it corresponds to

Usage::

    python dashboard/generate.py              # reuse the last junit.xml
    python dashboard/generate.py --run-tests  # re-run the suite first (~70s)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"
JUNIT = DASH / "junit.xml"
OUT = DASH / "data.json"
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}$")


# --------------------------------------------------------------------------
# Live source 1 — test results
# --------------------------------------------------------------------------
def run_suite() -> None:
    print("running the test suite (this takes ~70s) ...", flush=True)
    subprocess.run(
        ["uv", "run", "--extra", "dev", "python", "-m", "pytest", "tests/",
         "-q", "-p", "no:randomly", f"--junitxml={JUNIT}"],
        cwd=ROOT, check=False,
    )


def collect_tests() -> dict:
    if not JUNIT.is_file():
        return {
            "available": False,
            "note": "no junit.xml yet — run: python dashboard/generate.py --run-tests",
            "summary": {"passed": 0, "failed": 0, "skipped": 0, "total": 0},
            "suites": [], "failures": [],
        }
    root = ET.parse(JUNIT).getroot()
    suites: dict[str, dict] = {}
    failures: list[dict] = []
    passed = failed = skipped = 0
    for c in root.iter("testcase"):
        fname = c.get("classname") or c.get("file") or "?"
        s = suites.setdefault(
            fname, {"name": fname, "passed": 0, "failed": 0, "skipped": 0, "time": 0.0}
        )
        s["time"] += float(c.get("time") or 0.0)
        bad, err, skp = c.find("failure"), c.find("error"), c.find("skipped")
        if bad is not None or err is not None:
            node = bad if bad is not None else err
            failed += 1
            s["failed"] += 1
            failures.append({
                "name": f"{c.get('classname','')}::{c.get('name','')}",
                "file": fname,
                "message": (node.get("message") or "").strip()[:280],
            })
        elif skp is not None:
            skipped += 1
            s["skipped"] += 1
        else:
            passed += 1
            s["passed"] += 1
    suite_list = sorted(suites.values(), key=lambda s: (-s["failed"], s["name"]))
    for s in suite_list:
        s["time"] = round(s["time"], 2)
    return {
        "available": True,
        "summary": {"passed": passed, "failed": failed, "skipped": skipped,
                    "total": passed + failed + skipped},
        "suites": suite_list, "failures": failures,
    }


# --------------------------------------------------------------------------
# Live source 2 — the jobs tree (jobs/ → groups → runs → tasks → artifacts)
# --------------------------------------------------------------------------
def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _count_lines(p: Path) -> int:
    try:
        return sum(1 for _ in p.open())
    except Exception:
        return 0


# Each job group, the capability it exercises, and the advisory IDs it
# corresponds to — this is the jobs ↔ agent-advisor correspondence.
GROUP_META = {
    "main": {
        "label": "Unit-test rollouts",
        "blurb": "Rollouts the pytest suite writes while exercising the kernel.",
        "capability": None, "advisories": [],
    },
    "e2e": {
        "label": "Live e2e rollouts",
        "blurb": "Real agent runs on ClawsBench (Daytona + Docker) — capability 6.",
        "capability": 6, "advisories": ["MUST-3"],
    },
    "e2e-branch": {
        "label": "Branching rollout",
        "blurb": "The live Branch lifecycle — checkpoint → fork → V(root)=1.0 — capability 7.",
        "capability": 7, "advisories": [],
    },
    "environment": {
        "label": "Environment-plane probes",
        "blurb": "Manifest provisioning / readiness checks — capabilities 2 & 3.",
        "capability": 2, "advisories": ["MUST-2"],
    },
}

_ART_KIND = {
    "result.json": "result", "config.json": "config", "prompts.json": "prompts",
    "timing.json": "timing", "rewards.jsonl": "rewards",
}


def _task_artifacts(d: Path) -> list[dict]:
    """List the artifact files a rollout directory holds."""
    arts: list[dict] = []
    for child in sorted(d.iterdir()):
        rel = child.name
        if child.is_file():
            arts.append({"name": rel, "kind": _ART_KIND.get(rel, "file"), "info": ""})
        elif child.is_dir():
            files = sorted(p for p in child.rglob("*") if p.is_file())
            if rel == "trajectory" or rel == "agent":
                tj = next((p for p in files if p.name.endswith(".jsonl")), None)
                if tj is not None and rel == "trajectory":
                    arts.append({
                        "name": f"{rel}/{tj.name}", "kind": "trajectory",
                        "info": f"{_count_lines(tj)} events",
                    })
            if rel == "verifier":
                rf = child / "reward.txt"
                if rf.is_file():
                    try:
                        arts.append({"name": "verifier/reward.txt", "kind": "reward",
                                     "info": rf.read_text().strip()[:12]})
                    except Exception:
                        pass
            if files and rel in ("agent", "artifacts", "verifier"):
                arts.append({"name": f"{rel}/", "kind": "dir",
                             "info": f"{len(files)} file(s)"})
    return arts


def _task_row(d: Path) -> dict:
    result = _read_json(d / "result.json")
    config = _read_json(d / "config.json")
    traj = 0
    for cand in (d / "trajectory" / "acp_trajectory.jsonl",
                 d / "agent" / "acp_trajectory.jsonl"):
        if cand.is_file():
            traj = _count_lines(cand)
            break
    reward = None
    if isinstance(result.get("rewards"), dict):
        reward = result["rewards"].get("reward")
    if reward is None:
        rf = d / "verifier" / "reward.txt"
        if rf.is_file():
            try:
                reward = float(rf.read_text().strip())
            except Exception:
                reward = None
    timing = result.get("timing") or {}
    return {
        "name": result.get("task_name") or d.name.split("__")[0],
        "rollout": d.name,
        "agent": result.get("agent") or config.get("agent") or "—",
        "model": result.get("model") or config.get("model") or "—",
        "environment": config.get("environment") or "—",
        "reward": reward,
        "error": (result.get("error") or "")[:240] or None,
        "verifier_error": (result.get("verifier_error") or "")[:240] or None,
        "trajectory_events": traj,
        "total_time": round(float(timing.get("total") or 0.0), 2),
        "artifacts": _task_artifacts(d),
    }


def _is_task_dir(d: Path) -> bool:
    """A rollout directory — has a result/config file or a rollout subdir.

    Some rollouts (e.g. the ACP smoke runs) carry only ``agent/`` /
    ``verifier/`` / ``artifacts/`` subdirs and no top-level JSON, so the
    subdir markers matter too. A *run* directory (which only holds task
    dirs) has none of these directly, so it is not misread as a task.
    """
    if any((d / m).is_file() for m in
           ("result.json", "config.json", "timing.json",
            "prompts.json", "rewards.jsonl")):
        return True
    return any((d / s).is_dir() for s in
               ("trajectory", "verifier", "agent", "artifacts"))


def collect_jobs() -> dict:
    """Scan jobs/ into groups → runs → tasks, mirroring the folder layout."""
    jobs = ROOT / "jobs"
    if not jobs.is_dir():
        return {"groups": [], "total_tasks": 0}

    # group name -> { run id -> [task dirs] }
    groups: dict[str, dict[str, list[Path]]] = {}
    for top in sorted(jobs.iterdir()):
        if not top.is_dir():
            continue
        if TS_RE.match(top.name):
            # an ungrouped run directly under jobs/
            runs = groups.setdefault("main", {})
            runs[top.name] = [c for c in sorted(top.iterdir())
                              if c.is_dir() and _is_task_dir(c)]
        else:
            # a named group: e2e / e2e-branch / environment. Its children
            # are either run dirs (holding task dirs) or task dirs directly.
            runs = groups.setdefault(top.name, {})
            direct: list[Path] = []
            for child in sorted(top.iterdir()):
                if not child.is_dir():
                    continue
                if _is_task_dir(child):
                    direct.append(child)
                else:
                    runs[child.name] = [c for c in sorted(child.iterdir())
                                        if c.is_dir() and _is_task_dir(c)]
            if direct:
                runs["(tasks)"] = direct

    total = 0
    out_groups: list[dict] = []
    order = ["e2e-branch", "e2e", "environment", "main"]
    for name in sorted(groups, key=lambda n: (order.index(n) if n in order else 9, n)):
        meta = GROUP_META.get(name, {"label": name, "blurb": "",
                                     "capability": None, "advisories": []})
        run_list: list[dict] = []
        for run_id in sorted(groups[name], reverse=True):
            tasks = [_task_row(d) for d in groups[name][run_id]]
            total += len(tasks)
            run_list.append({"id": run_id, "tasks": tasks})
        out_groups.append({
            "name": name,
            "label": meta["label"],
            "blurb": meta["blurb"],
            "capability": meta["capability"],
            "advisories": meta["advisories"],
            "runs": run_list,
            "n_tasks": sum(len(r["tasks"]) for r in run_list),
        })
    return {"groups": out_groups, "total_tasks": total}


# --------------------------------------------------------------------------
# Live source 3 — the work timeline (git log main..v0.5-integration)
# --------------------------------------------------------------------------
def _commit_type(subject: str) -> str:
    s = subject.lower()
    if s.startswith("merge"):
        return "merge"
    for t in ("feat", "fix", "docs", "build", "refactor", "chore", "test"):
        if s.startswith(t + "(") or s.startswith(t + ":"):
            return t
    return "other"


def collect_timeline() -> list[dict]:
    """The v0.5 work timeline — every commit on main..v0.5-integration."""
    try:
        raw = subprocess.run(
            ["git", "log", "--pretty=format:%h\x1f%ad\x1f%s",
             "--date=format:%Y-%m-%d %H:%M", "main..v0.5-integration"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except Exception:
        return []
    events: list[dict] = []
    for line in raw.splitlines():
        parts = line.split("\x1f")
        if len(parts) != 3:
            continue
        h, date, subject = parts
        events.append({
            "hash": h, "date": date, "type": _commit_type(subject),
            "subject": subject,
        })
    return events


# --------------------------------------------------------------------------
# Authored — concept map
# --------------------------------------------------------------------------
CONCEPT_MAP = {
    "entry": "bench CLI · bf.run() · environment manifest",
    "kernel": {"name": "KERNEL",
               "blurb": "Rollout lifecycle · reward · trajectory. Imports only contracts/."},
    "planes": [
        {"name": "Sandbox", "role": "where it runs",
         "detail": "Compute substrate — Local, Docker, Daytona, Modal. BYO via the Sandbox protocol.",
         "han": "—"},
        {"name": "Agent", "role": "who acts",
         "detail": "The agent under test / policy under training. Protocol: ACP. The Session is its real surface.",
         "han": "H — Harness"},
        {"name": "Environment", "role": "the world",
         "detail": "The stateful world. Declarative environment.toml manifest; owns provision→snapshot→restore→teardown.",
         "han": "S — State"},
        {"name": "Reward", "role": "how it's scored",
         "detail": "RewardFunc / Rubric / verifier. Scores any RolloutNode across five spaces.",
         "han": "V — Verifier"},
    ],
    "execution_model": [
        {"name": "Job", "kind": "set",
         "detail": "A set of Rollouts run together — an eval sweep, a GRPO group, a CL sequence."},
        {"name": "Rollout", "kind": "PRIMITIVE",
         "detail": "One RL episode = a TREE of states. A linear rollout is a degree-1 tree."},
        {"name": "Step", "kind": "PRIMITIVE",
         "detail": "One edge of the tree: (reason → act) → (tool-in → tool-out). Han's atomic unit."},
        {"name": "Branch", "kind": "PRIMITIVE",
         "detail": "The snapshot-and-fork operation — a node with >1 child. The value-function engine."},
        {"name": "Trajectory", "kind": "DERIVED VIEW",
         "detail": "One root-to-leaf path. Computed from the tree, never declared. The trainer export unit."},
        {"name": "Scene", "kind": "AUTHORING SUGAR",
         "detail": "A declared role/skill span. Desugars completely to per-Step config; no runtime object."},
    ],
    "capabilities": [
        {"n": 1, "name": "SkillsBench", "planes": ["Environment", "Reward"],
         "status": "shipped", "issue": None,
         "fit": "Environment-plane benchmark package; the Reward plane's Memory space scores skill use + updates."},
        {"n": 2, "name": "ClawsBench", "planes": ["Environment"],
         "status": "shipped", "issue": "ENG-124",
         "fit": "The stateful-mock-service benchmark — base_image + [[services]], framework-started. The manifest's design partner."},
        {"n": 3, "name": "chi-bench", "planes": ["Environment"],
         "status": "shipped", "issue": "ENG-124",
         "fit": "Same SMSB archetype, owns_lifecycle=true. External proof: onboarded by a ~25-line manifest."},
        {"n": 4, "name": "NudgeBench", "planes": ["Agent", "Reward"],
         "status": "partial", "issue": "ENG-126",
         "fit": "ACP interaction model (nudges + ask_user) + tree-native Rollout; the Action space scores follow-up."},
        {"n": 5, "name": "Continual learning", "planes": ["Reward"],
         "status": "shipped", "issue": "ENG-127",
         "fit": "A Job in sequential-shared mode over a persistent, versioned learner store; the Memory space tracks it."},
        {"n": 6, "name": "RL-native", "planes": ["Reward"],
         "status": "shipped", "issue": "ENG-127",
         "fit": "The whole execution model — Rollout is a tree, Trajectory a path, exported as a trainer-ready record."},
        {"n": 7, "name": "Branching · rollback", "planes": ["Environment", "Reward"],
         "status": "shipped", "issue": "ENG-127",
         "fit": "The RL-native substrate itself — first-class Branch, Environment snapshot/restore. Live e2e: V(root)=1.0."},
        {"n": 8, "name": "Env adapters", "planes": ["Environment"],
         "status": "partial", "issue": "ENG-128",
         "fit": "Inbound adapters translate foreign formats — Harbor + Terminal-Bench shipped; ORS / PrimeIntellect pending."},
    ],
    "spaces": ["output", "action", "reasoning", "memory", "latent"],
}


# --------------------------------------------------------------------------
# Authored — roadmap
# --------------------------------------------------------------------------
ROADMAP = {
    "project": "BenchFlow v0.5 — architecture migration",
    "branch": "v0.5-integration",
    "milestones": [
        {"id": "M1", "name": "Cut dead architecture", "issues": [
            {"id": "ENG-117", "title": "RFC: v0.5 architecture — four planes, contracts, execution model", "status": "In Review"},
            {"id": "ENG-118", "title": "Cut dead architecture: delete orphaned modules + call-graph guard", "status": "Backlog"},
            {"id": "ENG-119", "title": "Collapse shim layers: sdk.py + runtime.py into one Rollout path", "status": "Backlog"},
            {"id": "ENG-104", "title": "chore: start v0.5 development after v0.4.0 release", "status": "In Progress"},
        ]},
        {"id": "M2", "name": "Four-plane contracts", "issues": [
            {"id": "ENG-120", "title": "Extract contracts/ package + uv workspace boundaries", "status": "Backlog"},
            {"id": "ENG-121", "title": "Sandbox plane: managed + BYO sandboxes", "status": "Backlog"},
            {"id": "ENG-122", "title": "Agent plane: managed + BYO agents (ACP + ACPX)", "status": "Backlog"},
            {"id": "ENG-123", "title": "Sandbox hardening + verifier isolation as a capability", "status": "Backlog"},
            {"id": "ENG-92", "title": "release blocker: reconcile Firecracker/K8s sandbox lanes", "status": "In Progress"},
        ]},
        {"id": "M3", "name": "Environment & Reward planes", "issues": [
            {"id": "ENG-124", "title": "Environment plane: tools/mock-services, TaskDatabase, AccountBroker", "status": "In Progress"},
            {"id": "ENG-125", "title": "Reward plane: wire Rubric/RewardFunc into Rollout, one schema", "status": "In Progress"},
        ]},
        {"id": "M4", "name": "RL-native, benchmarks & scale", "issues": [
            {"id": "ENG-126", "title": "Execution model: Scene / Round / Step + multi-* tasks", "status": "In Progress"},
            {"id": "ENG-127", "title": "RL-native results & reproducibility", "status": "In Progress"},
            {"id": "ENG-128", "title": "Adapter plane: support all benchmark sources", "status": "In Progress"},
            {"id": "ENG-129", "title": "Conformance suite: integration tests bound all behavior", "status": "Backlog"},
            {"id": "ENG-130", "title": "Failure semantics & eval integrity", "status": "Backlog"},
            {"id": "ENG-131", "title": "Scale: orchestration, concurrency, cost & task-schema versioning", "status": "Backlog"},
            {"id": "ENG-93", "title": "release blocker: complete trace-to-task e2e evidence", "status": "In Progress"},
            {"id": "ENG-98", "title": "release blocker: select real smoke tasks for the lanes", "status": "In Progress"},
        ]},
    ],
}


# --------------------------------------------------------------------------
# Authored — agent advisories (the 4×-review punch list), cross-linked
# --------------------------------------------------------------------------
ADVISORIES = {
    "source": "4× subagent review of the v0.5 capability fix pass (commits 35cdd47 → 0406a75)",
    "items": [
        {"id": "MUST-1", "severity": "must-fix", "status": "resolved", "agent": "Review consensus",
         "capability": 5, "group": "e2e",
         "title": "Continual learning — wire a real memory/skills producer",
         "detail": "The LearnerStore never received evolved skills and the Memory scorer read a memory_delta nothing wrote. New module learner_skills.py wires the rollout↔store data path end-to-end."},
        {"id": "MUST-2", "severity": "must-fix", "status": "resolved", "agent": "Review consensus",
         "capability": 8, "group": "environment",
         "title": "Adapter file-map collision silently drops a file",
         "detail": "TerminalBenchAdapter._build_file_map now raises ValueError when two sources map to the same native destination."},
        {"id": "MUST-3", "severity": "must-fix", "status": "resolved", "agent": "Review consensus",
         "capability": 6, "group": "e2e",
         "title": "Test deadlock on a concurrency regression",
         "detail": "test_parallel_independent_still_overlaps now wraps job.run() in asyncio.wait_for so a regression fails fast."},
        {"id": "MUST-4", "severity": "must-fix", "status": "resolved", "agent": "Reviewer 1 (correctness)",
         "capability": 5, "group": "e2e",
         "title": "Memory scorer was a tautology",
         "detail": "The scorer's `expected` answer-key was derived from the agent's own diff → precision=recall=1.0 always. Dropped; the scorer now honestly grades activity."},
        {"id": "SHOULD-1", "severity": "should-fix", "status": "resolved", "agent": "Reviewer 4 (quality)",
         "capability": 5, "group": "e2e",
         "title": "Store committed un-normalized skills",
         "detail": "_commit_learner_generation now commits the normalized after_skills so the store is byte-identical to the recorded delta."},
        {"id": "SHOULD-2", "severity": "should-fix", "status": "resolved", "agent": "Reviewers 1 & 4",
         "capability": 5, "group": "e2e",
         "title": "learner_nodes leaked across run() calls",
         "detail": "Reset per-run; RolloutNode ids index-prefixed so same-named tasks stay distinct."},
        {"id": "SHOULD-3", "severity": "should-fix", "status": "resolved", "agent": "Reviewer 4 (quality)",
         "capability": 5, "group": "e2e",
         "title": "Resumed continual-learning job silently restarts the curve",
         "detail": "run() now warns: the LearnerStore is process-local, so a resume restarts at generation 0."},
        {"id": "SHOULD-4", "severity": "should-fix", "status": "resolved", "agent": "Reviewer 3 (tests)",
         "capability": 5, "group": "e2e",
         "title": "Stale _revert docstrings + loose evolved_skills type",
         "detail": "_revert docstrings corrected; evolved_skills tightened to dict[str, str]."},
        {"id": "OPEN-1", "severity": "future-work", "status": "open", "agent": "Reviewer 1 (correctness)",
         "capability": 5, "group": "e2e",
         "title": "Real expected_skills fixture from the task definition",
         "detail": "The Memory scorer grades activity, not correctness, until tasks can declare an expected-skills fixture. Tracked on ENG-125."},
        {"id": "OPEN-2", "severity": "note", "status": "open", "agent": "Reviewer 4 (quality)",
         "capability": None, "group": None,
         "title": "thermo-nuclear code-quality-review skill not installed",
         "detail": "Reviewer 4 was assigned the cursor thermo-nuclear skill; it is not installed locally, so an adversarial review was used as the fallback."},
    ],
}


# --------------------------------------------------------------------------
def main() -> int:
    if "--run-tests" in sys.argv:
        run_suite()
    tests = collect_tests()
    jobs = collect_jobs()
    timeline = collect_timeline()

    done = sum(1 for c in CONCEPT_MAP["capabilities"] if c["status"] == "shipped")
    all_issues = [i for m in ROADMAP["milestones"] for i in m["issues"]]
    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "tests": tests["summary"],
            "capabilities_shipped": done,
            "capabilities_total": len(CONCEPT_MAP["capabilities"]),
            "issues_total": len(all_issues),
            "issues_active": sum(1 for i in all_issues
                                 if i["status"] in ("In Progress", "In Review")),
            "jobs_total": jobs["total_tasks"],
            "job_groups": len(jobs["groups"]),
            "commits": len(timeline),
            "advisories_open": sum(1 for a in ADVISORIES["items"]
                                   if a["status"] == "open"),
        },
        "concept_map": CONCEPT_MAP,
        "tests": tests,
        "roadmap": ROADMAP,
        "jobs": jobs,
        "timeline": timeline,
        "advisories": ADVISORIES,
    }
    OUT.write_text(json.dumps(data, indent=2))
    s = data["summary"]["tests"]
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  tests: {s['passed']}p/{s['failed']}f/{s['skipped']}s   "
          f"jobs: {jobs['total_tasks']} tasks in {len(jobs['groups'])} groups   "
          f"timeline: {len(timeline)} commits   "
          f"capabilities: {done}/{len(CONCEPT_MAP['capabilities'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
