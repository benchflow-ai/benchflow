#!/usr/bin/env python3
"""Generate ``dashboard/data.json`` — the feed for the BenchFlow v0.5 dashboard.

Sections, two live and the rest authored / derived:

  live      — test results (parsed from ``junit.xml``)
            — the jobs tree (scanned from ``jobs/``: groups → runs → tasks →
              artifacts, mirroring the folder layout; every artifact carries
              its actual file content, capped)
            — the experiments timeline (scanned from ``experiments/`` + ``labs/``)
  authored  — the concept map (the architecture)
            — the roadmap (the v0.5 milestones + Linear issues)
            — the agent advisories (the 4x-review punch list), each cross-linked
              to the capability and the job group it corresponds to

Usage::

    python dashboard/generate.py              # reuse the last junit.xml
    python dashboard/generate.py --run-tests  # re-run the suite first (~70s)
"""

from __future__ import annotations

import contextlib
import csv
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


def _is_empty(p: Path) -> bool:
    """A 0-byte file — nothing to view, so it is never surfaced as an artifact.

    Audit rule R1 (see dashboard/AUDIT.md). The *fact* that, e.g., a
    trajectory produced no events is carried on the task's ``trajectory_events``
    count — not as a free-standing "0 lines · empty file" row.
    """
    try:
        return p.stat().st_size == 0
    except Exception:
        return True


def _count_jsonl_events(p: Path) -> int:
    """Logical record count for a .jsonl — non-blank lines, not raw newlines.

    Audit rule R9: a user-facing "N events" count must reflect logical
    records (one JSON object per non-blank line), not the physical newline
    count, so a trailing blank line never inflates it.
    """
    try:
        with p.open() as fh:
            return sum(1 for line in fh if line.strip())
    except Exception:
        return 0


def _csv_rows(p: Path) -> int:
    """Logical CSV data-record count — quoted multi-line fields handled.

    Audit rule R9: ``content_lines - 1`` over-counts when a field contains an
    embedded newline inside quotes. ``csv.reader`` parses real records.
    """
    try:
        with p.open(newline="") as fh:
            n = sum(1 for _ in csv.reader(fh))
        return max(n - 1, 0)  # minus the header row
    except Exception:
        return 0


# File-content payloads — the dashboard shows the actual contents of every
# artifact, so each readable file is embedded (capped) into data.json.
_MAX_LINES = 240
_MAX_BYTES = 64000
_LANG = {
    ".json": "json", ".jsonl": "jsonl", ".ipynb": "json", ".csv": "csv",
    ".sh": "shell", ".py": "python", ".txt": "text", ".md": "text",
    ".toml": "text", ".log": "text", ".yaml": "text", ".yml": "text",
}


def _lang_for(p: Path) -> str:
    return _LANG.get(p.suffix.lower(), "text")


def _file_payload(p: Path) -> tuple[str | None, int, bool, str]:
    """Read a text file, capped. Returns (content, total_lines, truncated, lang).

    Binary and unreadable files return ``(None, 0, False, lang)``. The line
    count is the file's *real* length even when the embedded content is capped.
    """
    lang = _lang_for(p)
    try:
        raw = p.read_bytes()
    except Exception:
        return None, 0, False, lang
    if b"\x00" in raw[:4096]:  # binary
        return None, 0, False, lang
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    total = len(lines)
    truncated = False
    if total > _MAX_LINES:
        text = "\n".join(lines[:_MAX_LINES])
        truncated = True
    if len(text) > _MAX_BYTES:
        text = text[:_MAX_BYTES]
        truncated = True
    return text, total, truncated, lang


def _artifact(name: str, kind: str, path: Path) -> dict:
    """One artifact record — including the file's (capped) actual content."""
    content, lines, truncated, lang = _file_payload(path)
    if kind == "trajectory":
        info = f"{_count_jsonl_events(path)} events"
    elif kind == "reward":
        info = (content or "").strip().splitlines()[0][:16] if content else ""
    else:
        try:
            info = f"{path.stat().st_size:,} bytes"
        except Exception:
            info = ""
    return {
        "name": name, "kind": kind, "info": info, "lang": lang,
        "content": content, "content_lines": lines, "truncated": truncated,
    }


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
    """Every artifact file a rollout directory holds, with its contents.

    Top-level files plus the files inside the rollout's subdirs
    (``trajectory/`` ``verifier/`` ``agent/`` ``artifacts/``) — each record
    carries the file's embedded, capped content so the dashboard can show it
    verbatim.
    """
    arts: list[dict] = []
    for child in sorted(d.iterdir()):
        if child.is_file() and not _is_empty(child):  # R1: skip 0-byte files
            arts.append(_artifact(child.name,
                                  _ART_KIND.get(child.name, "file"), child))
    for sub in ("trajectory", "verifier", "agent", "artifacts"):
        subdir = d / sub
        if not subdir.is_dir():
            continue
        for f in sorted(p for p in subdir.rglob("*") if p.is_file()):
            if _is_empty(f):  # R1: skip 0-byte files
                continue
            rel = f.relative_to(d).as_posix()
            if f.name.endswith(".jsonl"):
                kind = "trajectory"
            elif f.name == "reward.txt":
                kind = "reward"
            else:
                kind = "file"
            arts.append(_artifact(rel, kind, f))
    return arts


def _task_row(d: Path) -> dict:
    result = _read_json(d / "result.json")
    config = _read_json(d / "config.json")
    traj = 0
    for cand in (d / "trajectory" / "acp_trajectory.jsonl",
                 d / "agent" / "acp_trajectory.jsonl"):
        if cand.is_file():
            traj = _count_jsonl_events(cand)
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
# Live source 3 — the experiments timeline (the experiments/ + labs/ folders)
# --------------------------------------------------------------------------
# (keyword, type, label, blurb) — the first rule whose keyword appears in a
# filename claims it. `reviewer` precedes `ablation` (reviewer_ablation.py).
EXP_RULES = [
    ("reviewer", "reviewer", "Reviewer ablation",
     "LLM-judge reviewer ablation."),
    ("ablation", "ablation", "Ablation studies",
     "Progressive-disclosure ablation runs and retries."),
    ("skillsbench", "skillsbench", "SkillsBench validation",
     "BYOS and skill-creator validation runs."),
    ("swebench", "swebench", "SWE-bench Pro",
     "SWE-bench Pro oracle-vs-baseline and progressive-disclosure results."),
    ("scene", "scene", "Scene-lifecycle validation",
     "Multi-scene lifecycle and TB2 scene validation."),
    ("tb2", "scene", "Scene-lifecycle validation",
     "Multi-scene lifecycle and TB2 scene validation."),
]


def _exp_file(p: Path, display: str | None = None) -> dict:
    """One experiment file — its kind, size, row count, and capped content."""
    content, lines, truncated, lang = _file_payload(p)
    suf = p.suffix.lower()
    kind = ("script" if suf in (".py", ".sh", ".ipynb")
            else "results" if suf in (".csv", ".json") else "data")
    rows = _csv_rows(p) if suf == ".csv" else None
    try:
        size = p.stat().st_size
    except Exception:
        size = 0
    return {
        "name": display or p.name, "kind": kind, "lang": lang, "size": size,
        "rows": rows, "content": content, "content_lines": lines,
        "truncated": truncated,
    }


def _ymd(mtimes: list[float]) -> str:
    return (datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d")
            if mtimes else "")


def collect_experiments() -> list[dict]:
    """The experiments timeline — scanned from experiments/ and labs/.

    Loose files in experiments/ are grouped into experiments by filename
    (see EXP_RULES); each subfolder of labs/ is its own experiment. Each
    experiment's files carry their embedded content, newest experiment first.
    """
    out: list[dict] = []

    exp = ROOT / "experiments"
    if exp.is_dir():
        buckets: dict[str, dict] = {}
        for f in sorted(exp.iterdir()):
            if not f.is_file() or f.name.startswith(".") or _is_empty(f):
                continue
            low = f.name.lower()
            match = next((r for r in EXP_RULES if r[0] in low), None)
            if match:
                _, typ, label, blurb = match
            else:
                typ, label, blurb = ("other", "Other experiments",
                                     "Assorted experiment scripts and data.")
            b = buckets.setdefault(
                label, {"type": typ, "blurb": blurb, "files": [], "mtimes": []})
            b["files"].append(_exp_file(f))
            with contextlib.suppress(Exception):
                b["mtimes"].append(f.stat().st_mtime)
        for label, b in buckets.items():
            files = sorted(b["files"], key=lambda x: x["name"])
            out.append({
                "name": label, "source": "experiments/", "type": b["type"],
                "blurb": b["blurb"], "date": _ymd(b["mtimes"]),
                "files": files,
                "files_truncated": False, "n_files_total": len(files),
            })

    labs = ROOT / "labs"
    if labs.is_dir():
        for sub in sorted(labs.iterdir()):
            if not sub.is_dir() or sub.name.startswith("."):
                continue
            all_files = [p for p in sorted(sub.rglob("*"))
                         if p.is_file() and not p.name.startswith(".")
                         and not _is_empty(p)]
            cap = 60
            files = all_files[:cap]
            mtimes = []
            for p in files:
                with contextlib.suppress(Exception):
                    mtimes.append(p.stat().st_mtime)
            out.append({
                "name": sub.name.replace("-", " ").replace("_", " ").title(),
                "source": f"labs/{sub.name}/", "type": "lab",
                "blurb": f"Lab experiment — {sub.name}.",
                "date": _ymd(mtimes),
                "files": [_exp_file(p, p.relative_to(sub).as_posix())
                          for p in files],
                "files_truncated": len(all_files) > cap,
                "n_files_total": len(all_files),
            })

    out.sort(key=lambda e: (e["date"], e["name"]), reverse=True)
    return out


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
# Authored — agent advisories (the 4x-review punch list), cross-linked
# --------------------------------------------------------------------------
ADVISORIES = {
    "source": "4x subagent review of the v0.5 capability fix pass (commits 35cdd47 → 0406a75)",
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
    experiments = collect_experiments()

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
            "experiments": len(experiments),
            "advisories_open": sum(1 for a in ADVISORIES["items"]
                                   if a["status"] == "open"),
        },
        "concept_map": CONCEPT_MAP,
        "tests": tests,
        "roadmap": ROADMAP,
        "jobs": jobs,
        "experiments": experiments,
        "advisories": ADVISORIES,
    }
    OUT.write_text(json.dumps(data, indent=2))
    s = data["summary"]["tests"]
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  tests: {s['passed']}p/{s['failed']}f/{s['skipped']}s   "
          f"jobs: {jobs['total_tasks']} tasks in {len(jobs['groups'])} groups   "
          f"experiments: {len(experiments)}   "
          f"capabilities: {done}/{len(CONCEPT_MAP['capabilities'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
