#!/usr/bin/env python3
"""Generate ``dashboard/data.json`` — the feed for the BenchFlow v0.5 dashboard.

Sections, two live and the rest authored / derived:

  live      — test results (parsed from ``junit.xml``)
            — the jobs tree (scanned from ``jobs/``: groups → runs → tasks →
              artifacts, mirroring the folder layout; every artifact carries
              its actual file content, capped)
            — the experiments timeline (scanned from ``experiments/`` + ``labs/``)
            — the roadmap (Linear project, or an explicit unavailable state)
  authored  — the concept map (the architecture)
            — the agent advisories (the 4x-review punch list), each cross-linked
              to the capability and the job group it corresponds to

Usage::

    LINEAR_API_KEY=... python dashboard/generate.py  # mirror roadmap from Linear
    python dashboard/generate.py --run-tests         # re-run tests, then mirror
    python dashboard/generate.py --allow-missing-linear  # local UI dev only
    python dashboard/generate.py --allow-stale-evidence  # local UI dev only

Production generation also refuses to publish when the junit test evidence is
missing or older than the release surface (the project version in
``pyproject.toml`` or the HEAD commit) — so the dashboard never ships a stale
or empty test-evidence summary as if it were release-fresh.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType

try:
    from dashboard.jobs_visibility import (
        RunTaskEvidence,
        RunVisibilityContext,
        decide_run_visibility,
    )
except ModuleNotFoundError:  # pragma: no cover - used when run as dashboard/generate.py
    from jobs_visibility import (  # type: ignore[no-redef]
        RunTaskEvidence,
        RunVisibilityContext,
        decide_run_visibility,
    )

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DASH = ROOT / "dashboard"
JUNIT = DASH / "junit.xml"
OUT = DASH / "data.json"
ARCHITECTURE_MD = DASH / "architecture.md"
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}$")
JOBS_ROOT_ENV = "BENCHFLOW_DASHBOARD_JOBS_ROOT"
_ROADMAP_MODULE: ModuleType | None = None
_SCORING_MODULE: ModuleType | None = None
_REWARD_EVENTS_MODULE: ModuleType | None = None
_PATHS_MODULE: ModuleType | None = None


def _load_local_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load dashboard helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_roadmap_module() -> ModuleType:
    """Load the sibling Linear roadmap helper, never a top-level ``roadmap``."""
    global _ROADMAP_MODULE
    if _ROADMAP_MODULE is None:
        _ROADMAP_MODULE = _load_local_module(
            "_benchflow_dashboard_roadmap", DASH / "roadmap.py"
        )
    return _ROADMAP_MODULE


def collect_roadmap() -> dict:
    return _load_roadmap_module().collect_roadmap()


def _slugify_heading(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _dashboard_source_path(path: Path) -> str:
    with contextlib.suppress(ValueError):
        return str(path.relative_to(ROOT))
    with contextlib.suppress(ValueError):
        return str(path.relative_to(DASH.parent))
    return str(path)


def _scrub_host_path(path: Path) -> str:
    """Return a portable, non-host-revealing rendering of an absolute path.

    The published ``data.json`` is served to anyone with dashboard access,
    so absolute host paths must not appear in it — they'd leak local
    usernames, worktree names, and tmp-dir layout. We render paths in
    order of preference:

    1. ``<relative>`` when ``path`` is under the repo root (``ROOT``)
    2. ``~/<relative>`` when ``path`` is under the operator's home dir
       (still round-trips through ``Path(...).expanduser()`` for callers
       like :func:`remembered_jobs_root`)
    3. ``<basename>`` for everything else — drops temp-dir layout, mount
       points, and other host-shape signals. Round-trip support outside
       HOME is not a goal: those paths are ephemeral and cannot survive
       a republish anyway.

    See issue #408.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        pass
    home = Path.home()
    try:
        rel = path.relative_to(home)
    except ValueError:
        # Outside ROOT and HOME — drop everything but the basename. For the
        # degenerate ``Path('/')`` case ``name`` is empty; treat that as the
        # root sentinel so we still don't echo the raw ``str(path)``.
        return path.name or "/"
    return f"~/{rel.as_posix()}" if rel.parts else "~"


def collect_architecture() -> dict:
    source = _dashboard_source_path(ARCHITECTURE_MD)
    if not ARCHITECTURE_MD.is_file():
        return {
            "available": False,
            "source": source,
            "title": "Architecture",
            "content": "",
            "headings": [],
            "error": "dashboard/architecture.md is missing",
        }

    content = ARCHITECTURE_MD.read_text()
    headings = []
    used: dict[str, int] = {}
    title = "Architecture"
    for line in content.splitlines():
        match = re.match(r"^(#{1,4})\s+(.+?)\s*$", line)
        if not match:
            continue
        level = len(match.group(1))
        text = match.group(2).strip()
        if level == 1 and title == "Architecture":
            title = text.lstrip("#").strip()
        base = _slugify_heading(text)
        count = used.get(base, 0)
        used[base] = count + 1
        anchor = base if count == 0 else f"{base}-{count + 1}"
        headings.append({"level": level, "title": text, "anchor": anchor})

    return {
        "available": True,
        "source": source,
        "title": title,
        "content": content,
        "lines": len(content.splitlines()),
        "headings": headings,
    }


def _load_scoring_module() -> ModuleType:
    """Load the pure scoring helper without importing ``benchflow`` package init."""
    global _SCORING_MODULE
    if _SCORING_MODULE is not None:
        return _SCORING_MODULE
    scoring_path = SRC / "benchflow" / "_utils" / "scoring.py"
    module = _load_local_module("_benchflow_dashboard_scoring", scoring_path)
    _SCORING_MODULE = module
    return module


def _classify_result_outcome(result: dict) -> str:
    return _load_scoring_module().classify_result_outcome(result)


def _load_reward_events_module() -> ModuleType:
    """Load the pure reward-event helper without importing ``benchflow``."""
    global _REWARD_EVENTS_MODULE
    if _REWARD_EVENTS_MODULE is not None:
        return _REWARD_EVENTS_MODULE
    reward_events_path = SRC / "benchflow" / "_utils" / "reward_events.py"
    module = _load_local_module(
        "_benchflow_dashboard_reward_events", reward_events_path
    )
    _REWARD_EVENTS_MODULE = module
    return module


def _memory_score_from_result(result: dict) -> float | None:
    return _load_reward_events_module().memory_score_from_result(result)


def _load_paths_module() -> ModuleType:
    """Load the symlink-safe traversal helpers without importing ``benchflow``."""
    global _PATHS_MODULE
    if _PATHS_MODULE is not None:
        return _PATHS_MODULE
    paths_path = SRC / "benchflow" / "_paths.py"
    _PATHS_MODULE = _load_local_module("_benchflow_dashboard_paths", paths_path)
    return _PATHS_MODULE


def _is_safe_regular_file(p: Path) -> bool:
    return bool(_load_paths_module().is_safe_regular_file(p))


def _iter_safe_children(directory: Path, *, context: str):
    yield from _load_paths_module().iter_safe_children(directory, context=context)


def _iter_safe_tree(root: Path, *, context: str):
    yield from _load_paths_module().iter_safe_tree(root, context=context)


def remembered_jobs_root(out: Path) -> Path | None:
    if not out.is_file():
        return None
    with contextlib.suppress(Exception):
        data = json.loads(out.read_text())
        raw = ((data.get("jobs") or {}).get("source") or {}).get("path")
        if not raw:
            return None
        remembered = Path(str(raw)).expanduser().resolve()
        if remembered.is_dir() and jobs_tree_has_rollouts(remembered):
            return remembered
    return None


def resolve_dashboard_jobs_root(root: Path, out: Path, raw: str | None = None) -> Path:
    raw = os.environ.get(JOBS_ROOT_ENV) if raw is None else raw
    if not raw:
        local_jobs = root / "jobs"
        if jobs_tree_has_rollouts(local_jobs):
            return local_jobs
        return remembered_jobs_root(out) or local_jobs
    candidate = Path(raw).expanduser()
    if candidate.name != "jobs" and (candidate / "jobs").is_dir():
        candidate = candidate / "jobs"
    return candidate.resolve()


def dashboard_jobs_root() -> Path:
    """Return the jobs tree the dashboard should mirror.

    Worktree-isolated agents often lack the git-ignored ``jobs/`` artifacts
    from the run-producing worktree. Operators can point the dashboard at
    either that worktree root or at its ``jobs/`` directory directly.
    """
    return resolve_dashboard_jobs_root(ROOT, OUT)


def _jobs_source(jobs: Path) -> dict:
    configured = bool(os.environ.get(JOBS_ROOT_ENV))
    local_jobs = (ROOT / "jobs").resolve()
    scrubbed = _scrub_host_path(jobs)
    return {
        # ``path`` is host-scrubbed (see _scrub_host_path / issue #408): repo
        # root → relative; HOME → ``~/…`` (round-trips through expanduser());
        # elsewhere → basename only. The published data.json never carries an
        # absolute host path here.
        "path": scrubbed,
        "label": str(jobs.relative_to(ROOT))
        if jobs.is_relative_to(ROOT)
        else scrubbed,
        "configured": configured,
        "remembered": not configured and jobs.resolve() != local_jobs,
        "available": jobs.is_dir(),
    }


def _latest_timestamp(items: list[str | None]) -> str | None:
    stamped = [item for item in items if item]
    return max(stamped) if stamped else None


# --------------------------------------------------------------------------
# Live source 1 — test results
# --------------------------------------------------------------------------
def run_suite() -> None:
    print("running the test suite (this takes ~70s) ...", flush=True)
    subprocess.run(
        [
            "uv",
            "run",
            "--extra",
            "dev",
            "python",
            "-m",
            "pytest",
            "tests/",
            "-q",
            "-p",
            "no:randomly",
            f"--junitxml={JUNIT}",
        ],
        cwd=ROOT,
        check=False,
    )


def collect_tests() -> dict:
    if not JUNIT.is_file():
        return {
            "available": False,
            "note": "no junit.xml yet — run: python dashboard/generate.py --run-tests",
            "summary": {"passed": 0, "failed": 0, "skipped": 0, "total": 0},
            "suites": [],
            "failures": [],
            "modified_at": None,
        }
    modified_at = datetime.fromtimestamp(JUNIT.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
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
            failures.append(
                {
                    "name": f"{c.get('classname', '')}::{c.get('name', '')}",
                    "message": (node.get("message") or "").strip()[:280],
                }
            )
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
        "summary": {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": passed + failed + skipped,
        },
        "suites": suite_list,
        "failures": failures,
        "modified_at": modified_at,
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
    ".json": "json",
    ".jsonl": "jsonl",
    ".ipynb": "json",
    ".csv": "csv",
    ".sh": "shell",
    ".py": "python",
    ".txt": "text",
    ".md": "text",
    ".toml": "text",
    ".log": "text",
    ".yaml": "text",
    ".yml": "text",
}


def _lang_for(p: Path) -> str:
    return _LANG.get(p.suffix.lower(), "text")


def _file_payload(p: Path) -> tuple[str | None, int, bool, str]:
    """Read a text file, capped. Returns (content, total_lines, truncated, lang).

    Binary and unreadable files return ``(None, 0, False, lang)``. Symlinks
    are refused outright (#390, #416) — an attacker-placed link inside an
    artifact or labs directory must never push host-readable content into
    ``data.json``. The line count is the file's *real* length even when the
    embedded content is capped.
    """
    lang = _lang_for(p)
    if not _is_safe_regular_file(p):
        return None, 0, False, lang
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
    """One artifact record — including the file's (capped) actual content.

    No ``info`` summary string is emitted: the file viewer renders the
    language, line count, and size itself, so an ``info`` field would be
    unread payload — and an unread field lets a correctness fix land where
    nobody can see it (audit rule R11, dashboard/AUDIT.md).
    """
    content, lines, truncated, lang = _file_payload(path)
    stat = path.stat()
    artifact = {
        "name": name,
        # ``path`` is host-scrubbed (see _scrub_host_path / issue #408): repo
        # root → relative; HOME → ``~/…``; elsewhere → basename only. The
        # published data.json never carries an absolute host path here.
        "path": _scrub_host_path(path),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "kind": kind,
        "lang": lang,
        "content": content,
        "content_lines": lines,
        "truncated": truncated,
    }
    if name == "rewards.jsonl":
        artifact["note"] = (
            "Reward event log derived from the canonical rollout reward; "
            "not a second score."
        )
    elif name in {"verifier/reward.txt", "verifier/reward.json"}:
        artifact["note"] = (
            "Raw verifier boundary output parsed into result.json.rewards."
        )
    elif name == "result.json":
        artifact["note"] = (
            "Canonical rollout result; result.json.rewards is the score "
            "used for summaries."
        )
    return artifact


# Each job group, the capability it exercises, and the advisory IDs it
# corresponds to — this is the jobs ↔ agent-advisor correspondence.
GROUP_META = {
    "main": {
        "label": "Unit-test rollouts",
        "blurb": "Rollouts the pytest suite writes while exercising the kernel.",
        "capability": None,
        "advisories": [],
    },
    "e2e": {
        "label": "Live e2e rollouts",
        "blurb": "Real agent runs on ClawsBench (Daytona + Docker) — capability 6.",
        "capability": 6,
        "advisories": ["MUST-3"],
    },
    "e2e-branch": {
        "label": "Branching rollout",
        "blurb": "The live Branch lifecycle — checkpoint → fork → V(root)=1.0 — capability 7.",
        "capability": 7,
        "advisories": [],
    },
    "environment": {
        "label": "Environment-plane probes",
        "blurb": "Manifest provisioning / readiness checks — capabilities 2 & 3.",
        "capability": 2,
        "advisories": ["MUST-2"],
    },
}

_ART_KIND = {
    "result.json": "result",
    "config.json": "config",
    "prompts.json": "prompts",
    "timing.json": "timing",
    "rewards.jsonl": "rewards",
}


def _task_artifacts(d: Path) -> list[dict]:
    """Every artifact file a rollout directory holds, with its contents.

    Top-level files plus the files inside the rollout's subdirs
    (``trajectory/`` ``verifier/`` ``agent/`` ``artifacts/``) — each record
    carries the file's embedded, capped content so the dashboard can show it
    verbatim.
    """
    arts: list[dict] = []
    for child in _iter_safe_children(d, context="rollout artifacts"):
        if _is_safe_regular_file(child) and not _is_empty(child):  # R1: skip 0-byte files
            arts.append(_artifact(child.name, _ART_KIND.get(child.name, "file"), child))
    for sub in ("trajectory", "verifier", "agent", "artifacts"):
        subdir = d / sub
        if not subdir.is_dir():
            continue
        # `iter_safe_tree` walks with followlinks=False and rejects symlinked
        # files; this is the #390 fix — agent-controlled artifact symlinks
        # must never embed host file content into ``data.json``.
        for f in _iter_safe_tree(subdir, context=f"rollout artifacts/{sub}"):
            if _is_empty(f):  # R1: skip 0-byte files
                continue
            rel = f.relative_to(d).as_posix()
            if f.name.endswith(".jsonl"):
                kind = "trajectory"
            elif f.name in {"reward.txt", "reward.json"}:
                kind = "reward"
            else:
                kind = "file"
            arts.append(_artifact(rel, kind, f))
    return arts


def _mark_ignored_reward_artifacts(artifacts: list[dict]) -> list[dict]:
    """Keep stale reward outputs visible, but mark them as ignored evidence."""
    reward_names = {
        "reward.txt",
        "reward.json",
        "verifier/reward.txt",
        "verifier/reward.json",
    }
    marked: list[dict] = []
    for artifact in artifacts:
        if artifact.get("name") in reward_names:
            artifact = {
                **artifact,
                "ignored_by_verifier": True,
                "note": "ignored by verifier because result.json reports a verifier error",
            }
        marked.append(artifact)
    return marked


def _task_row_from_parsed(d: Path, result: dict, config: dict) -> dict:
    traj = 0
    for cand in (
        d / "trajectory" / "acp_trajectory.jsonl",
        d / "agent" / "acp_trajectory.jsonl",
    ):
        if cand.is_file():
            traj = _count_jsonl_events(cand)
            break
    reward = None
    if isinstance(result.get("rewards"), dict):
        reward = result["rewards"].get("reward")
    memory_score = _memory_score_from_result(result)
    outcome = _classify_result_outcome(result)
    artifacts = _task_artifacts(d)
    if outcome == "verifier_errored":
        artifacts = _mark_ignored_reward_artifacts(artifacts)
    timing = result.get("timing") or {}
    latest_modified_at = _latest_timestamp(
        [artifact.get("modified_at") for artifact in artifacts]
    )
    return {
        "name": result.get("task_name") or d.name.split("__")[0],
        "rollout": d.name,
        "agent": result.get("agent") or config.get("agent") or "—",
        "model": result.get("model") or config.get("model") or "—",
        "environment": config.get("environment") or "—",
        "reward": reward,
        "memory_score": memory_score,
        "outcome": outcome,
        "error": (result.get("error") or "")[:240] or None,
        "verifier_error": (result.get("verifier_error") or "")[:240] or None,
        "trajectory_events": traj,
        "total_time": round(float(timing.get("total") or 0.0), 2),
        "latest_modified_at": latest_modified_at,
        "artifacts": artifacts,
    }


def _task_row(d: Path) -> dict:
    result = _read_json(d / "result.json")
    config = _read_json(d / "config.json")
    return _task_row_from_parsed(d, result, config)


def _task_visibility_evidence(
    d: Path, task: dict, result: dict, config: dict
) -> RunTaskEvidence:
    source = result.get("source") or config.get("source") or {}
    source_repo = None
    source_path = None
    if isinstance(source, dict):
        source_repo = source.get("repo")
        source_path = source.get("path")
    return RunTaskEvidence(
        path=d.as_posix(),
        name=str(task.get("name") or ""),
        rollout=str(task.get("rollout") or ""),
        agent=str(task.get("agent") or ""),
        model=str(task.get("model") or ""),
        environment=str(task.get("environment") or ""),
        outcome=task.get("outcome"),
        source_repo=str(source_repo) if source_repo else None,
        source_path=str(source_path) if source_path else None,
        reward_present=task.get("reward") is not None,
        memory_score_present=task.get("memory_score") is not None,
        trajectory_events=int(task.get("trajectory_events") or 0),
        artifact_count=len(task.get("artifacts") or []),
    )


def _task_record(d: Path) -> tuple[dict, RunTaskEvidence]:
    result = _read_json(d / "result.json")
    config = _read_json(d / "config.json")
    task = _task_row_from_parsed(d, result, config)
    return task, _task_visibility_evidence(d, task, result, config)


@dataclass(frozen=True)
class RunRecord:
    id: str
    run_dir: Path
    task_dirs: list[Path]
    summary_dir: Path | None = None


def _run_summary(run: RunRecord) -> dict:
    for summary_dir in (run.run_dir, run.summary_dir):
        if summary_dir is None:
            continue
        summary = _read_json(summary_dir / "summary.json")
        if summary:
            return summary
    return {}


def _run_summary_row(summary: dict) -> dict:
    keys = (
        "total",
        "passed",
        "failed",
        "errored",
        "verifier_errored",
        "score",
        "score_excl_errors",
        "concurrency",
        "agent",
        "model",
        "environment",
    )
    return {key: summary[key] for key in keys if key in summary}


def _is_task_dir(d: Path) -> bool:
    """A rollout directory — has a result/config file or a rollout subdir.

    Some rollouts (e.g. the ACP smoke runs) carry only ``agent/`` /
    ``verifier/`` / ``artifacts/`` subdirs and no top-level JSON, so the
    subdir markers matter too. A *run* directory (which only holds task
    dirs) has none of these directly, so it is not misread as a task.
    """
    if any(
        (d / m).is_file()
        for m in (
            "result.json",
            "config.json",
            "timing.json",
            "prompts.json",
            "rewards.jsonl",
        )
    ):
        return True
    return any(
        (d / s).is_dir() for s in ("trajectory", "verifier", "agent", "artifacts")
    )


def _task_dirs(run_dir: Path) -> list[Path]:
    return [c for c in sorted(run_dir.iterdir()) if c.is_dir() and _is_task_dir(c)]


_IGNORED_JOB_DIR_NAMES = {"configs", "_self_gen", "notes", "__pycache__"}


def _is_ignored_jobs_dir(d: Path) -> bool:
    return d.name in _IGNORED_JOB_DIR_NAMES or d.name.startswith(".")


def _job_child_dirs(root: Path) -> list[Path]:
    return [
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and not _is_ignored_jobs_dir(child)
    ]


def _nested_run_dirs(root: Path, group_dir: Path) -> list[RunRecord]:
    """Return explicit parent/timestamp and parent/mode/timestamp run dirs.

    The dashboard intentionally does not recurse arbitrary helper trees. Nested
    runs are used for bounded benchmark comparisons such as
    ``e2e/<experiment>/<timestamp>/<task rollout>`` and
    ``e2e/<experiment>/<mode>/<timestamp>/<task rollout>``.
    """
    runs: list[RunRecord] = []
    if (root / "summary.json").is_file():
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir() or _is_ignored_jobs_dir(run_dir):
                continue
            if not TS_RE.match(run_dir.name):
                continue
            task_dirs = _task_dirs(run_dir)
            if task_dirs:
                runs.append(
                    RunRecord(
                        id=run_dir.relative_to(group_dir).as_posix(),
                        run_dir=run_dir,
                        task_dirs=task_dirs,
                        summary_dir=root,
                    )
                )
    for mode_dir in sorted(root.iterdir()):
        if not mode_dir.is_dir() or _is_ignored_jobs_dir(mode_dir):
            continue
        if TS_RE.match(mode_dir.name):
            continue
        mode_summary_dir = mode_dir if (mode_dir / "summary.json").is_file() else None
        for run_dir in sorted(mode_dir.iterdir()):
            if not run_dir.is_dir() or _is_ignored_jobs_dir(run_dir):
                continue
            if not TS_RE.match(run_dir.name):
                continue
            task_dirs = _task_dirs(run_dir)
            if task_dirs:
                runs.append(
                    RunRecord(
                        id=run_dir.relative_to(group_dir).as_posix(),
                        run_dir=run_dir,
                        task_dirs=task_dirs,
                        summary_dir=mode_summary_dir,
                    )
                )
    return runs


def jobs_tree_runs(jobs: Path) -> dict[str, list[RunRecord]]:
    """Return the canonical dashboard groups/runs grammar for a jobs tree."""
    groups: dict[str, list[RunRecord]] = {}
    if not jobs.is_dir():
        return groups

    for top in _job_child_dirs(jobs):
        if TS_RE.match(top.name):
            groups.setdefault("main", []).append(
                RunRecord(id=top.name, run_dir=top, task_dirs=_task_dirs(top))
            )
            continue

        runs = groups.setdefault(top.name, [])
        direct: list[Path] = []
        for child in _job_child_dirs(top):
            if _is_task_dir(child):
                direct.append(child)
                continue
            task_dirs = _task_dirs(child)
            if task_dirs:
                runs.append(RunRecord(id=child.name, run_dir=child, task_dirs=task_dirs))
                continue
            nested_runs = _nested_run_dirs(child, top)
            if nested_runs:
                runs.extend(nested_runs)
        if direct:
            runs.append(
                RunRecord(id="(tasks)", run_dir=top, task_dirs=direct, summary_dir=top)
            )
    return groups


def jobs_tree_has_rollouts(jobs: Path) -> bool:
    """Return true when ``jobs`` contains a shape ``collect_jobs`` can collect."""
    with contextlib.suppress(Exception):
        return any(
            run.task_dirs for runs in jobs_tree_runs(jobs).values() for run in runs
        )
    return False


def collect_jobs() -> dict:
    """Scan jobs/ into groups → runs → tasks, mirroring the folder layout."""
    jobs = dashboard_jobs_root()
    source = _jobs_source(jobs)
    if not jobs.is_dir():
        return {
            "groups": [],
            "total_tasks": 0,
            "total_runs": 0,
            "archived_tasks": 0,
            "archived_runs": 0,
            "source": source,
        }

    groups = jobs_tree_runs(jobs)

    total = 0
    archived_tasks = 0
    archived_runs = 0
    total_runs = 0
    out_groups: list[dict] = []
    order = ["e2e-branch", "e2e", "environment", "main"]
    for name in sorted(groups, key=lambda n: (order.index(n) if n in order else 9, n)):
        meta = GROUP_META.get(
            name, {"label": name, "blurb": "", "capability": None, "advisories": []}
        )
        run_list: list[dict] = []
        for run in sorted(groups[name], key=lambda item: item.id, reverse=True):
            task_dirs = run.task_dirs
            records = [_task_record(d) for d in task_dirs]
            tasks = [task for task, _evidence in records]
            task_evidence = [evidence for _task, evidence in records]
            total_runs += 1
            summary = _run_summary(run)
            visibility = decide_run_visibility(
                RunVisibilityContext(
                    group_name=name,
                    run_id=run.id,
                    run_path=run.run_dir.as_posix(),
                    summary=summary,
                    tasks=task_evidence,
                )
            )
            if visibility.archived:
                archived_runs += 1
                archived_tasks += len(tasks)
                continue
            total += len(tasks)
            run_list.append(
                {
                    "id": run.id,
                    "latest_modified_at": _latest_timestamp(
                        [task.get("latest_modified_at") for task in tasks]
                    ),
                    "signals": list(visibility.signals),
                    "targets": list(visibility.targets),
                    "summary": _run_summary_row(summary),
                    "tasks": tasks,
                }
            )
        if not run_list:
            continue
        out_groups.append(
            {
                "name": name,
                "label": meta["label"],
                "blurb": meta["blurb"],
                "capability": meta["capability"],
                "advisories": meta["advisories"],
                "runs": run_list,
                "n_tasks": sum(len(r["tasks"]) for r in run_list),
                "latest_modified_at": _latest_timestamp(
                    [run.get("latest_modified_at") for run in run_list]
                ),
            }
        )
    source["latest_modified_at"] = _latest_timestamp(
        [group.get("latest_modified_at") for group in out_groups]
    )
    return {
        "groups": out_groups,
        "total_tasks": total,
        "total_runs": total_runs - archived_runs,
        "archived_tasks": archived_tasks,
        "archived_runs": archived_runs,
        "source": source,
    }


# --------------------------------------------------------------------------
# Live source 3 — the experiments timeline (the experiments/ + labs/ folders)
# --------------------------------------------------------------------------
# (keyword, type, label, blurb) — the first rule whose keyword appears in a
# filename claims it. `reviewer` precedes `ablation` (reviewer_ablation.py).
EXP_RULES = [
    ("reviewer", "reviewer", "Reviewer ablation", "LLM-judge reviewer ablation."),
    (
        "ablation",
        "ablation",
        "Ablation studies",
        "Progressive-disclosure ablation runs and retries.",
    ),
    (
        "skillsbench",
        "skillsbench",
        "SkillsBench validation",
        "BYOS and skill-creator validation runs.",
    ),
    (
        "swebench",
        "swebench",
        "SWE-bench Pro",
        "SWE-bench Pro oracle-vs-baseline and progressive-disclosure results.",
    ),
    (
        "scene",
        "scene",
        "Scene-lifecycle validation",
        "Multi-scene lifecycle and TB2 scene validation.",
    ),
    (
        "tb2",
        "scene",
        "Scene-lifecycle validation",
        "Multi-scene lifecycle and TB2 scene validation.",
    ),
]


def _exp_file(p: Path, display: str | None = None) -> dict:
    """One experiment file — its kind, size, row count, and capped content."""
    content, lines, truncated, lang = _file_payload(p)
    suf = p.suffix.lower()
    kind = (
        "script"
        if suf in (".py", ".sh", ".ipynb")
        else "results"
        if suf in (".csv", ".json")
        else "data"
    )
    rows = _csv_rows(p) if suf == ".csv" else None
    try:
        size = p.stat().st_size
    except Exception:
        size = 0
    return {
        "name": display or p.name,
        "kind": kind,
        "lang": lang,
        "size": size,
        "rows": rows,
        "content": content,
        "content_lines": lines,
        "truncated": truncated,
    }


def _ymd(mtimes: list[float]) -> str:
    return datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d") if mtimes else ""


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
        # Symlinks under experiments/ are refused (#416) so an attacker
        # cannot leak host files through the experiments timeline.
        for f in _iter_safe_children(exp, context="experiments timeline"):
            if not _is_safe_regular_file(f) or f.name.startswith(".") or _is_empty(f):
                continue
            low = f.name.lower()
            match = next((r for r in EXP_RULES if r[0] in low), None)
            if match:
                _, typ, label, blurb = match
            else:
                typ, label, blurb = (
                    "other",
                    "Other experiments",
                    "Assorted experiment scripts and data.",
                )
            b = buckets.setdefault(
                label, {"type": typ, "blurb": blurb, "files": [], "mtimes": []}
            )
            b["files"].append(_exp_file(f))
            with contextlib.suppress(Exception):
                b["mtimes"].append(f.stat().st_mtime)
        for label, b in buckets.items():
            files = sorted(b["files"], key=lambda x: x["name"])
            out.append(
                {
                    "name": label,
                    "source": "experiments/",
                    "type": b["type"],
                    "blurb": b["blurb"],
                    "date": _ymd(b["mtimes"]),
                    "files": files,
                    "files_truncated": False,
                    "n_files_total": len(files),
                }
            )

    labs = ROOT / "labs"
    if labs.is_dir():
        # Symlinks under labs/<sub> are refused (#416). `iter_safe_children`
        # rejects symlinked lab roots; `iter_safe_tree` walks without
        # following links inside each lab.
        for sub in _iter_safe_children(labs, context="labs timeline"):
            if not sub.is_dir() or sub.name.startswith("."):
                continue
            all_files = [
                p
                for p in _iter_safe_tree(sub, context=f"labs/{sub.name}")
                if not p.name.startswith(".") and not _is_empty(p)
            ]
            cap = 60
            files = all_files[:cap]
            mtimes = []
            for p in files:
                with contextlib.suppress(Exception):
                    mtimes.append(p.stat().st_mtime)
            out.append(
                {
                    "name": sub.name.replace("-", " ").replace("_", " ").title(),
                    "source": f"labs/{sub.name}/",
                    "type": "lab",
                    "blurb": f"Lab experiment — {sub.name}.",
                    "date": _ymd(mtimes),
                    "files": [
                        _exp_file(p, p.relative_to(sub).as_posix()) for p in files
                    ],
                    "files_truncated": len(all_files) > cap,
                    "n_files_total": len(all_files),
                }
            )

    out.sort(key=lambda e: (e["date"], e["name"]), reverse=True)
    return out


# --------------------------------------------------------------------------
# Authored — concept map
# --------------------------------------------------------------------------
CONCEPT_MAP = {
    "entry": "bench CLI · bf.run() · environment manifest",
    "kernel": {
        "name": "KERNEL",
        "blurb": "Rollout lifecycle · reward · trajectory. Imports only contracts/.",
    },
    "planes": [
        {
            "name": "Sandbox",
            "role": "where it runs",
            "detail": "Compute substrate — Local, Docker, Daytona, Modal. BYO via the Sandbox protocol.",
            "han": "—",
        },
        {
            "name": "Agent",
            "role": "who acts",
            "detail": "The agent under test / policy under training. Protocol: ACP. The Session is its real surface.",
            "han": "H — Harness",
        },
        {
            "name": "Environment",
            "role": "the world",
            "detail": "The stateful world. Declarative environment.toml manifest; owns provision→snapshot→restore→teardown.",
            "han": "S — State",
        },
        {
            "name": "Reward",
            "role": "how it's scored",
            "detail": "RewardFunc / Rubric / verifier. Scores any RolloutNode across five spaces.",
            "han": "V — Verifier",
        },
    ],
    "execution_model": [
        {
            "name": "Job",
            "kind": "set",
            "detail": "A set of Rollouts run together — an eval sweep, a GRPO group, a CL sequence.",
        },
        {
            "name": "Rollout",
            "kind": "PRIMITIVE",
            "detail": "One RL episode = a TREE of states. A linear rollout is a degree-1 tree.",
        },
        {
            "name": "Step",
            "kind": "PRIMITIVE",
            "detail": "One edge of the tree: (reason → act) → (tool-in → tool-out). Han's atomic unit.",
        },
        {
            "name": "Branch",
            "kind": "PRIMITIVE",
            "detail": "The snapshot-and-fork operation — a node with >1 child. The value-function engine.",
        },
        {
            "name": "Trajectory",
            "kind": "DERIVED VIEW",
            "detail": "One root-to-leaf path. Computed from the tree, never declared. The trainer export unit.",
        },
        {
            "name": "Scene",
            "kind": "AUTHORING SUGAR",
            "detail": "A declared role/skill span. Desugars completely to per-Step config; no runtime object.",
        },
    ],
    "capabilities": [
        {
            "n": 1,
            "name": "SkillsBench",
            "planes": ["Environment", "Reward"],
            "status": "shipped",
            "issue": None,
            "fit": "Environment-plane benchmark package; the Reward plane's Memory space scores skill use + updates.",
        },
        {
            "n": 2,
            "name": "ClawsBench",
            "planes": ["Environment"],
            "status": "shipped",
            "issue": "ENG-124",
            "fit": "The stateful-mock-service benchmark — base_image + [[services]], framework-started. The manifest's design partner.",
        },
        {
            "n": 3,
            "name": "chi-bench",
            "planes": ["Environment"],
            "status": "shipped",
            "issue": "ENG-124",
            "fit": "Same SMSB archetype, owns_lifecycle=true. External proof: onboarded by a ~25-line manifest.",
        },
        {
            "n": 4,
            "name": "NudgeBench",
            "planes": ["Agent", "Reward"],
            "status": "partial",
            "issue": "ENG-126",
            "fit": "ACP interaction model (nudges + ask_user) + tree-native Rollout; the Action space scores follow-up.",
        },
        {
            "n": 5,
            "name": "Continual learning",
            "planes": ["Reward"],
            "status": "shipped",
            "issue": "ENG-127",
            "fit": "A Job in sequential-shared mode over a persistent, versioned learner store; the Memory space tracks it.",
        },
        {
            "n": 6,
            "name": "RL-native",
            "planes": ["Reward"],
            "status": "shipped",
            "issue": "ENG-127",
            "fit": "The whole execution model — Rollout is a tree, Trajectory a path, exported as a trainer-ready record.",
        },
        {
            "n": 7,
            "name": "Branching · rollback",
            "planes": ["Environment", "Reward"],
            "status": "shipped",
            "issue": "ENG-127",
            "fit": "The RL-native substrate itself — first-class Branch, Environment snapshot/restore. Live e2e: V(root)=1.0.",
        },
        {
            "n": 8,
            "name": "Env adapters",
            "planes": ["Environment"],
            "status": "partial",
            "issue": "ENG-128",
            "fit": "Inbound adapters translate foreign formats — Harbor + Terminal-Bench shipped; ORS / PrimeIntellect pending.",
        },
    ],
    "spaces": ["output", "action", "reasoning", "memory", "latent"],
}


# --------------------------------------------------------------------------
# Authored — agent advisories (the 4x-review punch list), cross-linked
# --------------------------------------------------------------------------
ADVISORIES = {
    "source": "4x subagent review of the v0.5 capability fix pass, plus follow-up verification on codex/v05-integration-followup",
    "items": [
        {
            "id": "MUST-1",
            "severity": "must-fix",
            "status": "resolved",
            "agent": "Review consensus",
            "capability": 5,
            "group": "e2e",
            "title": "Continual learning — wire a real memory/skills producer",
            "detail": "The LearnerStore never received evolved skills and the Memory scorer read a memory_delta nothing wrote. New module learner_skills.py wires the rollout↔store data path end-to-end.",
        },
        {
            "id": "MUST-2",
            "severity": "must-fix",
            "status": "resolved",
            "agent": "Review consensus",
            "capability": 8,
            "group": "environment",
            "title": "Adapter file-map collision silently drops a file",
            "detail": "TerminalBenchAdapter._build_file_map now raises ValueError when two sources map to the same native destination.",
        },
        {
            "id": "MUST-3",
            "severity": "must-fix",
            "status": "resolved",
            "agent": "Review consensus",
            "capability": 6,
            "group": "e2e",
            "title": "Test deadlock on a concurrency regression",
            "detail": "test_parallel_independent_still_overlaps now wraps job.run() in asyncio.wait_for so a regression fails fast.",
        },
        {
            "id": "MUST-4",
            "severity": "must-fix",
            "status": "resolved",
            "agent": "Reviewer 1 (correctness)",
            "capability": 5,
            "group": "e2e",
            "title": "Memory scorer was a tautology",
            "detail": "The scorer's `expected` answer-key was derived from the agent's own diff → precision=recall=1.0 always. Dropped; the scorer now honestly grades activity.",
        },
        {
            "id": "SHOULD-1",
            "severity": "should-fix",
            "status": "resolved",
            "agent": "Reviewer 4 (quality)",
            "capability": 5,
            "group": "e2e",
            "title": "Store committed un-normalized skills",
            "detail": "_commit_learner_generation now commits the normalized after_skills so the store is byte-identical to the recorded delta.",
        },
        {
            "id": "SHOULD-2",
            "severity": "should-fix",
            "status": "resolved",
            "agent": "Reviewers 1 & 4",
            "capability": 5,
            "group": "e2e",
            "title": "learner_nodes leaked across run() calls",
            "detail": "Reset per-run; RolloutNode ids index-prefixed so same-named tasks stay distinct.",
        },
        {
            "id": "SHOULD-3",
            "severity": "should-fix",
            "status": "resolved",
            "agent": "Reviewer 4 (quality)",
            "capability": 5,
            "group": "e2e",
            "title": "Resumed continual-learning job silently restarts the curve",
            "detail": "run() now warns: the LearnerStore is process-local, so a resume restarts at generation 0.",
        },
        {
            "id": "SHOULD-4",
            "severity": "should-fix",
            "status": "resolved",
            "agent": "Reviewer 3 (tests)",
            "capability": 5,
            "group": "e2e",
            "title": "Stale _revert docstrings + loose evolved_skills type",
            "detail": "_revert docstrings corrected; evolved_skills tightened to dict[str, str].",
        },
        {
            "id": "OPEN-1",
            "severity": "should-fix",
            "status": "resolved",
            "agent": "Reviewer 1 (correctness)",
            "capability": 5,
            "group": "e2e",
            "title": "Real expected_skills fixture from the task definition",
            "detail": "ENG-125 follow-up wired: tasks can declare [verifier.memory].expected_skills, TaskConfig preserves it, and sequential-shared rollouts thread it into memory_delta. MemoryScorer can grade that fixture when invoked, without deriving an answer key from the agent diff.",
        },
        {
            "id": "OPEN-2",
            "severity": "note",
            "status": "resolved",
            "agent": "Reviewer 4 (quality)",
            "capability": None,
            "group": None,
            "title": "thermo-nuclear code-quality-review skill not installed",
            "detail": "Resolved: the thermo-nuclear code-quality-review skill is now installed and actively used by subagents in PRs #347-#356.",
        },
        {
            "id": "OPEN-3",
            "severity": "should-fix",
            "status": "resolved",
            "agent": "Follow-up review",
            "capability": 5,
            "group": "e2e",
            "title": "Memory-space scores surfaced in evaluation results",
            "detail": "MemoryScorer output is now a first-class additive metric: sequential-shared jobs persist sanitized memory_score/reward_events in result.json, aggregate memory_score and memory coverage in summary.json, expose it through metrics/eval CLI summaries, and render per-task Memory score in the dashboard.",
        },
    ],
}


def _roadmap_issues(roadmap: dict) -> list[dict]:
    return [i for m in roadmap.get("milestones", []) for i in m.get("issues", [])]


def _is_active_issue(issue: dict) -> bool:
    status_type = str(issue.get("status_type") or "").lower()
    status = str(issue.get("status") or "").lower()
    return status_type == "started" or "progress" in status or "review" in status


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).rstrip("\n")


def collect_repo_status() -> dict:
    """Capture the repo state that should cause dashboard data to refresh."""
    try:
        branch = _git(["branch", "--show-current"]) or _git(
            ["rev-parse", "--abbrev-ref", "HEAD"]
        )
        head = _git(["rev-parse", "--short", "HEAD"])
        status_lines = _git(["status", "--short"]).splitlines()
    except Exception as exc:
        return {"available": False, "error": str(exc), "dirty": None}

    staged = sum(1 for line in status_lines if line[:2] != "??" and line[:1].strip())
    unstaged = sum(
        1
        for line in status_lines
        if line[:2] != "??" and len(line) > 1 and line[1].strip()
    )
    untracked = sum(1 for line in status_lines if line.startswith("??"))
    return {
        "available": True,
        "branch": branch,
        "head": head,
        "dirty": bool(status_lines),
        "changes": len(status_lines),
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
    }


PYPROJECT_TOML = ROOT / "pyproject.toml"
_VERSION_RE = re.compile(r'^\s*version\s*=\s*"([^"]+)"', re.MULTILINE)


def _project_version() -> str | None:
    """Read ``version = "..."`` from the project ``pyproject.toml``."""
    try:
        text = PYPROJECT_TOML.read_text()
    except OSError:
        return None
    match = _VERSION_RE.search(text)
    return match.group(1) if match else None


def _git_iso_time(args: list[str]) -> str | None:
    """Return an ISO-8601 commit time from a git command, or None on failure."""
    try:
        return _git([*args, "--format=%cI"]) or None
    except Exception:
        return None


def _parse_iso(stamp: str | None) -> datetime | None:
    if not stamp:
        return None
    with contextlib.suppress(ValueError):
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return None


def _parse_local(stamp: str | None) -> datetime | None:
    if not stamp:
        return None
    with contextlib.suppress(ValueError):
        return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
    return None


def collect_release_evidence(tests: dict) -> dict:
    """Decide whether the bundled test evidence is fresh for the current release.

    The dashboard is published as a release-evidence surface; if the test
    evidence is missing or older than what the working copy claims to ship,
    refuse to publish (unless the operator explicitly opts into stale data).

    Freshness is checked against two signals on the release surface:

    * ``pyproject.toml`` — the file that carries the project ``version``.
      A version bump without re-running the suite makes the bundled evidence
      stale by definition.
    * The HEAD commit time — any code change since the last suite run makes
      the evidence stale relative to the code it is supposed to attest to.

    Returns a record with the freshness verdict, the stale reasons, and the
    timestamps used so the dashboard UI (and tests) can surface them.
    """
    junit_local = _parse_local(tests.get("modified_at"))
    junit_available = bool(tests.get("available"))

    version = _project_version()
    pyproject_at: datetime | None = None
    if PYPROJECT_TOML.is_file():
        with contextlib.suppress(OSError):
            pyproject_at = datetime.fromtimestamp(PYPROJECT_TOML.stat().st_mtime)
    head_at = _parse_iso(_git_iso_time(["log", "-1", "HEAD"]))

    reasons: list[str] = []
    if not junit_available or junit_local is None:
        reasons.append("junit.xml missing — no test evidence has been recorded")
    else:
        if pyproject_at is not None and junit_local < pyproject_at.replace(
            microsecond=0
        ):
            reasons.append(
                f"junit.xml ({tests.get('modified_at')}) is older than "
                f"pyproject.toml version={version!r}"
            )
        if head_at is not None:
            head_local = head_at.astimezone().replace(tzinfo=None, microsecond=0)
            if junit_local < head_local:
                reasons.append(
                    f"junit.xml ({tests.get('modified_at')}) is older than "
                    f"HEAD commit ({head_local.strftime('%Y-%m-%d %H:%M:%S')})"
                )

    def _fmt(dt: datetime | None) -> str | None:
        return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None

    return {
        "version": version,
        "junit_modified_at": tests.get("modified_at"),
        "pyproject_modified_at": _fmt(pyproject_at),
        "head_committed_at": _fmt(
            head_at.astimezone().replace(tzinfo=None) if head_at else None
        ),
        "stale_reasons": reasons,
        "fresh": not reasons,
    }


def build_data() -> dict:
    tests = collect_tests()
    jobs = collect_jobs()
    experiments = collect_experiments()
    roadmap = collect_roadmap()
    architecture = collect_architecture()
    repo = collect_repo_status()
    release_evidence = collect_release_evidence(tests)

    done = sum(1 for c in CONCEPT_MAP["capabilities"] if c["status"] == "shipped")
    all_issues = _roadmap_issues(roadmap)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "tests": tests["summary"],
            "capabilities_shipped": done,
            "capabilities_total": len(CONCEPT_MAP["capabilities"]),
            "issues_total": len(all_issues),
            "issues_active": sum(1 for i in all_issues if _is_active_issue(i)),
            "jobs_total": jobs["total_tasks"],
            "jobs_archived": jobs.get("archived_tasks", 0),
            "jobs_archived_runs": jobs.get("archived_runs", 0),
            "job_groups": len(jobs["groups"]),
            "experiments": len(experiments),
            "advisories_open": sum(
                1 for a in ADVISORIES["items"] if a["status"] == "open"
            ),
            "release_evidence_fresh": release_evidence["fresh"],
        },
        "concept_map": CONCEPT_MAP,
        "architecture": architecture,
        "tests": tests,
        "roadmap": roadmap,
        "repo": repo,
        "jobs": jobs,
        "experiments": experiments,
        "advisories": ADVISORIES,
        "release_evidence": release_evidence,
    }


# --------------------------------------------------------------------------
def main() -> int:
    if "--run-tests" in sys.argv:
        run_suite()
    data = build_data()
    if (
        data["roadmap"].get("source", {}).get("kind") != "linear-live"
        and "--allow-missing-linear" not in sys.argv
    ):
        error = (
            data["roadmap"].get("source", {}).get("error", "Linear roadmap unavailable")
        )
        print(f"error: Roadmap must mirror live Linear: {error}", file=sys.stderr)
        print(
            "hint: set LINEAR_API_KEY or pass --allow-missing-linear for local UI dev",
            file=sys.stderr,
        )
        return 1
    evidence = data["release_evidence"]
    if not evidence["fresh"] and "--allow-stale-evidence" not in sys.argv:
        print(
            "error: dashboard refuses to publish stale release evidence:",
            file=sys.stderr,
        )
        for reason in evidence["stale_reasons"]:
            print(f"  - {reason}", file=sys.stderr)
        print(
            "hint: run `python dashboard/generate.py --run-tests` to refresh,",
            file=sys.stderr,
        )
        print(
            "      or pass --allow-stale-evidence for local UI dev only",
            file=sys.stderr,
        )
        return 1
    OUT.write_text(json.dumps(data, indent=2))
    s = data["summary"]["tests"]
    jobs = data["jobs"]
    experiments = data["experiments"]
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(
        f"  tests: {s['passed']}p/{s['failed']}f/{s['skipped']}s   "
        f"jobs: {jobs['total_tasks']} rollout rows in {len(jobs['groups'])} groups   "
        f"archived: {jobs.get('archived_runs', 0)} runs/"
        f"{jobs.get('archived_tasks', 0)} rollout rows   "
        f"experiments: {len(experiments)}   "
        f"capabilities: {data['summary']['capabilities_shipped']}/"
        f"{data['summary']['capabilities_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
