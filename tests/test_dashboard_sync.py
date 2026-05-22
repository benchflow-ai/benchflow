"""Tests for dashboard repo-sync and ticket-browser behavior."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from dashboard import generate, serve


def test_collect_repo_status_counts_dirty_state(monkeypatch):
    """Guards the dashboard repo sync added after v0.5-integration@ffef85d."""
    outputs = {
        ("branch", "--show-current"): "codex/tickets",
        ("rev-parse", "--short", "HEAD"): "abc1234",
        ("status", "--short"): " M dashboard/index.html\nA  staged.py\n?? new.py\n",
    }

    monkeypatch.setattr(generate, "_git", lambda args: outputs[tuple(args)])

    status = generate.collect_repo_status()
    assert status == {
        "available": True,
        "branch": "codex/tickets",
        "head": "abc1234",
        "dirty": True,
        "changes": 3,
        "staged": 1,
        "unstaged": 1,
        "untracked": 1,
    }


def test_repo_fingerprint_changes_when_dirty_file_content_changes(
    tmp_path: Path, monkeypatch
):
    """Guards the dashboard repo sync added after v0.5-integration@ffef85d."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("one\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    monkeypatch.setattr(serve, "ROOT", tmp_path)
    clean = serve.repo_fingerprint()
    tracked.write_text("two\n")
    dirty_once = serve.repo_fingerprint()
    tracked.write_text("three\n")
    dirty_twice = serve.repo_fingerprint()
    untracked = tmp_path / "new.txt"
    untracked.write_text("alpha\n")
    untracked_once = serve.repo_fingerprint()
    untracked.write_text("beta\n")
    untracked_twice = serve.repo_fingerprint()

    assert clean != dirty_once
    assert dirty_once != dirty_twice
    assert dirty_twice != untracked_once
    assert untracked_once != untracked_twice


def test_syncing_handler_returns_503_on_failed_refresh(monkeypatch):
    """Guards the dashboard repo sync added after v0.5-integration@ffef85d."""
    handler = object.__new__(serve.SyncingDashboardHandler)
    sent: list[tuple[int, str | None]] = []
    handler.send_error = lambda code, message=None: sent.append((code, message))  # type: ignore[method-assign]

    class FailedResult:
        returncode = 1

    serve.SyncingDashboardHandler.gen_cmd = ["generate"]
    serve.SyncingDashboardHandler.last_repo_fingerprint = "old"
    serve.SyncingDashboardHandler.last_refresh_at = time.monotonic()
    serve.SyncingDashboardHandler.refresh_interval = 3600
    serve.SyncingDashboardHandler.last_failed_refresh_at = None
    serve.SyncingDashboardHandler.failed_refresh_error = None
    serve.SyncingDashboardHandler.sync_lock = threading.Lock()

    fingerprints = []
    runs = []
    monkeypatch.setattr(
        serve,
        "repo_fingerprint",
        lambda: fingerprints.append("new") or "new",
    )
    monkeypatch.setattr(
        serve.subprocess,
        "run",
        lambda _cmd, check=False: runs.append(_cmd) or FailedResult(),
    )

    assert handler._sync_data_json() is False
    assert handler._sync_data_json() is False
    assert sent == [
        (503, "data.json refresh failed; refusing to serve stale dashboard data"),
        (503, "data.json refresh failed; retrying after cooldown"),
    ]
    assert len(runs) == 1
    assert len(fingerprints) == 2


def test_index_ticket_view_and_sync_contract():
    """Guards the dashboard repo sync added after v0.5-integration@ffef85d."""
    html = Path("dashboard/index.html").read_text()
    for expected in (
        ".ticket-board",
        ".ticket-list-pane",
        ".ticket-detail-pane",
        ".progress-meter",
        ".ticket-progress",
        ".ticket-comments",
        ".architecture-board",
        ".architecture-doc",
        ".fb-resizer",
        ".vdate",
        ".vleft",
        ".sync-banner",
        "Dashboard data is not refreshing.",
        "safeLinearUrl",
        "linearLink",
        'target="_blank"',
        "aria-pressed",
        "SELECTED_TICKET_ID",
        "Architecture",
        "renderArchitecture",
        "compactPath",
        "Resize file tree",
        "file.modified_at",
        "file.path || node.path || file.name",
        "SYNC_ERROR",
        'nav.innerHTML = ""',
        "setInterval(() => loadData(false), 5000)",
        "d.jobs && d.jobs.source && d.jobs.source.path",
        "d.jobs && d.jobs.source && d.jobs.source.configured",
        "d.jobs && d.jobs.source && d.jobs.source.available",
        "d.jobs && d.jobs.source && d.jobs.source.remembered",
        "d.jobs && d.jobs.source && d.jobs.source.latest_modified_at",
        "Latest artifact:",
        "archived_runs",
        "low-signal run",
        "jobs latest",
        "artifactDisplayName",
        "reward events.jsonl",
        "raw verifier reward.txt",
        "final result.json",
        "sandbox boundary",
        "canonical",
        "Priority",
        "Milestone",
        "Assignee",
        "Team",
        "Creator",
        "Estimate",
        "Due",
        "Created",
        "Started",
        "Updated",
        "Completed",
        "Canceled",
        "Parent",
        "Description",
        "Comments",
        "Linear",
    ):
        assert expected in html


def test_ticket_board_is_bounded_to_short_desktop_viewports():
    """Guards the Tickets pane from growing taller than the visible screen."""
    html = Path("dashboard/index.html").read_text()
    match = re.search(r"\.ticket-board\s*\{(?P<body>[^}]*)\}", html)
    assert match is not None

    body = match.group("body")
    assert "height: calc(100dvh - 206px)" in body
    assert "min-height" not in body


def test_jobs_browser_tree_uses_artifact_workspace_width():
    """Guards Jobs against a cramped file-tree column."""
    html = Path("dashboard/index.html").read_text()
    match = re.search(r"\.jobs-browser\.fb\s*\{(?P<body>[^}]*)\}", html)
    assert match is not None

    body = match.group("body")
    assert "grid-template-columns" in body
    assert "--tree-width: clamp(260px, 28%, 380px)" in body
    assert "var(--tree-width) 6px minmax(0, 1fr)" in body
    assert 'fileBrowser(roots, { treeLabel: "jobs/", resizable: true })' in html
    assert 'setProperty("--tree-width"' in html


def test_collect_jobs_reports_latest_artifact_timestamp(tmp_path: Path, monkeypatch):
    """Guards Jobs against confusing feed freshness with artifact freshness."""
    jobs = tmp_path / "jobs"
    task_dir = jobs / "custom" / "2026-05-22__00-00-00" / "task-a"
    task_dir.mkdir(parents=True)
    result = task_dir / "result.json"
    result.write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "gemini",
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
            }
        )
    )
    config = task_dir / "config.json"
    config.write_text(json.dumps({"agent": "gemini"}))
    newer = datetime(2026, 5, 21, 17, 28, 3).timestamp()
    older = datetime(2026, 5, 21, 17, 27, 33).timestamp()
    os.utime(config, (older, older))
    os.utime(result, (newer, newer))

    monkeypatch.setenv(generate.JOBS_ROOT_ENV, str(jobs))

    data = generate.collect_jobs()

    assert data["source"]["latest_modified_at"] == "2026-05-21 17:28:03"
    group = data["groups"][0]
    assert group["latest_modified_at"] == "2026-05-21 17:28:03"
    run = group["runs"][0]
    assert run["latest_modified_at"] == "2026-05-21 17:28:03"
    assert run["tasks"][0]["latest_modified_at"] == "2026-05-21 17:28:03"


def test_collect_jobs_archives_generic_timestamp_rollouts(tmp_path: Path, monkeypatch):
    """Guards codex/v05-integration-followup@5f05423 dashboard cleanup."""
    jobs = tmp_path / "jobs"
    task_dir = jobs / "2026-05-18__11-55-00" / "task__fa1e623d"
    task_dir.mkdir(parents=True)
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "agent": "oracle",
                "rewards": {"reward": 1.0},
                "timing": {},
            }
        )
    )
    (task_dir / "config.json").write_text(json.dumps({"agent": "oracle"}))

    monkeypatch.setenv(generate.JOBS_ROOT_ENV, str(jobs))

    data = generate.collect_jobs()

    assert data["groups"] == []
    assert data["total_tasks"] == 0
    assert data["archived_runs"] == 1
    assert data["archived_tasks"] == 1


def test_collect_jobs_archives_acp_smoke_without_target_evidence(
    tmp_path: Path, monkeypatch
):
    """Guards codex/v05-integration-followup@5f05423 ACP smoke cleanup."""
    jobs = tmp_path / "jobs"
    task_dir = jobs / "environment" / "acp_smoke__fa1e623d"
    trajectory = task_dir / "agent"
    trajectory.mkdir(parents=True)
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "acp_smoke",
                "agent": "acp",
                "rewards": {"reward": 1.0},
                "timing": {},
            }
        )
    )
    (task_dir / "config.json").write_text(json.dumps({"agent": "acp"}))
    (trajectory / "acp_trajectory.jsonl").write_text(json.dumps({"event": "noop"}))

    monkeypatch.setenv(generate.JOBS_ROOT_ENV, str(jobs))

    data = generate.collect_jobs()

    assert data["groups"] == []
    assert data["total_tasks"] == 0
    assert data["archived_runs"] == 1
    assert data["archived_tasks"] == 1


def test_collect_jobs_archives_empty_task_dirs_and_target_named_placeholders(
    tmp_path: Path, monkeypatch
):
    """Guards codex/v05-integration-followup@5f05423 placeholder cleanup."""
    jobs = tmp_path / "jobs"
    (jobs / "2026-05-22__03-45-00" / "task__empty").mkdir(parents=True)
    (jobs / "smoke-test" / "programbench-daytona").mkdir(parents=True)
    (jobs / "programbench-daytona" / "programbench-daytona__abc123" / "agent").mkdir(
        parents=True
    )

    monkeypatch.setenv(generate.JOBS_ROOT_ENV, str(jobs))

    data = generate.collect_jobs()

    assert data["groups"] == []
    assert data["total_tasks"] == 0
    assert data["archived_runs"] == 3
    assert data["archived_tasks"] == 1


def test_collect_jobs_archives_generic_task_in_nonstandard_rollout(
    tmp_path: Path, monkeypatch
):
    """Guards codex/v05-integration-followup@5f05423 generic name cleanup."""
    jobs = tmp_path / "jobs"
    task_dir = jobs / "smoke-leftover" / "2026-05-22__03-55-00" / "leftover__abc123"
    task_dir.mkdir(parents=True)
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "agent": "oracle",
                "rewards": {"reward": 1.0},
                "timing": {},
            }
        )
    )
    (task_dir / "config.json").write_text(json.dumps({"agent": "oracle"}))

    monkeypatch.setenv(generate.JOBS_ROOT_ENV, str(jobs))

    data = generate.collect_jobs()

    assert data["groups"] == []
    assert data["total_tasks"] == 0
    assert data["archived_runs"] == 1
    assert data["archived_tasks"] == 1


def test_collect_jobs_keeps_target_runs_even_when_failed(tmp_path: Path, monkeypatch):
    """Guards codex/v05-integration-followup@5f05423 target evidence."""
    jobs = tmp_path / "jobs"
    task_dir = (
        jobs
        / "codex-daytona-programbench-provenance-20260522-020214"
        / "2026-05-22__02-02-14"
        / "abishekvashok-inference-provenance__7cf73b55"
    )
    task_dir.mkdir(parents=True)
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "abishekvashok-inference-provenance",
                "agent": "oracle",
                "rewards": None,
                "error": "agent stopped early",
                "timing": {},
                "source": {
                    "repo": "benchflow-ai/programbench",
                    "path": "datasets/programbench/abishekvashok-inference",
                },
            }
        )
    )
    (task_dir / "config.json").write_text(
        json.dumps({"agent": "oracle", "environment": "daytona"})
    )

    monkeypatch.setenv(generate.JOBS_ROOT_ENV, str(jobs))

    data = generate.collect_jobs()

    assert data["archived_runs"] == 0
    assert data["total_tasks"] == 1
    run = data["groups"][0]["runs"][0]
    assert "ProgramBench" in run["targets"]
    assert "hosted-env" in run["signals"]
    assert "provenance" in run["signals"]


def test_collect_jobs_archives_label_cleanup_without_lab_false_positive(
    tmp_path: Path, monkeypatch
):
    """Guards codex/v05-integration-followup@5f05423 lab token cleanup."""
    jobs = tmp_path / "jobs"
    task_dir = jobs / "label-cleanup" / "2026-05-22__04-00-00" / "task__abc123"
    task_dir.mkdir(parents=True)
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "agent": "oracle",
                "rewards": {"reward": 1.0},
                "timing": {},
            }
        )
    )
    (task_dir / "config.json").write_text(json.dumps({"agent": "oracle"}))

    monkeypatch.setenv(generate.JOBS_ROOT_ENV, str(jobs))

    data = generate.collect_jobs()

    assert data["groups"] == []
    assert data["total_tasks"] == 0
    assert data["archived_runs"] == 1
    assert data["archived_tasks"] == 1


def test_collect_jobs_archives_signal_words_without_structured_evidence(
    tmp_path: Path, monkeypatch
):
    """Guards codex/v05-integration-followup@5f05423 structured signals."""
    jobs = tmp_path / "jobs"
    for group_name in (
        "daytona-leftover",
        "modal-leftover",
        "provenance-leftover",
        "concurrency-leftover",
    ):
        task_dir = jobs / group_name / "2026-05-22__04-05-00" / "task__abc123"
        task_dir.mkdir(parents=True)
        (task_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "task",
                    "agent": "oracle",
                    "rewards": {"reward": 1.0},
                    "timing": {},
                }
            )
        )
        (task_dir / "config.json").write_text(json.dumps({"agent": "oracle"}))

    monkeypatch.setenv(generate.JOBS_ROOT_ENV, str(jobs))

    data = generate.collect_jobs()

    assert data["groups"] == []
    assert data["total_tasks"] == 0
    assert data["archived_runs"] == 4
    assert data["archived_tasks"] == 4


def test_collect_jobs_archives_hello_world_without_rollout_signal(
    tmp_path: Path, monkeypatch
):
    """Guards codex/v05-integration-followup@5f05423 hello-world cleanup."""
    jobs = tmp_path / "jobs"
    run_dir = jobs / "2026-05-22__04-15-00"
    hello_world = run_dir / "hello-world__abc123"
    hello_world_task = run_dir / "hello-world-task__abc123"
    hello_world.mkdir(parents=True)
    hello_world_task.mkdir()
    (hello_world / "result.json").write_text(
        json.dumps(
            {
                "task_name": "hello-world",
                "agent": "oracle",
                "rewards": {"reward": 1.0},
                "timing": {},
            }
        )
    )
    (hello_world / "config.json").write_text(json.dumps({"agent": "oracle"}))
    (hello_world_task / "result.json").write_text(
        json.dumps(
            {
                "task_name": "hello-world-task",
                "agent": "oracle",
                "rewards": {"reward": 1.0},
                "timing": {},
            }
        )
    )
    (hello_world_task / "config.json").write_text(json.dumps({"agent": "oracle"}))

    monkeypatch.setenv(generate.JOBS_ROOT_ENV, str(jobs))

    data = generate.collect_jobs()

    assert data["groups"] == []
    assert data["total_tasks"] == 0
    assert data["archived_runs"] == 1
    assert data["archived_tasks"] == 2


def test_collect_jobs_keeps_generic_task_inside_target_signal_rollout(
    tmp_path: Path, monkeypatch
):
    """Guards codex/v05-integration-followup@5f05423 generic target exception."""
    jobs = tmp_path / "jobs"
    task_dir = (
        jobs
        / "programbench-daytona-provenance"
        / "2026-05-22__04-30-00"
        / "hello-world-task__abc123"
    )
    task_dir.mkdir(parents=True)
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "hello-world-task",
                "agent": "oracle",
                "rewards": {"reward": 1.0},
                "timing": {},
            }
        )
    )
    (task_dir / "config.json").write_text(
        json.dumps({"agent": "oracle", "environment": "daytona"})
    )

    monkeypatch.setenv(generate.JOBS_ROOT_ENV, str(jobs))

    data = generate.collect_jobs()

    assert data["archived_runs"] == 0
    assert data["total_tasks"] == 1
    run = data["groups"][0]["runs"][0]
    assert "ProgramBench" in run["targets"]
    assert "hosted-env" in run["signals"]


def test_collect_architecture_hosts_markdown_file():
    """Guards the hosted Architecture dashboard tab."""
    doc = generate.collect_architecture()

    assert doc["available"] is True
    assert doc["source"] == "dashboard/architecture.md"
    assert doc["title"] == "BenchFlow — Architecture"
    assert doc["lines"] > 50
    assert "environment-and-rollout engine" in doc["content"]
    assert any(h["title"] == "What BenchFlow is" for h in doc["headings"])


def test_index_initializes_hash_tab_after_late_data_success():
    """Guards #Jobs after an initial data.json failure followed by a good poll."""
    html = Path("dashboard/index.html").read_text()

    assert "function tabFromLocation()" in html
    assert "if (initial || !CURRENT)" in html
    assert "show(tabFromLocation())" in html
    assert 'window.addEventListener("hashchange"' in html


def test_index_poll_success_renders_jobs_after_initial_data_json_failure():
    """Guards the actual DOM recovery path for #Jobs after an initial 503."""
    if not shutil.which("node"):
        pytest.skip("node is required for the dashboard DOM recovery contract")

    code = r"""
import fs from "node:fs";
import vm from "node:vm";
const html = fs.readFileSync("dashboard/index.html", "utf8");
let script = html.match(/<script>([\s\S]*)<\/script>/)[1];
script = script.replace(/loadData\(true\);\s*setInterval\(\(\) => loadData\(false\), 5000\);/, "");

function stripHtml(value) {
  return String(value || "").replace(/<[^>]*>/g, "");
}

class ClassList {
  constructor(el) { this.el = el; }
  _parts() { return this.el.className ? this.el.className.split(/\s+/).filter(Boolean) : []; }
  contains(name) { return this._parts().includes(name); }
  add(name) {
    const parts = new Set(this._parts());
    parts.add(name);
    this.el.className = Array.from(parts).join(" ");
  }
  remove(name) {
    this.el.className = this._parts().filter(part => part !== name).join(" ");
  }
  toggle(name, force) {
    const want = force === undefined ? !this.contains(name) : !!force;
    if (want) this.add(name); else this.remove(name);
    return want;
  }
}

class Element {
  constructor(tag) {
    this.tagName = tag.toLowerCase();
    this.children = [];
    this.parentNode = null;
    this.dataset = {};
    this.style = {};
    this.eventListeners = {};
    this.className = "";
    this.classList = new ClassList(this);
    this._innerHTML = "";
    this._textContent = "";
  }
  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  set innerHTML(value) {
    this._innerHTML = String(value);
    this._textContent = "";
    this.children = [];
  }
  get innerHTML() {
    return this._innerHTML + this.children.map(child => child.innerHTML || child.textContent).join("");
  }
  set textContent(value) {
    this._textContent = String(value);
    this._innerHTML = "";
    this.children = [];
  }
  get textContent() {
    return this._textContent + stripHtml(this._innerHTML) + this.children.map(child => child.textContent).join("");
  }
  get innerText() { return this.textContent; }
  addEventListener(type, fn) {
    (this.eventListeners[type] ||= []).push(fn);
  }
  setAttribute(name, value) {
    this[name] = String(value);
  }
  dispatchEvent(event) {
    (this.eventListeners[event.type] || []).forEach(fn => fn.call(this, event));
  }
  _descendants() {
    return this.children.flatMap(child => [child, ...child._descendants()]);
  }
  _matches(selector) {
    if (selector.startsWith(".")) {
      const classes = selector.slice(1).split(".");
      return classes.every(cls => this.classList.contains(cls));
    }
    return this.tagName === selector.toLowerCase();
  }
  _hasAncestor(selector) {
    let cur = this.parentNode;
    while (cur) {
      if (cur._matches(selector)) return true;
      cur = cur.parentNode;
    }
    return false;
  }
  querySelectorAll(selector) {
    if (selector === ".tnode.file .trow") {
      return this._descendants().filter(node => node._matches(".trow") && node._hasAncestor(".tnode.file"));
    }
    return this._descendants().filter(node => node._matches(selector));
  }
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
}

const main = new Element("main");
const nav = new Element("nav");
const gen = new Element("div");
const document = {
  createElement: tag => new Element(tag),
  getElementById: id => ({ main, nav, gen })[id],
  querySelectorAll: selector => selector === "nav a" ? nav._descendants().filter(node => node.tagName === "a") : [],
};

const data = {
  generated_at: "2026-05-22 00:00:00",
  summary: {
    tests: { passed: 1, failed: 0, skipped: 0, total: 1 },
    jobs_total: 1,
  },
  repo: { available: true, branch: "codex/test", head: "abc1234", dirty: false, changes: 0 },
  jobs: {
    total_tasks: 1,
    source: { path: "/tmp/producer/jobs", label: "/tmp/producer/jobs", configured: true, available: true },
    groups: [{
      name: "main", label: "Rollouts", blurb: "Real rollout", capability: 1, advisories: [],
      n_tasks: 1,
      runs: [{ id: "2026-05-21__23-57-23", tasks: [{
        name: "task", rollout: "task__abc123", agent: "codex", model: "test",
        environment: "docker", reward: 1, memory_score: 0.5, outcome: "passed", artifacts: [],
      }] }],
    }],
  },
  roadmap: { source: {} },
  tests: { summary: { passed: 1, failed: 0, skipped: 0, total: 1 }, suites: [], failures: [] },
  advisories: { items: [] },
  concept_map: { capabilities: [], planes: [], execution_model: [] },
  experiments: [],
};

let calls = 0;
const context = {
  console,
  document,
  location: { hash: "#Jobs" },
  window: { scrollTo() {} },
  MouseEvent: class { constructor(type) { this.type = type; } },
  setTimeout: fn => fn(),
  setInterval() {},
  fetch() {
    calls += 1;
    if (calls === 1) return Promise.resolve({ ok: false, status: 503, statusText: "Service Unavailable" });
    return Promise.resolve({ ok: true, json: () => Promise.resolve(data) });
  },
};
context.window = context;

vm.createContext(context);
vm.runInContext(script, context);
await context.loadData(true);
if (!main.textContent.includes("Could not load data.json")) {
  throw new Error("initial failure did not render the load error");
}
await context.loadData(false);
if (!main.textContent.includes("1 tasks in 1 groups")) {
  throw new Error("late successful poll did not render Jobs counts: " + main.textContent);
}
if (!nav.textContent.includes("jobs/main/")) {
  throw new Error("late successful poll did not rebuild Jobs sidebar: " + nav.textContent);
}
if (!main.textContent.includes("Memory 50%")) {
  throw new Error("Jobs task meta did not render memory score: " + main.textContent);
}
"""

    subprocess.run(
        ["node", "--input-type=module", "-e", code],
        cwd=generate.ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_task_row_uses_result_outcome_not_stale_reward_file(tmp_path: Path):
    """Guards dashboard reward sync with canonical BenchFlow result state."""
    rollout = tmp_path / "task-a__abc123"
    verifier = rollout / "verifier"
    verifier.mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "rewards": None,
                "error": "Agent prompt exceeded wall-clock budget 5s",
                "verifier_error": "verifier crashed: No reward file found",
                "timing": {},
            }
        )
    )
    (rollout / "config.json").write_text(
        json.dumps({"agent": "gemini", "model": "gemini-test", "environment": "docker"})
    )
    (verifier / "reward.txt").write_text("1.0")

    row = generate._task_row(rollout)

    assert row["outcome"] == "verifier_errored"
    assert row["reward"] is None
    reward_artifact = next(
        artifact
        for artifact in row["artifacts"]
        if artifact["name"] == "verifier/reward.txt"
    )
    assert reward_artifact["content"] == "1.0"
    assert reward_artifact["ignored_by_verifier"] is True
    assert "ignored" in reward_artifact["note"]


def test_task_row_artifacts_include_full_path_and_modified_date(tmp_path: Path):
    """Guards the file viewer header contract for full paths and dates."""
    rollout = tmp_path / "task-a__abc123"
    verifier = rollout / "verifier"
    verifier.mkdir(parents=True)
    result = rollout / "result.json"
    result.write_text(
        json.dumps({"task_name": "task-a", "rewards": {"reward": 1.0}, "timing": {}})
    )
    (rollout / "config.json").write_text(json.dumps({"agent": "codex"}))
    reward = verifier / "reward.txt"
    reward.write_text("1.0\n")
    stamp = datetime(2026, 5, 22, 1, 30, 0).timestamp()
    os.utime(reward, (stamp, stamp))

    row = generate._task_row(rollout)
    artifact = next(
        item for item in row["artifacts"] if item["name"] == "verifier/reward.txt"
    )

    assert artifact["path"] == str(reward)
    assert artifact["modified_at"] == "2026-05-22 01:30:00"
    assert "Raw verifier boundary output" in artifact["note"]


def test_task_row_reward_artifacts_explain_canonical_reward_pipeline(tmp_path: Path):
    """Guards Jobs against presenting reward artifacts as competing scores."""
    rollout = tmp_path / "task-a__abc123"
    verifier = rollout / "verifier"
    verifier.mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "oracle",
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
                "timing": {},
            }
        )
    )
    (rollout / "rewards.jsonl").write_text(
        json.dumps(
            {
                "type": "terminal",
                "source": "verifier",
                "value": 1.0,
                "tag": "reward",
            }
        )
    )
    (verifier / "reward.txt").write_text("1")

    row = generate._task_row(rollout)
    by_name = {artifact["name"]: artifact for artifact in row["artifacts"]}

    assert "Canonical rollout result" in by_name["result.json"]["note"]
    assert "Reward event log" in by_name["rewards.jsonl"]["note"]
    assert "Raw verifier boundary output" in by_name["verifier/reward.txt"]["note"]


def test_task_row_surfaces_memory_score_from_result_artifact(tmp_path: Path):
    """Guards OPEN-3 dashboard data from persisted rollout result.json."""
    rollout = tmp_path / "task-a__abc123"
    rollout.mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "rewards": {"reward": 1.0},
                "memory_score": 0.5,
                "timing": {},
            }
        )
    )
    (rollout / "config.json").write_text(json.dumps({"agent": "codex"}))

    row = generate._task_row(rollout)

    assert row["reward"] == 1.0
    assert row["memory_score"] == 0.5
    assert row["outcome"] == "passed"


def test_task_row_falls_back_to_memory_reward_event(tmp_path: Path):
    """Guards dashboard compatibility with memory-space reward events."""
    rollout = tmp_path / "task-a__abc123"
    rollout.mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "rewards": {"reward": 1.0},
                "reward_events": [
                    {
                        "type": "terminal",
                        "reward": 0.25,
                        "source": "memory",
                        "space": "memory",
                        "granularity": "terminal",
                    }
                ],
                "timing": {},
            }
        )
    )
    (rollout / "config.json").write_text(json.dumps({"agent": "codex"}))

    row = generate._task_row(rollout)

    assert row["memory_score"] == 0.25


def test_index_renders_canonical_task_outcome():
    """Guards dashboard status rendering against local reclassification drift."""
    html = Path("dashboard/index.html").read_text()
    assert "function taskOutcomeBadge(outcome)" in html
    assert "const status = taskOutcomeBadge(tk.outcome)" in html


def test_index_jobs_empty_state_uses_total_tasks():
    """Guards v0.5-integration@ffef85d against blank zero-task job groups."""
    html = Path("dashboard/index.html").read_text()
    assert "if (!j.total_tasks)" in html
    assert "if (!j.groups.length)" not in html


def test_collect_jobs_classifies_outcomes_without_importing_benchflow_package(
    tmp_path: Path,
):
    """Guards v0.5-integration@ffef85d against dashboard runtime dependency creep."""
    jobs_root = tmp_path / "previous-worktree"
    run = jobs_root / "jobs" / "2026-05-21__23-57-23"
    cases = {
        "passed": {"rewards": {"reward": 1.0}},
        "failed": {"rewards": {"reward": 0.0}},
        "errored": {"error": "agent crashed"},
        "verifier": {
            "rewards": {"reward": 1.0},
            "verifier_error": "verifier crashed before producing reward",
        },
        "unscored": {},
    }
    for name, result in cases.items():
        rollout = run / f"{name}__abc123"
        rollout.mkdir(parents=True)
        (rollout / "result.json").write_text(
            json.dumps({"task_name": name, "timing": {}, **result})
        )
        (rollout / "config.json").write_text(json.dumps({"agent": "codex"}))

    poison = tmp_path / "poison"
    (poison / "benchflow").mkdir(parents=True)
    (poison / "benchflow" / "__init__.py").write_text(
        'raise ModuleNotFoundError("poisoned benchflow import")\n'
    )
    code = textwrap.dedent(
        f"""
        import json
        import os
        import sys
        from dashboard import generate

        os.environ["BENCHFLOW_DASHBOARD_JOBS_ROOT"] = {str(jobs_root)!r}
        jobs = generate.collect_jobs()
        outcomes = {{
            task["name"]: task["outcome"]
            for group in jobs["groups"]
            for run in group["runs"]
            for task in run["tasks"]
        }}
        print(json.dumps({{
            "outcomes": outcomes,
            "imported_benchflow": any(
                name == "benchflow" or name.startswith("benchflow.")
                for name in sys.modules
            ),
        }}))
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(poison)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=generate.ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert json.loads(result.stdout) == {
        "outcomes": {
            "passed": "passed",
            "failed": "failed",
            "errored": "errored",
            "verifier": "verifier_errored",
            "unscored": "unscored",
        },
        "imported_benchflow": False,
    }


def test_generate_loads_local_roadmap_not_top_level_pythonpath_module(
    tmp_path: Path,
):
    """Guards v0.5-integration@ffef85d against dashboard roadmap import hijack."""
    (tmp_path / "roadmap.py").write_text(
        "def collect_roadmap():\n"
        "    return {'available': True, 'source': {'kind': 'poisoned'}}\n"
    )
    code = textwrap.dedent(
        """
        import json
        from dashboard import generate

        print(json.dumps(generate.collect_roadmap()["source"]))
        """
    )
    env = os.environ.copy()
    env.pop("LINEAR_API_KEY", None)
    env["PYTHONPATH"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=generate.ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert json.loads(result.stdout)["kind"] == "unavailable"


def test_expected_skills_advisory_is_no_longer_open():
    """Guards the ENG-125 follow-up on v0.5-integration@ffef85d."""
    advisory = next(
        item for item in generate.ADVISORIES["items"] if item["id"] == "OPEN-1"
    )
    follow_up = next(
        item for item in generate.ADVISORIES["items"] if item["id"] == "OPEN-3"
    )

    assert advisory["status"] == "resolved"
    assert advisory["severity"] == "should-fix"
    assert "expected_skills" in advisory["detail"]
    assert "memory_delta" in advisory["detail"]
    assert "grades activity, not correctness" not in advisory["detail"]
    assert "grades correctness" not in advisory["detail"]
    assert follow_up["status"] == "resolved"
    assert follow_up["severity"] == "should-fix"
    assert "Memory-space scores" in follow_up["title"]
    assert "result.json" in follow_up["detail"]
    assert "summary.json" in follow_up["detail"]
    assert "dashboard" in follow_up["detail"]


def test_collect_jobs_reads_configured_external_worktree_jobs(
    tmp_path: Path, monkeypatch
):
    """Guards v0.5-integration@ffef85d against missing git-ignored worktree jobs."""
    previous_worktree = tmp_path / "previous-worktree"
    rollout = previous_worktree / "jobs" / "2026-05-21__23-57-23" / "task__abc123"
    rollout.mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {"task_name": "external-evidence", "rewards": {"reward": 1.0}, "timing": {}}
        )
    )
    (rollout / "config.json").write_text(
        json.dumps({"agent": "codex", "model": "test", "environment": "docker"})
    )

    monkeypatch.setenv("BENCHFLOW_DASHBOARD_JOBS_ROOT", str(previous_worktree))

    jobs = generate.collect_jobs()

    assert jobs["source"]["path"] == str((previous_worktree / "jobs").resolve())
    assert jobs["source"]["configured"] is True
    assert jobs["source"]["available"] is True
    assert jobs["total_tasks"] == 1
    assert jobs["groups"][0]["name"] == "main"


def test_collect_jobs_reads_configured_external_jobs_dir(tmp_path: Path, monkeypatch):
    """Guards restart commands that point directly at the producer jobs/ dir."""
    previous_jobs = tmp_path / "previous-worktree" / "jobs"
    rollout = previous_jobs / "2026-05-21__23-57-23" / "task__abc123"
    rollout.mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {"task_name": "external-evidence", "rewards": {"reward": 1.0}, "timing": {}}
        )
    )
    (rollout / "config.json").write_text(json.dumps({"agent": "codex"}))

    monkeypatch.setenv("BENCHFLOW_DASHBOARD_JOBS_ROOT", str(previous_jobs))

    jobs = generate.collect_jobs()

    assert jobs["source"]["path"] == str(previous_jobs.resolve())
    assert jobs["source"]["configured"] is True
    assert jobs["source"]["available"] is True
    assert jobs["total_tasks"] == 1
    assert jobs["groups"][0]["name"] == "main"


def test_collect_jobs_transitions_from_empty_to_nonempty_external_root(
    tmp_path: Path, monkeypatch
):
    """Guards v0.5 dashboard refresh from publishing a permanent blank Jobs tab."""
    previous_worktree = tmp_path / "previous-worktree"
    (previous_worktree / "jobs").mkdir(parents=True)
    monkeypatch.setenv("BENCHFLOW_DASHBOARD_JOBS_ROOT", str(previous_worktree))

    before = generate.collect_jobs()
    assert before["source"]["path"] == str((previous_worktree / "jobs").resolve())
    assert before["source"]["configured"] is True
    assert before["groups"] == []
    assert before["total_tasks"] == 0

    rollout = previous_worktree / "jobs" / "2026-05-21__23-57-23" / "task__abc123"
    rollout.mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {"task_name": "external-evidence", "rewards": {"reward": 1.0}, "timing": {}}
        )
    )
    (rollout / "config.json").write_text(json.dumps({"agent": "codex"}))

    after = generate.collect_jobs()
    assert after["total_tasks"] == 1
    assert after["groups"][0]["name"] == "main"
    assert after["groups"][0]["runs"][0]["id"] == "2026-05-21__23-57-23"
    assert after["groups"][0]["runs"][0]["tasks"][0]["name"] == "external-evidence"


def test_collect_jobs_reuses_remembered_external_jobs_root(tmp_path: Path, monkeypatch):
    """Guards dashboard restarts from losing git-ignored jobs when env is absent."""
    repo = tmp_path / "repo"
    dash = repo / "dashboard"
    (repo / "jobs").mkdir(parents=True)
    dash.mkdir(parents=True)
    previous_jobs = tmp_path / "previous-worktree" / "jobs"
    rollout = previous_jobs / "2026-05-21__23-57-23" / "task__abc123"
    rollout.mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {"task_name": "external-evidence", "rewards": {"reward": 1.0}, "timing": {}}
        )
    )
    (rollout / "config.json").write_text(json.dumps({"agent": "codex"}))
    out = dash / "data.json"
    out.write_text(json.dumps({"jobs": {"source": {"path": str(previous_jobs)}}}))

    monkeypatch.delenv("BENCHFLOW_DASHBOARD_JOBS_ROOT", raising=False)
    monkeypatch.setattr(generate, "ROOT", repo)
    monkeypatch.setattr(generate, "DASH", dash)
    monkeypatch.setattr(generate, "OUT", out)

    jobs = generate.collect_jobs()

    assert jobs["source"]["path"] == str(previous_jobs.resolve())
    assert jobs["source"]["configured"] is False
    assert jobs["source"]["remembered"] is True
    assert jobs["total_tasks"] == 1
    assert jobs["groups"][0]["name"] == "main"


def test_repo_fingerprint_changes_when_configured_jobs_root_changes(
    tmp_path: Path, monkeypatch
):
    """Guards v0.5-integration@ffef85d auto-refresh for external jobs artifacts."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("one\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL
    )

    external = tmp_path / "previous-worktree" / "jobs" / "2026-05-21__23-57-23"
    external.mkdir(parents=True)
    artifact = external / "result.json"
    artifact.write_text("{}\n")

    monkeypatch.setattr(serve, "ROOT", repo)
    monkeypatch.setenv(
        "BENCHFLOW_DASHBOARD_JOBS_ROOT", str(tmp_path / "previous-worktree")
    )

    before = serve.repo_fingerprint()
    artifact.write_text("[]\n")
    after = serve.repo_fingerprint()

    assert before != after


def test_repo_fingerprint_changes_when_configured_jobs_root_gets_first_artifact(
    tmp_path: Path, monkeypatch
):
    """Guards data.json refresh when an external jobs root goes 0 -> nonzero."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("one\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL
    )

    previous_worktree = tmp_path / "previous-worktree"
    (previous_worktree / "jobs").mkdir(parents=True)
    monkeypatch.setattr(serve, "ROOT", repo)
    monkeypatch.setenv("BENCHFLOW_DASHBOARD_JOBS_ROOT", str(previous_worktree))

    before = serve.repo_fingerprint()
    rollout = previous_worktree / "jobs" / "2026-05-21__23-57-23" / "task__abc123"
    rollout.mkdir(parents=True)
    (rollout / "result.json").write_text("{}\n")
    after = serve.repo_fingerprint()

    assert before != after


def test_repo_fingerprint_tracks_remembered_external_jobs_root(
    tmp_path: Path, monkeypatch
):
    """Guards dashboard refresh after restart without the jobs-root env var."""
    repo = tmp_path / "repo"
    dash = repo / "dashboard"
    (repo / "jobs").mkdir(parents=True)
    dash.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("one\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL
    )

    previous_jobs = tmp_path / "previous-worktree" / "jobs"
    rollout = previous_jobs / "2026-05-21__23-57-23" / "task__abc123"
    rollout.mkdir(parents=True)
    artifact = rollout / "result.json"
    artifact.write_text("{}\n")
    (dash / "data.json").write_text(
        json.dumps({"jobs": {"source": {"path": str(previous_jobs)}}})
    )

    monkeypatch.delenv("BENCHFLOW_DASHBOARD_JOBS_ROOT", raising=False)
    monkeypatch.setattr(serve, "ROOT", repo)
    monkeypatch.setattr(serve, "DASH", dash)

    before = serve.repo_fingerprint()
    artifact.write_text("[]\n")
    after = serve.repo_fingerprint()

    assert before != after
