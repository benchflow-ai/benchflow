"""Branching: bench eval branch / bf.branch, kept checkpoints, branch views,
branch-tree export and the viewer's Lineage data.

Oracle scenarios fork a task at its start state; the agent scenario forks a
``claude-agent-acp`` run (scripted fake model, native route: branching
refuses a provider runtime) after its first prompt.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

import benchflow as bf
from tests.e2e import harness as h

DRAFT_PROMPT = "Write a draft first. [[fake-llm:draft]]"
PASS_CHILD = "label=pass,prompt=Write the final answer. [[fake-llm:hello-pass]]"
WRONG_CHILD = "label=wrong,prompt=Write a different answer. [[fake-llm:hello-wrong]]"


@pytest.fixture(scope="module")
def branch_task(tasks_root: Path) -> Path:
    return h.write_task(tasks_root / "branch", "e2e-branch")


@pytest.fixture(scope="module")
def oracle_branch(
    sandbox: str, branch_task: Path, jobs_root: Path, ledger: h.Ledger
) -> Path:
    job = jobs_root / "branch-oracle"
    if not h.needs_run(job):
        return job
    run = h.bench(
        "eval", "branch", "--tasks-dir", branch_task, "--agent", "oracle",
        "--sandbox", sandbox, "--parent", "discard", "--retain-snapshots",
        "--child", "label=a", "--child", "label=b",
        "--jobs-dir", jobs_root, "--job-name", job.name,
        log=jobs_root / "branch-oracle.log",
    )  # fmt: skip
    ledger.record("eval branch oracle, 2 children, --parent discard --retain-snapshots",
                  surface="CLI", seconds=run.seconds, job_dir=job)  # fmt: skip
    h.assert_exit(run, 0)
    return job


@pytest.fixture(scope="module")
def agent_branch(
    sandbox: str, tasks_root: Path, jobs_root: Path, ledger: h.Ledger
) -> Path:
    task = h.write_fake_llm_task(
        tasks_root / "branch-agent", "e2e-branch-agent", script="hello-pass"
    )
    job = jobs_root / "branch-agent"
    if not h.needs_run(job):
        return job
    run = h.bench(
        "eval", "branch", "--tasks-dir", task, *h.fake_agent_args("native"),
        "--sandbox", sandbox, "--prompt", DRAFT_PROMPT, "--prompt", "@instruction",
        "--checkpoints", "prompt:1", "--checkpoint-after-prompt", "1",
        "--concurrency", "2", "--child", PASS_CHILD, "--child", WRONG_CHILD,
        "--jobs-dir", jobs_root, "--job-name", job.name,
        log=jobs_root / "branch-agent.log", timeout=2400,
    )  # fmt: skip
    ledger.record("eval branch agent (fake model), --checkpoints prompt:1, --concurrency 2",
                  surface="CLI", seconds=run.seconds, job_dir=job)  # fmt: skip
    h.assert_exit(run, 0)
    return job


def test_oracle_branch_outputs(oracle_branch: Path):
    summary = h.read_json(oracle_branch / "summary.json")
    assert summary["kind"] == "benchflow-branch-job"
    assert summary["total"] == 2 and summary["passed"] == 2
    trial = h.trial_dirs(oracle_branch)[0]
    tree = h.read_json(trial / "tree.json")
    (fork,) = tree["forks"]
    assert fork["value"] == 1.0
    labels = sorted(c["intervention"]["label"] for c in fork["children"])
    assert labels == ["a", "b"]
    assert all(c["reward_source"] == "verifier" for c in fork["children"])
    assert (trial / "branches" / fork["id"] / "labels.json").is_file()
    result = h.read_json(trial / "result.json")
    assert result["branches"]["parent"] == "discarded"


def test_branch_again_from_kept_checkpoint(
    sandbox: str,
    oracle_branch: Path,
    branch_task: Path,
    jobs_root: Path,
    ledger: h.Ledger,
):
    trial = h.trial_dirs(oracle_branch)[0]
    job = jobs_root / "branch-from-checkpoint"
    run = h.bench(
        "eval", "branch", "--tasks-dir", branch_task, "--from-checkpoint", trial,
        "--agent", "oracle", "--child", "label=c", "--child", "label=d",
        "--jobs-dir", jobs_root, "--job-name", job.name,
        log=jobs_root / "branch-from-checkpoint.log",
    )  # fmt: skip
    ledger.record("eval branch --from-checkpoint (oracle)", surface="CLI",
                  seconds=run.seconds, job_dir=job)  # fmt: skip
    h.assert_exit(run, 0)
    new_trial = h.trial_dirs(job)[0]
    assert (new_trial / "checkpoint_source.json").is_file()
    tree = h.read_json(new_trial / "tree.json")
    assert tree["forks"][0]["value"] == 1.0


def test_branch_views_cli_and_python(oracle_branch: Path, jobs_root: Path):
    run, docs = h.bench_json("eval", "branches", oracle_branch, "--json")
    h.assert_exit(run, 0)
    assert isinstance(docs, list) and len(docs) == 1
    (doc,) = docs
    assert doc["kind"] == "benchflow.branch-view"
    py = bf.load_job(oracle_branch).branch_views()
    assert len(py) == 1
    py_doc = py[0] if isinstance(py[0], dict) else py[0].to_json_dict()
    assert py_doc["kind"] == doc["kind"]
    assert [f["id"] for f in py_doc["forks"]] == [f["id"] for f in doc["forks"]]
    run = h.bench("eval", "branches", jobs_root / "missing-branch-job")
    h.assert_exit(run, 2)


def test_python_branch_oracle_parallel(
    sandbox: str, branch_task: Path, jobs_root: Path, ledger: h.Ledger
):
    started = time.monotonic()
    events: list[dict] = []
    result = bf.branch(
        branch_task,
        agent="oracle",
        children={"x": None, "y": None},
        sandbox=sandbox,
        concurrency=2,
        jobs_dir=jobs_root,
        job_name="py-branch",
        on_event=events.append,
    )
    ledger.record("bf.branch oracle, concurrency=2 (isolated children)", surface="Python",
                  seconds=time.monotonic() - started, job_dir=jobs_root / "py-branch")  # fmt: skip
    assert result.ok
    assert result.value == 1.0
    assert sorted(c.label for c in result.children) == ["x", "y"]
    assert all(c.reward == 1.0 for c in result.children)
    assert events, "on_event received nothing"
    records = result.to_records()
    assert len(records) == 2


def test_agent_branch_reuses_checkpoint_and_scores_children(agent_branch: Path):
    trial = h.trial_dirs(agent_branch)[0]
    tree = h.read_json(trial / "tree.json")
    (fork,) = tree["forks"]
    rewards = {c["intervention"]["label"]: c["reward"] for c in fork["children"]}
    assert rewards == {"pass": 1.0, "wrong": 0.0}
    assert fork["value"] == pytest.approx(0.5)
    # The automatic checkpoint prompt:1 was kept, so the fork reused it.
    assert fork["snapshot"]["reused"] is True, fork["snapshot"]
    assert (trial / "checkpoints.json").is_file()


def test_branch_tree_export_and_validate(
    agent_branch: Path, jobs_root: Path, ledger: h.Ledger
):
    started = time.monotonic()
    children = jobs_root / "branch-children.jsonl"
    pairs = jobs_root / "branch-pairs.jsonl"
    run = h.bench(
        "train", "convert", agent_branch, "--format", "branch-tree", "--out", children,
        "--pairs", pairs, "--manifest", jobs_root / "branch-export.json",
    )  # fmt: skip
    h.assert_exit(run, 0)
    rows = h.read_jsonl(children)
    assert len(rows) == 2
    for row in rows:
        h.validate(row, "benchflow-branch-child-row.v2.schema.json")
        assert row["advantage"] == pytest.approx(row["child"]["reward"] - row["value"])
    # The children were asked different things: no pair by default ...
    assert h.read_jsonl(pairs) == []
    # ... one DPO pair (chosen = the rewarded child) with --pairs-any-request.
    run = h.bench(
        "train", "convert", agent_branch, "--format", "branch-tree", "--out", children,
        "--pairs", pairs, "--pairs-any-request",
    )  # fmt: skip
    h.assert_exit(run, 0)
    pair_rows = h.read_jsonl(pairs)
    assert len(pair_rows) == 1
    h.validate(pair_rows[0], "benchflow-branch-pair-row.v2.schema.json")
    for path in (children, pairs):
        run = h.bench("train", "validate", path, "--format", "branch-tree")
        h.assert_exit(run, 0)
    ledger.record("train convert/validate --format branch-tree", surface="CLI",
                  seconds=time.monotonic() - started)  # fmt: skip


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_viewer_serves_runs_and_lineage(
    agent_branch: Path, batch_job: Path, ledger: h.Ledger
):
    """bench eval view over a jobs folder: catalog page, rollout JSON, a branch child."""
    started = time.monotonic()
    root = agent_branch.parent
    port = _free_port()
    proc = subprocess.Popen(
        [h.bench_executable(), "eval", "view", str(root), "--port", str(port)],
        env=h.clean_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        base = f"http://localhost:{port}"
        deadline = time.monotonic() + 60
        status = 0
        while time.monotonic() < deadline:
            try:
                status, page = _get(base + "/")
                break
            except OSError:
                time.sleep(0.5)
        assert status == 200
        html = page.decode()
        trial = h.trial_dirs(agent_branch)[0]
        rid = str(trial.relative_to(root))
        assert rid in html or json.dumps(rid)[1:-1] in html
        status, body = _get(f"{base}/api/rollout?id={urllib.parse.quote(rid)}")
        assert status == 200
        payload = json.loads(body)
        assert payload
        tree = h.read_json(trial / "tree.json")
        fork = tree["forks"][0]
        child = fork["children"][0]
        ref = f"{fork['id']}/{child['node_id']}"
        status, body = _get(
            f"{base}/api/rollout?id={urllib.parse.quote(rid)}&branch={urllib.parse.quote(ref)}"
        )
        assert status == 200, body[:500]
        # Unknown ids and branch refs are refused, not guessed.
        status, _ = _get(f"{base}/api/rollout?id=../../etc")
        assert status == 404
        status, _ = _get(
            f"{base}/api/rollout?id={urllib.parse.quote(rid)}&branch=nope/nope"
        )
        assert status == 404
        # Only the printed localhost authority is answered (DNS-rebinding guard).
        status, _ = _get(f"http://127.0.0.1:{port}/")
        assert status == 403
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    ledger.record("eval view: catalog, /api/rollout, branch child, refusals", surface="CLI",
                  seconds=time.monotonic() - started)  # fmt: skip
