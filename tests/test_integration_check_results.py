from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from benchflow._utils import benchmark_repos as task_download
from benchflow._utils.benchmark_repos import task_file_hashes
from tests.integration import check_results as result_checker
from tests.integration.check_results import check_agent

EXAMPLE_TASK_ROOT = (
    Path(__file__).parents[1] / "tests" / "examples" / "hello-world-task"
)


def _build_source_repo() -> tuple[Path, str]:
    base = Path(tempfile.mkdtemp(prefix="benchflow-check-results-source-"))
    repo_root = base / "work"
    tasks_root = repo_root / "tasks"
    for task_name in ("task-a", "task-b", "weighted-gdp-calc"):
        shutil.copytree(EXAMPLE_TASK_ROOT, tasks_root / task_name)
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=repo_root, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/acme/benchmarks.git"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=BenchFlow Test",
            "-c",
            "user.email=benchflow-test@example.com",
            "commit",
            "-m",
            "seed tasks",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    sha = completed.stdout.strip()
    result_checker.REMOTE_REACHABILITY[("acme/benchmarks", sha, "main")] = True
    snapshot_root = task_download._cache_dir() / "acme" / "benchmarks__snapshots" / sha
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(repo_root), snapshot_root)
    return snapshot_root, sha


SOURCE_REPO_ROOT, SOURCE_REPO_SHA = _build_source_repo()
TASK_SOURCE_ROOT = SOURCE_REPO_ROOT / "tasks" / "task-a"


def _source(task_name: str = "task-a") -> dict:
    return {
        "type": "github",
        "repo": "acme/benchmarks",
        "requested_ref": "main",
        "resolved_sha": SOURCE_REPO_SHA,
        "path": f"tasks/{task_name}",
        "local_path": str(SOURCE_REPO_ROOT / "tasks" / task_name),
        "dirty": False,
        "file_hashes": task_file_hashes(SOURCE_REPO_ROOT / "tasks" / task_name),
    }


def _source_from_repo(
    repo_root: Path, sha: str, task_name: str = "task-a", *, dirty: bool = False
) -> dict:
    task_dir = repo_root / "tasks" / task_name
    return {
        "type": "github",
        "repo": "acme/benchmarks",
        "requested_ref": "main",
        "resolved_sha": sha,
        "path": f"tasks/{task_name}",
        "local_path": str(task_dir),
        "dirty": dirty,
        "file_hashes": task_file_hashes(task_dir),
    }


def _write_result_tree(
    tmp_path: Path,
    *,
    reward: float,
    summary: dict,
    include_config: bool = True,
    result_model: object = "test-model",
    config_model: object = "test-model",
    summary_model: object = "test-model",
    config_idle_timeout: object = 600,
    summary_idle_timeout: object = 600,
) -> Path:
    agent_dir = tmp_path / "agentA"
    run_dir = agent_dir / "2026-05-18__00-00-00" / "task-a"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "agentA",
                "model": result_model,
                "rewards": {"reward": reward},
                "error": None,
                "verifier_error": None,
                "source": _source(),
            }
        )
    )
    if include_config:
        _write_config(run_dir, model=config_model, idle_timeout=config_idle_timeout)
    summary_payload = {
        "agent": "agentA",
        "model": summary_model,
        "environment": "daytona",
        "concurrency": 64,
        "agent_idle_timeout_sec": summary_idle_timeout,
        **summary,
    }
    if "source" not in summary:
        summary_payload["source"] = _source()
    (agent_dir / "summary.json").write_text(json.dumps(summary_payload))
    return agent_dir


def _write_config(
    path: Path,
    source: dict | None = None,
    *,
    agent: str = "agentA",
    model: str = "test-model",
    idle_timeout: object = 600,
) -> None:
    config = {
        "task_path": "/tmp/tasks/task-a",
        "agent": agent,
        "model": model,
        "environment": "daytona",
        "concurrency": 64,
        "agent_idle_timeout_sec": idle_timeout,
        "source": source or _source(),
    }
    (path / "config.json").write_text(json.dumps(config))


def test_check_results_accepts_skill_invocation_metrics(tmp_path: Path) -> None:
    """Guards issue #507: result checker accepts valid structured skill counts."""
    agent_dir = tmp_path / "agentA"
    run_dir = agent_dir / "2026-05-24__00-00-00" / "task-a"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "agentA",
                "model": "test-model",
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
                "n_skill_invocations": 1,
                "agent_result": {"n_skill_invocations": 1},
                "source": _source(),
            }
        )
    )
    traj_dir = run_dir / "trajectory"
    traj_dir.mkdir()
    (traj_dir / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "tool_call", "kind": "skill", "title": "calculator"})
        + "\n"
    )
    _write_config(run_dir)
    (agent_dir / "summary.json").write_text(
        json.dumps(
            {
                "agent": "agentA",
                "model": "test-model",
                "environment": "daytona",
                "concurrency": 64,
                "agent_idle_timeout_sec": 600,
                "total": 1,
                "passed": 1,
                "failed": 0,
                "errored": 0,
                "verifier_errored": 0,
                "score": "100.0%",
                "total_skill_invocations": 1,
                "avg_skill_invocations": 1.0,
                "source": _source(),
            }
        )
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is True


def test_check_results_flags_skill_invocation_mismatch(tmp_path: Path) -> None:
    """Guards issue #507: result checker compares skill counts to ACP trajectory."""
    agent_dir = tmp_path / "agentA"
    run_dir = agent_dir / "2026-05-24__00-00-00" / "task-a"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "agentA",
                "model": "test-model",
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
                "n_skill_invocations": 0,
                "agent_result": {"n_skill_invocations": 0},
                "source": _source(),
            }
        )
    )
    traj_dir = run_dir / "trajectory"
    traj_dir.mkdir()
    (traj_dir / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "tool_call", "kind": "skill", "title": "calculator"})
        + "\n"
    )
    _write_config(run_dir)
    (agent_dir / "summary.json").write_text(
        json.dumps(
            {
                "agent": "agentA",
                "model": "test-model",
                "environment": "daytona",
                "concurrency": 64,
                "agent_idle_timeout_sec": 600,
                "total": 1,
                "passed": 1,
                "failed": 0,
                "errored": 0,
                "verifier_errored": 0,
                "score": "100.0%",
                "total_skill_invocations": 0,
                "avg_skill_invocations": 0.0,
                "source": _source(),
            }
        )
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any(
        "n_skill_invocations=0 but trajectory implies 1" in issue
        for issue in findings["issues"]
    )


def test_check_results_requires_source_provenance(tmp_path: Path) -> None:
    """Guards v0.5-integration@cb8759e against unaudited source artifacts."""
    agent_dir = tmp_path / "agentA"
    run_dir = agent_dir / "2026-05-22__00-00-00" / "task-a"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "agentA",
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
            }
        )
    )
    _write_config(run_dir)
    (agent_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 1,
                "passed": 1,
                "failed": 0,
                "errored": 0,
                "verifier_errored": 0,
                "score": "100.0%",
            }
        )
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("source" in issue and "missing" in issue for issue in findings["issues"])


def test_check_results_requires_config_source_provenance(tmp_path: Path) -> None:
    """Guards v0.5-integration@cb8759e against split config/result evidence."""
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        include_config=False,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any(
        "config.json" in issue and "missing" in issue for issue in findings["issues"]
    )


def test_check_results_rejects_mismatched_config_source(tmp_path: Path) -> None:
    """Guards v0.5-integration@cb8759e against config/result source divergence."""
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )
    rollout_dir = next(agent_dir.rglob("result.json")).parent
    _write_config(rollout_dir, {**_source(), "resolved_sha": "1" * 40})

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any(
        "config.json source does not match result.json" in issue
        for issue in findings["issues"]
    )


def test_check_results_rejects_bad_file_hash_digest(tmp_path: Path) -> None:
    """Guards v0.5-integration@cb8759e against weak source hash validation."""
    bad_source = _source()
    bad_source["file_hashes"] = {"task.toml": "sha256:abc"}
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
            "source": bad_source,
        },
    )
    result_path = next(agent_dir.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["source"] = bad_source
    result_path.write_text(json.dumps(result))

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("invalid source.file_hashes" in issue for issue in findings["issues"])


def test_check_results_requires_task_toml_source_hash(tmp_path: Path) -> None:
    """Guards v0.5-integration@cb8759e against incomplete task hash evidence."""
    bad_source = _source()
    bad_source["file_hashes"] = {"instruction.md": "sha256:" + "0" * 64}
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )
    result_path = next(agent_dir.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["source"] = bad_source
    result_path.write_text(json.dumps(result))
    _write_config(result_path.parent, bad_source)
    summary_path = agent_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["source"] = bad_source
    summary_path.write_text(json.dumps(summary))

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("task.toml" in issue for issue in findings["issues"])


def test_check_results_recomputes_source_hashes_when_local_path_exists(
    tmp_path: Path,
) -> None:
    """Guards v0.5-integration@cb8759e against forged local file hashes."""
    task_dir = tmp_path / "source" / "tasks" / "task-a"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("[task]\n")
    bad_source = {
        **_source(),
        "local_path": str(task_dir),
        "file_hashes": {"task.toml": "sha256:" + "0" * 64},
    }
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )
    result_path = next(agent_dir.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["source"] = bad_source
    result_path.write_text(json.dumps(result))
    _write_config(result_path.parent, bad_source)
    summary_path = agent_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["source"] = bad_source
    summary_path.write_text(json.dumps(summary))

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any(
        "source.file_hashes do not match local_path" in issue
        for issue in findings["issues"]
    )


def test_check_results_rejects_missing_local_source_path(tmp_path: Path) -> None:
    """Guards v0.5-integration@cb8759e against unauditable source snapshots."""
    missing_source = {**_source(), "local_path": str(tmp_path / "missing-task")}
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )
    result_path = next(agent_dir.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["source"] = missing_source
    result_path.write_text(json.dumps(result))
    _write_config(result_path.parent, missing_source)

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any(
        "source.local_path does not exist" in issue for issue in findings["issues"]
    )


def test_check_results_rejects_non_string_source_local_path(tmp_path: Path) -> None:
    """Guards v0.5-integration@cb8759e against null source snapshots."""
    bad_source = {**_source(), "local_path": None}
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )
    result_path = next(agent_dir.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["source"] = bad_source
    result_path.write_text(json.dumps(result))
    _write_config(result_path.parent, bad_source)

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any(
        "source.local_path must be a string" in issue for issue in findings["issues"]
    )


def test_check_results_rejects_source_git_identity_mismatch(tmp_path: Path) -> None:
    """Guards v0.5-integration@cb8759e against forged source repo/SHA evidence."""
    bad_source = {**_source(), "repo": "other/benchmarks", "resolved_sha": "0" * 40}
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )
    result_path = next(agent_dir.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["source"] = bad_source
    result_path.write_text(json.dumps(result))
    _write_config(result_path.parent, bad_source)

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("git HEAD" in issue for issue in findings["issues"])
    assert any("git remote" in issue for issue in findings["issues"])


def test_check_results_rejects_unreachable_spoofed_remote(
    tmp_path: Path,
) -> None:
    """Guards v0.5-integration@cb8759e against unreachable spoofed GitHub remotes."""
    repo_root = tmp_path / "spoofed"
    shutil.copytree(EXAMPLE_TASK_ROOT, repo_root / "tasks" / "task-a")
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=repo_root, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/acme/benchmarks.git"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=BenchFlow Test",
            "-c",
            "user.email=benchflow-test@example.com",
            "commit",
            "-m",
            "spoofed source",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result_checker.REMOTE_REACHABILITY[("acme/benchmarks", sha, "main")] = False
    source = _source_from_repo(repo_root, sha)
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )
    result_path = next(agent_dir.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["source"] = source
    result_path.write_text(json.dumps(result))
    _write_config(result_path.parent, source)
    summary_path = agent_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.pop("source", None)
    summary_path.write_text(json.dumps(summary))

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("not reachable" in issue for issue in findings["issues"])


def test_check_results_accepts_reachable_sibling_clone_source(tmp_path: Path) -> None:
    """Guards v0.5 feature rollout evidence from being tied to one checkout."""
    repo_root = tmp_path / "sibling"
    shutil.copytree(EXAMPLE_TASK_ROOT, repo_root / "tasks" / "task-a")
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=repo_root, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/acme/benchmarks.git"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=BenchFlow Test",
            "-c",
            "user.email=benchflow-test@example.com",
            "commit",
            "-m",
            "sibling source",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result_checker.REMOTE_REACHABILITY[("acme/benchmarks", sha, "main")] = True
    source = _source_from_repo(repo_root, sha)
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
            "source": source,
        },
    )
    result_path = next(agent_dir.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["source"] = source
    result_path.write_text(json.dumps(result))
    _write_config(result_path.parent, source)
    summary_path = agent_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["source"] = source
    summary_path.write_text(json.dumps(summary))

    findings = check_agent(agent_dir)

    assert findings["ok"] is True, findings["issues"]


def test_check_results_accepts_symlinked_current_repo_inferred_source(
    tmp_path: Path,
) -> None:
    """Guards v0.5 feature rollout tasksets symlinked to this repo."""
    taskset = tmp_path / "taskset"
    taskset.mkdir()
    linked_task = taskset / "hello-world-task"
    linked_task.symlink_to(EXAMPLE_TASK_ROOT, target_is_directory=True)
    source = task_download.infer_task_source_provenance(linked_task)
    assert source is not None
    result_checker.REMOTE_REACHABILITY[
        (
            source["repo"],
            source["resolved_sha"],
            source["requested_ref"],
        )
    ] = True
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
            "source": source,
        },
    )
    result_path = next(agent_dir.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["source"] = source
    result_path.write_text(json.dumps(result))
    _write_config(result_path.parent, source)
    summary_path = agent_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["source"] = source
    summary_path.write_text(json.dumps(summary))

    findings = check_agent(agent_dir)

    assert findings["ok"] is True, findings["issues"]


def test_check_results_rejects_unreachable_remote_sha(tmp_path: Path) -> None:
    """Guards v0.5-integration@cb8759e against unreachable source SHAs."""
    repo_root, sha = _build_source_repo()
    result_checker.REMOTE_REACHABILITY[("acme/benchmarks", sha, "main")] = False
    source = _source_from_repo(repo_root, sha)
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )
    result_path = next(agent_dir.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["source"] = source
    result_path.write_text(json.dumps(result))
    _write_config(result_path.parent, source)

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("not reachable" in issue for issue in findings["issues"])


def test_check_results_rejects_dirty_source_worktree(tmp_path: Path) -> None:
    """Guards v0.5-integration@cb8759e against forged clean source evidence."""
    repo_root, sha = _build_source_repo()
    source = _source_from_repo(repo_root, sha)
    task_dir = repo_root / "tasks" / "task-a"
    (task_dir / "task.toml").write_text("[task]\n# dirty\n")
    dirty_source = {**source, "file_hashes": task_file_hashes(task_dir), "dirty": False}
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )
    result_path = next(agent_dir.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["source"] = dirty_source
    result_path.write_text(json.dumps(result))
    _write_config(result_path.parent, dirty_source)
    summary_path = agent_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["source"] = {
        **dirty_source,
        "path": "tasks",
        "local_path": str(repo_root / "tasks"),
        "file_hashes": {},
    }
    summary_path.write_text(json.dumps(summary))

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("source.dirty does not match" in issue for issue in findings["issues"])


def test_check_results_rejects_result_outside_summary_source(tmp_path: Path) -> None:
    """Guards v0.5-integration@cb8759e against summary/result source divergence."""
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )
    result_path = next(agent_dir.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["source"] = {**_source(), "resolved_sha": "1" * 40}
    result_path.write_text(json.dumps(result))
    _write_config(result_path.parent, result["source"])

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("summary source does not cover" in issue for issue in findings["issues"])


def test_check_results_rejects_result_local_path_outside_summary_source(
    tmp_path: Path,
) -> None:
    """Guards v0.5-integration@cb8759e against local-path source divergence."""
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )
    result_path = next(agent_dir.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["source"] = {**_source(), "local_path": "/other/tasks/task-a"}
    result_path.write_text(json.dumps(result))
    _write_config(result_path.parent, result["source"])

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("summary source does not cover" in issue for issue in findings["issues"])


def test_check_results_treats_partial_reward_as_failure(tmp_path: Path) -> None:
    """Guards ENG-91 P1 integration checker partial-reward regression."""
    agent_dir = _write_result_tree(
        tmp_path,
        reward=0.5,
        summary={
            "total": 1,
            "passed": 0,
            "failed": 1,
            "errored": 0,
            "verifier_errored": 0,
            "score": 0.0,
        },
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is True
    assert findings["passed"] == 0
    assert findings["failed"] == 1


def test_check_results_reconciles_summary_counts(tmp_path: Path) -> None:
    """Guards ENG-91 P1 integration checker summary reconciliation."""
    agent_dir = _write_result_tree(
        tmp_path,
        reward=0.5,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": 1.0,
        },
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("summary.json passed=1" in issue for issue in findings["issues"])


def test_check_results_dedupes_retried_task_results(tmp_path: Path) -> None:
    """Guards the 2026-05-19 Gemini integration retry accounting bug."""
    agent_dir = tmp_path / "gemini"
    run_dir = agent_dir / "2026-05-19__00-00-00"
    first = run_dir / "weighted-gdp-calc__first"
    second = run_dir / "weighted-gdp-calc__second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "result.json").write_text(
        json.dumps(
            {
                "task_name": "weighted-gdp-calc",
                "agent": "gemini",
                "rewards": None,
                "error": "ACP error 400",
                "verifier_error": None,
                "source": _source("weighted-gdp-calc"),
            }
        )
    )
    (second / "result.json").write_text(
        json.dumps(
            {
                "task_name": "weighted-gdp-calc",
                "agent": "gemini",
                "rewards": {"reward": 0.0},
                "error": None,
                "verifier_error": None,
                "source": _source("weighted-gdp-calc"),
            }
        )
    )
    _write_config(first, _source("weighted-gdp-calc"), agent="gemini")
    _write_config(second, _source("weighted-gdp-calc"), agent="gemini")
    (agent_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 1,
                "passed": 0,
                "failed": 1,
                "errored": 0,
                "verifier_errored": 0,
                "score": "0.0%",
                "source": _source("weighted-gdp-calc"),
            }
        )
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is True
    assert findings["total"] == 1
    assert findings["errored"] == 0
    assert findings["failed"] == 1


def test_check_results_counts_mixed_agent_and_verifier_error_once(
    tmp_path: Path,
) -> None:
    """Guards timeout infra errors from being hidden by verifier precedence."""
    agent_dir = tmp_path / "gemini"
    run_dir = agent_dir / "2026-05-21__00-00-00" / "task-a"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "gemini",
                "rewards": None,
                "error": "Agent prompt exceeded wall-clock budget 5s",
                "verifier_error": "verifier crashed: No reward file found",
                "source": _source(),
            }
        )
    )
    _write_config(run_dir, agent="gemini")
    (agent_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 1,
                "passed": 0,
                "failed": 0,
                "errored": 0,
                "verifier_errored": 1,
                "score": "0.0%",
                "source": _source(),
            }
        )
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any(
        "Agent prompt exceeded wall-clock budget" in issue
        for issue in findings["issues"]
    )
    assert findings["errored"] == 0
    assert findings["verifier_errored"] == 1


@pytest.mark.parametrize(
    "error",
    [
        "Agent prompt exceeded wall-clock budget 5s",
        "Agent idle for 600s with no new tool call, message, or thought",
    ],
)
def test_check_results_flags_agent_timeout_infra_errors(
    tmp_path: Path, error: str
) -> None:
    """Guards the integration checker against v0.5 timeout false positives."""
    agent_dir = tmp_path / "gemini"
    run_dir = agent_dir / "2026-05-22__00-00-00" / "task-a"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "gemini",
                "rewards": None,
                "error": error,
                "verifier_error": None,
                "source": _source(),
            }
        )
    )
    _write_config(run_dir, agent="gemini")
    (agent_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 1,
                "passed": 0,
                "failed": 0,
                "errored": 1,
                "verifier_errored": 0,
                "score": "0.0%",
                "source": _source(),
            }
        )
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any(error in issue for issue in findings["issues"])


def test_check_results_requires_verifier_errored_summary_field(
    tmp_path: Path,
) -> None:
    """Guards the v0.5 verifier-error summary contract."""
    agent_dir = _write_result_tree(
        tmp_path,
        reward=0.5,
        summary={"total": 1, "passed": 0, "failed": 1, "errored": 0, "score": 0.0},
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("summary.json missing" in issue for issue in findings["issues"])
    assert any("verifier_errored" in issue for issue in findings["issues"])


def test_check_results_ignores_memory_score_for_output_counts(tmp_path: Path) -> None:
    """Guards OPEN-3: Memory-space score must not change output counts."""
    agent_dir = tmp_path / "agentA"
    run_dir = agent_dir / "2026-05-22__00-00-00" / "task-a"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "agentA",
                "rewards": {"reward": 0.0},
                "memory_score": 1.0,
                "error": None,
                "verifier_error": None,
                "source": _source(),
            }
        )
    )
    _write_config(run_dir)
    (agent_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 1,
                "passed": 0,
                "failed": 1,
                "errored": 0,
                "verifier_errored": 0,
                "score": "0.0%",
                "memory_score": 1.0,
                "source": _source(),
            }
        )
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is True
    assert findings["passed"] == 0
    assert findings["failed"] == 1


def test_check_results_flags_memory_score_summary_mismatch(tmp_path: Path) -> None:
    """Guards OPEN-3 summary consistency when memory scores are present."""
    agent_dir = tmp_path / "agentA"
    run_dir = agent_dir / "2026-05-22__00-00-00"
    for task_name, memory_score in (("task-a", 0.0), ("task-b", 1.0)):
        task_dir = run_dir / f"{task_name}__abc"
        task_dir.mkdir(parents=True)
        (task_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_name,
                    "agent": "agentA",
                    "rewards": {"reward": 1.0},
                    "memory_score": memory_score,
                    "error": None,
                    "verifier_error": None,
                    "source": _source(task_name),
                }
            )
        )
    (agent_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 2,
                "passed": 2,
                "failed": 0,
                "errored": 0,
                "verifier_errored": 0,
                "score": "100.0%",
                "memory_score": 1.0,
                "source": _source(),
            }
        )
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("summary.json memory_score" in issue for issue in findings["issues"])


def test_check_results_cli_accepts_single_rollout_artifact_root(
    tmp_path: Path,
) -> None:
    """Guards v0.5 rollout audit command against direct artifact-root failures."""
    rollout_root = tmp_path / "codex-feature-rollouts-20260522-021530"
    run_dir = rollout_root / "2026-05-22__02-15-31" / "task-a__abc"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "oracle",
                "model": "test-model",
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
                "source": _source(),
            }
        )
    )
    _write_config(run_dir, agent="oracle")
    (rollout_root / "summary.json").write_text(
        json.dumps(
            {
                "agent": "oracle",
                "model": "test-model",
                "environment": "daytona",
                "concurrency": 64,
                "total": 1,
                "passed": 1,
                "failed": 0,
                "errored": 0,
                "verifier_errored": 0,
                "score": "100.0%",
                "source": _source(),
            }
        )
    )

    findings = check_agent(rollout_root)

    assert findings["ok"] is True
    assert findings["total"] == 1


def test_check_results_cli_requires_expected_identity_for_artifact_root(
    tmp_path: Path,
) -> None:
    """Guards v0.5 direct artifact-root audits from certifying unknown identity."""
    rollout_root = tmp_path / "codex-feature-rollouts-20260522-021530"
    run_dir = rollout_root / "2026-05-22__02-15-31" / "task-a__abc"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "oracle",
                "model": "test-model",
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
                "source": _source(),
            }
        )
    )
    _write_config(run_dir, agent="oracle")
    (rollout_root / "summary.json").write_text(
        json.dumps(
            {
                "agent": "oracle",
                "model": "test-model",
                "environment": "daytona",
                "concurrency": 64,
                "total": 1,
                "passed": 1,
                "failed": 0,
                "errored": 0,
                "verifier_errored": 0,
                "score": "100.0%",
                "source": _source(),
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            "tests/integration/check_results.py",
            str(rollout_root),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert (
        "direct artifact-root audits require expected identity args" in completed.stdout
    )


def test_check_results_cli_rejects_missing_requested_agent_dir(
    tmp_path: Path,
) -> None:
    """Guards v0.5-integration@cb8759e against silently skipping missing agents."""
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()

    completed = subprocess.run(
        [
            sys.executable,
            "tests/integration/check_results.py",
            str(jobs_root),
            "gemini",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "missing requested agent directories: gemini" in completed.stdout


def test_check_results_cli_rejects_expected_model_mismatch(
    tmp_path: Path,
) -> None:
    """Guards v0.5-integration@cb8759e expected model evidence checks."""
    _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "tests/integration/check_results.py",
            str(tmp_path),
            "agentA",
            "agentA.model=gemini-3.1-flash-lite-preview",
            "environment=daytona",
            "concurrency=64",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "model" in completed.stdout
    assert "gemini-3.1-flash-lite-preview" in completed.stdout


def test_check_results_cli_accepts_expected_null_model(
    tmp_path: Path,
) -> None:
    """Guards v0.5-integration@c30e130 oracle audits from treating null as missing."""
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        result_model=None,
        config_model=None,
        summary_model=None,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )

    assert result_checker._parse_expected_value("null") is None
    previous = dict(result_checker.EXPECTED)
    try:
        result_checker.EXPECTED.clear()
        result_checker.EXPECTED.update(
            {
                "model": result_checker._parse_expected_value("null"),
                "environment": "daytona",
                "concurrency": "64",
            }
        )
        assert result_checker._has_expected("agentA", "model")

        findings = check_agent(agent_dir)
    finally:
        result_checker.EXPECTED.clear()
        result_checker.EXPECTED.update(previous)

    assert findings["ok"] is True, findings["issues"]


def test_check_results_cli_requires_expected_identity_for_agent_dirs(
    tmp_path: Path,
) -> None:
    """Guards v0.5 agent-dir audits from certifying unknown run identity."""
    _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "tests/integration/check_results.py",
            str(tmp_path),
            "agentA",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "requires expected identity args" in completed.stdout


def test_check_results_cli_rejects_expected_environment_mismatch(
    tmp_path: Path,
) -> None:
    """Guards v0.5-integration@cb8759e expected sandbox evidence checks."""
    _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "tests/integration/check_results.py",
            str(tmp_path),
            "agentA",
            "model=test-model",
            "environment=docker",
            "concurrency=64",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "environment" in completed.stdout
    assert "docker" in completed.stdout


def test_check_results_cli_rejects_expected_concurrency_mismatch(
    tmp_path: Path,
) -> None:
    """Guards v0.5-integration@cb8759e expected concurrency evidence checks."""
    _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "tests/integration/check_results.py",
            str(tmp_path),
            "agentA",
            "model=test-model",
            "environment=daytona",
            "concurrency=100",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "concurrency" in completed.stdout
    assert "100" in completed.stdout


def test_check_results_cli_rejects_expected_agent_idle_timeout_mismatch(
    tmp_path: Path,
) -> None:
    """Guards v0.5-idle-timeout audit against unaudited idle-timeout evidence."""
    _write_result_tree(
        tmp_path,
        reward=1.0,
        config_idle_timeout=600,
        summary_idle_timeout=600,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "tests/integration/check_results.py",
            str(tmp_path),
            "agentA",
            "model=test-model",
            "environment=daytona",
            "concurrency=64",
            "agent_idle_timeout_sec=45",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "agent_idle_timeout_sec" in completed.stdout
    assert "config.json agent_idle_timeout_sec" in completed.stdout
    assert "summary.json agent_idle_timeout_sec" in completed.stdout
    assert "45" in completed.stdout


def test_check_results_rejects_malformed_agent_idle_timeout_artifact(
    tmp_path: Path,
) -> None:
    """Guards v0.5-idle-timeout audit from accepting bool/float timeout evidence."""
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        config_idle_timeout=False,
        summary_idle_timeout=600,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any(
        "config.json agent_idle_timeout_sec must be null or integer seconds" in issue
        for issue in findings["issues"]
    )


def test_check_results_rejects_config_summary_agent_idle_timeout_mismatch(
    tmp_path: Path,
) -> None:
    """Guards v0.5-idle-timeout audit from needing an expected value to catch drift."""
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        config_idle_timeout=45,
        summary_idle_timeout=None,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any(
        "config.json agent_idle_timeout_sec 45 does not match expected null" in issue
        and "from summary.json" in issue
        for issue in findings["issues"]
    )


def test_check_results_cli_accepts_expected_agent_idle_timeout_alias_zero(
    tmp_path: Path,
) -> None:
    """Guards v0.5-idle-timeout audit from treating disabled idle timeout as missing."""
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        config_idle_timeout=None,
        summary_idle_timeout=None,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )

    previous = dict(result_checker.EXPECTED)
    try:
        result_checker.EXPECTED.clear()
        result_checker.EXPECTED.update(
            {
                "model": "test-model",
                "environment": "daytona",
                "concurrency": "64",
                "agent_idle_timeout": "0",
            }
        )

        findings = check_agent(agent_dir)
    finally:
        result_checker.EXPECTED.clear()
        result_checker.EXPECTED.update(previous)

    assert findings["ok"] is True, findings["issues"]


def test_check_results_cli_rejects_unknown_expected_key(
    tmp_path: Path,
) -> None:
    """Guards v0.5-integration@cb8759e against typo-tolerant audit expectations."""
    _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "tests/integration/check_results.py",
            str(tmp_path),
            "agentA",
            "enviroment=daytona",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "unknown expected field 'enviroment'" in completed.stdout


def test_check_results_cli_artifact_root_counts_all_rollouts(
    tmp_path: Path,
) -> None:
    """Guards v0.5-integration@cb8759e against auditing only the latest child."""
    rollout_root = tmp_path / "codex-feature-rollouts"
    for task_name in ("task-a", "task-b"):
        task_dir = rollout_root / "2026-05-22__02-15-31" / f"{task_name}__abc"
        task_dir.mkdir(parents=True)
        source = _source(task_name)
        (task_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_name,
                    "agent": "oracle",
                    "model": "test-model",
                    "rewards": {"reward": 1.0},
                    "error": None,
                    "verifier_error": None,
                    "source": source,
                }
            )
        )
        _write_config(task_dir, source, agent="oracle")
    (rollout_root / "summary.json").write_text(
        json.dumps(
            {
                "agent": "oracle",
                "model": "test-model",
                "environment": "daytona",
                "concurrency": 64,
                "total": 2,
                "passed": 2,
                "failed": 0,
                "errored": 0,
                "verifier_errored": 0,
                "score": "100.0%",
                "source": {
                    **_source(),
                    "path": "tasks",
                    "local_path": str(TASK_SOURCE_ROOT.parent),
                    "file_hashes": {},
                },
            }
        )
    )

    findings = check_agent(rollout_root)

    assert findings["ok"] is True
    assert findings["total"] == 2


def test_check_results_audits_every_duplicate_task_rollout(tmp_path: Path) -> None:
    """Guards v0.5-integration@cb8759e against hidden duplicate rollout artifacts."""
    agent_dir = tmp_path / "agentA"
    run_dir = agent_dir / "2026-05-22__00-00-00"
    good = run_dir / "task-a__good"
    bad = run_dir / "task-a__bad"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    source = _source()
    for rollout_dir, result_source in ((bad, None), (good, source)):
        result = {
            "task_name": "task-a",
            "agent": "agentA",
            "rewards": {"reward": 1.0},
            "error": None,
            "verifier_error": None,
        }
        if result_source is not None:
            result["source"] = result_source
        (rollout_dir / "result.json").write_text(json.dumps(result))
        _write_config(rollout_dir, source)
    (agent_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 1,
                "passed": 1,
                "failed": 0,
                "errored": 0,
                "verifier_errored": 0,
                "score": "100.0%",
                "source": source,
            }
        )
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("missing source provenance" in issue for issue in findings["issues"])


def test_check_results_rejects_malformed_duplicate_task_result(
    tmp_path: Path,
) -> None:
    """Guards v0.5-integration@cb8759e against unreadable duplicate rollouts."""
    agent_dir = _write_result_tree(
        tmp_path,
        reward=1.0,
        summary={
            "total": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "verifier_errored": 0,
            "score": "100.0%",
        },
    )
    bad = agent_dir / "2026-05-18__00-00-00" / "task-a__bad"
    bad.mkdir()
    (bad / "result.json").write_text("{")

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("bad result file" in issue for issue in findings["issues"])


def test_check_results_flags_sandbox_startup_error_with_diagnostics(
    tmp_path: Path,
) -> None:
    """Guards ENG-147: check_results surfaces structured sandbox_startup_info
    for sandbox_setup errors and flags them as INVALIDATED."""
    agent_dir = tmp_path / "agentA"
    run_dir = agent_dir / "2026-05-23__00-00-00" / "task-a"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "agentA",
                "model": "test-model",
                "rewards": None,
                "error": "Sandbox startup failed: Sandbox creation failed after retries: timeout",
                "verifier_error": None,
                "sandbox_startup_info": {
                    "reason": "sandbox_startup_failed",
                    "sandbox_id": "abc-123",
                    "sandbox_state": "creating",
                    "attempts": 3,
                    "build_timeout_sec": 600.0,
                    "raw_message": "timeout",
                },
                "source": _source(),
            }
        )
    )
    findings = check_agent(agent_dir)
    assert findings["ok"] is False
    assert any(
        "sandbox startup failed" in issue and "abc-123" in issue
        for issue in findings["issues"]
    )
    assert any(
        "INVALIDATED" in issue and "sandbox startup" in issue
        for issue in findings["issues"]
    )


def test_check_results_sandbox_startup_without_info(
    tmp_path: Path,
) -> None:
    """Guards ENG-147: check_results handles sandbox_setup errors without
    structured info (bare error string fallback)."""
    agent_dir = tmp_path / "agentA"
    run_dir = agent_dir / "2026-05-23__00-00-00" / "task-a"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "agentA",
                "model": "test-model",
                "rewards": None,
                "error": "Sandbox startup failed: unknown error",
                "verifier_error": None,
                "source": _source(),
            }
        )
    )
    findings = check_agent(agent_dir)
    assert findings["ok"] is False
    assert any(
        "INVALIDATED" in issue and "sandbox startup" in issue
        for issue in findings["issues"]
    )


def test_check_results_flags_transport_error_with_diagnostics(
    tmp_path: Path,
) -> None:
    """Guards ENG-148: check_results surfaces structured transport_error_info
    for pipe_closed errors and flags them as INVALIDATED."""
    agent_dir = tmp_path / "agentA"
    run_dir = agent_dir / "2026-05-18__00-00-00" / "task-a"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "agentA",
                "model": "test-model",
                "rewards": None,
                "error": "Process closed stdout (rc=255): Local subprocess exited",
                "verifier_error": None,
                "transport_error_info": {
                    "reason": "transport_closed",
                    "process_exit_code": 255,
                    "transport_diagnosis": "process_exited",
                    "sandbox_reachable": False,
                },
                "source": _source(),
            }
        )
    )
    _write_config(run_dir)
    (agent_dir / "summary.json").write_text(
        json.dumps(
            {
                "agent": "agentA",
                "model": "test-model",
                "environment": "daytona",
                "concurrency": 64,
                "agent_idle_timeout_sec": 600,
                "total": 1,
                "passed": 0,
                "failed": 0,
                "errored": 1,
                "verifier_errored": 0,
                "score": "0.0%",
                "source": _source(),
            }
        )
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("rc=255" in issue for issue in findings["issues"])
    assert any("transport closed" in issue for issue in findings["issues"])
    assert any(
        "INVALIDATED" in issue and "transport" in issue.lower()
        for issue in findings["issues"]
    )


def test_check_results_transport_error_without_info_still_flagged(
    tmp_path: Path,
) -> None:
    """Guards ENG-148: pipe_closed results without transport_error_info
    still get flagged as infra errors."""
    agent_dir = tmp_path / "agentA"
    run_dir = agent_dir / "2026-05-18__00-00-00" / "task-a"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "agentA",
                "model": "test-model",
                "rewards": None,
                "error": "Agent process closed stdout",
                "verifier_error": None,
                "source": _source(),
            }
        )
    )
    _write_config(run_dir)
    (agent_dir / "summary.json").write_text(
        json.dumps(
            {
                "agent": "agentA",
                "model": "test-model",
                "environment": "daytona",
                "concurrency": 64,
                "agent_idle_timeout_sec": 600,
                "total": 1,
                "passed": 0,
                "failed": 0,
                "errored": 1,
                "verifier_errored": 0,
                "score": "0.0%",
                "source": _source(),
            }
        )
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any(
        "INVALIDATED" in issue and "transport" in issue.lower()
        for issue in findings["issues"]
    )


def test_check_results_flags_verifier_dep_install_failure(
    tmp_path: Path,
) -> None:
    """Guards ENG-151: check_results detects verifier dependency install
    failures and flags them as INVALIDATED."""
    agent_dir = tmp_path / "agentA"
    run_dir = agent_dir / "2026-05-24__00-00-00" / "simpo-code-reproduction"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "simpo-code-reproduction",
                "agent": "agentA",
                "model": "test-model",
                "rewards": None,
                "error": None,
                "verifier_error": (
                    "verifier crashed: verifier exited with rc=1; "
                    "dependency install failed"
                ),
                "source": _source(),
            }
        )
    )
    _write_config(run_dir)
    (agent_dir / "summary.json").write_text(
        json.dumps(
            {
                "agent": "agentA",
                "model": "test-model",
                "environment": "daytona",
                "concurrency": 64,
                "agent_idle_timeout_sec": 600,
                "total": 1,
                "passed": 0,
                "failed": 0,
                "errored": 0,
                "verifier_errored": 1,
                "score": "0.0%",
                "source": _source(),
            }
        )
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("dependency install failed" in issue for issue in findings["issues"])
    assert any(
        "INVALIDATED" in issue and "dependency install" in issue.lower()
        for issue in findings["issues"]
    )


def test_check_results_flags_verifier_timeout(tmp_path: Path) -> None:
    """Guards ENG-152: check_results detects verifier timeout failures
    and flags them as INVALIDATED with budget/elapsed details."""
    agent_dir = tmp_path / "agentA"
    task_dir = agent_dir / "quantum-numerical-simulation" / "quantum__abc123"
    task_dir.mkdir(parents=True)
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "quantum-numerical-simulation",
                "rollout_name": "quantum__abc123",
                "rewards": None,
                "agent": "gemini",
                "agent_name": "gemini-cli",
                "model": "gemini-2.0-flash-lite",
                "n_tool_calls": 0,
                "n_prompts": 1,
                "error": None,
                "error_category": None,
                "verifier_error": "verifier timed out after 240s",
                "verifier_error_category": "verifier_timeout",
                "verifier_timeout_info": {
                    "timeout_budget_sec": 240.0,
                    "elapsed_sec": 240.1,
                    "task_name": "quantum-numerical-simulation",
                },
                "idle_timeout_info": None,
                "sandbox_startup_info": None,
                "transport_error_info": None,
                "partial_trajectory": False,
                "trajectory_source": None,
                "started_at": "2026-05-23 10:00:00",
                "finished_at": "2026-05-23 10:04:00",
                "timing": {"agent": 0.0, "verifier": 240.1, "total": 240.5},
            }
        )
    )
    traj_dir = task_dir / "trajectory"
    traj_dir.mkdir()
    (traj_dir / "acp_trajectory.jsonl").write_text("")
    (agent_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 1,
                "passed": 0,
                "failed": 0,
                "errored": 0,
                "verifier_errored": 1,
                "score": "0.0%",
                "source": _source(),
            }
        )
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any(
        "verifier timed out" in issue and "budget=240" in issue
        for issue in findings["issues"]
    )
    assert any(
        "INVALIDATED" in issue
        and "verifier" in issue.lower()
        and "timeout" in issue.lower()
        for issue in findings["issues"]
    )
