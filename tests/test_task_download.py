"""Tests for benchmark task repository materialization."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils import benchmark_repos as task_download
from benchflow._utils.benchmark_repos import (
    Source,
    resolve_source,
    resolve_source_with_metadata,
    task_source_provenance,
)


def _fake_worktree(cmd):
    repo_root = Path(cmd[2])
    snapshot = Path(cmd[-2])
    shutil.copytree(repo_root, snapshot, dirs_exist_ok=True)
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_skillsbench_alias_clones_main_branch(tmp_path, monkeypatch):
    """Guards PR #226: SkillsBench downloads must track GitHub main explicitly."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is True
        clone_dir = Path(cmd[-1])
        (clone_dir / ".git").mkdir(parents=True)
        (clone_dir / "tasks" / "sample-task").mkdir(parents=True)

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    target = task_download.ensure_tasks("skillsbench")

    assert (
        target
        == tmp_path / ".cache" / "datasets" / "benchflow-ai" / "skillsbench" / "tasks"
    )
    assert target.exists()
    assert calls == [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "main",
            "https://github.com/benchflow-ai/skillsbench.git",
            str(
                tmp_path / ".cache" / "datasets" / "benchflow-ai" / "_skillsbench_clone"
            ),
        ]
    ]


def test_programbench_alias_resolves_to_official_repo(tmp_path, monkeypatch):
    """Guards ENG-81: programbench alias resolves to current upstream metadata."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is True
        clone_dir = Path(cmd[-1])
        clone_dir.mkdir(parents=True)
        (clone_dir / ".git").mkdir()
        (clone_dir / "src" / "programbench" / "data" / "tasks" / "sample").mkdir(
            parents=True
        )

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    target = task_download.ensure_tasks("programbench")

    assert (
        target
        == tmp_path
        / ".cache"
        / "datasets"
        / "facebookresearch"
        / "programbench"
        / "src"
        / "programbench"
        / "data"
        / "tasks"
    )
    assert target.exists()
    assert "--branch" in calls[0]
    assert "main" in calls[0]
    assert "https://github.com/facebookresearch/programbench.git" in calls[0]


def test_resolve_source_with_path(tmp_path, monkeypatch):
    """resolve_source with repo + path clones the repo and returns the subpath."""
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, check):
        assert check is True
        clone_dir = Path(cmd[-1])
        clone_dir.mkdir(parents=True)
        (clone_dir / ".git").mkdir()
        (clone_dir / "my-benchmark" / "task.toml").parent.mkdir(parents=True)
        (clone_dir / "my-benchmark" / "task.toml").write_text("[task]\n")

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    target = resolve_source("acme-org/benchmarks", path="my-benchmark")

    assert (
        target
        == tmp_path / ".cache" / "datasets" / "acme-org" / "benchmarks" / "my-benchmark"
    )
    assert target.exists()


def test_resolve_source_with_metadata_records_sha_and_task_hashes(
    tmp_path, monkeypatch
):
    """Guards v0.5-integration@cb8759e against unauditable source artifacts."""
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, check=False, capture_output=False, text=False):
        del capture_output, text
        if cmd[:2] == ["git", "clone"]:
            assert check is True
            clone_dir = Path(cmd[-1])
            task = clone_dir / "tasks" / "sample"
            (task / "tests").mkdir(parents=True)
            (task / "task.toml").write_text("[task]\n")
            (task / "instruction.md").write_text("Solve it.\n")
            (task / "tests" / "test.sh").write_text("exit 0\n")
            (clone_dir / ".git").mkdir()
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "worktree" in cmd:
            assert check is True
            return _fake_worktree(cmd)
        if cmd[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(
                returncode=0,
                stdout="c65af83ae2c76fda3f1fd4d2fcf56563975e283e\n",
                stderr="",
            )
        if cmd[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    resolved = resolve_source_with_metadata(
        "acme-org/benchmarks",
        path="tasks/sample",
        ref="main",
    )

    assert resolved.path == (
        tmp_path
        / ".cache"
        / "datasets"
        / "acme-org"
        / "benchmarks__snapshots"
        / "c65af83ae2c76fda3f1fd4d2fcf56563975e283e"
        / "tasks"
        / "sample"
    )
    assert resolved.provenance["type"] == "github"
    assert resolved.provenance["repo"] == "acme-org/benchmarks"
    assert resolved.provenance["requested_ref"] == "main"
    assert (
        resolved.provenance["resolved_sha"]
        == "c65af83ae2c76fda3f1fd4d2fcf56563975e283e"
    )
    assert resolved.provenance["path"] == "tasks/sample"
    assert resolved.provenance["local_path"] == str(resolved.path)
    assert resolved.provenance["dirty"] is False
    assert set(resolved.provenance["file_hashes"]) == {
        "instruction.md",
        "task.toml",
        "tests/test.sh",
    }
    assert resolved.provenance["file_hashes"]["task.toml"].startswith("sha256:")


def test_resolve_source_with_metadata_records_task_md_hashes(tmp_path, monkeypatch):
    """Guards commit 67378ddd's 2026-06-04 task.md spike source hashes."""
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, check=False, capture_output=False, text=False):
        del capture_output, text
        if cmd[:2] == ["git", "clone"]:
            assert check is True
            clone_dir = Path(cmd[-1])
            task = clone_dir / "tasks" / "sample"
            (task / "tests").mkdir(parents=True)
            (task / "task.md").write_text(
                "---\nversion: '1.0'\n---\n## prompt\n\nSolve it.\n"
            )
            (task / "tests" / "test.sh").write_text("exit 0\n")
            (clone_dir / ".git").mkdir()
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "worktree" in cmd:
            assert check is True
            return _fake_worktree(cmd)
        if cmd[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="c" * 40 + "\n", stderr="")
        if cmd[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    resolved = resolve_source_with_metadata(
        "acme-org/benchmarks",
        path="tasks/sample",
        ref="main",
    )

    assert set(resolved.provenance["file_hashes"]) == {
        "task.md",
        "tests/test.sh",
    }
    assert resolved.provenance["file_hashes"]["task.md"].startswith("sha256:")


def test_resolve_source_with_metadata_uses_snapshot_sha_for_provenance(
    tmp_path, monkeypatch
):
    """Guards v0.5-integration@cb8759e against mutable-cache SHA races."""
    monkeypatch.chdir(tmp_path)
    initial_sha = "a" * 40
    mutated_sha = "b" * 40
    rev_parse_calls = 0

    def fake_run(cmd, check=False, capture_output=False, text=False):
        nonlocal rev_parse_calls
        del capture_output, text
        if cmd[:2] == ["git", "clone"]:
            assert check is True
            clone_dir = Path(cmd[-1])
            task = clone_dir / "tasks" / "sample"
            task.mkdir(parents=True)
            (task / "task.toml").write_text("[task]\n")
            (clone_dir / ".git").mkdir()
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "worktree" in cmd:
            assert check is True
            return _fake_worktree(cmd)
        if cmd[-2:] == ["rev-parse", "HEAD"]:
            rev_parse_calls += 1
            sha = (
                initial_sha
                if rev_parse_calls == 1 or "__snapshots" in cmd[2]
                else mutated_sha
            )
            return SimpleNamespace(returncode=0, stdout=f"{sha}\n", stderr="")
        if cmd[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    resolved = resolve_source_with_metadata(
        "acme-org/benchmarks",
        path="tasks/sample",
        ref="main",
    )

    assert initial_sha in str(resolved.path)
    assert resolved.provenance["resolved_sha"] == initial_sha


def test_resolve_source_with_metadata_snapshots_under_cache_lock(tmp_path, monkeypatch):
    """Guards v0.5-integration@cb8759e against concurrent checkout races."""
    monkeypatch.chdir(tmp_path)
    lock_path = tmp_path / ".cache" / "datasets" / "acme-org" / ".benchmarks.lock"

    def fake_run(cmd, check=False, capture_output=False, text=False):
        del capture_output, text
        if cmd[:2] == ["git", "clone"]:
            assert check is True
            clone_dir = Path(cmd[-1])
            task = clone_dir / "tasks" / "sample"
            task.mkdir(parents=True)
            (task / "task.toml").write_text("[task]\n")
            (clone_dir / ".git").mkdir()
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "worktree" in cmd:
            assert lock_path.exists()
            assert check is True
            return _fake_worktree(cmd)
        if cmd[-2:] == ["rev-parse", "HEAD"]:
            if "__snapshots" not in cmd[2]:
                assert lock_path.exists()
            return SimpleNamespace(returncode=0, stdout=f"{'c' * 40}\n", stderr="")
        if cmd[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    resolved = resolve_source_with_metadata(
        "acme-org/benchmarks",
        path="tasks/sample",
        ref="main",
    )

    assert resolved.provenance["resolved_sha"] == "c" * 40


def test_task_source_provenance_derives_batch_task_path_and_hashes(tmp_path):
    """Guards v0.5-integration@cb8759e against parent-directory batch evidence."""
    tasks_root = tmp_path / "tasks"
    task = tasks_root / "sample-task"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text("[task]\n")
    (task / "instruction.md").write_text("Solve it.\n")
    (task / "tests" / "test.sh").write_text("exit 0\n")

    provenance = {
        "type": "github",
        "repo": "benchflow-ai/benchmarks",
        "requested_ref": "main",
        "resolved_sha": "c65af83ae2c76fda3f1fd4d2fcf56563975e283e",
        "path": "datasets/programbench/tasks",
        "local_path": str(tasks_root),
        "file_hashes": {},
    }

    task_provenance = task_source_provenance(provenance, task)

    assert task_provenance["path"] == "datasets/programbench/tasks/sample-task"
    assert task_provenance["local_path"] == str(task)
    assert set(task_provenance["file_hashes"]) == {
        "instruction.md",
        "task.toml",
        "tests/test.sh",
    }
    assert "sample-task/task.toml" not in task_provenance["file_hashes"]


def test_task_source_provenance_rejects_task_outside_source_root(tmp_path):
    """Guards v0.5-integration@cb8759e against false out-of-tree provenance."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    outside_task = tmp_path / "outside" / "task-a"
    outside_task.mkdir(parents=True)
    (outside_task / "task.toml").write_text("[task]\n")
    provenance = {
        "type": "github",
        "repo": "benchflow-ai/benchmarks",
        "requested_ref": "main",
        "resolved_sha": "c65af83ae2c76fda3f1fd4d2fcf56563975e283e",
        "path": "datasets/programbench/tasks",
        "local_path": str(tasks_root),
        "file_hashes": {},
    }

    with pytest.raises(ValueError, match="outside source"):
        task_source_provenance(provenance, outside_task)


def test_resolve_source_with_metadata_records_canonical_source_path(
    tmp_path, monkeypatch
):
    """Guards v0.5-integration@cb8759e against non-canonical source evidence."""
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, check=False, capture_output=False, text=False):
        del capture_output, text
        if cmd[:2] == ["git", "clone"]:
            assert check is True
            clone_dir = Path(cmd[-1])
            task = clone_dir / "tasks" / "sample"
            task.mkdir(parents=True)
            (task / "task.toml").write_text("[task]\n")
            (clone_dir / ".git").mkdir()
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "worktree" in cmd:
            assert check is True
            return _fake_worktree(cmd)
        if cmd[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(
                returncode=0,
                stdout="c65af83ae2c76fda3f1fd4d2fcf56563975e283e\n",
                stderr="",
            )
        if cmd[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    resolved = resolve_source_with_metadata(
        "acme-org/benchmarks",
        path="tasks/../tasks/sample",
        ref="main",
    )

    assert resolved.provenance["path"] == "tasks/sample"


def test_task_file_hashes_rejects_symlink_files(tmp_path):
    """Guards v0.5-integration@cb8759e against source-provenance hash escape."""
    secret = tmp_path / "secret.txt"
    secret.write_text("do-not-hash\n")
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text("[task]\n")
    (task / "leak").symlink_to(secret)

    with pytest.raises(ValueError, match="symlink"):
        task_download.task_file_hashes(task)


def test_task_file_hashes_rejects_symlinked_task_directory(tmp_path):
    """Guards v0.5-integration@cb8759e against symlinked task-dir escape."""
    outside_task = tmp_path / "outside-task"
    outside_task.mkdir()
    (outside_task / "task.toml").write_text("[task]\n")
    source_root = tmp_path / "source" / "tasks"
    source_root.mkdir(parents=True)
    linked_task = source_root / "linked-task"
    linked_task.symlink_to(outside_task, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        task_download.task_file_hashes(linked_task)


def test_resolve_source_with_metadata_fails_without_git_sha(tmp_path, monkeypatch):
    """Guards v0.5-integration@cb8759e against unauditable source metadata."""
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, check=False, capture_output=False, text=False):
        del capture_output, text
        if cmd[:2] == ["git", "clone"]:
            assert check is True
            clone_dir = Path(cmd[-1])
            (clone_dir / ".git").mkdir(parents=True)
            (clone_dir / "task.toml").write_text("[task]\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=128, stdout="", stderr="not git")
        if cmd[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Unable to read resolved git SHA"):
        resolve_source_with_metadata("acme-org/benchmarks")


def test_resolve_source_with_metadata_rejects_path_escape(tmp_path, monkeypatch):
    """Guards v0.5-integration@cb8759e against source path escape evidence."""
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, check=False, capture_output=False, text=False):
        del capture_output, text
        if cmd[:2] == ["git", "clone"]:
            assert check is True
            clone_dir = Path(cmd[-1])
            (clone_dir / ".git").mkdir(parents=True)
            outside = clone_dir.parent / "outside-task"
            outside.mkdir()
            (outside / "task.toml").write_text("[task]\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(
                returncode=0,
                stdout="c65af83ae2c76fda3f1fd4d2fcf56563975e283e\n",
                stderr="",
            )
        if cmd[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="escapes repository root"):
        resolve_source_with_metadata("acme-org/benchmarks", path="../outside-task")


def test_resolve_source_without_path(tmp_path, monkeypatch):
    """resolve_source without path returns repo root."""
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, check):
        assert check is True
        clone_dir = Path(cmd[-1])
        clone_dir.mkdir(parents=True)
        (clone_dir / ".git").mkdir()
        (clone_dir / "task-1").mkdir()

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    target = resolve_source("acme-org/my-tasks")

    assert target == tmp_path / ".cache" / "datasets" / "acme-org" / "my-tasks"
    assert target.exists()


def test_resolve_source_with_ref(tmp_path, monkeypatch):
    """resolve_source passes ref as --branch to git clone."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is True
        clone_dir = Path(cmd[-1])
        clone_dir.mkdir(parents=True)
        (clone_dir / ".git").mkdir()

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    resolve_source("org/repo", ref="v2.0")

    assert "--branch" in calls[0]
    assert "v2.0" in calls[0]


def test_resolve_source_with_sha_ref_fetches_after_clone(tmp_path, monkeypatch):
    """Guards v0.5-integration@cb8759e against raw commit-SHA source refs."""
    monkeypatch.chdir(tmp_path)
    calls = []
    sha_ref = "c65af83ae2c76fda3f1fd4d2fcf56563975e283e"
    clone_tmp = tmp_path / ".cache" / "datasets" / "org" / "_repo_clone"

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is True
        if cmd[:2] == ["git", "clone"]:
            clone_dir = Path(cmd[-1])
            clone_dir.mkdir(parents=True)
            (clone_dir / ".git").mkdir()
            (clone_dir / "tasks").mkdir()

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    resolve_source("org/repo", path="tasks", ref=sha_ref)

    assert calls == [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/org/repo.git",
            str(clone_tmp),
        ],
        ["git", "-C", str(clone_tmp), "fetch", "--depth", "1", "origin", sha_ref],
        ["git", "-C", str(clone_tmp), "checkout", "--detach", "FETCH_HEAD"],
    ]


def test_resolve_source_with_ref_refreshes_cached_checkout(tmp_path, monkeypatch):
    """Guards v0.5-integration@cb8759e against stale branch cache evidence."""
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".cache" / "datasets" / "org" / "repo"
    (cache / ".git").mkdir(parents=True)
    (cache / "tasks").mkdir()
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is True

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    resolve_source("org/repo", path="tasks", ref="main")

    assert calls == [
        ["git", "-C", str(cache), "fetch", "--depth", "1", "origin", "main"],
        ["git", "-C", str(cache), "checkout", "--detach", "FETCH_HEAD"],
    ]


def test_resolve_source_caches_on_second_call(tmp_path, monkeypatch):
    """Second call to resolve_source should not clone again."""
    monkeypatch.chdir(tmp_path)
    call_count = 0

    def fake_run(cmd, check):
        nonlocal call_count
        call_count += 1
        clone_dir = Path(cmd[-1])
        clone_dir.mkdir(parents=True)
        (clone_dir / ".git").mkdir()
        (clone_dir / "tasks" / "t1").mkdir(parents=True)

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    resolve_source("test-org/test-repo", path="tasks")
    resolve_source("test-org/test-repo", path="tasks")

    assert call_count == 1


def test_source_dataclass_resolve(tmp_path, monkeypatch):
    """Source dataclass .resolve() delegates to resolve_source."""
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, check):
        clone_dir = Path(cmd[-1])
        clone_dir.mkdir(parents=True)
        (clone_dir / ".git").mkdir()
        (clone_dir / "tb2" / "task.toml").parent.mkdir(parents=True)
        (clone_dir / "tb2" / "task.toml").write_text("[task]\n")

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    src = Source(repo="benchflow-ai/benchmarks", path="tb2", ref="main")
    target = src.resolve()

    assert (
        target
        == tmp_path / ".cache" / "datasets" / "benchflow-ai" / "benchmarks" / "tb2"
    )
    assert target.exists()


def test_infer_task_source_provenance_for_cached_benchmark_repo(tmp_path, monkeypatch):
    """Guards #492: tasks under ``.cache/datasets/<org>/<repo>/...`` must
    attribute provenance to the cached benchmark repo (its remote/HEAD), not
    to the BenchFlow worktree. Without this, ``check_results.py`` warns
    ``source.resolved_sha does not match local_path git HEAD`` and friends
    on legitimate local cached-benchmark runs.
    """
    import subprocess as sp

    monkeypatch.chdir(tmp_path)
    # Make the BenchFlow worktree marker live at tmp_path so ``_repo_root()``
    # resolves here (matching the production layout under .cache/datasets).
    (tmp_path / ".git").mkdir()

    cache_root = tmp_path / ".cache" / "datasets" / "benchflow-ai" / "skillsbench"
    tasks_dir = cache_root / "tasks"
    task_dir = tasks_dir / "task-a"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("[task]\n")
    (task_dir / "instruction.md").write_text("Solve it.\n")

    # Stand up a real git repo at the cache root so ``_git_stdout`` produces
    # the expected HEAD/remote/status results.
    env = {
        "GIT_AUTHOR_NAME": "BenchFlow Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "BenchFlow Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    sp.run(
        ["git", "init", "-b", "main"], cwd=cache_root, check=True, capture_output=True
    )
    sp.run(
        [
            "git",
            "remote",
            "add",
            "origin",
            "https://github.com/benchflow-ai/skillsbench.git",
        ],
        cwd=cache_root,
        check=True,
        capture_output=True,
    )
    sp.run(
        ["git", "add", "."],
        cwd=cache_root,
        check=True,
        capture_output=True,
        env={**env},
    )
    sp.run(
        ["git", "commit", "-m", "seed"],
        cwd=cache_root,
        check=True,
        capture_output=True,
        env={**env},
    )
    head = sp.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cache_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    provenance = task_download.infer_task_source_provenance(task_dir)

    assert provenance is not None
    assert provenance["type"] == "github"
    assert provenance["repo"] == "benchflow-ai/skillsbench"
    assert provenance["resolved_sha"] == head
    assert provenance["path"] == "tasks/task-a"
    assert provenance["local_path"] == str(task_dir.resolve())
    assert provenance["dirty"] is False
