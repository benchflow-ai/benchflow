"""Tests for benchmark task repository materialization."""

from pathlib import Path

from benchflow._utils import benchmark_repos as task_download
from benchflow._utils.benchmark_repos import Source, resolve_source


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


def test_programbench_alias_resolves_to_benchmarks_repo(tmp_path, monkeypatch):
    """Guards TB2-removal + source-pattern migration: programbench alias resolves
    to benchflow-ai/benchmarks dataset repo."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is True
        clone_dir = Path(cmd[-1])
        clone_dir.mkdir(parents=True)
        (clone_dir / ".git").mkdir()
        (clone_dir / "datasets" / "programbench" / "tasks" / "sample").mkdir(
            parents=True
        )

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    target = task_download.ensure_tasks("programbench")

    assert (
        target
        == tmp_path
        / ".cache"
        / "datasets"
        / "benchflow-ai"
        / "benchmarks"
        / "datasets"
        / "programbench"
        / "tasks"
    )
    assert target.exists()
    assert "--branch" in calls[0]
    assert "main" in calls[0]


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
