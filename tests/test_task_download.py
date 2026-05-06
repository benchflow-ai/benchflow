"""Tests for benchmark task repository materialization."""

from pathlib import Path

from benchflow import task_download


def test_skillsbench_download_clones_main_branch(tmp_path, monkeypatch):
    """Guards PR #226: SkillsBench downloads must track GitHub main explicitly."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is True
        clone_dir = Path(cmd[-1])
        (clone_dir / "tasks" / "sample-task").mkdir(parents=True)

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    target = task_download.ensure_tasks("skillsbench")

    assert target == tmp_path / "benchmarks" / "skillsbench" / "tasks"
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
            str(tmp_path / "benchmarks" / "skillsbench" / "_clone"),
        ]
    ]


def test_unknown_benchmark_raises(tmp_path, monkeypatch):
    """Guards PR #237: unknown benchmark names raise ValueError."""
    monkeypatch.chdir(tmp_path)
    import pytest

    with pytest.raises(ValueError, match="Unknown benchmark"):
        task_download.ensure_tasks("nonexistent-benchmark")
