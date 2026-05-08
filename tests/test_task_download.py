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

    assert target == tmp_path / ".ref" / "skillsbench" / "tasks"
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
            str(tmp_path / ".ref" / "skillsbench" / "_clone"),
        ]
    ]


def test_download_without_ref_omits_branch_flag(tmp_path, monkeypatch):
    """Guards PR #226: existing task repos without refs keep their clone shape."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is True
        clone_dir = Path(cmd[-1])
        clone_dir.mkdir(parents=True)
        (clone_dir / ".git").mkdir()
        (clone_dir / "README.md").write_text("tasks\n")

    monkeypatch.setattr(task_download.subprocess, "run", fake_run)

    target = task_download.ensure_tasks("terminal-bench-2")

    assert target == tmp_path / ".ref" / "terminal-bench-2"
    assert target.exists()
    assert "--branch" not in calls[0]
    assert calls == [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/harbor-framework/terminal-bench-2.git",
            str(tmp_path / ".ref" / "_clone"),
        ]
    ]
