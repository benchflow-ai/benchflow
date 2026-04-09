"""Tests for exclude_tasks, agent_env, and sandbox_user in Job config."""

from pathlib import Path

import pytest

from benchflow.job import Job, JobConfig


def _make_tasks(tmp_path, names=("task-a", "task-b", "task-c")):
    """Create task dirs with task.toml files."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    for name in names:
        d = tasks_dir / name
        d.mkdir()
        (d / "task.toml").write_text('version = "1.0"')
    return tasks_dir


class TestExcludeTasksFilter:
    def test_no_exclusions(self, tmp_path):
        tasks_dir = _make_tasks(tmp_path)
        job = Job(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs")
        dirs = job._get_task_dirs()
        assert [d.name for d in dirs] == ["task-a", "task-b", "task-c"]

    def test_exclude_one(self, tmp_path):
        tasks_dir = _make_tasks(tmp_path)
        cfg = JobConfig(exclude_tasks={"task-b"})
        job = Job(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=cfg)
        dirs = job._get_task_dirs()
        assert [d.name for d in dirs] == ["task-a", "task-c"]

    def test_exclude_multiple(self, tmp_path):
        tasks_dir = _make_tasks(tmp_path)
        cfg = JobConfig(exclude_tasks={"task-a", "task-c"})
        job = Job(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=cfg)
        dirs = job._get_task_dirs()
        assert [d.name for d in dirs] == ["task-b"]

    def test_exclude_nonexistent_is_harmless(self, tmp_path):
        tasks_dir = _make_tasks(tmp_path)
        cfg = JobConfig(exclude_tasks={"no-such-task"})
        job = Job(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=cfg)
        dirs = job._get_task_dirs()
        assert len(dirs) == 3


class TestNativeYamlNewFields:
    def test_exclude_parsed(self, tmp_path):
        tasks_dir = _make_tasks(tmp_path)
        config = tmp_path / "config.yaml"
        config.write_text("""
tasks_dir: tasks
exclude:
  - task-a
  - task-c
""")
        job = Job.from_yaml(config)
        assert job._config.exclude_tasks == {"task-a", "task-c"}
        dirs = job._get_task_dirs()
        assert [d.name for d in dirs] == ["task-b"]

    def test_agent_env_parsed(self, tmp_path):
        _make_tasks(tmp_path)
        config = tmp_path / "config.yaml"
        config.write_text("""
tasks_dir: tasks
agent_env:
  MY_KEY: my-value
  OTHER_KEY: other-value
""")
        job = Job.from_yaml(config)
        assert job._config.agent_env == {"MY_KEY": "my-value", "OTHER_KEY": "other-value"}

    def test_sandbox_user_parsed(self, tmp_path):
        _make_tasks(tmp_path)
        config = tmp_path / "config.yaml"
        config.write_text("""
tasks_dir: tasks
sandbox_user: testuser
""")
        job = Job.from_yaml(config)
        assert job._config.sandbox_user == "testuser"

    def test_defaults_when_omitted(self, tmp_path):
        _make_tasks(tmp_path)
        config = tmp_path / "config.yaml"
        config.write_text("tasks_dir: tasks\n")
        job = Job.from_yaml(config)
        assert job._config.exclude_tasks == set()
        assert job._config.agent_env == {}
        assert job._config.sandbox_user is None
