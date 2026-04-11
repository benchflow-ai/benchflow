"""Tests for the exclude_tasks filter in Job._get_task_dirs."""

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
