"""Tests for YAML job config loading."""

from pathlib import Path

import pytest

from benchflow.job import Job


@pytest.fixture
def native_yaml(tmp_path):
    """Create a benchflow-native YAML config."""
    tasks = tmp_path / "tasks" / "task-a"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"')
    (tasks / "instruction.md").write_text("Do something")

    config = tmp_path / "config.yaml"
    config.write_text("""
tasks_dir: tasks
jobs_dir: output
agent: pi-acp
model: claude-haiku-4-5-20251001
environment: daytona
concurrency: 32
max_retries: 1
prompts:
  - null
  - "Review your solution."
""")
    return config


@pytest.fixture
def harbor_yaml(tmp_path):
    """Create a Harbor-compatible YAML config."""
    tasks = tmp_path / "tasks" / "task-a"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"')
    (tasks / "instruction.md").write_text("Do something")

    config = tmp_path / "config.yaml"
    config.write_text("""
jobs_dir: output
n_attempts: 2
orchestrator:
  type: local
  n_concurrent_trials: 8
environment:
  type: daytona
  env:
    - ANTHROPIC_API_KEY=test-key
agents:
  - name: claude-agent-acp
    model_name: anthropic/claude-haiku-4-5-20251001
datasets:
  - path: tasks
""")
    return config


def test_from_native_yaml(native_yaml):
    """Test loading benchflow-native YAML."""
    job = Job.from_yaml(native_yaml)
    cfg = job._config

    assert cfg.agent == "pi-acp"
    assert cfg.model == "claude-haiku-4-5-20251001"
    assert cfg.environment == "daytona"
    assert cfg.concurrency == 32
    assert cfg.retry.max_retries == 1
    assert cfg.prompts == [None, "Review your solution."]
    assert job._tasks_dir.name == "tasks"


def test_from_harbor_yaml(harbor_yaml):
    """Test loading Harbor-compatible YAML."""
    job = Job.from_yaml(harbor_yaml)
    cfg = job._config

    assert cfg.agent == "claude-agent-acp"
    assert cfg.model == "claude-haiku-4-5-20251001"
    assert cfg.environment == "daytona"
    assert cfg.concurrency == 8
    assert cfg.retry.max_retries == 1  # n_attempts=2 → max_retries=1
    assert cfg.agent_env.get("ANTHROPIC_API_KEY") == "test-key"
    assert job._tasks_dir.name == "tasks"


def test_from_harbor_yaml_defaults(tmp_path):
    """Test Harbor YAML with minimal config."""
    tasks = tmp_path / "tasks" / "task-a"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"')

    config = tmp_path / "config.yaml"
    config.write_text("""
agents:
  - name: pi-acp
datasets:
  - path: tasks
""")

    job = Job.from_yaml(config)
    cfg = job._config
    assert cfg.agent == "pi-acp"
    assert cfg.environment == "docker"
    assert cfg.concurrency == 4


def test_native_yaml_with_skills_dir(tmp_path):
    """skills_dir is parsed from native YAML and passed through verbatim.

    Relative paths are stored as-is — they're resolved against CWD when
    the agent setup actually opens the directory, not at parse time. See #135.
    """
    tasks = tmp_path / "tasks" / "task-a"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"')
    skills = tmp_path / "my-skills"
    skills.mkdir()

    config = tmp_path / "config.yaml"
    config.write_text("""
tasks_dir: tasks
agent: claude-agent-acp
skills_dir: my-skills
""")

    job = Job.from_yaml(config)
    assert job._config.skills_dir == "my-skills"


def test_native_yaml_without_skills_dir(native_yaml):
    """Test that skills_dir defaults to None."""
    job = Job.from_yaml(native_yaml)
    assert job._config.skills_dir is None


def _make_tasks(tmp_path, names=("task-a", "task-b", "task-c")):
    """Create task dirs with task.toml files."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    for name in names:
        d = tasks_dir / name
        d.mkdir()
        (d / "task.toml").write_text('version = "1.0"')
    return tasks_dir


class TestNativeYamlNewFields:
    def test_exclude_parsed(self, tmp_path, monkeypatch):
        _make_tasks(tmp_path)
        config = tmp_path / "config.yaml"
        config.write_text("""
tasks_dir: tasks
exclude:
  - task-a
  - task-c
""")
        # Paths in YAML are CWD-relative (#135), so chdir before iterating.
        monkeypatch.chdir(tmp_path)
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
        assert job._config.agent_env == {
            "MY_KEY": "my-value",
            "OTHER_KEY": "other-value",
        }

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
        assert job._config.sandbox_user == "agent"


class TestPathResolution:
    """Regression tests for #135: YAML paths must resolve against CWD,
    not the config file's parent directory. CLI flag semantics and Harbor
    convention both expect CWD-relative resolution.
    """

    def test_native_yaml_paths_are_cwd_relative_not_config_relative(
        self, tmp_path, monkeypatch
    ):
        """Loading subdir/config.yaml from tmp_path must NOT prepend subdir/
        to relative paths inside the YAML."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        config = subdir / "config.yaml"
        config.write_text("""
tasks_dir: tasks
jobs_dir: jobs/my-run
skills_dir: skills
""")
        monkeypatch.chdir(tmp_path)

        job = Job.from_yaml(config)

        # Pre-#135-fix, these would have been "subdir/tasks", "subdir/jobs/my-run",
        # "subdir/skills" — i.e. config-parent-relative.
        assert job._tasks_dir == Path("tasks")
        assert job._jobs_dir == Path("jobs/my-run")
        assert job._config.skills_dir == "skills"

    def test_harbor_yaml_paths_are_cwd_relative_not_config_relative(
        self, tmp_path, monkeypatch
    ):
        """Harbor format: same expectation."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        config = subdir / "harbor.yaml"
        config.write_text("""
jobs_dir: jobs/my-run
skills_dir: skills
agents:
  - name: claude-agent-acp
datasets:
  - path: tasks
""")
        monkeypatch.chdir(tmp_path)

        job = Job.from_yaml(config)

        assert job._tasks_dir == Path("tasks")
        assert job._jobs_dir == Path("jobs/my-run")
        assert job._config.skills_dir == "skills"

    def test_absolute_yaml_paths_are_preserved(self, tmp_path, monkeypatch):
        """Absolute paths in YAML must be preserved regardless of CWD."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        abs_tasks = tmp_path / "abs_tasks"
        abs_jobs = tmp_path / "abs_jobs"
        abs_skills = tmp_path / "abs_skills"

        config = tmp_path / "config.yaml"
        config.write_text(f"""
tasks_dir: {abs_tasks}
jobs_dir: {abs_jobs}
skills_dir: {abs_skills}
""")
        monkeypatch.chdir(elsewhere)

        job = Job.from_yaml(config)

        assert job._tasks_dir == abs_tasks
        assert job._jobs_dir == abs_jobs
        assert job._config.skills_dir == str(abs_skills)

    def test_get_task_dirs_finds_cwd_relative_tasks(self, tmp_path, monkeypatch):
        """End-to-end: with config in subdir/, _get_task_dirs() should walk
        the CWD-relative `tasks/` tree, not `subdir/tasks/`."""
        # CWD-relative tasks (the ones that should be found)
        cwd_tasks = tmp_path / "tasks"
        for name in ("alpha", "beta"):
            d = cwd_tasks / name
            d.mkdir(parents=True)
            (d / "task.toml").write_text('version = "1.0"')

        # Config-parent-relative tasks (the trap — these should NOT be found,
        # because we're testing the new semantics)
        subdir = tmp_path / "subdir"
        trap_tasks = subdir / "tasks" / "should-not-be-found"
        trap_tasks.mkdir(parents=True)
        (trap_tasks / "task.toml").write_text('version = "1.0"')

        config = subdir / "config.yaml"
        config.write_text("tasks_dir: tasks\n")
        monkeypatch.chdir(tmp_path)

        job = Job.from_yaml(config)
        names = sorted(d.name for d in job._get_task_dirs())
        assert names == ["alpha", "beta"]
