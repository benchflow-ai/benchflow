"""Tests for YAML job config loading."""

import pytest
from pathlib import Path

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
    """Test that skills_dir is parsed from native YAML."""
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
    assert job._config.skills_dir == str(tmp_path / "my-skills")


def test_native_yaml_without_skills_dir(native_yaml):
    """Test that skills_dir defaults to None."""
    job = Job.from_yaml(native_yaml)
    assert job._config.skills_dir is None
