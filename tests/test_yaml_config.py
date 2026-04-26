"""Tests for YAML job config loading."""

import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.job import Job
from benchflow.models import RunResult


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
sandbox_setup_timeout: 45
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
sandbox_setup_timeout: 75
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
    assert cfg.sandbox_setup_timeout == 45
    assert cfg.prompts == [None, "Review your solution."]
    assert job._tasks_dir == Path("tasks")
    assert job._jobs_dir == Path("output")


def test_from_harbor_yaml(harbor_yaml):
    """Test loading Harbor-compatible YAML."""
    job = Job.from_yaml(harbor_yaml)
    cfg = job._config

    assert cfg.agent == "claude-agent-acp"
    assert cfg.model == "anthropic/claude-haiku-4-5-20251001"
    assert cfg.environment == "daytona"
    assert cfg.concurrency == 8
    assert cfg.retry.max_retries == 1  # n_attempts=2 → max_retries=1
    assert cfg.agent_env.get("ANTHROPIC_API_KEY") == "test-key"
    assert cfg.sandbox_setup_timeout == 75
    assert job._tasks_dir == Path("tasks")
    assert job._jobs_dir == Path("output")


def test_harbor_yaml_preserves_provider_prefix(tmp_path):
    """Provider prefix must survive _from_harbor_yaml for downstream resolution."""
    tasks = tmp_path / "tasks" / "task-a"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"')

    config = tmp_path / "config.yaml"
    config.write_text("""
agents:
  - name: pi-acp
    model_name: vllm/Qwen/Qwen3.5-35B-A3B
datasets:
  - path: tasks
""")

    job = Job.from_yaml(config)
    assert job._config.model == "vllm/Qwen/Qwen3.5-35B-A3B"


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
    assert cfg.sandbox_setup_timeout == 120
    assert job._tasks_dir == Path("tasks")
    assert job._jobs_dir == Path("jobs")


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
    assert job._config.skills_dir == "my-skills"


def test_native_yaml_paths_are_cwd_relative(tmp_path):
    """Relative YAML paths are not rebased to the config file directory."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "config.yaml"
    config.write_text("""
tasks_dir: tasks
jobs_dir: jobs/my-run
skills_dir: skills
""")

    job = Job.from_yaml(config)
    assert job._tasks_dir == Path("tasks")
    assert job._jobs_dir == Path("jobs/my-run")
    assert job._config.skills_dir == "skills"


def test_harbor_yaml_paths_are_cwd_relative(tmp_path):
    """Harbor relative paths match CLI and SDK path behavior."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "config.yaml"
    config.write_text("""
jobs_dir: jobs/my-run
skills_dir: skills
agents:
  - name: pi-acp
datasets:
  - path: tasks
""")

    job = Job.from_yaml(config)
    assert job._tasks_dir == Path("tasks")
    assert job._jobs_dir == Path("jobs/my-run")
    assert job._config.skills_dir == "skills"


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
        monkeypatch.chdir(tmp_path)
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
        assert job._config.sandbox_setup_timeout == 120


# ── Scenes parsing (issue #4) ──


def _make_task_dir(tmp_path: Path, name: str = "task-a") -> Path:
    """Create a tasks/<name> directory with task.toml."""
    task_dir = tmp_path / "tasks" / name
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('version = "1.0"')
    return task_dir


def test_harbor_yaml_parses_issue_4_repro(tmp_path):
    """Guards the fix from commit 22f52b4 against the silent-scenes-drop regression in issue #4 — mirrors the literal repro YAML."""
    _make_task_dir(tmp_path)

    config = tmp_path / "config.yaml"
    config.write_text("""
agent: pi-acp
model_name: vllm/Qwen/Qwen3.6-27B
datasets:
  - path: tasks
scenes:
  - name: build-skill
    roles:
      - name: builder
        agent: pi-acp
        model: vllm/Qwen/Qwen3.6-27B
    turns:
      - role: builder
        prompt: "build the skill ..."
  - name: answer
    roles:
      - name: solver
        agent: pi-acp
        model: vllm/Qwen/Qwen3.6-27B
    turns:
      - role: solver
""")

    job = Job.from_yaml(config)
    scenes = job._config.scenes

    assert scenes is not None
    assert len(scenes) == 2
    assert scenes[0].name == "build-skill"
    assert scenes[0].roles[0].agent == "pi-acp"
    assert scenes[0].roles[0].model == "vllm/Qwen/Qwen3.6-27B"
    assert scenes[0].turns[0].prompt == "build the skill ..."
    assert scenes[1].name == "answer"
    assert scenes[1].turns[0].prompt is None


async def test_harbor_yaml_branch_passes_scenes_to_trial(tmp_path, monkeypatch):
    """Guards the fix from commit 22f52b4 against the bug where Harbor scenes never reached Trial — exercises the path that would have failed pre-fix (issue #4)."""
    task_dir = _make_task_dir(tmp_path)

    config = tmp_path / "config.yaml"
    config.write_text("""
datasets:
  - path: tasks
scenes:
  - name: build-skill
    roles:
      - name: builder
        agent: claude-agent-acp
        model: claude-haiku-4-5-20251001
    turns:
      - role: builder
        prompt: "build it"
  - name: answer
    roles:
      - name: solver
        agent: claude-agent-acp
        model: claude-haiku-4-5-20251001
    turns:
      - role: solver
""")

    job = Job.from_yaml(config)

    seen: dict = {}

    async def capturing_create(trial_config):
        seen["config"] = trial_config
        trial = AsyncMock()
        trial.run = AsyncMock(
            return_value=RunResult(task_name=task_dir.name, rewards={"reward": 1.0})
        )
        return trial

    monkeypatch.setattr("benchflow.trial.Trial.create", capturing_create)

    result = await job._run_single_task(task_dir, job._config)

    assert result.rewards == {"reward": 1.0}
    captured = seen["config"]
    assert captured.scenes == job._config.scenes
    assert [s.name for s in captured.scenes] == ["build-skill", "answer"]
    assert len(captured.scenes[0].roles) == 1
    assert len(captured.scenes[0].turns) == 1
    assert len(captured.scenes[1].turns) == 1


def test_harbor_yaml_without_scenes_unchanged(harbor_yaml):
    """Guards commit 22f52b4: harbor YAML without scenes: still produces scenes=None (legacy single-turn path, issue #4)."""
    job = Job.from_yaml(harbor_yaml)
    assert job._config.scenes is None


def test_native_yaml_parses_top_level_scenes(tmp_path):
    """Guards commit 22f52b4: native YAML scenes: parses just like Harbor (decision 1A, issue #4)."""
    _make_task_dir(tmp_path)

    config = tmp_path / "config.yaml"
    config.write_text("""
tasks_dir: tasks
scenes:
  - name: solve
    roles:
      - name: solver
        agent: claude-agent-acp
        model: claude-haiku-4-5-20251001
    turns:
      - role: solver
        prompt: "solve it"
""")

    job = Job.from_yaml(config)
    scenes = job._config.scenes

    assert scenes is not None
    assert len(scenes) == 1
    assert scenes[0].name == "solve"
    assert scenes[0].roles[0].agent == "claude-agent-acp"
    assert scenes[0].turns[0].prompt == "solve it"


def test_native_yaml_without_scenes_unchanged(native_yaml):
    """Guards commit 22f52b4: native YAML without scenes: still produces scenes=None (decision 1A, issue #4)."""
    job = Job.from_yaml(native_yaml)
    assert job._config.scenes is None


def test_harbor_yaml_top_level_singular_agent_and_model_name(tmp_path):
    """Guards commit 22f52b4: Harbor accepts singular top-level agent: + model_name: when agents[] is absent (TODO 1, issue #4)."""
    _make_task_dir(tmp_path)

    config = tmp_path / "config.yaml"
    config.write_text("""
agent: pi-acp
model_name: vllm/Qwen/Qwen3.6-27B
datasets:
  - path: tasks
""")

    job = Job.from_yaml(config)
    assert job._config.agent == "pi-acp"
    assert job._config.model == "vllm/Qwen/Qwen3.6-27B"


def test_harbor_yaml_empty_scenes_warns(tmp_path, caplog):
    """Guards commit 22f52b4: scenes: [] logs a warning and falls through to legacy (TODO 2, issue #4)."""
    _make_task_dir(tmp_path)

    config = tmp_path / "config.yaml"
    config.write_text("""
agents:
  - name: claude-agent-acp
    model_name: claude-haiku-4-5-20251001
datasets:
  - path: tasks
scenes: []
""")

    with caplog.at_level(logging.WARNING, logger="benchflow.job"):
        job = Job.from_yaml(config)

    assert job._config.scenes is None
    assert any(
        "scenes" in msg and "empty" in msg for msg in caplog.messages
    ), f"Expected warning about empty scenes; got: {caplog.messages!r}"


def test_harbor_yaml_summary_fields_match_first_scene_role(tmp_path):
    """Guards commit 22f52b4: when scenes is non-empty, JobConfig.agent/model derive from scenes[0].roles[0] so summary.json reflects what runs (TODO 3, issue #4)."""
    _make_task_dir(tmp_path)

    config = tmp_path / "config.yaml"
    config.write_text("""
agents:
  - name: claude-agent-acp
    model_name: claude-haiku-4-5-20251001
datasets:
  - path: tasks
scenes:
  - name: solve
    roles:
      - name: solver
        agent: pi-acp
        model: vllm/Qwen/Qwen3.6-27B
    turns:
      - role: solver
        prompt: "solve it"
""")

    job = Job.from_yaml(config)
    assert job._config.agent == "pi-acp"
    assert job._config.model == "vllm/Qwen/Qwen3.6-27B"
