"""Tests for YAML job config loading."""

from pathlib import Path

import pytest

from benchflow.evaluation import Evaluation
from benchflow.skill_policy import (
    SKILL_MODE_NO_SKILL,
    SKILL_MODE_SELF_GEN,
    SKILL_MODE_WITH_SKILL,
)


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
def legacy_yaml(tmp_path):
    """Create a legacy-format YAML config (agents + datasets style)."""
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
    job = Evaluation.from_yaml(native_yaml)
    cfg = job._config

    assert cfg.agent == "pi-acp"
    assert cfg.model == "claude-haiku-4-5-20251001"
    assert cfg.environment == "daytona"
    assert cfg.concurrency == 32
    assert cfg.retry.max_retries == 1
    assert cfg.sandbox_setup_timeout == 45
    assert cfg.prompts == [None, "Review your solution."]
    assert cfg.skill_mode == SKILL_MODE_NO_SKILL
    assert job._tasks_dir == Path("tasks")
    assert job._jobs_dir == Path("output")


def test_native_yaml_normalizes_agent_alias_and_root_sandbox_user(tmp_path):
    """Guards ENG-91 P0 dogfood config-boundary regression."""
    tasks = tmp_path / "tasks" / "task-a"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"')
    (tasks / "instruction.md").write_text("Do something")

    config = tmp_path / "config.yaml"
    config.write_text("""
tasks_dir: tasks
agent: codex
sandbox_user: none
""")

    job = Evaluation.from_yaml(config)

    assert job._config.agent == "codex-acp"
    assert job._config.sandbox_user is None


def test_rollout_yaml_loader_normalizes_alias_and_root_sandbox_user():
    """Guards ENG-91 P0 dogfood rollout YAML-loader regression."""
    from benchflow._utils.yaml_loader import rollout_config_from_dict

    cfg = rollout_config_from_dict(
        {
            "task_dir": "tests/examples/hello-world-task",
            "agent": "codex",
            "sandbox_user": "null",
        }
    )

    assert cfg.primary_agent == "codex-acp"
    assert cfg.sandbox_user is None


def test_native_yaml_zero_agent_idle_timeout_disables_watchdog(tmp_path):
    """Guards v0.5-idle-timeout@219906c against config/CLI semantic drift."""
    tasks = tmp_path / "tasks" / "task-a"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"')

    config = tmp_path / "config.yaml"
    config.write_text("""
tasks_dir: tasks
agent: gemini
agent_idle_timeout: 0
""")

    job = Evaluation.from_yaml(config)

    assert job._config.agent_idle_timeout is None


def test_legacy_yaml_zero_agent_idle_timeout_disables_watchdog(tmp_path):
    """Guards v0.5-idle-timeout@219906c against legacy config semantic drift."""
    tasks = tmp_path / "tasks" / "task-a"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"')

    config = tmp_path / "config.yaml"
    config.write_text("""
n_attempts: 1
agent_idle_timeout_sec: 0
orchestrator:
  n_concurrent_trials: 1
environment:
  type: docker
agents:
  - name: gemini
    model_name: gemini-3.1-flash-lite-preview
datasets:
  - path: tasks
""")

    job = Evaluation.from_yaml(config)

    assert job._config.agent_idle_timeout is None


def test_rollout_yaml_zero_agent_idle_timeout_disables_watchdog():
    """Guards v0.5-idle-timeout@219906c for direct RolloutConfig YAML loading."""
    from benchflow._utils.yaml_loader import rollout_config_from_dict

    cfg = rollout_config_from_dict(
        {
            "task_dir": "tests/examples/hello-world-task",
            "agent": "gemini",
            "agent_idle_timeout": 0,
        }
    )

    assert cfg.agent_idle_timeout is None


def test_native_yaml_rejects_bool_agent_idle_timeout(tmp_path):
    """Guards v0.5-idle-timeout@1566fed against bool-to-int coercion."""
    tasks = tmp_path / "tasks" / "task-a"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"')

    config = tmp_path / "config.yaml"
    config.write_text("""
tasks_dir: tasks
agent: gemini
agent_idle_timeout: true
""")

    with pytest.raises(ValueError, match="integer seconds"):
        Evaluation.from_yaml(config)


def test_legacy_yaml_rejects_fractional_agent_idle_timeout(tmp_path):
    """Guards v0.5-idle-timeout@1566fed against float truncation."""
    tasks = tmp_path / "tasks" / "task-a"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"')

    config = tmp_path / "config.yaml"
    config.write_text("""
n_attempts: 1
agent_idle_timeout_sec: 1.5
orchestrator:
  n_concurrent_trials: 1
environment:
  type: docker
agents:
  - name: gemini
    model_name: gemini-3.1-flash-lite-preview
datasets:
  - path: tasks
""")

    with pytest.raises(ValueError, match="integer seconds"):
        Evaluation.from_yaml(config)


def test_rollout_yaml_rejects_integral_float_agent_idle_timeout_contract():
    """Guards v0.5-idle-timeout@1566fed; integer seconds reject floats like 1.0."""
    from benchflow._utils.yaml_loader import rollout_config_from_dict

    with pytest.raises(ValueError, match="integer seconds"):
        rollout_config_from_dict(
            {
                "task_dir": "tests/examples/hello-world-task",
                "agent": "gemini",
                "agent_idle_timeout": 1.0,
            }
        )


def test_rollout_yaml_accepts_numeric_string_agent_idle_timeout():
    """Guards v0.5-idle-timeout@1566fed numeric-string compatibility."""
    from benchflow._utils.yaml_loader import rollout_config_from_dict

    cfg = rollout_config_from_dict(
        {
            "task_dir": "tests/examples/hello-world-task",
            "agent": "gemini",
            "agent_idle_timeout": "600",
        }
    )

    assert cfg.agent_idle_timeout == 600


def test_from_legacy_yaml(legacy_yaml):
    """Test loading legacy-format YAML."""
    job = Evaluation.from_yaml(legacy_yaml)
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


def test_legacy_yaml_preserves_provider_prefix(tmp_path):
    """Provider prefix must survive _from_legacy_yaml for downstream resolution."""
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

    job = Evaluation.from_yaml(config)
    assert job._config.model == "vllm/Qwen/Qwen3.5-35B-A3B"


def test_from_legacy_yaml_defaults(tmp_path):
    """Test legacy YAML with minimal config.

    Non-default agents must declare a model (either via ``model_name`` here or
    via ``AgentConfig.default_model``) — #343 stopped silent fallback to a
    Claude default for cross-provider agents like ``pi-acp``.
    """
    tasks = tmp_path / "tasks" / "task-a"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"')

    config = tmp_path / "config.yaml"
    config.write_text("""
agents:
  - name: pi-acp
    model_name: claude-haiku-4-5-20251001
datasets:
  - path: tasks
""")

    job = Evaluation.from_yaml(config)
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
skill_mode: with-skill
skills_dir: my-skills
""")

    job = Evaluation.from_yaml(config)
    assert job._config.skill_mode == SKILL_MODE_WITH_SKILL
    assert job._config.skills_dir == "my-skills"


def test_native_yaml_with_task_skill_mode(tmp_path):
    """Guards PR #586 so native YAML can explicitly enable task skills."""
    tasks = tmp_path / "tasks" / "task-a"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"')

    config = tmp_path / "config.yaml"
    config.write_text("""
tasks_dir: tasks
agent: claude-agent-acp
skill_mode: with-skill
""")

    job = Evaluation.from_yaml(config)
    assert job._config.skill_mode == SKILL_MODE_WITH_SKILL


def test_native_yaml_paths_are_cwd_relative(tmp_path):
    """Relative YAML paths are not rebased to the config file directory."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "config.yaml"
    config.write_text("""
tasks_dir: tasks
jobs_dir: jobs/my-run
skill_mode: with-skill
skills_dir: skills
""")

    job = Evaluation.from_yaml(config)
    assert job._tasks_dir == Path("tasks")
    assert job._jobs_dir == Path("jobs/my-run")
    assert job._config.skill_mode == SKILL_MODE_WITH_SKILL
    assert job._config.skills_dir == "skills"


def test_legacy_yaml_paths_are_cwd_relative(tmp_path):
    """Legacy relative paths match CLI and SDK path behavior."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "config.yaml"
    config.write_text("""
jobs_dir: jobs/my-run
skill_mode: with-skill
skills_dir: skills
agents:
  - name: pi-acp
    model_name: claude-haiku-4-5-20251001
datasets:
  - path: tasks
""")

    job = Evaluation.from_yaml(config)
    assert job._tasks_dir == Path("tasks")
    assert job._jobs_dir == Path("jobs/my-run")
    assert job._config.skill_mode == SKILL_MODE_WITH_SKILL
    assert job._config.skills_dir == "skills"


def test_native_yaml_without_skills_dir(native_yaml):
    """Test that skills_dir defaults to None."""
    job = Evaluation.from_yaml(native_yaml)
    assert job._config.skills_dir is None
    assert job._config.skill_mode == SKILL_MODE_NO_SKILL


def test_native_yaml_with_self_gen_skill_mode(tmp_path):
    """Self-gen job config is parsed for batch runs."""
    tasks = tmp_path / "tasks" / "task-a"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"')
    skills = tmp_path / "skills"
    skills.mkdir()

    config = tmp_path / "config.yaml"
    config.write_text("""
tasks_dir: tasks
agent: claude-agent-acp
skill_mode: self-gen
skill_creator_dir: skills
self_gen_no_internet: true
""")

    job = Evaluation.from_yaml(config)
    assert job._config.skill_mode == SKILL_MODE_SELF_GEN
    assert job._config.skill_creator_dir == "skills"
    assert job._config.self_gen_no_internet is True


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
        job = Evaluation.from_yaml(config)
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
        job = Evaluation.from_yaml(config)
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
        job = Evaluation.from_yaml(config)
        assert job._config.sandbox_user == "testuser"

    def test_defaults_when_omitted(self, tmp_path):
        _make_tasks(tmp_path)
        config = tmp_path / "config.yaml"
        config.write_text("tasks_dir: tasks\n")
        job = Evaluation.from_yaml(config)
        assert job._config.exclude_tasks == set()
        assert job._config.agent_env == {}
        assert job._config.sandbox_user == "agent"
        assert job._config.sandbox_setup_timeout == 120


def test_legacy_yaml_maps_include_exclude_filters(tmp_path):
    """Guards #500: legacy YAML must not silently drop include/exclude filters."""
    tasks = tmp_path / "tasks"
    for name in ("alpha", "beta", "gamma"):
        td = tasks / name
        td.mkdir(parents=True)
        (td / "task.toml").write_text('version = "1.0"')

    config = tmp_path / "config.yaml"
    config.write_text(
        """
agents:
  - name: claude-agent-acp
    model_name: anthropic/claude-haiku-4-5-20251001
datasets:
  - path: tasks
include:
  - alpha
  - beta
exclude:
  - gamma
"""
    )

    job = Evaluation.from_yaml(config)
    cfg = job._config
    assert cfg.include_tasks == {"alpha", "beta"}
    assert cfg.exclude_tasks == {"gamma"}


def test_legacy_yaml_accepts_plural_include_exclude(tmp_path):
    """Plural spellings ('includes'/'excludes') must also map (#500)."""
    tasks = tmp_path / "tasks"
    for name in ("alpha", "beta"):
        td = tasks / name
        td.mkdir(parents=True)
        (td / "task.toml").write_text('version = "1.0"')

    config = tmp_path / "config.yaml"
    config.write_text(
        """
agents:
  - name: claude-agent-acp
    model_name: anthropic/claude-haiku-4-5-20251001
datasets:
  - path: tasks
includes:
  - alpha
excludes:
  - beta
"""
    )

    job = Evaluation.from_yaml(config)
    cfg = job._config
    assert cfg.include_tasks == {"alpha"}
    assert cfg.exclude_tasks == {"beta"}
