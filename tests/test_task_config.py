"""Tests for native task config models."""

from pathlib import Path

from benchflow.tasks import Task, load_task_config


def test_load_task_config_parses_core_sections(tmp_path: Path) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text(
        """
[agent]
timeout_sec = 42

[verifier]
timeout_sec = 11
env = { FOO = "bar" }
pytest_plugins = ["pytester"]

[environment]
cpus = 2
memory_mb = 2048
storage_mb = 4096
allow_internet = false
env = { A = "B" }
"""
    )

    config = load_task_config(task_toml)

    assert config.agent.timeout_sec == 42
    assert config.verifier.timeout_sec == 11
    assert config.verifier.env == {"FOO": "bar"}
    assert config.verifier.pytest_plugins == ["pytester"]
    assert config.environment.cpus == 2
    assert config.environment.memory_mb == 2048
    assert config.environment.storage_mb == 4096
    assert config.environment.allow_internet is False
    assert config.environment.env == {"A": "B"}


def test_environment_config_to_sandbox_spec(tmp_path: Path) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text("[environment]\ncpus = 4\nallow_internet = false\n")

    spec = load_task_config(task_toml).environment.to_sandbox_spec("docker")

    assert spec.provider == "docker"
    assert spec.cpus == 4
    assert spec.allow_internet is False


def test_task_from_dir_exposes_paths(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "task.toml").write_text("[agent]\ntimeout_sec = 1\n")

    task = Task.from_dir(task_dir)

    assert task.task_dir == task_dir
    assert task.paths.instruction == task_dir / "instruction.md"
    assert task.paths.dockerfile == task_dir / "environment" / "Dockerfile"
