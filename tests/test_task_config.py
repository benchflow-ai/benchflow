"""Tests for task.toml parsing into TaskConfig."""

import pytest

from benchflow.task.config import (
    MultiStepRewardStrategy,
    NetworkMode,
    TaskConfig,
    TaskOS,
    VerifierEnvironmentMode,
)


def test_task_config_reads_expected_skills_from_verifier_memory():
    """Guards the ENG-125 follow-up on v0.5-integration@ffef85d."""
    cfg = TaskConfig.model_validate_toml(
        'version = "1.0"\n[verifier.memory]\nexpected_skills = ["git-bisect", "rg"]\n'
    )
    assert cfg.expected_skills == ["git-bisect", "rg"]


def test_task_config_public_toml_dump_omits_expected_skills_fixture():
    """Guards hidden Memory-space fixtures from public task serialization."""
    cfg = TaskConfig.model_validate_toml(
        'version = "1.0"\n[verifier.memory]\nexpected_skills = ["git-bisect", "rg"]\n'
    )

    dumped = cfg.model_dump_toml()

    assert cfg.expected_skills == ["git-bisect", "rg"]
    assert "expected_skills" not in dumped
    assert "git-bisect" not in dumped
    assert "rg" not in dumped


def test_task_config_preserves_empty_expected_skills_fixture():
    """Guards the ENG-125 follow-up on v0.5-integration@ffef85d."""
    cfg = TaskConfig.model_validate_toml(
        'version = "1.0"\n[verifier.memory]\nexpected_skills = []\n'
    )
    assert cfg.expected_skills == []


def test_task_config_absent_expected_skills_fixture_is_none():
    """Guards the ENG-125 follow-up on v0.5-integration@ffef85d."""
    cfg = TaskConfig.model_validate_toml('version = "1.0"\n')
    assert cfg.expected_skills is None


def test_task_config_rejects_malformed_expected_skills():
    """Guards the ENG-125 follow-up on v0.5-integration@ffef85d."""
    with pytest.raises(ValueError, match="expected_skills"):
        TaskConfig.model_validate_toml(
            'version = "1.0"\n[verifier.memory]\nexpected_skills = "git-bisect"\n'
        )


def test_task_config_accepts_current_harbor_task_toml_surface():
    """Guards commit 67378ddd's 2026-06-04 parity pass against schema shrinkage."""
    cfg = TaskConfig.model_validate_toml(
        """
schema_version = "1.3"
source = "harbor/parity"
multi_step_reward_strategy = "final"
artifacts = [{ source = "/logs/artifacts", destination = "artifacts" }]

[task]
name = "benchflow/harbor-parity"
description = "Exercises Harbor-compatible task.toml fields"
authors = [{ name = "BenchFlow", email = "benchflow@example.com" }]
keywords = ["parity", "task-md"]

[metadata]
category = "schema"

[metadata.custom]
kept = true

[agent]
timeout_sec = 120.0
user = "agent"
network_mode = "allowlist"
allowed_hosts = ["api.example.com"]

[verifier]
timeout_sec = 60.0
env = { JUDGE_API_KEY = "${JUDGE_API_KEY:-test}" }
user = "root"
network_mode = "public"
environment_mode = "separate"
pytest_plugins = ["pytest_playwright"]

[verifier.hardening]
cleanup_conftests = false

[verifier.environment]
docker_image = "ghcr.io/example/grader:latest"
cpus = 2
memory_mb = 1024
network_mode = "no-network"

[environment]
network_mode = "allowlist"
allowed_hosts = ["datasets.example.com"]
build_timeout_sec = 600.0
docker_image = "ghcr.io/example/task:latest"
os = "linux"
cpus = 4
memory_mb = 4096
storage_mb = 8192
gpus = 1
gpu_types = ["T4", "A100"]
env = { DATASET = "${DATASET:-sample}" }
skills_dir = "/skills"
workdir = "/workspace"

[environment.tpu]
type = "v6e"
topology = "2x4"

[environment.healthcheck]
command = "python -m app.healthcheck"
interval_sec = 2.0
timeout_sec = 10.0
retries = 5

[solution]
env = { SOLUTION_MODE = "oracle" }

[[steps]]
name = "scaffold"
min_reward = 0.5
artifacts = [{ source = "/app/scaffold.txt" }]

[steps.agent]
timeout_sec = 30.0

[steps.verifier]
timeout_sec = 15.0
env = { STEP = "scaffold" }
"""
    )

    assert cfg.schema_version == "1.3"
    assert cfg.task is not None
    assert cfg.task.name == "benchflow/harbor-parity"
    assert cfg.metadata["custom"]["kept"] is True
    assert cfg.agent.network_mode == NetworkMode.ALLOWLIST
    assert cfg.agent.allowed_hosts == ["api.example.com"]
    assert cfg.verifier.environment_mode == VerifierEnvironmentMode.SEPARATE
    assert cfg.verifier.hardening.cleanup_conftests is False
    assert cfg.verifier.environment is not None
    assert cfg.verifier.environment.allow_internet is False
    assert cfg.environment.network_mode == NetworkMode.ALLOWLIST
    assert cfg.environment.os == TaskOS.LINUX
    assert cfg.environment.tpu is not None
    assert cfg.environment.tpu.chip_count == 8
    assert cfg.environment.healthcheck is not None
    assert cfg.environment.healthcheck.retries == 5
    assert cfg.multi_step_reward_strategy == MultiStepRewardStrategy.FINAL
    assert cfg.artifacts[0].source == "/logs/artifacts"
    assert cfg.steps is not None
    assert cfg.steps[0].name == "scaffold"
    assert cfg.steps[0].artifacts[0].source == "/app/scaffold.txt"


def test_task_config_rejects_unknown_harbor_fields():
    """Guards commit 67378ddd's 2026-06-04 parity pass against lossy parsing."""
    with pytest.raises(ValueError, match="unknown_harbor_field"):
        TaskConfig.model_validate_toml(
            'schema_version = "1.3"\n[environment]\nunknown_harbor_field = true\n'
        )


def test_task_config_accepts_native_oracle_alias():
    """Guards commit 67378ddd's task.md vocabulary while preserving internals."""
    cfg = TaskConfig.model_validate({"oracle": {"env": {"MODE": "gold"}}})

    assert cfg.solution.env == {"MODE": "gold"}
    dumped = cfg.model_dump_toml()
    assert "[oracle.env]" in dumped
    assert "[solution.env]" not in dumped
