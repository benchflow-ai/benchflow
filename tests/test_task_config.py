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


def test_verifier_timeout_sec_omitted_inherits_default_budget():
    """Omitting [verifier].timeout_sec must yield the documented 600s default."""
    cfg = TaskConfig.model_validate_toml('version = "1.0"\n')
    assert cfg.verifier.timeout_sec == 600.0


def test_verifier_timeout_sec_explicit_value_is_used_exactly():
    cfg = TaskConfig.model_validate_toml(
        'version = "1.0"\n[verifier]\ntimeout_sec = 90.5\n'
    )
    assert cfg.verifier.timeout_sec == 90.5


@pytest.mark.parametrize("value", ["0", "0.0", "-30", "nan", "inf"])
def test_verifier_timeout_sec_rejects_unusable_budgets(value):
    """Zero/negative/non-finite budgets fail at parse time instead of
    producing an instant verifier timeout at execution time."""
    with pytest.raises(ValueError, match=r"verifier\.timeout_sec"):
        TaskConfig.model_validate_toml(
            f'version = "1.0"\n[verifier]\ntimeout_sec = {value}\n'
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


def test_task_config_accepts_current_skillsbench_task_toml_surface():
    """Guards SkillsBench main a8eefb4 against parser-only run failures."""
    cfg = TaskConfig.model_validate_toml(
        """
version = "1.0"

[reward]
reward = "(P0_passed/13 * 0.50) + (P1_passed/7 * 0.35)"

[verifier]
timeout_sec = 240.0

[agent]
timeout_sec = 1800.0

[environment]
image = "python:3.12-slim"
build_timeout_sec = 600.0
cpus = 4
memory = "4G"
storage = "8G"
allow_internet = true
bugswarm_image_tag = "google-auto-101506036"

[solution]
timeout_sec = 1800.0

[solution.env]
REPO_ID = "google/auto"
"""
    )

    assert cfg.schema_version == "1.0"
    assert cfg.reward["reward"].startswith("(P0_passed")
    assert cfg.environment.docker_image == "python:3.12-slim"
    assert cfg.environment.bugswarm_image_tag == "google-auto-101506036"
    assert cfg.environment.memory_mb == 4096
    assert cfg.environment.storage_mb == 8192
    assert cfg.solution.timeout_sec == 1800.0


def test_task_config_accepts_mcp_tool_filter_surface():
    """Guards cd8e250b MCP Atlas adapter work against MCP schema shrinkage."""
    cfg = TaskConfig.model_validate_toml(
        """
version = "1.0"

[[environment.mcp_servers]]
name = "atlas"
transport = "streamable-http"
url = "http://localhost:18765/mcp"
headers = { x_run = "smoke" }
tools = ["search", "fetch"]
include_tags = ["safe"]
exclude_tags = ["admin"]
"""
    )

    (server,) = cfg.environment.mcp_servers
    assert server.name == "atlas"
    assert server.transport == "streamable-http"
    assert server.headers == {"x_run": "smoke"}
    assert server.tools == ["search", "fetch"]
    assert server.include_tags == ["safe"]
    assert server.exclude_tags == ["admin"]


def test_task_config_accepts_mcp_stdio_cwd():
    """Guards PR #878 Toolathlon MCP cwd propagation."""
    cfg = TaskConfig.model_validate_toml(
        """
version = "1.0"

[[environment.mcp_servers]]
name = "word"
transport = "stdio"
command = "uvx"
args = ["--from", "office-word-mcp-server", "word_mcp_server"]
cwd = "/workspace/agent_workspace"
"""
    )

    (server,) = cfg.environment.mcp_servers
    assert server.cwd == "/workspace/agent_workspace"


def test_task_config_accepts_environment_setup_commands():
    """Guards cd8e250b Toolathlon adapter work against setup-hook schema loss."""
    cfg = TaskConfig.model_validate_toml(
        """
version = "1.0"

[environment]
workdir = "/workspace/agent_workspace"

[[environment.setup_commands]]
command = "python preprocess.py"
cwd = "/workspace"
timeout_sec = 120
service = "main"
[environment.setup_commands.env]
FOO = "${FOO:-bar}"
"""
    )

    (command,) = cfg.environment.setup_commands
    assert command.command == "python preprocess.py"
    assert command.cwd == "/workspace"
    assert command.timeout_sec == 120
    assert command.service == "main"
    assert command.env == {"FOO": "${FOO:-bar}"}


def test_task_config_accepts_skillsbench_solution_inline_env_shorthand():
    """Guards commit a8eefb4 SkillsBench tasks-extra/diff-transformer_impl."""
    cfg = TaskConfig.model_validate_toml(
        """
version = "1.0"

[solution]
MODAL_TOKEN_ID = "${MODAL_TOKEN_ID}"
MODAL_TOKEN_SECRET = "${MODAL_TOKEN_SECRET}"
"""
    )

    assert cfg.solution.env == {
        "MODAL_TOKEN_ID": "${MODAL_TOKEN_ID}",
        "MODAL_TOKEN_SECRET": "${MODAL_TOKEN_SECRET}",
    }


def test_task_config_rejects_non_env_unknown_solution_fields():
    """Guards commit a8eefb4 shorthand support from weakening schema checks."""
    with pytest.raises(ValueError, match="unexpected"):
        TaskConfig.model_validate_toml(
            """
version = "1.0"

[solution]
unexpected = "not an env shorthand"
"""
        )


def test_task_config_rejects_conflicting_solution_inline_env():
    """Guards commit a8eefb4 shorthand support from hiding env drift."""
    with pytest.raises(ValueError, match="Conflicting values"):
        TaskConfig.model_validate_toml(
            """
version = "1.0"

[solution]
MODAL_TOKEN_ID = "inline"

[solution.env]
MODAL_TOKEN_ID = "nested"
"""
        )
