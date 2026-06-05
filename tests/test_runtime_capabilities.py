"""Tests for P1 runtime capability gates (task-standard)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchflow.task import Task, TaskConfig
from benchflow.task.runtime_capabilities import (
    UnsupportedTaskRuntimeError,
    ensure_task_runtime_support,
    validate_task_runtime_support,
)

PROMPT_USER_SEMANTICS_TASK = Path(
    "docs/examples/task-standard/benchflow-wanted-features/prompt-user-semantics"
)


def _write_minimal_task(task_dir: Path, toml: str) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(toml)
    (task_dir / "instruction.md").write_text("Do the thing.\n")
    env = task_dir / "environment"
    env.mkdir()
    (env / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    tests = task_dir / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return task_dir


def _load_task(task_dir: Path) -> Task:
    return Task(task_dir)


@pytest.mark.parametrize("sandbox_type", ["docker", "daytona", "modal"])
class TestGatedSandboxesRejectUnsupportedFields:
    """Guards P1 fail-closed validation for docker, daytona, and modal."""

    def test_minimal_task_has_no_runtime_issues(self, tmp_path: Path, sandbox_type: str) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "ok-task",
            'version = "1.0"\n[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 120\n',
        )
        task = _load_task(task_dir)
        assert validate_task_runtime_support(task, sandbox_type, task_dir) == []

    def test_steps_fail_closed(self, tmp_path: Path, sandbox_type: str) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "steps-task",
            """\
version = "1.0"

[[steps]]
name = "scaffold"
""",
        )
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, sandbox_type, task_dir)
        assert any(issue.path == "steps" for issue in issues)

    def test_root_artifacts_fail_closed(self, tmp_path: Path, sandbox_type: str) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "artifacts-task",
            """\
version = "1.0"

[[artifacts]]
source = "/logs/artifacts"
destination = "artifacts"
""",
        )
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, sandbox_type, task_dir)
        assert any(issue.path == "artifacts" for issue in issues)

    def test_step_artifacts_fail_closed(self, tmp_path: Path, sandbox_type: str) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "step-artifacts-task",
            """\
version = "1.0"

[[steps]]
name = "collect"
artifacts = ["/app/out.txt"]
""",
        )
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, sandbox_type, task_dir)
        assert any(issue.path == "steps['collect'].artifacts" for issue in issues)

    def test_environment_allowlist_fail_closed(
        self, tmp_path: Path, sandbox_type: str
    ) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "allowlist-task",
            """\
version = "1.0"

[environment]
network_mode = "allowlist"
allowed_hosts = ["datasets.example.com"]
""",
        )
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, sandbox_type, task_dir)
        assert any(issue.path == "environment.network_mode" for issue in issues)

    def test_agent_allowlist_fail_closed(self, tmp_path: Path, sandbox_type: str) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "agent-allowlist-task",
            """\
version = "1.0"

[agent]
network_mode = "allowlist"
allowed_hosts = ["api.example.com"]
""",
        )
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, sandbox_type, task_dir)
        assert any(issue.path == "agent.network_mode" for issue in issues)

    def test_verifier_allowlist_fail_closed(
        self, tmp_path: Path, sandbox_type: str
    ) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "verifier-allowlist-task",
            """\
version = "1.0"

[verifier]
network_mode = "allowlist"
allowed_hosts = ["grader.example.com"]
""",
        )
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, sandbox_type, task_dir)
        assert any(issue.path == "verifier.network_mode" for issue in issues)

    def test_separate_verifier_environment_fail_closed(
        self, tmp_path: Path, sandbox_type: str
    ) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "separate-verifier-task",
            """\
version = "1.0"

[verifier]
environment_mode = "separate"

[verifier.environment]
docker_image = "ghcr.io/example/grader:latest"
cpus = 1
memory_mb = 512
""",
        )
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, sandbox_type, task_dir)
        assert any(issue.path.startswith("verifier.environment") for issue in issues)

    def test_windows_fail_closed(self, tmp_path: Path, sandbox_type: str) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "windows-task",
            """\
version = "1.0"

[environment]
os = "windows"
""",
        )
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, sandbox_type, task_dir)
        assert any(issue.path == "environment.os" for issue in issues)

    def test_tpu_fail_closed(self, tmp_path: Path, sandbox_type: str) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "tpu-task",
            """\
version = "1.0"

[environment.tpu]
type = "v6e"
topology = "2x4"
""",
        )
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, sandbox_type, task_dir)
        assert any(issue.path == "environment.tpu" for issue in issues)

    def test_healthcheck_fail_closed(self, tmp_path: Path, sandbox_type: str) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "healthcheck-task",
            """\
version = "1.0"

[environment.healthcheck]
command = "python -m app.healthcheck"
""",
        )
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, sandbox_type, task_dir)
        assert any(issue.path == "environment.healthcheck" for issue in issues)

    def test_workdir_fail_closed(self, tmp_path: Path, sandbox_type: str) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "workdir-task",
            """\
version = "1.0"

[environment]
workdir = "/workspace"
""",
        )
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, sandbox_type, task_dir)
        assert any(issue.path == "environment.workdir" for issue in issues)

    def test_non_main_verifier_service_without_compose_fail_closed(
        self, tmp_path: Path, sandbox_type: str
    ) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "target-verifier-task",
            """\
version = "1.0"

[verifier]
service = "target"
""",
        )
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, sandbox_type, task_dir)
        assert any(issue.path == "verifier.service" for issue in issues)

    def test_ensure_raises_with_actionable_message(
        self, tmp_path: Path, sandbox_type: str
    ) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "raise-task",
            """\
version = "1.0"

[environment]
workdir = "/repo"
""",
        )
        task = _load_task(task_dir)
        with pytest.raises(UnsupportedTaskRuntimeError, match=r"environment\.workdir"):
            ensure_task_runtime_support(task, sandbox_type, task_dir)


class TestNonMainVerifierServiceWithCompose:
    """Non-main verifier.service is allowed when compose topology exists."""

    def test_target_service_allowed_with_compose(self, tmp_path: Path) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "compose-target-task",
            """\
version = "1.0"

[verifier]
service = "target"
""",
        )
        (task_dir / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main: {}\n  target: {}\n"
        )
        task = _load_task(task_dir)
        for sandbox_type in ("docker", "daytona", "modal"):
            assert validate_task_runtime_support(task, sandbox_type, task_dir) == []


class TestUngatedSandboxes:
    """Unknown backends skip capability gating."""

    def test_unknown_backend_skips_validation(self, tmp_path: Path) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "k8s-task",
            """\
version = "1.0"

[environment]
workdir = "/workspace"
network_mode = "allowlist"
allowed_hosts = ["example.com"]
""",
        )
        task = _load_task(task_dir)
        assert validate_task_runtime_support(task, "kubernetes", task_dir) == []


class TestHarborParityFixture:
    """Schema-only harbor-parity example fields are rejected on docker."""

    def test_harbor_parity_frontmatter_is_blocked(self, tmp_path: Path) -> None:
        parity_toml = """\
version = "1.3"
multi_step_reward_strategy = "final"

[[artifacts]]
source = "/logs/artifacts"
destination = "artifacts"

[agent]
network_mode = "allowlist"
allowed_hosts = ["api.example.com"]

[verifier]
environment_mode = "separate"

[verifier.environment]
docker_image = "ghcr.io/example/grader:latest"

[environment]
network_mode = "allowlist"
allowed_hosts = ["datasets.example.com"]
workdir = "/workspace"

[environment.tpu]
type = "v6e"
topology = "2x4"

[environment.healthcheck]
command = "python -m app.healthcheck"

[[steps]]
name = "scaffold"
artifacts = ["/app/scaffold.txt"]
"""
        task_dir = _write_minimal_task(tmp_path / "harbor-parity", parity_toml)
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, "docker", task_dir)
        paths = {issue.path for issue in issues}
        assert "steps" in paths
        assert "artifacts" in paths
        assert "environment.network_mode" in paths
        assert "agent.network_mode" in paths
        assert "verifier.environment" in paths
        assert "environment.workdir" in paths
        assert "environment.tpu" in paths
        assert "environment.healthcheck" in paths


def test_create_sandbox_environment_fails_before_object_construction(
    tmp_path: Path,
) -> None:
    """Guards integration before Docker/Daytona sandbox object construction."""
    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task import RolloutPaths

    task_dir = _write_minimal_task(
        tmp_path / "setup-gate-task",
        """\
version = "1.0"

[environment]
workdir = "/repo"
""",
    )
    task = _load_task(task_dir)
    rollout_paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    rollout_paths.mkdir()

    with pytest.raises(UnsupportedTaskRuntimeError, match=r"environment\.workdir"):
        _create_sandbox_environment(
            "docker",
            task,
            task_dir,
            "rollout-name",
            rollout_paths,
        )


def test_rollout_setup_fails_before_environment_creation(tmp_path: Path) -> None:
    """Guards rollout.setup() wiring before sandbox creation."""
    import asyncio

    from benchflow.rollout import Rollout, RolloutConfig

    task_dir = _write_minimal_task(
        tmp_path / "rollout-gate-task",
        """\
version = "1.0"

[environment]
workdir = "/repo"
""",
    )
    rollout = Rollout(
        RolloutConfig(task_path=task_dir, environment="docker", skip_verify=True)
    )

    with pytest.raises(UnsupportedTaskRuntimeError, match=r"environment\.workdir"):
        asyncio.run(rollout.setup())


@pytest.mark.parametrize("sandbox_type", ["docker", "daytona", "modal"])
class TestPromptUserSemanticsDogfood:
    """Guards fail-closed user/nudge validation for prompt-user-semantics dogfood."""

    def test_simulated_user_nudges_supported_when_user_loop_executable(
        self, sandbox_type: str
    ) -> None:
        """Guards simulated-user nudge execution when document user loop compiles."""
        task = Task(PROMPT_USER_SEMANTICS_TASK)
        issues = validate_task_runtime_support(
            task, sandbox_type, PROMPT_USER_SEMANTICS_TASK
        )
        paths = {issue.path for issue in issues}
        assert "user" not in paths
        assert "benchflow.nudges" not in paths
        assert "prompt.user-persona" not in paths

    def test_nudges_fail_closed_without_executable_user_loop(
        self, tmp_path: Path, sandbox_type: str
    ) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "nudges-without-user-loop",
            """\
version = "1.0"

[agent]
timeout_sec = 300

[verifier]
timeout_sec = 120
""",
        )
        (task_dir / "task.md").write_text(
            """\
---
schema_version: "1.3"
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
benchflow:
  nudges:
    mode: simulated-user
    nudge_budget: 4
---

## prompt

Do the thing.
"""
        )
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, sandbox_type, task_dir)
        paths = {issue.path for issue in issues}
        assert "benchflow.nudges" in paths

    def test_metadata_only_user_runtime_skips_user_semantics_issues(
        self, tmp_path: Path, sandbox_type: str
    ) -> None:
        task_dir = _write_minimal_task(
            tmp_path / "metadata-only-user-task",
            """\
version = "1.0"

[agent]
timeout_sec = 300

[verifier]
timeout_sec = 120
""",
        )
        (task_dir / "task.md").write_text(
            """\
---
schema_version: "1.3"
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
user:
  model: claude-haiku
  stop_rule: satisfied-or-5-rounds
benchflow:
  user_runtime: metadata-only
  nudges:
    mode: simulated-user
---

## prompt

Do the thing.

## user-persona

You reveal private facts only when asked.
"""
        )
        task = _load_task(task_dir)
        issues = validate_task_runtime_support(task, sandbox_type, task_dir)
        paths = {issue.path for issue in issues}
        assert "user" not in paths
        assert "benchflow.nudges" not in paths
        assert "prompt.user-persona" not in paths


def test_validator_accepts_task_like_object(tmp_path: Path) -> None:
    """validate_task_runtime_support only needs config-bearing task objects."""
    task_dir = _write_minimal_task(
        tmp_path / "mock-task",
        """\
version = "1.0"

[environment]
workdir = "/repo"
""",
    )
    config = TaskConfig.model_validate_toml((task_dir / "task.toml").read_text())
    task = MagicMock()
    task.config = config
    issues = validate_task_runtime_support(task, "docker", task_dir)
    assert any(issue.path == "environment.workdir" for issue in issues)
