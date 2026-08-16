"""Offline tests for the Fystash sandbox backend (no live API)."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.sandbox.fystash import FystashSandbox, _has_cli_credentials
from benchflow.sandbox.providers import OPTIONAL_SANDBOX_EXTRAS, SANDBOX_PROVIDER_SET
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths


def _sandbox(
    tmp_path: Path, *, manifest_name: str = "environment.yaml"
) -> FystashSandbox:
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / manifest_name).write_text(
        "apiVersion: environments.fystash.dev/v1alpha1\nkind: Environment\n",
        encoding="utf-8",
    )
    return FystashSandbox(
        environment_dir=env_dir,
        environment_name="hello_exec",
        session_id="rollout-test",
        rollout_paths=RolloutPaths(tmp_path / "rollout"),
        task_env_config=SandboxConfig(),
    )


def test_registry_includes_fystash() -> None:
    assert "fystash" in SANDBOX_PROVIDER_SET
    assert OPTIONAL_SANDBOX_EXTRAS["fystash"] == "sandbox-fystash"


def test_validate_definition_accepts_environment_yaml(tmp_path: Path) -> None:
    box = _sandbox(tmp_path, manifest_name="environment.yaml")
    assert box.supports_snapshot is True
    assert box.is_mounted is False
    assert box.host == "127.0.0.1"


def test_validate_definition_accepts_sandbox_yaml(tmp_path: Path) -> None:
    _sandbox(tmp_path, manifest_name="sandbox.yaml")


def test_validate_definition_rejects_dockerfile_only(tmp_path: Path) -> None:
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    with pytest.raises(
        FileNotFoundError, match="does not build arbitrary guest Dockerfiles"
    ):
        FystashSandbox(
            environment_dir=env_dir,
            environment_name="skillsbench-task",
            session_id="rollout-test",
            rollout_paths=RolloutPaths(tmp_path / "rollout"),
            task_env_config=SandboxConfig(),
        )


def test_preflight_without_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("FYSTASH_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("FYSTASH_PROJECT_ID", raising=False)
    monkeypatch.delenv("FYSTASH_CONFIG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchflow.sandbox.fystash.Path.home", lambda: tmp_path / "no-home"
    )
    assert _has_cli_credentials() is False
    with pytest.raises(SystemExit, match="FYSTASH_ACCESS_TOKEN"):
        FystashSandbox.preflight()
