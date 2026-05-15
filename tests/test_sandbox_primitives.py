"""Tests for native sandbox value types and protocols."""

from pathlib import Path

from benchflow.sandboxes import ExecResult, ImageConfig, ImageRef, SandboxSpec


def test_exec_result_success_property() -> None:
    assert ExecResult(return_code=0).success is True
    assert ExecResult(return_code=2).success is False


def test_sandbox_spec_defaults_match_current_backend() -> None:
    spec = SandboxSpec()
    assert spec.provider == "docker"
    assert spec.allow_internet is True
    assert spec.env == {}


def test_image_config_and_ref_are_provider_neutral() -> None:
    config = ImageConfig(dockerfile=Path("environment/Dockerfile"))
    ref = ImageRef(provider="docker", ref="benchflow-task:latest")
    assert config.dockerfile == Path("environment/Dockerfile")
    assert ref.provider == "docker"
    assert ref.ref == "benchflow-task:latest"
