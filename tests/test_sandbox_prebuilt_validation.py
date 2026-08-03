"""Prebuilt-image tasks must pass definition validation without a Dockerfile.

Guards the fix from PR #942: `docker` and `daytona` `_validate_definition`
required an `environment/Dockerfile` even when `task_env_config.docker_image`
was set, although both backends' start paths fully support prebuilt images
(docker compose references the image directly; daytona uses ``Image.base``).
That mismatch broke any prebuilt-image-only task — including rubric-review
wrapper tasks — at setup time. `modal`, `agentcore`, and `apple-container`
already skipped the file requirement; docker and daytona now match.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox.daytona import DaytonaSandbox
from benchflow.sandbox.docker import DockerSandbox


def _bare(cls, environment_dir: Path, docker_image: str | None):
    """Construct without __init__ — validation is what's under test."""

    sandbox = object.__new__(cls)
    sandbox.environment_dir = environment_dir
    sandbox.task_env_config = SimpleNamespace(docker_image=docker_image)
    return sandbox


class TestDockerPrebuiltValidation:
    def test_prebuilt_image_needs_no_dockerfile(self, tmp_path):
        sandbox = _bare(DockerSandbox, tmp_path, "python:3.13-slim")
        sandbox._validate_definition()  # must not raise

    def test_missing_everything_still_rejected(self, tmp_path):
        sandbox = _bare(DockerSandbox, tmp_path, None)
        with pytest.raises(FileNotFoundError):
            sandbox._validate_definition()

    def test_dockerfile_alone_still_accepted(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
        sandbox = _bare(DockerSandbox, tmp_path, None)
        sandbox._validate_definition()


class TestDaytonaPrebuiltValidation:
    def test_prebuilt_image_needs_no_dockerfile(self, tmp_path):
        sandbox = _bare(DaytonaSandbox, tmp_path, "python:3.13-slim")
        sandbox._compose_mode = False
        sandbox._validate_definition()  # must not raise

    def test_missing_everything_still_rejected(self, tmp_path):
        sandbox = _bare(DaytonaSandbox, tmp_path, None)
        sandbox._compose_mode = False
        with pytest.raises(FileNotFoundError):
            sandbox._validate_definition()

    def test_compose_mode_still_requires_compose_file(self, tmp_path):
        """Compose side-services cannot come from a bare prebuilt image."""
        sandbox = _bare(DaytonaSandbox, tmp_path, "python:3.13-slim")
        sandbox._compose_mode = True
        with pytest.raises(FileNotFoundError):
            sandbox._validate_definition()
