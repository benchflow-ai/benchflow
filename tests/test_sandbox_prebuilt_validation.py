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


class TestUploadFileProtocolConformance:
    """Every backend must accept upload_file(..., mode=...).

    Guards the round-3 PR #942 finding: LiteLLM always passes mode="600",
    and ModalSandbox still had the old signature, so every Modal run died
    with TypeError. A backend list that drifts from the protocol is caught
    here by signature inspection instead of at runtime.
    """

    def test_every_backend_accepts_mode(self):
        import inspect

        from benchflow.sandbox._base import BaseSandbox

        def walk(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from walk(sub)

        import benchflow.sandbox.agentcore
        import benchflow.sandbox.apple_container
        import benchflow.sandbox.daytona
        import benchflow.sandbox.docker
        import benchflow.sandbox.modal_impl  # noqa: F401

        checked = 0
        for cls in walk(BaseSandbox):
            fn = cls.__dict__.get("upload_file")
            if fn is None:
                continue
            parameters = inspect.signature(fn).parameters
            assert "mode" in parameters, (
                f"{cls.__name__}.upload_file must accept mode= (protocol)"
            )
            checked += 1
        assert checked >= 4  # docker, daytona, agentcore, apple, modal
