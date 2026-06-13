"""Tests for the inbound-adapter -> EnvironmentManifest seam (#420).

The architecture's Environment plane is fed by a validated
:class:`EnvironmentManifest`, not by legacy ``task.toml`` files
(architecture.md, "The Environment plane & the manifest"). Inbound
adapters therefore have to produce a manifest as their primary contract,
keeping the legacy ``task.toml`` / file-map output as a downstream
compatibility layer.

These tests pin that contract: every inbound adapter populates
``InboundTask.manifest`` with a manifest the Environment plane can
actually consume, and the manifest survives a round-trip through the
plane's runtime adapter (:class:`ManifestEnvironment`).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchflow.adapters.harbor import HarborAdapter
from benchflow.adapters.inbound import (
    InboundTask,
    manifest_from_task_config,
)
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import ManifestEnvironment
from benchflow.task.config import TaskConfig

# Foreign task fixtures — minimal Harbor task dirs


_HARBOR_TASK_TOML = """\
schema_version = "1.0"

[task]
name = "openmoss/abc-bench__widget"

[environment]
cpus = 2
memory_mb = 4096

[environment.env]
OPENAI_API_KEY = "${OPENAI_API_KEY}"
ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"
"""


_HARBOR_TASK_TOML_PREBUILT = """\
schema_version = "1.0"

[task]
name = "openmoss/abc-bench__widget"

[environment]
docker_image = "ghcr.io/openmoss/abc-widget:1.2.3"
"""


def _write_harbor_task(root: Path, *, toml: str = _HARBOR_TASK_TOML) -> Path:
    task_dir = root / "harbor-task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(toml)
    (task_dir / "instruction.md").write_text("Build the widget service.\n")
    (task_dir / "environment").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\npytest\n")
    return task_dir


# manifest_from_task_config — the derivation seam


class TestManifestFromTaskConfig:
    def test_uses_prebuilt_docker_image_when_set(self) -> None:
        cfg = TaskConfig.model_validate_toml(_HARBOR_TASK_TOML_PREBUILT)
        manifest = manifest_from_task_config(name="openmoss/widget", config=cfg)
        assert manifest.image == "ghcr.io/openmoss/abc-widget:1.2.3"
        assert manifest.base_image is None

    def test_synthesizes_local_tag_when_no_prebuilt(self) -> None:
        # Foreign tasks without docker_image must still yield a manifest;
        # the synthesized tag mirrors the framework's Dockerfile build path
        # so the manifest stays consistent with what the builder produces.
        cfg = TaskConfig.model_validate_toml(_HARBOR_TASK_TOML)
        manifest = manifest_from_task_config(
            name="openmoss/abc-bench__widget", config=cfg
        )
        assert manifest.image is not None
        assert manifest.image.startswith("bf__")
        assert manifest.image.endswith(":latest")
        # Slash in org/name must not appear in the docker tag.
        assert "/" not in manifest.image

    def test_forwards_sandbox_env_keys(self) -> None:
        # The legacy [environment.env] keys (host env forwarded into the
        # sandbox) become the manifest's forward_env.keys so the seam stays
        # honest about what the legacy path forwarded.
        cfg = TaskConfig.model_validate_toml(_HARBOR_TASK_TOML)
        manifest = manifest_from_task_config(name="x/y", config=cfg)
        assert set(manifest.forward_env.keys) == {
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        }

    def test_single_container_owns_lifecycle(self) -> None:
        # Legacy single-container Harbor/Terminal-Bench tasks have no
        # separate service plane — owns_lifecycle = true so the manifest
        # validates without [[services]].
        cfg = TaskConfig.model_validate_toml(_HARBOR_TASK_TOML)
        manifest = manifest_from_task_config(name="x/y", config=cfg)
        assert manifest.owns_lifecycle is True
        assert manifest.services == []


# Adapter -> manifest contract — the core of #420


class TestHarborManifestSeam:
    def test_returns_validated_manifest(self, tmp_path: Path) -> None:
        task_dir = _write_harbor_task(tmp_path)
        result = HarborAdapter.from_task_dir(task_dir)
        assert isinstance(result, InboundTask)
        assert isinstance(result.manifest, EnvironmentManifest)

    def test_manifest_carries_task_identity(self, tmp_path: Path) -> None:
        task_dir = _write_harbor_task(tmp_path)
        result = HarborAdapter.from_task_dir(task_dir)
        assert result.manifest.name == "openmoss/abc-bench__widget"

    def test_manifest_forwards_env_keys(self, tmp_path: Path) -> None:
        task_dir = _write_harbor_task(tmp_path)
        result = HarborAdapter.from_task_dir(task_dir)
        assert set(result.manifest.forward_env.keys) == {
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        }

    def test_prefers_sibling_environment_toml(self, tmp_path: Path) -> None:
        """A Harbor task shipping its own ``environment.toml`` (the
        Environment-plane-native shape — chi-bench, clawsbench) bypasses
        the derived manifest and loads the file directly."""
        task_dir = _write_harbor_task(tmp_path)
        (task_dir / "environment.toml").write_text(
            "[environment]\n"
            'name = "explicit-env"\n'
            'image = "ghcr.io/example/explicit:9.9"\n'
            "owns_lifecycle = true\n"
        )
        result = HarborAdapter.from_task_dir(task_dir)
        assert result.manifest.name == "explicit-env"
        assert result.manifest.image == "ghcr.io/example/explicit:9.9"


# Manifest can be consumed by manifest-backed runtime


class _FakeSandbox:
    """Minimal Sandbox stand-in for ManifestEnvironment unit tests.

    ManifestEnvironment only needs ``exec`` on the sandbox; this records
    invocations so the test asserts the manifest reached the runtime
    without ever building a real container.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def exec(self, command: str, *, timeout_sec: int = 60) -> Any:
        self.calls.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")


@pytest.mark.asyncio
async def test_adapter_manifest_runs_through_manifest_environment(
    tmp_path: Path,
) -> None:
    """End-to-end: an inbound adapter's manifest is accepted by the
    Environment plane's default runtime adapter without further
    translation. This is the architectural contract #420 demands —
    adapter output flows through manifest-backed paths."""
    task_dir = _write_harbor_task(tmp_path)
    result = HarborAdapter.from_task_dir(task_dir)

    sandbox = _FakeSandbox()
    env = ManifestEnvironment(result.manifest, sandbox=sandbox)
    handle = await env.provision(ctx={"task_id": "demo"})

    # The manifest's owns_lifecycle = true path means no services are
    # framework-started; the handle just exposes endpoints for the
    # manifest's declared ports.
    assert handle.name == result.manifest.name
    assert isinstance(handle.endpoints, dict)


# Legacy compatibility — the file-map layer still works


def test_legacy_file_map_still_carried(tmp_path: Path) -> None:
    """The Environment-plane manifest is the new contract, but the
    ``files`` map (the legacy task-file compatibility layer) must keep
    working so downstream materialization paths are untouched."""
    task_dir = _write_harbor_task(tmp_path)
    result = HarborAdapter.from_task_dir(task_dir)
    assert (
        result.files["environment/Dockerfile"]
        == task_dir / "environment" / "Dockerfile"
    )
    assert result.files["tests/test.sh"] == task_dir / "tests" / "test.sh"
