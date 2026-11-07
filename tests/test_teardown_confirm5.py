"""Guards U10-CONFIRM-5 at a9829c79; all image mutations use dummy markers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from tests.test_teardown_confirmation import rig


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("separator", ["-", "_"])
@pytest.mark.parametrize(
    "retained", [None, "archive/dummy:keep", "benchflow-recovery-lease:dummy"]
)
async def test_compose_resolves_private_build_names(
    tmp_path, remote, separator, retained
):
    """Guards U10-CONFIRM-5 at a9829c79 for compatibility and default naming."""
    sandbox, daemon = rig(tmp_path)
    if remote:
        sandbox._remove_built_images = RemoteDockerSandbox._remove_built_images.__get__(
            sandbox
        )
    private = f"{sandbox.compose_project_name}{separator}api:latest"
    config = {
        "services": {
            "api": {
                "build": {"context": "./api", "tags": ["registry.test:5000/api:keep"]}
            },
            "main": {"image": "operator/main:keep", "build": "."},
            "pulled": {"image": "ubuntu:24.04"},
        }
    }

    async def compose(command, **kwargs):
        if command == ["config", "--format", "json"]:
            output = json.dumps(config)
        else:
            assert command == ["config", "--images"]
            output = (
                f"{private.removesuffix(':latest')}\noperator/main:keep\nubuntu:24.04\n"
            )
        return ExecResult(stdout=output, stderr="", return_code=0)

    sandbox._run_docker_compose_command = AsyncMock(side_effect=compose)
    await sandbox._record_built_image_tags()
    daemon.built("sha256:private", private, "api")
    if retained:
        daemon.images["sha256:private"]["tags"].append(retained)
    daemon.built("sha256:explicit", "operator/main:keep", "main")
    daemon.built("sha256:extra", "registry.test:5000/api:keep", "api")
    daemon.add_image("sha256:pulled", ["ubuntu:24.04"], "other-project")

    await sandbox._remove_built_images()

    assert private not in daemon.tags()
    assert ("sha256:private" in daemon.images) == bool(retained)
    assert {
        "operator/main:keep",
        "registry.test:5000/api:keep",
        "ubuntu:24.04",
    } <= daemon.tags()
    if retained:
        assert retained in daemon.tags()
    assert daemon.parent_present
    assert all(
        "--no-prune" in call and "--force" not in call
        for call in daemon.cli_calls
        if call[:2] == ["image", "rm"]
    )


@pytest.mark.parametrize(
    "output",
    ["", "first\nsecond\n", "warning: dummy\nproject_api\n", "dummy@sha256:abcd"],
)
async def test_unknown_resolved_names_discard_stale_ownership(tmp_path, output):
    """Guards U10-CONFIRM-5 at a9829c79 when Compose cannot resolve one image."""
    sandbox, daemon = rig(tmp_path)
    private = f"{sandbox.compose_project_name}_api:latest"
    config = {"services": {"api": {"build": "./api"}}}
    sandbox._run_docker_compose_command = AsyncMock(
        side_effect=[
            ExecResult(stdout=json.dumps(config), stderr="", return_code=0),
            ExecResult(stdout=output, stderr="", return_code=0),
        ]
    )
    sandbox._built_image_tags.add(private)
    sandbox._private_built_image_tags.add(private)
    daemon.built("sha256:unknown", private, "api")

    await sandbox._record_built_image_tags()
    await sandbox._remove_built_images()

    assert daemon._find(private) == "sha256:unknown"


async def test_explicit_compatibility_name_is_shared(tmp_path):
    """Guards U10-CONFIRM-5 at a9829c79 without inferring ownership from '_'."""
    sandbox, daemon = rig(tmp_path)
    ref = f"{sandbox.compose_project_name}_api:latest"
    config = {"services": {"api": {"image": ref, "build": "./api"}}}
    sandbox._run_docker_compose_command = AsyncMock(
        return_value=ExecResult(stdout=json.dumps(config), stderr="", return_code=0)
    )
    daemon.built("sha256:shared", ref, "api")

    await sandbox._record_built_image_tags()
    await sandbox._remove_built_images()

    assert daemon._find(ref) == "sha256:shared"
    sandbox._run_docker_compose_command.assert_awaited_once_with(
        ["config", "--format", "json"], timeout_sec=30, raw_output=True
    )


@pytest.mark.parametrize("separator", ["-", "_"])
async def test_generated_dependencies_and_pulled_digests(tmp_path, separator):
    """Guards U10-CONFIRM-5 at a9829c79 with multiple builds and dependencies."""
    sandbox, daemon = rig(tmp_path)
    refs = {
        f"{sandbox.compose_project_name}{separator}{s}:latest"
        for s in ("api", "worker")
    }
    pulled = "registry.test/dummy@sha256:" + "d" * 64
    config = {
        "services": {
            "api": {"build": ".", "depends_on": {"worker": {}}},
            "worker": {"build": ".", "depends_on": {"db": {}}},
            "db": {"image": pulled},
        }
    }
    sandbox._run_docker_compose_command = AsyncMock(
        side_effect=[
            ExecResult(stdout=json.dumps(config), stderr="", return_code=0),
            ExecResult(
                stdout="\n".join([*sorted(refs), pulled]), stderr="", return_code=0
            ),
        ]
    )
    for i, ref in enumerate(sorted(refs)):
        daemon.built(f"sha256:dummy-{i}", ref, "dummy")

    await sandbox._record_built_image_tags()
    await sandbox._remove_built_images()

    assert not daemon.images
    assert daemon.parent_present


async def test_resolution_failure_discards_stale_names(tmp_path):
    """Guards U10-CONFIRM-5 at a9829c79 when the second config command fails."""
    sandbox, daemon = rig(tmp_path)
    ref = f"{sandbox.compose_project_name}_api:latest"
    sandbox._built_image_tags.add(ref)
    sandbox._private_built_image_tags.add(ref)
    sandbox._run_docker_compose_command = AsyncMock(
        side_effect=[
            ExecResult(
                stdout=json.dumps({"services": {"api": {"build": "."}}}),
                stderr="",
                return_code=0,
            ),
            RuntimeError("dummy resolution failure"),
        ]
    )
    daemon.built("sha256:unknown", ref, "api")

    await sandbox._record_built_image_tags()
    await sandbox._remove_built_images()

    assert daemon._find(ref) == "sha256:unknown"
