"""Guards U10-CONFIRM-6 at ad6c0573; all daemon mutations use dummy markers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from tests.test_teardown_confirmation import rig


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("separator", ["-", "_"])
@pytest.mark.parametrize("dependency", ["independent", "direct", "transitive"])
@pytest.mark.parametrize(
    "retained", [None, "archive/dummy:keep", "benchflow-recovery-lease:dummy"]
)
async def test_provider_services_do_not_disable_build_cleanup(
    tmp_path, remote, separator, dependency, retained
):
    """Guards U10-CONFIRM-6 at ad6c0573 across names, backends and dependencies."""
    sandbox, daemon = rig(tmp_path)
    if remote:
        sandbox._record_built_image_tags = (
            RemoteDockerSandbox._record_built_image_tags.__get__(sandbox)
        )
        sandbox._remove_built_images = RemoteDockerSandbox._remove_built_images.__get__(
            sandbox
        )
    project = sandbox.compose_project_name
    provider = {"provider": {"type": "dummy", "options": {"type": "mysql"}}}
    pulled = "registry.test/dummy@sha256:" + "d" * 64
    services = {
        "api-worker": {"build": "./api"},
        "worker_jobs": {"build": {"context": "./worker"}},
        "database": provider,
        "database-extra": provider,
        "pulled": {"image": pulled},
        "shared": {
            "image": "operator/main:keep",
            "build": {"context": ".", "tags": ["registry.test:5000/api:keep"]},
        },
    }
    if dependency == "direct":
        services["api-worker"]["depends_on"] = {"database": {}}
    elif dependency == "transitive":
        services["api-worker"]["depends_on"] = {"worker_jobs": {}}
        services["worker_jobs"]["depends_on"] = {"database": {}, "pulled": {}}
    # Compose emits defaults even for services that have no image or build.
    names = [
        spec.get("image", f"{project}{separator}{name}")
        for name, spec in services.items()
    ]
    sandbox._run_docker_compose_command = AsyncMock(
        side_effect=[
            ExecResult(
                stdout=json.dumps({"services": services}), stderr="", return_code=0
            ),
            ExecResult(stdout="\n".join(reversed(names)), stderr="", return_code=0),
        ]
    )
    private = {
        f"{project}{separator}{name}:latest" for name in ("api-worker", "worker_jobs")
    }
    for i, ref in enumerate(sorted(private)):
        daemon.built(f"sha256:private-{i}", ref, "dummy")
        if retained:
            daemon.images[f"sha256:private-{i}"]["tags"].append(f"{retained}-{i}")
    # A project-labelled image does not make a provider's generated name ours.
    provider_ref = f"{project}{separator}database:latest"
    daemon.built("sha256:provider", provider_ref, "database")
    daemon.built("sha256:shared", "operator/main:keep", "shared")
    daemon.built("sha256:extra", "registry.test:5000/api:keep", "shared")
    daemon.add_image("sha256:pulled", [pulled], "other-project")

    await sandbox._record_built_image_tags()
    await sandbox._remove_built_images()

    assert not private.intersection(daemon.tags())
    assert sandbox._private_built_image_tags == private
    assert {
        provider_ref,
        pulled,
        "operator/main:keep",
        "registry.test:5000/api:keep",
    } <= daemon.tags()
    for i in range(2):
        assert (f"sha256:private-{i}" in daemon.images) == bool(retained)
        if retained:
            assert f"{retained}-{i}" in daemon.tags()
    assert daemon.parent_present
    assert all(
        "--no-prune" in call and "--force" not in call
        for call in daemon.cli_calls
        if call[:2] == ["image", "rm"]
    )


@pytest.mark.parametrize("separator", ["-", "_"])
@pytest.mark.parametrize("explicit_kind", ["image", "build.tags"])
async def test_explicit_reference_collision_keeps_generated_name(
    tmp_path, separator, explicit_kind
):
    """Guards U10-CONFIRM-6 at ad6c0573 without claiming shared references."""
    sandbox, daemon = rig(tmp_path)
    private = f"{sandbox.compose_project_name}{separator}api:latest"
    shared = (
        {"image": private}
        if explicit_kind == "image"
        else {"image": "shared:keep", "build": {"context": ".", "tags": [private]}}
    )
    config = {"services": {"api": {"build": "."}, "shared": shared}}
    sandbox._run_docker_compose_command = AsyncMock(
        side_effect=[
            ExecResult(stdout=json.dumps(config), stderr="", return_code=0),
            ExecResult(
                stdout=f"{private}\n{shared['image']}\n", stderr="", return_code=0
            ),
        ]
    )
    daemon.built("sha256:shared", private, "api")
    await sandbox._record_built_image_tags()
    await sandbox._remove_built_images()
    assert daemon._find(private) == "sha256:shared"
    assert not sandbox._private_built_image_tags
