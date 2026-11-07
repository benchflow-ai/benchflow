"""Guards U10-CONFIRM-7/8 at 28f01358; daemon mutations use dummy markers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._compose_teardown import built_image_refs
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from tests.test_teardown_confirmation import rig


def configured(pair, remote, services, names):
    sandbox, daemon = pair
    if remote:
        sandbox._record_built_image_tags = (
            RemoteDockerSandbox._record_built_image_tags.__get__(sandbox)
        )
        sandbox._remove_built_images = RemoteDockerSandbox._remove_built_images.__get__(
            sandbox
        )
    sandbox._run_docker_compose_command = AsyncMock(
        side_effect=[
            ExecResult(
                stdout=json.dumps({"services": services}), stderr="", return_code=0
            ),
            ExecResult(stdout="\n".join(names), stderr="", return_code=0),
        ]
    )
    return sandbox, daemon


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("separator", ["-", "_"])
@pytest.mark.parametrize("explicit_kind", ["image", "build.tags"])
@pytest.mark.parametrize("shared_first", [False, True])
@pytest.mark.parametrize(
    "retained", [None, "operator/dummy:keep", "benchflow-recovery-lease:dummy"]
)
async def test_shared_build_does_not_disable_unrelated_cleanup(
    tmp_path, remote, separator, explicit_kind, shared_first, retained
):
    """Guards U10-CONFIRM-7 at 28f01358 across ordering, backends and references."""
    pair = rig(tmp_path)
    project = pair[0].compose_project_name
    shared, private = (
        f"{project}{separator}{name}:latest" for name in ("api", "worker")
    )
    reader = (
        {"image": shared, "pull_policy": "never"}
        if explicit_kind == "image"
        else {
            "image": "operator/shared:keep",
            "build": {"context": ".", "tags": [shared]},
        }
    )
    services = {
        "api": {"build": "./api"},
        "worker": {"build": "./worker"},
        "reader": reader,
    }
    if not shared_first:
        services = dict(reversed(list(services.items())))
    sandbox, daemon = configured(
        pair, remote, services, [shared, private, reader["image"]]
    )
    daemon.built("sha256:shared", shared, "api")
    daemon.built("sha256:private", private, "worker")
    if retained:
        daemon.images["sha256:private"]["tags"].append(retained)

    await sandbox._record_built_image_tags()
    await sandbox._remove_built_images()

    assert daemon._find(shared) == "sha256:shared"
    assert private not in daemon.tags()
    assert sandbox._private_built_image_tags == {private}
    assert ("sha256:private" in daemon.images) == bool(retained)
    assert not retained or retained in daemon.tags()
    assert daemon.parent_present


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("unresolved", ["missing", "ambiguous"])
async def test_unresolved_build_does_not_discard_other_builds(
    tmp_path, remote, unresolved
):
    """Guards U10-CONFIRM-7 at 28f01358 with per-service ownership evidence."""
    pair = rig(tmp_path)
    project = pair[0].compose_project_name
    private = f"{project}-worker:latest"
    unknown = f"{project}-api:latest"
    names = [private]
    if unresolved == "ambiguous":
        names += [unknown, f"{project}_api:latest"]
    sandbox, daemon = configured(
        pair, remote, {"api": {"build": "."}, "worker": {"build": "."}}, names
    )
    daemon.built("sha256:unknown", unknown, "api")
    daemon.built("sha256:private", private, "worker")
    await sandbox._record_built_image_tags()
    await sandbox._remove_built_images()
    assert daemon._find(unknown) == "sha256:unknown"
    assert private not in daemon.tags()


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("separator", ["-", "_"])
async def test_explicit_other_separator_does_not_hide_private_build(
    tmp_path, remote, separator
):
    """Guards U10-CONFIRM-7/8 at 28f01358 against alternate-name collisions."""
    pair = rig(tmp_path)
    project = pair[0].compose_project_name
    private = f"{project}{separator}api:latest"
    alternate = f"{project}{'_' if separator == '-' else '-'}api:latest"
    sandbox, daemon = configured(
        pair,
        remote,
        {"api": {"build": "."}, "reader": {"image": "docker.io/library/" + alternate}},
        [private, alternate],
    )
    daemon.built("sha256:private", private, "api")
    daemon.built("sha256:shared", alternate, "reader")
    await sandbox._record_built_image_tags()
    await sandbox._remove_built_images()
    assert private not in daemon.tags()
    assert daemon._find(alternate) == "sha256:shared"


# The complete Docker Hub equivalence class for a single-component repository.
HUB_PREFIXES = [
    "",
    "library/",
    "docker.io/",
    "docker.io/library/",
    "index.docker.io/",
    "index.docker.io/library/",
]


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("separator", ["-", "_"])
@pytest.mark.parametrize("explicit_kind", ["image", "build.tags"])
@pytest.mark.parametrize("prefix", HUB_PREFIXES)
@pytest.mark.parametrize("suffix", ["", ":latest"])
async def test_equivalent_explicit_references_are_preserved(
    tmp_path, remote, separator, explicit_kind, prefix, suffix
):
    """Guards U10-CONFIRM-8 at 28f01358 for every default-registry spelling."""
    pair = rig(tmp_path)
    project = pair[0].compose_project_name
    name = f"{project}{separator}api"
    ref = prefix + name + suffix
    reader = (
        {"image": ref}
        if explicit_kind == "image"
        else {"image": "operator/shared:keep", "build": {"context": ".", "tags": [ref]}}
    )
    sandbox, daemon = configured(
        pair, remote, {"api": {"build": "."}, "reader": reader}, [name, reader["image"]]
    )
    familiar = name + ":latest"
    daemon.built("sha256:shared", familiar, "api")
    await sandbox._record_built_image_tags()
    await sandbox._remove_built_images()
    assert daemon._find(familiar) == "sha256:shared"
    assert not sandbox._private_built_image_tags


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("separator", ["-", "_"])
@pytest.mark.parametrize("prefix", HUB_PREFIXES)
@pytest.mark.parametrize("path", ["resolved", "inventory"])
async def test_private_ownership_matches_equivalent_returned_references(
    tmp_path, remote, separator, prefix, path
):
    """Guards U10-CONFIRM-8 at 28f01358 at both ownership boundaries."""
    pair = rig(tmp_path)
    project = pair[0].compose_project_name
    name = f"{project}{separator}worker"
    resolved = (prefix if path == "resolved" else "") + name
    inventory = (prefix if path == "inventory" else "") + name + ":latest"
    sandbox, daemon = configured(pair, remote, {"worker": {"build": "."}}, [resolved])
    daemon.built("sha256:private", inventory, "worker")
    daemon.images["sha256:private"]["tags"].append("operator/dummy:keep")
    await sandbox._record_built_image_tags()
    await sandbox._remove_built_images()
    assert inventory not in daemon.tags()
    assert daemon.tags() == {"operator/dummy:keep"}
    assert daemon.parent_present


@pytest.mark.parametrize(
    "pattern",
    [
        "{name}:other",
        "team/{name}",
        "docker.io/team/{name}",
        "localhost/{name}",
        "registry.test/{name}",
        "registry.test:5000/{name}",
        "docker.io:443/library/{name}",
        "registry-1.docker.io/library/{name}",
        "Docker.io/library/{name}",
        "[::1]:5000/{name}",
        "{name}@sha256:" + "d" * 64,
    ],
)
async def test_distinct_references_do_not_hide_private_builds(tmp_path, pattern):
    """Guards U10-CONFIRM-8 at 28f01358 without merging distinct references."""
    pair = rig(tmp_path)
    project = pair[0].compose_project_name
    name = f"{project}-api"
    distinct = pattern.format(name=name)
    sandbox, daemon = configured(
        pair,
        False,
        {"api": {"build": "."}, "reader": {"image": distinct}},
        [name, distinct],
    )
    daemon.built("sha256:private", name + ":latest", "api")
    await sandbox._record_built_image_tags()
    await sandbox._remove_built_images()
    assert not daemon.images


@pytest.mark.parametrize("prefix", HUB_PREFIXES)
@pytest.mark.parametrize("lease", [False, True])
def test_alias_cleanup_uses_the_same_reference_identity(prefix, lease):
    """Guards U10-CONFIRM-8 at 28f01358 for alias eligibility and lease protection."""
    name = "benchflow-recovery-lease:dummy" if lease else "dummy-build:latest"
    listing = json.dumps(
        [{"Id": "sha256:dummy", "RepoTags": [prefix + name], "RepoDigests": []}]
    )
    assert built_image_refs(listing, build_tags={name, prefix + name}) == (
        [] if lease else ["sha256:dummy"]
    )
