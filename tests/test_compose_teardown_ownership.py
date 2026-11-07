"""Guards PY8 U10's ownership fixes to 30518669; every daemon call is a double."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, call

import pytest

from benchflow.sandbox._base import ExecResult
from tests.test_compose_teardown import LEASE, _docker_sandbox


@pytest.mark.parametrize("started", [False, True])
async def test_reused_named_volume_is_not_given_to_compose_down(tmp_path, started):
    """Guards the named-volume ownership fix, including failed startup."""
    sandbox, daemon = _docker_sandbox(tmp_path)
    sandbox._compose_up_attempted = started
    owned = f"{sandbox.compose_project_name}_data"
    removed = []

    async def cli(args, check=True, *, raw_output=False):
        if args[:2] == ["volume", "ls"]:
            assert (
                f"label=com.docker.compose.project={sandbox.compose_project_name}"
                in args
            )
            return ExecResult(stdout=owned, stderr="", return_code=0)
        if args[:2] == ["volume", "rm"]:
            removed.extend(args[2:])
        return await daemon.cli(args, check=check)

    sandbox._docker_cli = cli
    await sandbox.stop(delete=True)

    assert all("--volumes" not in c for c in daemon.compose_calls if c[0] == "down")
    assert not any(c[0] == "rm" for c in daemon.compose_calls)
    assert removed == [
        owned
    ]  # the adopted, unlabelled shared volume is never a candidate


async def test_retagged_build_is_removed_by_captured_identity(tmp_path):
    """B moves A's tag after A lists it."""
    sandbox, daemon = _docker_sandbox(tmp_path)
    daemon.built("sha256:old", "bf__hello:latest", "main")

    async def cli(args, check=True, *, raw_output=False):
        result = await daemon.cli(args, check=check)
        if args[:2] == ["image", "ls"]:
            daemon.images["sha256:old"]["tags"].clear()
            daemon.add_image("sha256:new", ["bf__hello:latest"], "other-project")
        return result

    sandbox._docker_cli = cli
    await sandbox.stop(delete=True)

    assert "sha256:new" in daemon.images
    assert "sha256:old" not in daemon.images
    # U10-CONFIRM-3 after 22672877: remove only the private pin, never an ID.
    pin = next(c for c in daemon.cli_calls if c[:2] == ["image", "tag"])
    assert pin[2] == "sha256:old"
    assert ["image", "rm", "--no-prune", pin[3]] in daemon.cli_calls


@pytest.mark.parametrize("tags", [["archive/hello:keep"], ["bf__hello:latest", LEASE]])
async def test_retained_tags_keep_the_build(tmp_path, tags):
    """Guards the fix and the recovery-lease sibling of that class."""
    sandbox, daemon = _docker_sandbox(tmp_path)
    daemon.add_image("sha256:old", tags, sandbox.compose_project_name)
    await sandbox.stop(delete=True)
    assert daemon.images["sha256:old"]["tags"] == tags
    assert not any(c[:2] == ["image", "rm"] for c in daemon.cli_calls)


async def test_image_removal_does_not_prune_unowned_parents(tmp_path):
    """The CLI must not recursively prune parents."""
    sandbox, daemon = _docker_sandbox(tmp_path)
    daemon.built("sha256:child", f"{sandbox._env_vars.main_image_name}:latest", "main")
    sandbox._docker_cli = AsyncMock(side_effect=daemon.cli)
    await sandbox.stop(delete=True)
    removals = [
        c.args[0]
        for c in sandbox._docker_cli.await_args_list
        if c.args[0][:2] == ["image", "rm"]
    ]
    assert removals
    assert all("--no-prune" in c and "--force" not in c for c in removals)


async def test_custom_build_tags_are_exact_and_pulled_services_are_not_owned(tmp_path):
    """Record effective image/build.tags, not name prefixes."""
    sandbox, daemon = _docker_sandbox(tmp_path)
    project = sandbox.compose_project_name
    config = {
        "services": {
            "main": {"image": "custom/main", "build": {"context": "."}},
            "api": {
                "build": {"context": ".", "tags": ["registry.test:5000/api:build"]}
            },
            "db": {"image": "operator/db:keep"},
        }
    }
    sandbox._run_docker_compose_command = AsyncMock(
        side_effect=[
            ExecResult(stdout=json.dumps(config), stderr="", return_code=0),
            ExecResult(stdout=f"{project}-api", stderr="", return_code=0),
        ]
    )
    await sandbox._record_built_image_tags()
    daemon.built("sha256:main", "custom/main:latest", "main")
    daemon.built("sha256:api", f"{project}-api:latest", "api")
    daemon.built("sha256:tag", "registry.test:5000/api:build", "api")
    daemon.built("sha256:retained", f"{project}-operator:keep", "main")
    daemon.add_image("sha256:pulled", ["operator/db:keep"])
    await sandbox._remove_built_images()
    # U10-CONFIRM-5 after 94c7fb07: the generated API tag is trial-private;
    # explicit image/build.tags names can be shared by later trials.
    assert set(daemon.images) == {
        "sha256:main",
        "sha256:tag",
        "sha256:retained",
        "sha256:pulled",
    }
    sandbox._run_docker_compose_command.assert_has_awaits(
        [
            call(["config", "--format", "json"], timeout_sec=30, raw_output=True),
            call(["config", "--images"], timeout_sec=30, raw_output=True),
        ]
    )
