"""Guards U10-CONFIRM-4/5 after 94c7fb07; daemon mutations are simulated."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from benchflow.cli import sandbox as sandbox_cli
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from tests.test_teardown_confirmation import rig


def test_retained_checkpoint_cleanup_preserves_unowned_parent(monkeypatch):
    """Guards U10-CONFIRM-4 at 94c7fb07 through the CLI cleanup entry point."""
    images = {"checkpoint", "unowned-parent"}
    created = (datetime.now(UTC) - timedelta(days=1)).strftime(
        "%Y-%m-%d %H:%M:%S %z UTC"
    )

    def docker(args, **kwargs):
        if args[:2] == ["docker", "images"]:
            output = f"child\tbf-snap-dummy:latest\t{created}"
        else:
            assert args[:2] == ["docker", "rmi"]
            assert args[-1] == "bf-snap-dummy:latest"
            assert "--force" not in args
            images.remove("checkpoint")
            if "--no-prune" not in args:
                images.remove("unowned-parent")
            output = ""
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(sandbox_cli.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(sandbox_cli.subprocess, "run", docker)

    counts = sandbox_cli.cleanup_docker_snapshots(dry_run=False, max_age_minutes=0)

    assert counts == {"found": 1, "deleted": 1, "skipped": 0, "failed": 0}
    assert images == {"unowned-parent"}


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("retained", ["none", "tag", "digest", "lease"])
async def test_generated_private_build_tag_is_released(tmp_path, remote, retained):
    """Guards U10-CONFIRM-5 at 94c7fb07 for local/remote custom Compose builds."""
    sandbox, daemon = rig(tmp_path)
    if remote:
        # Remote Docker inherits the same cleanup, using its own CLI transport.
        sandbox._remove_built_images = RemoteDockerSandbox._remove_built_images.__get__(
            sandbox
        )
    private = f"{sandbox.compose_project_name}-api:latest"
    config = {
        "services": {"api": {"build": "./api"}, "pulled": {"image": "ubuntu:24.04"}}
    }
    sandbox._run_docker_compose_command = AsyncMock(
        side_effect=[
            ExecResult(stdout=json.dumps(config), stderr="", return_code=0),
            ExecResult(stdout=private, stderr="", return_code=0),
        ]
    )
    await sandbox._record_built_image_tags()
    tags = [private]
    keep = (
        "archive/dummy:keep" if retained == "tag" else "benchflow-recovery-lease:dummy"
    )
    if retained in {"tag", "lease"}:
        tags.append(keep)
    daemon.built("sha256:private", private, "api")
    daemon.images["sha256:private"]["tags"] = tags
    if retained == "digest":
        daemon.digests["sha256:private"] = ["registry.test/dummy@sha256:" + "d" * 64]
        # Docker keeps a digest reference when its last ordinary tag is removed.
        # The shared legacy double models only tags, so add digest custody here.
        remove = daemon._rm

        def remove_with_digest(ref):
            image_id = daemon._find(ref)
            image = daemon.images.get(image_id)
            removed = remove(ref)
            if image is not None and daemon.digests.get(image_id):
                daemon.images[image_id] = image
            return removed

        daemon._rm = remove_with_digest
    daemon.add_image("sha256:pulled", ["ubuntu:24.04"], "other-project")

    await sandbox._remove_built_images()

    assert private not in daemon.tags()
    assert daemon._find("ubuntu:24.04") == "sha256:pulled"
    if retained == "none":
        assert "sha256:private" not in daemon.images
    else:
        assert "sha256:private" in daemon.images
        if retained in {"tag", "lease"}:
            assert daemon._find(keep) == "sha256:private"
    assert daemon.parent_present
    assert all(
        "--no-prune" in c and "--force" not in c
        for c in daemon.cli_calls
        if c[:2] == ["image", "rm"]
    )


async def test_explicit_image_names_are_not_private_even_if_they_match(tmp_path):
    """Guards U10-CONFIRM-5 after 94c7fb07 without guessing from a prefix."""
    sandbox, daemon = rig(tmp_path)
    ref = f"{sandbox.compose_project_name}-api:latest"
    config = {"services": {"api": {"image": ref, "build": "./api"}}}
    sandbox._run_docker_compose_command = AsyncMock(
        return_value=ExecResult(stdout=json.dumps(config), stderr="", return_code=0)
    )
    await sandbox._record_built_image_tags()
    daemon.built("sha256:shared", ref, "api")

    await sandbox._remove_built_images()

    assert daemon._find(ref) == "sha256:shared"


async def test_config_failure_discards_previous_private_build_names(tmp_path):
    """Guards U10-CONFIRM-5 after 94c7fb07 when effective config is unknown."""
    sandbox, daemon = rig(tmp_path)
    ref = f"{sandbox.compose_project_name}-api:latest"
    config = {"services": {"api": {"build": "./api"}}}
    sandbox._run_docker_compose_command = AsyncMock(
        side_effect=[
            ExecResult(stdout=json.dumps(config), stderr="", return_code=0),
            ExecResult(stdout=ref, stderr="", return_code=0),
        ]
    )
    await sandbox._record_built_image_tags()
    sandbox._run_docker_compose_command.side_effect = RuntimeError(
        "dummy config failure"
    )
    await sandbox._record_built_image_tags()
    daemon.built("sha256:unknown", ref, "api")

    await sandbox._remove_built_images()

    assert daemon._find(ref) == "sha256:unknown"
