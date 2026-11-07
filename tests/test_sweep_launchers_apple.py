"""Guards FF-sw-launchers against host symlink reads at 363d815b."""

import shutil
from unittest.mock import patch

import pytest

from benchflow.sandbox.apple_container import AppleContainerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["absolute-file", "relative-file", "directory", "cycle", "fifo"]
)
async def test_mounted_copy_does_not_follow_untrusted_entries(tmp_path, kind):
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    root = tmp_path / "rollout"
    source = root / "artifacts"
    source.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "dummy").write_text("DUMMY_HOST_MARKER")
    link = source / "entry"
    if kind == "absolute-file":
        link.symlink_to(outside / "dummy")
    elif kind == "relative-file":
        link.symlink_to("../../outside/dummy")
    elif kind == "directory":
        link.symlink_to(outside, target_is_directory=True)
    elif kind == "cycle":
        link.symlink_to("entry")
    else:
        import os

        os.mkfifo(link)
    with patch.object(AppleContainerSandbox, "preflight"):
        env = AppleContainerSandbox(
            environment_dir=env_dir,
            environment_name="task",
            session_id="s",
            rollout_paths=RolloutPaths(root),
            task_env_config=SandboxConfig(),
        )
    env._container_name = "dummy"
    target = tmp_path / "copy"
    with pytest.raises((ValueError, OSError, shutil.Error)):
        await env.download_dir("/logs/artifacts", target)
    assert not any(
        p.is_file() and b"DUMMY_HOST_MARKER" in p.read_bytes()
        for p in target.rglob("*")
        if not p.is_symlink()
    )


@pytest.mark.parametrize("directory", [False, True])
def test_opened_source_never_follows_replaced_descendant(
    tmp_path, monkeypatch, directory
):
    """Guards FF-sw-launchers against a check/use swap at 363d815b."""
    from benchflow.sandbox._host_copy import copy_mounted

    root = tmp_path / "root"
    root.mkdir()
    child = root / "child"
    child.mkdir()
    (child / "file").write_text("original")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("DUMMY_HOST_MARKER")
    from benchflow._utils import reachable_write

    original = reachable_write._open_beneath
    swapped = False

    def opening(parent, path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "child" and not swapped:
            child.rename(root / "saved")
            child.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original(parent, path, flags, *args, **kwargs)

    monkeypatch.setattr(reachable_write, "_open_beneath", opening)
    with pytest.raises(OSError):
        copy_mounted(
            root,
            ("child",) if directory else ("child", "file"),
            tmp_path / "out",
            directory=directory,
        )
    assert not (tmp_path / "out").exists()
