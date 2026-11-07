"""Guard CONFIRM-taskmd-rewards-R8-1 and R8-2 at f2367f92 on every publishing backend.

R8-1: a download to a destination under an OS alias above the trusted root
(macOS /tmp -> private/tmp, /var/folders) was refused, because every component
from / was opened with O_NOFOLLOW. R8-2: Windows publication staged in the
system temporary folder and renamed onto the destination's volume, which fails
across volumes. The fix canonicalises only the controller-owned prefix, keeps
no-follow traversal from the first sandbox-reachable root down, and stages on
the destination's own filesystem. Linux simulates the alias with a symlinked
prefix, the second volume with /dev/shm (tmpfs), and the Windows branch by
giving reachable_write an ``os`` whose ``name`` is ``"nt"``.
"""

import os
import shutil
import tempfile
import uuid
from contextlib import suppress
from pathlib import Path

import pytest

from benchflow._utils import reachable_write
from benchflow.task.paths import RolloutPaths
from tests._trust_admission import admitted
from tests.test_sweep_download_boundary import _backend_env

BACKENDS = [
    "docker",
    "remote-docker",
    "apple",
    "daytona",
    "daytona-dind-log",
    "daytona-dind-compose",
    "modal",
    "agentcore",
]
MARKER = b"DUMMY_CONTROLLER_MARKER"


class _WindowsOs:
    """``os`` as reachable_write sees it on Windows; everything else is the real module."""

    name = "nt"

    def __getattr__(self, attribute):
        return getattr(os, attribute)


@pytest.fixture(params=["posix", "nt"])
def branch(request, monkeypatch):
    if request.param == "nt":
        monkeypatch.setattr(reachable_write, "os", _WindowsOs())
    return request.param


def _remote(backend):
    return "/logs/artifacts" if backend == "daytona-dind-log" else "/input"


async def _download(env, backend, directory, target):
    if directory:
        await env.download_dir(_remote(backend), target)
    else:
        await env.download_file(_remote(backend) + "/file", target)


def _received(target, directory):
    return (Path(target) / "file" if directory else Path(target)).read_bytes()


def _no_staging_left(folder: Path):
    assert not [p for p in folder.iterdir() if p.name.startswith(".benchflow-")]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("directory", [False, True])
async def test_honest_download_under_an_aliased_temporary_root(
    tmp_path, monkeypatch, branch, backend, directory
):
    """R8-1: /tmp -> private/tmp style alias above the trusted root must not refuse."""
    real = tmp_path / "private" / "tmp"
    real.mkdir(parents=True)
    alias = tmp_path / "tmp"
    alias.symlink_to("private/tmp", target_is_directory=True)
    _pinned = reachable_write.pin_destination(alias)
    target = alias / "py8" / ("out" if directory else "output.json")
    env = _backend_env(backend, tmp_path, monkeypatch, directory)
    await _download(env, backend, directory, target)
    canonical = real / "py8" / target.name
    assert _received(canonical, directory) == b"submission"
    assert _received(target, directory) == b"submission"
    _no_staging_left(real / "py8")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("link", ["absolute", "relative"])
# The harness's own form: the rollout capability. A plain path names the
# caller's trusted root itself: see tests/test_sweep_taskmd_r12.py.
@pytest.mark.parametrize("form", ["capability"])
async def test_a_link_planted_below_the_trusted_root_is_still_refused(
    tmp_path, monkeypatch, branch, backend, directory, link, form
):
    """R8-1 keeps R7-2/LAUNCHERS-C1: a sandbox link under verifier/ is never followed."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_bytes(MARKER)
    (outside / "output.json").write_bytes(MARKER)
    # The rollout folder sits under an aliased prefix too: the alias above is
    # resolved, the planted link below the reachable root is not.
    (tmp_path / "jobs-real").mkdir()
    (tmp_path / "jobs").symlink_to("jobs-real", target_is_directory=True)
    paths = RolloutPaths(tmp_path / "jobs" / "trial")
    paths.mkdir()
    planted = tmp_path / "jobs-real" / "trial" / "verifier" / "planted"
    planted.symlink_to(
        outside if link == "absolute" else os.path.relpath(outside, planted.parent),
        target_is_directory=True,
    )
    name = "out" if directory else "output.json"
    target = (
        tmp_path / "jobs" / "trial" / "verifier" / "planted" / name
        if form == "path"
        else paths.verifier_dir / "planted" / name
    )
    env = _backend_env(backend, tmp_path, monkeypatch, directory)
    with pytest.raises((OSError, ValueError, TypeError, RuntimeError)):
        await _download(env, backend, directory, target)
    assert (outside / "file").read_bytes() == MARKER
    assert (outside / "output.json").read_bytes() == MARKER
    assert sorted(p.name for p in outside.iterdir()) == ["file", "output.json"]


@pytest.fixture
def other_volume(tmp_path):
    """A destination folder on a filesystem other than the system temporary folder."""
    shm = Path("/dev/shm")
    if not shm.is_dir() or not os.access(shm, os.W_OK):
        pytest.skip("no writable /dev/shm for a second filesystem")
    folder = shm / f"bf-r8-{uuid.uuid4().hex}"
    folder.mkdir()
    try:
        if os.stat(folder).st_dev == os.stat(tempfile.gettempdir()).st_dev:
            pytest.skip("/dev/shm shares the temporary folder's filesystem")
        yield folder
    finally:
        shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("directory", [False, True])
async def test_honest_download_onto_another_filesystem(
    tmp_path, monkeypatch, other_volume, branch, backend, directory
):
    """R8-2: staging is on the destination's filesystem, so publication never crosses devices."""
    _pinned = reachable_write.pin_destination(other_volume)
    target = other_volume / "jobs" / ("out" if directory else "output.json")
    env = _backend_env(backend, tmp_path, monkeypatch, directory)
    await _download(env, backend, directory, target)
    assert _received(target, directory) == b"submission"
    _no_staging_left(other_volume / "jobs")


def test_staging_on_another_filesystem_is_refused(tmp_path, other_volume):
    """The code asserts staging and destination share st_dev before publication."""
    destination = other_volume / "dest"
    destination.mkdir()
    with (
        pytest.raises(reachable_write.ReachableWriteError, match="filesystem"),
        reachable_write._private_staging(
            [tmp_path], os.stat(destination).st_dev, temp=False
        ),
    ):
        pass
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".benchflow-")]


@pytest.mark.parametrize("directory", [False, True])
def test_staging_for_shares_the_destination_device(tmp_path, other_volume, directory):
    _pinned = reachable_write.pin_destination(other_volume)
    for target in (
        other_volume / "a" / "out",
        RolloutPaths(other_volume / "trial").verifier_dir / "out",
    ):
        with reachable_write.staging_for(target, same_filesystem=True) as staging:
            capability = reachable_write.destination(target)
            assert os.stat(staging).st_dev == os.stat(capability._anchor.path).st_dev
            assert os.stat(staging).st_mode & 0o777 == 0o700
        assert not staging.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", [False, True])
async def test_apple_upload_into_logs_is_held_at_the_rollout_folder(
    tmp_path, directory
):
    """Apple uploads to /logs publish through a capability, never a resolved path."""
    from unittest.mock import patch

    from benchflow.sandbox.apple_container import AppleContainerSandbox
    from tests.test_sweep_download_boundary import _sandbox

    env = _sandbox(AppleContainerSandbox, tmp_path)
    env._container_name = "dummy"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x").write_bytes(MARKER)
    verifier = tmp_path / "rollout" / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "planted").symlink_to(outside, target_is_directory=True)
    source = tmp_path / "upload"
    if directory:
        source.mkdir()
        (source / "x").write_text("upload")
    else:
        source.write_text("upload")
    with (
        patch.object(env, "_copy", side_effect=AssertionError("CLI copy")),
        pytest.raises((OSError, ValueError)),
    ):
        if directory:
            await env.upload_dir(
                source, "/logs/verifier/planted/x", admission=admitted(env)
            )
        else:
            await env.upload_file(
                source, "/logs/verifier/planted/x", admission=admitted(env)
            )
    assert (outside / "x").read_bytes() == MARKER
    with patch.object(env, "_copy", side_effect=AssertionError("CLI copy")):
        if directory:
            await env.upload_dir(
                source, "/logs/verifier/sub/x", admission=admitted(env)
            )
            assert (verifier / "sub" / "x" / "x").read_text() == "upload"
        else:
            await env.upload_file(
                source, "/logs/verifier/sub/x", admission=admitted(env)
            )
            assert (verifier / "sub" / "x").read_text() == "upload"


@pytest.mark.asyncio
async def test_failed_transfer_leaves_no_staging(tmp_path, monkeypatch):
    env = _backend_env("docker", tmp_path, monkeypatch, False)

    async def failing(*argv, **kwargs):
        raise RuntimeError("transport failed")

    monkeypatch.setattr("asyncio.create_subprocess_exec", failing)
    target = tmp_path / "dest" / "output.json"
    with suppress(RuntimeError):
        await env.download_file("/input/file", target)
    assert not target.exists()
    _no_staging_left(tmp_path)
