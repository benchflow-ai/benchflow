"""Publication by construction, no reachability classification.

Every controller write below a trusted destination root is resolved through
held descriptors that follow no symbolic link and cross no mount (Linux
openat2 RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_XDEV |
RESOLVE_BENEATH), whether or not a sandbox can reach the path. The trusted
root is the folder the harness or user chose, canonicalised once. There is
no launch-time mount inventory: remote daemons, NFS, named volumes and
shared log folders launch as on integ. The earlier alias, case and
nested-mount inputs (R9 to R11) are refused by the resolution itself.
"""

import json
import os
import shutil
import tempfile
import uuid
from contextlib import suppress
from pathlib import Path, PureWindowsPath

import pytest

from benchflow._utils import reachable_write
from benchflow._utils.reachable_write import ReachablePath
from benchflow.task.paths import RolloutPaths
from tests._trust_admission import admitted
from tests.test_sweep_download_boundary import _backend_env
from tests.test_sweep_taskmd_r8 import BACKENDS, _download, _received
from tests.test_sweep_taskmd_r8 import branch as branch

MARKER = b"DUMMY_CONTROLLER_MARKER"


def _private(base: Path) -> Path:
    private = base / "controller-private"
    private.mkdir()
    (private / "marker.json").write_bytes(MARKER)
    return private


def _intact(private: Path) -> None:
    assert (private / "marker.json").read_bytes() == MARKER
    assert sorted(p.name for p in private.iterdir()) == ["marker.json"]


def _real(path: Path) -> Path:
    return Path(os.path.realpath(path))


@pytest.fixture
def shm_folder():
    """A fresh folder on /dev/shm, a mount below /dev in a Docker container."""
    shm = Path("/dev/shm")
    if not shm.is_dir() or not os.access(shm, os.W_OK):
        pytest.skip("no writable /dev/shm")
    if os.stat(shm).st_dev == os.stat(shm.parent).st_dev:
        pytest.skip("/dev/shm is not a mount here")
    folder = shm / f"bf-r12-{uuid.uuid4().hex}"
    folder.mkdir()
    try:
        yield folder
    finally:
        shutil.rmtree(folder, ignore_errors=True)


# 1. Links and mounts below the trusted root are refused, naming the path --------


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("link", ["absolute", "relative"])
async def test_a_link_below_the_trusted_root_is_refused_on_every_backend(
    tmp_path, monkeypatch, branch, backend, directory, link
):
    """R9-1/R9-2/R10-2/REVIEW-1/R11-1/R11-3 inputs: the agent's results link below
    the output folder the harness chose is never followed, whatever mounts or
    spellings lead there."""
    shared = tmp_path / "srv" / "shared"
    shared.mkdir(parents=True)
    private = _private(tmp_path)
    (shared / "results").symlink_to(
        private if link == "absolute" else os.path.relpath(private, shared),
        target_is_directory=True,
    )
    name = "out" if directory else "marker.json"
    target = ReachablePath.directory(shared) / "results" / name
    env = _backend_env(backend, tmp_path, monkeypatch, directory)
    with pytest.raises((OSError, ValueError, RuntimeError)) as refused:
        await _download(env, backend, directory, target)
    _intact(private)
    if branch == "posix":
        assert str(_real(shared) / "results") in str(refused.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("directory", [False, True])
async def test_a_mount_below_the_trusted_root_is_refused_on_every_backend(
    tmp_path, monkeypatch, shm_folder, backend, directory
):
    """NO_XDEV: /dev/shm is another mount below the root /dev; nothing is written."""
    target = ReachablePath.directory(Path("/dev")) / "shm" / shm_folder.name / "out"
    env = _backend_env(backend, tmp_path, monkeypatch, directory)
    with pytest.raises((OSError, ValueError, RuntimeError)) as refused:
        await _download(env, backend, directory, target)
    assert "/dev/shm is another mount" in str(refused.value)
    assert list(shm_folder.iterdir()) == []


def test_an_honest_destination_with_a_link_below_its_root_is_refused_by_name(tmp_path):
    """R11-4's honest topology: current -> releases/v1 below the chosen root."""
    root = tmp_path / "exports" / "shared"
    (root / "releases" / "v1").mkdir(parents=True)
    (root / "current").symlink_to("releases/v1", target_is_directory=True)
    with pytest.raises(reachable_write.ReachableWriteError) as refused:
        reachable_write.write_host_text(
            ReachablePath.directory(root) / "current" / "output.json", "{}"
        )
    message = str(refused.value)
    assert str(_real(root) / "current") in message and "symbolic link" in message
    assert list((root / "releases" / "v1").iterdir()) == []


def test_a_plain_destination_uses_its_preagent_canonical_root(tmp_path):
    """R13 corrects 7c24f4d88: only an explicitly pre-pinned alias is trusted."""
    real = tmp_path / "private" / "tmp"
    real.mkdir(parents=True)
    (tmp_path / "tmp").symlink_to("private/tmp", target_is_directory=True)
    _pinned = reachable_write.pin_destination(tmp_path / "tmp")
    reachable_write.write_host_text(tmp_path / "tmp" / "py8" / "out.json", "{}")
    assert (real / "py8" / "out.json").read_text() == "{}"


def test_an_explicit_root_keeps_everything_below_it_no_follow(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    private = _private(tmp_path)
    (root / "trial").symlink_to(private, target_is_directory=True)
    capability = reachable_write.destination(root / "trial" / "marker.json", root=root)
    with pytest.raises(reachable_write.ReachableWriteError):
        reachable_write.write_host_text(capability, "{}")
    _intact(private)
    with pytest.raises(ValueError):
        reachable_write.destination(tmp_path / "elsewhere.json", root=root)


def test_a_planted_hard_link_or_fifo_at_the_final_name_is_replaced(tmp_path):
    root = tmp_path / "out"
    root.mkdir()
    shared = tmp_path / "shared.json"
    shared.write_bytes(MARKER)
    os.link(shared, root / "marker.json")
    reachable_write.write_host_text(ReachablePath.directory(root) / "marker.json", "{}")
    assert shared.read_bytes() == MARKER
    assert (root / "marker.json").read_text() == "{}"


# 2. Honest topologies launch as on integ --------------------------------------


class _Daemon:
    """docker ps/inspect/volume inspect answers, so 1f1416ab's inventory runs."""

    def __init__(self, fake, mounts, volumes=()):
        from tests.test_docker_direct_exec import MAIN_ID

        self.fake, self.mounts, self.volumes, self.main = fake, mounts, volumes, MAIN_ID
        self.calls: list[list[str]] = []

    async def __call__(self, *argv, **kwargs):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        proc = await self.fake(*argv, **kwargs)
        out = None
        if argv[:2] == ["docker", "ps"]:
            out = f"{self.main}\n".encode()
        elif argv[:3] == ["docker", "volume", "inspect"]:
            out = json.dumps(list(self.volumes)).encode()
        elif argv[:2] == ["docker", "inspect"]:
            out = json.dumps([{"Id": self.main, "Mounts": self.mounts}]).encode()
        if out is not None:
            proc.communicate.return_value = (out, b"")
        return proc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mounts", "volumes", "logs_shared"),
    [
        # R11-5: a remote daemon sharing the job folder, a remote-only bind.
        (
            [{"Type": "bind", "Source": "/srv/task-data-remote", "Destination": "/w"}],
            [],
            True,
        ),
        # REVIEW-4: a remote daemon not sharing the job folder.
        (
            [{"Type": "bind", "Source": "/srv/task-data-remote", "Destination": "/w"}],
            [],
            False,
        ),
        # Named NFS volume whose mounted source this host cannot see.
        (
            [
                {
                    "Type": "volume",
                    "Name": "nfs",
                    "Source": "/var/lib/docker/volumes/nfs/_data",
                    "Destination": "/d",
                }
            ],
            [
                {
                    "Name": "nfs",
                    "Driver": "local",
                    "Options": {"type": "nfs", "device": ":/x"},
                }
            ],
            True,
        ),
        # A volume driver BenchFlow does not know.
        (
            [
                {
                    "Type": "volume",
                    "Name": "ext",
                    "Source": "/mnt/ext/_data",
                    "Destination": "/d",
                }
            ],
            [{"Name": "ext", "Driver": "vieux/sshfs", "Options": {}}],
            True,
        ),
    ],
)
async def test_honest_topologies_launch_as_on_integ(
    tmp_path, monkeypatch, mounts, volumes, logs_shared
):
    from tests.test_docker_direct_exec import FakeDocker, _sandbox, _started

    daemon = _Daemon(FakeDocker(), mounts, volumes)
    monkeypatch.setattr("asyncio.create_subprocess_exec", daemon)
    sandbox = _sandbox(tmp_path)

    async def probe():
        sandbox._logs_are_mounted = logs_shared

    sandbox._probe_verifier_log_mount = probe
    await _started(sandbox)
    # No launch-time mount inventory: nothing inspects the mounts.
    assert not [c for c in daemon.calls if c[:3] == ["docker", "volume", "inspect"]]


# 3. Staging: no transport writes by pathname near a sandbox mount ---------------


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", [False, True])
async def test_docker_streams_downloads_and_stages_nothing(
    tmp_path, monkeypatch, shm_folder, directory
):
    """R10-1's input: host /tmp is bind-mounted into the task and the destination
    is another mount. Docker streams docker cp's tar through held descriptors,
    so nothing is ever staged by pathname, in /tmp or anywhere."""
    host_tmp = tmp_path / "host-tmp"
    host_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(host_tmp))
    _pinned = reachable_write.pin_destination(shm_folder)
    target = shm_folder / ("out" if directory else "output.json")
    env = _backend_env("docker", tmp_path, monkeypatch, directory)
    await _download(env, "docker", directory, target)
    assert _received(target, directory) == b"submission"
    assert list(host_tmp.iterdir()) == []
    assert [p.name for p in shm_folder.iterdir()] == [target.name]


@pytest.mark.asyncio
async def test_a_docker_stream_with_a_link_member_is_refused(tmp_path, monkeypatch):
    import asyncio
    import io
    import tarfile
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    env = _backend_env("docker", tmp_path, monkeypatch, True)
    private = _private(tmp_path)

    async def docker_cp(*argv, stdout=None, **kwargs):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            folder = tarfile.TarInfo("out")
            folder.type = tarfile.DIRTYPE
            tar.addfile(folder)
            link = tarfile.TarInfo("out/leak")
            link.type = tarfile.SYMTYPE
            link.linkname = str(private)
            tar.addfile(link)
        os.write(stdout, buf.getvalue())
        stderr = asyncio.StreamReader()
        stderr.feed_eof()
        return SimpleNamespace(
            stderr=stderr, wait=AsyncMock(return_value=0), kill=lambda: None
        )

    monkeypatch.setattr("asyncio.create_subprocess_exec", docker_cp)
    target = tmp_path / "dest" / "out"
    with pytest.raises(reachable_write.ReachableWriteError, match="link or special"):
        await env.download_dir("/input", target)
    assert not target.exists()
    assert [p.name for p in (tmp_path / "dest").iterdir()] == []
    _intact(private)


def test_transport_staging_is_the_private_temporary_folder(tmp_path):
    rollout = RolloutPaths(tmp_path / "trial")
    rollout.mkdir()
    for target in (tmp_path / "out" / "x.json", rollout.verifier_dir / "x.json"):
        with reachable_write.staging_for(target) as staging:
            assert staging.parent == _real(Path(tempfile.gettempdir()))
            assert os.stat(staging).st_mode & 0o777 == 0o700
        assert not staging.exists()


def test_rename_staging_never_lands_in_a_harness_mount(tmp_path):
    """Apple mounts the whole rollout folder: rename staging goes above it."""
    rollout = RolloutPaths(tmp_path / "jobs" / "trial")
    rollout.mkdir()
    reachable_write.declare_harness_mount(tmp_path / "jobs" / "trial")
    with reachable_write.private_folder(rollout.verifier_dir) as staging:
        assert staging.parent == _real(tmp_path / "jobs")


# 4. Earlier honest-transfer and root cases keep working --------------------------


@pytest.mark.parametrize(
    "root", [PureWindowsPath("D:\\"), PureWindowsPath("\\\\server\\share\\")]
)
def test_windows_drive_and_unc_roots_have_zero_components(root):
    """R9-3: a drive or UNC root is opened itself, never with a "" component."""
    from benchflow.rewards._safe_evidence import _root_components

    assert _root_components(root) == (None, ())
    assert _root_components(root / "jobs") == (root, ("jobs",))


def test_the_filesystem_root_opens_as_an_evidence_root():
    from benchflow.rewards._safe_evidence import open_directory

    with open_directory(Path("/"), evidence_root=Path("/")) as root:
        assert root.names()


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", [False, True])
async def test_apple_upload_from_an_aliased_temporary_source(tmp_path, directory):
    """R9-5: /tmp -> private/tmp above an upload's source folder is the controller's own."""
    from unittest.mock import patch

    from benchflow.sandbox.apple_container import AppleContainerSandbox
    from tests.test_sweep_download_boundary import _sandbox

    env = _sandbox(AppleContainerSandbox, tmp_path)
    env._container_name = "dummy"
    env.rollout_paths.mkdir()
    (tmp_path / "private" / "tmp" / "py8-upload").mkdir(parents=True)
    (tmp_path / "tmp").symlink_to("private/tmp", target_is_directory=True)
    source = tmp_path / "tmp" / "py8-upload" / ("tree" if directory else "input.txt")
    if directory:
        source.mkdir()
        (source / "input.txt").write_text("upload")
    else:
        source.write_text("upload")
    artifacts = tmp_path / "rollout" / "artifacts"
    with patch.object(env, "_copy", side_effect=AssertionError("CLI copy")):
        if directory:
            await env.upload_dir(
                source, "/logs/artifacts/tree", admission=admitted(env)
            )
            assert (artifacts / "tree" / "input.txt").read_text() == "upload"
        else:
            await env.upload_file(
                source, "/logs/artifacts/input.txt", admission=admitted(env)
            )
            assert (artifacts / "input.txt").read_text() == "upload"


# 5. Real topologies (throwaway container, BENCHFLOW_TEST_TOPOLOGY prepared as root) --


def _topology() -> Path:
    base = os.environ.get("BENCHFLOW_TEST_TOPOLOGY")
    if not base:
        pytest.skip("needs a prepared mount topology (BENCHFLOW_TEST_TOPOLOGY)")
    return Path(base)


@pytest.mark.parametrize(
    "relative",
    [
        # REVIEW-1: exports/output binds a nested tmpfs's output/.
        ("output", "results", "marker.json"),
        # R11-1: exports/v binds the volume's v/ above its unreadable _data.
        ("v", "_data", "results", "marker.json"),
        # REVIEW-3: a casefolded mountpoint spelled otherwise.
        ("ci", "OUTPUT", "results", "marker.json"),
    ],
)
def test_real_alias_topologies_are_refused(relative):
    base = _topology()
    with suppress(OSError, ValueError):
        reachable_write.write_host_text(
            ReachablePath.directory(base / "exports") / Path(*relative), "{}"
        )
    _intact(base / "private")


def test_real_honest_mount_below_the_root_is_refused_by_name():
    base = _topology()
    with pytest.raises(reachable_write.ReachableWriteError, match="another mount"):
        reachable_write.write_host_text(
            ReachablePath.directory(base / "exports") / "output" / "honest.json", "{}"
        )
    assert not (base / "srv" / "shared" / "data" / "output" / "honest.json").exists()


@pytest.mark.parametrize("resolution", ["openat2", "portable"])
def test_both_resolutions_refuse_links_and_mounts(
    tmp_path, monkeypatch, shm_folder, resolution
):
    """The portable walk (O_NOFOLLOW per component, st_dev and mount ID) refuses
    exactly what openat2 refuses."""
    if resolution == "portable":
        monkeypatch.setattr(reachable_write, "_openat2_usable", [False])
    private = _private(tmp_path)
    root = tmp_path / "out"
    root.mkdir()
    (root / "link").symlink_to(private, target_is_directory=True)
    with pytest.raises(reachable_write.ReachableWriteError, match="symbolic link"):
        reachable_write.write_host_text(
            ReachablePath.directory(root) / "link" / "marker.json", "{}"
        )
    _intact(private)
    with pytest.raises(reachable_write.ReachableWriteError, match="another mount"):
        reachable_write.write_host_text(
            ReachablePath.directory(Path("/dev")) / "shm" / shm_folder.name / "x", "{}"
        )
    assert list(shm_folder.iterdir()) == []
    reachable_write.write_host_text(
        ReachablePath.directory(root) / "a" / "b.json", "{}"
    )
    assert (root / "a" / "b.json").read_text() == "{}"
