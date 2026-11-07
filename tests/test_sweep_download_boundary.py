"""Guards R7-2 at a2052fa6 through export and backend download entry points."""

import asyncio
import base64
import io
import os
import tarfile
from contextlib import suppress
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths


def _sandbox(cls, tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    with patch.object(cls, "preflight"):
        return cls(
            environment_dir=tmp_path,
            environment_name="dummy",
            session_id="s",
            rollout_paths=RolloutPaths(tmp_path / "rollout"),
            task_env_config=SandboxConfig(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["file", "directory", "fifo"])
async def test_apple_skill_export_never_admits_controller_bytes(
    tmp_path, monkeypatch, kind
):
    """Guards a2052fa6's real cleanup export/capture path with dummy controller data."""
    from benchflow.rollout._user_loop import _export_generated_skills
    from benchflow.sandbox.apple_container import AppleContainerSandbox

    env = _sandbox(AppleContainerSandbox, tmp_path)
    env._container_name = "dummy"
    source = tmp_path / "rollout/artifacts/generated"
    source.mkdir(parents=True)
    marker = tmp_path / "outside/SKILL.md"
    marker.parent.mkdir()
    marker.write_text("DUMMY_CONTROLLER_MARKER")
    if kind == "directory":
        (source / "leak").symlink_to(marker.parent, target_is_directory=True)
    else:
        (source / "leak").mkdir()
        if kind == "file":
            (source / "leak/SKILL.md").symlink_to(marker)
        else:
            os.mkfifo(source / "leak/SKILL.md")
    rollout = SimpleNamespace(
        _env=env,
        _evolved_skills={},
        _config=SimpleNamespace(
            export_generated_skills_to=tmp_path / "export",
            generated_skills_root="/logs/artifacts/generated",
            recorded_skill_mode=None,
        ),
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    with suppress(OSError, RuntimeError, ValueError):
        await _export_generated_skills(rollout)
    assert "DUMMY_CONTROLLER_MARKER" not in str(rollout._evolved_skills)
    for item in (tmp_path / "export").rglob("*"):
        if item.is_file() and not item.is_symlink():
            assert b"DUMMY_CONTROLLER_MARKER" not in item.read_bytes()


def _backend_env(backend, tmp_path, monkeypatch, directory, payload=b"submission"):
    """A real backend whose transport writes ``payload`` where it is told to."""

    def materialize(path):
        path = Path(path)
        if directory:
            path.mkdir(parents=True, exist_ok=True)
            path = path / "file"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    if backend in ("docker", "remote-docker"):
        from benchflow.sandbox.docker import DockerSandbox
        from benchflow.sandbox.remote_docker import (
            RemoteDockerHost,
            RemoteDockerSandbox,
        )

        cls = DockerSandbox if backend == "docker" else RemoteDockerSandbox
        env = object.__new__(cls)
        if backend == "remote-docker":
            env.docker_host = RemoteDockerHost("ssh://fixture.invalid")
        env._chown_to_host_user = AsyncMock()
        env._main_container = "a" * 12
        env._docker_compose_global_args = lambda: []
        env._docker_compose_env = lambda: {}
        env._compose_transport_env = lambda: {}

        async def docker_cp(*argv, stdout=None, **kwargs):
            if argv[-2:] == ("system", "dial-stdio"):
                return SimpleNamespace(
                    returncode=0,
                    communicate=AsyncMock(
                        return_value=(
                            b"HTTP/1.1 200 OK\r\n"
                            b"X-Docker-Container-Path-Stat: eyJtb2RlIjo0MjB9\r\n\r\n",
                            b"",
                        )
                    ),
                )
            # `docker cp CONTAINER:SRC -`: the copied entry as a tar on stdout.
            assert argv[-1] == "-" and isinstance(stdout, int)
            base = PurePosixPath(argv[-2].split(":", 1)[1]).name
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                if directory:
                    folder = tarfile.TarInfo(base)
                    folder.type = tarfile.DIRTYPE
                    folder.mode = 0o755
                    tar.addfile(folder)
                    name = f"{base}/file"
                else:
                    name = base
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                member.mode = 0o644
                tar.addfile(member, io.BytesIO(payload))
            os.write(stdout, buf.getvalue())
            stderr = asyncio.StreamReader()
            stderr.feed_eof()
            return SimpleNamespace(
                stderr=stderr, wait=AsyncMock(return_value=0), kill=lambda: None
            )

        monkeypatch.setattr("asyncio.create_subprocess_exec", docker_cp)
    elif backend == "apple":
        from benchflow.sandbox.apple_container import AppleContainerSandbox

        env = _sandbox(AppleContainerSandbox, tmp_path)
        env._container_name = "dummy"

        async def copying(source, target, **kwargs):
            materialize(target)

        env._copy = copying
    elif backend.startswith("daytona"):
        from benchflow.sandbox import daytona
        from benchflow.sandbox.daytona import DaytonaSandbox

        monkeypatch.setattr(daytona, "FileDownloadRequest", SimpleNamespace)
        env = object.__new__(DaytonaSandbox)
        env.logger = Mock()
        fs = SimpleNamespace(
            list_files=AsyncMock(return_value=[]),
            search_files=AsyncMock(
                side_effect=lambda root, pattern: SimpleNamespace(
                    files=[root + "/file"]
                )
            ),
            get_file_info=AsyncMock(return_value=SimpleNamespace(is_dir=False)),
        )

        async def file_copy(source, destination):
            materialize(destination)

        async def files_copy(*, files):
            # One successful response per request, as the SDK reports it.
            for request in files:
                Path(request.destination).parent.mkdir(parents=True, exist_ok=True)
                Path(request.destination).write_bytes(payload)
            return [
                SimpleNamespace(
                    source=request.source, error=None, result=request.destination
                )
                for request in files
            ]

        fs.download_file = file_copy
        fs.download_files = files_copy
        env._require_sandbox = lambda: SimpleNamespace(fs=fs)
        if backend == "daytona":
            from benchflow.sandbox.daytona_strategies import _DaytonaDirect

            env._strategy = _DaytonaDirect(env)
        else:
            from benchflow.sandbox.daytona_dind import _DaytonaDinD

            strategy = object.__new__(_DaytonaDinD)
            strategy._env = env
            # Guards de80dad: a successful transport identifies the container
            # and supplies Docker's regular-file metadata before publication.
            strategy._vm_exec = AsyncMock(
                return_value=ExecResult(
                    stdout="HTTP/1.1 200 OK\r\n"
                    "X-Docker-Container-Path-Stat: eyJtb2RlIjo0MjB9\r\n\r\n",
                    return_code=0,
                )
            )
            strategy._compose_exec = AsyncMock(
                side_effect=lambda argv, **_: ExecResult(
                    stdout="capture-container\n"
                    if argv == ["ps", "--all", "-q", "main"]
                    else "",
                    return_code=0,
                )
            )
            env._strategy = strategy
    elif backend == "modal":
        from benchflow.sandbox.modal_impl import ModalSandbox

        env = object.__new__(ModalSandbox)

        async def copy_to_local(remote_path, local_path):
            Path(local_path).write_bytes(payload)

        # Sandbox.filesystem: Modal refuses the legacy Sandbox.open.
        env._sandbox = SimpleNamespace(
            filesystem=SimpleNamespace(
                copy_to_local=SimpleNamespace(aio=AsyncMock(side_effect=copy_to_local))
            )
        )
        from tests.receipt_fakes import receipt_answer

        # The listing crosses as a receipt of NUL-framed names.
        listing = ExecResult(stdout="/input/file\0", return_code=0)
        env.exec = AsyncMock(
            side_effect=lambda command, *a, **k: receipt_answer(
                command, lambda read: listing
            )
        )
    else:
        from benchflow.sandbox.agentcore import AgentCoreSandbox

        env = _sandbox(AgentCoreSandbox, tmp_path)
        if directory:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                member = tarfile.TarInfo("file")
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))
            payload = buf.getvalue()
        env.exec = AsyncMock(
            return_value=ExecResult(
                stdout=base64.b64encode(payload).decode(), return_code=0
            )
        )
    return env


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend",
    [
        "docker",
        "remote-docker",
        "apple",
        "daytona",
        "daytona-dind-log",
        "daytona-dind-compose",
        "modal",
        "agentcore",
    ],
)
@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("destination", ["safe", "link"])
async def test_backend_download_publishes_through_no_follow_destination(
    tmp_path, monkeypatch, backend, directory, destination
):
    """Guards a2052fa6: real public backend methods cannot overwrite an ancestor-link marker."""
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "file"
    marker.write_bytes(b"DUMMY_CONTROLLER_MARKER")
    # The link sits where a sandbox can plant one: under a rollout's
    # verifier/ (R8-1 canonicalises only the controller-owned prefix).
    # The harness makes the rollout's folders, so their identities are known.
    links = RolloutPaths(tmp_path / "links")
    links.mkdir()
    alias = links.rollout_dir / "verifier" / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    # The rollout's own capability: the trusted root is verifier/, and the
    # planted link below it is never followed.
    target = (
        links.verifier_dir / "alias"
        if directory
        else links.verifier_dir / "alias" / "file"
    )
    if destination == "safe":
        target = tmp_path / "normal"
    remote = "/logs/artifacts" if backend == "daytona-dind-log" else "/input"

    env = _backend_env(backend, tmp_path, monkeypatch, directory)
    with suppress(OSError, RuntimeError, ValueError):
        if directory:
            await env.download_dir(remote, target)
        else:
            await env.download_file(remote + "/file", target)
    assert marker.read_bytes() == b"DUMMY_CONTROLLER_MARKER"
    if destination == "safe":
        assert (target / "file" if directory else target).read_bytes() == b"submission"


@pytest.mark.parametrize("directory", [False, True])
def test_source_root_ancestors_are_opened_without_following(tmp_path, directory):
    """Guards R7-2's shared copier at b72df1ed, including ancestors of its source root.

    The trusted source root is the rollout's artifacts/ folder; the link a
    sandbox planted below it is never followed. The root itself is
    canonicalised once.
    """
    from benchflow.sandbox._host_copy import copy_mounted

    outside = tmp_path / "outside"
    (outside / "nested").mkdir(parents=True)
    (outside / "nested/file").write_text("DUMMY_CONTROLLER_MARKER")
    links = RolloutPaths(tmp_path / "links")
    links.mkdir()
    alias = links.rollout_dir / "artifacts" / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    target = tmp_path / "out"
    with pytest.raises((OSError, ValueError)):
        copy_mounted(
            links.rollout_dir / "artifacts",
            ("alias", "nested") if directory else ("alias", "nested", "file"),
            target,
            directory=directory,
        )
    assert not target.exists()


def _received_tree(tmp_path, name):
    source = tmp_path / "received"
    source.mkdir()
    (source / "safe").write_text("submission")
    (source / name).write_text("dummy")
    return source


@pytest.mark.parametrize("bad_name", [r"C:\dummy", "dummy\x01"])
def test_received_tree_is_preflighted_before_publication(tmp_path, bad_name):
    """Guards R7-2's native-transport publication boundary at b72df1ed."""
    from benchflow.sandbox._host_copy import copy_mounted

    source = _received_tree(tmp_path, bad_name)
    target = tmp_path / "out"
    with pytest.raises((OSError, ValueError)):
        copy_mounted(source, (), target, directory=True)
    assert not target.exists()


WINDOWS_ONLY = ["dummy:stream", "NUL", "dummy.", "LONGNA~1", "aux.py"]


@pytest.mark.parametrize("bad_name", WINDOWS_ONLY)
def test_received_tree_is_preflighted_on_windows(tmp_path, monkeypatch, bad_name):
    """Guards R7-2 at b72df1ed on a Windows controller, where these are aliases."""
    from benchflow._utils import archive_paths
    from benchflow.sandbox._host_copy import copy_mounted

    monkeypatch.setattr(archive_paths, "os", SimpleNamespace(name="nt"))
    source = _received_tree(tmp_path, bad_name)
    target = tmp_path / "out"
    with pytest.raises((OSError, ValueError)):
        copy_mounted(source, (), target, directory=True)
    assert not target.exists()


@pytest.mark.skipif(os.name == "nt", reason="these names are aliases on Windows")
@pytest.mark.parametrize("name", WINDOWS_ONLY)
def test_received_windows_names_publish_on_posix(tmp_path, name):
    """A downloaded verifier or artifacts
    tree holding such a name is published as written on a POSIX controller,
    as in 0.7.11, instead of failing the download."""
    from benchflow.sandbox._host_copy import copy_mounted

    source = _received_tree(tmp_path, name)
    target = tmp_path / "out"
    copy_mounted(source, (), target, directory=True)
    assert (target / "safe").read_text() == "submission"
    assert (target / name).read_text() == "dummy"


def test_download_copy_preserves_file_and_directory_metadata(tmp_path):
    """Guards b72df1ed's new backend boundary against dropping transport metadata."""
    from benchflow.sandbox._host_copy import copy_mounted

    source = tmp_path / "source"
    source.mkdir()
    (source / "file").write_bytes(b"submission")
    moment = 1234567890123456789
    for path in (source / "file", source):
        path.chmod(0o751)
        os.utime(path, ns=(moment, moment))
    target = tmp_path / "out"
    copy_mounted(source, (), target, directory=True)
    assert (target / "file").read_bytes() == b"submission"
    for path in (target / "file", target):
        assert path.stat().st_mode & 0o777 == 0o751
        assert path.stat().st_mtime_ns == moment


@pytest.mark.parametrize("newline", [None, "", "\n", "\r", "\r\n"])
def test_reachable_text_preserves_path_newline_contract(tmp_path, newline):
    """Guards the b72df1ed merge: diagnostic fixtures use Path.write_text(newline=...)."""
    from benchflow._utils.reachable_write import ReachablePath

    root = tmp_path / "output"
    root.mkdir()
    expected = tmp_path / "expected"
    marker = tmp_path / "marker"
    marker.write_text("DUMMY_CONTROLLER_MARKER")
    (root / "log").symlink_to(marker)
    data = "alpha\nbeta\r\ngamma\n"
    expected_count = expected.write_text(data, newline=newline)
    result = (ReachablePath.directory(root) / "log").write_text(data, newline=newline)
    assert result == expected_count
    assert (root / "log").read_bytes() == expected.read_bytes()
    assert marker.read_text() == "DUMMY_CONTROLLER_MARKER"
