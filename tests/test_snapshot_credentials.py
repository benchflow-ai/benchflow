"""Agent credential files never enter a container snapshot.

Regression test: a container snapshot captured everything ``install_agent``
wrote, including ``~/.codex/auth.json`` (codex-acp) and
``~/.claude/.credentials.json`` (host subscription login), so a kept Daytona
snapshot would store the credential at rest in provider storage. Snapshot
capture now reads those files into host memory, removes them, captures, and
writes them back (content, owner, mode) to the live sandbox and to every
sandbox restored from that snapshot.

Unit tests over an in-memory filesystem; no Docker, Daytona or credentials.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._snapshot_credentials import (
    CredentialScrubError,
    put_back_credentials,
    scrub_credentials,
)
from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDirect
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import ExecResult, SandboxImage
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths

SECRET = b'{"tokens": {"refresh_token": "fake-refresh-value"}}'


class FakeFs:
    """Just enough of a container to answer the scrub's commands."""

    def __init__(self, files: dict[str, tuple[bytes, str, str, str]]):
        # path -> (content, uid, gid, mode); symlinks: content None
        self.files = dict(files)
        self.fail_rm = False

    def listing(self) -> str:
        from benchflow.agents.credentials import CREDENTIAL_EVIDENCE_PATHS

        lines = []
        for path, (content, uid, gid, mode) in sorted(self.files.items()):
            home = (
                "/root" if path.startswith("/root/") else "/".join(path.split("/")[:3])
            )
            rel = path[len(home) + 1 :]
            if not any(
                rel == cand or rel.startswith(cand + "/")
                for cand in CREDENTIAL_EVIDENCE_PATHS
            ):
                continue
            kind = "symbolic link" if content is None else "regular file"
            lines.append(f"{uid}:{gid}:{mode}:{kind}:{path}")
        return "\n".join(lines) + ("\n" if lines else "")


class FakeOps:
    def __init__(self, fs: FakeFs):
        self.fs = fs
        self.commands: list[str] = []

    async def run(self, command: str) -> ExecResult:
        self.commands.append(command)
        if command.startswith("rm -f"):
            if self.fs.fail_rm:
                return ExecResult(1, "", "read-only file system")
            for path in list(self.fs.files):
                if f"'{path}'" in command or f" {path}" in command:
                    del self.fs.files[path]
            return ExecResult(0, "", "")
        if command.startswith("chown"):
            _, owner, _, _, mode, path = command.replace("&&", "").split()
            uid, gid = owner.split(":")
            content = self.fs.files[path.strip("'")][0]
            self.fs.files[path.strip("'")] = (content, uid, gid, mode)
            return ExecResult(0, "", "")
        if command.startswith("mkdir"):
            return ExecResult(0, "", "")
        return ExecResult(0, self.fs.listing(), "")

    async def read(self, path: str) -> bytes:
        return self.fs.files[path][0]

    async def write(self, path: str, content: bytes) -> None:
        self.fs.files[path] = (content, "0", "0", "644")


def _fs() -> FakeFs:
    return FakeFs(
        {
            "/root/.codex/auth.json": (SECRET, "0", "0", "600"),
            "/home/agent/.claude/.credentials.json": (SECRET, "1000", "1000", "600"),
            "/root/notes.txt": (b"not a credential", "0", "0", "644"),
        }
    )


async def test_scrub_removes_credentials_and_put_back_restores_them():
    fs = _fs()
    ops = FakeOps(fs)
    stash = await scrub_credentials(ops)
    assert sorted(f.path for f in stash) == [
        "/home/agent/.claude/.credentials.json",
        "/root/.codex/auth.json",
    ]
    assert set(fs.files) == {"/root/notes.txt"}
    assert all(SECRET not in command.encode() for command in ops.commands)
    await put_back_credentials(ops, stash)
    assert fs.files["/root/.codex/auth.json"] == (SECRET, "0", "0", "600")
    assert fs.files["/home/agent/.claude/.credentials.json"] == (
        SECRET,
        "1000",
        "1000",
        "600",
    )
    assert "fake-refresh-value" not in repr(stash)


async def test_no_credentials_means_no_changes():
    fs = FakeFs({"/root/notes.txt": (b"x", "0", "0", "644")})
    assert await scrub_credentials(FakeOps(fs)) == []
    assert set(fs.files) == {"/root/notes.txt"}


async def test_symlinked_credential_is_refused_before_anything_changes():
    fs = _fs()
    fs.files["/root/.claude/.credentials.json"] = (None, "0", "0", "777")
    with pytest.raises(CredentialScrubError, match="symbolic link"):
        await scrub_credentials(FakeOps(fs))
    assert "/root/.codex/auth.json" in fs.files


async def test_failed_removal_puts_files_back_and_refuses():
    fs = _fs()
    fs.fail_rm = True
    with pytest.raises(CredentialScrubError, match="could not remove"):
        await scrub_credentials(FakeOps(fs))
    assert fs.files["/root/.codex/auth.json"][0] == SECRET


@pytest.fixture
def docker_sandbox(tmp_path):
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    return DockerSandbox(
        environment_dir=environment,
        environment_name="scrub",
        session_id="bf-scrub",
        rollout_paths=paths,
        task_env_config=SandboxConfig(),
    )


async def test_docker_commit_sees_no_credentials_and_restore_puts_them_back(
    docker_sandbox, monkeypatch
):
    fs = _fs()
    committed: list[dict] = []
    docker_sandbox._main_container_id = AsyncMock(return_value="old")
    docker_sandbox._credential_ops = lambda container: FakeOps(fs)

    class Proc:
        returncode = 0

        async def communicate(self):
            committed.append(dict(fs.files))
            return b"sha256:" + b"0" * 64, b""

    async def fake_exec(*args, **kwargs):
        assert args[:2] == ("docker", "commit")
        return Proc()

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    image = await docker_sandbox.snapshot()
    assert set(committed[0]) == {"/root/notes.txt"}
    assert fs.files["/root/.codex/auth.json"][0] == SECRET  # live world intact

    # A restore replaces the container: the new one starts without the files.
    fs.files = {"/root/notes.txt": (b"not a credential", "0", "0", "644")}
    docker_sandbox._inspect_container = AsyncMock(
        return_value={"HostConfig": {}, "Config": {}}
    )
    docker_sandbox._docker_cli = AsyncMock(return_value=ExecResult(0, "", ""))
    monkeypatch.setattr(
        "benchflow.sandbox.docker._replayed_run_args", lambda *a, **k: []
    )
    await docker_sandbox.restore(image)
    assert fs.files["/root/.codex/auth.json"] == (SECRET, "0", "0", "600")

    # Once the snapshot is released, the host copy is dropped too.
    await docker_sandbox.delete_snapshot(image)
    assert docker_sandbox._snapshot_credentials == {}


async def test_daytona_snapshot_sees_no_credentials_and_restore_puts_them_back(
    monkeypatch,
):
    pytest.importorskip("daytona")  # sandbox-daytona optional dependency
    from benchflow.sandbox import daytona as daytona_mod

    daytona_mod._load_daytona_sdk()
    fs = _fs()
    captured: list[dict] = []

    async def create_snapshot(name):
        captured.append(dict(fs.files))

    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.logger = daytona_mod.logger
    sandbox.environment_name = "scrub"
    sandbox._sandbox = SimpleNamespace(
        id="sb-1", _experimental_create_snapshot=create_snapshot, delete=AsyncMock()
    )
    strategy = _DaytonaDirect(sandbox)
    sandbox._strategy = strategy
    strategy._credential_ops = lambda: FakeOps(fs)
    image = await sandbox.snapshot()
    assert set(captured[0]) == {"/root/notes.txt"}
    assert fs.files["/root/.codex/auth.json"][0] == SECRET

    fs.files = {"/root/notes.txt": (b"not a credential", "0", "0", "644")}
    sandbox._auto_delete_interval = 0
    sandbox._auto_stop_interval = 0
    sandbox._network_block_all = False
    sandbox._create_sandbox = AsyncMock()
    await sandbox.restore(image)
    assert fs.files["/home/agent/.claude/.credentials.json"] == (
        SECRET,
        "1000",
        "1000",
        "600",
    )
    assert isinstance(image, SandboxImage)


# ── adopt_snapshot: credentials for an image this sandbox did not take ──
# (tests/test_branch_reuse_snapshot.py): a fork that reuses a kept checkpoint
# as its snapshot needs the live sandbox's credential files for restores.


async def test_docker_adopt_snapshot_remembers_the_live_credentials(
    docker_sandbox, monkeypatch
):
    fs = _fs()
    docker_sandbox._main_container_id = AsyncMock(return_value="c")
    docker_sandbox._credential_ops = lambda container: FakeOps(fs)
    image = SandboxImage(provider="docker", ref="bf-snap-kept")
    await docker_sandbox.adopt_snapshot(image)
    # Nothing in the live sandbox changed; the stash is keyed by the image.
    assert fs.files["/root/.codex/auth.json"][0] == SECRET
    assert docker_sandbox._snapshot_credentials["bf-snap-kept"]


async def test_daytona_adopt_snapshot_remembers_the_live_credentials():
    pytest.importorskip("daytona")
    from benchflow.sandbox import daytona as daytona_mod

    daytona_mod._load_daytona_sdk()
    fs = _fs()
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.logger = daytona_mod.logger
    sandbox._sandbox = SimpleNamespace(id="sb-1")
    strategy = _DaytonaDirect(sandbox)
    sandbox._strategy = strategy
    strategy._credential_ops = lambda: FakeOps(fs)
    await sandbox.adopt_snapshot(SandboxImage(provider="daytona", ref="bf-snap-kept"))
    assert fs.files["/root/.codex/auth.json"][0] == SECRET
    assert strategy._snapshot_credentials["bf-snap-kept"]
