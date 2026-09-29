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

import json
import shlex
from pathlib import Path
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
    """In-memory stand-in that models the symlink-safe put-back flow.

    ``put_back_credentials`` now stages each file in a fresh root-only dir and
    runs the ``python3`` write-back that creates the destination without
    following links (``benchflow.sandbox._credential_writeback``). This fake
    reproduces the observable contract: a destination that reappeared as a
    symlink (content ``None``) is refused; a plain regular file is overwritten
    (the scrub-rollback case); an absent destination is created.
    """

    _STAGE = "/tmp/bf-credstage"

    def __init__(self, fs: FakeFs):
        self.fs = fs
        self.commands: list[str] = []
        self._staging: dict[str, bytes] = {}

    async def run(self, command: str) -> ExecResult:
        self.commands.append(command)
        if "mktemp -d" in command:
            return ExecResult(0, self._STAGE, "")
        if command.startswith("rm -rf"):
            self._staging.clear()
            return ExecResult(0, "", "")
        if command.startswith("rm -f"):
            if self.fs.fail_rm:
                return ExecResult(1, "", "read-only file system")
            for path in list(self.fs.files):
                if f"'{path}'" in command or f" {path}" in command:
                    del self.fs.files[path]
            return ExecResult(0, "", "")
        tokens = shlex.split(command)
        if tokens[:3] == ["python3", "-I", "-c"]:
            return self._writeback(json.loads(tokens[-1]))
        return ExecResult(0, self.fs.listing(), "")

    def _writeback(self, manifest: list[dict]) -> ExecResult:
        refused: list[str] = []
        for entry in manifest:
            dest = entry["path"]
            existing = self.fs.files.get(dest)
            if existing is not None and existing[0] is None:
                # A symlink where a credential file was scrubbed: the swap.
                refused.append(dest)
                continue
            content = self._staging[entry["staged"]]
            self.fs.files[dest] = (content, entry["uid"], entry["gid"], entry["mode"])
        if refused:
            return ExecResult(3, "", "REFUSED: " + ", ".join(refused))
        return ExecResult(0, "", "")

    async def read(self, path: str) -> bytes:
        return self.fs.files[path][0]

    async def write(self, path: str, content: bytes) -> None:
        # Content is staged into the fresh root-only dir, never the destination.
        assert path.startswith(self._STAGE + "/"), path
        self._staging[path] = content


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


async def test_put_back_refuses_a_credential_swapped_for_a_symlink():
    """The attack: after the scrub removes ~/.codex/auth.json, the still-alive
    agent plants a symlink there (to site-packages). The put-back must refuse
    it, not write through the link, and record why."""
    fs = _fs()
    ops = FakeOps(fs)
    stash = await scrub_credentials(ops)
    assert "/root/.codex/auth.json" not in fs.files
    # Agent re-creates the scrubbed path as a symlink (content None).
    fs.files["/root/.codex/auth.json"] = (None, "0", "0", "777")
    with pytest.raises(CredentialScrubError, match="without following a link"):
        await put_back_credentials(ops, stash)
    # The link is left as it was — its target was never written through.
    assert fs.files["/root/.codex/auth.json"] == (None, "0", "0", "777")


def test_credential_writeback_refuses_symlink_swap_on_a_real_filesystem(tmp_path):
    """The syscall-level guard, exercised end to end on a real filesystem.

    A credential path is swapped for a symlink to a victim file; the write-back
    must refuse it (exit 3) and leave the victim untouched, while a sibling
    non-symlinked path is written with the requested mode and owner.
    """
    import os

    from benchflow.sandbox import _credential_writeback

    root = Path(os.path.realpath(tmp_path))  # a symlink-free parent chain
    victim = root / "victim.py"
    victim.write_text("# original, must not change\n")

    codex = root / "home" / ".codex"
    codex.mkdir(parents=True)
    swapped = codex / "auth.json"
    swapped.symlink_to(victim)  # the attacker's redirect

    staged_bad = root / "stage-bad"
    staged_bad.write_bytes(b'{"secret": "x"}')
    staged_ok = root / "stage-ok"
    staged_ok.write_bytes(SECRET)
    ok_dest = root / "home" / ".claude" / ".credentials.json"

    manifest = [
        {
            "staged": str(staged_bad),
            "path": str(swapped),
            "uid": str(os.getuid()),
            "gid": str(os.getgid()),
            "mode": "600",
        },
        {
            "staged": str(staged_ok),
            "path": str(ok_dest),
            "uid": str(os.getuid()),
            "gid": str(os.getgid()),
            "mode": "600",
        },
    ]
    rc = _credential_writeback.main(["_credential_writeback.py", json.dumps(manifest)])
    assert rc == 3  # a link appeared -> refused
    # The victim was not written through the link, and the link still points at it.
    assert victim.read_text() == "# original, must not change\n"
    assert swapped.is_symlink()
    # The clean sibling was still written safely, owner/mode set on the fd.
    assert ok_dest.read_bytes() == SECRET
    assert not ok_dest.is_symlink()
    assert oct(ok_dest.stat().st_mode)[-3:] == "600"


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
