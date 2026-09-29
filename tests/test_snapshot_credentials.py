"""Agent credential files never enter a container snapshot, and the scrub never
acts as root on a path the agent controls.

Regression tests: a container snapshot captured everything ``install_agent``
wrote, including ``~/.codex/auth.json`` (codex-acp) and
``~/.claude/.credentials.json`` (host subscription login), so a kept Daytona
snapshot would store the credential at rest in provider storage. Snapshot
capture now reads those files into host memory, removes them, captures, and
writes them back to the live sandbox and to every sandbox restored from that
snapshot.

The agent's processes can be alive during a checkpoint, so every read, removal
and write runs as the owner of the credential home (never as root on a path the
agent controls), a symbolic link on the path is refused and named, a newline in
a file name cannot forge a listing entry, and a restored mode never carries
setuid bits.

The provider tests run over an in-memory filesystem (no Docker, Daytona or
credentials); the script tests run the real scrub scripts with ``sh`` on Linux.
"""

from __future__ import annotations

import base64
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._snapshot_credentials import (
    AsUserResult,
    CredentialScrubError,
    StashedCredential,
    _read_script,
    _remove_script,
    _write_script,
    put_back_credentials,
    scrub_credentials,
)
from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDirect
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import ExecResult, SandboxImage
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths

SECRET = b'{"tokens": {"refresh_token": "fake-refresh-value"}}'

# Owners of the credential homes in the fake container.
HOMES = {"/root": ("0", "0"), "/home/agent": ("1000", "1000")}


class FakeFs:
    """Just enough of a container to answer the scrub's commands."""

    def __init__(self, files: dict[str, tuple[bytes | None, str, str, str]]):
        # path -> (content, uid, gid, mode); symlinks: content None
        self.files = dict(files)
        self.fail_rm = False
        self.newline_in: str | None = None  # a home whose tree has a newline name
        self.forged: list[str] = []  # extra raw listing lines

    def listing(self) -> str:
        from benchflow.agents.credentials import CREDENTIAL_EVIDENCE_PATHS

        lines: list[str] = []
        seen: set[str] = set()
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
            lines.append(f"F:{uid}:{gid}:{mode}:{kind}:{path}")
            seen.add(home)
        if self.newline_in:
            lines.insert(0, f"!newline:{self.newline_in}")
        lines.extend(self.forged)
        for home in sorted(seen):
            uid, gid = HOMES[home]
            lines.append(f"H:{uid}:{gid}:755:directory:{home}")
        return "\n".join(lines) + ("\n" if lines else "")


class FakeOps:
    """Runs the scrub against FakeFs, recording which uid ran each action.

    ``run_as`` models the privilege-separated scripts (``read``, ``remove``,
    ``write`` of one path; a symbolic link on it is refused). A write run as
    root restores the recorded owner; one run as a user leaves the file owned
    by that user, as ``sh`` would.
    """

    def __init__(self, fs: FakeFs):
        self.fs = fs
        self.commands: list[str] = []
        self.scripts: list[str] = []
        self.actions: list[tuple[str, int, str]] = []  # (action, uid, path)

    async def run(self, command: str) -> ExecResult:
        self.commands.append(command)
        return ExecResult(0, self.fs.listing(), "")

    async def run_as(
        self, uid: int, gid: int, script: str, *, stdin: bytes | None = None
    ) -> AsUserResult:
        self.scripts.append(script)
        action = script.split("\n", 1)[0].removeprefix("# benchflow-credential ")
        values = {
            name: shlex.split(value)[0]
            for name, value in re.findall(r"^(p|m|o)=(.*)$", script, flags=re.M)
        }
        path = values["p"]
        self.actions.append((action, int(uid), path))
        existing = self.fs.files.get(path)
        if existing is not None and existing[0] is None:
            return AsUserResult(4, b"", f"refused: {path} is a symbolic link")
        if action == "read":
            if existing is None or existing[0] is None:
                return AsUserResult(4, b"", f"refused: {path} is not a regular file")
            return AsUserResult(0, base64.b64encode(existing[0]), "")
        if action == "remove":
            if self.fs.fail_rm:
                return AsUserResult(1, b"", "read-only file system")
            self.fs.files.pop(path, None)
            return AsUserResult(0, b"", "")
        assert action == "write", action
        assert stdin is not None
        owner = values["o"].split(":") if int(uid) == 0 else [str(uid), str(gid)]
        self.fs.files[path] = (base64.b64decode(stdin), owner[0], owner[1], values["m"])
        return AsUserResult(0, b"", "")


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
    await put_back_credentials(ops, stash)
    assert fs.files["/root/.codex/auth.json"] == (SECRET, "0", "0", "600")
    assert fs.files["/home/agent/.claude/.credentials.json"] == (
        SECRET,
        "1000",
        "1000",
        "600",
    )
    # Contents never travel in a command or a script, only over stdin/stdout.
    assert all(SECRET not in text.encode() for text in ops.commands + ops.scripts)
    assert "fake-refresh-value" not in repr(stash)


async def test_nothing_under_an_agent_home_runs_as_root():
    fs = _fs()
    ops = FakeOps(fs)
    await put_back_credentials(ops, await scrub_credentials(ops))
    assert {action for action, _, _ in ops.actions} == {"read", "remove", "write"}
    for action, uid, path in ops.actions:
        assert uid == (0 if path.startswith("/root/") else 1000), (action, uid, path)


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
    """The attack: after the scrub removes the agent's credential file, the
    still-running agent plants a symlink there (to a sitecustomize.py). The
    put-back runs as the agent, refuses the link, names it, and never writes
    through it."""
    fs = _fs()
    ops = FakeOps(fs)
    stash = await scrub_credentials(ops)
    swapped = "/home/agent/.claude/.credentials.json"
    fs.files[swapped] = (None, "1000", "1000", "777")
    with pytest.raises(CredentialScrubError, match="symbolic link"):
        await put_back_credentials(ops, stash)
    assert fs.files[swapped] == (None, "1000", "1000", "777")
    assert ("write", 1000, swapped) in ops.actions
    # The other credential was still restored.
    assert fs.files["/root/.codex/auth.json"] == (SECRET, "0", "0", "600")


async def test_a_newline_in_a_credential_file_name_refuses_the_snapshot():
    # A directory name holding a newline makes `stat` print a second line the
    # agent wrote, e.g. "1000:1000:666:regular file:/etc/shadow".
    fs = _fs()
    fs.newline_in = "/home/agent"
    ops = FakeOps(fs)
    with pytest.raises(CredentialScrubError, match="newline"):
        await scrub_credentials(ops)
    assert ops.actions == []  # nothing read or removed
    assert "/root/.codex/auth.json" in fs.files


async def test_a_listing_entry_outside_the_credential_paths_refuses_the_snapshot():
    fs = _fs()
    fs.forged = ["F:1000:1000:666:regular file:/etc/shadow"]
    ops = FakeOps(fs)
    with pytest.raises(CredentialScrubError, match="unexpected credential path"):
        await scrub_credentials(ops)
    assert ops.actions == []


async def test_put_back_never_restores_setuid_bits():
    fs = FakeFs({})
    ops = FakeOps(fs)
    item = StashedCredential(
        "/root/.ssh/tool", "0", "0", "4755", b"#!/bin/sh\n", "0", "0"
    )
    await put_back_credentials(ops, [item])
    assert fs.files["/root/.ssh/tool"][3] == "755"


# ── The scrub's shell scripts, run for real ──────────────────────────────
# Linux ``sh`` with ``stat -c`` and ``base64``, as the current user (a
# non-root credential home owner); skipped elsewhere (e.g. macOS).


def _posix_tools() -> bool:
    if sys.platform != "linux" or shutil.which("sh") is None:
        return False
    try:
        probe = subprocess.run(["stat", "-c", "%u", "/"], capture_output=True)
    except FileNotFoundError:
        return False
    return probe.returncode == 0


needs_posix_sh = pytest.mark.skipif(
    not _posix_tools(), reason="runs the scrub's sh scripts: Linux sh, stat -c"
)


def _sh(script: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sh", "-c", script], input=stdin, capture_output=True, check=False
    )


@needs_posix_sh
def test_write_script_refuses_a_symlink_swap_and_writes_nothing(tmp_path):
    root = Path(os.path.realpath(tmp_path))
    victim = root / "sitecustomize.py"
    victim.write_text("# original\n")
    codex = root / "home" / ".codex"
    codex.mkdir(parents=True)
    swapped = codex / "auth.json"
    swapped.symlink_to(victim)
    uid, gid = os.getuid(), os.getgid()
    result = _sh(
        _write_script(uid, str(swapped), "600", f"{uid}:{gid}"),
        base64.b64encode(b"import os  # agent code\n"),
    )
    assert result.returncode == 4 and b"symbolic link" in result.stderr
    assert victim.read_text() == "# original\n" and swapped.is_symlink()


@needs_posix_sh
def test_write_script_refuses_a_symlinked_parent_directory(tmp_path):
    root = Path(os.path.realpath(tmp_path))
    site_packages = root / "site-packages"
    site_packages.mkdir()
    home = root / "home"
    home.mkdir()
    (home / ".codex").symlink_to(site_packages)
    uid, gid = os.getuid(), os.getgid()
    result = _sh(
        _write_script(uid, str(home / ".codex" / "auth.json"), "600", f"{uid}:{gid}"),
        base64.b64encode(SECRET),
    )
    assert result.returncode == 4 and b"symbolic link" in result.stderr
    assert not (site_packages / "auth.json").exists()


@needs_posix_sh
def test_read_script_refuses_a_symlinked_credential(tmp_path):
    root = Path(os.path.realpath(tmp_path))
    secret_elsewhere = root / "shadow"
    secret_elsewhere.write_text("root-only\n")
    (root / "home").mkdir()
    link = root / "home" / "auth.json"
    link.symlink_to(secret_elsewhere)
    result = _sh(_read_script(os.getuid(), str(link)))
    assert result.returncode == 4 and b"root-only" not in result.stdout


@needs_posix_sh
def test_scripts_round_trip_a_credential_with_its_mode(tmp_path):
    root = Path(os.path.realpath(tmp_path))
    path = root / "home" / ".config" / "gcloud" / "adc.json"  # parents created
    uid, gid = os.getuid(), os.getgid()
    wrote = _sh(
        _write_script(uid, str(path), "640", f"{uid}:{gid}"), base64.b64encode(SECRET)
    )
    assert wrote.returncode == 0, wrote.stderr
    assert path.read_bytes() == SECRET
    assert oct(path.stat().st_mode & 0o777) == "0o640"
    read = _sh(_read_script(uid, str(path)))
    assert read.returncode == 0
    assert base64.b64decode(b"".join(read.stdout.split())) == SECRET
    removed = _sh(_remove_script(uid, str(path)))
    assert removed.returncode == 0 and not path.exists()


@needs_posix_sh
def test_scripts_refuse_to_run_as_another_user():
    if os.getuid() == 0:
        pytest.skip("the check needs a non-root caller")
    # A root script run by a user (a provider that failed to switch) stops.
    result = _sh(_read_script(0, "/etc/hostname"))
    assert result.returncode == 98 and b"not running as uid 0" in result.stderr


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
