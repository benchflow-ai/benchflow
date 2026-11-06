"""The folders on PATH are locked to root before the agent runs (Rust 0.8.2).

Every command BenchFlow sends as root finds ``sh`` and its tools through the
sandbox's PATH, and the verifier's PATH keeps the image's folders that only
root can write. An image that leaves a folder on PATH open to others (a
0777 ``/usr/local/cargo/bin``, say) let the agent user replace a program root
runs, and dropped that folder from the verifier's PATH, so a verifier that
needed a tool there ended unscored. ``setup_sandbox_user`` now locks such
folders, and the folders above them, to root, the image's last PATH folder
and the system folders appended after it included; a folder that cannot be
locked is recorded ``failed``.

The lock runs here for real in temp folders: as a plain user ``chown root``
fails, so the folder is reported unlockable; as root (a fresh container) it
is locked.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._exec_receipt import read_receipt
from benchflow.sandbox.lockdown import setup_sandbox_user


def _ok(stdout: str = "") -> MagicMock:
    return MagicMock(return_code=0, stdout=stdout, stderr="")


def _run_as_root_in_sh(command: str, path: str) -> str:
    """Run a command BenchFlow sent as root under /bin/sh, with this PATH."""
    done = subprocess.run(
        ["/bin/sh", "-c", command],
        capture_output=True,
        text=True,
        env={"PATH": path},
        timeout=60,
        check=False,
    )
    # Lockdown reads the report from a receipt: give back what the command
    # itself printed, as production does.
    found = re.search(r"BF-RECEIPT-([0-9a-f]{24}) ", done.stdout)
    if found:
        result = ExecResult(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )
        return read_receipt(result, found.group(1)).stdout or ""
    return done.stdout


async def _path_lock_command(monkeypatch, root: Path, workspace: Path) -> str | None:
    """The root command setup_sandbox_user sends that locks PATH's folders."""
    # The test's folders live under the temp folder, which the lock leaves
    # alone in a real sandbox; this test lets it look there, and stops its
    # walk up at the test's own folder. The test host's own system folders
    # (on the test's PATH for sh's tools, and appended by the lock) are left
    # alone instead, so a run as root changes nothing outside the test.
    monkeypatch.setattr(
        lockdown,
        "_PATH_LOCK_SKIP",
        tuple(lockdown._SYSTEM_PATH_FOLDERS.split(":")),
        raising=False,
    )
    monkeypatch.setattr(lockdown, "_PATH_LOCK_TOP", str(root.resolve()), raising=False)
    env = MagicMock()
    env.supports_system_shell = True
    env.exec = AsyncMock(return_value=_ok())
    await setup_sandbox_user(env, "agent", str(workspace), guard_daemon=False)
    for call in env.exec.call_args_list:
        command = call.args[0] if call.args else call.kwargs.get("command", "")
        if call.kwargs.get("user") == "root" and "BF locked-path" in command:
            return command
    return None


def _reported(out: str) -> list[str]:
    """The folders the lock locked or could not lock."""
    return [
        line.split(" ", 2)[2]
        for line in out.splitlines()
        if line.startswith(("BF locked-path ", "BF unlockable-path "))
    ]


def _open_folder(path: Path) -> Path:
    path.mkdir(parents=True)
    path.chmod(0o777)
    return path


@pytest.mark.asyncio
async def test_the_last_path_folder_open_to_others_is_locked(tmp_path, monkeypatch):
    workspace = tmp_path / "app"
    workspace.mkdir()
    open_bin = _open_folder(tmp_path / "opt" / "tool" / "bin")
    closed_bin = tmp_path / "closed" / "bin"
    closed_bin.mkdir(parents=True)
    closed_bin.chmod(0o755)
    for d in (tmp_path / "opt", tmp_path / "opt" / "tool", tmp_path / "closed"):
        d.chmod(0o755)

    command = await _path_lock_command(monkeypatch, tmp_path, workspace)
    assert command is not None, "setup_sandbox_user locks no folder on PATH"

    # The open folder is the image's LAST PATH entry: the system folders
    # appended after it must not stay glued to it.
    out = _run_as_root_in_sh(command, f"{closed_bin}:/usr/bin:/bin:{open_bin}")
    reported = _reported(out)
    assert str(open_bin) in reported, out
    if os.geteuid() == 0:
        assert f"BF locked-path {open_bin}" in out
        assert not open_bin.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        # Root's own closed folder is left as it is.
        assert str(closed_bin) not in reported
    else:
        # As a plain user chown fails: the folder is named, not hidden.
        assert f"BF unlockable-path {open_bin}" in out
    assert "/usr/bin" not in reported and "/bin" not in reported


@pytest.mark.asyncio
async def test_an_open_folder_above_a_path_folder_is_locked(tmp_path, monkeypatch):
    workspace = tmp_path / "app"
    workspace.mkdir()
    cargo = _open_folder(tmp_path / "cargo")
    cargo_bin = cargo / "bin"
    cargo_bin.mkdir()
    cargo_bin.chmod(0o755)

    command = await _path_lock_command(monkeypatch, tmp_path, workspace)
    assert command is not None, "setup_sandbox_user locks no folder on PATH"

    out = _run_as_root_in_sh(command, f"{cargo_bin}:/usr/bin:/bin")
    # Whoever can write the folder above can move bin aside and put another
    # in its place.
    assert str(cargo) in _reported(out), out
    if os.geteuid() == 0:
        assert str(cargo_bin) not in _reported(out), out


@pytest.mark.asyncio
async def test_the_workspace_and_homes_on_path_are_left_alone(tmp_path, monkeypatch):
    workspace = _open_folder(tmp_path / "app")
    venv_bin = _open_folder(workspace / ".venv" / "bin")

    command = await _path_lock_command(monkeypatch, tmp_path, workspace)
    assert command is not None, "setup_sandbox_user locks no folder on PATH"

    out = _run_as_root_in_sh(command, f"{venv_bin}:/usr/bin:/bin")
    assert not any(
        folder == str(workspace) or folder.startswith(f"{workspace}/")
        for folder in _reported(out)
    ), out


def test_temp_folders_are_never_locked():
    command = lockdown._lock_system_path_cmd("/app")
    for folder in ("/tmp", "/var/tmp", "/dev/shm"):
        assert f"|{folder}|{folder}/*" in command


def test_the_lock_is_posix_sh():
    command = lockdown._lock_system_path_cmd("/app")
    done = subprocess.run(
        ["/bin/sh", "-n", "-c", command], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    # No Python in the image is needed.
    assert "python" not in command


@pytest.mark.asyncio
async def test_a_linked_path_entry_locks_the_folder_that_holds_the_link(
    tmp_path, monkeypatch
):
    """Guards PT-R3-A-1 at c2db61b8, not only links to PATH programs."""
    workspace = tmp_path / "app"
    workspace.mkdir()
    holder = _open_folder(tmp_path / "opt1")
    target = tmp_path / "srv/venvs/py/bin"
    target.mkdir(parents=True)
    (holder / "venv").symlink_to(target.parent)
    command = await _path_lock_command(monkeypatch, tmp_path, workspace)
    assert command is not None
    out = _run_as_root_in_sh(command, f"{holder}/venv/bin:/usr/bin:/bin")
    assert str(holder) in _reported(out), out
    if os.geteuid() == 0:
        assert not holder.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)


@pytest.mark.asyncio
async def test_the_record_says_repaired_or_failed(tmp_path, monkeypatch):
    from benchflow._utils import path_lock_record
    from benchflow.sandbox._exec_receipt import encode_receipt, unwrap_receipt_command

    def answering(lock_stdout: str):
        """An exec that answers the lock's receipt read with *lock_stdout*."""

        async def exec_(command, **kwargs):
            wrapped = unwrap_receipt_command(command)
            if wrapped is not None and "BF locked-path" in wrapped[0]:
                return _ok(encode_receipt(lock_stdout.encode(), 0, wrapped[1]))
            return _ok()

        return AsyncMock(side_effect=exec_)

    env = MagicMock()
    env.supports_system_shell = True
    env.exec = answering("BF locked-path /usr/local/bin\n")
    record: dict = {}
    with path_lock_record.recording_into(record):
        await setup_sandbox_user(env, "agent", "/app", guard_daemon=False)
    assert record["state"] == "enforced"
    assert record["locked"] == ["/usr/local/bin"]
    assert record["note"].startswith("repaired:")

    env.exec = answering("BF unlockable-path /opt/tool/bin\n")
    record = {}
    with path_lock_record.recording_into(record):
        await setup_sandbox_user(env, "agent", "/app", guard_daemon=False)
    assert record["state"] == "failed"
    assert record["unlockable"] == ["/opt/tool/bin"]
    json.dumps(record)


@pytest.mark.asyncio
async def test_the_oracle_and_nop_get_the_lock_too(monkeypatch):
    """guard_daemon=False is the oracle's and nop's setup: still locked, so the
    verifier keeps the same tool folders whoever ran."""
    command = None
    monkeypatch.setattr(lockdown, "_PATH_LOCK_SKIP", (), raising=False)
    env = MagicMock()
    env.supports_system_shell = True
    env.exec = AsyncMock(return_value=_ok())
    await setup_sandbox_user(env, "agent", "/app", guard_daemon=False)
    for call in env.exec.call_args_list:
        if "BF locked-path" in (call.args[0] if call.args else ""):
            command = call.args[0]
    assert command is not None


@pytest.mark.asyncio
async def test_a_program_others_may_write_in_a_path_folder_is_locked(
    tmp_path, monkeypatch
):
    """The official rust image's cargo is 0777 in its 0777 folder: locking the
    folder alone would leave a program the agent can rewrite in place, in a
    folder the verifier's PATH then trusts."""
    workspace = tmp_path / "app"
    workspace.mkdir()
    tool_bin = tmp_path / "cargo" / "bin"
    tool_bin.mkdir(parents=True)
    tool_bin.chmod(0o755)
    (tmp_path / "cargo").chmod(0o755)
    cargo = tool_bin / "cargo"
    cargo.write_text("#!/bin/sh\necho cargo\n")
    cargo.chmod(0o777)
    closed = tool_bin / "rustc"
    closed.write_text("#!/bin/sh\necho rustc\n")
    closed.chmod(0o755)

    command = await _path_lock_command(monkeypatch, tmp_path, workspace)
    assert command is not None, "setup_sandbox_user locks no folder on PATH"

    out = _run_as_root_in_sh(command, f"{tool_bin}:/usr/bin:/bin")
    # The test owns the file, so chmod works as a plain user too.
    assert not cargo.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    assert cargo.stat().st_mode & stat.S_IXUSR
    if os.geteuid() == 0:
        assert f"BF locked-path {cargo}" in out, out
        assert str(closed) not in _reported(out)
    else:
        # A plain user's file is not root's: chown fails, so it is named.
        assert f"BF unlockable-path {cargo}" in out, out


@pytest.mark.asyncio
async def test_programs_in_the_workspace_are_left_alone(tmp_path, monkeypatch):
    workspace = tmp_path / "app"
    venv_bin = workspace / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    tool = venv_bin / "tool"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o777)

    command = await _path_lock_command(monkeypatch, tmp_path, workspace)
    assert command is not None, "setup_sandbox_user locks no folder on PATH"

    _run_as_root_in_sh(command, f"{venv_bin}:/usr/bin:/bin")
    assert tool.stat().st_mode & stat.S_IWOTH


def _tool_folder(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "app"
    workspace.mkdir()
    tool_bin = tmp_path / "local" / "bin"
    tool_bin.mkdir(parents=True)
    for d in (tmp_path / "local", tool_bin):
        d.chmod(0o755)
    return workspace, tool_bin


def _program(path: Path, mode: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho tool\n")
    path.chmod(mode)
    return path


@pytest.mark.asyncio
async def test_a_program_another_user_owns_in_a_path_folder_is_locked(
    tmp_path, monkeypatch
):
    """Guards PT-R1-B-3: a program unpacked as root from an archive keeps the
    archive's uid (often 1000, the uid useradd then gives the agent user), so
    the agent could rewrite it though no group or other write bit is set."""
    workspace, tool_bin = _tool_folder(tmp_path)
    tool = _program(tool_bin / "tool", 0o755)
    if os.geteuid() == 0:
        os.chown(tool, 1000, 1000)

    command = await _path_lock_command(monkeypatch, tmp_path, workspace)
    assert command is not None, "setup_sandbox_user locks no folder on PATH"

    out = _run_as_root_in_sh(command, f"{tool_bin}:/usr/bin:/bin")
    if os.geteuid() == 0:
        assert f"BF locked-path {tool}" in out, out
        assert tool.stat().st_uid == 0
        assert not tool.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    else:
        # The test's own file is another user's: chown fails, so it is named.
        assert f"BF unlockable-path {tool}" in out, out


@pytest.mark.asyncio
async def test_a_link_on_path_is_judged_by_the_file_it_resolves_to(
    tmp_path, monkeypatch
):
    """Guards PT-R1-B-3: ``-type f`` skipped a link in a PATH folder, so a link
    to a file (in a folder) others may write let the agent replace what root
    runs through it. The file and its folder are locked now."""
    workspace, tool_bin = _tool_folder(tmp_path)
    target_dir = _open_folder(tmp_path / "opt" / "x")
    (tmp_path / "opt").chmod(0o755)
    target = _program(target_dir / "tool2", 0o777)
    link = tool_bin / "tool2"
    link.symlink_to(target)

    command = await _path_lock_command(monkeypatch, tmp_path, workspace)
    assert command is not None, "setup_sandbox_user locks no folder on PATH"

    out = _run_as_root_in_sh(command, f"{tool_bin}:/usr/bin:/bin")
    reported = _reported(out)
    assert f"{link} -> {target}" in reported, out
    assert str(target_dir) in reported, out
    # chmod works on the test's own file and folder as a plain user too.
    assert not target.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    assert not target_dir.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if os.geteuid() == 0:
        assert f"BF locked-path {link} -> {target}" in out, out
        assert f"BF locked-path {target_dir}" in out, out


@pytest.mark.asyncio
async def test_a_link_into_the_workspace_is_reported_not_locked(tmp_path, monkeypatch):
    """The agent user owns the workspace, so a program on PATH that resolves
    there cannot be made root's alone; the lock names it and leaves the
    workspace as it is."""
    workspace, tool_bin = _tool_folder(tmp_path)
    ws_bin = _open_folder(workspace / "bin")
    target = _program(ws_bin / "tool3", 0o777)
    (tool_bin / "tool3").symlink_to(target)

    command = await _path_lock_command(monkeypatch, tmp_path, workspace)
    assert command is not None, "setup_sandbox_user locks no folder on PATH"

    out = _run_as_root_in_sh(command, f"{tool_bin}:/usr/bin:/bin")
    assert any(
        line.startswith("BF unlockable-path ") and str(workspace) in line
        for line in out.splitlines()
    ), out
    assert not any(
        line.startswith("BF locked-path ") and str(workspace) in line
        for line in out.splitlines()
    ), out
    assert target.stat().st_mode & stat.S_IWOTH
    assert ws_bin.stat().st_mode & stat.S_IWOTH


@pytest.mark.asyncio
async def test_the_folder_of_a_middle_hop_in_a_link_chain_is_locked(
    tmp_path, monkeypatch
):
    """Guards PT-R2-B-1: ``py -> /opt/venv/bin/python -> python3`` resolves to
    root's ``python3``, so only that file's folder was checked; whoever may
    write ``/opt/venv/bin`` (a venv made for the image's user) could put
    another ``python`` in the chain and have root run it. Each hop's folder is
    locked now, for a relative hop as well as an absolute one."""
    workspace, tool_bin = _tool_folder(tmp_path)
    python3 = _program(tool_bin / "python3", 0o755)
    venv_bin = _open_folder(tmp_path / "opt" / "venv" / "bin")
    for d in (tmp_path / "opt", tmp_path / "opt" / "venv"):
        d.chmod(0o755)
    (venv_bin / "python").symlink_to(python3)
    (tool_bin / "py").symlink_to(venv_bin / "python")
    rel_bin = _open_folder(tmp_path / "opt" / "rel" / "bin")
    (tmp_path / "opt" / "rel").chmod(0o755)
    (rel_bin / "python").symlink_to(python3)
    (tool_bin / "py2").symlink_to(Path("..", "..", "opt", "rel", "bin", "python"))

    command = await _path_lock_command(monkeypatch, tmp_path, workspace)
    assert command is not None, "setup_sandbox_user locks no folder on PATH"

    out = _run_as_root_in_sh(command, f"{tool_bin}:/usr/bin:/bin")
    reported = _reported(out)
    assert str(venv_bin) in reported, out
    assert str(rel_bin) in reported, out
    # chmod works on the test's own folders as a plain user too.
    assert not venv_bin.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    assert not rel_bin.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if os.geteuid() == 0:
        assert f"BF locked-path {venv_bin}" in out, out


@pytest.mark.asyncio
async def test_a_folder_link_in_a_chain_is_followed_hop_by_hop(tmp_path, monkeypatch):
    """A link among the folders of a hop (``/opt/cur -> /opt/v2``, itself a
    link to the real folder) is a chain too: the folder each of its hops sits
    in is locked."""
    workspace, tool_bin = _tool_folder(tmp_path)
    real = tmp_path / "srv" / "real"
    real.mkdir(parents=True)
    for d in (tmp_path / "srv", real):
        d.chmod(0o755)
    tool = _program(real / "tool", 0o755)
    opt = tmp_path / "opt"
    opt.mkdir()
    opt.chmod(0o755)
    hop = _open_folder(tmp_path / "hop")
    (hop / "v2").symlink_to(real)
    (opt / "cur").symlink_to(hop / "v2")
    (tool_bin / "tool").symlink_to(opt / "cur" / "tool")
    assert (tool_bin / "tool").resolve() == tool.resolve()

    command = await _path_lock_command(monkeypatch, tmp_path, workspace)
    assert command is not None, "setup_sandbox_user locks no folder on PATH"

    out = _run_as_root_in_sh(command, f"{tool_bin}:/usr/bin:/bin")
    assert str(hop) in _reported(out), out
    assert not hop.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)


@pytest.mark.asyncio
async def test_a_dangling_link_the_agent_could_fill_is_reported(tmp_path, monkeypatch):
    """Guards PT-R2-B-1: a link to a file in a folder that does not exist yet
    (``/usr/local/bin/tool3 -> /tmp/bfx/tool3``) was dropped, since ``readlink
    -f`` fails on it, so the agent could make the folder and the file and have
    root run it while the record said enforced. It is reported now, and the
    open folder it would be made in is locked when it can be."""
    workspace, tool_bin = _tool_folder(tmp_path)
    workspace.chmod(0o777)
    into_ws = tool_bin / "tool3"
    into_ws.symlink_to(workspace / "bfx" / "tool3")
    open_dir = _open_folder(tmp_path / "shared")
    (tool_bin / "tool4").symlink_to(open_dir / "sub" / "tool4")

    command = await _path_lock_command(monkeypatch, tmp_path, workspace)
    assert command is not None, "setup_sandbox_user locks no folder on PATH"

    out = _run_as_root_in_sh(command, f"{tool_bin}:/usr/bin:/bin")
    assert f"BF unlockable-path {into_ws} -> {workspace.resolve()}/bfx/tool3" in out, (
        out
    )
    assert str(open_dir) in _reported(out), out
    assert not open_dir.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    # The workspace is left as it is.
    assert workspace.stat().st_mode & stat.S_IWOTH


@pytest.mark.asyncio
async def test_a_link_past_the_folder_link_depth_is_reported(tmp_path, monkeypatch):
    """Guards PT-CONF-4 at 7a220231: a PATH entry reached through nested
    folder links (``bin -> h1/l/d``, ``h1/l -> h2/l/d``, ... ``h4/l -> tree``)
    left the folder holding the last link unchecked and unreported, so the
    agent that owned it could swap the link while the record said enforced."""
    workspace = tmp_path / "app"
    workspace.mkdir()
    tree = tmp_path / "srv" / "tree"
    (tree / "d" / "d" / "d" / "d").mkdir(parents=True)
    agents = _open_folder(tmp_path / "h4")
    (agents / "l").symlink_to(tree)
    below = agents / "l"
    for n in (3, 2, 1):
        holder = tmp_path / f"h{n}"
        holder.mkdir()
        holder.chmod(0o755)
        (holder / "l").symlink_to(below / "d")
        below = holder / "l"
    entry_holder = tmp_path / "e"
    entry_holder.mkdir()
    (entry_holder / "bin").symlink_to(below / "d")
    assert (entry_holder / "bin").resolve() == (tree / "d" / "d" / "d" / "d")

    command = await _path_lock_command(monkeypatch, tmp_path, workspace)
    assert command is not None
    out = _run_as_root_in_sh(command, f"{entry_holder}/bin:/usr/bin:/bin")
    reported = _reported(out)
    assert str(agents) in reported or f"{agents}/l" in out, out
