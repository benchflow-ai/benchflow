"""Guard PT-R3-B-2 at c2db61b8: resolve missing tails on BusyBox too."""

import stat

import pytest

from tests.test_locked_system_path import _path_lock_command, _run_as_root_in_sh


@pytest.mark.asyncio
async def test_dangling_link_through_a_folder_link_is_reported_not_chmodded(
    tmp_path, monkeypatch
):
    """Guards c2db61b8 without needing a second host readlink implementation."""
    workspace = tmp_path / "app"
    workspace.mkdir(mode=0o777)
    workspace.chmod(0o777)
    holder = tmp_path / "opt"
    holder.mkdir()
    (holder / "x").symlink_to(workspace)
    programs = tmp_path / "bin"
    programs.mkdir()
    (programs / "tool").symlink_to(holder / "x/a/b/tool")
    command = await _path_lock_command(monkeypatch, tmp_path, workspace)
    assert command is not None
    # BusyBox lacks -m; its -f fails on the two missing components. GNU -e
    # gives the same failure for this fixture while retaining real resolution
    # of existing ancestors and links. No tested command is replaced by echo.
    shim = 'readlink() { case "$1" in -m) return 1;; -f) /usr/bin/readlink -e "$2";; *) /usr/bin/readlink "$@";; esac; };\n'
    out = _run_as_root_in_sh(shim + command, f"{programs}:/usr/bin:/bin")
    assert f"BF unlockable-path {workspace}" in out, out
    assert not any(
        line.startswith(f"BF locked-path {holder}/x") for line in out.splitlines()
    )
    assert stat.S_IMODE(workspace.stat().st_mode) == 0o777
