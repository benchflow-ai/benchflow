"""Focused contract tests for setup_sandbox_user() shell command generation."""

import re
import shlex
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._sandbox import setup_sandbox_user
from benchflow.agents.registry import get_sandbox_home_dirs


async def _run_setup_sandbox_user(
    *, sandbox_user: str = "agent", workspace: str = "/app"
):
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))

    await setup_sandbox_user(env, sandbox_user, workspace)

    env.exec.assert_awaited_once()
    return env.exec.call_args.args[0], env.exec.call_args.kwargs


def _assert_conditional_legacy_symlink(cmd: str, *, source: str, dest: str) -> None:
    """Legacy tool dirs should link only when a root-only install exists."""
    assert re.search(
        rf"if \[ -e [\"']?{re.escape(source)}[\"']? \].*ln -s(?:f|[a-zA-Z-])* [\"']?{re.escape(source)}[\"']? [\"']?{re.escape(dest)}[\"']?.*fi",
        cmd,
    ), f"expected explicit symlink from {source} to {dest} in setup command: {cmd}"


def _get_copy_loop_dirs(cmd: str) -> list[str]:
    """Extract the general home-dir copy loop payload from the shell command."""
    match = re.search(r"for d in (?P<dirs>.*?); do", cmd)
    assert match, f"expected general home-dir copy loop in setup command: {cmd}"
    return match.group("dirs").split()


class TestSetupSandboxUser:
    @pytest.mark.asyncio
    async def test_setup_command_avoids_recursive_root_tool_copies(self):
        """Heavy root-owned tool dirs should no longer be recursively copied."""
        cmd, kwargs = await _run_setup_sandbox_user()

        assert "cp -aL /root/.local/bin/." not in cmd
        assert "cp -a /root/.nvm/." not in cmd
        assert kwargs["timeout_sec"] == 120

    @pytest.mark.asyncio
    async def test_setup_command_still_creates_user_prepares_home_and_chowns_workspace(
        self,
    ):
        """The non-copy setup contract still creates the user and grants access."""
        cmd, _ = await _run_setup_sandbox_user()

        assert "id -u agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent" in cmd
        assert "mkdir -p /home/agent/.local/bin" not in cmd
        assert "chown -R agent:agent /home/agent" in cmd
        assert f"chown -R agent:agent {shlex.quote('/app')}" in cmd

    @pytest.mark.asyncio
    async def test_setup_command_keeps_heavy_root_tool_dirs_on_shared_paths(self):
        """Legacy root-only tool dirs should use conditional symlinks, not duplication."""
        cmd, _ = await _run_setup_sandbox_user()

        _assert_conditional_legacy_symlink(
            cmd,
            source="/root/.local/bin",
            dest="/home/agent/.local/bin",
        )
        _assert_conditional_legacy_symlink(
            cmd, source="/root/.nvm", dest="/home/agent/.nvm"
        )
        assert "cp -aL /root/.local/bin/. /home/agent/.local/bin/" not in cmd
        assert "cp -a /root/.nvm/. /home/agent/.nvm/" not in cmd

    @pytest.mark.asyncio
    async def test_setup_command_copy_loop_excludes_local_dir(self):
        """General home-dir copying should narrow to small config/auth dirs only."""
        cmd, _ = await _run_setup_sandbox_user()

        copy_loop_dirs = _get_copy_loop_dirs(cmd)

        assert copy_loop_dirs == sorted(
            d for d in get_sandbox_home_dirs() if d != ".local"
        )
        assert ".local" not in copy_loop_dirs
        assert "mkdir -p /home/agent/$d" in cmd
        assert "cp -a /root/$d/. /home/agent/$d/ 2>/dev/null || true" in cmd

    @pytest.mark.asyncio
    async def test_setup_command_does_not_copy_heavy_tool_trees_into_home(self):
        """BenchFlow-installed agents must not rely on sandbox-home tool copies.

        Agent binaries are placed in /usr/local/bin by the registered install_cmd
        values, so setup_sandbox_user() must not bulk-copy heavyweight tool trees
        (e.g. /root/.nvm, /root/.local/bin) to make them executable for the user.
        """
        cmd, _ = await _run_setup_sandbox_user()

        for heavy_source in ("/root/.nvm", "/root/.local/bin"):
            assert f"cp -a {heavy_source}/." not in cmd
            assert f"cp -aL {heavy_source}/." not in cmd

        copy_loop_dirs = _get_copy_loop_dirs(cmd)
        for heavy_dir in (".nvm", ".local"):
            assert heavy_dir not in copy_loop_dirs, (
                f"sandbox copy loop must not include heavy tool dir {heavy_dir!r}"
            )
