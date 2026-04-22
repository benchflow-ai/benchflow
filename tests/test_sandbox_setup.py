"""Focused contract tests for setup_sandbox_user() shell command generation."""

import re
import shlex
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._sandbox import setup_sandbox_user


async def _run_setup_sandbox_user(*, sandbox_user: str = "agent", workspace: str = "/app"):
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))

    await setup_sandbox_user(env, sandbox_user, workspace)

    env.exec.assert_awaited_once()
    return env.exec.call_args.args[0], env.exec.call_args.kwargs


def _assert_explicit_symlink(cmd: str, *, source: str, dest: str) -> None:
    """Heavy tool dirs must use an explicit symlink compatibility path."""
    assert re.search(
        rf"ln -s(?:f|[a-zA-Z-])* [\"']?{re.escape(source)}[\"']? [\"']?{re.escape(dest)}[\"']?",
        cmd,
    ), f"expected explicit symlink from {source} to {dest} in setup command: {cmd}"


class TestSetupSandboxUser:
    @pytest.mark.asyncio
    async def test_setup_command_avoids_recursive_root_tool_copies(self):
        """Heavy root-owned tool dirs should no longer be recursively copied."""
        cmd, kwargs = await _run_setup_sandbox_user()

        assert "cp -aL /root/.local/bin/." not in cmd
        assert "cp -a /root/.nvm/." not in cmd
        assert kwargs["timeout_sec"] == 120

    @pytest.mark.asyncio
    async def test_setup_command_still_creates_user_prepares_home_and_chowns_workspace(self):
        """The non-copy setup contract still creates the user and grants access."""
        cmd, _ = await _run_setup_sandbox_user()

        assert "id -u agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent" in cmd
        assert "mkdir -p /home/agent/.local/bin" in cmd
        assert "chown -R agent:agent /home/agent" in cmd
        assert f"chown -R agent:agent {shlex.quote('/app')}" in cmd

    @pytest.mark.asyncio
    async def test_setup_command_keeps_heavy_root_tool_dirs_on_shared_paths(self):
        """Heavy root-owned tool dirs should use explicit symlinks, not duplication."""
        cmd, _ = await _run_setup_sandbox_user()

        _assert_explicit_symlink(
            cmd,
            source="/root/.local/bin",
            dest="/home/agent/.local/bin",
        )
        _assert_explicit_symlink(cmd, source="/root/.nvm", dest="/home/agent/.nvm")
        assert "cp -aL /root/.local/bin/. /home/agent/.local/bin/" not in cmd
        assert "cp -a /root/.nvm/. /home/agent/.nvm/" not in cmd
