"""Tests for process.py env handling (no Docker required)."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.process import DockerProcess


class TestDockerProcessEnv:
    """Verify env vars are injected via export prefix, not -e args."""

    def _make_process(self):
        return DockerProcess(
            project_name="test-project",
            project_dir="/tmp/test",
            compose_files=["/tmp/test/docker-compose.yml"],
        )

    @pytest.mark.asyncio
    async def test_env_not_dash_e(self):
        """Env vars must not appear as -e K=V docker compose args."""
        proc = self._make_process()
        captured_cmd = []

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(args)
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.returncode = None
            mock_proc.stdin = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await proc.start(
                command="echo hello",
                env={"SECRET_KEY": "hunter2", "OTHER": "value"},
            )

        cmd_str = " ".join(captured_cmd)
        # Must not contain the secret as a -e argument
        assert "-e SECRET_KEY=hunter2" not in cmd_str
        assert "-e OTHER=value" not in cmd_str
        # Must not use --env-file (not supported in all Docker Compose versions)
        assert "--env-file" not in cmd_str

    @pytest.mark.asyncio
    async def test_env_exported_in_command(self):
        """Env vars are prepended as export statements in the bash command."""
        proc = self._make_process()
        captured_cmd = []

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(args)
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.returncode = None
            mock_proc.stdin = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await proc.start(
                command="echo hello",
                env={"API_KEY": "secret123"},
            )

        # The bash -c argument should contain the export + original command
        cmd_str = " ".join(captured_cmd)
        assert "export API_KEY=" in cmd_str
        assert "echo hello" in cmd_str

    @pytest.mark.asyncio
    async def test_env_values_shell_quoted(self):
        """Env values with special chars are shell-quoted."""
        proc = self._make_process()
        captured_cmd = []

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(args)
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.returncode = None
            mock_proc.stdin = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await proc.start(
                command="echo hello",
                env={"KEY": "val with spaces & special; chars"},
            )

        # Find the bash -c argument (last arg)
        bash_cmd = captured_cmd[-1]
        assert "KEY=" in bash_cmd
        # Value should be quoted (not raw)
        assert "val with spaces & special; chars" not in bash_cmd or "'" in bash_cmd

    @pytest.mark.asyncio
    async def test_no_env_no_export(self):
        """When no env is passed, no export prefix is added."""
        proc = self._make_process()
        captured_cmd = []

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(args)
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.returncode = None
            mock_proc.stdin = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await proc.start(command="echo hello")

        assert "--env-file" not in captured_cmd
        # The bash -c command should be the original command
        assert captured_cmd[-1] == "echo hello"
