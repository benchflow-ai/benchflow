"""Tests for process.py env-file handling (no Docker required)."""

import asyncio
import os
import stat
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.process import DockerProcess


class TestDockerProcessEnvFile:
    """Verify env vars are written to a temp file, not passed as -e args."""

    def _make_process(self):
        return DockerProcess(
            project_name="test-project",
            project_dir="/tmp/test",
            compose_files=["/tmp/test/docker-compose.yml"],
        )

    @pytest.mark.asyncio
    async def test_env_file_not_dash_e(self):
        """Env vars must go through --env-file, not -e K=V."""
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
        # Must use --env-file
        assert "--env-file" in cmd_str

    @pytest.mark.asyncio
    async def test_env_file_permissions(self):
        """Env file must be created with 0600 permissions."""
        proc = self._make_process()
        observed_path = None
        observed_mode = None
        observed_content = None

        original_exec = asyncio.create_subprocess_exec

        async def spy_exec(*args, **kwargs):
            nonlocal observed_path, observed_mode, observed_content
            # Find the --env-file arg
            args_list = list(args)
            for i, arg in enumerate(args_list):
                if arg == "--env-file" and i + 1 < len(args_list):
                    observed_path = args_list[i + 1]
                    observed_mode = stat.S_IMODE(os.stat(observed_path).st_mode)
                    with open(observed_path) as f:
                        observed_content = f.read()
                    break

            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.returncode = None
            mock_proc.stdin = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=spy_exec):
            await proc.start(
                command="echo hello",
                env={"API_KEY": "secret123"},
            )

        assert observed_path is not None
        assert observed_mode == 0o600
        assert "API_KEY=secret123\n" in observed_content

    @pytest.mark.asyncio
    async def test_env_file_cleaned_up(self):
        """Env file must be deleted after subprocess starts."""
        proc = self._make_process()
        env_file_path = None

        async def fake_exec(*args, **kwargs):
            nonlocal env_file_path
            args_list = list(args)
            for i, arg in enumerate(args_list):
                if arg == "--env-file" and i + 1 < len(args_list):
                    env_file_path = args_list[i + 1]
                    break

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
                env={"KEY": "val"},
            )

        assert env_file_path is not None
        assert not os.path.exists(env_file_path), "Env file should be deleted after start"

    @pytest.mark.asyncio
    async def test_no_env_no_file(self):
        """When no env is passed, no --env-file arg should appear."""
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
