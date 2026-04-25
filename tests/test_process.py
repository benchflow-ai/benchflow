"""Tests for process.py env handling (no Docker required)."""

from unittest.mock import AsyncMock, patch

import pytest

from benchflow.process import DockerProcess


class TestDockerProcessEnv:
    """Verify env vars are written inside the container, not leaked in ps aux."""

    def _make_process(self):
        return DockerProcess(
            project_name="test-project",
            project_dir="/tmp/test",
            compose_files=["/tmp/test/docker-compose.yml"],
        )

    def _mock_exec(self, captured_calls: list):
        """Return a fake create_subprocess_exec that records calls."""

        async def fake_exec(*args, **kwargs):
            captured_calls.append(list(args))
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.returncode = 0
            mock_proc.stdin = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            return mock_proc

        return fake_exec

    @pytest.mark.asyncio
    async def test_env_not_in_main_cmd(self):
        """Secrets must not appear as args in the main exec command."""
        calls = []
        with patch(
            "asyncio.create_subprocess_exec", side_effect=self._mock_exec(calls)
        ):
            proc = self._make_process()
            await proc.start(
                command="echo hello",
                env={"SECRET_KEY": "hunter2", "OTHER": "value"},
            )

        # Two calls: one to write env file, one for the main command
        assert len(calls) == 2
        main_cmd_str = " ".join(calls[1])
        assert "hunter2" not in main_cmd_str
        assert "-e SECRET_KEY" not in main_cmd_str
        assert "--env-file" not in main_cmd_str

    @pytest.mark.asyncio
    async def test_env_written_to_container(self):
        """Env vars are written to a file inside the container via stdin."""
        calls = []
        communicate_inputs = []

        async def fake_exec(*args, **kwargs):
            calls.append(list(args))
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.returncode = 0
            mock_proc.stdin = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()

            async def capture_communicate(data=None):
                if data:
                    communicate_inputs.append(data)
                return (b"", b"")

            mock_proc.communicate = capture_communicate
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            proc = self._make_process()
            await proc.start(
                command="echo hello",
                env={"API_KEY": "secret123"},
            )

        # First call writes the env file
        write_cmd_str = " ".join(calls[0])
        assert "cat >" in write_cmd_str
        assert "chmod 600" in write_cmd_str

        # Env content was piped to stdin
        assert len(communicate_inputs) == 1
        content = communicate_inputs[0].decode()
        assert "export API_KEY=" in content
        assert "secret123" in content

    @pytest.mark.asyncio
    async def test_main_cmd_sources_and_deletes_env(self):
        """Main command sources the env file and then removes it."""
        calls = []
        with patch(
            "asyncio.create_subprocess_exec", side_effect=self._mock_exec(calls)
        ):
            proc = self._make_process()
            await proc.start(
                command="codex-acp",
                env={"KEY": "val"},
            )

        main_bash_cmd = calls[1][-1]  # last arg is the bash -c string
        assert "source /tmp/.benchflow_env" in main_bash_cmd
        assert "rm -f /tmp/.benchflow_env" in main_bash_cmd
        assert "codex-acp" in main_bash_cmd

    @pytest.mark.asyncio
    async def test_no_env_single_call(self):
        """When no env is passed, only the main exec runs (no env write step)."""
        calls = []
        with patch(
            "asyncio.create_subprocess_exec", side_effect=self._mock_exec(calls)
        ):
            proc = self._make_process()
            await proc.start(command="echo hello")

        assert len(calls) == 1
        assert calls[0][-1] == "echo hello"

    @pytest.mark.asyncio
    async def test_env_values_shell_quoted(self):
        """Env values with special chars are shell-quoted."""
        communicate_inputs = []

        async def fake_exec(*args, **kwargs):
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.returncode = 0
            mock_proc.stdin = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()

            async def capture_communicate(data=None):
                if data:
                    communicate_inputs.append(data)
                return (b"", b"")

            mock_proc.communicate = capture_communicate
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            proc = self._make_process()
            await proc.start(
                command="echo hello",
                env={"KEY": "val with spaces & special; chars"},
            )

        content = communicate_inputs[0].decode()
        # Value must be quoted so bash interprets it correctly
        assert "'" in content or '"' in content

    @pytest.mark.asyncio
    async def test_dangerous_chars_in_env_values(self):
        """Env values with shell metacharacters are safely quoted."""
        communicate_inputs = []

        async def fake_exec(*args, **kwargs):
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.returncode = 0
            mock_proc.stdin = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()

            async def capture_communicate(data=None):
                if data:
                    communicate_inputs.append(data)
                return (b"", b"")

            mock_proc.communicate = capture_communicate
            return mock_proc

        dangerous_env = {
            "CMD_INJECT": "value; rm -rf /",
            "NEWLINE_VAL": "line1\nline2",
            "BACKTICK": "$(whoami)",
            "SINGLE_QUOTE": "it's a test",
        }

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            proc = self._make_process()
            await proc.start(command="echo hello", env=dangerous_env)

        content = communicate_inputs[0].decode()
        # Each value must be shell-quoted — shlex.quote wraps in single quotes
        for key in dangerous_env:
            assert f"export {key}=" in content
        # shlex.quote wraps values in single quotes
        import shlex

        assert f"export CMD_INJECT={shlex.quote('value; rm -rf /')}" in content
        assert f"export NEWLINE_VAL={shlex.quote('line1\nline2')}" in content
        assert f"export BACKTICK={shlex.quote('$(whoami)')}" in content
        assert (
            f"export SINGLE_QUOTE={shlex.quote('it' + chr(39) + 's a test')}" in content
        )


class TestDaytonaProcessEnvFilePath:
    """Regression: env-file path must be unique without relying on shell `$$` expansion.

    Guards the fix from PR #198 against the regression introduced by PR #193
    (DinD compose ACP via Daytona PTY WebSocket, commit cdccac7).

    The DinD branch builds an inner `docker compose exec --env-file PATH ...`
    command and runs it through `shlex.join()`, which single-quotes any `$$`
    (preventing remote shell expansion). The `cat > PATH` heredoc that writes
    the file uses raw f-string interpolation where `$$` IS expanded. If the
    path contains `$$`, the file is written to one path and read from another
    — env vars silently disappear.

    The direct (non-DinD) branch uses raw f-string in both write and read, so
    `$$` would expand consistently — but uuid is robust against future quoting
    changes. Pin both branches.
    """

    @pytest.mark.asyncio
    async def test_dind_env_file_path_does_not_use_shell_pid_expansion(self):
        """DinD path must not use $$ — shlex.join would quote it literally."""
        from unittest.mock import MagicMock

        from benchflow.process import DaytonaProcess

        sandbox = MagicMock()
        sandbox.create_ssh_access = AsyncMock(return_value=MagicMock(token="abc"))
        proc = DaytonaProcess(
            sandbox=sandbox,
            is_dind=True,
            compose_cmd_prefix="",
            compose_cmd_base="docker compose -p test",
        )

        captured = []

        async def fake_exec(*args, **kwargs):
            captured.append(list(args))
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.returncode = 0
            mock_proc.stdin = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await proc.start(command="echo hi", env={"FOO": "bar"})

        # Last arg of ssh is the remote command. Search it for $$
        remote_cmd = captured[0][-1]
        assert "$$" not in remote_cmd, (
            "$$ in remote command — shlex.join() will quote it, mismatching "
            f"the cat heredoc that does expand it. Got: {remote_cmd[:200]!r}"
        )
        # And: a real path was used (literal hex suffix, no shell variable)
        assert "/tmp/benchflow_env_" in remote_cmd
        assert "--env-file" in remote_cmd

    @pytest.mark.asyncio
    async def test_direct_sandbox_env_file_path_does_not_use_shell_pid_expansion(self):
        """Direct (non-DinD) path is currently safe with $$, but pin the uuid form for robustness."""
        from unittest.mock import MagicMock

        from benchflow.process import DaytonaProcess

        sandbox = MagicMock()
        sandbox.create_ssh_access = AsyncMock(return_value=MagicMock(token="abc"))
        proc = DaytonaProcess(sandbox=sandbox, is_dind=False)

        captured = []

        async def fake_exec(*args, **kwargs):
            captured.append(list(args))
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.returncode = 0
            mock_proc.stdin = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await proc.start(command="echo hi", env={"FOO": "bar"})

        remote_cmd = captured[0][-1]
        assert "$$" not in remote_cmd
        assert "/tmp/benchflow_env_" in remote_cmd
