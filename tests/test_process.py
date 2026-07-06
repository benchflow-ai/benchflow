"""Tests for process.py env handling (no Docker required)."""

import os
import shlex
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.sandbox.process import DaytonaProcess, DaytonaPtyProcess, DockerProcess


class _FakeStdin:
    def __init__(self):
        self.writes = []
        self.drain = AsyncMock()
        self.closed = False

    def write(self, data):
        self.writes.append(data)

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(
        self,
        *,
        stdin=None,
        stdout=None,
        stderr=None,
        communicate=None,
        returncode=0,
    ):
        self.pid = 12345
        self.returncode = returncode
        self.stdin = stdin if stdin is not None else AsyncMock()
        self.stdout = stdout if stdout is not None else AsyncMock()
        self.stderr = stderr if stderr is not None else AsyncMock()
        self.communicate = communicate if communicate is not None else AsyncMock()


class _DaytonaExecHarness:
    def __init__(self):
        self.calls = []
        self.live_stdin = _FakeStdin()

    async def fake_exec(self, *args, **kwargs):
        self.calls.append(list(args))
        return _FakeProcess(stdin=self.live_stdin)


def _make_daytona_sandbox(token="abc", exit_code=0, result=None):
    sandbox = MagicMock()
    sandbox.create_ssh_access = AsyncMock(return_value=MagicMock(token=token))
    if result is None:
        result = "__BENCHFLOW_BOOTSTRAP_DONE__\n"
    sandbox.process.exec = AsyncMock(
        return_value=MagicMock(exit_code=exit_code, result=result)
    )
    return sandbox


class _FakeDaytonaPty:
    def __init__(
        self,
        on_data,
        send_error=None,
        reject_exec_env_file=True,
        emit_marker=True,
    ):
        self._on_data = on_data
        self._send_error = send_error
        self._reject_exec_env_file = reject_exec_env_file
        self._emit_marker = emit_marker
        self.inputs = []
        self.killed = False
        self.disconnected = False

    async def wait_for_connection(self):
        return None

    async def send_input(self, payload):
        self.inputs.append(payload)
        if self._reject_exec_env_file:
            parts = shlex.split(payload)
            if "exec" in parts:
                exec_index = parts.index("exec")
                assert "--env-file" not in parts[exec_index + 1 :]
        if self._send_error:
            raise self._send_error
        if self._emit_marker and "echo '" in payload:
            marker = payload.split("echo '", 1)[1].split("'", 1)[0]
            await self._on_data(f"{marker}\n".encode())

    async def kill(self):
        self.killed = True

    async def disconnect(self):
        self.disconnected = True


def _make_daytona_pty_sandbox(
    result=None,
    send_error=None,
    exit_code=0,
    emit_marker=True,
):
    sandbox = MagicMock()
    if result is None:
        result = "__BENCHFLOW_BOOTSTRAP_DONE__\n"
    ptys = []

    async def create_pty_session(*, id, on_data, envs=None):
        pty = _FakeDaytonaPty(
            on_data=on_data,
            send_error=send_error,
            emit_marker=emit_marker,
        )
        ptys.append(pty)
        return pty

    sandbox.process.exec = AsyncMock(
        return_value=MagicMock(exit_code=exit_code, result=result)
    )
    sandbox.process.create_pty_session = AsyncMock(side_effect=create_pty_session)
    return sandbox, ptys


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

    The DinD branch writes provider values through a Daytona SDK process
    bootstrap, then sources that remote env file before running compose with
    only env names in argv. A literal `$$` path risks mismatched expansion
    across these shell boundaries, so both Daytona branches use a uuid path.
    """

    @pytest.mark.asyncio
    async def test_dind_env_file_path_does_not_use_shell_pid_expansion(self):
        """DinD path must not use $$ — shlex.join would quote it literally."""
        sandbox = _make_daytona_sandbox()
        proc = DaytonaProcess(
            sandbox=sandbox,
            is_dind=True,
            compose_cmd_prefix="",
            compose_cmd_base="docker compose -p test",
        )

        harness = _DaytonaExecHarness()

        with patch("asyncio.create_subprocess_exec", side_effect=harness.fake_exec):
            await proc.start(command="echo hi", env={"FOO": "bar"})

        assert len(harness.calls) == 1
        bootstrap_cmd = sandbox.process.exec.await_args.args[0]
        live_cmd = harness.calls[0][-1]
        assert "$$" not in live_cmd, (
            "$$ in remote command — shlex.join() will quote it, mismatching "
            f"the SDK bootstrap command. Got: {live_cmd[:200]!r}"
        )

        # And: a real path was used in the SDK bootstrap (literal hex suffix,
        # no shell variable), while env values travel through Daytona's env map.
        assert "$$" not in bootstrap_cmd
        assert "/tmp/benchflow_env_" in bootstrap_cmd
        assert sandbox.process.exec.await_args.kwargs["env"] == {"FOO": "bar"}
        assert "--env FOO" in live_cmd
        assert "--env-file" not in live_cmd
        assert "bar" not in live_cmd
        assert ". /tmp/benchflow_env_" in live_cmd
        assert "rm -f /tmp/benchflow_env_" in live_cmd

    @pytest.mark.asyncio
    async def test_direct_sandbox_env_file_path_does_not_use_shell_pid_expansion(self):
        """Direct (non-DinD) path must not rely on shell PID expansion."""
        sandbox = _make_daytona_sandbox()
        proc = DaytonaProcess(sandbox=sandbox, is_dind=False)

        harness = _DaytonaExecHarness()

        with patch("asyncio.create_subprocess_exec", side_effect=harness.fake_exec):
            await proc.start(command="echo hi", env={"FOO": "bar"})

        assert len(harness.calls) == 1
        bootstrap_cmd = sandbox.process.exec.await_args.args[0]
        live_cmd = harness.calls[0][-1]
        assert "$$" not in live_cmd

        assert "$$" not in bootstrap_cmd
        assert "/tmp/benchflow_env_" in bootstrap_cmd
        assert sandbox.process.exec.await_args.kwargs["env"] == {"FOO": "bar"}
        assert "rm -f /tmp/benchflow_env_" in live_cmd

    def test_direct_bootstrap_env_file_sources_single_quote_values(self, tmp_path):
        """Guards the 2026-05-22 Daytona direct env quoting fix."""
        env_file = tmp_path / "benchflow-env"
        command = DaytonaProcess._bootstrap_env_command(
            remote_env_path=str(env_file),
            env_keys=["SAFE_QUOTE", "SPACE_VALUE"],
            shell_exports=True,
        )
        env = {
            **os.environ,
            "SAFE_QUOTE": "it's fine",
            "SPACE_VALUE": "hello world",
        }

        bootstrap = subprocess.run(
            command,
            shell=True,
            executable="/bin/sh",
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert bootstrap.returncode == 0, bootstrap.stderr
        assert "__BENCHFLOW_BOOTSTRAP_DONE__" in bootstrap.stdout.splitlines()

        source = subprocess.run(
            ". "
            + shlex.quote(str(env_file))
            + '; printf "%s\\n%s\\n" "$SAFE_QUOTE" "$SPACE_VALUE"',
            shell=True,
            executable="/bin/sh",
            text=True,
            capture_output=True,
            check=False,
        )

        assert source.returncode == 0, source.stderr
        assert source.stdout.splitlines() == ["it's fine", "hello world"]


class TestDaytonaProcessSecretArgv:
    """Regression tests for Daytona env values staying out of local argv."""

    @staticmethod
    def _joined_calls(calls):
        return "\n".join(" ".join(str(arg) for arg in call) for call in calls)

    @pytest.mark.asyncio
    async def test_direct_sandbox_env_value_not_passed_in_subprocess_argv(self):
        """Guards the 2026-05-22 Daytona argv leak blocker fix."""
        token = "ssh-token-abc"
        sandbox = _make_daytona_sandbox(token=token)
        proc = DaytonaProcess(sandbox=sandbox, is_dind=False)

        harness = _DaytonaExecHarness()
        secret = "bf_live_provider_key_should_not_be_in_argv"

        with patch("asyncio.create_subprocess_exec", side_effect=harness.fake_exec):
            await proc.start(command="codex-acp", env={"OPENAI_API_KEY": secret})

        assert harness.calls
        for call in harness.calls:
            for arg in call:
                assert secret not in arg
                assert token not in arg
        bootstrap_cmd = sandbox.process.exec.await_args.args[0]
        assert secret not in bootstrap_cmd
        assert sandbox.process.exec.await_args.kwargs["env"] == {
            "OPENAI_API_KEY": secret
        }
        ssh_config_path = harness.calls[0][2]
        assert harness.calls[0][:2] == ["ssh", "-F"]
        assert harness.calls[0][3] == "benchflow-daytona"
        assert os.path.exists(ssh_config_path)

        assert harness.live_stdin.writes == []
        await proc.writeline('{"jsonrpc":"2.0"}')
        assert harness.live_stdin.writes[-1] == b'{"jsonrpc":"2.0"}\n'
        await proc.close()
        assert not os.path.exists(ssh_config_path)

    @pytest.mark.asyncio
    async def test_dind_sandbox_env_value_not_passed_in_subprocess_argv(self):
        """Guards the 2026-05-22 Daytona compose argv leak blocker fix."""
        token = "ssh-token-abc"
        sandbox = _make_daytona_sandbox(token=token)
        proc = DaytonaProcess(
            sandbox=sandbox,
            is_dind=True,
            compose_cmd_prefix="",
            compose_cmd_base="docker compose -p test",
        )

        harness = _DaytonaExecHarness()
        secret = "bf_compose_provider_key_should_not_be_in_argv"

        with patch("asyncio.create_subprocess_exec", side_effect=harness.fake_exec):
            await proc.start(command="codex-acp", env={"ANTHROPIC_API_KEY": secret})

        assert harness.calls
        for call in harness.calls:
            for arg in call:
                assert secret not in arg
                assert token not in arg
        bootstrap_cmd = sandbox.process.exec.await_args.args[0]
        assert secret not in bootstrap_cmd
        assert sandbox.process.exec.await_args.kwargs["env"] == {
            "ANTHROPIC_API_KEY": secret
        }
        ssh_config_path = harness.calls[0][2]
        assert harness.calls[0][:2] == ["ssh", "-F"]
        assert harness.calls[0][3] == "benchflow-daytona"
        await proc.close()
        assert not os.path.exists(ssh_config_path)

    @pytest.mark.asyncio
    async def test_daytona_debug_log_does_not_include_ssh_token(self, caplog):
        """Daytona SSH access tokens must not be written to BenchFlow logs."""
        sandbox = _make_daytona_sandbox()
        proc = DaytonaProcess(sandbox=sandbox, is_dind=False)
        harness = _DaytonaExecHarness()

        with patch("asyncio.create_subprocess_exec", side_effect=harness.fake_exec):
            await proc.start(command="codex-acp", env={"OPENAI_API_KEY": "secret"})

        assert "abc@ssh.app.daytona.io" not in caplog.text
        assert "User=abc" not in caplog.text
        await proc.close()

    @pytest.mark.asyncio
    async def test_two_step_bootstrap_leaves_live_stdin_for_agent_frames(self):
        """Guards against bootstrap shell text reaching the live ACP stdin."""
        sandbox = _make_daytona_sandbox()
        proc = DaytonaProcess(sandbox=sandbox, is_dind=False)
        harness = _DaytonaExecHarness()

        with patch("asyncio.create_subprocess_exec", side_effect=harness.fake_exec):
            await proc.start(command="codex-acp", env={"OPENAI_API_KEY": "secret"})

        sandbox.process.exec.assert_awaited_once()
        assert harness.live_stdin.writes == []

        first_frame = '{"jsonrpc":"2.0"}'
        await proc.writeline(first_frame)
        assert harness.live_stdin.writes == [b'{"jsonrpc":"2.0"}\n']
        await proc.close()

    @pytest.mark.asyncio
    async def test_bootstrapped_env_file_is_removed_when_ssh_launch_fails(self):
        """Guards the 2026-05-22 Daytona launch-failure cleanup fix."""
        sandbox = _make_daytona_sandbox()
        proc = DaytonaProcess(sandbox=sandbox, is_dind=False)

        with (
            patch(
                "asyncio.create_subprocess_exec",
                side_effect=OSError("ssh launch failed"),
            ),
            pytest.raises(OSError),
        ):
            await proc.start(command="codex-acp", env={"OPENAI_API_KEY": "secret"})

        calls = sandbox.process.exec.await_args_list
        assert len(calls) == 2
        bootstrap_cmd = calls[0].args[0]
        cleanup_cmd = calls[1].args[0]
        assert "/tmp/benchflow_env_" in bootstrap_cmd
        assert cleanup_cmd.startswith("rm -f /tmp/benchflow_env_")
        assert "secret" not in cleanup_cmd
        assert proc._ssh_config_path is None

    @pytest.mark.asyncio
    async def test_ssh_config_not_created_when_env_bootstrap_fails(self):
        """Guards temp SSH config cleanup when SDK bootstrap fails early."""
        sandbox = _make_daytona_sandbox(exit_code=1, result="boom")
        proc = DaytonaProcess(sandbox=sandbox, is_dind=False)

        with (
            patch.object(DaytonaProcess, "_write_ssh_config") as write_config,
            pytest.raises(RuntimeError, match="Failed to bootstrap Daytona agent env"),
        ):
            await proc.start(command="codex-acp", env={"OPENAI_API_KEY": "secret"})

        write_config.assert_not_called()
        assert proc._ssh_config_path is None

    @pytest.mark.asyncio
    async def test_bootstrapped_env_file_is_removed_when_ssh_access_fails(self):
        """Guards remote env cleanup when SSH access fails after bootstrap."""
        sandbox = _make_daytona_sandbox()
        sandbox.create_ssh_access = AsyncMock(side_effect=RuntimeError("ssh access"))
        proc = DaytonaProcess(sandbox=sandbox, is_dind=False)

        with pytest.raises(RuntimeError, match="ssh access"):
            await proc.start(command="codex-acp", env={"OPENAI_API_KEY": "secret"})

        calls = sandbox.process.exec.await_args_list
        assert len(calls) == 2
        assert calls[0].args[0].startswith("sh -c ")
        assert calls[1].args[0].startswith("rm -f /tmp/benchflow_env_")
        assert proc._ssh_config_path is None

    @pytest.mark.asyncio
    async def test_bootstrapped_env_file_is_removed_when_ssh_config_write_fails(self):
        """Guards remote env cleanup when local SSH config creation fails."""
        sandbox = _make_daytona_sandbox()
        proc = DaytonaProcess(sandbox=sandbox, is_dind=False)

        with (
            patch.object(
                DaytonaProcess,
                "_write_ssh_config",
                side_effect=OSError("config failed"),
            ),
            pytest.raises(OSError, match="config failed"),
        ):
            await proc.start(command="codex-acp", env={"OPENAI_API_KEY": "secret"})

        calls = sandbox.process.exec.await_args_list
        assert len(calls) == 2
        assert calls[0].args[0].startswith("sh -c ")
        assert calls[1].args[0].startswith("rm -f /tmp/benchflow_env_")
        assert proc._ssh_config_path is None


class TestDaytonaPtyProcessSecretTransport:
    """Regression tests for Daytona PTY env values staying out of PTY input."""

    @pytest.mark.asyncio
    async def test_provider_env_value_not_sent_through_pty_input(self):
        """Guards the 2026-05-22 Daytona PTY env transport leak fix."""
        sandbox, ptys = _make_daytona_pty_sandbox()
        proc = DaytonaPtyProcess(
            sandbox=sandbox,
            compose_cmd_prefix="",
            compose_cmd_base="docker compose -p test",
        )
        secret = "bf_pty_provider_key_should_not_be_in_input"

        await proc.start(command="codex-acp", env={"GEMINI_API_KEY": secret})

        assert ptys
        assert len(ptys[0].inputs) == 2
        assert "echo '__BENCHFLOW_ACP_" in ptys[0].inputs[0]
        assert "stty raw -echo" in ptys[0].inputs[0]
        assert "docker compose -p test exec" not in ptys[0].inputs[0]
        assert ptys[0].inputs[1].startswith("exec sh /tmp/benchflow_pty_exec_")
        assert "docker compose -p test exec" not in ptys[0].inputs[1]
        sent_payload = "\n".join(ptys[0].inputs)
        assert secret not in sent_payload
        bootstrap_call = sandbox.process.exec.await_args_list[0]
        assert bootstrap_call.kwargs["env"] == {"GEMINI_API_KEY": secret}
        assert secret not in bootstrap_call.args[0]
        script_call = sandbox.process.exec.await_args_list[1]
        assert secret not in script_call.args[0]
        assert "docker compose -p test exec" in script_call.args[0]
        assert "--env GEMINI_API_KEY" in script_call.args[0]
        assert ". /tmp/benchflow_env_" not in sent_payload
        assert "rm -f /tmp/benchflow_env_" not in sent_payload
        assert "--env GEMINI_API_KEY" not in sent_payload
        assert "--env-file" not in sent_payload
        assert "exec ." not in sent_payload

        await proc.close()

        cleanup_calls = sandbox.process.exec.await_args_list[-2:]
        assert cleanup_calls[0].args[0].startswith("rm -f /tmp/benchflow_env_")
        assert cleanup_calls[1].args[0].startswith("rm -f /tmp/benchflow_pty_exec_")
        assert secret not in cleanup_calls[0].args[0]
        assert secret not in cleanup_calls[1].args[0]

    @pytest.mark.asyncio
    async def test_direct_daytona_pty_runs_without_compose_and_hides_env(self):
        """Direct Daytona tasks use PTY transport without docker compose."""
        sandbox, ptys = _make_daytona_pty_sandbox()
        proc = DaytonaPtyProcess(
            sandbox=sandbox,
            compose_cmd_prefix="",
            compose_cmd_base="",
        )
        secret = "bf_direct_pty_provider_key_should_not_be_in_input"

        await proc.start(
            command="openhands acp --always-approve",
            env={"OPENAI_API_KEY": secret},
            cwd="/workspace/agent_workspace",
        )

        assert ptys
        assert len(ptys[0].inputs) == 2
        sent_payload = "\n".join(ptys[0].inputs)
        assert "docker compose" not in sent_payload
        assert "cd /workspace/agent_workspace" not in sent_payload
        assert ". /tmp/benchflow_env_" not in sent_payload
        assert "rm -f /tmp/benchflow_env_" not in sent_payload
        assert "exec bash -lc 'openhands acp --always-approve'" not in sent_payload
        assert ptys[0].inputs[1].startswith("exec sh /tmp/benchflow_pty_exec_")
        assert "--env OPENAI_API_KEY" not in sent_payload
        assert secret not in sent_payload
        bootstrap_call = sandbox.process.exec.await_args_list[0]
        assert bootstrap_call.kwargs["env"] == {"OPENAI_API_KEY": secret}
        assert secret not in bootstrap_call.args[0]

        await proc.close()

    @pytest.mark.asyncio
    async def test_bootstrapped_env_file_is_removed_when_pty_handoff_fails(self):
        """Guards remote env cleanup when Daytona PTY start fails after bootstrap."""
        sandbox, ptys = _make_daytona_pty_sandbox(
            send_error=RuntimeError("pty handoff failed")
        )
        proc = DaytonaPtyProcess(
            sandbox=sandbox,
            compose_cmd_prefix="",
            compose_cmd_base="docker compose -p test",
        )

        with pytest.raises(RuntimeError, match="pty handoff failed"):
            await proc.start(command="codex-acp", env={"GEMINI_API_KEY": "secret"})

        assert ptys
        assert ptys[0].killed
        assert ptys[0].disconnected
        calls = sandbox.process.exec.await_args_list
        assert len(calls) == 4
        assert calls[0].args[0].startswith("sh -c ")
        assert calls[0].kwargs["env"] == {"GEMINI_API_KEY": "secret"}
        assert "docker compose -p test exec" in calls[1].args[0]
        assert calls[2].args[0].startswith("rm -f /tmp/benchflow_env_")
        assert calls[3].args[0].startswith("rm -f /tmp/benchflow_pty_exec_")
        assert "secret" not in calls[2].args[0]
        assert "secret" not in calls[3].args[0]

    @pytest.mark.asyncio
    async def test_pty_marker_timeout_raises_typed_transport_error(self, monkeypatch):
        """Guards PR #561: PTY startup marker timeouts stay retryable."""
        from benchflow.diagnostics import TransportClosedError

        monkeypatch.setattr(DaytonaPtyProcess, "_START_MARKER_TIMEOUT_SEC", 0.01)
        sandbox, ptys = _make_daytona_pty_sandbox(emit_marker=False)
        proc = DaytonaPtyProcess(
            sandbox=sandbox,
            compose_cmd_prefix="",
            compose_cmd_base="docker compose -p test",
        )

        with pytest.raises(TransportClosedError) as exc_info:
            await proc.start(command="codex-acp", env={"GEMINI_API_KEY": "secret"})

        assert ptys
        assert ptys[0].killed
        assert ptys[0].disconnected
        assert exc_info.value.diagnostic.transport_diagnosis == "pty_startup_timeout"
        calls = sandbox.process.exec.await_args_list
        assert len(calls) == 4
        assert calls[0].args[0].startswith("sh -c ")
        assert calls[0].kwargs["env"] == {"GEMINI_API_KEY": "secret"}
        assert "docker compose -p test exec" in calls[1].args[0]
        assert calls[2].args[0].startswith("rm -f /tmp/benchflow_env_")
        assert calls[3].args[0].startswith("rm -f /tmp/benchflow_pty_exec_")
        assert "secret" not in calls[2].args[0]
        assert "secret" not in calls[3].args[0]

    @pytest.mark.asyncio
    async def test_pty_readline_timeout_uses_default_when_env_is_missing(
        self, monkeypatch
    ):
        """Guards the fix for the Daytona PTY timeout issue seen at cb65fe8."""
        from benchflow.diagnostics import TransportClosedError

        monkeypatch.delenv("BENCHFLOW_DAYTONA_PTY_READLINE_TIMEOUT", raising=False)
        sandbox, _ptys = _make_daytona_pty_sandbox()
        proc = DaytonaPtyProcess(
            sandbox=sandbox,
            compose_cmd_prefix="",
            compose_cmd_base="docker compose -p test",
        )

        async def fake_wait_for(awaitable, *, timeout):
            awaitable.close()
            raise TimeoutError

        with (
            patch("asyncio.wait_for", side_effect=fake_wait_for) as wait_for,
            pytest.raises(TransportClosedError) as exc_info,
        ):
            await proc.readline()

        assert wait_for.call_args.kwargs["timeout"] == 900.0
        assert exc_info.value.diagnostic.raw_message == "PTY readline timeout (900s)"
        assert exc_info.value.diagnostic.transport_diagnosis == "pty_error"

    @pytest.mark.asyncio
    async def test_pty_readline_timeout_can_be_extended_for_long_agent_thinking(
        self, monkeypatch
    ):
        """Guards the fix for the Daytona PTY timeout issue seen at cb65fe8."""
        from benchflow.diagnostics import TransportClosedError

        monkeypatch.setenv("BENCHFLOW_DAYTONA_PTY_READLINE_TIMEOUT", "0.01")
        sandbox, _ptys = _make_daytona_pty_sandbox()
        proc = DaytonaPtyProcess(
            sandbox=sandbox,
            compose_cmd_prefix="",
            compose_cmd_base="docker compose -p test",
        )

        with pytest.raises(TransportClosedError) as exc_info:
            await proc.readline()

        assert exc_info.value.diagnostic.raw_message == "PTY readline timeout (0.01s)"
        assert exc_info.value.diagnostic.transport_diagnosis == "pty_error"

    @pytest.mark.asyncio
    async def test_bootstrapped_env_file_is_removed_when_pty_bootstrap_marker_missing(
        self,
    ):
        """Guards remote env cleanup when Daytona PTY SDK bootstrap is ambiguous."""
        sandbox, ptys = _make_daytona_pty_sandbox(result="no marker\n")
        proc = DaytonaPtyProcess(
            sandbox=sandbox,
            compose_cmd_prefix="",
            compose_cmd_base="docker compose -p test",
        )

        with pytest.raises(RuntimeError, match="Failed to bootstrap Daytona PTY"):
            await proc.start(command="codex-acp", env={"GEMINI_API_KEY": "secret"})

        assert ptys
        assert ptys[0].killed
        assert ptys[0].disconnected
        calls = sandbox.process.exec.await_args_list
        assert len(calls) == 2
        assert calls[0].args[0].startswith("sh -c ")
        assert calls[0].kwargs["env"] == {"GEMINI_API_KEY": "secret"}
        assert calls[1].args[0].startswith("rm -f /tmp/benchflow_env_")
        assert "secret" not in calls[1].args[0]

    @pytest.mark.asyncio
    async def test_bootstrapped_env_file_is_removed_when_pty_bootstrap_fails(self):
        """Guards remote env cleanup when Daytona PTY env bootstrap fails."""
        sandbox, ptys = _make_daytona_pty_sandbox(result="boom", exit_code=1)
        proc = DaytonaPtyProcess(
            sandbox=sandbox,
            compose_cmd_prefix="",
            compose_cmd_base="docker compose -p test",
        )

        with pytest.raises(RuntimeError, match="Failed to bootstrap Daytona PTY"):
            await proc.start(command="codex-acp", env={"GEMINI_API_KEY": "secret"})

        assert ptys
        assert ptys[0].killed
        assert ptys[0].disconnected
        calls = sandbox.process.exec.await_args_list
        assert len(calls) == 2
        assert calls[0].args[0].startswith("sh -c ")
        assert calls[0].kwargs["env"] == {"GEMINI_API_KEY": "secret"}
        assert calls[1].args[0].startswith("rm -f /tmp/benchflow_env_")
        assert "secret" not in calls[1].args[0]
