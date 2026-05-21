"""Tests for multi-container / multi-service sandbox exec support.

Covers issue #248: vulhub-style tasks need to run commands in a specific
compose service (target/database container) rather than only the agent's
``main`` container — for flag injection before the agent runs and for
target-side verification afterwards.

All tests are unit tests with mocked subprocess/strategy layers — no
Docker or Daytona infrastructure required.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.process import DockerProcess


class TestDockerSandboxServiceExec:
    """#248: DockerSandbox.exec must be able to target any compose service."""

    def _make_sandbox(self) -> DockerSandbox:
        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox._persistent_env = {}
        sandbox.default_user = None
        return sandbox

    @pytest.mark.asyncio
    async def test_exec_defaults_to_main_service(self) -> None:
        """#248: default exec still targets the agent's ``main`` container."""
        sandbox = self._make_sandbox()
        captured: list[list[str]] = []

        async def fake_run(command, check=False, timeout_sec=None):
            captured.append(command)
            return ExecResult(stdout="", stderr="", return_code=0)

        sandbox._run_docker_compose_command = fake_run  # type: ignore[method-assign]
        await sandbox.exec("echo hi")

        assert captured[0] == ["exec", "main", "bash", "-c", "echo hi"]

    @pytest.mark.asyncio
    async def test_exec_targets_named_service(self) -> None:
        """#248: exec(service=...) runs the command in the named target container."""
        sandbox = self._make_sandbox()
        captured: list[list[str]] = []

        async def fake_run(command, check=False, timeout_sec=None):
            captured.append(command)
            return ExecResult(stdout="", stderr="", return_code=0)

        sandbox._run_docker_compose_command = fake_run  # type: ignore[method-assign]
        await sandbox.exec("test -f /tmp/pwned", service="target")

        assert captured[0] == ["exec", "target", "bash", "-c", "test -f /tmp/pwned"]

    @pytest.mark.asyncio
    async def test_exec_in_service_wrapper(self) -> None:
        """#248: exec_in_service is sugar for exec(..., service=...)."""
        sandbox = self._make_sandbox()
        captured: list[list[str]] = []

        async def fake_run(command, check=False, timeout_sec=None):
            captured.append(command)
            return ExecResult(stdout="", stderr="", return_code=0)

        sandbox._run_docker_compose_command = fake_run  # type: ignore[method-assign]
        await sandbox.exec_in_service("db", "mysql -e 'SELECT 1'")

        assert captured[0] == ["exec", "db", "bash", "-c", "mysql -e 'SELECT 1'"]

    @pytest.mark.asyncio
    async def test_exec_service_with_user_and_cwd(self) -> None:
        """#248: service selection composes with user/cwd/env flags."""
        sandbox = self._make_sandbox()
        captured: list[list[str]] = []

        async def fake_run(command, check=False, timeout_sec=None):
            captured.append(command)
            return ExecResult(stdout="", stderr="", return_code=0)

        sandbox._run_docker_compose_command = fake_run  # type: ignore[method-assign]
        await sandbox.exec(
            "id", cwd="/work", user="root", env={"K": "v"}, service="attacker"
        )

        cmd = captured[0]
        # service name precedes `bash -c`, after all flags
        assert cmd[cmd.index("bash") - 1] == "attacker"
        assert "-w" in cmd and "/work" in cmd
        assert "-u" in cmd and "root" in cmd
        assert "-e" in cmd and "K=v" in cmd

    @pytest.mark.asyncio
    async def test_services_lists_compose_services(self) -> None:
        """#248: services() enumerates every container defined in the task."""
        sandbox = self._make_sandbox()

        async def fake_run(command, check=True, timeout_sec=None):
            assert command == ["config", "--services"]
            return ExecResult(stdout="main\ntarget\ndb\n", stderr="", return_code=0)

        sandbox._run_docker_compose_command = fake_run  # type: ignore[method-assign]
        services = await sandbox.services()

        assert services == ["main", "target", "db"]


class TestDockerProcessServiceSelection:
    """#248: ACP agent process can be launched into a non-main service."""

    def test_from_sandbox_env_defaults_to_main(self) -> None:
        """#248: by default the agent process runs in the main container."""
        env = MagicMock()
        env.session_id = "task.abc"
        env.environment_dir.resolve.return_value.absolute.return_value = "/env"
        env._docker_compose_paths = []

        proc = DockerProcess.from_sandbox_env(env)
        assert proc._service == "main"

    def test_from_sandbox_env_honors_service(self) -> None:
        """#248: agent can run in a dedicated attacker container (e.g. Kali)."""
        env = MagicMock()
        env.session_id = "task.abc"
        env.environment_dir.resolve.return_value.absolute.return_value = "/env"
        env._docker_compose_paths = []

        proc = DockerProcess.from_sandbox_env(env, service="kali")
        assert proc._service == "kali"

    @pytest.mark.asyncio
    async def test_started_process_targets_selected_service(self) -> None:
        """#248: the docker compose exec command uses the selected service."""
        from unittest.mock import patch

        proc = DockerProcess(
            project_name="p",
            project_dir="/d",
            compose_files=["/d/docker-compose.yml"],
            service="kali",
        )
        captured: list[list[str]] = []

        async def fake_exec(*args, **kwargs):
            captured.append(list(args))
            mock_proc = AsyncMock()
            mock_proc.pid = 1
            mock_proc.returncode = 0
            mock_proc.stdin = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await proc.start(command="echo hi")

        # `<service> bash -c <command>` — service is the arg before "bash"
        cmd = captured[0]
        assert cmd[cmd.index("bash") - 1] == "kali"


class TestEnvironmentServicePassthrough:
    """#248: Runtime Environment exposes service selection to task authors."""

    @pytest.mark.asyncio
    async def test_environment_exec_threads_service(self) -> None:
        """#248: Environment.exec forwards a service kwarg to the sandbox."""
        from benchflow.runtime import Environment

        inner = MagicMock()
        inner.exec = AsyncMock(
            return_value=ExecResult(stdout="", stderr="", return_code=0)
        )
        env = Environment.__new__(Environment)
        env._inner = inner

        await env.exec("cat /flag", service="target")
        inner.exec.assert_awaited_once_with("cat /flag", service="target")

    @pytest.mark.asyncio
    async def test_environment_exec_in_service(self) -> None:
        """#248: Environment.exec_in_service is sugar for exec(service=...)."""
        from benchflow.runtime import Environment

        inner = MagicMock()
        inner.exec = AsyncMock(
            return_value=ExecResult(stdout="", stderr="", return_code=0)
        )
        env = Environment.__new__(Environment)
        env._inner = inner

        await env.exec_in_service("db", "mysql -e 'SELECT 1'")
        inner.exec.assert_awaited_once_with("mysql -e 'SELECT 1'", service="db")


class TestDaytonaServiceExec:
    """#248: Daytona DinD compose strategy must support service selection."""

    @pytest.mark.asyncio
    async def test_dind_exec_targets_named_service(self) -> None:
        """#248: DinD compose exec runs the command in the named service."""
        pytest.importorskip("tenacity")  # sandbox-daytona optional dependency
        from benchflow.sandbox.daytona import _DaytonaDinD

        strategy = _DaytonaDinD.__new__(_DaytonaDinD)
        captured: list[list[str]] = []

        async def fake_compose_exec(subcommand, timeout_sec=None):
            captured.append(subcommand)
            return ExecResult(stdout="", stderr="", return_code=0)

        strategy._compose_exec = fake_compose_exec  # type: ignore[method-assign]
        await strategy.exec("cat /flag", service="target")

        sub = captured[0]
        assert sub[sub.index("bash") - 1] == "target"

    @pytest.mark.asyncio
    async def test_dind_exec_defaults_to_main(self) -> None:
        """#248: DinD compose exec still defaults to the agent's main container."""
        pytest.importorskip("tenacity")  # sandbox-daytona optional dependency
        from benchflow.sandbox.daytona import _DaytonaDinD

        strategy = _DaytonaDinD.__new__(_DaytonaDinD)
        captured: list[list[str]] = []

        async def fake_compose_exec(subcommand, timeout_sec=None):
            captured.append(subcommand)
            return ExecResult(stdout="", stderr="", return_code=0)

        strategy._compose_exec = fake_compose_exec  # type: ignore[method-assign]
        await strategy.exec("echo hi")

        sub = captured[0]
        assert sub[sub.index("bash") - 1] == "main"

    @pytest.mark.asyncio
    async def test_direct_strategy_rejects_non_main_service(self) -> None:
        """#248: direct (non-compose) Daytona sandbox rejects multi-service."""
        pytest.importorskip("tenacity")  # sandbox-daytona optional dependency
        from benchflow.sandbox.daytona import _DaytonaDirect

        strategy = _DaytonaDirect.__new__(_DaytonaDirect)

        with pytest.raises(ValueError, match="single-container"):
            await strategy.exec("echo hi", service="target")


class TestModalRejectsMultiService:
    """#248: single-container backends reject non-main services with a clear error."""

    @pytest.mark.asyncio
    async def test_modal_exec_rejects_non_main_service(self) -> None:
        """#248: Modal is single-container — targeting another service must error."""
        pytest.importorskip("tenacity")  # modal_impl optional dependency
        from benchflow.sandbox.modal_impl import ModalSandbox

        sandbox = ModalSandbox.__new__(ModalSandbox)
        sandbox._persistent_env = {}
        sandbox.default_user = None

        with pytest.raises(ValueError, match="single-container"):
            await sandbox.exec("echo hi", service="target")
