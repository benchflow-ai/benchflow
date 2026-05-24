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

        assert captured[0] == ["exec", "-T", "main", "sh", "-c", "echo hi"]

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

        assert captured[0] == [
            "exec",
            "-T",
            "target",
            "sh",
            "-c",
            "test -f /tmp/pwned",
        ]

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

        assert captured[0] == ["exec", "-T", "db", "sh", "-c", "mysql -e 'SELECT 1'"]

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
        # service name precedes `sh -c`, after all flags
        assert cmd[cmd.index("sh") - 1] == "attacker"
        assert "-w" in cmd and "/work" in cmd
        assert "-u" in cmd and "root" in cmd
        # Env vars are sourced from a file inside the container, never passed
        # as `-e KEY=VALUE` flags (which leak onto host `ps aux`).
        assert "-e" not in cmd
        sh_cmd = cmd[cmd.index("sh") + 2]
        assert "base64 -d" in sh_cmd
        for arg in cmd:
            assert "K=v" not in arg

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

    @pytest.mark.asyncio
    async def test_services_filters_merged_stderr_warnings(self) -> None:
        """#248: warning lines merged into stdout are not mistaken for services.

        ``_run_docker_compose_command`` redirects stderr into stdout, so a
        compose warning could otherwise become a spurious service name.
        """
        sandbox = self._make_sandbox()

        async def fake_run(command, check=True, timeout_sec=None):
            assert command == ["config", "--services"]
            return ExecResult(
                stdout=(
                    "WARN[0000] /env/docker-compose.yaml: `version` is obsolete\n"
                    'time="2024-01-01" level=warning msg="foo"\n'
                    "main\n"
                    "target\n"
                ),
                stderr="",
                return_code=0,
            )

        sandbox._run_docker_compose_command = fake_run  # type: ignore[method-assign]
        services = await sandbox.services()

        assert services == ["main", "target"]


class TestDockerSandboxServiceFileTransfer:
    """#248: upload_dir/download_dir must be able to target any service.

    Target-side ``test.sh`` verification needs the ``/tests`` dir copied into
    the target container and the resulting ``reward.txt`` copied back out.
    """

    def _make_sandbox(self) -> DockerSandbox:
        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox._persistent_env = {}
        sandbox.default_user = None
        return sandbox

    @pytest.mark.asyncio
    async def test_upload_dir_targets_named_service(self) -> None:
        """#248: upload_dir(service=...) copies into the target container."""
        sandbox = self._make_sandbox()
        cp_calls: list[list[str]] = []

        async def fake_run(command, check=False, timeout_sec=None):
            cp_calls.append(command)
            return ExecResult(stdout="", stderr="", return_code=0)

        sandbox._run_docker_compose_command = fake_run  # type: ignore[method-assign]
        await sandbox.upload_dir("/host/tests", "/tests", service="target")

        cp = next(c for c in cp_calls if c and c[0] == "cp")
        assert cp == ["cp", "/host/tests/.", "target:/tests"]

    @pytest.mark.asyncio
    async def test_upload_dir_defaults_to_main(self) -> None:
        """#248: upload_dir without a service still copies into main."""
        sandbox = self._make_sandbox()
        cp_calls: list[list[str]] = []

        async def fake_run(command, check=False, timeout_sec=None):
            cp_calls.append(command)
            return ExecResult(stdout="", stderr="", return_code=0)

        sandbox._run_docker_compose_command = fake_run  # type: ignore[method-assign]
        await sandbox.upload_dir("/host/tests", "/tests")

        cp = next(c for c in cp_calls if c and c[0] == "cp")
        assert cp == ["cp", "/host/tests/.", "main:/tests"]

    @pytest.mark.asyncio
    async def test_download_dir_targets_named_service(self) -> None:
        """#248: download_dir(service=...) copies out of the target container."""
        sandbox = self._make_sandbox()
        cp_calls: list[list[str]] = []

        async def fake_run(command, check=False, timeout_sec=None):
            cp_calls.append(command)
            return ExecResult(stdout="", stderr="", return_code=0)

        sandbox._run_docker_compose_command = fake_run  # type: ignore[method-assign]
        await sandbox.download_dir("/logs/verifier", "/host/out", service="target")

        cp = next(c for c in cp_calls if c and c[0] == "cp")
        assert cp == ["cp", "target:/logs/verifier/.", "/host/out"]


class TestSingleContainerFileTransferRejection:
    """#248: single-container backends reject non-main dir transfers."""

    @pytest.mark.asyncio
    async def test_modal_upload_dir_rejects_non_main(self) -> None:
        """#248: Modal upload_dir cannot target a non-main service."""
        pytest.importorskip("modal")  # sandbox-modal optional dependency
        from benchflow.sandbox.modal_impl import ModalSandbox

        sandbox = ModalSandbox.__new__(ModalSandbox)
        with pytest.raises(ValueError, match="single-container"):
            await sandbox.upload_dir("/host/tests", "/tests", service="target")

    @pytest.mark.asyncio
    async def test_modal_download_dir_rejects_non_main(self) -> None:
        """#248: Modal download_dir cannot target a non-main service."""
        pytest.importorskip("modal")  # sandbox-modal optional dependency
        from benchflow.sandbox.modal_impl import ModalSandbox

        sandbox = ModalSandbox.__new__(ModalSandbox)
        with pytest.raises(ValueError, match="single-container"):
            await sandbox.download_dir("/logs/verifier", "/host/out", service="target")

    @pytest.mark.asyncio
    async def test_daytona_direct_upload_dir_rejects_non_main(self) -> None:
        """#248: direct (non-compose) Daytona upload_dir rejects multi-service."""
        pytest.importorskip("daytona")  # sandbox-daytona optional dependency
        from benchflow.sandbox.daytona import _DaytonaDirect

        strategy = _DaytonaDirect.__new__(_DaytonaDirect)
        with pytest.raises(ValueError, match="single-container"):
            await strategy.upload_dir("/host/tests", "/tests", service="target")


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
        pytest.importorskip("daytona")  # sandbox-daytona optional dependency
        from benchflow.sandbox.daytona import _DaytonaDinD

        strategy = _DaytonaDinD.__new__(_DaytonaDinD)
        captured: list[list[str]] = []

        async def fake_compose_exec(subcommand, timeout_sec=None):
            captured.append(subcommand)
            return ExecResult(stdout="", stderr="", return_code=0)

        strategy._compose_exec = fake_compose_exec  # type: ignore[method-assign]
        await strategy.exec("cat /flag", service="target")

        sub = captured[0]
        assert sub[sub.index("sh") - 1] == "target"

    @pytest.mark.asyncio
    async def test_dind_exec_defaults_to_main(self) -> None:
        """#248: DinD compose exec still defaults to the agent's main container."""
        pytest.importorskip("daytona")  # sandbox-daytona optional dependency
        from benchflow.sandbox.daytona import _DaytonaDinD

        strategy = _DaytonaDinD.__new__(_DaytonaDinD)
        captured: list[list[str]] = []

        async def fake_compose_exec(subcommand, timeout_sec=None):
            captured.append(subcommand)
            return ExecResult(stdout="", stderr="", return_code=0)

        strategy._compose_exec = fake_compose_exec  # type: ignore[method-assign]
        await strategy.exec("echo hi")

        sub = captured[0]
        assert sub[sub.index("sh") - 1] == "main"

    @pytest.mark.asyncio
    async def test_direct_strategy_rejects_non_main_service(self) -> None:
        """#248: direct (non-compose) Daytona sandbox rejects multi-service."""
        pytest.importorskip("daytona")  # sandbox-daytona optional dependency
        from benchflow.sandbox.daytona import _DaytonaDirect

        strategy = _DaytonaDirect.__new__(_DaytonaDirect)

        with pytest.raises(ValueError, match="single-container"):
            await strategy.exec("echo hi", service="target")

    @pytest.mark.asyncio
    async def test_dind_services_lists_compose_services(self) -> None:
        """#248: DinD services() enumerates every compose service in the task."""
        pytest.importorskip("daytona")  # sandbox-daytona optional dependency
        from benchflow.sandbox.daytona import _DaytonaDinD

        strategy = _DaytonaDinD.__new__(_DaytonaDinD)
        captured: list[list[str]] = []

        async def fake_compose_exec(subcommand, timeout_sec=None):
            captured.append(subcommand)
            return ExecResult(stdout="main\ntarget\ndb\n", stderr="", return_code=0)

        strategy._compose_exec = fake_compose_exec  # type: ignore[method-assign]
        services = await strategy.services()

        assert captured[0] == ["config", "--services"]
        assert services == ["main", "target", "db"]

    @pytest.mark.asyncio
    async def test_dind_services_filters_warning_lines(self) -> None:
        """#248: warning lines merged into stdout are not mistaken for services."""
        pytest.importorskip("daytona")  # sandbox-daytona optional dependency
        from benchflow.sandbox.daytona import _DaytonaDinD

        strategy = _DaytonaDinD.__new__(_DaytonaDinD)

        async def fake_compose_exec(subcommand, timeout_sec=None):
            return ExecResult(
                stdout=(
                    "WARN[0000] /benchflow/environment/docker-compose.yaml: "
                    "`version` is obsolete\n"
                    "main\n"
                    "target\n"
                ),
                stderr="",
                return_code=0,
            )

        strategy._compose_exec = fake_compose_exec  # type: ignore[method-assign]
        services = await strategy.services()

        assert services == ["main", "target"]

    @pytest.mark.asyncio
    async def test_direct_strategy_services_rejects_single_container(self) -> None:
        """#248: direct (non-compose) Daytona sandbox has no compose topology."""
        pytest.importorskip("daytona")  # sandbox-daytona optional dependency
        from benchflow.sandbox.daytona import _DaytonaDirect

        strategy = _DaytonaDirect.__new__(_DaytonaDirect)

        with pytest.raises(NotImplementedError, match="single-container"):
            await strategy.services()


class TestModalRejectsMultiService:
    """#248: single-container backends reject non-main services with a clear error."""

    @pytest.mark.asyncio
    async def test_modal_exec_rejects_non_main_service(self) -> None:
        """#248: Modal is single-container — targeting another service must error."""
        pytest.importorskip("modal")  # sandbox-modal optional dependency
        from benchflow.sandbox.modal_impl import ModalSandbox

        sandbox = ModalSandbox.__new__(ModalSandbox)
        sandbox._persistent_env = {}
        sandbox.default_user = None

        with pytest.raises(ValueError, match="single-container"):
            await sandbox.exec("echo hi", service="target")

    @pytest.mark.asyncio
    async def test_modal_services_rejects_single_container(self) -> None:
        """#248: Modal has no compose topology — services() must error clearly."""
        pytest.importorskip("modal")  # sandbox-modal optional dependency
        from benchflow.sandbox.modal_impl import ModalSandbox

        sandbox = ModalSandbox.__new__(ModalSandbox)

        with pytest.raises(NotImplementedError, match="single-container"):
            await sandbox.services()

    @pytest.mark.asyncio
    async def test_modal_is_dir_rejects_non_main_service(self) -> None:
        """Guards PR #310: ModalSandbox.is_dir gained a ``service`` param.

        PR #310 added ``service`` to ``BaseSandbox.is_dir``/``is_file`` but
        left ``ModalSandbox``'s overrides on the old ``(path, user)``
        signature, so ``is_dir(path, service="target")`` raised an opaque
        ``TypeError`` instead of the actionable ``ValueError`` that
        ``ModalSandbox.exec`` raises for non-main services.
        """
        pytest.importorskip("modal")  # sandbox-modal optional dependency
        from benchflow.sandbox.modal_impl import ModalSandbox

        sandbox = ModalSandbox.__new__(ModalSandbox)

        with pytest.raises(ValueError, match="single-container"):
            await sandbox.is_dir("/etc", service="target")

    @pytest.mark.asyncio
    async def test_modal_is_file_rejects_non_main_service(self) -> None:
        """Guards PR #310: ModalSandbox.is_file gained a ``service`` param.

        Same regression as ``is_dir`` — without the ``service`` parameter,
        ``is_file(path, service="target")`` raised ``TypeError`` rather than
        the clear ``ValueError`` Modal uses for unsupported multi-service
        access.
        """
        pytest.importorskip("modal")  # sandbox-modal optional dependency
        from benchflow.sandbox.modal_impl import ModalSandbox

        sandbox = ModalSandbox.__new__(ModalSandbox)

        with pytest.raises(ValueError, match="single-container"):
            await sandbox.is_file("/etc/hosts", service="target")


class TestServiceExecUsesPosixShell:
    """Guards PR #310: service-targeted exec must use POSIX ``sh``, not ``bash``.

    PR #310 made ``exec(..., service=...)`` route commands to arbitrary task
    containers, but the Docker and Daytona-DinD exec paths hardcoded
    ``bash -c`` / ``bash -lc``. Alpine/distroless/minimal DB images ship no
    ``/bin/bash``, so service-targeted verifier/setup commands failed even on
    healthy containers. The exec wrapper uses only POSIX constructs, so the
    fix is to invoke ``sh -c``.
    """

    def _make_docker_sandbox(self) -> DockerSandbox:
        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox._persistent_env = {}
        sandbox.default_user = None
        return sandbox

    @pytest.mark.asyncio
    async def test_docker_exec_invokes_sh_not_bash(self) -> None:
        """Guards PR #310: DockerSandbox.exec must invoke ``sh``, never ``bash``."""
        sandbox = self._make_docker_sandbox()
        captured: list[list[str]] = []

        async def fake_run(command, check=False, timeout_sec=None):
            captured.append(command)
            return ExecResult(stdout="", stderr="", return_code=0)

        sandbox._run_docker_compose_command = fake_run  # type: ignore[method-assign]
        await sandbox.exec("echo hi", service="target")

        cmd = captured[0]
        assert "sh" in cmd
        assert "bash" not in cmd
        # ``sh -c <command>`` is the tail of the exec invocation.
        assert cmd[-3:] == ["sh", "-c", "echo hi"]

    @pytest.mark.asyncio
    async def test_docker_exec_env_wrapper_stays_posix(self) -> None:
        """Guards PR #310: the env-file wrapper sourced under ``sh`` is POSIX.

        The wrapper relies on ``trap``, ``printf``, ``base64 -d``, ``umask``,
        ``set -a`` and ``. file`` — all POSIX, so it runs correctly under
        ``sh`` on shells without ``bash``.
        """
        sandbox = self._make_docker_sandbox()
        captured: list[list[str]] = []

        async def fake_run(command, check=False, timeout_sec=None):
            captured.append(command)
            return ExecResult(stdout="", stderr="", return_code=0)

        sandbox._run_docker_compose_command = fake_run  # type: ignore[method-assign]
        await sandbox.exec("id", env={"K": "v"}, service="target")

        cmd = captured[0]
        assert cmd[cmd.index("sh") - 1] == "target"
        wrapped = cmd[cmd.index("sh") + 2]
        # ``source`` is a bashism; the wrapper must use the POSIX ``.`` builtin.
        assert "source " not in wrapped
        assert ". /tmp/.benchflow_exec_env_" in wrapped

    @pytest.mark.asyncio
    async def test_daytona_dind_exec_invokes_sh_not_bash(self) -> None:
        """Guards PR #310: Daytona DinD exec must invoke ``sh``, never ``bash``."""
        pytest.importorskip("daytona")  # sandbox-daytona optional dependency
        from benchflow.sandbox.daytona import _DaytonaDinD

        strategy = _DaytonaDinD.__new__(_DaytonaDinD)
        captured: list[list[str]] = []

        async def fake_compose_exec(subcommand, timeout_sec=None):
            captured.append(subcommand)
            return ExecResult(stdout="", stderr="", return_code=0)

        strategy._compose_exec = fake_compose_exec  # type: ignore[method-assign]
        await strategy.exec("cat /flag", service="target")

        sub = captured[0]
        assert "sh" in sub
        assert "bash" not in sub
        assert sub[-3:] == ["sh", "-c", "cat /flag"]
