"""Each sandbox backend owns its ACP transport choice.

These cases previously lived in ``tests/test_acp.py`` and drove
``connect_acp`` with a provider-name string, because transport selection was
an ``if environment == ...`` chain inside the ACP layer. That chain is now a
single ``await env.live_process(agent=...)`` call, so the same guarantees are
asserted directly against the backend that makes the decision. The regression
each case guards is unchanged and named in its docstring.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.sandbox import process as process_pkg
from benchflow.sandbox._base import BaseSandbox


def _stub_sandbox(**attrs: object) -> SimpleNamespace:
    """A stand-in for a started sandbox.

    ``live_process`` only forwards ``self`` to the transport's
    ``from_sandbox_env``, so the concrete sandbox never has to be built.
    """
    return SimpleNamespace(**attrs)


class TestDaytonaTransportSelection:
    @pytest.mark.asyncio
    async def test_direct_uses_pty_transport(self) -> None:
        """Direct Daytona tasks use PTY transport, not SSH pipes."""
        from benchflow.sandbox.daytona import DaytonaSandbox

        env = _stub_sandbox()
        with (
            patch.object(
                process_pkg.DaytonaPtyProcess,
                "from_sandbox_env",
                new_callable=AsyncMock,
            ) as pty,
            patch.object(
                process_pkg.DaytonaProcess,
                "from_sandbox_env",
                new_callable=AsyncMock,
            ) as ssh,
        ):
            await DaytonaSandbox.live_process(env, agent="test-agent")

        pty.assert_awaited_once_with(env)
        ssh.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_compose_task_uses_pty_transport(self) -> None:
        """Daytona compose (DinD) tasks avoid SSH pipe-closed failures."""
        from benchflow.sandbox.daytona import DaytonaSandbox

        strategy = MagicMock()
        strategy._compose_cmd = MagicMock(return_value="docker compose -p t")
        env = _stub_sandbox(_strategy=strategy)
        with (
            patch.object(
                process_pkg.DaytonaPtyProcess,
                "from_sandbox_env",
                new_callable=AsyncMock,
            ) as pty,
            patch.object(
                process_pkg.DaytonaProcess,
                "from_sandbox_env",
                new_callable=AsyncMock,
            ) as ssh,
        ):
            await DaytonaSandbox.live_process(env, agent="test-agent")

        pty.assert_awaited_once_with(env)
        ssh.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_can_opt_into_ssh_transport(self, monkeypatch) -> None:
        """Guards PR #921 fallback for PTY post-tool controller deadlocks."""
        from benchflow.sandbox.daytona import DaytonaSandbox

        monkeypatch.setenv("BENCHFLOW_DAYTONA_ACP_TRANSPORT", "ssh")
        env = _stub_sandbox()
        with (
            patch.object(
                process_pkg.DaytonaPtyProcess,
                "from_sandbox_env",
                new_callable=AsyncMock,
            ) as pty,
            patch.object(
                process_pkg.DaytonaProcess,
                "from_sandbox_env",
                new_callable=AsyncMock,
            ) as ssh,
        ):
            await DaytonaSandbox.live_process(env, agent="openhands")

        ssh.assert_awaited_once_with(env)
        pty.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_transport_falls_back_to_pty(self, monkeypatch) -> None:
        """Guards PR #921 against invalid transport config disabling Daytona."""
        from benchflow.sandbox.daytona import DaytonaSandbox

        monkeypatch.setenv("BENCHFLOW_DAYTONA_ACP_TRANSPORT", "invalid")
        env = _stub_sandbox()
        with (
            patch.object(
                process_pkg.DaytonaPtyProcess,
                "from_sandbox_env",
                new_callable=AsyncMock,
            ) as pty,
            patch.object(
                process_pkg.DaytonaProcess,
                "from_sandbox_env",
                new_callable=AsyncMock,
            ) as ssh,
        ):
            await DaytonaSandbox.live_process(env, agent="openhands")

        pty.assert_awaited_once_with(env)
        ssh.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gemini_uses_ssh_transport(self) -> None:
        """Guards the Gemini regression introduced by PR #896's PTY migration."""
        from benchflow.sandbox.daytona import DaytonaSandbox

        env = _stub_sandbox()
        with (
            patch.object(
                process_pkg.DaytonaPtyProcess,
                "from_sandbox_env",
                new_callable=AsyncMock,
            ) as pty,
            patch.object(
                process_pkg.DaytonaProcess,
                "from_sandbox_env",
                new_callable=AsyncMock,
            ) as ssh,
        ):
            await DaytonaSandbox.live_process(env, agent="gemini")

        ssh.assert_awaited_once_with(env)
        pty.assert_not_awaited()


class TestSingleTransportBackends:
    @pytest.mark.asyncio
    async def test_apple_container_uses_native_transport(self) -> None:
        """Guards PR #936 against treating Apple Container as Daytona."""
        from benchflow.sandbox.apple_container import AppleContainerSandbox

        env = _stub_sandbox(_container_name="bf_run")
        result = await AppleContainerSandbox.live_process(env)

        assert isinstance(result, process_pkg.AppleContainerProcess)

    @pytest.mark.asyncio
    async def test_docker_uses_compose_exec_transport(self) -> None:
        """Docker runs the agent through `docker compose exec -i`."""
        from benchflow.sandbox.docker import DockerSandbox

        env = _stub_sandbox()
        with patch.object(
            process_pkg.DockerProcess, "from_sandbox_env", return_value="docker-proc"
        ) as docker:
            result = await DockerSandbox.live_process(env)

        docker.assert_called_once_with(env)
        assert result == "docker-proc"

    @pytest.mark.asyncio
    async def test_agentcore_uses_shell_websocket_transport(self) -> None:
        """AgentCore hosts the agent on its runtime-session shell WebSocket."""
        from benchflow.sandbox.agentcore import AgentCoreSandbox

        env = _stub_sandbox(
            runtime_arn="arn:aws:bedrock-agentcore:us-west-2:1:runtime/x",
            runtime_session_id="s" * 40,
            region="us-west-2",
        )
        result = await AgentCoreSandbox.live_process(env)

        assert isinstance(result, process_pkg.AgentCoreProcess)

    @pytest.mark.asyncio
    async def test_backend_without_transport_fails_actionably(self) -> None:
        """A backend with no live transport must say so, not borrow another's.

        Modal previously fell through the ACP layer's ``else`` branch and was
        handed a ``DaytonaProcess``, which failed deep inside Daytona SSH setup
        with an unrelated error.
        """

        class _NoTransportSandbox(BaseSandbox):
            def __init__(self) -> None:  # no BaseSandbox init needed here
                pass

            def _validate_definition(self) -> None: ...

            @classmethod
            def preflight(cls) -> None: ...

            async def start(self, force_build: bool) -> None: ...
            async def stop(self, delete: bool) -> None: ...
            async def upload_file(self, source_path, target_path) -> None: ...
            async def upload_dir(
                self, source_dir, target_dir, service="main"
            ) -> None: ...
            async def download_file(self, source_path, target_path) -> None: ...
            async def download_dir(
                self, source_dir, target_dir, service="main"
            ) -> None: ...
            async def exec(self, command, **kwargs): ...

        with pytest.raises(NotImplementedError) as excinfo:
            await _NoTransportSandbox().live_process()

        assert "does not provide a live agent transport" in str(excinfo.value)
