"""Each sandbox backend owns its ACP transport choice.

These cases previously lived in ``tests/test_acp.py`` and drove
``connect_acp`` with a provider-name string, because transport selection was
an ``if environment == ...`` chain inside the ACP layer. That chain is now a
single ``await env.live_process(agent=...)`` call, so the same guarantees are
asserted directly against the backend that makes the decision. The regression
each case guards is unchanged and named in its docstring.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.container_transport import ContainerTransport
from benchflow.rollout import Rollout
from benchflow.sandbox import process as process_pkg
from benchflow.sandbox._base import BaseSandbox
from benchflow.sandbox.process import _base as process_base
from benchflow.sandbox.process._base import (
    LiveProcess,
    SubprocessLiveProcess,
    _RemoteProcessGroupLiveProcess,
)


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


def _mutating_adapter_program(path: str) -> str:
    child = (
        "import signal, time\n"
        "from pathlib import Path\n"
        f"path = Path({path!r})\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "while True:\n"
        "    with path.open('a') as stream:\n"
        "        stream.write('x')\n"
        "        stream.flush()\n"
        "    time.sleep(0.01)\n"
    )
    return (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL, close_fds=True)\n"
        "print('ready', flush=True)\n"
        "time.sleep(3600)\n"
    )


async def _wait_for_mutation(path, *, previous_size: int = -1) -> int:
    for _ in range(500):
        if path.exists():
            size = path.stat().st_size
            if size > previous_size:
                return size
        await asyncio.sleep(0.01)
    raise AssertionError(f"child did not mutate {path}")


class _LocalProcessGroup(SubprocessLiveProcess):
    async def start(self, command, env=None, cwd=None) -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._set_process(process, owns_process_group=True)


class _LocalRemoteProcessGroup(_RemoteProcessGroupLiveProcess):
    async def start(self, command, env=None, cwd=None) -> None:
        wrapped = self._wrap_remote_process_group(
            shlex.join([sys.executable, "-c", command])
        )
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            wrapped,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._set_process(process)

    async def _exec_remote_process_group_command(self, command: str) -> int:
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return await process.wait()


class _AdapterOnlyProcess(LiveProcess):
    def __init__(self) -> None:
        self.process = None

    async def start(self, command, env=None, cwd=None) -> None:
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

    async def readline(self) -> bytes:
        assert self.process is not None and self.process.stdout is not None
        return await self.process.stdout.readline()

    async def writeline(self, data: str) -> None:
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write((data + "\n").encode())
        await self.process.stdin.drain()

    async def close(self) -> None:
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            await self.process.wait()

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None


@pytest.mark.asyncio
async def test_owned_process_group_stops_adapter_and_mutating_child(
    tmp_path, monkeypatch
) -> None:
    mutation_path = tmp_path / "mutations"
    monkeypatch.setattr(
        process_base, "_PROCESS_TREE_TERM_TIMEOUT_SEC", 0.1, raising=False
    )
    monkeypatch.setattr(
        process_base, "_PROCESS_TREE_KILL_TIMEOUT_SEC", 1.0, raising=False
    )
    process = _LocalProcessGroup()
    transport = ContainerTransport(
        container_process=process,
        command=_mutating_adapter_program(str(mutation_path)),
    )
    await transport.start()
    assert await asyncio.wait_for(process.readline(), timeout=5) == b"ready\n"
    await _wait_for_mutation(mutation_path)

    await transport.close()

    termination = transport.termination_status
    assert termination.graceful_termination is False
    assert termination.force_kill_required is True
    assert termination.process_tree_stopped is True
    stopped_size = mutation_path.stat().st_size
    await asyncio.sleep(0.1)
    assert mutation_path.stat().st_size == stopped_size


@pytest.mark.asyncio
async def test_remote_process_group_identity_stops_mutating_descendants(
    tmp_path, monkeypatch
) -> None:
    mutation_path = tmp_path / "remote-mutations"
    monkeypatch.setattr(
        process_base, "_PROCESS_TREE_TERM_TIMEOUT_SEC", 0.1, raising=False
    )
    monkeypatch.setattr(
        process_base, "_PROCESS_TREE_KILL_TIMEOUT_SEC", 1.0, raising=False
    )
    process = _LocalRemoteProcessGroup()
    transport = ContainerTransport(
        container_process=process,
        command=_mutating_adapter_program(str(mutation_path)),
    )
    await transport.start()
    assert await asyncio.wait_for(process.readline(), timeout=5) == b"ready\n"
    await _wait_for_mutation(mutation_path)

    await transport.close()

    assert transport.termination_status.force_kill_required is True
    assert transport.termination_status.process_tree_stopped is True
    stopped_size = mutation_path.stat().st_size
    await asyncio.sleep(0.1)
    assert mutation_path.stat().st_size == stopped_size


@pytest.mark.asyncio
async def test_stopping_only_adapter_cannot_claim_capture_safe(tmp_path) -> None:
    mutation_path = tmp_path / "mutations"
    process = _AdapterOnlyProcess()
    transport = ContainerTransport(
        container_process=process,
        command=_mutating_adapter_program(str(mutation_path)),
    )
    await transport.start()
    assert await asyncio.wait_for(process.readline(), timeout=5) == b"ready\n"
    first_size = await _wait_for_mutation(mutation_path)
    assert process.process is not None
    process_group_id = process.process.pid

    try:
        await transport.close()
        await _wait_for_mutation(mutation_path, previous_size=first_size)
        assert transport.termination_status.process_tree_stopped is False
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGKILL)


class _UnknownInitialLivenessProcess(_RemoteProcessGroupLiveProcess):
    def __init__(self) -> None:
        super().__init__()
        self._remote_process_group_path = "/tmp/exact-agent-group"
        self.events: list[str] = []
        self._liveness = iter([None, True, False])

    async def start(self, command, env=None, cwd=None) -> None:
        raise AssertionError("this fake is never launched")

    async def close_stdin(self) -> bool:
        self.events.append("session_closed")
        return True

    async def _remote_process_group_alive(self) -> bool | None:
        self.events.append("liveness")
        return next(self._liveness)

    async def _signal_remote_process_group(self, signal_name: str) -> bool:
        self.events.append(signal_name)
        return True

    async def _cleanup_remote_process_group_identity(self) -> None:
        self.events.append("cleanup")


@pytest.mark.asyncio
async def test_remote_group_still_signals_when_initial_liveness_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_base, "_PROCESS_TREE_TERM_TIMEOUT_SEC", 0.0, raising=False
    )
    process = _UnknownInitialLivenessProcess()

    result = await process.terminate_process_tree()

    assert process.events == [
        "session_closed",
        "liveness",
        "TERM",
        "KILL",
        "liveness",
        "liveness",
        "cleanup",
    ]
    assert result.graceful_termination is False
    assert result.force_kill_required is True
    assert result.process_tree_stopped is True


_MUTATING_ACP_AGENT = Path(__file__).parent / "fixtures" / "mock_acp_mutating_agent.py"


class _FixtureProcessGroup(SubprocessLiveProcess):
    def __init__(self) -> None:
        self.events: list[str] = []
        self.protocol_event_path: Path | None = None
        self.signal_protocol_events: list[list[str]] = []

    async def start(self, command, env=None, cwd=None) -> None:
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            cwd=cwd,
            env={**os.environ, **(env or {})},
        )
        self._set_process(process, owns_process_group=True)

    async def close_stdin(self) -> bool:
        if not getattr(self, "_stdin_closed", False):
            self.events.append("session_close")
        return await super().close_stdin()

    async def process_tree_stopped(self) -> bool:
        self.events.append("liveness")
        return await super().process_tree_stopped()

    async def _wait_for_process_tree(self, timeout: float) -> bool:
        self.events.append("wait")
        return await super()._wait_for_process_tree(timeout)

    def _signal_process_group(self, sig: signal.Signals) -> bool:
        self.events.append(sig.name)
        if self.protocol_event_path is not None:
            self.signal_protocol_events.append(
                self.protocol_event_path.read_text().splitlines()
            )
        return super()._signal_process_group(sig)


class _FixtureAdapterOnlyProcess(LiveProcess):
    def __init__(self) -> None:
        self.process = None

    async def start(self, command, env=None, cwd=None) -> None:
        self.process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            cwd=cwd,
            env={**os.environ, **(env or {})},
        )

    async def readline(self) -> bytes:
        assert self.process is not None and self.process.stdout is not None
        return await self.process.stdout.readline()

    async def writeline(self, data: str) -> None:
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write((data + "\n").encode())
        await self.process.stdin.drain()

    async def close_stdin(self) -> bool:
        if self.process is None or self.process.stdin is None:
            return False
        self.process.stdin.close()
        return True

    async def close(self) -> None:
        if self.process is None:
            return
        await self.close_stdin()
        if self.process.returncode is None:
            try:
                await asyncio.wait_for(self.process.wait(), timeout=1)
            except TimeoutError:
                self.process.terminate()
                await self.process.wait()

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None


async def _mutating_acp_rollout(tmp_path, process: LiveProcess):
    mutation_path = tmp_path / "acp-mutations"
    event_path = tmp_path / "acp-events"
    transport = ContainerTransport(
        container_process=process,
        command=shlex.join([sys.executable, str(_MUTATING_ACP_AGENT)]),
        env={
            "MUTATION_PATH": str(mutation_path),
            "EVENT_PATH": str(event_path),
        },
    )
    client = ACPClient(transport)
    await client.connect()
    await client.initialize()
    session = await client.session_new()
    await _wait_for_mutation(mutation_path)

    rollout = Rollout.__new__(Rollout)
    rollout._acp_client = client
    rollout._session = session
    rollout._session_adapter = None
    rollout._is_session_factory = False
    rollout._active_role = None
    rollout._session_tool_count = 0
    rollout._session_traj_count = 0
    rollout._n_tool_calls = 0
    rollout._trajectory = []
    rollout._rollout_dir = tmp_path
    rollout._phase = "executed"
    rollout._termination_receipt = None
    rollout._acp_session_observation = None
    return rollout, mutation_path, event_path


@pytest.mark.asyncio
async def test_rollout_stop_agent_kills_acp_adapter_and_mutating_child(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_base, "_PROCESS_TREE_TERM_TIMEOUT_SEC", 0.1, raising=False
    )
    process = _FixtureProcessGroup()
    rollout, mutation_path, event_path = await _mutating_acp_rollout(tmp_path, process)
    process.protocol_event_path = event_path

    receipt = await rollout.stop_agent(cancel_requested=True)

    assert event_path.read_text().splitlines() == ["cancel", "session_closed"]
    assert receipt.cancel_acknowledged is True
    assert receipt.session_closed is True
    assert receipt.graceful_termination is False
    assert receipt.force_kill_required is True
    assert receipt.process_tree_stopped is True
    assert receipt.capture_safe is True
    stopped_size = mutation_path.stat().st_size
    await asyncio.sleep(0.1)
    assert mutation_path.stat().st_size == stopped_size

    assert process.events == [
        "session_close",
        "liveness",
        "SIGTERM",
        "wait",
        "SIGKILL",
        "wait",
        "liveness",
    ]
    assert process.signal_protocol_events == [
        ["cancel", "session_closed"],
        ["cancel", "session_closed"],
    ]


@pytest.mark.asyncio
async def test_rollout_stop_agent_marks_adapter_only_death_unsafe(tmp_path) -> None:
    process = _FixtureAdapterOnlyProcess()
    rollout, mutation_path, event_path = await _mutating_acp_rollout(tmp_path, process)
    assert process.process is not None
    process_group_id = process.process.pid

    try:
        receipt = await rollout.stop_agent(cancel_requested=True)

        assert event_path.read_text().splitlines() == ["cancel", "session_closed"]
        assert receipt.cancel_acknowledged is True
        assert receipt.session_closed is True
        assert receipt.process_tree_stopped is False
        assert receipt.capture_safe is False
        prior_size = mutation_path.stat().st_size
        await _wait_for_mutation(mutation_path, previous_size=prior_size)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGKILL)


class _PollingRemoteProcess(_RemoteProcessGroupLiveProcess):
    def __init__(self, observations) -> None:
        super().__init__()
        self._observations = iter(observations)

    async def start(self, command, env=None, cwd=None) -> None:
        raise AssertionError("this fake is never launched")

    async def _remote_process_group_alive(self) -> bool | None:
        return next(self._observations)


@pytest.mark.asyncio
async def test_remote_liveness_retries_unknown_until_death_is_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_base, "_PROCESS_TREE_POLL_INTERVAL_SEC", 0.001, raising=False
    )
    process = _PollingRemoteProcess([None, False])

    stopped = await process._wait_for_remote_process_tree(0.1)

    assert stopped is True


class _SlowRemoteProbeProcess(_RemoteProcessGroupLiveProcess):
    async def start(self, command, env=None, cwd=None) -> None:
        raise AssertionError("this fake is never launched")

    async def _remote_process_group_alive(self) -> bool | None:
        await asyncio.sleep(1)
        return True


@pytest.mark.asyncio
async def test_remote_liveness_probe_is_bounded_by_remaining_deadline() -> None:
    process = _SlowRemoteProbeProcess()

    stopped = await asyncio.wait_for(
        process._wait_for_remote_process_tree(0.01),
        timeout=0.2,
    )

    assert stopped is False


class _TrackedControlProcess:
    def __init__(self, process) -> None:
        self._process = process
        self.kill_called = False
        self.wait_completed = False

    @property
    def returncode(self):
        return self._process.returncode

    async def communicate(self):
        return await self._process.communicate()

    def kill(self) -> None:
        self.kill_called = True
        self._process.kill()

    async def wait(self):
        return_code = await self._process.wait()
        self.wait_completed = True
        return return_code


async def _assert_control_process_cleanup_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    backend_module,
    execute_command,
) -> None:
    original_create_subprocess_exec = asyncio.create_subprocess_exec
    started = asyncio.Event()
    control_process = None

    async def create_control_process(*args, **kwargs):
        nonlocal control_process
        child = await original_create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        control_process = _TrackedControlProcess(child)
        started.set()
        return control_process

    monkeypatch.setattr(
        backend_module.asyncio,
        "create_subprocess_exec",
        create_control_process,
    )
    task = asyncio.create_task(execute_command())
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        assert control_process is not None
        assert control_process.returncode is None

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert control_process.kill_called is True
        assert control_process.wait_completed is True
        assert control_process.returncode is not None
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if control_process is not None and control_process.returncode is None:
            control_process.kill()
            await control_process.wait()


@pytest.mark.asyncio
async def test_apple_control_process_is_reaped_before_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchflow.sandbox.process import apple as apple_module
    from benchflow.sandbox.process.apple import AppleContainerProcess

    process = AppleContainerProcess("benchflow-test")

    await _assert_control_process_cleanup_on_cancellation(
        monkeypatch,
        apple_module,
        lambda: process._exec_remote_process_group_command("true"),
    )


@pytest.mark.asyncio
async def test_docker_control_process_is_reaped_before_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchflow.sandbox.process import docker as docker_module
    from benchflow.sandbox.process.docker import DockerProcess

    process = DockerProcess("benchflow-test", "/tmp", [])
    monkeypatch.setattr(process, "_host_env", lambda: {})

    await _assert_control_process_cleanup_on_cancellation(
        monkeypatch,
        docker_module,
        lambda: process._exec_remote_process_group_command("true"),
    )
