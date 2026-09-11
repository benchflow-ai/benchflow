"""Timeout teardown must survive a child that exits inside the timeout window (#1065).

Every ``except TimeoutError`` teardown in the codebase signals the child before
reporting the timeout. ``terminate()``/``kill()`` raise ``ProcessLookupError``
once the child has exited and been reaped, and asyncio reaps as soon as the
process ends — so the signal is a race against a child that may already be gone.

Losing that race is not a cosmetic failure. ``ProcessLookupError`` escapes the
handler, the ``RuntimeError("Command timed out ...")`` underneath never runs,
and the caller sees an exception whose ``args`` are empty. ``_verify_rollout``
renders it with ``f"verifier crashed: {e}"``, so a finished rollout is scored
``rewards: null`` under a message that ends at the colon, with the one fact that
would explain it — that this was a timeout — destroyed on the way out.

The race is timing-dependent in production. These tests reproduce it
deterministically: the child is genuinely spawned, exited and reaped, and
``communicate`` raises ``TimeoutError`` instead of sleeping, so each test
exercises the real teardown sequence without waiting on a real clock or
gambling on a scheduling coin flip.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

# Captured before any monkeypatching so the helpers below can still spawn while
# the code under test sees a patched ``create_subprocess_exec``.
_spawn = asyncio.create_subprocess_exec


async def _reaped_child() -> asyncio.subprocess.Process:
    """A real, already-exited, already-reaped asyncio child process.

    Not a mock: ``ProcessLookupError`` here is raised by the OS through
    asyncio's own transport, so these tests keep pinning real behavior rather
    than a fixture's idea of it.
    """
    process = await _spawn(
        "sh",
        "-c",
        "exit 0",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    await process.communicate()
    return process


def _patch_spawn_with_timed_out_child(monkeypatch, *, grace_expires: bool = False):
    """Make every spawn yield a reaped child whose ``communicate`` times out.

    ``grace_expires`` extends the timeout to the post-``terminate`` drain as
    well, which is what pushes the teardown on to ``kill()`` — the second half
    of the same race.
    """
    timeouts = 2 if grace_expires else 1

    async def spawn(*_args, **_kwargs):
        process = await _reaped_child()
        calls = {"n": 0}

        async def communicate(*_a, **_k):
            calls["n"] += 1
            if calls["n"] <= timeouts:
                raise TimeoutError
            return (b"", b"")

        # wraps= keeps attribute access working; the process-control methods
        # are bound to the real child so the signals hit a real reaped pid.
        wrapper = MagicMock(wraps=process)
        wrapper.communicate = communicate
        wrapper.terminate = process.terminate
        wrapper.kill = process.kill
        wrapper.wait = process.wait
        wrapper.returncode = process.returncode
        return wrapper

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)


class TestReapedChildPremise:
    """The premise the rest of the module depends on."""

    @pytest.mark.asyncio
    async def test_signalling_a_reaped_child_raises_an_empty_error(self) -> None:
        process = await _reaped_child()

        with pytest.raises(ProcessLookupError) as excinfo:
            process.terminate()

        # Empty args are what make this failure mode invisible downstream: the
        # generic handler in _verify_rollout formats the exception, not its type.
        assert excinfo.value.args == ()
        assert str(excinfo.value) == ""
        assert f"verifier crashed: {excinfo.value}" == "verifier crashed: "


class TestDockerComposeExecTimeout:
    """``DockerSandbox.exec`` — the path that carries the hardening execs."""

    @staticmethod
    def _sandbox(tmp_path: Path):
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.session_id = "teardown-race"
        sandbox.environment_dir = tmp_path
        return sandbox

    @staticmethod
    def _stub_compose_config():
        from benchflow.sandbox.docker import DockerSandbox

        return (
            patch.object(
                DockerSandbox,
                "_docker_compose_paths",
                new_callable=PropertyMock,
                return_value=[],
            ),
            patch.object(DockerSandbox, "_docker_compose_env", return_value={}),
        )

    @pytest.mark.asyncio
    async def test_timeout_is_reported_as_a_timeout(self, tmp_path, monkeypatch):
        """The RuntimeError must survive a terminate() that finds no child."""
        _patch_spawn_with_timed_out_child(monkeypatch)
        sandbox = self._sandbox(tmp_path)
        paths, env = self._stub_compose_config()

        with paths, env, pytest.raises(RuntimeError, match="timed out after"):
            await sandbox._run_docker_compose_command(
                ["exec", "-T", "main", "sh", "-c", "printenv PATH"],
                check=False,
                timeout_sec=1,
            )

    @pytest.mark.asyncio
    async def test_timeout_survives_a_racing_kill(self, tmp_path, monkeypatch):
        """The escalation path — grace period elapses, then kill() — races too."""
        _patch_spawn_with_timed_out_child(monkeypatch, grace_expires=True)
        sandbox = self._sandbox(tmp_path)
        paths, env = self._stub_compose_config()

        with paths, env, pytest.raises(RuntimeError, match="timed out after"):
            await sandbox._run_docker_compose_command(
                ["exec", "-T", "main", "sh", "-c", "printenv PATH"],
                check=False,
                timeout_sec=1,
            )


class TestPreComposeHookTimeout:
    """The pre-compose hook repeats the teardown, so it repeats the race."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("grace_expires", [False, True])
    async def test_hook_timeout_is_reported_as_a_timeout(
        self, tmp_path, monkeypatch, grace_expires
    ):
        from benchflow.sandbox.docker import DockerSandbox

        hook = tmp_path / "pre_compose.sh"
        hook.write_text("#!/bin/sh\nexit 0\n")

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.environment_dir = tmp_path
        sandbox.environment_name = "teardown-race"
        sandbox.task_env_config = MagicMock(build_timeout_sec=1)

        _patch_spawn_with_timed_out_child(monkeypatch, grace_expires=grace_expires)

        with (
            patch.object(
                DockerSandbox,
                "_pre_compose_hook_path",
                new_callable=PropertyMock,
                return_value=hook,
            ),
            patch.object(DockerSandbox, "_docker_compose_env", return_value={}),
            pytest.raises(RuntimeError, match="timed out after"),
        ):
            await sandbox._run_pre_compose_hook()


class TestAppleContainerExecTimeout:
    """The Apple backend re-raises the TimeoutError, so the race replaces it."""

    @pytest.mark.asyncio
    async def test_timeout_is_reraised_as_a_timeout(self, monkeypatch) -> None:
        from benchflow.sandbox import apple_container

        _patch_spawn_with_timed_out_child(monkeypatch)

        with pytest.raises(TimeoutError):
            await apple_container._run_cli("ls", timeout=1)

    @pytest.mark.asyncio
    async def test_env_write_timeout_is_reraised_as_a_timeout(self, monkeypatch):
        """The Apple LiveProcess env write repeats the same teardown."""
        from benchflow.sandbox.process.apple import AppleContainerProcess

        _patch_spawn_with_timed_out_child(monkeypatch)

        process = AppleContainerProcess.__new__(AppleContainerProcess)
        process._container_name = "teardown-race"
        process._env_path = "/tmp/agent-env"

        with pytest.raises(TimeoutError):
            await process._write_env_to_container({"A": "b"})


class TestVerifierErrorIsNeverDetailFree:
    """A recorded verifier error must name something, whatever was raised.

    ``self._error = describe_exception(e)`` on the agent side already documents
    why ``str(e)`` is not enough for a persisted artifact. The verifier side
    used plain interpolation, so the exception that #1065 delivers — raised
    with no args — was recorded as a bare prefix.
    """

    def test_describe_exception_names_an_argument_less_exception(self) -> None:
        from benchflow._utils.text import describe_exception

        assert describe_exception(ProcessLookupError()) == (
            "ProcessLookupError (no message)"
        )

    @pytest.mark.asyncio
    async def test_hardening_failure_is_recorded_with_its_type(self, tmp_path) -> None:
        """The exact shape of #1065: hardening raises, scoring records it.

        ``harden_before_verify`` is where the timed-out execs live, so an
        exception escaping it lands in the generic handler that writes
        ``verifier_error``. It must not write a bare prefix.
        """
        from benchflow.rollout._setup import _verify_rollout

        planes = MagicMock()
        planes.harden_before_verify = AsyncMock(side_effect=ProcessLookupError())

        rollout_paths = MagicMock()
        rollout_paths.verifier_dir = tmp_path / "verifier"

        task = MagicMock()
        task.config.verifier.timeout_sec = 60

        rewards, verifier_error, verifier_timeout = await _verify_rollout(
            env=MagicMock(),
            task=task,
            rollout_paths=rollout_paths,
            timing={},
            planes=planes,
        )

        assert rewards is None
        assert verifier_timeout is None
        assert verifier_error == "verifier crashed: ProcessLookupError (no message)"
        # The precise regression: a message that stops at the colon names
        # neither the failure nor the fact that it carried no detail.
        assert not verifier_error.endswith(": ")


class TestAcpTransportClose:
    """Closing an ACP transport after the agent exits must not raise.

    ``close()`` runs in teardown, typically while another exception is in
    flight, so raising here replaces the real failure with an empty
    ``ProcessLookupError``. ``sandbox/process/_base.py`` already guards its
    equivalent ``close()`` with a ``returncode`` check and documents it as
    "safe to call after process death"; this asserts the ACP transport keeps
    the same promise.
    """

    @pytest.mark.asyncio
    async def test_close_after_the_agent_process_exits(self) -> None:
        from benchflow.acp.transport import StdioTransport

        transport = StdioTransport.__new__(StdioTransport)
        transport._process = await _reaped_child()

        await transport.close()

    @pytest.mark.asyncio
    async def test_subprocess_live_process_close_after_death_stays_safe(self) -> None:
        """Guard the neighbour that already behaves, so it cannot regress."""
        from benchflow.sandbox.process._base import SubprocessLiveProcess

        class _Concrete(SubprocessLiveProcess):
            async def start(self, *args, **kwargs) -> None: ...

        process = _Concrete()
        process._process = await _reaped_child()

        await process.close()
