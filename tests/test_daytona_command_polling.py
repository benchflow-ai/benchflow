"""Tests for Daytona SDK session command polling behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


class TestDaytonaCommandPolling:
    @pytest.mark.asyncio
    async def test_transient_exec_deletes_completed_session(self) -> None:
        """Guards the Daytona live-capture session fix in PR #921."""
        pytest.importorskip("daytona")
        from benchflow.sandbox.daytona import (
            DaytonaSandbox,
            _DaytonaDirect,
            _load_daytona_sdk,
        )

        _load_daytona_sdk()

        class FakeProcess:
            def __init__(self) -> None:
                self.created: list[str] = []
                self.deleted: list[str] = []

            async def create_session(self, session_id):
                self.created.append(session_id)

            async def execute_session_command(self, session_id, request, timeout=None):
                return SimpleNamespace(cmd_id="cmd-1")

            async def get_session_command(self, session_id, command_id):
                return SimpleNamespace(id="cmd-1", exit_code=0)

            async def get_session_command_logs(self, session_id, command_id):
                return SimpleNamespace(stdout="ok", stderr="")

            async def delete_session(self, session_id):
                self.deleted.append(session_id)

        process = FakeProcess()
        sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
        sandbox.default_user = None
        sandbox._persistent_env = {}
        sandbox._sandbox = SimpleNamespace(process=process)
        sandbox._strategy = _DaytonaDirect(sandbox)

        result = await sandbox.exec_transient("echo ok", timeout_sec=5)

        assert result.return_code == 0
        assert result.stdout == "ok"
        assert process.deleted == process.created

    @pytest.mark.asyncio
    async def test_exec_times_out_when_daytona_command_never_exits(self) -> None:
        """Guards the v0.5 Daytona polling timeout regression."""
        pytest.importorskip("daytona")  # sandbox-daytona optional dependency
        from benchflow.sandbox.daytona import (
            DaytonaSandbox,
            _DaytonaDirect,
            _load_daytona_sdk,
        )

        # ``__init__`` is what normally materializes the SDK handles
        # ``_sandbox_exec`` consumes. This test bypasses ``__init__`` via
        # ``__new__``, so trigger the same lazy-load explicitly.
        _load_daytona_sdk()

        class FakeProcess:
            async def create_session(self, session_id):
                pass

            async def execute_session_command(self, session_id, request, timeout=None):
                assert timeout == 0.01
                return SimpleNamespace(cmd_id="cmd-1")

            async def get_session_command(self, session_id, command_id):
                return SimpleNamespace(id="cmd-1", exit_code=None)

            async def get_session_command_logs(self, session_id, command_id):
                raise AssertionError("logs should not be fetched before timeout")

        sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
        sandbox.default_user = None
        sandbox._persistent_env = {}
        sandbox._sandbox = SimpleNamespace(process=FakeProcess())
        sandbox._strategy = _DaytonaDirect(sandbox)

        with pytest.raises(RuntimeError, match=r"Command timed out after 0.01 seconds"):
            await asyncio.wait_for(
                sandbox.exec("sleep forever", timeout_sec=0.01),
                timeout=0.5,
            )
