"""Tests for Daytona SDK session command polling behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


class TestDaytonaCommandPolling:
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
