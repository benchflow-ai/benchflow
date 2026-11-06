"""Provider and session fakes for command-channel lifecycle tests."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import daytona as module
from benchflow.sandbox import daytona_strategies as strategies
from benchflow.sandbox.daytona import DaytonaSandbox
from benchflow.sandbox.daytona_dind import VM_ROUTE
from benchflow.sandbox.protocol import SandboxImage


class _Process:
    def __init__(self):
        self.created = []
        self.executed = []
        self.deleted = []

    async def create_session(self, session_id):
        self.created.append(session_id)

    async def delete_session(self, session_id):
        self.deleted.append(session_id)

    async def execute_session_command(self, session_id, request, timeout=None):
        assert session_id in self.created
        assert session_id not in self.executed
        assert request.run_async
        self.executed.append(session_id)
        return SimpleNamespace(cmd_id=session_id)

    async def get_session_command(self, session_id, command_id):
        return SimpleNamespace(exit_code=0)

    async def get_session_command_logs(self, session_id, command_id):
        return SimpleNamespace(stdout="BF daemon-log root-only\n", stderr="")


class _Remote:
    def __init__(self, name):
        self.id = name
        self.process = _Process()
        self.deleted = False
        self.deletes = 0
        self.deleting = asyncio.Event()
        self.release = asyncio.Event()
        self.fail_delete = False

    async def delete(self, **kwargs):
        self.deletes += 1
        self.deleting.set()
        await self.release.wait()
        if self.fail_delete:
            raise RuntimeError("provider deletion refused")
        self.deleted = True


class _Ledger:
    def __init__(self):
        self.entries = []

    def open_sandbox(self, client, sandbox_id, reason, **kwargs):
        self.entries.append(sandbox_id)

    def ensure_sweeper(self):
        pass

    def close_sandbox(self, sandbox_id):
        pass

    async def drain(self, holder):
        pass


def _lifecycle_sandbox(monkeypatch, strategy=module._DaytonaDirect):
    pytest.importorskip("daytona")
    module._load_daytona_sdk()
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._sandbox = _Remote("old-box")
    sandbox._auto_delete_interval = sandbox._auto_stop_interval = 0
    sandbox._network_block_all = False
    sandbox._client_manager = None
    sandbox.rollout_paths = None
    sandbox.logger = logging.getLogger("test.daytona.channel.lifecycle")
    if strategy is module._DaytonaDinD:
        sandbox._strategy = strategy.__new__(strategy)
        sandbox._strategy._env = sandbox
        sandbox._strategy._compose_teardown = AsyncMock()
    else:
        sandbox._strategy = strategy(sandbox)
    ledger = _Ledger()
    monkeypatch.setattr(module, "LEDGER", ledger)
    return sandbox, ledger


def _restore_setup(monkeypatch, sandbox, new):
    monkeypatch.setattr(
        strategies, "system_shell_route", AsyncMock(return_value=VM_ROUTE)
    )
    monkeypatch.setattr(strategies, "_deliver_bfx", AsyncMock())

    async def create(*args, **kwargs):
        sandbox._sandbox = new
        sandbox._on_sandbox_created()

    sandbox._create_sandbox_without_toolbox_bash_env = create


def _checkpoint(meta=None):
    startup = {
        module._TOOLBOX_STARTUP_META: json.dumps(
            dict.fromkeys(module.SYSTEM_SHELL_EMPTY_VARS, "")
        )
    }
    return SandboxImage(
        provider="daytona", ref="snapshot", meta={**startup, **(meta or {})}
    )
