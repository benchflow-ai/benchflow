"""Guards FF-sw-launchers against echoed readiness nonces at 363d815b."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox.process.daytona import DaytonaPtyProcess
from tests.test_daytona_pty_liveness import _BOOTSTRAP_OK, FakePtyHandle


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prefix,suffix", [("", ""), ("\x1b[?2004l", "\x1b[0m"), ("\x1b]0;title\x07", "")]
)
async def test_echo_is_not_readiness(prefix, suffix):
    echoed = asyncio.Event()
    release = asyncio.Event()
    handles = []

    class Handle(FakePtyHandle):
        async def send_input(self, data):
            self.inputs.append(data)
            if "echo '" in data:
                await self._on_data(data.encode())
                echoed.set()

                async def ready():
                    await release.wait()
                    marker = data.split("echo '", 1)[1].split("'", 1)[0]
                    await self._on_data(f"{prefix}{marker}{suffix}\n".encode())

                self.ready_task = asyncio.create_task(ready())

    sandbox = MagicMock()

    async def create(*, id, on_data, envs=None):
        handle = Handle(on_data)
        handles.append(handle)
        return handle

    sandbox.process.create_pty_session = AsyncMock(side_effect=create)
    sandbox.process.exec = AsyncMock(return_value=_BOOTSTRAP_OK)
    proc = DaytonaPtyProcess(sandbox, "", "")
    task = asyncio.create_task(proc.start("agent"))
    try:
        await asyncio.wait_for(echoed.wait(), 2)
        await asyncio.sleep(0.01)
        assert not task.done()
        assert len(handles[0].inputs) == 1
        release.set()
        await asyncio.wait_for(task, 2)
        assert len(handles[0].inputs) == 2
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await proc.close()
        if handles:
            await handles[0].ready_task
