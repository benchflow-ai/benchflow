"""Guards EN-925: an agent's blank stderr at close writes no agent log.

``ContainerTransport.close()`` terminates ``docker compose exec -i`` right
after closing its stdin, and a terminated ``docker compose exec`` prints a
lone newline to stderr. Whether the agent had already exited on its closed
stdin was a race, so a protocol-only run's ``agent/<agent>.txt`` (that
newline) was there or not at random: test_local_daemon_docker's records of
a cold and a warm perfacp run differed by that file.
"""

from unittest.mock import AsyncMock

import pytest

from benchflow.acp.container_transport import ContainerTransport
from benchflow.task.paths import RolloutPaths


async def _run(tmp_path, stderr_tail: str):
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    process = AsyncMock()
    process.stderr_tail = stderr_tail
    process.readline.side_effect = [b'{"jsonrpc":"2.0","id":1,"result":{}}\n']
    transport = ContainerTransport(
        process, "perfacp", agent_log_path=paths.agent_dir / "perfacp.txt"
    )
    await transport.start()
    await transport.receive()
    await transport.close()
    return tmp_path / "trial" / "agent" / "perfacp.txt"


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", ["", "\n", " \r\n\n"])
async def test_a_blank_stderr_tail_leaves_no_agent_log(tmp_path, tail):
    assert not (await _run(tmp_path, tail)).exists()


@pytest.mark.asyncio
async def test_stderr_with_text_is_still_the_agent_log(tmp_path):
    log = await _run(tmp_path, "\nwarning: slow start\n")
    assert log.read_text() == "\nwarning: slow start\n"
