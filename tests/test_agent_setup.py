from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._agent_setup import install_agent
from benchflow.models import AgentInstallError


@pytest.mark.asyncio
async def test_install_agent_writes_command_stdout_and_stderr_on_failure(tmp_path: Path):
    env = SimpleNamespace()
    env.exec = AsyncMock(
        side_effect=[
            SimpleNamespace(return_code=1, stdout="", stderr="uv: command not found\n"),
            SimpleNamespace(
                stdout="OS:\nID=ubuntu\nNode:\nv22.0.0\nAgent:\nnot found\n",
                stderr="",
                return_code=0,
            ),
        ]
    )

    with pytest.raises(AgentInstallError) as exc_info:
        await install_agent(env, "openhands", tmp_path)

    err = exc_info.value
    log_path = Path(err.log_path)
    assert log_path == tmp_path / "agent" / "install-stdout.txt"
    assert log_path.exists()
    log_text = log_path.read_text()
    assert log_text.startswith("$ ")
    assert "uv tool install openhands --python 3.12" in log_text
    assert "=== stderr ===" in log_text
    assert "uv: command not found" in log_text
    assert err.stdout == log_text
    assert "ID=ubuntu" in err.diagnostics
