from unittest.mock import AsyncMock, MagicMock

import pytest


def _oracle_env():
    env = MagicMock()

    async def exec_side_effect(cmd, **kwargs):
        if "tail -c" in cmd:
            return MagicMock(return_code=0, stdout="oracle preview")
        return MagicMock(return_code=0, stdout="")

    env.exec = AsyncMock(side_effect=exec_side_effect)
    return env


@pytest.mark.asyncio
async def test_run_oracle_redirects_to_container_log(tmp_path):
    from benchflow.sdk import SDK

    solve = tmp_path / "solution" / "solve.sh"
    solve.parent.mkdir()
    solve.write_text("#!/bin/sh\necho ok\n")
    env = _oracle_env()

    trajectory, agent_name = await SDK()._run_oracle(env, tmp_path, timeout=123)

    calls = env.exec.call_args_list
    solve_call = calls[0]
    assert solve_call.args[0] == "bash /solution/solve.sh > /logs/agent/oracle.txt 2>&1"
    assert "tee" not in solve_call.args[0]
    assert solve_call.kwargs["env"] == {"DEBIAN_FRONTEND": "noninteractive"}
    assert solve_call.kwargs["timeout_sec"] == 123
    assert trajectory == [
        {
            "type": "oracle",
            "command": "solution/solve.sh",
            "return_code": 0,
            "stdout": "oracle preview",
        }
    ]
    assert agent_name == "oracle"


@pytest.mark.asyncio
async def test_run_oracle_redirects_sandbox_user_output(tmp_path):
    from benchflow.sdk import SDK

    solve = tmp_path / "solution" / "solve.sh"
    solve.parent.mkdir()
    solve.write_text("#!/bin/sh\necho ok\n")
    env = _oracle_env()

    await SDK()._run_oracle(env, tmp_path, timeout=123, sandbox_user="agent")

    solve_cmd = env.exec.call_args_list[0].args[0]
    assert solve_cmd == (
        "su -s /bin/bash agent -c "
        "'DEBIAN_FRONTEND=noninteractive bash /solution/solve.sh' "
        "> /logs/agent/oracle.txt 2>&1"
    )
    assert "tee" not in solve_cmd
