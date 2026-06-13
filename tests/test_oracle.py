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


def _scaffold_task(tmp_path, solution_env=None):
    """Create minimal task structure for Task() to parse."""
    solve = tmp_path / "solution" / "solve.sh"
    solve.parent.mkdir(parents=True, exist_ok=True)
    solve.write_text("#!/bin/sh\necho ok\n")
    (tmp_path / "instruction.md").write_text("Solve the task.\n")
    env_section = ""
    if solution_env:
        env_section = "\n[solution.env]\n" + "\n".join(
            f'{k} = "{v}"' for k, v in solution_env.items()
        )
    (tmp_path / "task.toml").write_text(f'version = "1.0"\n{env_section}\n')
    (tmp_path / "environment").mkdir(exist_ok=True)


def _scaffold_native_task(tmp_path, oracle_env=None):
    """Create minimal native task.md structure for Task() to parse."""
    solve = tmp_path / "oracle" / "solve.sh"
    solve.parent.mkdir(parents=True, exist_ok=True)
    solve.write_text("#!/bin/sh\necho ok\n")
    oracle_section = ""
    if oracle_env:
        env_lines = "\n".join(f"    {k}: {v}" for k, v in oracle_env.items())
        oracle_section = f"\noracle:\n  env:\n{env_lines}\n"
    (tmp_path / "task.md").write_text(
        f"""---
version: "1.0"
{oracle_section}---
## prompt

Solve the task.
"""
    )
    (tmp_path / "environment").mkdir(exist_ok=True)


@pytest.mark.asyncio
async def test_run_oracle_redirects_to_container_log(tmp_path):
    from benchflow.sdk import SDK

    _scaffold_task(tmp_path)
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
async def test_run_oracle_uses_native_oracle_mount(tmp_path):
    """Native oracle/ packages execute from /oracle, not legacy /solution."""
    from benchflow.sdk import SDK

    _scaffold_native_task(tmp_path)
    env = _oracle_env()

    trajectory, _agent_name = await SDK()._run_oracle(env, tmp_path, timeout=123)

    solve_call = env.exec.call_args_list[0]
    assert solve_call.args[0] == "bash /oracle/solve.sh > /logs/agent/oracle.txt 2>&1"
    assert trajectory[0]["command"] == "oracle/solve.sh"


@pytest.mark.asyncio
async def test_run_oracle_redirects_sandbox_user_output(tmp_path):
    from benchflow.sdk import SDK

    _scaffold_task(tmp_path)
    env = _oracle_env()

    await SDK()._run_oracle(env, tmp_path, timeout=123, sandbox_user="agent")

    solve_cmd = env.exec.call_args_list[0].args[0]
    assert solve_cmd == (
        "su -s /bin/bash agent -c "
        "'DEBIAN_FRONTEND=noninteractive bash /solution/solve.sh' "
        "> /logs/agent/oracle.txt 2>&1"
    )
    assert env.exec.call_args_list[0].kwargs["user"] == "root"
    assert "tee" not in solve_cmd


@pytest.mark.asyncio
async def test_run_oracle_passes_solution_env(tmp_path):
    """solution.env from task.toml is forwarded to the oracle exec call."""
    from benchflow.sdk import SDK

    _scaffold_task(
        tmp_path,
        solution_env={
            "MY_TOKEN": "secret123",
            "REPO_ID": "org/repo",
        },
    )
    env = _oracle_env()

    await SDK()._run_oracle(env, tmp_path, timeout=60)

    solve_call = env.exec.call_args_list[0]
    passed_env = solve_call.kwargs["env"]
    assert passed_env["MY_TOKEN"] == "secret123"
    assert passed_env["REPO_ID"] == "org/repo"
    assert passed_env["DEBIAN_FRONTEND"] == "noninteractive"


@pytest.mark.asyncio
async def test_run_oracle_no_solution_env_only_has_debian_frontend(tmp_path):
    """Without [solution.env], only DEBIAN_FRONTEND is passed."""
    from benchflow.sdk import SDK

    _scaffold_task(tmp_path)
    env = _oracle_env()

    await SDK()._run_oracle(env, tmp_path, timeout=60)

    solve_call = env.exec.call_args_list[0]
    assert solve_call.kwargs["env"] == {"DEBIAN_FRONTEND": "noninteractive"}


@pytest.mark.asyncio
async def test_run_oracle_solution_env_resolves_host_vars(tmp_path, monkeypatch):
    """${VAR} references in solution.env are resolved from the host environment."""
    from benchflow.sdk import SDK

    monkeypatch.setenv("HOST_SECRET", "resolved_value")
    _scaffold_task(tmp_path, solution_env={"TOKEN": "${HOST_SECRET}"})
    env = _oracle_env()

    await SDK()._run_oracle(env, tmp_path, timeout=60)

    solve_call = env.exec.call_args_list[0]
    assert solve_call.kwargs["env"]["TOKEN"] == "resolved_value"
