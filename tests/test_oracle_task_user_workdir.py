"""The oracle runs where and as whom the task says the agent runs.

Harbor runs a task's solve.sh through the environment's default user and
working directory, so ``[agent].user`` and ``[environment].workdir`` apply to
the oracle (Harbor's own ``hello-user`` and ``hello-workdir`` examples check
exactly this). BenchFlow gave the agent the declared workdir but ran the
oracle as root in the image's default directory, so the reference solution of
such a task scored 0 and a task author could not tell a broken solution from a
runner gap. With neither field declared the oracle keeps running as root in
the default directory.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


def _env() -> MagicMock:
    env = MagicMock()

    async def exec_side_effect(cmd, **kwargs):
        return MagicMock(return_code=0, stdout="")

    env.exec = AsyncMock(side_effect=exec_side_effect)
    return env


def _task(root: Path, extra_toml: str = "") -> Path:
    (root / "solution").mkdir(parents=True)
    (root / "solution" / "solve.sh").write_text("#!/bin/bash\npwd > where.txt\n")
    (root / "instruction.md").write_text("Write where.txt.\n")
    (root / "environment").mkdir()
    (root / "task.toml").write_text('version = "1.0"\n' + extra_toml)
    return root


@pytest.mark.asyncio
async def test_oracle_runs_in_the_declared_workdir(tmp_path: Path) -> None:
    from benchflow.rollout._setup import _run_oracle

    env = _env()
    await _run_oracle(
        env,
        _task(tmp_path, '[environment]\nworkdir = "/custom-workdir"\n'),
        timeout=30,
    )

    solve_call = env.exec.call_args_list[0]
    assert solve_call.kwargs.get("cwd") == "/custom-workdir"


@pytest.mark.asyncio
async def test_oracle_runs_as_the_declared_agent_user(tmp_path: Path) -> None:
    from benchflow.rollout._setup import _run_oracle

    env = _env()
    await _run_oracle(env, _task(tmp_path, '[agent]\nuser = "agent"\n'), timeout=30)

    commands = [c.args[0] for c in env.exec.call_args_list]
    solve_cmd = next(c for c in commands if "> /logs/agent/oracle.txt" in c)
    assert solve_cmd.startswith("su -s /bin/bash agent -c ")
    # The uploaded oracle dir is root-owned; the declared user must be able
    # to read and run it (otherwise solve.sh fails with rc=126).
    grant = commands.index("chmod -R a+rX /solution")
    assert grant < commands.index(solve_cmd)


@pytest.mark.asyncio
async def test_oracle_default_stays_root_in_the_default_directory(
    tmp_path: Path,
) -> None:
    from benchflow.rollout._setup import _run_oracle

    env = _env()
    await _run_oracle(env, _task(tmp_path), timeout=30)

    solve_call = env.exec.call_args_list[0]
    assert solve_call.args[0].startswith("bash /solution/solve.sh")
    assert solve_call.kwargs.get("cwd") is None
