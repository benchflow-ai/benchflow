"""The oracle runs where and as whom the task says the agent runs.

The oracle must use the task's declared ``[agent].user`` and
``[environment].workdir``. BenchFlow gave the agent the declared workdir but ran the
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


@pytest.mark.asyncio
async def test_an_agent_user_oracle_that_keeps_services_writes_outside_logs(
    tmp_path: Path,
) -> None:
    """K2p-R1-A-2: a service solve.sh starts without a redirect of its own
    inherits the oracle's output file, and the release kills a kept process
    that holds a file in /logs. With kept services the output goes to a new
    file under /tmp while solve.sh runs and is copied into
    /logs/agent/oracle.txt after it returns. K2p-R2-B-2: root opens
    /logs/agent/oracle.txt (in a folder the agent user owns) before solve.sh
    runs, solve.sh does not inherit it, and the copy goes through it, so a
    link solve.sh puts there is not followed."""
    from benchflow.rollout._setup import _run_oracle

    env = _env()
    toml = (
        '[metadata]\nx-keep-services = [{ port = 5000 }]\n\n[agent]\nuser = "agent"\n'
    )
    # Guards DT2-CONF-7 at b891c49a: this direct caller has no install phase.
    from benchflow.sandbox._identities import Identities

    env._benchflow_identities = Identities(
        "root:x:0:0::/:/bin/sh\nagent:x:1000:1000::/:/bin/sh\n",
        "root:x:0:\nagent:x:1000:\n",
    )

    # Guards DT2-R11-1: setup asks the oracle's own su for the account.
    async def exec_side_effect(cmd, **kwargs):
        stdout = "benchflow-id:1000:1000\n" if "benchflow-id" in cmd else ""
        return MagicMock(return_code=0, stdout=stdout)

    env.exec = AsyncMock(side_effect=exec_side_effect)
    await _run_oracle(env, _task(tmp_path, toml), timeout=30)

    commands = [c.args[0] for c in env.exec.call_args_list]
    probe = next(c for c in commands if "benchflow-id" in c)
    assert probe.startswith("su -s /bin/bash agent -c ")
    solve_cmd = next(
        c for c in commands if "su -s /bin/bash agent -c " in c and c != probe
    )
    assert solve_cmd.startswith("o=$(mktemp /tmp/benchflow-oracle.XXXXXX")
    opened = solve_cmd.index("exec 3>/logs/agent/oracle.txt; ")
    run = solve_cmd.index('/solution/solve.sh\' 3>&- > "$o" 2>&1')
    copy = solve_cmd.index('cat "$o" >&3; exec 3>&-;')
    assert opened < run < copy
    assert solve_cmd.count("/logs/agent/oracle.txt") == 2  # the open, the fallback
    assert solve_cmd.rstrip().endswith("exit $rc")
