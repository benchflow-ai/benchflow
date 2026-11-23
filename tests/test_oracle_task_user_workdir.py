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

from benchflow.rollout._setup import ORACLE_IMAGE_PATH


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
    assert solve_call.args[0].startswith(ORACLE_IMAGE_PATH + "bash /solution/solve.sh")
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
    assert solve_cmd.startswith("o=$(mktemp /tmp/runtime-oracle.XXXXXX")
    opened = solve_cmd.index("exec 3>/logs/agent/oracle.txt; ")
    run = solve_cmd.index('/solution/solve.sh\' 3>&- > "$o" 2>&1')
    copy = solve_cmd.index('cat "$o" >&3; exec 3>&-;')
    assert opened < run < copy
    assert solve_cmd.count("/logs/agent/oracle.txt") == 2  # the open, the fallback
    assert solve_cmd.rstrip().endswith("exit $rc")


@pytest.mark.asyncio
async def test_an_oracle_that_switches_user_runs_su_as_root(tmp_path: Path) -> None:
    """su to the task's account runs from root whatever the image's USER."""
    from benchflow.rollout._setup import _run_oracle

    env = _env()
    await _run_oracle(env, _task(tmp_path), timeout=30, sandbox_user="agent")

    solve = next(
        c for c in env.exec.call_args_list if "> /logs/agent/oracle.txt" in c.args[0]
    )
    assert solve.args[0].startswith("su -s /bin/bash agent -c ")
    assert solve.kwargs.get("user") == "root"


@pytest.mark.asyncio
async def test_the_oracle_identity_probe_runs_su_as_root(tmp_path: Path) -> None:
    from benchflow.rollout._setup import _run_oracle
    from benchflow.sandbox._identities import Identities

    env = _env()
    env._benchflow_identities = Identities(
        "root:x:0:0::/:/bin/sh\nagent:x:1000:1000::/:/bin/sh\n",
        "root:x:0:\nagent:x:1000:\n",
    )

    async def exec_side_effect(cmd, **kwargs):
        stdout = "benchflow-id:1000:1000\n" if "benchflow-id" in cmd else ""
        return MagicMock(return_code=0, stdout=stdout)

    env.exec = AsyncMock(side_effect=exec_side_effect)
    toml = (
        '[metadata]\nx-keep-services = [{ port = 5000 }]\n\n[agent]\nuser = "agent"\n'
    )
    await _run_oracle(env, _task(tmp_path, toml), timeout=30)

    probe = next(c for c in env.exec.call_args_list if "benchflow-id" in c.args[0])
    assert probe.args[0].startswith("su -s /bin/bash agent -c ")
    assert probe.kwargs.get("user") == "root"


@pytest.mark.asyncio
async def test_an_oracle_with_no_account_runs_as_root_whatever_the_image_user(
    tmp_path: Path,
) -> None:
    """With no account named, solve.sh runs as root, not as the image's USER:
    setup gives the workspace and /logs/agent to the sandbox user, and the
    reference solution of an image ending with USER nobody never ran."""
    from benchflow.rollout._setup import _run_oracle

    env = _env()
    await _run_oracle(env, _task(tmp_path), timeout=30)

    solve = env.exec.call_args_list[0]
    assert solve.args[0].startswith(ORACLE_IMAGE_PATH + "bash /solution/solve.sh")
    assert solve.kwargs.get("user") == "root"


_IMAGE_PATH = 'export PATH="${_BENCHFLOW_IMAGE_PATH-$PATH}"; '


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox_user", [None, "agent"], ids=["root", "su"])
async def test_the_root_launch_gives_solve_sh_the_images_path(
    tmp_path: Path, sandbox_user: str | None
) -> None:
    """An exec that names root starts on root's fixed PATH, the image's kept
    aside: solve.sh gets the image's back, as it had when the exec named no
    user. A conda image's only python is in /opt/conda/bin, and its oracle
    scored 0 (test_pt_integ_r3_conda_path_docker)."""
    import shlex

    from benchflow.rollout._setup import _run_oracle

    env = _env()
    await _run_oracle(env, _task(tmp_path), timeout=30, sandbox_user=sandbox_user)

    solve = next(
        c for c in env.exec.call_args_list if "> /logs/agent/oracle.txt" in c.args[0]
    )
    assert solve.kwargs.get("user") == "root"
    if sandbox_user:
        words = shlex.split(solve.args[0])
        assert words[:4] == ["su", "-s", "/bin/bash", "agent"]
        inner = words[words.index("-c") + 1]
        assert inner.startswith(_IMAGE_PATH + "DEBIAN_FRONTEND=noninteractive bash ")
    else:
        assert solve.args[0].startswith(_IMAGE_PATH + "bash /solution/solve.sh")


@pytest.mark.asyncio
async def test_a_path_the_solution_env_sets_is_kept(tmp_path: Path) -> None:
    from benchflow.rollout._setup import _run_oracle

    env = _env()
    toml = '[solution.env]\nPATH = "/opt/tool/bin:/usr/bin:/bin"\n'
    await _run_oracle(env, _task(tmp_path, toml), timeout=30)

    solve = env.exec.call_args_list[0]
    assert solve.args[0].startswith("bash /solution/solve.sh")
    assert solve.kwargs["env"]["PATH"] == "/opt/tool/bin:/usr/bin:/bin"


def _receipt_env(shown: dict[str, str]) -> MagicMock:
    """An env whose receipt reads answer as a sandbox would: the workspace
    step's ``BF workspace-mode`` line (*shown*: the mode by the step's chmod),
    every other command answering 0 with no output."""
    from benchflow.sandbox._exec_receipt import encode_receipt, unwrap_receipt_command

    env = _env()

    async def answers(cmd, **kwargs):
        unwrapped = unwrap_receipt_command(cmd)
        if unwrapped is None:
            return MagicMock(return_code=0, stdout="")
        inner, nonce = unwrapped
        out = "".join(
            f"BF workspace-mode {mode} 0 /root\n"
            for chmod, mode in shown.items()
            if chmod in inner
        )
        return MagicMock(
            return_code=0, stdout=encode_receipt(out.encode(), 0, nonce), stderr=""
        )

    env.exec = AsyncMock(side_effect=answers)
    return env


@pytest.mark.asyncio
async def test_a_root_oracle_gets_the_granted_workspace_mode_back_first(
    tmp_path: Path,
) -> None:
    """#1249: a workspace the grant made sticky loses group and other write
    before a root solve.sh and gets the grant's 1777 back after it; an oracle
    that runs as the task's [agent].user keeps the grant (row A2's unit)."""
    from benchflow._runtime_tree import WORKSPACE_MODE_ATTR
    from benchflow.rollout._setup import _run_oracle

    env = _receipt_env({"chmod go-w /root": "1755", "chmod 1777 /root": "1777"})
    setattr(env, WORKSPACE_MODE_ATTR, {"/root": "700"})
    await _run_oracle(env, _task(tmp_path / "root"), timeout=30)
    commands = [c.args[0] for c in env.exec.call_args_list]
    solve = next(i for i, c in enumerate(commands) if "solve.sh" in c)
    assert "chmod go-w /root" in "\n".join(commands[:solve])
    assert "chmod 1777 /root" in "\n".join(commands[solve + 1 :])
    assert "chmod 700" not in "\n".join(commands)

    env = _receipt_env({"chmod go-w /root": "1755", "chmod 1777 /root": "1777"})
    setattr(env, WORKSPACE_MODE_ATTR, {"/root": "700"})
    await _run_oracle(
        env, _task(tmp_path / "user", '[agent]\nuser = "agent"\n'), timeout=30
    )
    assert not any("chmod go-w /root" in c.args[0] for c in env.exec.call_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shown",
    [{}, {"chmod go-w /root": "1777"}],
    ids=["nothing-shown", "still-1777"],
)
async def test_a_root_oracles_restore_that_does_not_take_is_a_trial_error(
    tmp_path: Path, shown: dict[str, str]
) -> None:
    """Row A1's check: the step before a root solve.sh shows the mode it
    left; none shown, or group and other write still there, fails the oracle
    before solve.sh runs, never a quiet 0, and the grant's mode is still
    asked back.

    Mutant: ignore what the step shows."""
    from benchflow._runtime_tree import WORKSPACE_MODE_ATTR
    from benchflow.rollout._setup import _run_oracle

    env = _receipt_env(shown)
    setattr(env, WORKSPACE_MODE_ATTR, {"/root": "700"})
    with pytest.raises(
        RuntimeError, match=r"Verifier hardening failed: .* before solve\.sh"
    ):
        await _run_oracle(env, _task(tmp_path), timeout=30)
    commands = [c.args[0] for c in env.exec.call_args_list]
    assert not any("solve.sh" in c for c in commands)
    assert "chmod 1777 /root" in commands[-1]


@pytest.mark.asyncio
async def test_an_agent_user_oracle_gets_the_grant_back_after_a_verify(
    tmp_path: Path,
) -> None:
    """Row S4 (unit): an oracle that runs as the task's [agent].user, after a
    step's verify took group and other write (a multi-step task), gets the
    grant's 1777 back before its solve.sh.

    Mutant: no re-open for an [agent].user oracle."""
    from benchflow._runtime_tree import WORKSPACE_CLOSED_ATTR
    from benchflow.rollout._setup import _run_oracle

    env = _receipt_env({"chmod 1777 /root": "1777"})
    setattr(env, WORKSPACE_CLOSED_ATTR, {"/root": "700"})
    await _run_oracle(env, _task(tmp_path, '[agent]\nuser = "agent"\n'), timeout=30)
    commands = [c.args[0] for c in env.exec.call_args_list]
    solve = next(i for i, c in enumerate(commands) if "solve.sh" in c)
    assert "chmod 1777 /root" in "\n".join(commands[:solve])
    assert getattr(env, WORKSPACE_CLOSED_ATTR) is None
