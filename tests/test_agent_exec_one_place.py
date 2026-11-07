"""The one place agent commands get no_new_privs, its refusal and opt-in.

K1-R11-1: every exec path that runs agent code goes through S2's launcher,
``lockdown.build_priv_drop_cmd`` (the ACP and native-harness launches call it;
a session-factory agent's commands and the task runtime's bash reach it
through :func:`benchflow.sandbox.agent_exec.agent_exec_call`); an image with
no usable setpriv refuses the command with what to do; a task that names the
accounts its agent switches to (``x-agent-identity-switch``) keeps that barrier
too (K1-R15-2: such a task runs only where the container has no_new_privs from
creation), and the kills stop those accounts too.
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox.agent_exec import agent_exec_call
from benchflow.task import identity_switch

SRC = Path(__file__).parents[1] / "src" / "benchflow"

# Every place BenchFlow runs agent code, with the helper each must use.
AGENT_EXEC_PATHS = {
    ("rollout/session_factory_runtime.py", "SessionFactorySandbox"): "_agent_call",
    ("rollout/task_runtime.py", "bash"): "agent_exec_call",
}
AGENT_LAUNCHES = ("acp/runtime.py", "native_harness/client.py")


def _scope(path: str, name: str) -> ast.AST:
    tree = ast.parse((SRC / path).read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f"{path}: no {name}")


def _command_argument(call: ast.Call, method: str, offset: int = 0) -> ast.AST | None:
    position = (1 if method == "exec_in_service" else 0) + offset
    if len(call.args) > position:
        return call.args[position]
    for keyword in call.keywords:
        if keyword.arg == "command":
            return keyword.value
    return None


@pytest.mark.parametrize("where", sorted(AGENT_EXEC_PATHS))
def test_every_agent_exec_goes_through_the_one_place(where):
    path, name = where
    helper = AGENT_EXEC_PATHS[where]
    scope = _scope(path, name)
    # Names the helper's answer is unpacked into (``command, user = ...``).
    assigned = {
        name.id
        for node in ast.walk(scope)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func).endswith(helper)
        for target in node.targets
        for name in (target.elts if isinstance(target, ast.Tuple) else [target])
        if isinstance(name, ast.Name)
    }
    # A direct call (sandbox.exec(cmd)) or one through exec_raw
    # (exec_raw(sandbox.exec, cmd)): the command it runs.
    commands = []
    for node in ast.walk(scope):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "exec",
            "exec_in_service",
        }:
            commands.append((node, _command_argument(node, node.func.attr)))
        elif (
            ast.unparse(node.func) == "exec_raw"
            and node.args
            and isinstance(node.args[0], ast.Attribute)
            and node.args[0].attr in {"exec", "exec_in_service"}
        ):
            commands.append(
                (node, _command_argument(node, node.args[0].attr, offset=1))
            )
    assert commands, f"{path}:{name} runs no command"
    for call, command in commands:
        ok = (
            isinstance(command, ast.Call) and ast.unparse(command.func).endswith(helper)
        ) or (isinstance(command, ast.Name) and command.id in assigned)
        assert ok, f"{path}:{call.lineno} runs a command without {helper}"


@pytest.mark.parametrize("path", AGENT_LAUNCHES)
def test_agent_launches_drop_privileges_with_no_new_privs(path):
    assert "build_priv_drop_cmd(" in (SRC / path).read_text()
    assert "--no-new-privs" in lockdown.build_priv_drop_cmd("true", "agent")


def test_the_helper_keeps_agent_commands_under_the_backend_user():
    command = "echo hi"
    wrapped, user = agent_exec_call(command, "agent")
    assert user == "agent"
    assert "/usr/bin/setpriv --no-new-privs" in wrapped
    launched = (wrapped, user)
    assert agent_exec_call(command, "agent") == launched
    assert agent_exec_call(command, None, sandbox_user="agent") == launched
    # Root's own commands are not agent-user code; they run as asked.
    assert agent_exec_call(command, "root") == (command, "root")
    assert agent_exec_call(command, None) == (command, None)


def test_an_image_without_setpriv_refuses_with_what_to_do(tmp_path):
    """The launcher's own refusal: no setpriv on PATH, the command never runs."""
    tools = tmp_path / "bin"
    tools.mkdir()
    for tool in ("bash", "grep", "touch"):
        found = subprocess.run(
            ["bash", "-c", f"command -v {tool}"], capture_output=True, text=True
        ).stdout.strip()
        os.symlink(found, tools / tool)
    ran = tmp_path / "ran"
    command = lockdown.build_priv_drop_cmd(f"touch {ran}", "agent")
    result = subprocess.run(
        [str(tools / "bash"), "-c", command],
        capture_output=True,
        text=True,
        env={"PATH": str(tools)},
    )
    assert result.returncode != 0
    # PT's launch takes setpriv by absolute path, never from PATH (C8), so
    # the image's own setpriv may refuse first; either way nothing runs.
    assert "requires util-linux setpriv" in result.stderr or "setpriv:" in result.stderr
    assert not ran.exists()


@pytest.mark.parametrize(
    "value, accounts",
    [(["worker"], ("worker",)), (["a", "b", "a"], ("a", "b"))],
)
def test_the_opt_in_names_accounts(value, accounts):
    assert identity_switch.read(value) == accounts


@pytest.mark.parametrize(
    "value", [[], "worker", ["root"], ["Worker"], ["a b"], [1], ["x"] * 9]
)
def test_the_opt_in_refuses_what_is_not_a_list_of_accounts(value):
    with pytest.raises(ValueError, match="x-agent-identity-switch"):
        identity_switch.read(value)


def test_a_task_declares_the_opt_in_in_its_metadata():
    task = SimpleNamespace(
        config=SimpleNamespace(metadata={"x-agent-identity-switch": ["worker"]})
    )
    assert identity_switch.for_task(task) == ("worker",)
    assert identity_switch.for_task(SimpleNamespace(config=None)) == ()


def test_with_the_opt_in_agent_commands_keep_the_barrier():
    """K1-R15-2: at 6c7d11fa5 the opt-in returned the bare command, so on a
    backend without the container's no_new_privs a sudo could cross accounts
    and restart one already cleaned."""
    command, user = agent_exec_call("id", "agent", identity_switch=("worker",))
    assert user == "agent"
    assert command != "id"
    assert "--no-new-privs" in command
    assert getattr(command, "agent_command", None) == "id"


@pytest.mark.asyncio
async def test_with_the_opt_in_the_session_factory_still_wraps_commands():
    from benchflow.rollout.session_factory_runtime import SessionFactorySandbox

    inner = SimpleNamespace(
        exec=AsyncMock(return_value="done"), agent_identity_switch=("worker",)
    )
    await SessionFactorySandbox(inner, {}, None, "agent").exec("id", user="agent")
    assert inner.exec.await_args.args[0] != "id"
    assert "--no-new-privs" in inner.exec.await_args.args[0]
    assert inner.exec.await_args.kwargs["user"] == "agent"


@pytest.mark.asyncio
async def test_the_kills_stop_the_declared_accounts_too(monkeypatch):
    killed = []

    async def kill(env, user, *, proc_root="/proc", record=True):
        killed.append((user, record))
        return 2

    monkeypatch.setattr(lockdown, "_kill_sandbox_user_procs", kill)
    env = SimpleNamespace(agent_identity_switch=("worker", "builder"))
    assert await lockdown.kill_switched_accounts(env) == 4
    # Each account is stopped; the agent user's kill record is left alone.
    assert killed == [("worker", False), ("builder", False)]
    assert await lockdown.kill_switched_accounts(SimpleNamespace()) == 0


@pytest.mark.asyncio
async def test_the_turn_end_kill_stops_the_declared_accounts(monkeypatch):
    from benchflow import rollout_planes

    order = []

    async def switched(env):
        order.append("accounts")
        return 0

    async def kill(env, user):
        order.append(user)
        return 0

    monkeypatch.setattr(rollout_planes, "kill_switched_accounts", switched)
    monkeypatch.setattr(rollout_planes, "_kill_sandbox_user_procs", kill)
    await rollout_planes.DefaultRolloutPlanes().quiesce_agent(object(), "agent")
    assert order == ["accounts", "agent"]
