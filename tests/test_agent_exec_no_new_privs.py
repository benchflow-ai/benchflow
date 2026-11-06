"""Every agent exec goes through S2's launcher (K1-R11-1).

The kills before grading find the agent's processes by uid; a process under
no_new_privs can never change uid, nor can any child of it.
``lockdown.build_priv_drop_cmd`` (S2) drops from root to the agent user under
``setpriv --no-new-privs`` with the firewall capabilities gone, and refuses
the launch where no usable setpriv is there. At d6850512 a session-factory
agent's commands and the task runtime's bash went to the backend as the
agent user, bare, bypassing it. Here each path must hand the backend exactly
the launcher's command, to run as root.

Except the default-user case (new), these import nothing d6850512 lacks, so
they fail there by behaviour. That
the launcher's process and its children read ``NoNewPrivs: 1`` is shown in
throwaway containers: dropping users
needs root, which a test process does not have.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox.agent_exec import agent_exec_call


def _sandbox():
    return SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(stdout="", stderr="", return_code=0)
        ),
        exec_in_service=AsyncMock(return_value="done"),
        agent_identity_switch=(),
    )


@pytest.mark.asyncio
async def test_a_session_factory_agents_exec_goes_through_the_launcher():
    from benchflow.rollout.session_factory_runtime import SessionFactorySandbox

    inner = _sandbox()
    wrapper = SessionFactorySandbox(inner, {}, None)
    await wrapper.exec("make answer", cwd="/app", user="agent")
    call = inner.exec.await_args
    assert call.args[0] == agent_exec_call("make answer", "agent")[0]
    assert call.kwargs["user"] == "agent"
    assert call.kwargs["cwd"] == "/app"


@pytest.mark.asyncio
async def test_an_exec_that_names_no_user_runs_as_the_trials_agent_user():
    from benchflow.rollout.session_factory_runtime import SessionFactorySandbox

    inner = _sandbox()
    wrapper = SessionFactorySandbox(inner, {}, None, exec_user="agent")
    await wrapper.exec("make answer", cwd="/app")
    call = inner.exec.await_args
    assert call.args[0] == agent_exec_call("make answer", "agent")[0]
    assert call.kwargs["user"] == "agent"
    assert call.kwargs["cwd"] == "/app"


@pytest.mark.asyncio
async def test_a_session_factory_agents_service_exec_goes_through_the_launcher():
    from benchflow.rollout.session_factory_runtime import SessionFactorySandbox

    inner = _sandbox()
    wrapper = SessionFactorySandbox(inner, {}, None)
    await wrapper.exec_in_service("main", "make answer", user="agent")
    call = inner.exec_in_service.await_args
    assert call.args[:2] == (
        "main",
        agent_exec_call("make answer", "agent")[0],
    )
    assert call.kwargs["user"] == "agent"


@pytest.mark.asyncio
async def test_the_task_runtimes_bash_goes_through_the_launcher(tmp_path):
    from benchflow.rollout.task_runtime import TaskRuntime

    inner = _sandbox()
    runtime = TaskRuntime.__new__(TaskRuntime)
    runtime.config = SimpleNamespace(sandbox_user="agent")
    runtime._started = True
    runtime._rollout = SimpleNamespace(
        env=inner,
        _agent_cwd="/app",
        record_external_tool_call=lambda **kwargs: None,
    )
    await runtime.bash("make answer")
    call = inner.exec.await_args
    assert (
        call.args[0] == agent_exec_call("bash -lc 'cd /app && make answer'", "agent")[0]
    )
    assert call.kwargs["user"] == "agent"
