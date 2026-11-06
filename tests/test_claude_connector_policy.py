"""Guards PR #79 managed Claude connector restrictions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.agents.claude_connector_policy import (
    MCP_REFUSAL,
    enforce_claude_connector_policy,
)
from benchflow.agents.registry import AGENTS

pytestmark = pytest.mark.claude_connector_policy


@pytest.mark.asyncio
@pytest.mark.parametrize("user", [None, "root", "0"])
async def test_policy_refuses_root(user):
    with pytest.raises(RuntimeError, match="non-root"):
        await enforce_claude_connector_policy(
            SimpleNamespace(),
            agent="claude-agent-acp",
            agent_launch=AGENTS["claude-agent-acp"].launch_cmd,
            agent_env={},
            sandbox_user=user,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, stdout", [(1, ""), (0, ""), (0, "connector-policy-installed")]
)
async def test_policy_requires_successful_install_and_verification(
    monkeypatch, code, stdout
):
    execute = AsyncMock(return_value=SimpleNamespace(return_code=code, stdout=stdout))
    monkeypatch.setattr("benchflow.agents.claude_connector_policy.exec_raw", execute)
    env = {"CLAUDE_CODE_OAUTH_TOKEN": "not-for-probe"}
    call = enforce_claude_connector_policy(
        SimpleNamespace(exec=object()),
        agent="claude-agent-acp",
        agent_launch=AGENTS["claude-agent-acp"].launch_cmd,
        agent_env=env,
        sandbox_user="agent",
    )
    if code or not stdout:
        with pytest.raises(RuntimeError, match="could not be verified"):
            await call
    else:
        assert await call == env
    assert "not-for-probe" not in str(execute.call_args)


@pytest.mark.asyncio
async def test_task_mcp_is_refused_explicitly_before_launch(tmp_path):
    from benchflow.acp.runtime import connect_acp
    from benchflow.acp.types import McpServerSpec

    with pytest.raises(
        RuntimeError, match="task requires an agent with task MCP support"
    ):
        await connect_acp(
            None,
            "claude-agent-acp",
            "unused",
            {},
            "agent",
            None,
            tmp_path,
            "docker",
            "/app",
            mcp_servers=[McpServerSpec(name="task", command="server")],
        )


def _claude_trial(tmp_path, harness, *servers):
    from benchflow.rollout import Rollout, RolloutConfig
    from tests.test_acp_mcp_servers import _fake_planes, _task_with_mcp

    trial = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            agent="claude-agent-acp",
            model="test-model",
            harness=harness,
        )
    )
    trial._env = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(
                return_code=0, stdout="connector-policy-installed\n", stderr=""
            )
        )
    )
    trial._rollout_dir = tmp_path
    trial._agent_launch = AGENTS["claude-agent-acp"].launch_cmd
    trial._phase = "installed"
    trial._task = _task_with_mcp(*servers)
    trial._planes = _fake_planes()
    return trial


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["acp", "native"])
async def test_a_task_with_mcp_servers_is_refused_on_either_harness(
    tmp_path, harness, monkeypatch
):
    """Guards the merge of PR #79: its refusal was in connect_acp alone, while
    the managed settings hold for both harnesses. The native harness passed
    the task's servers with --mcp-config, Claude Code 2.1.316 dropped them
    ("MCP server blocked by enterprise policy: task") and the trial ran on
    without its tools, scored as the model's failure."""
    from benchflow.task.config import MCPServerConfig

    trial = _claude_trial(
        tmp_path,
        harness,
        MCPServerConfig(name="task", transport="stdio", command="server"),
    )
    native = AsyncMock(side_effect=AssertionError("the native harness started"))
    monkeypatch.setattr(trial, "_connect_live_agent", native)
    with pytest.raises(RuntimeError) as refused:
        await trial.connect()
    assert str(refused.value) == MCP_REFUSAL
    native.assert_not_awaited()
    trial._planes.connect_acp.assert_not_awaited()
    trial._env.exec.assert_not_awaited()  # no managed settings were written


@pytest.mark.asyncio
async def test_a_connection_runs_the_policy_before_the_agent(tmp_path):
    """Guards PR #79: connect() installs and verifies the managed settings,
    as root, before ACP starts Claude Code."""
    trial = _claude_trial(tmp_path, "acp")
    await trial.connect()
    policy = [
        call
        for call in trial._env.exec.await_args_list
        if "managed-settings.json" in call.args[0]
    ]
    assert len(policy) == 1 and policy[0].kwargs["user"] == "root"
    trial._planes.connect_acp.assert_awaited_once()
