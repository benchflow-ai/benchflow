"""Guards #982: request summarized thinking without changing budgets."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.runtime import connect_acp
from benchflow.acp.session import ACPSession


@pytest.mark.parametrize("agent", ["claude-agent-acp", "codex-acp"])
async def test_runtime_requests_thinking_display_for_claude_only(agent, tmp_path):
    acp = AsyncMock(spec=ACPClient)
    acp.initialize.return_value = SimpleNamespace(agent_info=None)
    acp.session_new.return_value = ACPSession("s")
    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=acp) as factory,
    ):
        await connect_acp(
            env=AsyncMock(),
            agent=agent,
            agent_launch=agent,
            agent_env={"MAX_THINKING_TOKENS": "8192"},
            sandbox_user=None,
            model=None,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )
    meta = factory.call_args.kwargs.get("session_meta")
    if agent == "claude-agent-acp":
        assert meta == {
            "claudeCode": {"options": {"extraArgs": {"thinking-display": "summarized"}}}
        }
    else:
        assert not meta


@pytest.mark.parametrize("load", [False, True])
async def test_session_metadata_reaches_new_and_resumed_sessions(load):
    meta = {
        "claudeCode": {"options": {"extraArgs": {"thinking-display": "summarized"}}}
    }
    client = ACPClient(AsyncMock(), session_meta=meta)
    with patch.object(
        client, "_send_request", AsyncMock(return_value={"sessionId": "s"})
    ) as request:
        if load:
            await client.session_load("s")
        else:
            await client.session_new()
    assert request.call_args.args[1]["_meta"] == meta


async def test_print_mode_asks_for_the_display_the_acp_session_asks_for(tmp_path):
    """Guards the wire parity an earlier change broke: the ACP session asked Claude Code
    for summarized thinking while print mode kept its own default
    (``updates``, with its beta header), so the native harness sent the
    model a different request and recorded different thinking."""
    from benchflow.native_harness.claude_code import claude_code_launch
    from benchflow.native_harness.spec import NativeTurn

    acp = AsyncMock(spec=ACPClient)
    acp.initialize.return_value = SimpleNamespace(agent_info=None)
    acp.session_new.return_value = ACPSession("s")
    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=acp) as factory,
    ):
        await connect_acp(
            env=AsyncMock(),
            agent="claude-agent-acp",
            agent_launch="claude-agent-acp",
            agent_env={},
            sandbox_user=None,
            model=None,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )
    extra = factory.call_args.kwargs["session_meta"]["claudeCode"]["options"][
        "extraArgs"
    ]
    for turn in (
        NativeTurn(cwd="/app", new_session_id="u-1"),
        NativeTurn(cwd="/app", resume_id="u-1"),
    ):
        argv = claude_code_launch(turn).argv
        flag = argv.index("--thinking-display")
        assert argv[flag + 1] == extra["thinking-display"] == "summarized"
