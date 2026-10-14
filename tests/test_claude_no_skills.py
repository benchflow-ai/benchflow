"""Guards PR516's explicit native Claude no-skills session configuration."""

from unittest.mock import AsyncMock

import pytest

from benchflow.acp.client import ACPClient
from tests.test_acp_mcp_servers import _RecordingTransport
from tests.test_acp_model_config_dispatch import _connect, _make_mocks

METADATA = {"claudeCode": {"options": {"extraArgs": {"disable-slash-commands": None}}}}


@pytest.mark.asyncio
async def test_client_preserves_supported_native_session_metadata():
    """PR516: ACP metadata reaches the official adapter without CLI patching."""
    client = ACPClient(_RecordingTransport())
    client._send_request = AsyncMock(return_value={"sessionId": "s1"})
    await client.session_new(cwd="/root/project", metadata=METADATA)
    client._send_request.assert_awaited_once_with(
        "session/new",
        {
            "cwd": "/root/project",
            "mcpServers": [],
            "_meta": METADATA,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent,enabled", [("claude-agent-acp", True), ("codex-acp", False)]
)
async def test_only_explicit_claude_flag_disables_skills(tmp_path, agent, enabled):
    """PR516: do not alter other harnesses or their native tool/safety surface."""
    mocked = _make_mocks()
    await _connect(
        mocked,
        agent=agent,
        model=None,
        tmp_path=tmp_path,
        agent_env={"BENCHFLOW_CLAUDE_DISABLE_SKILLS": "1"},
    )
    assert (mocked.session_new.call_args.kwargs.get("metadata") == METADATA) is enabled


@pytest.mark.asyncio
async def test_default_claude_session_is_unchanged(tmp_path):
    """PR516: existing skill-enabled or unspecified sessions keep defaults."""
    mocked = _make_mocks()
    await _connect(mocked, agent="claude-agent-acp", model=None, tmp_path=tmp_path)
    assert "metadata" not in mocked.session_new.call_args.kwargs
