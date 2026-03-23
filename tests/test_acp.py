"""Tests for ACP client ↔ mock agent — Step 10."""

import sys
from pathlib import Path

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.transport import StdioTransport
from benchflow.acp.types import StopReason

MOCK_AGENT = str(Path(__file__).parent / "fixtures" / "mock_acp_agent.py")


class TestACPClient:
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            result = await client.initialize()
            assert result.protocol_version == 0
            assert result.agent_info is not None
            assert result.agent_info.name == "mock-agent"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_session_new(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            await client.initialize()
            session = await client.session_new()
            assert session.session_id == "mock-session-1"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_prompt_and_response(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            await client.initialize()
            await client.session_new()

            result = await client.prompt("Hello, agent!")
            assert result.stop_reason == StopReason.END_TURN

            session = client.session
            assert session is not None
            assert "I received: Hello, agent!" in session.full_message
            assert len(session.tool_calls) == 1
            assert session.tool_calls[0].kind == "bash"
            assert session.tool_calls[0].title == "echo hello"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_from_config_stdio(self) -> None:
        client = ACPClient.from_config(
            command=f"{sys.executable} {MOCK_AGENT}",
            transport_type="stdio",
        )
        try:
            await client.connect()
            result = await client.initialize()
            assert result.agent_info.name == "mock-agent"
        finally:
            await client.close()

    def test_from_config_missing_command(self) -> None:
        with pytest.raises(ValueError, match="command required"):
            ACPClient.from_config(transport_type="stdio")

    def test_from_config_missing_url(self) -> None:
        with pytest.raises(ValueError, match="url required"):
            ACPClient.from_config(transport_type="sse")
