"""Task-configured MCP servers must reach the agent via ACP ``session/new``.

Before this wiring, ``task.toml`` ``[[environment.mcp_servers]]`` was parsed
into ``SandboxConfig.mcp_servers`` but never plumbed anywhere: both ACP session
entry points hardcoded an empty MCP list, and the spec type ``McpServerSpec``
was never constructed. These tests pin the full path — config → spec → wire —
plus the rollout-layer translation (``rollout._task_mcp_specs``) that feeds it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.transport import Transport
from benchflow.acp.types import McpServerSpec
from benchflow.rollout import Rollout, RolloutConfig, _task_mcp_specs
from benchflow.task.config import MCPServerConfig


class _RecordingTransport(Transport):
    """Captures outbound messages without doing real I/O."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def start(self) -> None:  # pragma: no cover - trivial
        pass

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def receive(self) -> dict[str, Any]:  # pragma: no cover - unused
        raise RuntimeError("not used in this test")

    async def close(self) -> None:  # pragma: no cover - trivial
        pass


# ── McpServerSpec → per-transport session/new wire dict ──


def test_stdio_spec_omits_type_and_carries_command() -> None:
    """stdio servers map to the SDK's ``McpServerStdio`` shape (no ``type``)."""
    spec = McpServerSpec(
        name="playwright", type="stdio", command="npx", args=["-y", "x"]
    )
    assert spec.to_new_session_param() == {
        "name": "playwright",
        "command": "npx",
        "args": ["-y", "x"],
        "env": [],
    }


def test_sse_spec_carries_url_and_type() -> None:
    spec = McpServerSpec(name="research", type="sse", url="http://localhost:8931/sse")
    assert spec.to_new_session_param() == {
        "type": "sse",
        "name": "research",
        "url": "http://localhost:8931/sse",
        "headers": [],
    }


def test_http_spec_carries_url_and_type() -> None:
    spec = McpServerSpec(name="api", type="http", url="http://localhost:9000/mcp")
    assert spec.to_new_session_param() == {
        "type": "http",
        "name": "api",
        "url": "http://localhost:9000/mcp",
        "headers": [],
    }


# ── rollout._task_mcp_specs: task config → ACP specs (incl. transport naming) ──


def _task_with_mcp(*configs: MCPServerConfig) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(environment=SimpleNamespace(mcp_servers=list(configs)))
    )


def test_task_mcp_specs_maps_each_transport() -> None:
    """streamable-http → http; stdio/sse pass through; fields carried across."""
    task = _task_with_mcp(
        MCPServerConfig(name="pw", transport="stdio", command="npx", args=["-y", "x"]),
        MCPServerConfig(name="r", transport="sse", url="http://x/sse"),
        MCPServerConfig(name="h", transport="streamable-http", url="http://x/mcp"),
    )
    assert [spec.to_new_session_param() for spec in _task_mcp_specs(task)] == [
        {"name": "pw", "command": "npx", "args": ["-y", "x"], "env": []},
        {"type": "sse", "name": "r", "url": "http://x/sse", "headers": []},
        {"type": "http", "name": "h", "url": "http://x/mcp", "headers": []},
    ]


def test_task_mcp_specs_handles_absent_config() -> None:
    """No task, or a task with no MCP servers, yields an empty spec list."""
    assert _task_mcp_specs(None) == []
    assert (
        _task_mcp_specs(
            SimpleNamespace(config=SimpleNamespace(environment=SimpleNamespace()))
        )
        == []
    )


# ── ACPClient.session_new / session_load wire output ──


async def _drive_session_new(
    mcp_servers: list[McpServerSpec] | None,
) -> dict[str, Any]:
    """Run ``session_new`` against a recording transport; return the wire msg."""
    transport = _RecordingTransport()
    client = ACPClient(transport)

    async def _fake_read(_request_id: int) -> dict[str, Any]:
        return {"sessionId": "s-1"}

    client._read_until_response = _fake_read  # type: ignore[method-assign]
    await client.session_new(cwd="/app", mcp_servers=mcp_servers)
    (sent,) = [m for m in transport.sent if m.get("method") == "session/new"]
    return sent


@pytest.mark.asyncio
async def test_session_new_attaches_configured_mcp_servers() -> None:
    """A configured stdio + SSE server must land on the ``session/new`` wire."""
    specs = [
        McpServerSpec(name="playwright", type="stdio", command="npx", args=["-y", "x"]),
        McpServerSpec(name="research", type="sse", url="http://localhost:8931/sse"),
    ]
    sent = await _drive_session_new(specs)
    assert sent["params"]["mcpServers"] == [
        {"name": "playwright", "command": "npx", "args": ["-y", "x"], "env": []},
        {
            "type": "sse",
            "name": "research",
            "url": "http://localhost:8931/sse",
            "headers": [],
        },
    ]


@pytest.mark.asyncio
async def test_session_new_defaults_to_no_mcp_servers() -> None:
    """No configured servers → empty list, preserving the historical default."""
    sent = await _drive_session_new(None)
    assert sent["params"]["mcpServers"] == []


@pytest.mark.asyncio
async def test_session_load_attaches_configured_mcp_servers() -> None:
    """``session/load`` mirrors ``session/new``: the same servers are attached."""
    transport = _RecordingTransport()
    client = ACPClient(transport)

    async def _fake_read(_request_id: int) -> dict[str, Any]:
        return {"sessionId": "loaded-1"}

    client._read_until_response = _fake_read  # type: ignore[method-assign]
    spec = McpServerSpec(
        name="playwright", type="stdio", command="npx", args=["-y", "x"]
    )
    await client.session_load("loaded-1", cwd="/app", mcp_servers=[spec])
    (sent,) = [m for m in transport.sent if m.get("method") == "session/load"]
    assert sent["params"]["mcpServers"] == [
        {"name": "playwright", "command": "npx", "args": ["-y", "x"], "env": []}
    ]


# ── Rollout-layer threading (task config → connect_acp) ──


def _fake_planes() -> MagicMock:
    planes = MagicMock()
    planes.ensure_bedrock_proxy_runtime = AsyncMock(
        side_effect=lambda **kwargs: (kwargs["agent_env"], None)
    )
    planes.ensure_usage_proxy_runtime = AsyncMock(
        side_effect=lambda **kwargs: (kwargs["agent_env"], None)
    )
    planes.connect_acp = AsyncMock(
        return_value=(MagicMock(), MagicMock(), MagicMock(), "agent")
    )
    return planes


@pytest.mark.asyncio
async def test_connect_threads_task_mcp_servers_into_connect_acp(tmp_path) -> None:
    """``Rollout.connect()`` passes the task's mapped MCP specs to connect_acp."""
    cfg = RolloutConfig(
        task_path=tmp_path / "task", agent="claude-agent-acp", model="test-model"
    )
    trial = Rollout.__new__(Rollout)
    trial._config = cfg
    trial._env = {}
    trial._rollout_dir = tmp_path
    trial._timing = {}
    trial._agent_cwd = "/app"
    trial._agent_env = {}
    trial._agent_launch = "claude-agent-acp"
    trial._phase = "installed"
    trial._task = _task_with_mcp(
        MCPServerConfig(
            name="playwright",
            transport="stdio",
            command="npx",
            args=["-y", "@playwright/mcp@latest"],
        )
    )
    trial._planes = _fake_planes()

    await trial.connect()

    specs = trial._planes.connect_acp.await_args.kwargs["mcp_servers"]
    assert [s.to_new_session_param() for s in specs] == [
        {
            "name": "playwright",
            "command": "npx",
            "args": ["-y", "@playwright/mcp@latest"],
            "env": [],
        }
    ]
