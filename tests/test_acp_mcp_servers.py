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
from benchflow.rollout import (
    Role,
    Rollout,
    RolloutConfig,
    _openhands_mcp_config,
    _task_mcp_specs,
    _task_mcp_specs_for_agent,
)
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
        MCPServerConfig(
            name="pw",
            transport="stdio",
            command="npx",
            args=["-y", "x"],
            cwd="/workspace",
        ),
        MCPServerConfig(name="r", transport="sse", url="http://x/sse"),
        MCPServerConfig(name="h", transport="streamable-http", url="http://x/mcp"),
    )
    assert [spec.to_new_session_param() for spec in _task_mcp_specs(task)] == [
        {
            "name": "pw",
            "command": "npx",
            "args": ["-y", "x"],
            "env": [],
            "cwd": "/workspace",
        },
        {"type": "sse", "name": "r", "url": "http://x/sse", "headers": []},
        {"type": "http", "name": "h", "url": "http://x/mcp", "headers": []},
    ]


def test_task_mcp_specs_carries_openhands_tool_filters() -> None:
    """Guards MCP Atlas adapter work against invalid raw ACP tool filters."""
    task = _task_with_mcp(
        MCPServerConfig(
            name="atlas",
            transport="streamable-http",
            url="http://localhost:18765/mcp",
            headers={"x-run": "smoke"},
            tools=["search", "fetch"],
            include_tags=["safe"],
            exclude_tags=["admin"],
        )
    )

    assert [spec.to_new_session_param() for spec in _task_mcp_specs(task)] == [
        {
            "type": "http",
            "name": "atlas",
            "url": "http://localhost:18765/mcp",
            "headers": [{"name": "x-run", "value": "smoke"}],
            "_meta": {
                "benchflow": {
                    "mcp_tool_filters": {
                        "tools": ["search", "fetch"],
                        "include_tags": ["safe"],
                        "exclude_tags": ["admin"],
                    }
                }
            },
        }
    ]


def test_openhands_mcp_config_uses_fastmcp_shape() -> None:
    """Guards native OpenHands MCP config against ACP/FastMCP shape drift."""
    task = _task_with_mcp(
        MCPServerConfig(
            name="atlas",
            transport="streamable-http",
            url="http://localhost:18765/mcp",
            headers={"x-run": "smoke"},
            tools=["search", "fetch"],
        ),
        MCPServerConfig(
            name="local",
            transport="stdio",
            command="python",
            args=["server.py"],
            cwd="/workspace/agent_workspace",
            env={"TOKEN": "abc"},
            exclude_tags=["unsafe"],
        ),
    )

    assert _openhands_mcp_config(task) == {
        "mcpServers": {
            "atlas": {
                "url": "http://localhost:18765/mcp",
                "transport": "http",
                "headers": {"x-run": "smoke"},
                "enabled": True,
                "tools": ["search", "fetch"],
            },
            "local": {
                "command": "python",
                "args": ["server.py"],
                "cwd": "/workspace/agent_workspace",
                "env": {"TOKEN": "abc"},
                "transport": "stdio",
                "enabled": True,
                "exclude_tags": ["unsafe"],
            },
        }
    }


def test_openhands_mcp_servers_are_not_sent_over_acp() -> None:
    """OpenHands loads task MCP config from ~/.openhands/mcp.json, not session/new."""
    task = _task_with_mcp(
        MCPServerConfig(name="h", transport="streamable-http", url="http://x/mcp")
    )

    assert _task_mcp_specs_for_agent("openhands", task) == []
    assert len(_task_mcp_specs_for_agent("claude-agent-acp", task)) == 1


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
    # release/v0.6.0 (#613) replaced the per-provider proxies with one LiteLLM
    # runtime — connect() awaits ensure_litellm_runtime, returning
    # (agent_env, usage_runtime).
    planes.ensure_litellm_runtime = AsyncMock(
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


@pytest.mark.asyncio
async def test_connect_as_omits_openhands_mcp_servers_from_acp(tmp_path) -> None:
    """Guards the role execution path used by single-shot OpenHands scenes."""
    cfg = RolloutConfig(task_path=tmp_path / "task", agent="openhands", model="qwen")
    trial = Rollout.__new__(Rollout)
    trial._config = cfg
    trial._env = {}
    trial._rollout_dir = tmp_path
    trial._timing = {}
    trial._agent_cwd = "/app"
    trial._agent_cfg = MagicMock()
    trial._agent_launch = "openhands"
    trial._phase = "installed"
    trial._disallow_web_tools = False
    trial._task = _task_with_mcp(
        MCPServerConfig(name="atlas", transport="streamable-http", url="http://x/mcp")
    )
    trial._planes = _fake_planes()
    trial._planes.agent_launch.return_value = "openhands"
    trial._planes.resolve_agent_env.side_effect = lambda _agent, _model, env: env or {}
    trial._planes.install_agent = AsyncMock()
    trial._planes.write_credential_files = AsyncMock()
    trial._planes.upload_subscription_auth = AsyncMock()
    trial._planes.apply_web_tool_policy = AsyncMock()

    await trial.connect_as(Role(name="agent", agent="openhands", model="qwen"))

    assert trial._planes.connect_acp.await_args.kwargs["mcp_servers"] == []
