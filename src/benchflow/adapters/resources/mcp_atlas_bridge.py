#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

UPSTREAM_URL = os.environ.get("MCP_ATLAS_UPSTREAM_URL", "http://127.0.0.1:1984").rstrip(
    "/"
)
ENABLED_TOOLS_FILE = os.environ.get("MCP_ENABLED_TOOLS_FILE", "/enabled_tools.txt")
HOST = os.environ.get("MCP_ATLAS_BRIDGE_HOST", "0.0.0.0")
PORT = int(os.environ.get("MCP_ATLAS_BRIDGE_PORT", "18765"))


def load_enabled_tools() -> set[str]:
    path = Path(ENABLED_TOOLS_FILE)
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


async def fetch_upstream_tools() -> list[dict[str, Any]]:
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(180):
            try:
                response = await client.post(f"{UPSTREAM_URL}/list-tools")
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return [item for item in payload if isinstance(item, dict)]
                raise RuntimeError(
                    f"unexpected /list-tools payload: {type(payload).__name__}"
                )
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(1)
    raise RuntimeError(f"MCP Atlas upstream never became ready: {last_error}")


def tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    schema = (
        tool.get("inputSchema")
        or tool.get("input_schema")
        or tool.get("parameters")
        or {"type": "object", "properties": {}}
    )
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    schema = dict(schema)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


def content_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = content_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        kind = value.get("type")
        if kind == "text" and isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("text"), str):
            return value["text"]
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def make_proxy(tool_name: str):
    async def proxy(**kwargs: Any) -> str:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                f"{UPSTREAM_URL}/call-tool",
                json={"tool_name": tool_name, "tool_args": kwargs},
            )
            response.raise_for_status()
            return content_to_text(response.json())

    proxy.__name__ = "call_" + "".join(ch if ch.isalnum() else "_" for ch in tool_name)
    return proxy


async def build_server() -> FastMCP:
    enabled_tools = load_enabled_tools()
    upstream_tools = await fetch_upstream_tools()
    server = FastMCP("mcp-atlas-bridge")
    registered = 0
    for tool in upstream_tools:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        if enabled_tools and name not in enabled_tools:
            continue
        server.add_tool(
            FunctionTool(
                name=name,
                description=str(tool.get("description") or ""),
                parameters=tool_schema(tool),
                fn=make_proxy(name),
                run_in_thread=False,
            )
        )
        registered += 1
    if registered == 0:
        raise RuntimeError(
            f"MCP Atlas bridge registered zero tools from {len(upstream_tools)} upstream tools"
        )
    print(f"MCP Atlas bridge registered {registered} tools on port {PORT}", flush=True)
    return server


if __name__ == "__main__":
    mcp = asyncio.run(build_server())
    mcp.run(transport="streamable-http", host=HOST, port=PORT, path="/mcp")
