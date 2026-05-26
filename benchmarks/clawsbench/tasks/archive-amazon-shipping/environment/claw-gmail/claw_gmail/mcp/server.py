"""MCP server setup — exposes all FastAPI routes as MCP tools."""

from __future__ import annotations


def mount_mcp(app):
    """Mount MCP endpoint on the FastAPI app.

    Requires `fastapi-mcp` to be installed.
    """
    try:
        from fastapi_mcp import FastApiMCP

        mcp = FastApiMCP(
            app,
            name="claw-gmail",
            description="Mock Gmail API — Gmail-compatible REST API for AI agents",
        )
        mcp.mount_http()  # -> /mcp endpoint
        return mcp
    except ImportError:
        import warnings
        warnings.warn(
            "fastapi-mcp not installed. MCP endpoint will not be available. "
            "Install with: pip install fastapi-mcp"
        )
        return None
