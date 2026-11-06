"""ACP (Agent Client Protocol) — JSON-RPC 2.0 interface to sandbox agents.

Provides the client that drives agent sessions (connect → initialize →
prompt loop → close), the session state tracker, and stdio/container
transports. See acp/client.py for the main entry point.

The names below resolve on first use (PEP 562): importing a submodule such as
``benchflow.acp.types`` for ``StopReason`` runs this file first, and the
client and session pull in the ACP SDK's schema (about 0.2 s), which a run
that starts no ACP agent (oracle, nop) never needs.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import ACPClient, ACPError
    from .session import ACPSession
    from .transport import StdioTransport, Transport
    from .types import ContentBlock, StopReason, ToolKind

_LAZY = {
    "ACPClient": ".client",
    "ACPError": ".client",
    "ACPSession": ".session",
    "StdioTransport": ".transport",
    "Transport": ".transport",
    "ContentBlock": ".types",
    "StopReason": ".types",
    "ToolKind": ".types",
}

__all__ = [
    "ACPClient",
    "ACPError",
    "ACPSession",
    "ContentBlock",
    "StdioTransport",
    "StopReason",
    "ToolKind",
    "Transport",
]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value
