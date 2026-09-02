"""ACP (Agent Client Protocol) — JSON-RPC 2.0 interface to sandbox agents.

Provides the client that drives agent sessions (connect → initialize →
prompt loop → close), the session state tracker, and stdio/container
transports. See acp/client.py for the main entry point.
"""

from .client import ACPClient, ACPError
from .session import ACPSession
from .termination import AcpSessionObservation, AgentTerminationReceipt
from .transport import StdioTransport, Transport
from .types import ContentBlock, StopReason, ToolKind

__all__ = [
    "AcpSessionObservation",
    "ACPClient",
    "ACPError",
    "AgentTerminationReceipt",
    "ACPSession",
    "ContentBlock",
    "StdioTransport",
    "StopReason",
    "ToolKind",
    "Transport",
]
