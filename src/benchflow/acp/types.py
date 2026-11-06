"""ACP protocol types — SDK-backed schema for the Agent Client Protocol.

The protocol schema (request/response messages, content blocks, capabilities)
is sourced from the official ``agent-client-protocol`` SDK (importable as the
top-level ``acp`` package — no collision with ``benchflow.acp``). BenchFlow's
historic public names are kept as compatibility aliases so consumers in
``acp/client.py`` and ``acp/session.py`` change minimally.

A few types stay vendored on purpose:

* ``StopReason`` / ``ToolKind`` / ``ToolCallStatus`` — the SDK exposes these as
  ``typing.Literal`` aliases, but BenchFlow uses them as ``StrEnum`` (member
  access like ``ToolCallStatus.PENDING``, the ``.value`` attribute, and the
  callable ``ToolCallStatus(...)`` constructor). The SDK ``ToolCallStatus``
  Literal also has no ``cancelled`` member, which ``session.py`` requires.
* The ``session/update`` notification union (``ToolCallUpdate`` etc.) — the SDK
  models it as ``ToolCallStart`` / ``ToolCallProgress``, structurally different
  from BenchFlow's ``ToolCallUpdate`` / ``ToolCallStatusUpdate``. ``session.py``
  parses raw ``session/update`` dicts directly, so these vendored models are a
  thin documentation layer rather than a parse path.
* The JSON-RPC envelope and the fs/terminal/permission request params — these
  are BenchFlow framework transport types, not protocol schema.
"""

import contextlib
import importlib
import sys
import threading
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

# The SDK-backed names (the aliases below, the capabilities and
# ``ACP_PROTOCOL_VERSION``) resolve on first use (PEP 562, ``__getattr__`` at
# the bottom): the SDK's schema takes about 0.2 s to import, and the vendored
# names here (``StopReason``, ``McpServerSpec``) are imported by modules every
# run loads, ACP agent or not. Type checkers read the same names from the
# block below.
if TYPE_CHECKING:
    from acp.schema import (
        AgentCapabilities,
        AuthCapabilities,
        ClientCapabilities,
        FileSystemCapabilities,
        ImageContentBlock,
        Implementation,
        InitializeRequest,
        InitializeResponse,
        McpCapabilities,
        NewSessionRequest,
        NewSessionResponse,
        PromptCapabilities,
        PromptRequest,
        PromptResponse,
        ResourceContentBlock,
        TextContentBlock,
    )

    # The ACP protocol version this client implements. Sourced from the SDK so
    # it tracks upstream; v1 is current. ``client.py`` imports this name.
    ACP_PROTOCOL_VERSION: int


# Enums (vendored — SDK exposes these as typing.Literal, not enums)


class StopReason(StrEnum):
    """Why the agent stopped generating after a prompt.

    Vendored as a StrEnum: BenchFlow code compares against members
    (``StopReason.END_TURN``). As a StrEnum, members also compare equal to the
    plain strings the SDK ``PromptResponse.stop_reason`` carries.
    """

    END_TURN = "end_turn"  # Agent finished normally
    MAX_TOKENS = "max_tokens"  # Hit output token limit
    MAX_TURN_REQUESTS = "max_turn_requests"  # Hit tool-call-per-turn cap
    REFUSAL = "refusal"  # Agent refused the prompt
    CANCELLED = "cancelled"  # Client cancelled the request


class ToolKind(StrEnum):
    """Category tag for tool calls, used for metrics and trajectory display."""

    OTHER = "other"
    BASH = "bash"
    SEARCH = "search"
    BROWSER = "browser"
    READ = "read"
    WRITE = "write"
    SKILL = "skill"


class ToolCallStatus(StrEnum):
    """Lifecycle state of a tool call within a session.

    Vendored: the SDK ``ToolCallStatus`` Literal lacks ``cancelled``, and
    ``session.py`` constructs/compares this as an enum with ``.value`` access.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Content blocks (SDK-backed)
#
# BenchFlow's content-block names map to the SDK's discriminated ``*ContentBlock``
# variants (which carry the ``type`` discriminator field).

if TYPE_CHECKING:
    TextContent = TextContentBlock
    ImageContent = ImageContentBlock
    ResourceLink = ResourceContentBlock

    ContentBlock = TextContentBlock | ImageContentBlock | ResourceContentBlock


# Capabilities & identity (SDK-backed)

if TYPE_CHECKING:
    FsCapabilities = FileSystemCapabilities
    ClientInfo = Implementation
    AgentInfo = Implementation

# ``ClientCapabilities``, ``AgentCapabilities``, ``PromptCapabilities`` and
# ``McpCapabilities`` are re-exported directly from the SDK under their
# original names (imported at module top, listed in ``__all__``).


# Requests / Responses (SDK-backed)
#
# BenchFlow historically named these ``*Params`` / ``*Result``; the SDK names
# them ``*Request`` / ``*Response``. Keep both working.

if TYPE_CHECKING:
    InitializeParams = InitializeRequest
    InitializeResult = InitializeResponse
    NewSessionParams = NewSessionRequest
    NewSessionResult = NewSessionResponse
    PromptParams = PromptRequest
    PromptResult = PromptResponse

# Each SDK-backed name and the ``acp.schema`` class it is.
_SDK_NAMES = {
    "AgentCapabilities": "AgentCapabilities",
    "AuthCapabilities": "AuthCapabilities",
    "ClientCapabilities": "ClientCapabilities",
    "FileSystemCapabilities": "FileSystemCapabilities",
    "ImageContentBlock": "ImageContentBlock",
    "Implementation": "Implementation",
    "InitializeRequest": "InitializeRequest",
    "InitializeResponse": "InitializeResponse",
    "McpCapabilities": "McpCapabilities",
    "NewSessionRequest": "NewSessionRequest",
    "NewSessionResponse": "NewSessionResponse",
    "PromptCapabilities": "PromptCapabilities",
    "PromptRequest": "PromptRequest",
    "PromptResponse": "PromptResponse",
    "ResourceContentBlock": "ResourceContentBlock",
    "TextContentBlock": "TextContentBlock",
    "TextContent": "TextContentBlock",
    "ImageContent": "ImageContentBlock",
    "ResourceLink": "ResourceContentBlock",
    "FsCapabilities": "FileSystemCapabilities",
    "ClientInfo": "Implementation",
    "AgentInfo": "Implementation",
    "InitializeParams": "InitializeRequest",
    "InitializeResult": "InitializeResponse",
    "NewSessionParams": "NewSessionRequest",
    "NewSessionResult": "NewSessionResponse",
    "PromptParams": "PromptRequest",
    "PromptResult": "PromptResponse",
}


class McpServerSpec(BaseModel):
    """MCP server to attach to a session (stdio or SSE/HTTP).

    Vendored: BenchFlow uses a single flat shape across stdio/SSE/HTTP, while
    the SDK splits these into separate ``McpServerStdio`` / ``SseMcpServer`` /
    ``HttpMcpServer`` models. :meth:`to_new_session_param` projects this flat
    shape onto the exact per-transport dict ``session/new`` expects: every
    variant carries ``name``; stdio adds ``command``/``args``/``env`` and omits
    the ``type`` discriminator (the SDK's ``McpServerStdio`` has none); sse/http
    add ``url``/``headers`` and keep ``type``. BenchFlow stores task-authored
    env/header maps as dictionaries for ergonomics, then projects them to ACP's
    list-of-name/value wire shape. Benchmark-specific MCP tool filters are kept
    under ACP ``_meta`` so they do not violate the standard server schema.
    """

    name: str  # required on every ACP variant; the rest are transport-dependent
    type: str = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    tools: list[str] | None = None
    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None

    @staticmethod
    def _mapping_to_name_value(mapping: dict[str, str]) -> list[dict[str, str]]:
        return [{"name": name, "value": value} for name, value in mapping.items()]

    def _meta_param(self) -> dict[str, Any]:
        filters: dict[str, list[str]] = {}
        if self.tools is not None:
            filters["tools"] = list(self.tools)
        if self.include_tags is not None:
            filters["include_tags"] = list(self.include_tags)
        if self.exclude_tags is not None:
            filters["exclude_tags"] = list(self.exclude_tags)
        if not filters:
            return {}
        return {"_meta": {"benchflow": {"mcp_tool_filters": filters}}}

    def to_new_session_param(self) -> dict[str, Any]:
        """Project onto the per-transport ``session/new`` server dict.

        stdio servers carry ``command``/``args``/``env`` and no ``type``; sse and
        http servers carry ``url``/``headers`` and the ``type`` discriminator.
        Header and env maps are projected to ACP's required list shape, and
        non-standard BenchFlow filter hints are carried under ``_meta``.
        """
        meta = self._meta_param()
        if self.type == "stdio":
            payload = {
                "name": self.name,
                "command": self.command,
                "args": list(self.args),
                "env": self._mapping_to_name_value(self.env),
                **meta,
            }
            if self.cwd is not None:
                payload["cwd"] = self.cwd
            return payload
        return {
            "type": self.type,
            "name": self.name,
            "url": self.url,
            "headers": self._mapping_to_name_value(self.headers),
            **meta,
        }


class CancelParams(BaseModel):
    """Parameters for session/cancel — aborts the current prompt execution."""

    session_id: str = Field(alias="sessionId")


# Session update notifications (vendored)
#
# Kept vendored: the SDK models session updates as ``ToolCallStart`` /
# ``ToolCallProgress``, structurally different from BenchFlow's pair below.
# ``session.py`` parses raw ``session/update`` dicts, so these are a
# documentation layer rather than a live parse path.


class ToolCallUpdate(BaseModel):
    """Notification: agent started a new tool call."""

    session_update: Literal["tool_call"] = Field(alias="sessionUpdate")
    tool_call_id: str = Field(alias="toolCallId")
    title: str = ""
    kind: ToolKind = ToolKind.OTHER
    status: ToolCallStatus = ToolCallStatus.PENDING


class ToolCallStatusUpdate(BaseModel):
    """Notification: existing tool call changed status or produced output."""

    session_update: Literal["tool_call_update"] = Field(alias="sessionUpdate")
    tool_call_id: str = Field(alias="toolCallId")
    status: ToolCallStatus
    content: list[dict[str, Any]] = Field(default_factory=list)


class AgentMessageChunk(BaseModel):
    """Notification: streaming chunk of agent visible-to-user text."""

    session_update: Literal["agent_message_chunk"] = Field(alias="sessionUpdate")
    content: dict[str, Any]


class AgentThoughtChunk(BaseModel):
    """Notification: streaming chunk of agent internal reasoning."""

    session_update: Literal["agent_thought_chunk"] = Field(alias="sessionUpdate")
    content: dict[str, Any]


SessionUpdate = (
    ToolCallUpdate | ToolCallStatusUpdate | AgentMessageChunk | AgentThoughtChunk
)


# JSON-RPC envelope (vendored — BenchFlow transport framing)


class JsonRpcRequest(BaseModel):
    """Outbound JSON-RPC 2.0 request (has ``id``, expects a response)."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcResponse(BaseModel):
    """Inbound JSON-RPC 2.0 response — exactly one of result/error is set."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class JsonRpcNotification(BaseModel):
    """JSON-RPC 2.0 notification (no ``id``, no response expected)."""

    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


# File system / Terminal requests from agent (vendored)
#
# Kept vendored: these mirror requests the agent makes to BenchFlow's
# framework transport, which proxies them to the sandbox. The SDK's
# ``ReadTextFileRequest`` etc. exist but BenchFlow's auto-approve client
# handles these as raw dicts.


class ReadFileParams(BaseModel):
    """Agent → client request: read a file from the sandbox filesystem."""

    session_id: str = Field(alias="sessionId")
    path: str
    line: int | None = None
    limit: int | None = None


class WriteFileParams(BaseModel):
    """Agent → client request: write a file to the sandbox filesystem."""

    session_id: str = Field(alias="sessionId")
    path: str
    contents: str


class CreateTerminalParams(BaseModel):
    """Agent → client request: spawn a terminal process in the sandbox."""

    session_id: str = Field(alias="sessionId")
    command: str = "bash"
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: list[dict[str, str]] = Field(default_factory=list)


class TerminalOutputParams(BaseModel):
    """Agent → client request: read stdout/stderr from a running terminal."""

    session_id: str = Field(alias="sessionId")
    terminal_id: str = Field(alias="terminalId")


class WaitForExitParams(BaseModel):
    """Agent → client request: block until a terminal process exits."""

    session_id: str = Field(alias="sessionId")
    terminal_id: str = Field(alias="terminalId")


class PermissionRequestParams(BaseModel):
    """Agent → client request: ask the user to approve a sensitive action."""

    session_id: str = Field(alias="sessionId")
    tool_call_id: str = Field(alias="toolCallId")
    title: str
    description: str = ""
    options: list[dict[str, Any]] = Field(default_factory=list)


__all__ = [
    "ACP_PROTOCOL_VERSION",
    "AgentCapabilities",
    "AgentInfo",
    "AgentMessageChunk",
    "AgentThoughtChunk",
    "AuthCapabilities",
    "CancelParams",
    "ClientCapabilities",
    "ClientInfo",
    "ContentBlock",
    "CreateTerminalParams",
    "FsCapabilities",
    "ImageContent",
    "InitializeParams",
    "InitializeResult",
    "JsonRpcNotification",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "McpCapabilities",
    "McpServerSpec",
    "NewSessionParams",
    "NewSessionResult",
    "PermissionRequestParams",
    "PromptCapabilities",
    "PromptParams",
    "PromptResult",
    "ReadFileParams",
    "ResourceLink",
    "SessionUpdate",
    "StopReason",
    "TerminalOutputParams",
    "TextContent",
    "ToolCallStatus",
    "ToolCallStatusUpdate",
    "ToolCallUpdate",
    "ToolKind",
    "WaitForExitParams",
    "WriteFileParams",
]


_sdk_import_started = False


def prewarm_sdk() -> None:
    """Import the ACP SDK's schema in a thread of this process, once.

    The names above load it on first use, when the agent connects: about
    0.25 s of the event loop on a build VM, right before the agent starts.
    Started once the sandbox is up (Rollout._prewarm_acp), the import overlaps
    the sandbox's setup; the later import waits on the module lock for what is
    left and gets the same module. The SDK imports nothing of BenchFlow's
    (SPEED-FIX)."""
    global _sdk_import_started
    if _sdk_import_started or "acp.schema" in sys.modules:
        return
    _sdk_import_started = True

    def _import() -> None:
        with contextlib.suppress(Exception):
            importlib.import_module("acp.schema")
            importlib.import_module("acp.meta")

    threading.Thread(target=_import, name="acp-import", daemon=True).start()


def __getattr__(name: str) -> Any:
    if name == "ACP_PROTOCOL_VERSION":
        from acp import meta

        value: Any = meta.PROTOCOL_VERSION
    elif name == "ContentBlock":
        from acp.schema import (
            ImageContentBlock,
            ResourceContentBlock,
            TextContentBlock,
        )

        value = TextContentBlock | ImageContentBlock | ResourceContentBlock
    elif name in _SDK_NAMES:
        from acp import schema

        value = getattr(schema, _SDK_NAMES[name])
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
