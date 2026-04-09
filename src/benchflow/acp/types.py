"""ACP protocol types — JSON-RPC 2.0 messages for Agent Client Protocol.

Defines the full request/response/notification schema for ACP. Types mirror
the wire format (camelCase aliases) and are used by ``acp/client.py`` to
construct and parse messages. Related: acp/session.py (consumes SessionUpdate).
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# --- Enums ---


class StopReason(str, Enum):
    """Why the agent stopped generating after a prompt."""

    END_TURN = "end_turn"  # Agent finished normally
    MAX_TOKENS = "max_tokens"  # Hit output token limit
    MAX_TURN_REQUESTS = "max_turn_requests"  # Hit tool-call-per-turn cap
    REFUSAL = "refusal"  # Agent refused the prompt
    CANCELLED = "cancelled"  # Client cancelled the request


class ToolKind(str, Enum):
    """Category tag for tool calls, used for metrics and trajectory display."""

    OTHER = "other"
    BASH = "bash"
    SEARCH = "search"
    BROWSER = "browser"
    READ = "read"
    WRITE = "write"


class ToolCallStatus(str, Enum):
    """Lifecycle state of a tool call within a session."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# --- Content blocks ---


class TextContent(BaseModel):
    """Plain text content block in an agent message."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Base64-encoded image content block."""

    type: Literal["image"] = "image"
    data: str
    mime_type: str = Field(alias="mimeType", default="image/png")


class ResourceLink(BaseModel):
    """Reference to an external resource (file, URL) attached to a message."""

    type: Literal["resource_link"] = "resource_link"
    uri: str
    title: str | None = None
    mime_type: str | None = Field(default=None, alias="mimeType")


ContentBlock = TextContent | ImageContent | ResourceLink


# --- Capabilities ---


class FsCapabilities(BaseModel):
    """File-system operations the client offers to the agent."""

    read_text_file: bool = Field(default=True, alias="readTextFile")
    write_text_file: bool = Field(default=True, alias="writeTextFile")


class ClientCapabilities(BaseModel):
    """Capabilities the benchflow client advertises during initialize."""

    fs: FsCapabilities = Field(default_factory=FsCapabilities)
    terminal: bool = True


class PromptCapabilities(BaseModel):
    """Media types the agent accepts in prompt content blocks."""

    image: bool = False
    audio: bool = False
    embedded_context: bool = Field(default=False, alias="embeddedContext")


class McpCapabilities(BaseModel):
    """MCP transport modes the agent supports."""

    sse: bool = False
    http: bool = False


class AgentCapabilities(BaseModel):
    """Capabilities the agent reports back during initialize."""

    prompt_capabilities: PromptCapabilities | None = Field(
        default=None, alias="promptCapabilities"
    )
    mcp_capabilities: McpCapabilities | None = Field(
        default=None, alias="mcpCapabilities"
    )
    load_session: bool = Field(default=False, alias="loadSession")


class ClientInfo(BaseModel):
    """Identity block sent by the client during initialize."""

    name: str = "benchflow"
    version: str = "2.0.0"


class AgentInfo(BaseModel):
    """Identity block returned by the agent during initialize."""

    name: str
    version: str


# --- Requests / Responses ---


class InitializeParams(BaseModel):
    """Client → agent: start the ACP handshake."""

    protocol_version: int = Field(default=0, alias="protocolVersion")
    client_capabilities: ClientCapabilities = Field(
        default_factory=ClientCapabilities, alias="clientCapabilities"
    )
    client_info: ClientInfo = Field(default_factory=ClientInfo, alias="clientInfo")


class InitializeResult(BaseModel):
    """Agent → client: handshake response with agent identity and capabilities."""

    protocol_version: int = Field(alias="protocolVersion")
    agent_capabilities: AgentCapabilities | None = Field(
        default=None, alias="agentCapabilities"
    )
    agent_info: AgentInfo | None = Field(default=None, alias="agentInfo")


class McpServerSpec(BaseModel):
    """MCP server to attach to a session (stdio or SSE/HTTP)."""

    type: str = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: list[dict[str, str]] = Field(default_factory=list)
    url: str | None = None


class NewSessionParams(BaseModel):
    """Parameters for session/new — creates a new agent workspace session."""

    cwd: str = "/app"
    mcp_servers: list[McpServerSpec] = Field(default_factory=list, alias="mcpServers")


class NewSessionResult(BaseModel):
    """Response from session/new — contains the session identifier."""

    session_id: str = Field(alias="sessionId")


class PromptParams(BaseModel):
    """Parameters for prompt/send — delivers user content to the agent."""

    session_id: str = Field(alias="sessionId")
    prompt: list[dict[str, Any]]


class PromptResult(BaseModel):
    """Response from prompt/send — indicates why the agent stopped."""

    stop_reason: StopReason = Field(alias="stopReason")


class CancelParams(BaseModel):
    """Parameters for prompt/cancel — aborts the current prompt execution."""

    session_id: str = Field(alias="sessionId")


# --- Session update notifications ---


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


# --- JSON-RPC envelope ---


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


# --- File system / Terminal requests from agent ---


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
