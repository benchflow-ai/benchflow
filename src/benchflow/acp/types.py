"""ACP protocol types — JSON-RPC 2.0 messages for Agent Client Protocol."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# --- Enums ---


class StopReason(str, Enum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    MAX_TURN_REQUESTS = "max_turn_requests"
    REFUSAL = "refusal"
    CANCELLED = "cancelled"


class ToolKind(str, Enum):
    OTHER = "other"
    BASH = "bash"
    SEARCH = "search"
    BROWSER = "browser"
    READ = "read"
    WRITE = "write"


class ToolCallStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# --- Content blocks ---


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    data: str
    mime_type: str = Field(alias="mimeType", default="image/png")


class ResourceLink(BaseModel):
    type: Literal["resource_link"] = "resource_link"
    uri: str
    title: str | None = None
    mime_type: str | None = Field(default=None, alias="mimeType")


ContentBlock = TextContent | ImageContent | ResourceLink


# --- Capabilities ---


class FsCapabilities(BaseModel):
    read_text_file: bool = Field(default=True, alias="readTextFile")
    write_text_file: bool = Field(default=True, alias="writeTextFile")


class ClientCapabilities(BaseModel):
    fs: FsCapabilities = Field(default_factory=FsCapabilities)
    terminal: bool = True


class PromptCapabilities(BaseModel):
    image: bool = False
    audio: bool = False
    embedded_context: bool = Field(default=False, alias="embeddedContext")


class McpCapabilities(BaseModel):
    sse: bool = False
    http: bool = False


class AgentCapabilities(BaseModel):
    prompt_capabilities: PromptCapabilities | None = Field(
        default=None, alias="promptCapabilities"
    )
    mcp_capabilities: McpCapabilities | None = Field(
        default=None, alias="mcpCapabilities"
    )
    load_session: bool = Field(default=False, alias="loadSession")


class ClientInfo(BaseModel):
    name: str = "benchflow"
    version: str = "2.0.0"


class AgentInfo(BaseModel):
    name: str
    version: str


# --- Requests / Responses ---


class InitializeParams(BaseModel):
    protocol_version: int = Field(default=0, alias="protocolVersion")
    client_capabilities: ClientCapabilities = Field(
        default_factory=ClientCapabilities, alias="clientCapabilities"
    )
    client_info: ClientInfo = Field(default_factory=ClientInfo, alias="clientInfo")


class InitializeResult(BaseModel):
    protocol_version: int = Field(alias="protocolVersion")
    agent_capabilities: AgentCapabilities | None = Field(
        default=None, alias="agentCapabilities"
    )
    agent_info: AgentInfo | None = Field(default=None, alias="agentInfo")


class McpServerSpec(BaseModel):
    type: str = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: list[dict[str, str]] = Field(default_factory=list)
    url: str | None = None


class NewSessionParams(BaseModel):
    cwd: str = "/app"
    mcp_servers: list[McpServerSpec] = Field(default_factory=list, alias="mcpServers")


class NewSessionResult(BaseModel):
    session_id: str = Field(alias="sessionId")


class PromptParams(BaseModel):
    session_id: str = Field(alias="sessionId")
    prompt: list[dict[str, Any]]


class PromptResult(BaseModel):
    stop_reason: StopReason = Field(alias="stopReason")


class CancelParams(BaseModel):
    session_id: str = Field(alias="sessionId")


# --- Session update notifications ---


class ToolCallUpdate(BaseModel):
    session_update: Literal["tool_call"] = Field(alias="sessionUpdate")
    tool_call_id: str = Field(alias="toolCallId")
    title: str = ""
    kind: ToolKind = ToolKind.OTHER
    status: ToolCallStatus = ToolCallStatus.PENDING


class ToolCallStatusUpdate(BaseModel):
    session_update: Literal["tool_call_update"] = Field(alias="sessionUpdate")
    tool_call_id: str = Field(alias="toolCallId")
    status: ToolCallStatus
    content: list[dict[str, Any]] = Field(default_factory=list)


class AgentMessageChunk(BaseModel):
    session_update: Literal["agent_message_chunk"] = Field(alias="sessionUpdate")
    content: dict[str, Any]


class AgentThoughtChunk(BaseModel):
    session_update: Literal["agent_thought_chunk"] = Field(alias="sessionUpdate")
    content: dict[str, Any]


SessionUpdate = (
    ToolCallUpdate | ToolCallStatusUpdate | AgentMessageChunk | AgentThoughtChunk
)


# --- JSON-RPC envelope ---


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class JsonRpcNotification(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


# --- File system / Terminal requests from agent ---


class ReadFileParams(BaseModel):
    session_id: str = Field(alias="sessionId")
    path: str
    line: int | None = None
    limit: int | None = None


class WriteFileParams(BaseModel):
    session_id: str = Field(alias="sessionId")
    path: str
    contents: str


class CreateTerminalParams(BaseModel):
    session_id: str = Field(alias="sessionId")
    command: str = "bash"
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: list[dict[str, str]] = Field(default_factory=list)


class TerminalOutputParams(BaseModel):
    session_id: str = Field(alias="sessionId")
    terminal_id: str = Field(alias="terminalId")


class WaitForExitParams(BaseModel):
    session_id: str = Field(alias="sessionId")
    terminal_id: str = Field(alias="terminalId")


class PermissionRequestParams(BaseModel):
    session_id: str = Field(alias="sessionId")
    tool_call_id: str = Field(alias="toolCallId")
    title: str
    description: str = ""
    options: list[dict[str, Any]] = Field(default_factory=list)
