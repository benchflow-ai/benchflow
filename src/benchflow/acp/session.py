"""ACP session lifecycle management."""

import logging
from datetime import datetime

from .types import (
    AgentCapabilities,
    AgentInfo,
    StopReason,
    ToolCallStatus,
)

logger = logging.getLogger(__name__)


class ToolCallRecord:
    """Record of a single tool call within a session."""

    def __init__(self, tool_call_id: str, title: str, kind: str):
        self.tool_call_id = tool_call_id
        self.title = title
        self.kind = kind
        self.status = ToolCallStatus.PENDING
        self.content: list[dict] = []
        self.started_at = datetime.now()
        self.finished_at: datetime | None = None

    def update_status(
        self, status: ToolCallStatus, content: list[dict] | None = None
    ) -> None:
        self.status = status
        if content:
            self.content.extend(content)
        if status in (
            ToolCallStatus.COMPLETED,
            ToolCallStatus.FAILED,
            ToolCallStatus.CANCELLED,
        ):
            self.finished_at = datetime.now()


class ACPSession:
    """Tracks state for one ACP session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.agent_info: AgentInfo | None = None
        self.agent_capabilities: AgentCapabilities | None = None
        self.message_chunks: list[str] = []
        self.thought_chunks: list[str] = []
        self.tool_calls: list[ToolCallRecord] = []
        self._tool_call_map: dict[str, ToolCallRecord] = {}
        self.stop_reason: StopReason | None = None
        self.created_at = datetime.now()

    def handle_update(self, update: dict) -> None:
        """Process a session/update notification."""
        update_type = update.get("sessionUpdate")

        if update_type == "tool_call":
            record = ToolCallRecord(
                tool_call_id=update.get("toolCallId", ""),
                title=update.get("title", ""),
                kind=update.get("kind", "other"),
            )
            self.tool_calls.append(record)
            self._tool_call_map[record.tool_call_id] = record

        elif update_type == "tool_call_update":
            tc_id = update.get("toolCallId", "")
            record = self._tool_call_map.get(tc_id)
            if record:
                record.update_status(
                    ToolCallStatus(update.get("status", "in_progress")),
                    update.get("content"),
                )

        elif update_type == "agent_message_chunk":
            content = update.get("content", {})
            if content.get("type") == "text":
                self.message_chunks.append(content.get("text", ""))

        elif update_type == "agent_thought_chunk":
            content = update.get("content", {})
            if content.get("type") == "text":
                self.thought_chunks.append(content.get("text", ""))

    @property
    def full_message(self) -> str:
        return "".join(self.message_chunks)

    @property
    def full_thought(self) -> str:
        return "".join(self.thought_chunks)
